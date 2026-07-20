"""
evaluate_triad.py – Walk-forward, cost-aware evaluation of the retrained
triad on the held-out validation slice.  This is the go/no-go gate before
deployment.

The full serving pipeline is replayed day by day:
  Hunter signals → drop already-held symbols → Executive approve-probability
  ranking (must exceed EXECUTIVE_APPROVE_THRESHOLD, default 0.5) → entries
  fill free slots → Guardian manages every open position (force-close at
  20 bars, matching training).

Trades pay ``ROUND_TRIP_COST`` (2 × COST_BPS per side).  The baseline is
buy-and-hold ^NSEI over the same window.

VERDICT RULE: if the strategy does not beat the baseline after costs on this
out-of-sample slice, DO NOT deploy the models — iterate on the reward design
in train_triad.py instead.

Usage:
  python training/evaluate_triad.py                    # models from repo models/
  python training/evaluate_triad.py --models-dir /kaggle/working/models --smoke
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

# Allow `python training/evaluate_triad.py` from the repo root (or Kaggle).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.common import (  # noqa: E402
    MAX_SLOTS,
    REENTRY_COOLDOWN_BARS,
    MAX_TRADE_BARS,
    ROUND_TRIP_COST,
    Dataset,
    build_dataset,
    download_universe,
    nifty_ret_lookup,
)

from app.feature_engine import build_executive_obs, build_guardian_obs  # noqa: E402
from app.risk import overlay_exit_reason, regime_allows_entries  # noqa: E402
from app.stock_list import NSE_SYMBOLS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class Position:
    __slots__ = ("symbol", "entry_price", "entry_idx", "bars", "peak", "alloc")

    def __init__(self, symbol: str, entry_price: float, entry_idx: int, alloc: float):
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_idx = entry_idx
        self.bars = 0
        self.peak = entry_price
        self.alloc = alloc


def run_backtest(ds: Dataset, brains) -> dict:
    """Replay the Hunter → Executive → Guardian pipeline over the val slice."""
    # Per-symbol date → row-index lookup for the validation slice.
    date_idx: dict[str, dict[pd.Timestamp, int]] = {
        sym: {d: i for i, d in enumerate(sd.dates)} for sym, sd in ds.val.items()
    }

    # Master calendar: every trading day seen in the validation slice.
    all_dates = sorted({d for sd in ds.val.values() for d in sd.dates})
    if not all_dates:
        raise RuntimeError("Validation slice is empty – lower the train cutoff")

    cash = 1.0
    positions: dict[str, Position] = {}
    cooldown: dict[str, int] = {}  # symbol -> bar index it was closed at
    trade_returns: list[float] = []
    equity_curve: list[float] = []

    def mark_equity(today: pd.Timestamp) -> float:
        value = cash
        for pos in positions.values():
            sd = ds.val[pos.symbol]
            i = date_idx[pos.symbol].get(today)
            price = sd.close[i] if i is not None else sd.close[pos.entry_idx + pos.bars]
            value += pos.alloc * (price / pos.entry_price)
        return value

    for today in all_dates:
        # ── Guardian pass: manage open positions ──────────────────────────
        for sym in list(positions.keys()):
            pos = positions[sym]
            i = date_idx[sym].get(today)
            if i is None or i <= pos.entry_idx + pos.bars:
                continue  # no bar for this symbol today
            pos.bars = i - pos.entry_idx
            sd = ds.val[sym]
            price = float(sd.close[i])
            pos.peak = max(pos.peak, price)

            pnl_pct = price / pos.entry_price - 1.0
            peak_pnl = pos.peak / pos.entry_price - 1.0
            # Deterministic risk overlays run BEFORE the Guardian, exactly as
            # in the live loop (app/main.py) - the backtest grades the SYSTEM.
            close_now = overlay_exit_reason(pnl_pct, peak_pnl, pos.bars) is not None
            if not close_now:
                obs = build_guardian_obs(sd.features[i], pnl_pct, pos.bars)
                close_now = brains.guardian.should_close(obs, sym)
            if close_now:
                net = max(pnl_pct, -0.05) - ROUND_TRIP_COST
                cash += pos.alloc * (1.0 + net)
                trade_returns.append(net)
                cooldown[sym] = i
                del positions[sym]

        # ── Entry pass: Hunter → dedup → Executive ────────────────────────
        todays_features = {
            sym: ds.val[sym].features[date_idx[sym][today]]
            for sym in ds.val
            if today in date_idx[sym]
        }
        nifty_hist = None
        if ds.nifty_close is not None:
            nifty_hist = ds.nifty_close[ds.nifty_close.index <= today].to_frame("close")
        if not regime_allows_entries(nifty_hist):
            equity_curve.append(mark_equity(today))
            continue
        signals = brains.hunter.find_signals(todays_features)
        signals = [
            s for s in signals
            if s not in positions
            and date_idx[s][today] - cooldown.get(s, -10**9) > REENTRY_COOLDOWN_BARS
        ]

        open_slots = MAX_SLOTS - len(positions)
        if signals and open_slots > 0:
            open_frac = len(positions) / MAX_SLOTS
            nifty = nifty_ret_lookup(ds, today)
            exec_obs = {
                s: build_executive_obs(todays_features[s], open_frac, nifty)
                for s in signals
            }
            selected = brains.executive.select_slots(signals, exec_obs, open_slots)
            equity_now = mark_equity(today)
            for sym in selected:
                i = date_idx[sym][today]
                alloc = min(cash, equity_now / MAX_SLOTS)
                if alloc <= 0:
                    break
                cash -= alloc
                positions[sym] = Position(
                    sym, float(ds.val[sym].close[i]), i, alloc
                )

        equity_curve.append(mark_equity(today))

    # Liquidate whatever is still open at the final mark.
    final_date = all_dates[-1]
    for sym, pos in list(positions.items()):
        sd = ds.val[sym]
        i = date_idx[sym].get(final_date, pos.entry_idx + pos.bars)
        net = float(sd.close[i]) / pos.entry_price - 1.0 - ROUND_TRIP_COST
        cash += pos.alloc * (1.0 + net)
        trade_returns.append(net)
    positions.clear()

    equity = np.array(equity_curve, dtype=float)
    running_peak = np.maximum.accumulate(equity)
    max_drawdown = float(((equity - running_peak) / running_peak).min()) if len(equity) else 0.0

    # Buy-and-hold ^NSEI over the same window.
    baseline = 0.0
    if ds.nifty_close is not None:
        window = ds.nifty_close[
            (ds.nifty_close.index >= all_dates[0])
            & (ds.nifty_close.index <= all_dates[-1])
        ]
        if len(window) > 1:
            baseline = float(window.iloc[-1] / window.iloc[0] - 1.0)

    returns = np.array(trade_returns, dtype=float)
    return {
        "window": (all_dates[0].date(), all_dates[-1].date()),
        "total_return": cash - 1.0,
        "trade_count": len(returns),
        "mean_trade_return": float(returns.mean()) if len(returns) else 0.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "max_drawdown": max_drawdown,
        "baseline_nifty": baseline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "models"),
        help="Directory containing the three model zips (default: repo models/)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Evaluate on 5 symbols only (pairs with train_triad.py --smoke)",
    )
    args = parser.parse_args()

    # Point ai_brains at the requested directory BEFORE importing it, then
    # reuse the exact serving-side wrappers (including the obs-dim load
    # assertions) so evaluation and production share one code path.
    os.environ["MODELS_DIR"] = os.path.abspath(args.models_dir)
    from app.ai_brains import BrainManager

    brains = BrainManager()
    brains.load_all()
    if not brains.all_ready:
        raise SystemExit(
            "Models failed to load or failed the observation-dim assertion – "
            "nothing to evaluate."
        )

    symbols_subset = NSE_SYMBOLS[:5] if args.smoke else None
    ohlcv = download_universe(symbols=symbols_subset)
    ds = build_dataset(ohlcv)
    if not ds.val:
        raise SystemExit("Validation slice is empty – check data / cutoff")

    metrics = run_backtest(ds, brains)

    print("\n──────────── Out-of-sample evaluation ────────────")
    print(f"Window            : {metrics['window'][0]} → {metrics['window'][1]}")
    print(f"Total return      : {metrics['total_return']:+.2%}")
    print(f"Trades            : {metrics['trade_count']}")
    print(f"Mean trade return : {metrics['mean_trade_return']:+.3%}")
    print(f"Win rate          : {metrics['win_rate']:.1%}")
    print(f"Max drawdown      : {metrics['max_drawdown']:.2%}")
    print(f"^NSEI buy & hold  : {metrics['baseline_nifty']:+.2%}  (baseline)")

    beats = metrics["total_return"] > metrics["baseline_nifty"]
    if beats:
        print("\nVERDICT: PASS – strategy beats buy-and-hold ^NSEI after costs. "
              "OK to commit the model zips to models/.")
    else:
        print("\nVERDICT: FAIL – strategy does NOT beat buy-and-hold ^NSEI after "
              "costs. DO NOT deploy; iterate on the reward design instead.")


if __name__ == "__main__":
    main()
