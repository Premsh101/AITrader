"""
evaluate_legacy.py – Cost-aware walk-forward backtest of the LEGACY
``apex_1500`` models using their recovered observation recipes
(training/legacy_features.py).  Answers, with data, whether the old models
are worth reviving or the retrained triad should replace them.

Semantics replayed as originally designed:
  • Hunter: Discrete(2) per symbol on the 5-dim obs → buy signals,
    ranked by the policy's buy probability.
  • Executive: Discrete(3) on the 3-dim obs → accept 0 / 1 / up to 3 of
    today's signals (its trained meaning — NOT per-symbol scoring).
  • Guardian: Discrete(2) on the 5-dim trade obs; the training env's forced
    exits also apply (−5% stop, 20-bar cap) since they were part of the
    design.
Costs: the same ROUND_TRIP_COST as the new models' evaluation.

Usage:  python training/evaluate_legacy.py [--models-dir models/legacy] [--smoke]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.common import (  # noqa: E402
    MAX_SLOTS,
    MAX_TRADE_BARS,
    ROUND_TRIP_COST,
    build_dataset,
    download_universe,
)
from training.legacy_features import (  # noqa: E402
    compute_legacy_frame,
    executive_obs,
    guardian_obs,
    hunter_obs,
)
from app.stock_list import NSE_SYMBOLS  # noqa: E402
from app.symbols import to_base  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

LEGACY_STOP_LOSS = -0.05
EXEC_ACCEPT = {0: 0, 1: 1, 2: 3}


def _load(path: str, expected_dim: int):
    from stable_baselines3 import PPO

    model = PPO.load(path, device="cpu")
    actual = tuple(model.observation_space.shape)
    if actual != (expected_dim,):
        raise SystemExit(f"{path}: obs shape {actual} != ({expected_dim},)")
    return model


def _action(model, obs: np.ndarray) -> int:
    a, _ = model.predict(obs.reshape(1, -1), deterministic=True)
    return int(np.asarray(a).reshape(-1)[0])


def _buy_prob(model, obs: np.ndarray) -> float:
    import torch

    with torch.no_grad():
        t, _ = model.policy.obs_to_tensor(obs.reshape(1, -1))
        return float(model.policy.get_distribution(t).distribution.probs[0, 1].item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=os.path.join(REPO_ROOT, "models", "legacy"))
    parser.add_argument("--smoke", action="store_true", help="5 symbols only")
    args = parser.parse_args()

    hunter = _load(os.path.join(args.models_dir, "hunter_apex_1500_brain.zip"), 5)
    guardian = _load(os.path.join(args.models_dir, "guardian_apex_1500_brain.zip"), 5)
    executive = _load(os.path.join(args.models_dir, "executive_apex_manager.zip"), 3)
    logger.info("Legacy models loaded from %s", args.models_dir)

    symbols_subset = NSE_SYMBOLS[:5] if args.smoke else None
    ohlcv = download_universe(symbols=symbols_subset)
    ds = build_dataset(ohlcv)  # reuse the walk-forward split for closes/dates
    if not ds.val:
        raise SystemExit("Validation slice is empty")

    # Legacy feature frames aligned to the SAME validation slice.
    legacy: dict[str, pd.DataFrame] = {}
    for sym, df in ohlcv.items():
        if sym == to_base("^NSEI") or sym not in ds.val:
            continue
        frame = compute_legacy_frame(df)
        if frame is None:
            continue
        legacy[sym] = frame.reindex(ds.val[sym].dates).fillna(0.0)

    date_idx = {sym: {d: i for i, d in enumerate(sd.dates)} for sym, sd in ds.val.items()}
    all_dates = sorted({d for sym in legacy for d in ds.val[sym].dates})

    cash, positions = 1.0, {}
    trade_returns: list[float] = []
    equity_curve: list[float] = []

    def mark(today):
        v = cash
        for sym, pos in positions.items():
            i = date_idx[sym].get(today)
            price = ds.val[sym].close[i] if i is not None else pos["entry"]
            v += pos["alloc"] * (price / pos["entry"])
        return v

    for today in all_dates:
        # Guardian pass (+ designed forced exits)
        for sym in list(positions):
            pos = positions[sym]
            i = date_idx[sym].get(today)
            if i is None or i <= pos["i"]:
                continue
            pos["i"] = i
            price = float(ds.val[sym].close[i])
            pnl = price / pos["entry"] - 1.0
            bars = i - pos["entry_i"]
            row = legacy[sym].iloc[i]
            sell = (
                pnl <= LEGACY_STOP_LOSS
                or bars >= MAX_TRADE_BARS
                or _action(guardian, guardian_obs(pnl, bars, row)) == 1
            )
            if sell:
                net = pnl - ROUND_TRIP_COST
                cash += pos["alloc"] * (1.0 + net)
                trade_returns.append(net)
                del positions[sym]

        # Hunter signals ranked by buy probability
        scored = []
        for sym, frame in legacy.items():
            i = date_idx[sym].get(today)
            if i is None or sym in positions:
                continue
            obs = hunter_obs(frame.iloc[i])
            if _action(hunter, obs) == 1:
                scored.append((_buy_prob(hunter, obs), sym, i))
        scored.sort(reverse=True)

        # Executive decides how many to accept (its trained meaning)
        free = MAX_SLOTS - len(positions)
        if scored and free > 0:
            vols = [legacy[s].iloc[date_idx[s][today]]["volatility"]
                    for s in legacy if today in date_idx[s]]
            open_pnls = [
                ds.val[s].close[date_idx[s][today]] / p["entry"] - 1.0
                for s, p in positions.items() if today in date_idx[s]
            ]
            obs = executive_obs(
                free / MAX_SLOTS,
                float(np.mean(vols)) if vols else 0.0,
                float(np.mean(open_pnls)) if open_pnls else 0.0,
            )
            n = min(EXEC_ACCEPT[_action(executive, obs)], free, len(scored))
            equity_now = mark(today)
            for _, sym, i in scored[:n]:
                alloc = min(cash, equity_now / MAX_SLOTS)
                if alloc <= 0:
                    break
                cash -= alloc
                positions[sym] = {"entry": float(ds.val[sym].close[i]),
                                  "entry_i": i, "i": i, "alloc": alloc}

        equity_curve.append(mark(today))

    # Liquidate remainder
    last = all_dates[-1]
    for sym, pos in positions.items():
        i = date_idx[sym].get(last, pos["i"])
        net = float(ds.val[sym].close[i]) / pos["entry"] - 1.0 - ROUND_TRIP_COST
        cash += pos["alloc"] * (1.0 + net)
        trade_returns.append(net)

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min()) if len(equity) else 0.0
    baseline = 0.0
    if ds.nifty_close is not None:
        w = ds.nifty_close[(ds.nifty_close.index >= all_dates[0]) & (ds.nifty_close.index <= last)]
        if len(w) > 1:
            baseline = float(w.iloc[-1] / w.iloc[0] - 1.0)
    r = np.array(trade_returns)

    print("\n──────── LEGACY apex_1500 out-of-sample evaluation ────────")
    print(f"Window            : {all_dates[0].date()} → {last.date()}")
    print(f"Total return      : {cash - 1.0:+.2%}")
    print(f"Trades            : {len(r)}")
    print(f"Mean trade return : {(r.mean() if len(r) else 0.0):+.3%}")
    print(f"Win rate          : {((r > 0).mean() if len(r) else 0.0):.1%}")
    print(f"Max drawdown      : {max_dd:.2%}")
    print(f"^NSEI buy & hold  : {baseline:+.2%}  (baseline)")
    if cash - 1.0 > baseline:
        print("\nVERDICT: LEGACY PASS – the old models beat buy-and-hold after "
              "costs. Reviving them (with a 5-dim serving adapter) is justified.")
    else:
        print("\nVERDICT: LEGACY FAIL – the old models do not beat buy-and-hold "
              "after costs. Retire them; deploy the retrained triad instead.")


if __name__ == "__main__":
    main()
