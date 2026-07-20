"""
common.py – Shared data loading and configuration for the AITrader
retraining package.

CRITICAL DESIGN RULE: all feature computation is imported from
``app.feature_engine`` (:func:`compute_feature_frame`,
:func:`build_guardian_obs`, :func:`build_executive_obs`).  Training must
never re-implement feature code — duplicated feature definitions are what
made the previous generation of models unusable in production.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Make the repo root importable when this script is run from training/ or
# from a Kaggle working directory containing the cloned repo.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.feature_engine import (  # noqa: E402
    FEATURE_COLUMNS,
    WARMUP_BARS,
    compute_feature_frame,
)
from app.stock_list import NSE_SYMBOLS  # noqa: E402
from app.symbols import to_base, to_yahoo  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Round-trip friction, basis points PER SIDE.  25 bps ≈ brokerage + STT +
# exchange charges + slippage for NSE cash-segment trades; deliberately
# conservative so the models only learn edges that survive real costs.
COST_BPS = 25.0
COST_PER_SIDE = COST_BPS / 10_000.0
ROUND_TRIP_COST = 2.0 * COST_PER_SIDE

# TRAINING friction: a modest multiplier over the true cost gives the models
# a small selectivity margin.  In v5 the dominant selectivity pressure comes
# from the EXCESS-RETURN reward (below), not from inflating costs, so this is
# 1.5 rather than the v2-v4 value of 2.0 (which over-suppressed trades when
# combined with the benchmark subtraction).  EVALUATION always uses the true
# ROUND_TRIP_COST.
TRAIN_COST_MULT = 1.5
TRAIN_ROUND_TRIP_COST = ROUND_TRIP_COST * TRAIN_COST_MULT

# v5 CORE FIX — reward EXCESS return over the ^NSEI benchmark, not the raw
# forward return.  The go/no-go gate is "beat buy-and-hold ^NSEI", so the
# reward must optimise exactly that: a trade only earns reward for how much it
# OUTPERFORMS the index over the holding window.  Buying a stock that rises
# while the index rises just as much is worth ~0 (you could have held the
# index for free); it is a LOSS after costs.  This is what forces genuine
# selectivity — most stocks do NOT beat the index, so the models learn to
# trade only the setups that do, instead of buying every up-drift (the v1-v4
# overtrading disease: 350-850 trades, sub-40% win, always below baseline).
REWARD_EXCESS_OVER_NIFTY = True

# Data window (yfinance period string, ≥ 5y required).
DATA_PERIOD = "6y"

# Walk-forward split: everything ON OR BEFORE the cutoff trains, everything
# strictly after validates.  Default: 18 months before today.
TRAIN_CUTOFF_MONTHS_BACK = 18

# Forward horizon (bars) used by the Hunter/Executive reward.  v5: 10 bars
# (was 5) — matches the ~20-bar live hold better, halves the noise in the
# excess-return signal, and lets a real trend show through the benchmark.
FORWARD_BARS = 10

# Hold penalised as a missed opportunity only when the EXCESS forward return
# (stock minus ^NSEI) exceeded this threshold — i.e. only skipping a genuine
# outperformer is penalised, never skipping a stock that merely tracked the
# index.
MISSED_OPPORTUNITY_THRESHOLD = 0.03  # v2: only real moves count as missed
MISSED_OPPORTUNITY_PENALTY = -0.005   # v2: patience is cheaper than churn

# Guardian episode cap (bars) – must match the bars_in_trade scaling in
# app.feature_engine.build_guardian_obs.
MAX_TRADE_BARS = 20

# Portfolio slots – must match app.ai_brains.MAX_SLOTS.
MAX_SLOTS = 5

# Bars a just-closed symbol stays un-buyable (churn brake), serve + backtest.
REENTRY_COOLDOWN_BARS = 5

# Hard stop-loss fraction (matches app/risk.py RISK_STOP_LOSS_PCT).  Training
# rewards cap downside here, teaching the TRUE risk profile: max loss ~5%,
# target +10%-style moves -> ~2:1 reward-to-risk per trade.
TRAIN_STOP_LOSS = 0.05

NIFTY_TICKER = "^NSEI"

# Exact filenames the app loads (app/ai_brains.py).
HUNTER_FILE = "hunter_apex_1500_brain.zip"
GUARDIAN_FILE = "guardian_apex_1500_brain.zip"
EXECUTIVE_FILE = "executive_apex_manager.zip"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class SymbolData:
    """Aligned per-symbol arrays for one date slice."""

    dates: pd.DatetimeIndex
    close: np.ndarray          # (T,) float64
    features: np.ndarray       # (T, 15) float32


@dataclass
class Dataset:
    """Train/validation split of the whole universe plus the NIFTY index."""

    train: dict[str, SymbolData] = field(default_factory=dict)
    val: dict[str, SymbolData] = field(default_factory=dict)
    nifty_ret_5d: pd.Series | None = None      # indexed by date, full history
    nifty_close: pd.Series | None = None       # indexed by date, full history
    nifty_fwd: pd.Series | None = None         # FORWARD FORWARD_BARS return
    cutoff: pd.Timestamp | None = None


def default_cutoff(today: date | None = None) -> pd.Timestamp:
    today = today or date.today()
    return pd.Timestamp(today - timedelta(days=TRAIN_CUTOFF_MONTHS_BACK * 30))


def download_universe(
    symbols: list[str] | None = None,
    period: str = DATA_PERIOD,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for the universe + ^NSEI via yfinance.

    Returns a dict keyed by BASE symbol (the index keeps its ``^NSEI`` name).
    """
    import yfinance as yf

    symbols = symbols if symbols is not None else NSE_SYMBOLS
    tickers = [to_yahoo(s) for s in symbols] + [NIFTY_TICKER]

    logger.info("Downloading %d tickers (period=%s)…", len(tickers), period)
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data – check connectivity")

    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = raw[ticker].copy() if len(tickers) > 1 else raw.copy()
        except (KeyError, TypeError):
            continue
        df = df.dropna(how="all")
        if df.empty or len(df) < WARMUP_BARS + FORWARD_BARS + 10:
            logger.warning("Skipping %s – insufficient history (%d rows)", ticker, len(df))
            continue
        df.columns = [c.lower() for c in df.columns]
        out[to_base(ticker)] = df

    logger.info("Downloaded usable data for %d / %d tickers", len(out), len(tickers))
    return out


def build_dataset(
    ohlcv: dict[str, pd.DataFrame],
    cutoff: pd.Timestamp | None = None,
) -> Dataset:
    """Compute features for every symbol and split train/val by date.

    The split is strictly chronological: bars dated ≤ *cutoff* go to train,
    bars dated > *cutoff* go to validation.  No shuffling ever crosses the
    boundary.
    """
    cutoff = cutoff if cutoff is not None else default_cutoff()
    ds = Dataset(cutoff=cutoff)

    nifty_df = ohlcv.get(to_base(NIFTY_TICKER))
    if nifty_df is not None:
        close = nifty_df["close"].astype(float)
        ds.nifty_close = close
        ds.nifty_ret_5d = close.pct_change(5).fillna(0.0)
        # Forward FORWARD_BARS return of the index at each date (NaN at the
        # tail).  Used only by the training reward to compute per-trade alpha.
        ds.nifty_fwd = close.shift(-FORWARD_BARS) / close - 1.0

    for sym, df in ohlcv.items():
        if sym == to_base(NIFTY_TICKER):
            continue
        frame = compute_feature_frame(df)
        if frame is None:
            logger.warning("Feature computation failed for %s – skipped", sym)
            continue

        # Drop indicator warm-up rows: they are zero-filled, not real signal.
        frame = frame.iloc[WARMUP_BARS:]
        close = df["close"].astype(float).iloc[WARMUP_BARS:]

        idx = frame.index
        train_mask = idx <= cutoff
        val_mask = idx > cutoff

        for mask, bucket in ((train_mask, ds.train), (val_mask, ds.val)):
            if mask.sum() < FORWARD_BARS + 5:
                continue
            bucket[sym] = SymbolData(
                dates=pd.DatetimeIndex(idx[mask]),
                close=close[mask].to_numpy(),
                features=frame.loc[mask, FEATURE_COLUMNS].to_numpy(np.float32),
            )

    logger.info(
        "Dataset built: %d train symbols, %d val symbols, cutoff=%s",
        len(ds.train), len(ds.val), cutoff.date(),
    )
    return ds


def nifty_ret_lookup(ds: Dataset, when: pd.Timestamp) -> float:
    """5-day ^NSEI return at *when* (0.0 if unavailable)."""
    if ds.nifty_ret_5d is None:
        return 0.0
    try:
        pos = ds.nifty_ret_5d.index.get_indexer([when], method="ffill")[0]
        if pos < 0:
            return 0.0
        return float(ds.nifty_ret_5d.iloc[pos])
    except Exception:
        return 0.0


def nifty_fwd_lookup(ds: Dataset, when: pd.Timestamp) -> float:
    """Forward FORWARD_BARS ^NSEI return at *when* (the benchmark a trade
    opened on this date must beat).  0.0 if unavailable or at the tail."""
    if ds.nifty_fwd is None:
        return 0.0
    try:
        pos = ds.nifty_fwd.index.get_indexer([when], method="ffill")[0]
        if pos < 0:
            return 0.0
        v = ds.nifty_fwd.iloc[pos]
        return float(v) if pd.notna(v) else 0.0
    except Exception:
        return 0.0
