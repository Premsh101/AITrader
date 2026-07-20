"""Tests for the v5 excess-over-benchmark entry reward (train_triad._entry_reward).

The go/no-go gate is "beat buy-and-hold ^NSEI", so the entry reward must score
a trade on its ALPHA (out-performance of the index), not on raw forward return.
These tests pin that contract so the overtrading regression (buying every
up-drift the index gives for free) cannot silently return.
"""

import numpy as np
import pandas as pd

from app.feature_engine import MARKET_FEATURE_DIM
from training.common import (
    Dataset,
    SymbolData,
    FORWARD_BARS,
    TRAIN_ROUND_TRIP_COST,
)
import training.train_triad as tt

ACT = tt.ACTION_ACT
HOLD = tt.ACTION_HOLD


def _ds_with(stock_fwd: float, nifty_fwd: float) -> tuple[Dataset, SymbolData, int]:
    """Build a 1-symbol dataset where the stock rises *stock_fwd* over the
    horizon and the index rises *nifty_fwd*, so alpha = stock_fwd - nifty_fwd."""
    n = FORWARD_BARS + 5
    dates = pd.bdate_range("2023-01-02", periods=n)
    # Geometric ramp so the return from index 0 to index FORWARD_BARS (the exact
    # window _entry_reward measures) equals the target precisely.
    ramp = np.arange(n) / FORWARD_BARS
    close = 100.0 * (1 + stock_fwd) ** ramp
    feats = np.zeros((n, MARKET_FEATURE_DIM), np.float32)
    sd = SymbolData(dates=pd.DatetimeIndex(dates), close=close, features=feats)

    ds = Dataset()
    ds.train["SYM"] = sd
    nclose = 100.0 * (1 + nifty_fwd) ** ramp
    ds.nifty_close = pd.Series(nclose, index=dates)
    ds.nifty_fwd = ds.nifty_close.shift(-FORWARD_BARS) / ds.nifty_close - 1.0
    return ds, sd, 0


def test_riding_the_index_is_not_rewarded():
    """A stock that only matches the index earns a NEGATIVE reward when bought
    (it pays costs for zero alpha) — so the model won't learn to overtrade."""
    ds, sd, t = _ds_with(stock_fwd=0.05, nifty_fwd=0.05)
    r_buy = tt._entry_reward(sd, t, ds, ACT)
    assert r_buy < 0, f"riding the index should be a net loss, got {r_buy}"
    # Skipping an index-tracker is free (no missed alpha).
    assert tt._entry_reward(sd, t, ds, HOLD) == 0.0


def test_real_outperformer_is_rewarded():
    """A genuine outperformer (beats the index by well over costs) earns a
    positive reward when bought, and skipping it is penalised."""
    ds, sd, t = _ds_with(stock_fwd=0.10, nifty_fwd=0.02)  # +8% alpha
    r_buy = tt._entry_reward(sd, t, ds, ACT)
    assert r_buy > 0, f"an 8% outperformer should pay off, got {r_buy}"
    assert tt._entry_reward(sd, t, ds, HOLD) < 0  # missed a real opportunity


def test_underperformer_buy_is_worse_than_skip():
    """Buying a stock that lags the index is strictly worse than skipping it."""
    ds, sd, t = _ds_with(stock_fwd=0.01, nifty_fwd=0.06)  # -5% alpha
    r_buy = tt._entry_reward(sd, t, ds, ACT)
    r_skip = tt._entry_reward(sd, t, ds, HOLD)
    assert r_buy < r_skip
    assert r_skip == 0.0


def test_buy_reward_equals_alpha_minus_cost():
    """The buy reward is exactly (alpha - training cost) within the clip."""
    ds, sd, t = _ds_with(stock_fwd=0.07, nifty_fwd=0.03)  # +4% alpha
    r_buy = tt._entry_reward(sd, t, ds, ACT)
    assert abs(r_buy - (0.04 - TRAIN_ROUND_TRIP_COST)) < 1e-4
