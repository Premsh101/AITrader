"""
risk.py – Deterministic risk overlays for AITrader (Phase 1 safety net).

These rules run BEFORE the Guardian model and are independent of any ML:
a trained exit model can be wrong or degenerate; these floors cannot be
overridden by it.

  • Hard stop-loss           (default −5%)
  • Maximum holding period   (default 20 bars)
  • Profit ladder            (+5% peak → exit at break-even;
                              +10% peak → exit if price falls back to +8%)
  • Death protocol           (equity < 75% of peak equity → flatten & halt)
  • VIX filter               (^INDIAVIX daily spike > 15% → no new entries)

All thresholds are env-configurable; defaults follow the original design
conversation ("AI Survival Trading With 5000 INR").
"""

from __future__ import annotations

import os

import pandas as pd

STOP_LOSS_PCT = float(os.environ.get("RISK_STOP_LOSS_PCT", "0.05"))
MAX_HOLD_BARS = int(os.environ.get("RISK_MAX_HOLD_BARS", "20"))
BREAKEVEN_TRIGGER = float(os.environ.get("RISK_BREAKEVEN_TRIGGER", "0.05"))
TRAIL_TRIGGER = float(os.environ.get("RISK_TRAIL_TRIGGER", "0.10"))
TRAIL_LOCK = float(os.environ.get("RISK_TRAIL_LOCK", "0.08"))
DEATH_DRAWDOWN = float(os.environ.get("RISK_DEATH_DRAWDOWN", "0.25"))
# Notional account size used for equity/death-protocol tracking (₹).
BASE_CAPITAL = float(os.environ.get("BASE_CAPITAL", "50000"))
# Estimated friction per side, basis points (brokerage+STT+slippage proxy).
# Matches training/common.py COST_BPS so live P&L and backtests agree.
TRADING_COST_BPS = float(os.environ.get("TRADING_COST_BPS", "25"))


def round_trip_charges(buy_value: float, sell_value: float) -> float:
    """Estimated total charges (₹) for a completed round trip."""
    return (buy_value + sell_value) * TRADING_COST_BPS / 10_000.0
VIX_SPIKE_PCT = float(os.environ.get("RISK_VIX_SPIKE_PCT", "0.15"))

VIX_YAHOO = "^INDIAVIX"

# Regime filter: new entries only while NIFTY is above its N-day average.
REGIME_FILTER_ENABLED = os.environ.get("RISK_REGIME_FILTER", "1") == "1"
REGIME_SMA_BARS = int(os.environ.get("RISK_REGIME_SMA_BARS", "200"))
# Liquidity floor: minimum 20-day average turnover (₹) to enter a stock.
MIN_TURNOVER = float(os.environ.get("RISK_MIN_TURNOVER", "50000000"))  # ₹5 crore
# Volatility-scaled sizing: shrink the slot in wilder stocks.
TARGET_ATR_PCT = float(os.environ.get("RISK_TARGET_ATR_PCT", "0.02"))
MIN_SIZE_SCALE = float(os.environ.get("RISK_MIN_SIZE_SCALE", "0.25"))


def overlay_exit_reason(
    pnl_pct: float,
    peak_pnl_pct: float,
    bars_in_trade: int,
) -> str | None:
    """Return the overlay exit reason for an open long, or None to defer to
    the Guardian model.

    Args:
        pnl_pct:       (current_price − buy_price) / buy_price.
        peak_pnl_pct:  (highest close since entry − buy_price) / buy_price.
        bars_in_trade: bars the position has been open.
    """
    if pnl_pct <= -STOP_LOSS_PCT:
        return "stop-loss"
    if bars_in_trade >= MAX_HOLD_BARS:
        return "time-exit"
    if peak_pnl_pct >= TRAIL_TRIGGER and pnl_pct <= TRAIL_LOCK:
        return "profit-trail"
    if peak_pnl_pct >= BREAKEVEN_TRIGGER and pnl_pct <= 0.0:
        return "breakeven-stop"
    return None


def vix_entries_blocked(vix_df: pd.DataFrame | None) -> bool:
    """True when India VIX jumped more than VIX_SPIKE_PCT on the last bar."""
    if vix_df is None or len(vix_df) < 2:
        return False
    try:
        last = float(vix_df["close"].iloc[-1])
        prev = float(vix_df["close"].iloc[-2])
        return prev > 0 and (last / prev - 1.0) > VIX_SPIKE_PCT
    except Exception:
        return False


def death_threshold(peak_equity: float) -> float:
    """Equity below this value triggers the death protocol."""
    return peak_equity * (1.0 - DEATH_DRAWDOWN)


def regime_allows_entries(nifty_df: pd.DataFrame | None) -> bool:
    """Trend-following regime filter: allow new entries only while the NIFTY
    trades above its REGIME_SMA_BARS-day average.

    Bear markets are where small accounts die; staying out while the index is
    below its long average is the oldest, best-evidenced protection there is.
    Fails open when disabled or when there isn't enough history.
    """
    if not REGIME_FILTER_ENABLED:
        return True
    if nifty_df is None or len(nifty_df) < REGIME_SMA_BARS:
        return True
    try:
        close = nifty_df["close"].astype(float)
        return float(close.iloc[-1]) > float(close.rolling(REGIME_SMA_BARS).mean().iloc[-1])
    except Exception:
        return True


def is_liquid(df: pd.DataFrame | None) -> bool:
    """True when 20-day average turnover (close × volume) ≥ MIN_TURNOVER.

    The cost model assumes ~25 bps of slippage — only realistic in liquid
    names; illiquid small-caps can cost several times that.
    """
    if df is None or len(df) < 20:
        return False
    try:
        turnover = (df["close"].astype(float) * df["volume"].astype(float)).tail(20).mean()
        return float(turnover) >= MIN_TURNOVER
    except Exception:
        return False


def vol_scaled_quantity(
    reference_price: float,
    atr_pct: float,
    capital_per_slot: float,
) -> int:
    """Shares to buy with volatility-scaled slot capital.

    The slot is scaled by TARGET_ATR_PCT / atr_pct (capped at 1×, floored at
    MIN_SIZE_SCALE) so a stock moving 4%/day gets half the capital of one
    moving 2%/day — equalising rupee risk per position.
    """
    if reference_price <= 0:
        raise ValueError(f"reference_price must be positive, got {reference_price}")
    scale = 1.0
    if atr_pct and atr_pct > 0:
        scale = min(1.0, max(MIN_SIZE_SCALE, TARGET_ATR_PCT / atr_pct))
    import math

    return max(1, math.floor(capital_per_slot * scale / reference_price))
