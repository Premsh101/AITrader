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
VIX_SPIKE_PCT = float(os.environ.get("RISK_VIX_SPIKE_PCT", "0.15"))

VIX_YAHOO = "^INDIAVIX"


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
