"""Tests for the Phase 1 deterministic risk overlays and death protocol."""

import pandas as pd

from app.risk import (
    is_liquid,
    regime_allows_entries,
    vol_scaled_quantity,
    BASE_CAPITAL,
    death_threshold,
    overlay_exit_reason,
    vix_entries_blocked,
)


class TestOverlayExit:
    def test_no_exit_on_healthy_position(self):
        assert overlay_exit_reason(0.02, 0.03, bars_in_trade=3) is None

    def test_stop_loss(self):
        assert overlay_exit_reason(-0.05, 0.0, 1) == "stop-loss"
        assert overlay_exit_reason(-0.12, 0.0, 1) == "stop-loss"
        assert overlay_exit_reason(-0.049, 0.0, 1) is None

    def test_time_exit(self):
        assert overlay_exit_reason(0.01, 0.02, 20) == "time-exit"
        assert overlay_exit_reason(0.01, 0.02, 19) is None

    def test_profit_trail_locks_gains(self):
        # Peaked above +10%, fell back to +8% or less → lock the gain.
        assert overlay_exit_reason(0.08, 0.12, 5) == "profit-trail"
        assert overlay_exit_reason(0.09, 0.12, 5) is None  # still above lock

    def test_breakeven_stop(self):
        # Peaked above +5%, gave it all back → exit at breakeven.
        assert overlay_exit_reason(0.0, 0.06, 5) == "breakeven-stop"
        assert overlay_exit_reason(-0.01, 0.06, 5) == "breakeven-stop"
        assert overlay_exit_reason(0.01, 0.06, 5) is None

    def test_stop_loss_beats_ladder(self):
        assert overlay_exit_reason(-0.06, 0.12, 5) == "stop-loss"


class TestVixFilter:
    @staticmethod
    def _vix(prev: float, last: float) -> pd.DataFrame:
        return pd.DataFrame({"close": [prev, last]})

    def test_spike_blocks(self):
        assert vix_entries_blocked(self._vix(14.0, 17.0))  # +21%

    def test_calm_allows(self):
        assert not vix_entries_blocked(self._vix(14.0, 15.0))  # +7%

    def test_missing_data_allows(self):
        assert not vix_entries_blocked(None)
        assert not vix_entries_blocked(pd.DataFrame({"close": [15.0]}))


class TestDeathProtocol:
    def test_threshold_is_75_pct_of_peak(self):
        assert death_threshold(100_000.0) == 75_000.0

    def test_base_capital_positive(self):
        assert BASE_CAPITAL > 0


class TestRegimeFilter:
    @staticmethod
    def _nifty(n: int, trend: float) -> pd.DataFrame:
        import numpy as np
        close = 20000 * np.cumprod(1 + np.full(n, trend))
        return pd.DataFrame({"close": close})

    def test_uptrend_allows(self):
        assert regime_allows_entries(self._nifty(250, +0.001))

    def test_downtrend_blocks(self):
        assert not regime_allows_entries(self._nifty(250, -0.001))

    def test_insufficient_history_fails_open(self):
        assert regime_allows_entries(self._nifty(50, -0.001))
        assert regime_allows_entries(None)


class TestLiquidityFloor:
    @staticmethod
    def _df(price: float, volume: float) -> pd.DataFrame:
        return pd.DataFrame({"close": [price] * 25, "volume": [volume] * 25})

    def test_liquid_passes(self):
        assert is_liquid(self._df(1000.0, 1_000_000))  # ₹100 crore/day

    def test_illiquid_fails(self):
        assert not is_liquid(self._df(100.0, 10_000))  # ₹10 lakh/day

    def test_missing_data_fails_closed(self):
        assert not is_liquid(None)


class TestVolScaledSizing:
    def test_calm_stock_full_size(self):
        # ATR 1.5% < 2% target → full ₹10,000 slot at ₹100 → 100 shares.
        assert vol_scaled_quantity(100.0, 0.015, 10_000) == 100

    def test_wild_stock_half_size(self):
        # ATR 4% = 2x target → half slot → 50 shares.
        assert vol_scaled_quantity(100.0, 0.04, 10_000) == 50

    def test_extreme_vol_floored(self):
        # ATR 20% → scale floored at 0.25 → 25 shares, never zero.
        assert vol_scaled_quantity(100.0, 0.20, 10_000) == 25

    def test_missing_atr_full_size(self):
        assert vol_scaled_quantity(100.0, 0.0, 10_000) == 100

    def test_never_zero_shares(self):
        assert vol_scaled_quantity(9_999.0, 0.20, 10_000) == 1
