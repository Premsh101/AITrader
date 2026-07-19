"""Tests for app.feature_engine – shapes, NaN handling, train/serve parity."""

import numpy as np

from app.feature_engine import (
    EXECUTIVE_DIM,
    FEATURE_COLUMNS,
    GUARDIAN_DIM,
    MARKET_FEATURE_DIM,
    build_executive_obs,
    build_guardian_obs,
    compute_feature_frame,
    generate_features,
)


def test_generate_features_shape_and_no_nan(synthetic_ohlcv):
    feat = generate_features(synthetic_ohlcv)
    assert feat is not None
    assert feat.shape == (MARKET_FEATURE_DIM,)
    assert feat.dtype == np.float32
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()
    assert (feat >= -5.0).all() and (feat <= 5.0).all()


def test_compute_feature_frame_full_history(synthetic_ohlcv):
    frame = compute_feature_frame(synthetic_ohlcv)
    assert frame is not None
    assert frame.shape == (len(synthetic_ohlcv), MARKET_FEATURE_DIM)
    assert list(frame.columns) == FEATURE_COLUMNS
    assert not frame.isna().any().any()


def test_generate_features_is_last_frame_row(synthetic_ohlcv):
    """Serving (generate_features) must equal training (compute_feature_frame)."""
    frame = compute_feature_frame(synthetic_ohlcv)
    feat = generate_features(synthetic_ohlcv)
    np.testing.assert_allclose(frame.iloc[-1].to_numpy(), feat)


def test_generate_features_too_short_returns_none(synthetic_ohlcv):
    assert generate_features(synthetic_ohlcv.head(10)) is None


def test_build_guardian_obs_shape_and_clipping(synthetic_ohlcv):
    feat = generate_features(synthetic_ohlcv)
    obs = build_guardian_obs(feat, unrealized_pnl_pct=0.03, bars_in_trade=4)
    assert obs.shape == (GUARDIAN_DIM,)
    assert obs.dtype == np.float32
    assert obs[15] == np.float32(0.03)
    assert obs[16] == np.float32(4 / 20)

    # P&L clipped to [-0.5, 0.5]; bars capped at 20.
    obs = build_guardian_obs(feat, unrealized_pnl_pct=-2.0, bars_in_trade=99)
    assert obs[15] == np.float32(-0.5)
    assert obs[16] == np.float32(1.0)


def test_build_executive_obs_shape_and_clipping(synthetic_ohlcv):
    feat = generate_features(synthetic_ohlcv)
    obs = build_executive_obs(feat, open_positions_frac=0.4, nifty_ret_5d=0.01)
    assert obs.shape == (EXECUTIVE_DIM,)
    assert obs.dtype == np.float32
    assert obs[15] == np.float32(0.4)
    assert abs(obs[16] - 0.01) < 1e-6

    obs = build_executive_obs(feat, open_positions_frac=3.0, nifty_ret_5d=-0.9)
    assert obs[15] == np.float32(1.0)
    assert obs[16] == np.float32(-0.2)
