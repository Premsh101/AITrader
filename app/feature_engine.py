"""
feature_engine.py – Technical-indicator feature generation for AITrader.

This module is the SINGLE source of truth for feature definitions.  The
training package (``training/train_triad.py``) imports
:func:`compute_feature_frame` so that training and serving share one
implementation — any drift between the two is exactly the class of bug that
made the original models unusable.

Market feature vector layout (15 values, ``MARKET_FEATURE_DIM``):
  0  close_ratio        – close / 20-period SMA (measures price relative to trend)
  1  rsi                – RSI(14), scaled to [0, 1]
  2  macd               – MACD line, normalised by close
  3  macd_signal        – MACD signal line, normalised by close
  4  macd_hist          – MACD histogram, normalised by close
  5  bb_pct             – Bollinger Band %B  (0 = at lower, 1 = at upper)
  6  ema9_ratio         – EMA(9) / EMA(21)
  7  ema21_ratio        – EMA(21) / EMA(50)
  8  vol_ratio          – today's volume / 20-period average volume
  9  atr_norm           – ATR(14) / close
  10 ret_1d             – 1-day return
  11 ret_5d             – 5-day return
  12 rsi_delta          – RSI change over last 3 bars
  13 stoch_k            – Stochastic %K, scaled to [0, 1]
  14 stoch_d            – Stochastic %D, scaled to [0, 1]

Guardian observation (17 values, ``GUARDIAN_DIM``):
  0–14  market features above
  15    unrealized P&L %, clipped to [-0.5, 0.5]
  16    bars in trade, capped at 20 and scaled to [0, 1]

Executive observation (17 values, ``EXECUTIVE_DIM``):
  0–14  market features above
  15    open positions fraction (open_count / MAX_SLOTS)
  16    NIFTY (^NSEI) 5-day return, clipped
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MARKET_FEATURE_DIM = 15
GUARDIAN_DIM = 17
EXECUTIVE_DIM = 17

# Backwards-compatible alias (pre-refactor name).
FEATURE_DIM = MARKET_FEATURE_DIM

FEATURE_COLUMNS: list[str] = [
    "close_ratio",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pct",
    "ema9_ratio",
    "ema21_ratio",
    "vol_ratio",
    "atr_norm",
    "ret_1d",
    "ret_5d",
    "rsi_delta",
    "stoch_k",
    "stoch_d",
]

# Number of leading bars whose indicator values are dominated by rolling-window
# warm-up (longest window is EMA(50)/SMA(20)/RSI(14) chains → ~50 bars).
WARMUP_BARS = 50


def compute_feature_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute the full history of all 15 market features for one symbol.

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
            (case-insensitive) sorted by date ascending.  At least 30 rows
            are required; ≥60 recommended so all indicators have history.

    Returns:
        DataFrame indexed like *df* with the 15 ``FEATURE_COLUMNS``, NaN/Inf
        replaced and values clipped to [-5, 5]; or ``None`` if the input is
        too short or indicator computation fails.  The first ``WARMUP_BARS``
        rows contain zero-filled warm-up values — training code should skip
        them.
    """
    if df is None or len(df) < 30:
        return None

    # Work on a copy; ensure lower-case column names
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    try:
        import pandas_ta as ta  # type: ignore[import]

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # ── Moving averages ───────────────────────────────────────────────
        sma20 = close.rolling(20).mean()
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        # ── RSI ───────────────────────────────────────────────────────────
        rsi_series = ta.rsi(close, length=14)
        if rsi_series is None:
            rsi_series = pd.Series(np.nan, index=close.index)

        # ── MACD ──────────────────────────────────────────────────────────
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is None or macd_df.empty:
            return None
        # pandas_ta MACD columns follow the pattern: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        macd_col = next(
            (c for c in macd_df.columns if c.upper().startswith("MACD_")), None
        )
        sig_col = next(
            (c for c in macd_df.columns if c.upper().startswith("MACDS_")), None
        )
        hist_col = next(
            (c for c in macd_df.columns if c.upper().startswith("MACDH_")), None
        )
        if not (macd_col and sig_col and hist_col):
            return None
        macd_line = macd_df[macd_col]
        macd_sig = macd_df[sig_col]
        macd_hist = macd_df[hist_col]

        # ── Bollinger Bands ───────────────────────────────────────────────
        bb = ta.bbands(close, length=20, std=2)
        if bb is None or bb.empty:
            return None
        # pandas_ta BBands columns: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        bb_upper_col = next(
            (c for c in bb.columns if c.upper().startswith("BBU_")), None
        )
        bb_lower_col = next(
            (c for c in bb.columns if c.upper().startswith("BBL_")), None
        )
        if not (bb_upper_col and bb_lower_col):
            return None
        bb_upper = bb[bb_upper_col]
        bb_lower = bb[bb_lower_col]
        bb_pct = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

        # ── Volume ratio ──────────────────────────────────────────────────
        vol_ma20 = volume.rolling(20).mean()
        vol_ratio = volume / (vol_ma20 + 1e-9)

        # ── ATR ───────────────────────────────────────────────────────────
        atr = ta.atr(high, low, close, length=14)
        if atr is None:
            atr = pd.Series(np.nan, index=close.index)
        atr_norm = atr / (close + 1e-9)

        # ── Returns ───────────────────────────────────────────────────────
        ret_1d = close.pct_change(1)
        ret_5d = close.pct_change(5)

        # ── RSI delta ─────────────────────────────────────────────────────
        rsi_delta = rsi_series.diff(3)

        # ── Stochastic ────────────────────────────────────────────────────
        stoch = ta.stoch(high, low, close, k=14, d=3)
        if stoch is None or stoch.empty:
            stoch_k = pd.Series(50.0, index=close.index)
            stoch_d = pd.Series(50.0, index=close.index)
        else:
            k_col = [c for c in stoch.columns if "STOCHk" in c]
            d_col = [c for c in stoch.columns if "STOCHd" in c]
            stoch_k = stoch[k_col[0]] if k_col else pd.Series(50.0, index=close.index)
            stoch_d = stoch[d_col[0]] if d_col else pd.Series(50.0, index=close.index)

        frame = pd.DataFrame(
            {
                "close_ratio": close / (sma20 + 1e-9),
                "rsi": rsi_series / 100.0,
                "macd": macd_line / (close + 1e-9),
                "macd_signal": macd_sig / (close + 1e-9),
                "macd_hist": macd_hist / (close + 1e-9),
                "bb_pct": bb_pct,
                "ema9_ratio": ema9 / (ema21 + 1e-9),
                "ema21_ratio": ema21 / (ema50 + 1e-9),
                "vol_ratio": vol_ratio,
                "atr_norm": atr_norm,
                "ret_1d": ret_1d,
                "ret_5d": ret_5d,
                "rsi_delta": rsi_delta / 100.0,
                "stoch_k": stoch_k / 100.0,
                "stoch_d": stoch_d / 100.0,
            },
            index=df.index,
        )[FEATURE_COLUMNS]

        # Replace any remaining NaN / Inf, then clip to reasonable bounds to
        # avoid exploding inputs (same handling the models were trained with).
        frame = frame.replace([np.inf, -np.inf], [1.0, -1.0]).fillna(0.0)
        frame = frame.clip(-5.0, 5.0).astype(np.float32)

        return frame

    except Exception:
        logger.exception("Feature computation failed for provided DataFrame")
        return None


def generate_features(df: pd.DataFrame) -> np.ndarray | None:
    """Compute the 15-feature observation vector for the most recent bar.

    Thin wrapper over :func:`compute_feature_frame` that returns the last row
    as a ``float32`` array of shape ``(MARKET_FEATURE_DIM,)``.

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
            sorted by date ascending.  At least 60 rows are recommended so
            that all indicators have enough history.

    Returns:
        A numpy float32 array of shape ``(MARKET_FEATURE_DIM,)``, or ``None``
        if the DataFrame is too short or feature computation fails.
    """
    frame = compute_feature_frame(df)
    if frame is None or frame.empty:
        return None

    last_close = float(df["close" if "close" in df.columns else "Close"].iloc[-1])
    if last_close == 0:
        return None

    return frame.iloc[-1].to_numpy(dtype=np.float32)


def build_guardian_obs(
    market_features: np.ndarray,
    unrealized_pnl_pct: float,
    bars_in_trade: float,
) -> np.ndarray:
    """Build the 17-dim Guardian observation for an open position.

    Args:
        market_features:    The 15-dim market feature vector for the symbol.
        unrealized_pnl_pct: (current_price - buy_price) / buy_price.
        bars_in_trade:      Number of bars the position has been open.

    Returns:
        float32 array of shape ``(GUARDIAN_DIM,)``.
    """
    pnl = float(np.clip(unrealized_pnl_pct, -0.5, 0.5))
    bars = min(float(bars_in_trade), 20.0) / 20.0
    obs = np.concatenate(
        [np.asarray(market_features, dtype=np.float32), [pnl, bars]]
    ).astype(np.float32)
    assert obs.shape == (GUARDIAN_DIM,)
    return obs


def build_executive_obs(
    market_features: np.ndarray,
    open_positions_frac: float,
    nifty_ret_5d: float,
) -> np.ndarray:
    """Build the 17-dim Executive observation for a buy candidate.

    Args:
        market_features:     The 15-dim market feature vector for the symbol.
        open_positions_frac: open_count / MAX_SLOTS, in [0, 1].
        nifty_ret_5d:        5-day return of ^NSEI (clipped to [-0.2, 0.2]).

    Returns:
        float32 array of shape ``(EXECUTIVE_DIM,)``.
    """
    frac = float(np.clip(open_positions_frac, 0.0, 1.0))
    nifty = float(np.clip(nifty_ret_5d, -0.2, 0.2))
    obs = np.concatenate(
        [np.asarray(market_features, dtype=np.float32), [frac, nifty]]
    ).astype(np.float32)
    assert obs.shape == (EXECUTIVE_DIM,)
    return obs
