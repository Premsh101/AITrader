"""
feature_engine.py – Technical-indicator feature generation for AITrader.

Given a single-symbol OHLCV DataFrame (columns: open, high, low, close, volume),
``generate_features`` returns a 1-D numpy float32 array that can be fed directly
into a Stable-Baselines3 model.

Feature vector layout (15 values):
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
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_DIM = 15


def generate_features(df: pd.DataFrame) -> np.ndarray | None:
    """Compute the 15-feature observation vector from an OHLCV DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close, volume] sorted by
            date ascending.  At least 60 rows are recommended so that all
            indicators have enough history.

    Returns:
        A numpy float32 array of shape ``(FEATURE_DIM,)``, or ``None`` if the
        DataFrame is too short or contains too many NaN values.
    """
    if df is None or len(df) < 30:
        return None

    # Work on a copy; ensure lower-case column names
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    try:
        import pandas_ta as ta  # type: ignore[import]

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ── Moving averages ───────────────────────────────────────────────
        sma20 = close.rolling(20).mean()
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        # ── RSI ───────────────────────────────────────────────────────────
        rsi_series = ta.rsi(close, length=14)

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

        # ── Assemble last row ─────────────────────────────────────────────
        last = -1  # index of the most recent bar
        last_close = float(close.iloc[last])
        if last_close == 0:
            return None

        features = np.array(
            [
                float(close.iloc[last]) / (float(sma20.iloc[last]) + 1e-9),  # 0
                float(rsi_series.iloc[last]) / 100.0,                         # 1
                float(macd_line.iloc[last]) / last_close,                     # 2
                float(macd_sig.iloc[last]) / last_close,                      # 3
                float(macd_hist.iloc[last]) / last_close,                     # 4
                float(bb_pct.iloc[last]),                                      # 5
                float(ema9.iloc[last]) / (float(ema21.iloc[last]) + 1e-9),    # 6
                float(ema21.iloc[last]) / (float(ema50.iloc[last]) + 1e-9),   # 7
                float(vol_ratio.iloc[last]),                                   # 8
                float(atr_norm.iloc[last]),                                    # 9
                float(ret_1d.iloc[last]),                                      # 10
                float(ret_5d.iloc[last]),                                      # 11
                float(rsi_delta.iloc[last]) / 100.0,                          # 12
                float(stoch_k.iloc[last]) / 100.0,                            # 13
                float(stoch_d.iloc[last]) / 100.0,                            # 14
            ],
            dtype=np.float32,
        )

        # Replace any remaining NaN / Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clip to reasonable bounds to avoid exploding inputs
        features = np.clip(features, -5.0, 5.0)

        return features

    except Exception:
        logger.exception("Feature generation failed for provided DataFrame")
        return None
