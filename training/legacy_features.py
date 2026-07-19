"""
legacy_features.py – The RECOVERED observation recipes of the legacy
``apex_1500`` models (models/legacy/), reconstructed from the original
design conversation.  Used ONLY by ``evaluate_legacy.py`` to give the old
models a fair, cost-aware backtest — never by the serving app.

Legacy observations (order matters):
  Hunter    (5,): [RSI(14)/100, MACD(12,26,9), VWAP_Dist, ATR(14)/Close, OBV_Trend]
  Guardian  (5,): [profit_pct, days_held/20, RSI(14)/100, MACD_Slope, ATR(14)/Close]
  Executive (3,): [free_slots/5, avg 14d volatility, portfolio profit%]
                  actions Discrete(3): 0=reject all, 1=accept 1, 2=accept up to 3

NaN fallbacks per the original scripts: RSI→0.5 (post-scaling), others→0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEGACY_HUNTER_DIM = 5
LEGACY_GUARDIAN_DIM = 5
LEGACY_EXECUTIVE_DIM = 3

LEGACY_COLUMNS = ["rsi_n", "macd", "vwap_dist", "atr_ratio", "obv_trend",
                  "macd_slope", "volatility"]


def compute_legacy_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    """Full history of every legacy feature for one symbol's OHLCV frame."""
    if df is None or len(df) < 30:
        return None

    import pandas_ta as ta

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    rsi = ta.rsi(close, length=14)
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is None or macd_df.empty:
        return None
    macd_col = next((c for c in macd_df.columns if c.upper().startswith("MACD_")), None)
    if macd_col is None:
        return None
    macd = macd_df[macd_col]

    # "Bulletproof VWMA": 20-day volume-weighted moving average.
    vwma = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    vwap_dist = (close - vwma) / (vwma + 1e-9)

    atr = ta.atr(high, low, close, length=14)
    if atr is None:
        atr = pd.Series(np.nan, index=close.index)
    atr_ratio = atr / (close + 1e-9)

    obv = ta.obv(close, volume)
    obv_trend = obv.pct_change() if obv is not None else pd.Series(np.nan, index=close.index)

    frame = pd.DataFrame(
        {
            "rsi_n": (rsi / 100.0).fillna(0.5),
            "macd": macd.fillna(0.0),
            "vwap_dist": vwap_dist.fillna(0.0),
            "atr_ratio": atr_ratio.fillna(0.0),
            "obv_trend": obv_trend.fillna(0.0),
            "macd_slope": macd.diff().fillna(0.0),
            "volatility": close.pct_change().rolling(14).std().fillna(0.0),
        },
        index=df.index,
    ).replace([np.inf, -np.inf], 0.0)

    return frame.astype(np.float32)


def hunter_obs(row: pd.Series) -> np.ndarray:
    return np.array(
        [row["rsi_n"], row["macd"], row["vwap_dist"], row["atr_ratio"], row["obv_trend"]],
        dtype=np.float32,
    )


def guardian_obs(profit_pct: float, days_held: float, row: pd.Series) -> np.ndarray:
    return np.array(
        [profit_pct, min(days_held, 20.0) / 20.0, row["rsi_n"], row["macd_slope"], row["atr_ratio"]],
        dtype=np.float32,
    )


def executive_obs(free_slots_frac: float, avg_volatility: float, portfolio_profit: float) -> np.ndarray:
    return np.array([free_slots_frac, avg_volatility, portfolio_profit], dtype=np.float32)
