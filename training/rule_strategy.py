"""
rule_strategy.py – "CLASSIC" rule-based challenger: the best-documented
retail-viable strategy (trend-following momentum with volume confirmation)
encoded as plain rules, exposed through the SAME interface as the RL triad
so ``run_backtest`` grades both identically.

The rules (all long-standing, published results, none invented here):
  ENTRY  – price above its 20-SMA in an aligned EMA uptrend, positive 5-day
           momentum, above-average volume, RSI strong but not overbought.
  RANK   – strongest 5-day momentum first (classic cross-sectional momentum).
  EXIT   – trend break (EMA alignment lost / price back under the SMA) or
           overbought blow-off; the deterministic overlays (-5% stop, profit
           ladder, 20-bar cap) and regime filter apply exactly as for the RL
           models because the backtest enforces them for every contestant.

Feature indices (app.feature_engine.FEATURE_COLUMNS):
  0 close_ratio  1 rsi  6 ema9_ratio  7 ema21_ratio  8 vol_ratio  11 ret_5d
"""

from __future__ import annotations

import numpy as np


class RuleHunter:
    ready = True

    def find_signals(self, symbol_features: dict[str, np.ndarray]) -> list[str]:
        signals = []
        for sym, f in symbol_features.items():
            if (
                f[0] > 1.03          # price 3% above 20-SMA (established trend)
                and f[6] > 1.0       # EMA9 > EMA21
                and f[7] > 1.0       # EMA21 > EMA50 (aligned uptrend)
                and f[11] > 0.03     # +3% momentum over 5 days (higher bar →
                                     # fewer, higher-conviction entries)
                and f[8] > 1.3       # volume 30% above average (confirmation)
                and 0.55 < f[1] < 0.72  # RSI strong, not overbought
            ):
                signals.append(sym)
        return signals


class RuleGuardian:
    ready = True

    def should_close(self, obs: np.ndarray, symbol: str = "?") -> bool:
        # v5: exit only on a CLEAR trend break or blow-off, so winners are not
        # whipsawed out on a one-day dip (the main drag on the v4 rule run —
        # it cut winners early and churned).  The -5% stop, profit ladder and
        # 20-bar cap overlays still bound risk and time; here we simply stop
        # closing on shallow noise.
        return bool(obs[6] < 0.97 or obs[0] < 0.95 or obs[1] > 0.82)


class RuleExecutive:
    ready = True

    def select_slots(self, signals, observations, open_slots):
        ranked = sorted(
            (s for s in signals if s in observations),
            key=lambda s: float(observations[s][11]),  # 5-day momentum
            reverse=True,
        )
        return ranked[:open_slots]


class RuleBrains:
    """Drop-in replacement for BrainManager in run_backtest."""

    all_ready = True

    def __init__(self) -> None:
        self.hunter = RuleHunter()
        self.guardian = RuleGuardian()
        self.executive = RuleExecutive()
