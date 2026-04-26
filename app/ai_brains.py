"""
ai_brains.py – Stable-Baselines3 model wrappers for the AITrader system.

Three AI brains:
  • HunterBrain    – scans feature vectors and emits BUY signals.
  • GuardianBrain  – watches open-trade feature vectors and emits CLOSE signals.
  • ExecutiveBrain – selects at most MAX_SLOTS signals from the Hunter's output.

All brains load their models lazily and degrade gracefully when the .zip files
are absent (useful for local development / CI without model artefacts).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

MODELS_DIR: str = os.environ.get("MODELS_DIR", "/app/models")
MAX_SLOTS: int = 5

# Action constants (must match the training environment)
ACTION_HOLD = 0
ACTION_BUY_CLOSE = 1  # "BUY" for Hunter; "CLOSE" for Guardian


# ---------------------------------------------------------------------------
# Base helper
# ---------------------------------------------------------------------------


def _load_sb3_model(path: str):
    """Load a Stable-Baselines3 PPO model from *path*.

    Returns the model object, or ``None`` if loading fails.
    """
    try:
        from stable_baselines3 import PPO  # type: ignore[import]

        model = PPO.load(path)
        logger.info("Model loaded from %s", path)
        return model
    except FileNotFoundError:
        logger.warning("Model file not found at %s – brain will use random fallback.", path)
        return None
    except Exception:
        logger.exception("Failed to load model from %s", path)
        return None


# ---------------------------------------------------------------------------
# Hunter Brain
# ---------------------------------------------------------------------------


class HunterBrain:
    """Scans market feature vectors and identifies potential BUY candidates."""

    MODEL_FILE = "hunter_apex_1500_brain.zip"

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        path = os.path.join(MODELS_DIR, self.MODEL_FILE)
        self._model = _load_sb3_model(path)

    def find_signals(self, symbol_features: dict[str, np.ndarray]) -> list[str]:
        """Return a list of symbol strings that the Hunter flags as BUY.

        Args:
            symbol_features: Mapping of ``{symbol: feature_vector}``.

        Returns:
            List of symbols with a positive BUY signal.
        """
        signals: list[str] = []
        for symbol, obs in symbol_features.items():
            try:
                action = self._predict(obs)
                if action == ACTION_BUY_CLOSE:
                    signals.append(symbol)
            except Exception:
                logger.debug("Hunter inference failed for %s", symbol, exc_info=True)
        return signals

    def _predict(self, obs: np.ndarray) -> int:
        if self._model is not None:
            obs_2d = obs.reshape(1, -1)
            action, _ = self._model.predict(obs_2d, deterministic=True)
            return int(action)
        # Fallback heuristic: RSI < 35 (oversold territory) → BUY signal.
        # 0.35 corresponds to RSI=35 (scaled to [0,1]) – a conservative oversold threshold.
        return ACTION_BUY_CLOSE if float(obs[1]) < 0.35 else ACTION_HOLD


# ---------------------------------------------------------------------------
# Guardian Brain
# ---------------------------------------------------------------------------


class GuardianBrain:
    """Monitors open-trade feature vectors and decides when to CLOSE positions."""

    MODEL_FILE = "guardian_apex_1500_brain.zip"

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        path = os.path.join(MODELS_DIR, self.MODEL_FILE)
        self._model = _load_sb3_model(path)

    def should_close(self, obs: np.ndarray) -> bool:
        """Return ``True`` if the Guardian recommends closing this position."""
        try:
            action = self._predict(obs)
            return action == ACTION_BUY_CLOSE
        except Exception:
            logger.debug("Guardian inference failed", exc_info=True)
            return False

    def _predict(self, obs: np.ndarray) -> int:
        if self._model is not None:
            obs_2d = obs.reshape(1, -1)
            action, _ = self._model.predict(obs_2d, deterministic=True)
            return int(action)
        # Fallback heuristic:
        #   RSI > 70 (overbought, scaled: 0.70) → take profit.
        #   1-day return < -2% → stop-loss trigger.
        rsi = float(obs[1])
        ret_1d = float(obs[10])
        return ACTION_BUY_CLOSE if (rsi > 0.70 or ret_1d < -0.02) else ACTION_HOLD


# ---------------------------------------------------------------------------
# Executive Brain
# ---------------------------------------------------------------------------


class ExecutiveBrain:
    """Selects up to MAX_SLOTS signals from the Hunter's BUY candidates."""

    MODEL_FILE = "executive_apex_manager.zip"

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        path = os.path.join(MODELS_DIR, self.MODEL_FILE)
        self._model = _load_sb3_model(path)

    def select_slots(
        self,
        signals: list[str],
        symbol_features: dict[str, np.ndarray],
        open_slots: int,
    ) -> list[str]:
        """Choose at most *open_slots* symbols from *signals* to act on.

        Args:
            signals:         Symbols flagged by the Hunter.
            symbol_features: Feature vectors keyed by symbol.
            open_slots:      Number of free portfolio slots remaining.

        Returns:
            Ordered list of symbols the Executive approves for entry.
        """
        if not signals or open_slots <= 0:
            return []

        limit = min(open_slots, MAX_SLOTS)

        if self._model is None:
            # Fallback: rank by lowest RSI (most oversold first)
            ranked = sorted(
                signals,
                key=lambda s: float(symbol_features.get(s, np.zeros(15))[1]),
            )
            return ranked[:limit]

        # Score each candidate using the Executive model
        scores: list[tuple[float, str]] = []
        for sym in signals:
            obs = symbol_features.get(sym)
            if obs is None:
                continue
            try:
                obs_2d = obs.reshape(1, -1)
                action, _ = self._model.predict(obs_2d, deterministic=True)
                # Higher action value → higher priority
                scores.append((float(action), sym))
            except Exception:
                scores.append((0.0, sym))

        scores.sort(reverse=True)
        return [sym for _, sym in scores[:limit]]


# ---------------------------------------------------------------------------
# Brain Manager (convenience facade)
# ---------------------------------------------------------------------------


class BrainManager:
    """Loads and exposes all three AI brains as a single object."""

    def __init__(self) -> None:
        self.hunter = HunterBrain()
        self.guardian = GuardianBrain()
        self.executive = ExecutiveBrain()

    def load_all(self) -> None:
        """Load all three brain models from disk."""
        self.hunter.load()
        self.guardian.load()
        self.executive.load()
        logger.info("All AI brains loaded.")
