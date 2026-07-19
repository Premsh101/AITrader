"""
ai_brains.py – Stable-Baselines3 model wrappers for the AITrader system.

Three AI brains:
  • HunterBrain    – scans 15-dim market feature vectors and emits BUY signals.
  • GuardianBrain  – watches 17-dim open-trade observations and emits CLOSE signals.
  • ExecutiveBrain – approves/ranks Hunter candidates from 17-dim observations.

There are NO heuristic fallbacks.  If a model file is missing, fails to load,
or its observation space does not match the expected dimension, the brain is
simply not ready (``BrainManager.all_ready`` is False) and the trading loop
must skip the cycle.  Trading on anything other than the trained models is
never allowed — that silent degradation is what previously masked a total
observation-shape mismatch.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from app.feature_engine import EXECUTIVE_DIM, GUARDIAN_DIM, MARKET_FEATURE_DIM

logger = logging.getLogger(__name__)

MODELS_DIR: str = os.environ.get("MODELS_DIR", "/app/models")
MAX_SLOTS: int = 5

# Action constants (must match the training environments in training/)
ACTION_HOLD = 0
ACTION_BUY_CLOSE = 1  # "BUY" for Hunter; "CLOSE" for Guardian; "APPROVE" for Executive

# Executive approval threshold: candidates whose approve-probability is at or
# below this are rejected regardless of ranking.
EXECUTIVE_APPROVE_THRESHOLD = float(
    os.environ.get("EXECUTIVE_APPROVE_THRESHOLD", "0.5")
)


def _load_sb3_model(path: str, expected_dim: int):
    """Load a Stable-Baselines3 PPO model and validate its observation space.

    Returns the model object, or ``None`` if the file is missing, loading
    fails, or ``observation_space.shape`` does not equal ``(expected_dim,)``.
    The shape assertion makes the original silent-mismatch failure impossible
    to reintroduce: a model trained against different features refuses to
    load rather than predicting garbage.
    """
    try:
        from stable_baselines3 import PPO  # type: ignore[import]

        if not os.path.exists(path):
            logger.error("Model file not found at %s – brain NOT ready.", path)
            return None

        # device="cpu": inference is a single small MLP forward pass; this also
        # keeps evaluation working on hosts whose GPU the installed PyTorch
        # doesn't support (e.g. Kaggle's Tesla P100).
        model = PPO.load(path, device="cpu")
        actual_shape = tuple(model.observation_space.shape)
        if actual_shape != (expected_dim,):
            logger.error(
                "Model %s observation shape %s does not match expected (%d,) "
                "– brain NOT ready. Retrain with training/train_triad.py.",
                path,
                actual_shape,
                expected_dim,
            )
            return None

        logger.info("Model loaded from %s (obs dim %d)", path, expected_dim)
        return model
    except Exception:
        logger.exception("Failed to load model from %s – brain NOT ready.", path)
        return None


# ---------------------------------------------------------------------------
# Hunter Brain
# ---------------------------------------------------------------------------


class HunterBrain:
    """Scans market feature vectors and identifies potential BUY candidates."""

    MODEL_FILE = "hunter_apex_1500_brain.zip"
    OBS_DIM = MARKET_FEATURE_DIM

    def __init__(self) -> None:
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        path = os.path.join(MODELS_DIR, self.MODEL_FILE)
        self._model = _load_sb3_model(path, self.OBS_DIM)

    def find_signals(self, symbol_features: dict[str, np.ndarray]) -> list[str]:
        """Return a list of symbols the Hunter flags as BUY.

        Args:
            symbol_features: Mapping of ``{symbol: 15-dim feature vector}``.

        Returns:
            List of symbols with a positive BUY signal.
        """
        if not self.ready:
            raise RuntimeError("Hunter model not loaded")

        signals: list[str] = []
        for symbol, obs in symbol_features.items():
            try:
                obs_2d = np.asarray(obs, dtype=np.float32).reshape(1, -1)
                action, _ = self._model.predict(obs_2d, deterministic=True)
                if int(np.asarray(action).reshape(-1)[0]) == ACTION_BUY_CLOSE:
                    signals.append(symbol)
            except Exception:
                logger.warning("Hunter inference failed for %s", symbol, exc_info=True)
        return signals


# ---------------------------------------------------------------------------
# Guardian Brain
# ---------------------------------------------------------------------------


class GuardianBrain:
    """Monitors open-trade observations and decides when to CLOSE positions."""

    MODEL_FILE = "guardian_apex_1500_brain.zip"
    OBS_DIM = GUARDIAN_DIM

    def __init__(self) -> None:
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        path = os.path.join(MODELS_DIR, self.MODEL_FILE)
        self._model = _load_sb3_model(path, self.OBS_DIM)

    def should_close(self, obs: np.ndarray, symbol: str = "?") -> bool:
        """Return ``True`` if the Guardian recommends closing this position.

        Args:
            obs:    17-dim observation from ``build_guardian_obs``.
            symbol: Symbol for log context.
        """
        if not self.ready:
            raise RuntimeError("Guardian model not loaded")

        try:
            obs_2d = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            action, _ = self._model.predict(obs_2d, deterministic=True)
            return int(np.asarray(action).reshape(-1)[0]) == ACTION_BUY_CLOSE
        except Exception:
            logger.warning("Guardian inference failed for %s", symbol, exc_info=True)
            return False


# ---------------------------------------------------------------------------
# Executive Brain
# ---------------------------------------------------------------------------


class ExecutiveBrain:
    """Approves and ranks Hunter candidates by approve probability."""

    MODEL_FILE = "executive_apex_manager.zip"
    OBS_DIM = EXECUTIVE_DIM

    def __init__(self) -> None:
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        path = os.path.join(MODELS_DIR, self.MODEL_FILE)
        self._model = _load_sb3_model(path, self.OBS_DIM)

    def approve_probability(self, obs: np.ndarray) -> float:
        """Return the policy's probability of the APPROVE action for *obs*.

        Ranking by probability (rather than the argmax action) breaks the ties
        a Discrete action value would produce, giving a real priority ordering.
        """
        if not self.ready:
            raise RuntimeError("Executive model not loaded")

        import torch

        obs_2d = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            obs_t, _ = self._model.policy.obs_to_tensor(obs_2d)
            dist = self._model.policy.get_distribution(obs_t)
            prob = float(dist.distribution.probs[0, ACTION_BUY_CLOSE].item())
        return prob

    def select_slots(
        self,
        signals: list[str],
        observations: dict[str, np.ndarray],
        open_slots: int,
    ) -> list[str]:
        """Choose at most *open_slots* symbols from *signals* to act on.

        Args:
            signals:      Symbols flagged by the Hunter (already deduplicated
                          against currently held positions by the caller).
            observations: 17-dim Executive observations keyed by symbol
                          (from ``build_executive_obs``).
            open_slots:   Number of free portfolio slots remaining.

        Returns:
            Symbols approved for entry, ordered by descending approve
            probability.  Only candidates with probability strictly above
            ``EXECUTIVE_APPROVE_THRESHOLD`` are approved at all.
        """
        if not signals or open_slots <= 0:
            return []
        if not self.ready:
            raise RuntimeError("Executive model not loaded")

        limit = min(open_slots, MAX_SLOTS)

        scores: list[tuple[float, str]] = []
        for sym in signals:
            obs = observations.get(sym)
            if obs is None:
                continue
            try:
                prob = self.approve_probability(obs)
            except Exception:
                logger.warning("Executive inference failed for %s", sym, exc_info=True)
                continue
            if prob > EXECUTIVE_APPROVE_THRESHOLD:
                scores.append((prob, sym))

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

    @property
    def all_ready(self) -> bool:
        """True only when every brain has a validated model loaded."""
        return self.hunter.ready and self.guardian.ready and self.executive.ready

    def load_all(self) -> None:
        """Load all three brain models from disk and report readiness."""
        self.hunter.load()
        self.guardian.load()
        self.executive.load()
        if self.all_ready:
            logger.info("All AI brains loaded and validated.")
        else:
            logger.error(
                "AI brains NOT ready (hunter=%s guardian=%s executive=%s) – "
                "trading is disabled until valid models are present.",
                self.hunter.ready,
                self.guardian.ready,
                self.executive.ready,
            )
