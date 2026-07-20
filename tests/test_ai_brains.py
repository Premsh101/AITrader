"""Tests for ai_brains load-time observation-dimension assertions."""

import numpy as np
import pytest

import app.ai_brains as ai_brains
from app.feature_engine import MARKET_FEATURE_DIM


def _save_tiny_ppo(path: str, obs_dim: int, n_actions: int = 2) -> None:
    """Train-free PPO with the given obs dim, saved to *path*."""
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO

    class DummyEnv(gym.Env):
        observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        action_space = spaces.Discrete(n_actions)

        def reset(self, *, seed=None, options=None):
            return np.zeros(obs_dim, np.float32), {}

        def step(self, action):
            return np.zeros(obs_dim, np.float32), 0.0, True, False, {}

    model = PPO("MlpPolicy", DummyEnv(), n_steps=8, batch_size=8,
                policy_kwargs={"net_arch": [8]})
    model.save(path)


def test_missing_model_file_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_brains, "MODELS_DIR", str(tmp_path))
    brain = ai_brains.HunterBrain()
    brain.load()
    assert not brain.ready


def test_wrong_dim_model_refused(tmp_path, monkeypatch):
    """A 5-dim model (the legacy shape) must FAIL the 15-dim assertion."""
    monkeypatch.setattr(ai_brains, "MODELS_DIR", str(tmp_path))
    _save_tiny_ppo(str(tmp_path / ai_brains.HunterBrain.MODEL_FILE), obs_dim=5)

    brain = ai_brains.HunterBrain()
    brain.load()
    assert not brain.ready


def test_correct_dim_model_accepted_and_predicts(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_brains, "MODELS_DIR", str(tmp_path))
    _save_tiny_ppo(
        str(tmp_path / ai_brains.HunterBrain.MODEL_FILE),
        obs_dim=MARKET_FEATURE_DIM,
    )

    brain = ai_brains.HunterBrain()
    brain.load()
    assert brain.ready

    signals = brain.find_signals({"TCS": np.zeros(MARKET_FEATURE_DIM, np.float32)})
    assert isinstance(signals, list)


def test_manager_not_all_ready_when_any_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_brains, "MODELS_DIR", str(tmp_path))
    _save_tiny_ppo(
        str(tmp_path / ai_brains.HunterBrain.MODEL_FILE),
        obs_dim=MARKET_FEATURE_DIM,
    )
    manager = ai_brains.BrainManager()
    manager.load_all()
    assert manager.hunter.ready
    assert not manager.guardian.ready
    assert not manager.all_ready


def test_executive_probability_ranking(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_brains, "MODELS_DIR", str(tmp_path))
    _save_tiny_ppo(
        str(tmp_path / ai_brains.ExecutiveBrain.MODEL_FILE),
        obs_dim=ai_brains.ExecutiveBrain.OBS_DIM,
    )
    brain = ai_brains.ExecutiveBrain()
    brain.load()
    assert brain.ready

    obs = np.zeros(ai_brains.ExecutiveBrain.OBS_DIM, np.float32)
    prob = brain.approve_probability(obs)
    assert 0.0 <= prob <= 1.0

    selected = brain.select_slots(["A", "B"], {"A": obs, "B": obs}, open_slots=2)
    # Untrained model → prob ≈ 0.5; either it approves both or neither, but
    # never errors and never exceeds the slot budget.
    assert len(selected) <= 2


def test_executive_threshold_gates_selection(tmp_path, monkeypatch):
    """Patching the module-level threshold (as tournament.py's sweep does)
    must tighten/loosen the entry bar without reloading the model."""
    monkeypatch.setattr(ai_brains, "MODELS_DIR", str(tmp_path))
    _save_tiny_ppo(
        str(tmp_path / ai_brains.ExecutiveBrain.MODEL_FILE),
        obs_dim=ai_brains.ExecutiveBrain.OBS_DIM,
    )
    brain = ai_brains.ExecutiveBrain()
    brain.load()
    obs = np.zeros(ai_brains.ExecutiveBrain.OBS_DIM, np.float32)

    monkeypatch.setattr(ai_brains, "EXECUTIVE_APPROVE_THRESHOLD", 0.99)
    assert brain.select_slots(["A", "B"], {"A": obs, "B": obs}, open_slots=2) == []

    monkeypatch.setattr(ai_brains, "EXECUTIVE_APPROVE_THRESHOLD", 0.0)
    assert len(brain.select_slots(["A", "B"], {"A": obs, "B": obs}, open_slots=2)) == 2


def test_brains_raise_when_not_loaded():
    hunter = ai_brains.HunterBrain()
    with pytest.raises(RuntimeError):
        hunter.find_signals({"TCS": np.zeros(MARKET_FEATURE_DIM, np.float32)})

    guardian = ai_brains.GuardianBrain()
    with pytest.raises(RuntimeError):
        guardian.should_close(np.zeros(ai_brains.GuardianBrain.OBS_DIM, np.float32))
