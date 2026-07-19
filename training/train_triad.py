"""
train_triad.py – Retrain the AITrader model triad (Hunter, Guardian,
Executive) against the repo's committed feature engine.

All observations are produced by ``app.feature_engine`` — the same module the
live app serves from — so train/serve parity is guaranteed by construction.

Environments (all PPO, MlpPolicy, Discrete(2)):
  • HunterEnv    (obs 15): BUY → forward 5-bar return − round-trip cost;
                 HOLD → small missed-opportunity penalty only when the move
                 was actually worth taking.
  • GuardianEnv  (obs 17): simulates an open long; CLOSE realises P&L − cost;
                 HOLD pays a time penalty plus a drawdown-from-peak penalty;
                 force-close at 20 bars.
  • ExecutiveEnv (obs 17): APPROVE → forward return − round-trip cost;
                 SKIP → ~0 with a small missed-opportunity penalty.

Usage:
  python training/train_triad.py                 # full run (1M steps each)
  python training/train_triad.py --smoke         # 5 symbols, 5k steps
  python training/train_triad.py --timesteps 5000000 --brains hunter

Checkpoints are written every 100k steps so a 12-hour Kaggle session limit
never loses a run; resume by pointing --resume-from at a checkpoint zip.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Allow `python training/train_triad.py` from the repo root (or Kaggle).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.common import (  # noqa: E402
    EXECUTIVE_FILE,
    FORWARD_BARS,
    GUARDIAN_FILE,
    HUNTER_FILE,
    MAX_SLOTS,
    MAX_TRADE_BARS,
    MISSED_OPPORTUNITY_PENALTY,
    MISSED_OPPORTUNITY_THRESHOLD,
    ROUND_TRIP_COST,
    Dataset,
    SymbolData,
    build_dataset,
    download_universe,
    nifty_ret_lookup,
)

from app.feature_engine import (  # noqa: E402
    EXECUTIVE_DIM,
    GUARDIAN_DIM,
    MARKET_FEATURE_DIM,
    build_executive_obs,
    build_guardian_obs,
)
from app.stock_list import NSE_SYMBOLS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Default training budget per brain.  The previous (lost) Hunter/Guardian
# models were trained for ~5M steps; 1M is a faster starting point — raise
# --timesteps to 5_000_000 for a comparable budget once rewards look sane.
TOTAL_TIMESTEPS = 1_000_000

CHECKPOINT_EVERY = 100_000
EPISODE_LEN = 64          # steps per episode for the sampling envs
REWARD_CLIP = 0.25

ACTION_HOLD = 0
ACTION_ACT = 1  # BUY / CLOSE / APPROVE


def _random_sample_point(
    rng: np.random.Generator,
    data: dict[str, SymbolData],
    min_forward: int,
) -> tuple[str, SymbolData, int]:
    """Pick a random (symbol, t) leaving at least *min_forward* bars ahead."""
    syms = list(data.keys())
    while True:
        sym = syms[rng.integers(len(syms))]
        sd = data[sym]
        max_t = len(sd.close) - min_forward - 1
        if max_t <= 0:
            continue
        t = int(rng.integers(0, max_t))
        return sym, sd, t


def _forward_return(sd: SymbolData, t: int, horizon: int = FORWARD_BARS) -> float:
    return float(sd.close[t + horizon] / sd.close[t] - 1.0)


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------


class HunterEnv(gym.Env):
    """Contextual-bandit style scanner: each step is a fresh (symbol, t)."""

    metadata = {"render_modes": []}

    def __init__(self, data: dict[str, SymbolData], seed: int = 0) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(MARKET_FEATURE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self._data = data
        self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._sd: SymbolData | None = None
        self._t = 0

    def _next_obs(self) -> np.ndarray:
        _, self._sd, self._t = _random_sample_point(
            self._rng, self._data, FORWARD_BARS
        )
        return self._sd.features[self._t]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        return self._next_obs(), {}

    def step(self, action):
        fwd = _forward_return(self._sd, self._t)
        if action == ACTION_ACT:
            reward = fwd - ROUND_TRIP_COST
        else:
            reward = (
                MISSED_OPPORTUNITY_PENALTY
                if fwd > MISSED_OPPORTUNITY_THRESHOLD
                else 0.0
            )
        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

        self._steps += 1
        terminated = self._steps >= EPISODE_LEN
        return self._next_obs(), reward, terminated, False, {}


class GuardianEnv(gym.Env):
    """Simulates managing one open long position from a random entry."""

    metadata = {"render_modes": []}

    TIME_PENALTY = -0.001
    DRAWDOWN_PENALTY_SCALE = 0.02

    def __init__(self, data: dict[str, SymbolData], seed: int = 0) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(GUARDIAN_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self._data = data
        self._rng = np.random.default_rng(seed)
        self._sd: SymbolData | None = None
        self._entry_t = 0
        self._bars = 0
        self._entry_price = 0.0
        self._peak = 0.0

    def _obs(self) -> np.ndarray:
        t = self._entry_t + self._bars
        price = self._sd.close[t]
        pnl_pct = price / self._entry_price - 1.0
        return build_guardian_obs(self._sd.features[t], pnl_pct, self._bars)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _, self._sd, self._entry_t = _random_sample_point(
            self._rng, self._data, MAX_TRADE_BARS + 1
        )
        self._bars = 0
        self._entry_price = float(self._sd.close[self._entry_t])
        self._peak = self._entry_price
        return self._obs(), {}

    def _close_reward(self) -> float:
        t = self._entry_t + self._bars
        pnl_pct = self._sd.close[t] / self._entry_price - 1.0
        return float(np.clip(pnl_pct - ROUND_TRIP_COST, -REWARD_CLIP, REWARD_CLIP))

    def step(self, action):
        if action == ACTION_ACT:  # CLOSE
            return self._obs(), self._close_reward(), True, False, {}

        # HOLD: advance one bar
        self._bars += 1
        t = self._entry_t + self._bars
        price = float(self._sd.close[t])
        self._peak = max(self._peak, price)
        drawdown = (self._peak - price) / self._peak

        if self._bars >= MAX_TRADE_BARS:
            # Forced exit at the bar cap (mirrors serving-side reality).
            return self._obs(), self._close_reward(), True, False, {}

        reward = self.TIME_PENALTY - self.DRAWDOWN_PENALTY_SCALE * drawdown
        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
        return self._obs(), reward, False, False, {}


class ExecutiveEnv(gym.Env):
    """Approve/skip gate over randomly sampled entry candidates."""

    metadata = {"render_modes": []}

    def __init__(self, data: dict[str, SymbolData], ds: Dataset, seed: int = 0) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(EXECUTIVE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self._data = data
        self._ds = ds
        self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._sd: SymbolData | None = None
        self._t = 0

    def _next_obs(self) -> np.ndarray:
        _, self._sd, self._t = _random_sample_point(
            self._rng, self._data, FORWARD_BARS
        )
        # Portfolio fullness is randomised so the policy sees every regime;
        # the market context is the REAL ^NSEI 5-day return at that date.
        open_frac = float(self._rng.integers(0, MAX_SLOTS + 1)) / MAX_SLOTS
        nifty = nifty_ret_lookup(self._ds, self._sd.dates[self._t])
        return build_executive_obs(self._sd.features[self._t], open_frac, nifty)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        return self._next_obs(), {}

    def step(self, action):
        fwd = _forward_return(self._sd, self._t)
        if action == ACTION_ACT:  # APPROVE
            reward = fwd - ROUND_TRIP_COST
        else:  # SKIP
            reward = (
                MISSED_OPPORTUNITY_PENALTY
                if fwd > MISSED_OPPORTUNITY_THRESHOLD
                else 0.0
            )
        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

        self._steps += 1
        terminated = self._steps >= EPISODE_LEN
        return self._next_obs(), reward, terminated, False, {}


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def train_one(
    name: str,
    env: gym.Env,
    out_path: str,
    timesteps: int,
    checkpoint_dir: str,
    resume_from: str | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(env)

    # Default is CPU: PPO with MlpPolicy gains nothing from a GPU (SB3 itself
    # warns about this — the policy net is tiny and the bottleneck is the env),
    # and Kaggle sometimes hands out GPUs (e.g. Tesla P100/sm_60) that current
    # PyTorch wheels no longer ship kernels for, crashing with "no kernel image
    # is available for execution on the device".  --device cuda is accepted for
    # supported GPUs (e.g. T4) but is not expected to be faster.
    if resume_from:
        logger.info("[%s] resuming from %s", name, resume_from)
        model = PPO.load(resume_from, env=env, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            ent_coef=0.01,
            seed=seed,
            verbose=1,
            device=device,
        )

    callback = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY,
        save_path=checkpoint_dir,
        name_prefix=name,
    )
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    model.save(out_path)
    logger.info("[%s] saved → %s", name, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"PPO timesteps per brain (default {TOTAL_TIMESTEPS:,}; the old "
        "Hunter/Guardian used ~5M)",
    )
    parser.add_argument(
        "--brains", nargs="+", default=["hunter", "guardian", "executive"],
        choices=["hunter", "guardian", "executive"],
        help="Which brains to train (default: all three)",
    )
    parser.add_argument(
        "--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "models"),
        help="Directory for the final model zips (default: repo models/)",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Directory for periodic checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Checkpoint zip to resume the (single) selected brain from",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "auto"],
        help="Torch device for PPO (default cpu; cuda works on supported GPUs "
        "like the T4 but is typically no faster for MlpPolicy)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Fast end-to-end check: 5 symbols, 5k timesteps per brain",
    )
    args = parser.parse_args()

    symbols_subset = NSE_SYMBOLS[:5] if args.smoke else None
    timesteps = 5_000 if args.smoke else args.timesteps

    ohlcv = download_universe(symbols=symbols_subset)
    ds = build_dataset(ohlcv)
    if not ds.train:
        raise RuntimeError("No training data – check the symbol universe / network")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    jobs = {
        "hunter": (HunterEnv(ds.train, seed=args.seed), HUNTER_FILE),
        "guardian": (GuardianEnv(ds.train, seed=args.seed), GUARDIAN_FILE),
        "executive": (ExecutiveEnv(ds.train, ds, seed=args.seed), EXECUTIVE_FILE),
    }

    for name in args.brains:
        env, filename = jobs[name]
        train_one(
            name=name,
            env=env,
            out_path=os.path.join(out_dir, filename),
            timesteps=timesteps,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from if len(args.brains) == 1 else None,
            seed=args.seed,
            device=args.device,
        )

    logger.info(
        "Training complete. Run `python training/evaluate_triad.py` and only "
        "deploy if the strategy beats buy-and-hold ^NSEI after costs."
    )


if __name__ == "__main__":
    main()
