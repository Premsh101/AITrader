# AITrader retraining package

Retrains the three PPO brains (Hunter, Guardian, Executive) **against the
repo's committed feature engine**. All feature code is imported from
`app/feature_engine.py` — the exact module the live app serves from — so a
train/serve observation mismatch (the bug that killed the previous models)
is impossible by construction.

| Brain     | Obs dim | Actions               | Output file                     |
|-----------|---------|-----------------------|---------------------------------|
| Hunter    | 15      | Discrete(2) hold/buy  | `hunter_apex_1500_brain.zip`    |
| Guardian  | 17      | Discrete(2) hold/close| `guardian_apex_1500_brain.zip`  |
| Executive | 17      | Discrete(2) skip/approve | `executive_apex_manager.zip` |

Key design points:

- **Data**: yfinance daily OHLCV for the repo universe (`app/stock_list.py`)
  plus `^NSEI`, period 6y.
- **Walk-forward split**: bars dated ≤ cutoff (18 months before today) train;
  bars strictly after validate. Nothing ever shuffles across the boundary.
- **Costs**: every reward pays `COST_BPS = 25` bps **per side**
  (`training/common.py`) — an approximation of brokerage + STT + slippage on
  NSE. A model that only learns cost-free edges is worthless.
- **Budget**: `--timesteps` defaults to 1M per brain. The previous
  Hunter/Guardian were trained ~5M steps; raise to `--timesteps 5000000` once
  the 1M run's evaluation looks sane.
- **Checkpoints** every 100k steps into `checkpoints/`, so Kaggle's 12-hour
  session limit can't lose a run (`--resume-from checkpoints/<file>.zip` with
  a single `--brains` selection to continue).

## Exact Kaggle steps

1. Create a new Kaggle notebook (CPU is fine; GPU doesn't help MlpPolicy at
   this size). In **Settings → Internet**, turn internet **ON**.

2. Cell 1 — clone the repo and install pinned deps (match `requirements.txt`):

   ```python
   !git clone https://github.com/Premsh101/AITrader.git
   %cd AITrader
   !pip install -q "stable-baselines3==2.8.0" "numpy>=2.0" "pandas_ta==0.4.67b0" "yfinance>=0.2.40"
   ```

3. Cell 2 — smoke-check the pipeline end to end (~2 minutes):

   ```python
   !python training/train_triad.py --smoke
   !python training/evaluate_triad.py --smoke
   ```

4. Cell 3 — full training run:

   ```python
   !python training/train_triad.py --timesteps 1000000
   ```

   (~a few hours on Kaggle CPU for all three. To fit inside one session you
   can train one brain per session: `--brains hunter`, then `guardian`, then
   `executive`, resuming from `checkpoints/` if interrupted.)

5. Cell 4 — evaluate (the go/no-go gate):

   ```python
   !python training/evaluate_triad.py
   ```

6. Download the three zips from `AITrader/models/` (right-hand *Output* panel
   or `Files` browser), then commit them to the repo's `models/` directory
   with the exact same filenames and redeploy.

## Go / no-go rule

`evaluate_triad.py` replays the **full pipeline** (Hunter → dedup →
Executive probability gate → Guardian, force-close at 20 bars) on the
held-out validation slice **with costs**, and prints total return, trade
count, mean per-trade return, win rate, max drawdown, and buy-and-hold
`^NSEI` over the same window, followed by a verdict line.

**If the strategy does not beat the `^NSEI` buy-and-hold baseline after
costs out-of-sample, do not deploy.** Do not "just try it live", do not
lower costs to make the number pass — iterate on the reward design in
`train_triad.py` (missed-opportunity threshold, drawdown penalty, entropy
coefficient, timesteps) and re-run the gate.

The app itself enforces the obs contract: `app/ai_brains.py` refuses to mark
a brain ready unless the loaded model's `observation_space` is exactly
(15,)/(17,)/(17,), so an incompatible zip can never silently trade again.
