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

## v5 strategy: reward alpha, not drift

The gate is "beat buy-and-hold ^NSEI after costs", so the entry brains
(Hunter, Executive) are now rewarded for **excess return over ^NSEI**, not
raw forward return. Buying a stock that rises while the index rises just as
much earns ≈0 and is a net loss after costs — you could have held the index
for free. Only genuine out-performers pay off.

This is the fix for the v1–v4 failure mode: the old reward
(`forward_return − cost`) taught the models to buy *any* up-drift, so in a
rising market they made 350–850 trades, none with a real edge, and always
finished below the baseline. Rewarding alpha forces selectivity — most
stocks do **not** beat the index, so the models learn to trade only the ones
that do. Supporting changes: reward horizon 5→10 bars (matches the ~20-bar
live hold, less noise), training-cost multiplier 2.0→1.5 (the benchmark
subtraction now supplies the selectivity pressure), and a gentler Guardian
time-penalty so winners run long enough to amortise costs.

Key design points:

- **Data**: yfinance daily OHLCV for the repo universe (`app/stock_list.py`)
  plus `^NSEI`, period 6y.
- **Walk-forward split**: bars dated ≤ cutoff (18 months before today) train;
  bars strictly after validate. Nothing ever shuffles across the boundary.
- **Costs**: every reward pays `COST_BPS = 25` bps **per side**
  (`training/common.py`) — an approximation of brokerage + STT + slippage on
  NSE. A model that only learns cost-free edges is worthless.
- **Budget**: `--timesteps` defaults to 1M per brain — a fast first
  iteration, NOT the final word. The previous Hunter/Guardian were trained
  ~5M steps (the Executive ~1M). To match that budget exactly:

  ```
  python training/train_triad.py --brains hunter    --timesteps 5000000
  python training/train_triad.py --brains guardian  --timesteps 5000000
  python training/train_triad.py --brains executive --timesteps 1000000
  ```

  Expect very roughly 2–5 hours per 5M-step brain on Kaggle CPU. More steps
  generally help but with diminishing returns — the out-of-sample evaluation
  verdict, not wall-clock time, is the measure of quality. Run the 1M pass
  first to check the pipeline, then spend the 5M budget.
- **Checkpoints** every 100k steps into `checkpoints/`, so Kaggle's 12-hour
  session limit can't lose a run (`--resume-from checkpoints/<file>.zip` with
  a single `--brains` selection to continue).

## Exact Kaggle steps

1. Create a new Kaggle notebook. In **Settings → Internet**, turn internet
   **ON**. Accelerator can stay **None (CPU)** — training is forced onto the
   CPU regardless (a GPU doesn't help MlpPolicy, and some Kaggle GPUs like the
   Tesla P100 aren't supported by current PyTorch wheels anyway).

2. Cell 1 — clone the repo and install pinned deps (match `requirements.txt`):

   ```python
   !git clone https://github.com/Premsh101/AITrader.git
   %cd AITrader
   !pip install -q "stable-baselines3==2.8.0" "numpy>=2.0" "pandas_ta==0.4.67b0" "yfinance>=0.2.40"
   ```

   pip may print "dependency resolver" complaints about Kaggle's own
   pre-installed packages (bigframes, google-colab, dopamine-rl, …) — they
   are unrelated to this project and safe to ignore.

3. Cell 2 — smoke-check the pipeline end to end (~2 minutes):

   ```python
   !python training/train_triad.py --smoke
   !python training/evaluate_triad.py --smoke
   ```

   Any ticker that fails to download (renamed/delisted on Yahoo) is skipped
   with a logged warning and training continues with the rest — if you see
   failures, update or remove those names in `app/stock_list.py`.

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

## Multi-seed tournament & threshold sweep

`train_triad.py --seeds N` trains N independent triads (saved with `_s0`,
`_s1`, … suffixes). `tournament.py --seeds N` then backtests every variant
on the held-out window **and sweeps the Executive approve threshold**
(default `0.50,0.60,0.65,0.70`) for each one. Raising the threshold makes
the Executive reject lower-confidence entries — fewer, higher-conviction
trades — and because the sweep only re-grades existing zips it takes
minutes, not a training run:

```python
!python training/tournament.py --seeds 3
# or a custom sweep on zips you already have:
!python training/tournament.py --seeds 3 --thresholds 0.5,0.65,0.75
```

The winner is the best (variant, threshold) pair; its zips are promoted to
the canonical filenames. If the verdict is PASS, deploy with the printed
`EXECUTIVE_APPROVE_THRESHOLD` value in `.env` so serving uses the same
entry bar the winner was graded at.

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
