# AITrader repair: obs-dim parity, symbol mapping, auth, retraining package

A code review found the system could not place a single correct trade. This
changeset repairs every finding and adds the retraining package needed to
produce usable models.

## Why the old models are gone from `models/`

The committed zips were trained on 5-dim (Hunter/Guardian) and 3-dim
(Executive) observations, while `app/feature_engine.py` produces 15 dims.
Every `predict()` raised a shape error that was swallowed at debug level, so
the Hunter emitted zero signals and the Guardian never closed a position.
The notebooks defining those features are lost, so the zips are unusable.
They now live in `models/legacy/` (see its README). **The app will refuse to
trade until you train new models** — see `training/README.md` for exact
Kaggle steps and the go/no-go evaluation gate.

## Backend changes

- **`app/symbols.py` (new)** – single source of truth for symbol formats.
  The DB and all internal logic store the base symbol (`RELIANCE`); Yahoo
  (`RELIANCE.NS`) and Shoonya (`NSE`/`RELIANCE-EQ`) formats exist only at
  the fetch/broker edges. This fixes the bug where Yahoo-format symbols were
  sent to the broker and failed on every automated order.
- **`app/feature_engine.py`** – refactored for train/serve parity:
  `compute_feature_frame(df)` returns the full 15-feature history (training
  imports this), `generate_features(df)` is now just its last row. Added
  `build_guardian_obs` (15 market features + clipped unrealized P&L % +
  bars-in-trade/20 → 17 dims) and `build_executive_obs` (15 + open-slot
  fraction + ^NSEI 5-day return → 17 dims), plus `MARKET_FEATURE_DIM`,
  `GUARDIAN_DIM`, `EXECUTIVE_DIM`.
- **`app/ai_brains.py`** – all heuristic fallbacks deleted. A brain is ready
  only if its model file loads AND its `observation_space` matches the
  expected dims (15/17/17); `BrainManager.all_ready` gates the trading loop.
  Inference errors now log at WARNING with the symbol. The Executive ranks
  candidates by approve *probability* (`policy.get_distribution`) instead of
  the old argmax "priority score" that produced arbitrary tie-broken
  rankings; approval requires probability > 0.5.
- **`app/security.py` (new)** – `require_api_key` dependency checks
  `X-API-Key` against `AITRADER_API_KEY`; if the env var is unset the
  protected routes return **503 (fail closed)**. Applied to
  `POST /toggle-mode`, `PATCH /config`, `POST /order`. GETs and `/health`
  stay open. CORS origins come from `ALLOWED_ORIGINS`. Switching to LIVE
  additionally requires `confirm: true` in the body.
- **`app/main.py` trading loop** –
  - Runs only Mon–Fri 09:15–15:30 IST (`TRADING_ALWAYS_ON=1` bypasses for
    testing); no new entries after 15:15 IST (broker squares off intraday
    product ~15:20), Guardian runs until close.
  - Cycle interval `TRADING_LOOP_INTERVAL` (default 300 s) with
    `FETCH_TTL_SECONDS` (default 900 s) caching of Yahoo data — daily bars
    don't change intraday, so Yahoo is no longer hammered every 60 s.
  - Skips the whole cycle with an ERROR (visible in the dashboard activity
    feed) when brains aren't ready. Never trades on heuristics.
  - Held symbols are excluded from Hunter signals before the Executive sees
    them (no more duplicate positions).
  - Guardian gets real 17-dim observations (unrealized P&L from last close,
    bars-in-trade from `created_at`; calendar days ≈ bars is a documented
    approximation). Executive gets the real ^NSEI 5-day return.
  - Orders carry a `reference_price` (last close) and are sized as
    `floor(TRADE_CAPITAL_PER_SLOT / price)` (env, default ₹10,000/slot)
    instead of hardcoded quantity 1.
- **`app/shoonya_service.py`** –
  - Paper orders never touch the broker or the live price feed; the fill is
    the `reference_price` and a missing/zero price raises `ValueError`
    (0.0-price trades are impossible now).
  - Live orders store `broker_order_id` (`norenordno`) and record the fill
    best-effort (post-placement LTP, falling back to reference price — true
    fills need the trade-book API; noted in code).
  - `get_live_price(symbol_base)` builds the Shoonya format internally; no
    more `ValueError` on the hot path.
- **`app/models.py`** – `Trade.broker_order_id` (String(50), nullable).
- **`app/stock_list.py`** – `ADANITRANS.NS` → `ADANIENSOL.NS` (2023 rename);
  `MCDOWELL-N.NS` removed because its successor `UNITDSPR.NS` was already in
  the list. *Note:* outbound access to Yahoo is blocked from the CI sandbox
  this change was authored in, so the replacements couldn't be re-verified
  live here; both are the documented NSE renames and the training smoke run
  (`train_triad.py --smoke`, which downloads real data) will confirm them —
  drop any symbol that fails there.
- **`requirements.txt`** – `stable-baselines3==2.8.0` + `numpy>=2.0`
  (matches the toolchain the models are trained with; verified to resolve
  cleanly on Python 3.12, the Dockerfile's base).

## Database migrations (`alembic/versions/` — new)

Previously the directory didn't exist, so `alembic upgrade head` in
`entrypoint.sh` did nothing and fresh deploys had no tables. There is now a
single idempotent `0001_initial`:

- Fresh database → creates `trades` (including `broker_order_id`) and
  `system_config`.
- **Existing deployments** (tables created outside Alembic) → detects the
  tables, only adds the missing `broker_order_id` column, and stamps the
  revision. Just run `alembic upgrade head` (the entrypoint already does);
  **no manual `alembic stamp` is needed.** Both paths were tested.

## Deployment / docker-compose

- API (`127.0.0.1:8005`), dashboard (`127.0.0.1:8006`) and Postgres
  (`127.0.0.1:5438`) are now bound to **loopback**. A reverse proxy
  (Coolify) should be the only public entry; if your proxy reaches the
  containers over a Docker network rather than the host loopback, adjust the
  bindings accordingly.
- New env vars (see `.env.example`): `AITRADER_API_KEY`, `ALLOWED_ORIGINS`,
  `NEXT_PUBLIC_API_KEY` (build arg for the dashboard), `TRADING_LOOP_INTERVAL`,
  `FETCH_TTL_SECONDS`, `TRADING_ALWAYS_ON`, `TRADE_CAPITAL_PER_SLOT`.

## Dashboard

- `lib/api.ts` sends `X-API-Key` (from `NEXT_PUBLIC_API_KEY`, baked in at
  build time) on mutating requests and passes `confirm: true` on the mode
  toggle; `Header.tsx` shows a browser confirmation before switching to LIVE.

## Training package (`training/` — new)

`train_triad.py` (with `--smoke`), `evaluate_triad.py`, and a README with
exact Kaggle steps. Feature code is **imported from `app.feature_engine`**
(never duplicated), data is split walk-forward (train ≤ cutoff 18 months
back, validate strictly after), every reward pays 25 bps/side costs, PPO
checkpoints every 100k steps, and the models save under the exact filenames
the app loads. `evaluate_triad.py` replays the full Hunter → dedup →
Executive → Guardian pipeline on the held-out slice and prints a verdict
against buy-and-hold ^NSEI. **If it doesn't beat the baseline after costs,
do not deploy.**

## Tests (`tests/` — new)

37 pytest cases covering: symbol mapper round-trips; feature shapes and
NaN-freeness; guardian/executive obs builders; paper-order safety (no broker
calls, `reference_price` required); API-key 401/503 behavior; the LIVE
confirm guard; and the ai_brains dim assertions (including that the legacy
5-dim/3-dim zips are refused).
