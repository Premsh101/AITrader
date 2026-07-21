# Deploying AITrader on Coolify with Neon Postgres

This is the paper-trading deployment: the app runs in **PAPER mode** (no real
broker orders) and retrains itself **daily**, keeping any improvement that
beats buy-and-hold ^NSEI after costs.

## 1. Create the Neon database

1. Create a Neon project and a database named `aitrader`.
2. Copy the **pooled** connection string (the host contains `-pooler`). It
   looks like:
   ```
   postgresql://<user>:<password>@ep-xxxx-pooler.<region>.aws.neon.tech/aitrader?sslmode=require
   ```
   `sslmode=require` must be present — Neon only accepts TLS connections.
   The pooled endpoint (PgBouncer) suits the app's connection pool; psycopg2
   does not use server-side prepared statements, so transaction pooling is fine.
3. No schema setup needed — the app runs `alembic upgrade head` on boot.

## 2. Create the Coolify resource

1. **New Resource → Docker Compose**, pointing at this GitHub repo, branch
   `main`.
2. Set the compose file path to **`docker-compose.coolify.yml`** (not the
   default `docker-compose.yml`, which is the local-dev stack with its own
   Postgres).
3. Assign a domain to each service:
   - `app` → e.g. `https://api.yourdomain` (container port 8000)
   - `dashboard` → e.g. `https://app.yourdomain` (container port 3000)
4. Mark the **`aitrader_models`** volume as persistent. This is what lets the
   daily retraining accumulate — the image only seeds it when empty, so
   redeploys never wipe the learned models.

## 3. Environment variables (set in Coolify)

| Variable | Value |
|---|---|
| `DATABASE_URL` | your Neon pooled string from step 1 |
| `AITRADER_API_KEY` | `openssl rand -hex 32` |
| `NEXT_PUBLIC_API_KEY` | **same value** as `AITRADER_API_KEY` (build arg — baked into the dashboard) |
| `NEXT_PUBLIC_API_URL` | the **public** URL of the `app` service (e.g. `https://api.yourdomain`) — a build arg, so a rebuild is needed if it changes |
| `ALLOWED_ORIGINS` | the dashboard URL (e.g. `https://app.yourdomain`) |
| `AUTO_RETRAIN` | `1` |
| `AUTO_RETRAIN_SCHEDULE` | `daily` |

Leave the `SHOONYA_*` variables unset for the paper period — paper mode never
calls the broker. Do **not** enable live mode; the app defaults to paper.

## 4. First-boot verification

Deploy, then check the `app` service logs for, in order:

```
Seeding models volume if empty...
  seeding hunter_apex_1500_brain.zip from image
  seeding guardian_apex_1500_brain.zip from image
  seeding executive_apex_manager.zip from image
Running database migrations...
All AI brains loaded and validated.
```

Open the dashboard domain — it should load and show **PAPER** mode. On later
redeploys the "seeding" lines are skipped (the volume already has models,
possibly ones the daily trainer improved), which is exactly the intended
behaviour.

## Notes

- **NEXT_PUBLIC_API_URL is baked at build time.** If the browser can't reach
  the API, it's almost always because this was left as `localhost` — set it to
  the API's public domain and redeploy.
- The daily trainer runs at `AUTO_RETRAIN_HOUR` IST (default 02:00). Early on
  it will rarely promote a new model (only genuine gate-beating improvements
  ship); that's expected while live trade/ghost history accumulates.
- When you're ready to go live: fill the `SHOONYA_*` variables, then flip to
  LIVE from the dashboard (it requires an explicit confirm).
