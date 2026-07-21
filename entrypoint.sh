#!/bin/sh
# entrypoint.sh – container start-up script
#
# 1. Waits for the database to be ready (alembic will fail otherwise).
# 2. Runs alembic upgrade head to create or migrate tables.
# 3. Starts the FastAPI application with uvicorn.

set -e

# Seed the models volume from the image ONLY when a model is missing, so a
# redeploy (which rebuilds the image) never clobbers the models the weekend
# auto-trainer has since improved in place on the persistent volume.
echo "Seeding models volume if empty..."
mkdir -p /app/models
for f in hunter_apex_1500_brain.zip guardian_apex_1500_brain.zip executive_apex_manager.zip; do
    if [ ! -f "/app/models/$f" ] && [ -f "/app/model_seeds/$f" ]; then
        echo "  seeding $f from image"
        cp "/app/model_seeds/$f" "/app/models/$f"
    fi
done

# Fail fast with a clear, actionable message if the DB connection string is
# missing — otherwise alembic dies with an opaque SQLAlchemy URL parse error
# and the container crash-loops.
if [ -z "$DATABASE_URL" ]; then
    echo "FATAL: DATABASE_URL is not set." >&2
    echo "       Set it in Coolify (app service env) to your Neon connection" >&2
    echo "       string, e.g.:" >&2
    echo "       postgresql://user:pass@ep-xxx-pooler.<region>.aws.neon.tech/aitrader?sslmode=require" >&2
    exit 1
fi

echo "Running database migrations..."
alembic upgrade head

echo "Starting AITrader FastAPI application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
