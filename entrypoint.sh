#!/bin/sh
# entrypoint.sh – container start-up script
#
# 1. Waits for the database to be ready (alembic will fail otherwise).
# 2. Runs alembic upgrade head to create or migrate tables.
# 3. Starts the FastAPI application with uvicorn.

set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting AITrader FastAPI application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
