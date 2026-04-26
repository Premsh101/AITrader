# ── Stage 1: Frontend builder ────────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /dashboard

# Copy package manifest and lockfile to ensure reproducible installs
COPY dashboard/package.json dashboard/package-lock.json* ./
RUN npm install --no-audit

# Copy the rest of the dashboard source and build the Next.js app
COPY dashboard/ .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ── Stage 2: Frontend production runner ──────────────────────────────────────
FROM node:20-alpine AS frontend

WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

COPY --from=frontend-builder /dashboard/public ./public
COPY --from=frontend-builder --chown=nextjs:nodejs /dashboard/.next/standalone ./
COPY --from=frontend-builder --chown=nextjs:nodejs /dashboard/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT=3000
CMD ["node", "server.js"]

# ── Stage 3: Backend ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS backend

# Install system dependencies required by psycopg2-binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create required directories
RUN mkdir -p /app/models /app/logs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy AI model zip files into the image so they are available at runtime.
# Expected files:
#   ./models/hunter_apex_1500_brain.zip
#   ./models/guardian_apex_1500_brain.zip
#   ./models/executive_apex_manager.zip
COPY ./models /app/models

# Copy application source and supporting files
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
