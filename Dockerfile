# ── Stage 1: Frontend builder ────────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /dashboard

# Copy package manifest and lockfile to ensure reproducible installs
COPY dashboard/package.json dashboard/package-lock.json* ./
RUN npm install --no-audit

# Copy the rest of the dashboard source and build the Next.js app.
# NEXT_PUBLIC_* values are embedded into the client bundle at build time,
# so they must be provided as build args (see docker-compose.yml).
COPY dashboard/ .
ARG NEXT_PUBLIC_API_URL=http://localhost:8005
ARG NEXT_PUBLIC_API_KEY=
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_KEY=$NEXT_PUBLIC_API_KEY
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
# Install the CPU-only PyTorch build FIRST, from PyTorch's CPU index, so the
# multi-GB CUDA wheels are never pulled. Both inference and the self-learning
# retrains run on CPU (the app forces device="cpu"), so the GPU libraries are
# dead weight — and shipping them bloats the image by ~5 GB, which overflows
# disk on modest deploy hosts during image unpack. requirements.txt then
# installs everything else from PyPI; torch is already satisfied so SB3 does
# not re-pull it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the shipped model zips into a SEED directory (not /app/models itself).
# At runtime /app/models is a persistent volume; entrypoint.sh copies these
# seeds in ONLY when the volume has no model yet, so a redeploy never
# overwrites the improvements the weekend auto-trainer writes to the volume.
# Expected files:
#   ./models/hunter_apex_1500_brain.zip
#   ./models/guardian_apex_1500_brain.zip
#   ./models/executive_apex_manager.zip
COPY ./models /app/model_seeds

# Copy application source and supporting files
COPY app ./app
COPY training ./training
COPY scripts ./scripts
COPY alembic ./alembic
COPY alembic.ini .
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
