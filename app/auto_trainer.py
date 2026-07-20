"""
auto_trainer.py – Autonomous self-learning, with the walk-forward evaluation
gate as the promotion criterion.

Runs on a configurable cadence (``AUTO_RETRAIN_SCHEDULE`` = ``daily`` or
``weekly``) at AUTO_RETRAIN_HOUR IST (default 02:00), when AUTO_RETRAIN=1:

  1. Export the system's decision history (trades + ghosts) to CSV.
  2. Copy the current models to a STAGING dir and continue training them
     there on freshly self-fetched market data (--finetune --experience).
  3. Run the walk-forward evaluation gate against the staged models.
  4. PASS → staged zips replace the live ones and brains hot-reload.
     FAIL → live models stay untouched; an ERROR lands in the activity log.

The live models can therefore only ever improve — a bad refresh never ships.
DAILY is useful during the paper-trading period: each run replays the full
and growing decision history (every executed trade AND every ghost trade the
Hunter/Executive declined), so the system keeps learning from "what it
missed" without needing to over-trade. Because promotion still requires
beating buy-and-hold ^NSEI after costs, a daily run that doesn't improve
simply keeps the incumbent — daily cadence adds attempts, never risk.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "1") == "1"
AUTO_RETRAIN_TIMESTEPS = int(os.environ.get("AUTO_RETRAIN_TIMESTEPS", "250000"))
AUTO_RETRAIN_HOUR = int(os.environ.get("AUTO_RETRAIN_HOUR", "2"))  # IST
# Cadence: "daily" retrains every day at AUTO_RETRAIN_HOUR (useful during the
# paper-trading period); "weekly" retrains every Saturday. Default weekly.
AUTO_RETRAIN_SCHEDULE = os.environ.get("AUTO_RETRAIN_SCHEDULE", "weekly").lower()

MODEL_FILES = [
    "hunter_apex_1500_brain.zip",
    "guardian_apex_1500_brain.zip",
    "executive_apex_manager.zip",
]


def seconds_until_next_saturday(now_ist: datetime) -> float:
    """Seconds from *now_ist* to the next Saturday AUTO_RETRAIN_HOUR IST."""
    days_ahead = (5 - now_ist.weekday()) % 7  # Saturday == 5
    target = (now_ist + timedelta(days=days_ahead)).replace(
        hour=AUTO_RETRAIN_HOUR, minute=0, second=0, microsecond=0
    )
    if target <= now_ist:
        target += timedelta(days=7)
    return (target - now_ist).total_seconds()


def seconds_until_next_daily(now_ist: datetime) -> float:
    """Seconds from *now_ist* to the next AUTO_RETRAIN_HOUR IST (today or tomorrow)."""
    target = now_ist.replace(
        hour=AUTO_RETRAIN_HOUR, minute=0, second=0, microsecond=0
    )
    if target <= now_ist:
        target += timedelta(days=1)
    return (target - now_ist).total_seconds()


def seconds_until_next_run(now_ist: datetime) -> float:
    """Seconds to the next scheduled retrain, per AUTO_RETRAIN_SCHEDULE."""
    if AUTO_RETRAIN_SCHEDULE == "daily":
        return seconds_until_next_daily(now_ist)
    return seconds_until_next_saturday(now_ist)


def run_retrain_cycle(models_dir: str, repo_root: str, log_activity) -> bool:
    """One full self-learning cycle. Returns True if new models shipped."""
    staging = os.path.join(models_dir, "staging")
    os.makedirs(staging, exist_ok=True)
    experience_csv = os.path.join(staging, "experience.csv")

    # 1. Decision history (trades + ghosts) → CSV.
    from app.database import SessionLocal
    from app.models import GhostTrade, Trade

    db = SessionLocal()
    try:
        with open(experience_csv, "w") as fh:
            fh.write("symbol,date,kind\n")
            for t in db.query(Trade).all():
                if t.created_at:
                    fh.write(f"{t.symbol},{t.created_at.date()},trade\n")
            for g in db.query(GhostTrade).all():
                if g.created_at:
                    fh.write(f"{g.symbol},{g.created_at.date()},ghost\n")
    finally:
        db.close()

    # 2. Stage current models and fine-tune them on fresh data (the training
    #    script fetches everything it needs from Yahoo itself).
    have_models = all(os.path.exists(os.path.join(models_dir, f)) for f in MODEL_FILES)
    for f in MODEL_FILES:
        src = os.path.join(models_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(staging, f))

    train_cmd = [
        sys.executable, os.path.join(repo_root, "training", "train_triad.py"),
        "--timesteps", str(AUTO_RETRAIN_TIMESTEPS),
        "--out-dir", staging,
        "--checkpoint-dir", os.path.join(staging, "checkpoints"),
        "--experience", experience_csv,
    ]
    if have_models:
        train_cmd.append("--finetune")
    log_activity("Trainer",
                 f"Self-learning started ({AUTO_RETRAIN_SCHEDULE}, finetune={have_models})")
    result = subprocess.run(train_cmd, cwd=repo_root, capture_output=True, text=True,
                            timeout=6 * 3600)
    if result.returncode != 0:
        log_activity("Trainer", f"Training failed: {result.stderr[-300:]}",
                     level=logging.ERROR)
        return False

    # 3. Evaluation gate on the staged models.
    eval_cmd = [
        sys.executable, os.path.join(repo_root, "training", "evaluate_triad.py"),
        "--models-dir", staging,
    ]
    result = subprocess.run(eval_cmd, cwd=repo_root, capture_output=True, text=True,
                            timeout=3600)
    verdict_pass = "VERDICT: PASS" in result.stdout

    # 4. Promote or reject.
    if verdict_pass:
        for f in MODEL_FILES:
            shutil.copy2(os.path.join(staging, f), os.path.join(models_dir, f))
        log_activity("Trainer", "Self-learning refresh PASSED the gate – new models live")
        return True

    log_activity(
        "Trainer",
        "Self-learning refresh FAILED the evaluation gate – keeping current "
        "models. See container logs for the metrics.",
        level=logging.ERROR,
    )
    logger.error("Evaluation output:\n%s", result.stdout[-2000:])
    return False


async def self_learning_loop(models_dir: str, repo_root: str, ist_tz, log_activity, brains) -> None:
    """Background task: sleep to the next scheduled run, retrain, repeat."""
    if not AUTO_RETRAIN:
        logger.info("AUTO_RETRAIN=0 – self-learning disabled")
        return
    logger.info("Self-learning enabled (schedule=%s, hour=%02d IST)",
                AUTO_RETRAIN_SCHEDULE, AUTO_RETRAIN_HOUR)
    while True:
        wait = seconds_until_next_run(datetime.now(ist_tz))
        logger.info("Next %s self-learning in %.1f hours",
                    AUTO_RETRAIN_SCHEDULE, wait / 3600)
        await asyncio.sleep(wait)
        try:
            shipped = await asyncio.to_thread(
                run_retrain_cycle, models_dir, repo_root, log_activity
            )
            if shipped:
                brains.load_all()  # hot-reload the promoted models
        except Exception:
            logger.exception("Self-learning cycle crashed")
        await asyncio.sleep(3600)  # avoid double-fire within the hour


# Backward-compatible aliases (older imports / call sites).
run_weekend_retrain = run_retrain_cycle
weekend_loop = self_learning_loop
