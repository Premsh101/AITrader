"""Tests for the self-learning scheduler (weekly + daily cadences)."""

import importlib
from datetime import datetime
from zoneinfo import ZoneInfo

from app.auto_trainer import seconds_until_next_saturday, seconds_until_next_daily

IST = ZoneInfo("Asia/Kolkata")


def test_midweek_waits_until_saturday_2am():
    wed = datetime(2026, 7, 15, 12, 0, tzinfo=IST)  # Wednesday noon
    secs = seconds_until_next_saturday(wed)
    target = wed.replace(day=18, hour=2, minute=0)
    assert abs(secs - (target - wed).total_seconds()) < 1


def test_saturday_after_hour_waits_a_week():
    sat = datetime(2026, 7, 18, 3, 0, tzinfo=IST)  # Saturday 03:00, past 02:00
    assert seconds_until_next_saturday(sat) > 6 * 24 * 3600


def test_daily_before_hour_waits_until_today():
    # 00:30, before the 02:00 anchor → ~1.5h wait, same day.
    t = datetime(2026, 7, 15, 0, 30, tzinfo=IST)
    secs = seconds_until_next_daily(t)
    assert abs(secs - 1.5 * 3600) < 1


def test_daily_after_hour_waits_until_tomorrow():
    # 03:00, past the 02:00 anchor → next run is tomorrow, < 24h away.
    t = datetime(2026, 7, 15, 3, 0, tzinfo=IST)
    secs = seconds_until_next_daily(t)
    assert 22 * 3600 < secs <= 24 * 3600


def test_schedule_dispatch_honours_env(monkeypatch):
    monkeypatch.setenv("AUTO_RETRAIN_SCHEDULE", "daily")
    import app.auto_trainer as at
    importlib.reload(at)
    try:
        wed = datetime(2026, 7, 15, 12, 0, tzinfo=IST)
        # Daily dispatch must be far sooner than the weekly (Saturday) path.
        assert at.seconds_until_next_run(wed) < at.seconds_until_next_saturday(wed)
    finally:
        monkeypatch.delenv("AUTO_RETRAIN_SCHEDULE", raising=False)
        importlib.reload(at)
