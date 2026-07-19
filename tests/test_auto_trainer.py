"""Tests for the weekend self-learning scheduler."""

from datetime import datetime
from zoneinfo import ZoneInfo

from app.auto_trainer import seconds_until_next_saturday

IST = ZoneInfo("Asia/Kolkata")


def test_midweek_waits_until_saturday_2am():
    wed = datetime(2026, 7, 15, 12, 0, tzinfo=IST)  # Wednesday noon
    secs = seconds_until_next_saturday(wed)
    target = wed.replace(day=18, hour=2, minute=0)
    assert abs(secs - (target - wed).total_seconds()) < 1


def test_saturday_after_hour_waits_a_week():
    sat = datetime(2026, 7, 18, 3, 0, tzinfo=IST)  # Saturday 03:00, past 02:00
    assert seconds_until_next_saturday(sat) > 6 * 24 * 3600
