"""Tests for the trading-loop active-window gate (NSE hours ± buffer).

The loop must only do work — and only touch the market-data API — inside NSE
hours plus a small buffer, and sleep until the next session otherwise.
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ.setdefault("MODELS_DIR", "/nonexistent-for-tests")

import app.main as m

IST = ZoneInfo("Asia/Kolkata")


def test_inside_window_during_market_hours():
    assert m._within_active_window(datetime(2026, 7, 21, 12, 0, tzinfo=IST))  # Tue noon
    assert m._seconds_until_active_window(datetime(2026, 7, 21, 12, 0, tzinfo=IST)) == 0.0


def test_buffer_minutes_are_inside_the_window():
    # 15 min before open and after close (default buffer) are active.
    assert m._within_active_window(datetime(2026, 7, 21, 9, 5, tzinfo=IST))    # 09:05
    assert m._within_active_window(datetime(2026, 7, 21, 15, 40, tzinfo=IST))  # 15:40


def test_before_open_waits_until_today_open():
    # 08:00 Tue → window opens at 09:00 (09:15 − 15 min) → 1 hour.
    secs = m._seconds_until_active_window(datetime(2026, 7, 21, 8, 0, tzinfo=IST))
    assert abs(secs - 3600) < 1
    assert not m._within_active_window(datetime(2026, 7, 21, 8, 0, tzinfo=IST))


def test_after_close_waits_until_next_morning():
    # 16:00 Tue → next open 09:00 Wed → 17 hours.
    secs = m._seconds_until_active_window(datetime(2026, 7, 21, 16, 0, tzinfo=IST))
    assert abs(secs - 17 * 3600) < 1


def test_weekend_rolls_to_monday():
    # Sat 12:00 → Monday 09:00 → 45 hours.
    secs = m._seconds_until_active_window(datetime(2026, 7, 25, 12, 0, tzinfo=IST))
    assert abs(secs - 45 * 3600) < 1
    assert not m._within_active_window(datetime(2026, 7, 25, 12, 0, tzinfo=IST))
