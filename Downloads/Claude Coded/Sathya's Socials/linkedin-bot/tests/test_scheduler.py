import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def make_ist(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=IST)


def test_next_slot_from_monday_morning():
    """From Monday morning, next slot is Tuesday 8 AM."""
    # Monday 2026-03-16 09:00 IST
    now = make_ist(2026, 3, 16, 9, 0)
    from scheduler import next_optimal_slot
    slot = next_optimal_slot(now=now)
    assert slot.weekday() == 1  # Tuesday
    assert slot.hour == 8
    assert slot.minute == 0


def test_next_slot_from_tuesday_before_8am():
    """From Tuesday 7 AM, next slot is Tuesday 8 AM same day."""
    now = make_ist(2026, 3, 17, 7, 0)
    from scheduler import next_optimal_slot
    slot = next_optimal_slot(now=now)
    assert slot.weekday() == 1  # Tuesday
    assert slot.hour == 8
    assert slot.date() == now.date()


def test_next_slot_from_tuesday_after_8am_before_5pm():
    """From Tuesday 10 AM, next slot is Tuesday 5 PM."""
    now = make_ist(2026, 3, 17, 10, 0)
    from scheduler import next_optimal_slot
    slot = next_optimal_slot(now=now)
    assert slot.weekday() == 1  # Tuesday
    assert slot.hour == 17


def test_next_slot_from_tuesday_evening():
    """From Tuesday 6 PM, next slot is Wednesday 8 AM."""
    now = make_ist(2026, 3, 17, 18, 30)
    from scheduler import next_optimal_slot
    slot = next_optimal_slot(now=now)
    assert slot.weekday() == 2  # Wednesday
    assert slot.hour == 8


def test_next_slot_from_thursday_evening():
    """From Thursday 7 PM, next slot is Friday 8 AM (Mon/Fri fallback tier)."""
    now = make_ist(2026, 3, 19, 19, 0)
    from scheduler import next_optimal_slot
    slot = next_optimal_slot(now=now)
    assert slot.weekday() == 4  # Friday
    assert slot.hour == 8


def test_next_slot_skips_within_24h(tmp_path):
    """Skips a slot if a post is already scheduled within 24h of it."""
    pending_file = tmp_path / "pending_posts.json"
    # Already scheduled Tuesday 8 AM
    pending_file.write_text(json.dumps([
        {"scheduled_for": "2026-03-17T08:00:00+05:30"}
    ]))

    now = make_ist(2026, 3, 16, 9, 0)  # Monday 9 AM
    from scheduler import next_optimal_slot
    slot = next_optimal_slot(now=now, pending_path=str(pending_file))

    # Should skip Tuesday 8 AM and give Tuesday 5 PM or later
    assert not (slot.weekday() == 1 and slot.hour == 8)


def test_schedule_post_success(tmp_path):
    """Returns True and does not write to pending on success."""
    pending_file = tmp_path / "pending_posts.json"

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"id": "post123"}

    from datetime import timezone, timedelta
    slot = datetime(2026, 3, 17, 8, 0, tzinfo=IST)

    with patch("requests.post", return_value=mock_response):
        from scheduler import schedule_post
        success = schedule_post(
            content="Test post",
            account_id="acc123",
            late_api_key="fake_key",
            slot=slot,
            pending_path=str(pending_file)
        )

    assert success is True
    assert not pending_file.exists()


def test_schedule_post_failure_saves_pending(tmp_path):
    """Returns False and saves to pending_posts.json on API failure."""
    pending_file = tmp_path / "pending_posts.json"

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = Exception("Server error")

    slot = datetime(2026, 3, 17, 8, 0, tzinfo=IST)

    with patch("requests.post", side_effect=Exception("Connection error")):
        from scheduler import schedule_post
        success = schedule_post(
            content="Test post",
            account_id="acc123",
            late_api_key="fake_key",
            slot=slot,
            pending_path=str(pending_file)
        )

    assert success is False
    saved = json.loads(pending_file.read_text())
    assert len(saved) == 1
    assert saved[0]["content"] == "Test post"
