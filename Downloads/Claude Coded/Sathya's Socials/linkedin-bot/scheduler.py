import json
import os
from datetime import datetime, timedelta, timezone

import requests
from zoneinfo import ZoneInfo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PENDING_PATH = os.path.join(BASE_DIR, "pending_posts.json")
GETLATE_BASE = "https://getlate.dev/api/v1"
IST = ZoneInfo("Asia/Kolkata")

# (weekday, hour) pairs in priority order. weekday: Mon=0 ... Sun=6
PRIORITY_SLOTS = [
    (1, 8),   # Tuesday 8 AM
    (2, 8),   # Wednesday 8 AM
    (3, 8),   # Thursday 8 AM
    (1, 17),  # Tuesday 5 PM
    (2, 17),  # Wednesday 5 PM
    (3, 17),  # Thursday 5 PM
    (0, 8),   # Monday 8 AM
    (4, 8),   # Friday 8 AM
]


def _load_pending(pending_path: str) -> list:
    if os.path.exists(pending_path):
        with open(pending_path) as f:
            return json.loads(f.read())
    return []


def _is_within_24h(slot: datetime, scheduled_posts: list) -> bool:
    """Return True if any existing post is within 24h of this slot."""
    for post in scheduled_posts:
        raw = post.get("scheduled_for", "")
        if not raw:
            continue
        try:
            existing = datetime.fromisoformat(raw)
            if existing.tzinfo is None:
                existing = existing.replace(tzinfo=IST)
            if abs((slot - existing).total_seconds()) < 86400:
                return True
        except ValueError:
            continue
    return False


def next_optimal_slot(now: datetime = None, pending_path: str = DEFAULT_PENDING_PATH) -> datetime:
    """Return the next available optimal IST posting slot."""
    if now is None:
        now = datetime.now(tz=IST)

    pending = _load_pending(pending_path)

    # Search up to 14 days forward
    for days_ahead in range(14):
        candidate_date = (now + timedelta(days=days_ahead)).date()

        for (weekday, hour) in PRIORITY_SLOTS:
            if candidate_date.weekday() != weekday:
                continue

            slot = datetime(
                candidate_date.year, candidate_date.month, candidate_date.day,
                hour, 0, 0, tzinfo=IST
            )

            # Must be at least 1 hour in the future
            if slot < now + timedelta(hours=1):
                continue

            # Must not conflict with existing scheduled posts
            if _is_within_24h(slot, pending):
                continue

            return slot

    # Absolute fallback: next Monday 8 AM
    days_to_monday = (7 - now.weekday()) % 7 or 7
    fallback = now + timedelta(days=days_to_monday)
    return datetime(fallback.year, fallback.month, fallback.day, 8, 0, 0, tzinfo=IST)


def schedule_post(
    content: str,
    account_id: str,
    late_api_key: str,
    slot: datetime,
    pending_path: str = DEFAULT_PENDING_PATH,
) -> bool:
    """POST to GetLate. Return True on success, False on failure."""
    scheduled_for = slot.strftime("%Y-%m-%dT%H:%M:%S")

    payload = {
        "content": content,
        "scheduledFor": scheduled_for,
        "timezone": "Asia/Kolkata",
        "platforms": [{"platform": "linkedin", "accountId": account_id}],
    }

    try:
        resp = requests.post(
            f"{GETLATE_BASE}/posts",
            headers={
                "Authorization": f"Bearer {late_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        return True

    except Exception as e:
        _save_pending(content, slot, pending_path, error=str(e))
        return False


def _save_pending(content: str, slot: datetime, pending_path: str, error: str = ""):
    """Append a failed post to pending_posts.json."""
    pending = _load_pending(pending_path)
    pending.append({
        "content": content,
        "scheduled_for": slot.isoformat(),
        "error": error,
    })
    with open(pending_path, "w") as f:
        json.dump(pending, f, indent=2)
