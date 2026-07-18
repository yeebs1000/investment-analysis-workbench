"""US trading-session date, shared by every nightly stage.

The machine runs in Singapore (~12h ahead of New York), so one LOCAL calendar
date spans two US sessions and `dt.date.today()` is wrong after local midnight:
a stage that starts at 23:40 SGT and retries at 00:10 SGT would look for the
NEXT day's chains/signals and find nothing (this bit us on 2026-07-16).
Key everything off the date in America/New_York instead — during the entire
US session (09:30-16:00 ET = 21:30-04:00 SGT) it names one consistent day.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def session_date(now: dt.datetime | None = None) -> str:
    """ISO date of the current US trading session (date in New York)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.astimezone()
    return now.astimezone(ET).date().isoformat()


def is_us_weekend(now: dt.datetime | None = None) -> bool:
    """True when it's Saturday/Sunday in New York (no session to trade)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.astimezone()
    return now.astimezone(ET).weekday() >= 5
