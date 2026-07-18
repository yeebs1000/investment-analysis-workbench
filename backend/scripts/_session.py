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


# NYSE full-closure holidays. Static on purpose (no dependency); extend each
# January -- the nightly runner logs a reminder when the horizon runs short.
NYSE_HOLIDAYS = {
    "2026-09-07",   # Labor Day
    "2026-11-26",   # Thanksgiving
    "2026-12-25",   # Christmas
    "2027-01-01",   # New Year's Day
    "2027-01-18",   # MLK Day
    "2027-02-15",   # Presidents' Day
    "2027-03-26",   # Good Friday
    "2027-05-31",   # Memorial Day
    "2027-06-18",   # Juneteenth (observed)
    "2027-07-05",   # Independence Day (observed)
}


def is_trading_session(now: dt.datetime | None = None) -> bool:
    """True when New York has a session today (not weekend, not a holiday).
    A holiday Monday used to run the whole pipeline into a closed market --
    the 07-11 Saturday-batch class: five structures guaranteed to expire."""
    return not is_us_weekend(now) and session_date(now) not in NYSE_HOLIDAYS


def holiday_horizon_days(now: dt.datetime | None = None) -> int:
    """Days until the static holiday list runs out (runner logs when < 60)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    last = max(dt.date.fromisoformat(d) for d in NYSE_HOLIDAYS)
    return (last - now.astimezone(ET).date()).days
