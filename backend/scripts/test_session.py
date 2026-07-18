"""Self-check for the US session-date helper (scripts/_session.py).

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/test_session.py
"""
import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._session import is_us_weekend, session_date  # noqa: E402

SGT = ZoneInfo("Asia/Singapore")


def demo() -> None:
    # 23:40 SGT Thu = 11:40 ET Thu -> session date is Thursday
    t = dt.datetime(2026, 7, 16, 23, 40, tzinfo=SGT)
    assert session_date(t) == "2026-07-16", session_date(t)

    # THE bug this fixes: 00:10 SGT Fri = 12:10 ET Thu -> STILL Thursday's session
    t = dt.datetime(2026, 7, 17, 0, 10, tzinfo=SGT)
    assert session_date(t) == "2026-07-16", session_date(t)

    # 03:59 SGT Fri = 15:59 ET Thu (one minute before US close) -> Thursday
    t = dt.datetime(2026, 7, 17, 3, 59, tzinfo=SGT)
    assert session_date(t) == "2026-07-16", session_date(t)

    # 22:30 SGT Fri = 10:30 ET Fri -> Friday
    t = dt.datetime(2026, 7, 17, 22, 30, tzinfo=SGT)
    assert session_date(t) == "2026-07-17", session_date(t)

    # weekend guard: Sat 22:30 SGT = Sat 10:30 ET -> weekend
    assert is_us_weekend(dt.datetime(2026, 7, 18, 22, 30, tzinfo=SGT))
    # Mon 01:00 SGT = Sun 13:00 ET -> still US weekend
    assert is_us_weekend(dt.datetime(2026, 7, 20, 1, 0, tzinfo=SGT))
    # Mon 22:30 SGT = Mon 10:30 ET -> trading day
    assert not is_us_weekend(dt.datetime(2026, 7, 20, 22, 30, tzinfo=SGT))

    # naive datetimes are accepted (interpreted as local time)
    assert isinstance(session_date(dt.datetime(2026, 7, 16, 23, 40)), str)

    # holidays: Labor Day Monday is NOT a trading session; the Tuesday after is
    from scripts._session import holiday_horizon_days, is_trading_session
    assert not is_trading_session(dt.datetime(2026, 9, 7, 22, 30, tzinfo=SGT))
    assert is_trading_session(dt.datetime(2026, 9, 8, 22, 30, tzinfo=SGT))
    # weekend still blocks regardless of holiday list
    assert not is_trading_session(dt.datetime(2026, 7, 18, 22, 30, tzinfo=SGT))
    assert holiday_horizon_days(dt.datetime(2026, 7, 18, tzinfo=SGT)) > 300

    print("session_date: all checks passed")


if __name__ == "__main__":
    demo()
