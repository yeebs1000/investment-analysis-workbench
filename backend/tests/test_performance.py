"""Correctness checks for the performance-vs-SPY tracker. Run directly:

    python -m tests.test_performance

Uses a temp store path via monkeypatching _store_path so the real
data_store/performance.parquet is never touched.
"""
from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app.analytics import performance as perf


def _use_temp_store():
    tmp = Path(tempfile.mkdtemp()) / "perf.parquet"
    perf._store_path = lambda: tmp   # type: ignore[assignment]
    return tmp


def test_no_data_status():
    _use_temp_store()
    r = perf.compute_performance()
    assert r["status"] == "no_data"
    assert r["days_tracked"] == 0


def test_building_status_and_cumulative_returns():
    _use_temp_store()
    start = date(2026, 1, 1)
    # 10 days: account +10%, SPY +5% over the window
    for i in range(10):
        eq = 100_000 * (1 + 0.10 * i / 9)
        spy = 500.0 * (1 + 0.05 * i / 9)
        perf.record_snapshot(eq, spy, on=start + timedelta(days=i))
    r = perf.compute_performance()
    assert r["status"] == "building"      # < MIN_DAYS_FOR_STATS
    assert r["days_tracked"] == 10
    assert abs(r["account_return_pct"] - 10.0) < 0.01
    assert abs(r["spy_return_pct"] - 5.0) < 0.01
    assert abs(r["excess_return_pct"] - 5.0) < 0.01
    assert r["beating_spy"] is True
    assert "beta" not in r               # ratios withheld until enough data


def test_ready_status_computes_ratios():
    _use_temp_store()
    start = date(2026, 1, 1)
    # 40 business-ish days; account outperforms SPY steadily
    for i in range(40):
        eq = 100_000 * (1 + 0.20 * i / 39)
        spy = 500.0 * (1 + 0.10 * i / 39)
        perf.record_snapshot(eq, spy, on=start + timedelta(days=i))
    r = perf.compute_performance()
    assert r["status"] == "ready"
    assert r["days_tracked"] == 40
    assert r["beating_spy"] is True
    assert r["excess_return_pct"] > 0
    # ratios present
    for k in ("beta", "alpha_pct", "tracking_error_pct", "information_ratio",
              "max_drawdown_pct", "spy_max_drawdown_pct"):
        assert k in r, k


def test_daily_dedup_last_write_wins():
    _use_temp_store()
    d = date(2026, 3, 1)
    perf.record_snapshot(100_000, 500.0, on=d)
    perf.record_snapshot(111_111, 510.0, on=d)   # same day -> overwrite
    df = perf._load()
    assert len(df) == 1
    assert abs(float(df.iloc[0]["equity_usd"]) - 111_111) < 1


def test_bad_input_is_noop():
    tmp = _use_temp_store()
    perf.record_snapshot(None, 500.0)
    perf.record_snapshot(100_000, 0.0)
    perf.record_snapshot(-5, -5)
    assert not tmp.exists(), "bad snapshots must not create a store file"


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} performance tests passed.")


if __name__ == "__main__":
    main()
