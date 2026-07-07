"""Correctness checks for the entry-risk flag (chase/capitulation detector)
and the two-axis quality×timing verdict. Run directly:

    python -m tests.test_entry_risk

Design being proven: the flag keys on VELOCITY (ATRs traveled in 10 bars) and
STRETCH (ATRs from the 20-EMA) — never on proximity to highs — so a steady
uptrend never fires, a parabolic ramp always does, a capitulation flush fires
symmetrically, and it is all ATR-relative so a volatile name isn't punished
for its normal swing size. The verdict must keep "is it good" and "is this a
good moment" as independent axes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics import technical
from app.analytics.scoring import two_axis_verdict


def _bars(n: int = 80, drift: float = 0.0, burst_days: int = 0,
          burst_ret: float = 0.0, rng_pct: float = 0.016,
          gap_on_first_burst: bool = False, seed: int = 1) -> pd.DataFrame:
    """Synthetic OHLCV: `n` bars around 100 with per-bar `drift`, the last
    `burst_days` replaced by `burst_ret` daily moves. Opens sit at the prior
    close (no gaps) unless `gap_on_first_burst` injects a 4% open gap on the
    first burst bar."""
    rng = np.random.default_rng(seed)
    rets = np.full(n, drift) + rng.normal(0, 0.001, n)
    if burst_days:
        rets[-burst_days:] = burst_ret
    close = 100.0 * np.cumprod(1 + rets)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    if gap_on_first_burst and burst_days:
        open_[n - burst_days] = close[n - burst_days - 1] * (1.04 if burst_ret > 0 else 0.96)
    high = np.maximum(open_, close) * (1 + rng_pct / 2)
    low = np.minimum(open_, close) * (1 - rng_pct / 2)
    vol = rng.uniform(1e5, 1e6, n)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


def test_parabolic_ramp_flags_chase_risk():
    ta = technical.analyze("TEST", "Test", _bars(burst_days=10, burst_ret=0.025))
    er = ta.entry_risk
    assert er is not None, "a +25%-in-10-bars ramp must flag"
    assert er["direction"] == "up" and er["level"] == "high", er
    assert er["move_atr_10"] >= technical.ENTRY_RISK_MOVE_ATR_HIGH, er
    assert "chase" in er["label"].lower()


def test_capitulation_flush_flags_symmetrically():
    ta = technical.analyze("TEST", "Test", _bars(burst_days=10, burst_ret=-0.025))
    er = ta.entry_risk
    assert er is not None, "a -22%-in-10-bars flush must flag"
    assert er["direction"] == "down" and er["level"] == "high", er
    assert "panic-sell" in er["label"].lower()


def test_steady_uptrend_never_fires():
    """A healthy grind — the case a naive 'near highs = don't chase' rule would
    wrongly punish (and which our momentum components correctly reward)."""
    ta = technical.analyze("TEST", "Test", _bars(drift=0.0025))
    assert ta.entry_risk is None, ta.entry_risk
    assert ta.stop is not None and ta.target is not None  # rest of the read intact


def test_atr_relative_no_flag_on_volatile_name():
    """Same +25% move but on a name whose normal daily range is ~5%: measured
    in ITS OWN ATRs the move isn't parabolic, so no flag (a fixed-% rule would
    misfire here — the GRAB case)."""
    ta = technical.analyze("TEST", "Test",
                           _bars(burst_days=10, burst_ret=0.025, rng_pct=0.055, seed=7))
    er = ta.entry_risk
    assert er is None or er["level"] == "caution", er


def test_gap_led_move_marks_event_gap():
    with_gap = technical.analyze(
        "TEST", "Test", _bars(burst_days=10, burst_ret=0.025, gap_on_first_burst=True))
    without = technical.analyze(
        "TEST", "Test", _bars(burst_days=10, burst_ret=0.025))
    assert with_gap.entry_risk is not None and with_gap.entry_risk["event_gap"] is True
    assert without.entry_risk is not None and without.entry_risk["event_gap"] is False


Q_GOOD = {"score_0_100": 80.0, "label": "High quality"}
Q_WEAK = {"score_0_100": 30.0, "label": "Weak"}
ER_UP = {"direction": "up", "level": "high"}
ER_DOWN = {"direction": "down", "level": "high"}


def test_verdict_requires_quality_axis():
    assert two_axis_verdict(None, 70.0, None) is None


def test_verdict_axes_stay_independent():
    """The cyberagent discipline: 'is it good' and 'is this the moment' are
    never conflated — a strong score with an active chase flag must read
    'extended', not 'favorable'."""
    v_fav = two_axis_verdict(Q_GOOD, 70.0, None)
    v_ext = two_axis_verdict(Q_GOOD, 70.0, ER_UP)
    assert v_fav["timing_axis"] == "favorable"
    assert v_ext["timing_axis"] == "extended"
    assert "chasing" in v_ext["guidance"]


def test_verdict_weak_quality_strong_tape_is_a_trade():
    v = two_axis_verdict(Q_WEAK, 65.0, None, stop=42.5)
    assert v["timing_axis"] == "favorable"
    assert "trade" in v["guidance"] and "42.5" in v["guidance"]


def test_verdict_good_quality_flush_warns_against_selling():
    v = two_axis_verdict(Q_GOOD, 40.0, ER_DOWN)
    assert v["timing_axis"] == "flush"
    assert "worst moment to sell" in v["guidance"]


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} entry-risk tests passed.")


if __name__ == "__main__":
    main()
