"""Correctness checks for the structure-aware stop/target (and therefore a
non-constant reward:risk). Run directly:

    python -m tests.test_technical_levels

Regression target: the old fixed price±(2,3)×ATR multiples made reward:risk a
constant 1.5 for every stock — no information. The new logic anchors to 20-day
support/resistance with ATR bounds, so R:R must vary with structure and stay
inside its documented bounds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics import technical


def _bars_with_range(n: int, price_now: float, low20: float, high20: float,
                     seed: int = 1) -> pd.DataFrame:
    """Synthetic OHLCV whose last-20-bar range is [low20, high20] and whose
    final close is price_now. Earlier bars stay inside that band too, so the
    20-day extremes are exactly the injected ones."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    mid = (low20 + high20) / 2.0
    close = mid + (high20 - low20) * 0.25 * np.sin(np.linspace(0, 6, n)) \
        + rng.normal(0, (high20 - low20) * 0.02, n)
    close = np.clip(close, low20 + 0.01, high20 - 0.01)
    close[-1] = price_now
    # inject the exact extremes into the final 20 bars' highs/lows
    high = close * 1.005
    low = close * 0.995
    high[-5] = high20
    low[-10] = low20
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = rng.uniform(1e5, 1e6, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


def _rr(ta) -> float | None:
    return ta.reward_risk


def test_stop_target_within_atr_bounds():
    bars = _bars_with_range(n=80, price_now=100.0, low20=95.0, high20=110.0)
    ta = technical.analyze("TEST", "Test", bars)
    assert ta.stop is not None and ta.target is not None
    atr = ta.indicators["atr"]
    price = ta.price
    assert price - 3.0 * atr - 0.01 <= ta.stop <= price - 1.0 * atr + 0.01, (ta.stop, price, atr)
    assert price + 1.5 * atr - 0.01 <= ta.target <= price + 4.5 * atr + 0.01, (ta.target, price, atr)
    assert ta.stop < price < ta.target


def test_reward_risk_varies_with_structure():
    """A name just above support (tight structural stop, far resistance) must
    show a better R:R than the same name pressed just under resistance (far
    stop, tiny room to the range top). Under the old constant-multiple logic
    both were exactly 1.5."""
    near_support = _bars_with_range(n=80, price_now=96.0, low20=95.0, high20=112.0, seed=2)
    near_resist = _bars_with_range(n=80, price_now=111.0, low20=95.0, high20=112.0, seed=3)
    rr_support = _rr(technical.analyze("TEST", "Test", near_support))
    rr_resist = _rr(technical.analyze("TEST", "Test", near_resist))
    assert rr_support is not None and rr_resist is not None
    assert rr_support != rr_resist, "R:R must not be a constant"
    assert rr_support > rr_resist, (rr_support, rr_resist)


def test_stop_always_below_price_even_at_new_lows():
    """Price breaking below the prior support must not produce a stop above
    price (the min() clamp guarantees >= 1 ATR below)."""
    bars = _bars_with_range(n=80, price_now=95.5, low20=95.0, high20=112.0, seed=4)
    bars.iloc[-1, bars.columns.get_loc("close")] = 94.0   # close below prior support
    bars.iloc[-1, bars.columns.get_loc("low")] = 93.8
    ta = technical.analyze("TEST", "Test", bars)
    assert ta.stop is not None and ta.stop < ta.price


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} technical-levels tests passed.")


if __name__ == "__main__":
    main()
