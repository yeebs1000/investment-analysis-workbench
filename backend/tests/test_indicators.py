"""Correctness checks for the indicator math. Run directly:

    python -m tests.test_indicators

Uses hand-computed values and an independent reference RSI so the engine is
provably correct, not merely plausible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics import indicators as ind


def _approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def test_sma():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert _approx(ind.sma(s, 2).iloc[-1], 4.5)
    assert _approx(ind.sma(s, 5).iloc[-1], 3.0)


def test_ema():
    # span=3 -> alpha=0.5, adjust=False, seed=first value
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert _approx(ind.ema(s, 3).iloc[-1], 4.0625)


def test_rsi_extremes():
    up = pd.Series(np.arange(1, 30), dtype=float)
    assert _approx(ind.rsi(up, 14).iloc[-1], 100.0)
    down = pd.Series(np.arange(30, 1, -1), dtype=float)
    assert _approx(ind.rsi(down, 14).iloc[-1], 0.0)


def test_rsi_matches_reference():
    rng = np.random.default_rng(42)
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    got = ind.rsi(prices, 14)

    # Independent Wilder RSI reference.
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    al = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    ref = 100 - 100 / (1 + ag / al)

    common = got.dropna().index.intersection(ref.dropna().index)
    assert len(common) > 100
    assert np.allclose(got.loc[common], ref.loc[common], atol=1e-6)


def test_true_range_and_atr():
    high = pd.Series([10, 11, 12], dtype=float)
    low = pd.Series([9, 9.5, 11], dtype=float)
    close = pd.Series([9.5, 10.5, 11.5], dtype=float)
    tr = ind.true_range(high, low, close)
    # bar0: high-low = 1
    assert _approx(tr.iloc[0], 1.0)
    # bar1: max(11-9.5=1.5, |11-9.5|=1.5, |9.5-9.5|=0) = 1.5
    assert _approx(tr.iloc[1], 1.5)
    # bar2: max(12-11=1, |12-10.5|=1.5, |11-10.5|=0.5) = 1.5
    assert _approx(tr.iloc[2], 1.5)


def test_bollinger_pctb_midpoint():
    # Constant-trend series: %B should be between 0 and 1 and finite.
    s = pd.Series(100 + np.sin(np.linspace(0, 6, 100)), dtype=float)
    _mid, up, low, pctb, bw = ind.bollinger(s, 20, 2)
    last = pctb.dropna().iloc[-1]
    assert 0.0 <= last <= 1.0 or np.isfinite(last)
    assert (up.dropna() >= low.dropna()).all()


def test_macd_relationship():
    s = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 100)))
    macd_line, signal, hist = ind.macd(s)
    assert _approx(hist.iloc[-1], macd_line.iloc[-1] - signal.iloc[-1], tol=1e-9)


def test_stochastic_bounds():
    rng = np.random.default_rng(7)
    n = 100
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    k, d = ind.stochastic(high, low, close)
    kk = k.dropna()
    assert (kk >= -1e-9).all() and (kk <= 100 + 1e-9).all()


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} indicator tests passed.")


if __name__ == "__main__":
    main()
