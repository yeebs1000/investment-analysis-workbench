"""Technical indicators implemented from first principles (pandas/numpy only).

Kept dependency-free and explicit so every value is auditable and provably
correct (see tests/test_indicators.py). All functions take/return pandas Series
aligned to the input index; trailing NaNs appear until enough lookback exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _wilder(series: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (a.k.a. RMA): EMA with alpha = 1/n."""
    return series.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder(gain, n)
    avg_loss = _wilder(loss, n)
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # When there have been no losses, RSI is defined as 100.
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(
    close: pd.Series, n: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = sma(close, n)
    sd = close.rolling(n).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    width = upper - lower
    pct_b = (close - lower) / width.replace(0, np.nan)
    bandwidth = width / mid.replace(0, np.nan)
    return mid, upper, lower, pct_b, bandwidth


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return _wilder(true_range(high, low, close), n)


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr_ = _wilder(true_range(high, low, close), n)
    plus_di = 100.0 * _wilder(plus_dm, n) / atr_
    minus_di = 100.0 * _wilder(minus_dm, n) / atr_
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = _wilder(dx, n)
    return adx_, plus_di, minus_di


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    pct_k = 100.0 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    pct_d = pct_k.rolling(d).mean()
    return pct_k, pct_d


def rolling_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 20
) -> pd.Series:
    typical = (high + low + close) / 3.0
    pv = typical * volume
    return pv.rolling(n).sum() / volume.rolling(n).sum().replace(0, np.nan)


def slope_pct(series: pd.Series, n: int = 5) -> float:
    """Percent change of a series over the last n steps (trend of an indicator)."""
    s = series.dropna()
    if len(s) <= n or s.iloc[-1 - n] == 0:
        return 0.0
    return float((s.iloc[-1] - s.iloc[-1 - n]) / abs(s.iloc[-1 - n]) * 100.0)


def roc(series: pd.Series, n: int) -> float | None:
    """Rate of change (percent return) over the last n bars."""
    s = series.dropna()
    if len(s) <= n or s.iloc[-1 - n] == 0:
        return None
    return float((s.iloc[-1] - s.iloc[-1 - n]) / abs(s.iloc[-1 - n]) * 100.0)


def donchian(high: pd.Series, low: pd.Series, n: int = 55) -> tuple[pd.Series, pd.Series]:
    """Donchian channel (Turtle/trend-following): rolling highest-high & lowest-low."""
    return high.rolling(n).max(), low.rolling(n).min()


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def ann_return_pct(close: pd.Series, periods_per_year: float = 252.0) -> float | None:
    r = log_returns(close).dropna()
    if len(r) < 5:
        return None
    return float((np.exp(r.mean() * periods_per_year) - 1.0) * 100.0)


def ann_vol_pct(close: pd.Series, periods_per_year: float = 252.0) -> float | None:
    r = log_returns(close).dropna()
    if len(r) < 5:
        return None
    return float(r.std(ddof=0) * np.sqrt(periods_per_year) * 100.0)


def sharpe(close: pd.Series, periods_per_year: float = 252.0, rf: float = 0.0) -> float | None:
    """Annualized Sharpe ratio of the price series' periodic returns (rf annual %)."""
    r = log_returns(close).dropna()
    if len(r) < 20 or r.std(ddof=0) == 0:
        return None
    rf_per = rf / 100.0 / periods_per_year
    return float((r.mean() - rf_per) / r.std(ddof=0) * np.sqrt(periods_per_year))


def max_drawdown_pct(close: pd.Series, lookback: int | None = None) -> float | None:
    """Largest peak-to-trough decline (%) over the window (returns a negative number)."""
    s = close.dropna()
    if lookback:
        s = s.tail(lookback)
    if len(s) < 5:
        return None
    roll_max = s.cummax()
    dd = (s / roll_max - 1.0) * 100.0
    return float(dd.min())


def beta_alpha(
    stock_close: pd.Series, bench_close: pd.Series, ppy: float = 252.0,
) -> tuple[float | None, float | None, float | None]:
    """Beta, annualized alpha (%), and relative strength (%) vs a benchmark,
    computed over the overlapping window of the two price series."""
    a = log_returns(stock_close).dropna()
    b = log_returns(bench_close).dropna()
    idx = a.index.intersection(b.index)
    if len(idx) < 20:
        return None, None, None
    a, b = a.loc[idx], b.loc[idx]
    var_b = float(b.var(ddof=0))
    beta = float(a.cov(b) / var_b) if var_b > 0 else None
    alpha = None
    if beta is not None:
        alpha_per = float(a.mean() - beta * b.mean())
        alpha = (np.exp(alpha_per * ppy) - 1.0) * 100.0
    rel = float((a.sum() - b.sum()) * 100.0)   # cumulative log-return spread, %
    return (round(beta, 2) if beta is not None else None,
            round(alpha, 1) if alpha is not None else None,
            round(rel, 1))


def volume_profile_poc(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    bins: int = 24, lookback: int = 120,
) -> tuple[float | None, float | None]:
    """Volume-by-price point of control (POC).

    Buckets the last `lookback` bars' volume by typical price and returns
    (poc_price, poc_share) where poc_price is the most-traded price level and
    poc_share is the fraction of window volume transacted in that bucket — a
    high share marks a strong support/resistance shelf institutions defended.
    """
    h, l, c, v = (s.dropna().tail(lookback) for s in (high, low, close, volume))
    idx = c.index
    h, l, v = h.reindex(idx), l.reindex(idx), v.reindex(idx)
    typ = (h + l + c) / 3.0
    typ, v = typ.dropna(), v.reindex(typ.index).fillna(0.0)
    if len(typ) < 10 or v.sum() <= 0:
        return None, None
    lo, hi = float(typ.min()), float(typ.max())
    if hi <= lo:
        return float(typ.iloc[-1]), 1.0
    edges = np.linspace(lo, hi, bins + 1)
    which = np.clip(np.digitize(typ.values, edges) - 1, 0, bins - 1)
    vol_by_bin = np.zeros(bins)
    for b, vol in zip(which, v.values):
        vol_by_bin[b] += vol
    poc_bin = int(vol_by_bin.argmax())
    poc_price = float((edges[poc_bin] + edges[poc_bin + 1]) / 2.0)
    poc_share = float(vol_by_bin[poc_bin] / vol_by_bin.sum())
    return poc_price, poc_share


def divergence(price: pd.Series, osc: pd.Series, lookback: int = 40) -> tuple[int, str]:
    """Detect a regular price/oscillator divergence over the last `lookback` bars.

    Returns (+1, msg) for a bullish divergence (price lower-low, oscillator
    higher-low), (-1, msg) for bearish (price higher-high, oscillator lower-high),
    or (0, "") if none. A coarse two-window comparison — robust and explainable.
    """
    p = price.dropna()
    o = osc.reindex(p.index).dropna()
    p = p.loc[o.index]
    if len(p) < lookback:
        return 0, ""
    p, o = p.tail(lookback), o.tail(lookback)
    half = len(p) // 2
    old_p, new_p = p.iloc[:half], p.iloc[half:]
    old_o, new_o = o.iloc[:half], o.iloc[half:]
    if new_p.max() > old_p.max() and new_o.max() < old_o.max():
        return -1, "Bearish divergence: price made a higher high but momentum made a lower high"
    if new_p.min() < old_p.min() and new_o.min() > old_o.min():
        return 1, "Bullish divergence: price made a lower low but momentum made a higher low"
    return 0, ""
