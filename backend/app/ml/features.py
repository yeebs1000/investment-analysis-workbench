"""Causal, point-in-time feature construction for ML training.

Reuses every primitive from `app.analytics.indicators` as-is -- no duplicate
indicator math, single source of truth with the live rule-based engine.

LOAD-BEARING INVARIANT: the feature row for date `t` must only ever be able to
"see" `bars.iloc[:t+1]`. Two families of indicator function, handled differently:

- "Rolling-Series" functions (sma/ema/rsi/macd/bollinger/atr/adx/obv/stochastic/
  donchian) are provably causal by construction (pandas .rolling()/.ewm() never
  look forward -- verified against every function body). Computing them ONCE
  over the FULL bars frame and then indexing `.iloc[t]` is therefore exactly
  equivalent to truncating the frame to `.iloc[:t+1]` and recomputing --
  cheaper, and proven equivalent by `tests/test_ml_features.py::test_causal_equivalence`.
- "Scalar as-of-latest" functions (roc, slope_pct, sharpe, max_drawdown_pct,
  ann_vol_pct, beta_alpha, divergence, volume_profile_poc) return ONE number
  describing "as of the last element of the series you passed in" -- these have
  NO safe "compute once" shortcut and MUST be called on a `.iloc[:t+1]`-truncated
  slice for every t, or they silently leak the future. This is the single
  easiest lookahead bug to introduce; every call site below is truncated.

Higher-timeframe (weekly) trend uses the SAME invariant: resample only
`bars.iloc[:t+1]` (never the live `AnalysisService._bars(code, "week")`, which
always fetches through "today").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics import indicators as ind
from app.analytics import technical

MIN_BARS = technical.MIN_BARS         # 30 -- same floor the live engine uses
SHARPE_WINDOW = 60
BETA_WINDOW = 120
REL_STRENGTH_WINDOW = 60

FEATURE_COLUMNS = [
    "rsi14", "rsi2", "rsi_slope_pct", "macd", "macd_signal", "macd_hist",
    "pct_b", "bandwidth", "atr_pct", "adx", "plus_di", "minus_di",
    "stoch_k", "stoch_d", "volume_ratio", "obv_slope_pct",
    "sma50_vs_sma200", "donchian_pos", "roc20", "roc63", "roc126", "ts_momentum",
    "sharpe_60", "max_dd_60", "ann_vol_60", "beta_120", "rel_strength_60",
    "rsi_divergence", "poc_rel_pos", "htf_trend_score",
]


def build_feature_frame(
    code: str, bars: pd.DataFrame, bench_bars: pd.DataFrame | None = None,
    ppy: float = 252.0, last_row_only: bool = False,
) -> pd.DataFrame:
    """One feature row per date, for every date with >= MIN_BARS of prior history.

    `last_row_only=True` computes just the final row -- rows are independent
    of each other, so this is exactly the last row of the full frame, at ~1/n
    of the cost. Used by request-time inference, which only ever reads
    `.iloc[-1]` (equivalence proven by test_last_row_only_matches_full)."""
    n = len(bars)
    if n < MIN_BARS + 1:
        return pd.DataFrame(columns=["date", "code", *FEATURE_COLUMNS])

    close, high, low, vol = bars["close"], bars["high"], bars["low"], bars["volume"]

    # --- family 1: rolling-Series indicators, computed ONCE (causal; index-at-t is exact) ---
    rsi14_s = ind.rsi(close, 14)
    rsi2_s = ind.rsi(close, 2)
    macd_line, macd_sig, macd_hist = ind.macd(close)
    _mid, _bb_up, _bb_lo, pct_b_s, bandwidth_s = ind.bollinger(close, 20, 2)
    atr_s = ind.atr(high, low, close, 14)
    adx_s, plus_di_s, minus_di_s = ind.adx(high, low, close, 14)
    obv_s = ind.obv(close, vol)
    stoch_k_s, stoch_d_s = ind.stochastic(high, low, close)
    don_n = min(55, n - 1)
    don_up_s, don_lo_s = ind.donchian(high, low, don_n)
    sma50_s = ind.sma(close, 50)
    sma200_s = ind.sma(close, 200)
    vol_avg20_s = vol.rolling(20).mean()

    # --- higher-timeframe weekly closes, precomputed ONCE (was an O(n^2) full
    # resample of `bars.iloc[:t+1]` per row). Causally equivalent because
    # technical.trend_score only reads weekly *closes*, every completed week's
    # close is fixed regardless of t, and the current (possibly partial)
    # week's close at bar t is exactly close.iloc[t]. resample("W") means
    # W-SUN buckets with empty weeks dropna'd -- identical to grouping bars by
    # to-period("W") and taking each group's last close. Equivalence proven by
    # tests/test_ml_features.py::test_htf_weekly_equivalence.
    week_ids = pd.PeriodIndex(bars.index, freq="W")
    first_of_week = np.flatnonzero(np.r_[True, week_ids[1:] != week_ids[:-1]])
    week_last_close = close.to_numpy()[np.r_[first_of_week[1:] - 1, n - 1]]

    rows: list[dict] = []
    start_t = max(MIN_BARS, n - 1) if last_row_only else MIN_BARS
    for t in range(start_t, n):
        date = bars.index[t]
        price = close.iloc[t]
        if pd.isna(price) or price == 0:
            continue

        row: dict = {"date": date, "code": code}

        # family 1 -- direct index at t
        row["rsi14"] = rsi14_s.iloc[t]
        row["rsi2"] = rsi2_s.iloc[t]
        row["macd"] = macd_line.iloc[t]
        row["macd_signal"] = macd_sig.iloc[t]
        row["macd_hist"] = macd_hist.iloc[t]
        row["pct_b"] = pct_b_s.iloc[t]
        row["bandwidth"] = bandwidth_s.iloc[t]
        atr_v = atr_s.iloc[t]
        row["atr_pct"] = float(atr_v / price * 100.0) if pd.notna(atr_v) else None
        row["adx"] = adx_s.iloc[t]
        row["plus_di"] = plus_di_s.iloc[t]
        row["minus_di"] = minus_di_s.iloc[t]
        row["stoch_k"] = stoch_k_s.iloc[t]
        row["stoch_d"] = stoch_d_s.iloc[t]
        v20 = vol_avg20_s.iloc[t]
        row["volume_ratio"] = float(vol.iloc[t] / v20) if (pd.notna(v20) and v20) else None
        s50, s200 = sma50_s.iloc[t], sma200_s.iloc[t]
        row["sma50_vs_sma200"] = (1.0 if s50 > s200 else 0.0) if (pd.notna(s50) and pd.notna(s200)) else None
        du, dl = don_up_s.iloc[t], don_lo_s.iloc[t]
        row["donchian_pos"] = float((price - dl) / (du - dl)) if (pd.notna(du) and pd.notna(dl) and du > dl) else None

        # family 2 -- MUST truncate to .iloc[:t+1] per date, no shortcut
        close_upto = close.iloc[:t + 1]
        row["rsi_slope_pct"] = ind.slope_pct(rsi14_s.iloc[:t + 1], 3)
        row["obv_slope_pct"] = ind.slope_pct(obv_s.iloc[:t + 1], 10)
        row["roc20"] = ind.roc(close_upto, 20)
        # Intermediate-horizon momentum (~3mo / ~6mo): the Jegadeesh-Titman band
        # that sits between the short roc20 (which often mean-reverts) and the
        # 12-month ts_momentum. min(N, t) matches ts_momentum's convention so
        # early rows still get a (shorter-window) value instead of being dropped.
        row["roc63"] = ind.roc(close_upto, min(63, t))
        row["roc126"] = ind.roc(close_upto, min(126, t))
        row["ts_momentum"] = ind.roc(close_upto, min(252, t))
        row["sharpe_60"] = ind.sharpe(close_upto.tail(SHARPE_WINDOW), ppy)
        row["max_dd_60"] = ind.max_drawdown_pct(close_upto, lookback=SHARPE_WINDOW)
        row["ann_vol_60"] = ind.ann_vol_pct(close_upto.tail(SHARPE_WINDOW), ppy)

        div_dir, _msg = ind.divergence(close_upto, rsi14_s.iloc[:t + 1], lookback=40)
        row["rsi_divergence"] = float(div_dir)

        poc_price, poc_share = ind.volume_profile_poc(
            high.iloc[:t + 1], low.iloc[:t + 1], close_upto, vol.iloc[:t + 1],
            lookback=min(120, t + 1),
        )
        row["poc_rel_pos"] = (
            float((price - poc_price) / price * 100.0)
            if (poc_price is not None and poc_share is not None and poc_share >= 0.10)
            else None
        )

        if bench_bars is not None and not bench_bars.empty:
            bench_close_upto = bench_bars["close"].loc[:date]
            beta120, _a1, _r1 = ind.beta_alpha(close_upto.tail(BETA_WINDOW), bench_close_upto.tail(BETA_WINDOW), ppy)
            _b2, _a2, rel60 = ind.beta_alpha(close_upto.tail(REL_STRENGTH_WINDOW), bench_close_upto.tail(REL_STRENGTH_WINDOW), ppy)
            row["beta_120"] = beta120
            row["rel_strength_60"] = rel60
        else:
            row["beta_120"] = None
            row["rel_strength_60"] = None

        # higher-timeframe (weekly) trend -- completed weeks' closes from the
        # precomputed buckets above, plus the current partial week's close at
        # t (never the live weekly fetch, which sees through "today").
        w = int(np.searchsorted(first_of_week, t, side="right")) - 1
        weekly_close = np.append(week_last_close[:w], close.iloc[t])
        htf_score, _summary = technical.trend_score(pd.DataFrame({"close": weekly_close}))
        row["htf_trend_score"] = htf_score

        rows.append(row)

    return pd.DataFrame(rows)


def build_universe_features(
    store, codes: list[str], bench_code: str = "US.SPY", ppy: float = 252.0,
) -> pd.DataFrame:
    """Load each symbol's persisted bars, build its feature frame, concatenate.

    `bench_code` supplies bench_bars for every other symbol but is never itself
    trained on (it's a feature input, not a name we'd ever recommend trading).
    """
    bench_bars = store.load(bench_code)
    frames = []
    for code in codes:
        if code == bench_code:
            continue
        bars = store.load(code)
        if bars.empty:
            continue
        f = build_feature_frame(code, bars, bench_bars, ppy)
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["date", "code", *FEATURE_COLUMNS])
    return pd.concat(frames, ignore_index=True)
