"""Correctness/leakage checks for the ML pipeline. Run directly (after
`pip install -r requirements.txt`, which pulls in the ML deps these tests need):

    python -m tests.test_ml_features

`test_no_lookahead_feature_frame` is the single most important test here: it
is the automated, direct proof that no lookahead bug exists in feature
construction -- not just an assertion in prose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics import indicators as ind
from app.ml import features, labels as labels_mod, validation


def _approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def _synthetic_bars(n=100, seed=1, shock_at=None, shock_mult=5.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    price = np.maximum(100 + np.cumsum(rng.normal(0, 1, n)), 1.0)
    if shock_at is not None:
        price = price.copy()
        price[shock_at:] *= shock_mult
    close = pd.Series(price, index=dates)
    high, low = close * 1.01, close * 0.99
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(rng.uniform(1e5, 1e6, n), index=dates)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_causal_equivalence():
    """Family-1 indicators (rolling/ewm Series) must give identical values
    whether computed once over the full series and indexed at t, or
    recomputed on a series truncated to t+1 -- this is the invariant
    features.py relies on for performance."""
    bars = _synthetic_bars(n=120, seed=3)
    close = bars["close"]
    rsi_full = ind.rsi(close, 14)
    macd_full, _sig_full, _hist_full = ind.macd(close)
    for t in (30, 50, 80, 119):
        rsi_trunc = ind.rsi(close.iloc[: t + 1], 14).iloc[-1]
        assert _approx(rsi_full.iloc[t], rsi_trunc, tol=1e-9)
        macd_trunc = ind.macd(close.iloc[: t + 1])[0].iloc[-1]
        assert _approx(macd_full.iloc[t], macd_trunc, tol=1e-9)


def test_no_lookahead_feature_frame():
    """THE critical test: inject a large price shock after a cutoff date and
    assert every feature row dated BEFORE the cutoff is identical whether or
    not the shock exists later in the same DataFrame passed in."""
    n, shock_at = 100, 60
    normal = _synthetic_bars(n=n, seed=1)
    shocked = _synthetic_bars(n=n, seed=1, shock_at=shock_at, shock_mult=8.0)

    f_normal = features.build_feature_frame("TEST", normal)
    f_shocked = features.build_feature_frame("TEST", shocked)

    cutoff = normal.index[shock_at]
    a = f_normal[f_normal["date"] < cutoff].drop(columns=["date", "code"]).reset_index(drop=True)
    b = f_shocked[f_shocked["date"] < cutoff].drop(columns=["date", "code"]).reset_index(drop=True)

    assert len(a) > 0, "test setup produced no pre-cutoff rows -- widen the window"
    assert a.shape == b.shape
    for col in a.columns:
        av, bv = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        both_nan = np.isnan(av) & np.isnan(bv)
        close_enough = np.isclose(av, bv, atol=1e-9, equal_nan=False)
        assert np.all(both_nan | close_enough), f"lookahead leak detected in feature column '{col}'"


def test_label_horizon_drop():
    n, horizon = 50, 10
    bars_by_code = {f"SYM{i}": _synthetic_bars(n=n, seed=i) for i in range(6)}
    rows = [
        {"date": d, "code": code, "dummy": 0.0}
        for code, bars in bars_by_code.items() for d in bars.index
    ]
    feat_df = pd.DataFrame(rows)
    out = labels_mod.add_forward_labels(feat_df, bars_by_code, horizon=horizon, min_symbols_per_date=5)

    too_late = set(bars_by_code["SYM0"].index[-horizon:])
    present = set(out[out["code"] == "SYM0"]["date"])
    assert too_late.isdisjoint(present), "rows within the label horizon must never get a label"


def test_cross_sectional_median_threshold():
    n, horizon = 50, 5
    bars_by_code = {f"SYM{i}": _synthetic_bars(n=n, seed=i) for i in range(6)}
    thin_date = list(bars_by_code["SYM0"].index)[20]

    rows = []
    for i, (code, bars) in enumerate(bars_by_code.items()):
        for d in bars.index:
            if d == thin_date and i >= 2:  # only 2 of 6 symbols present on thin_date
                continue
            rows.append({"date": d, "code": code, "dummy": 0.0})
    feat_df = pd.DataFrame(rows)
    out = labels_mod.add_forward_labels(feat_df, bars_by_code, horizon=horizon, min_symbols_per_date=5)

    assert thin_date not in set(out["date"]), "a date with too few symbols must be dropped"
    normal_date = list(bars_by_code["SYM0"].index)[10]
    assert normal_date in set(out["date"]), "a normal, fully-populated date must survive"


def test_vol_adjusted_label_mode():
    """vol_adjusted must (a) keep fwd_ret RAW, (b) still emit a balanced binary
    label, and (c) actually differ from the median label when names have very
    different vols -- otherwise the mode is a silent no-op."""
    n, horizon = 60, 5
    # Mix of calm and jumpy names so raw-return ranking != vol-adjusted ranking.
    bars_by_code = {}
    for i in range(6):
        mult = 5.0 if i % 2 else 1.0  # every other name is 5x more volatile
        b = _synthetic_bars(n=n, seed=100 + i)
        c = 100 + (b["close"] - 100) * mult
        bars_by_code[f"SYM{i}"] = b.assign(close=c, high=c * 1.01, low=c * 0.99)
    rows = [{"date": d, "code": code, "dummy": 0.0}
            for code, bars in bars_by_code.items() for d in bars.index]
    feat_df = pd.DataFrame(rows)

    med = labels_mod.add_forward_labels(feat_df, bars_by_code, horizon, label_mode="median")
    vol = labels_mod.add_forward_labels(feat_df, bars_by_code, horizon, label_mode="vol_adjusted")

    assert not vol.empty
    assert set(vol["label"].unique()) <= {0.0, 1.0}
    # cross-sectional median split stays ~balanced (ties/odd counts drift it a bit)
    assert 0.3 <= float(vol["label"].mean()) <= 0.7, "vol_adjusted label should be ~balanced"
    # target-scaling helpers must not leak into the output frame; fwd_ret raw
    assert "trail_vol" not in vol.columns and "_target" not in vol.columns
    # the two modes must disagree on at least some rows given the vol spread
    joined = med.merge(vol, on=["code", "date"], suffixes=("_med", "_vol"))
    assert (joined["label_med"] != joined["label_vol"]).any(), "vol_adjusted collapsed to the median label"

    try:
        labels_mod.add_forward_labels(feat_df, bars_by_code, horizon, label_mode="bogus")
        assert False, "unknown label_mode should raise"
    except ValueError:
        pass


def test_non_overlapping_subsample():
    """Kept rows per symbol must be >= horizon bars apart (non-overlapping label
    windows), preserve columns, and no-op for horizon<=1."""
    n, horizon = 80, 10
    bars_by_code = {f"SYM{i}": _synthetic_bars(n=n, seed=200 + i) for i in range(6)}
    rows = [{"date": d, "code": code, "dummy": 0.0}
            for code, bars in bars_by_code.items() for d in bars.index]
    feat_df = pd.DataFrame(rows)
    labeled = labels_mod.add_forward_labels(feat_df, bars_by_code, horizon)

    thinned = labels_mod.subsample_non_overlapping(labeled, horizon)
    assert len(thinned) < len(labeled), "subsample should drop overlapping rows"
    assert set(thinned.columns) == set(labeled.columns)
    # within each symbol, consecutive kept dates are >= horizon business days apart
    for _code, grp in thinned.groupby("code"):
        ds = grp["date"].sort_values().to_list()
        for a, b in zip(ds, ds[1:]):
            gap = len(pd.bdate_range(a, b)) - 1
            assert gap >= horizon, f"kept rows only {gap} bars apart (< {horizon})"
    # horizon<=1 is a no-op
    assert len(labels_mod.subsample_non_overlapping(labeled, 1)) == len(labeled)


def test_htf_weekly_equivalence():
    """The precomputed weekly-close buckets in features.py must produce the
    exact same htf_trend_score as the original per-row full resample of the
    truncated daily frame -- for EVERY row, not a sample."""
    from app.analytics import technical

    bars = _synthetic_bars(n=140, seed=7)
    f = features.build_feature_frame("TEST", bars)
    assert not f.empty

    close = bars["close"]
    for _, r in f.iterrows():
        t = bars.index.get_loc(r["date"])
        weekly_naive = bars.iloc[:t + 1].resample("W").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        expected, _msg = technical.trend_score(weekly_naive)
        assert _approx(r["htf_trend_score"], expected, tol=1e-12), (
            f"weekly precompute diverged from naive resample at {r['date']}"
        )


def test_last_row_only_matches_full():
    """last_row_only=True must return exactly the final row of the full frame
    (rows are independent -- this is the invariant inference.py relies on)."""
    bars = _synthetic_bars(n=120, seed=9)
    full = features.build_feature_frame("TEST", bars)
    last = features.build_feature_frame("TEST", bars, last_row_only=True)

    assert len(last) == 1
    a = full.iloc[[-1]].reset_index(drop=True)
    b = last.reset_index(drop=True)
    assert a["date"].iloc[0] == b["date"].iloc[0]
    for col in features.FEATURE_COLUMNS:
        av, bv = a[col].iloc[0], b[col].iloc[0]
        if pd.isna(av) and pd.isna(bv):
            continue
        assert _approx(av, bv, tol=1e-12), f"last_row_only mismatch in '{col}'"


def test_purge_embargo_no_overlap():
    dates = pd.bdate_range("2023-01-01", periods=500)
    horizon = 10
    splits = validation.purged_walkforward_splits(dates, horizon=horizon, n_folds=6, min_train_months=6)
    assert len(splits) >= 3, "2 years of synthetic business days should yield real folds"

    buffer = pd.Timedelta(days=int(round(horizon * 1.5)))
    for train_dates, test_dates in splits:
        test_start, test_end = min(test_dates), max(test_dates)
        assert set(train_dates).isdisjoint(set(test_dates))
        for d in train_dates:
            assert not (test_start - buffer <= d < test_start), f"purge violated: {d} near test_start {test_start}"
            assert not (test_end < d <= test_end + buffer), f"embargo violated: {d} near test_end {test_end}"


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} ML pipeline tests passed.")


if __name__ == "__main__":
    main()
