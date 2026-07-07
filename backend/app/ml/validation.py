"""Purged, expanding-window walk-forward cross-validation.

Offline-only (imported by `app/ml/train.py`, never the live request path) --
requires scikit-learn/scipy to be installed; unlike `inference.py` there's no
need for a lazy-import/graceful-degrade here, since this only ever runs when
the user has deliberately invoked the training CLI.

Why purged walk-forward and not a simpler split: a random train/test split (or
plain k-fold) leaks on overlapping-label time series data -- a training row's
forward-return label window can overlap a test row's feature window whenever
they're temporally close, regardless of which "fold" they landed in. Expanding
walk-forward (train on everything before month T, test on month T) is already
close to leak-free by construction; purging (drop training rows whose label
window would reach into the test period) and embargo (drop a buffer after
each test fold from ever being used as training) close the remaining gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score


def purged_walkforward_splits(
    dates, horizon: int, n_folds: int = 6, min_train_months: int = 6,
) -> list[tuple[list, list]]:
    """Returns a list of (train_dates, test_dates). Each fold's test period is
    one calendar month; training is everything strictly before it, minus a
    purge/embargo buffer (~1.5x `horizon` trading bars, in calendar days) drawn
    around EVERY fold's test window. Returns [] (a hard stop, not a fabricated
    smaller answer) if there isn't enough history for at least 3 real folds."""
    uniq_dates = sorted(pd.to_datetime(pd.Series(list(dates))).unique())
    if not uniq_dates:
        return []
    uniq_dates = [pd.Timestamp(d) for d in uniq_dates]
    months = sorted({(d.year, d.month) for d in uniq_dates})
    if len(months) <= min_train_months:
        return []

    testable_months = months[min_train_months:]
    n_folds = min(n_folds, len(testable_months))
    if n_folds < 3:
        return []

    buffer = pd.Timedelta(days=int(round(horizon * 1.5)))
    fold_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for ym in testable_months[:n_folds]:
        td = [d for d in uniq_dates if (d.year, d.month) == ym]
        if td:
            fold_windows.append((min(td), max(td)))

    # union of purge (before each test start) + embargo (after each test end)
    # zones, applied uniformly -- conservative, and simple enough to unit test.
    excluded: set[pd.Timestamp] = set()
    for test_start, test_end in fold_windows:
        for d in uniq_dates:
            if test_start - buffer <= d < test_start:
                excluded.add(d)
            if test_end < d <= test_end + buffer:
                excluded.add(d)

    splits = []
    for test_start, test_end in fold_windows:
        test_dates = [d for d in uniq_dates if test_start <= d <= test_end]
        train_dates = [d for d in uniq_dates if d < test_start and d not in excluded]
        if train_dates and test_dates:
            splits.append((train_dates, test_dates))
    return splits


def evaluate_fold(model, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> dict | None:
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test, y_test = test_df[feature_cols], test_df["label"]
    if y_train.nunique() < 2 or len(X_test) < 5:
        return None  # not enough signal/data to fit or evaluate meaningfully

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() >= 2 else None
    acc = float(accuracy_score(y_test, pred))
    ic_val, _p = spearmanr(proba, test_df["fwd_ret"]) if len(proba) >= 3 else (None, None)
    ic = float(ic_val) if ic_val is not None and not pd.isna(ic_val) else None

    return {"auc": auc, "accuracy": acc, "ic": ic, "n_train": len(X_train), "n_test": len(X_test)}


def run_walkforward(
    df: pd.DataFrame, feature_cols: list[str], model_factory, horizon: int,
    n_folds: int = 6, min_train_months: int = 6,
) -> pd.DataFrame:
    """Fit+evaluate `model_factory()` on every purged walk-forward fold. Rows
    with any NaN feature are dropped up front (conservative -- no imputation,
    so a thin sample size is visible in the result, not hidden)."""
    df = df.dropna(subset=[*feature_cols, "label", "fwd_ret"]).copy()
    if df.empty:
        return pd.DataFrame()
    splits = purged_walkforward_splits(df["date"], horizon, n_folds, min_train_months)
    results = []
    for i, (train_dates, test_dates) in enumerate(splits):
        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"].isin(test_dates)]
        metrics = evaluate_fold(model_factory(), train_df, test_df, feature_cols)
        if metrics is not None:
            metrics.update(fold=i + 1, test_start=min(test_dates), test_end=max(test_dates))
            results.append(metrics)
    return pd.DataFrame(results)


def shuffle_labels_within_date(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Leakage sanity control: shuffle `label` within each date (preserves the
    cross-sectional class balance). Re-running the walk-forward harness on
    this should collapse OOS AUC to ~0.50 -- if it doesn't, a leak exists
    somewhere upstream and no real result from this pipeline should be trusted
    until it's found."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    out["label"] = out.groupby("date")["label"].transform(
        lambda s: rng.permutation(s.values)
    )
    return out


def block_bootstrap_ci(
    fold_metrics: pd.DataFrame, metric: str = "auc", n_boot: int = 500, ci: float = 0.90, seed: int = 42,
) -> tuple[float, float] | None:
    """Resample FOLDS (not rows) with replacement -- respects within-fold
    autocorrelation, unlike a naive row-level bootstrap. Returns a percentile
    CI on the mean of `metric` across folds, or None if too few folds."""
    vals = fold_metrics[metric].dropna().to_numpy()
    if len(vals) < 3:
        return None
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return float(np.percentile(boot_means, lo)), float(np.percentile(boot_means, hi))
