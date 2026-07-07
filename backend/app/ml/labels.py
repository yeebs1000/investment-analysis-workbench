"""Forward-return label construction.

This is the ONE place in the pipeline that is allowed to look into the future
-- that's the entire point of a training label (grading what already
happened). Deliberately kept in its own module, separate from `features.py`,
so the leakage boundary is visually obvious in review: this file only ever
*adds* label columns onto an already-built feature frame; it never touches or
recomputes a feature column, and the feature frame passed in must already be
fully built (by `features.py`, which has the opposite invariant: never look
forward) before this runs.
"""
from __future__ import annotations

import pandas as pd


VOL_WINDOW = 20  # trailing-return-vol window for the vol_adjusted label target


def add_forward_labels(
    feat_df: pd.DataFrame,
    bars_by_code: dict[str, pd.DataFrame],
    horizon: int,
    min_symbols_per_date: int = 5,
    label_mode: str = "median",
) -> pd.DataFrame:
    """Add `fwd_ret` (raw forward return) and `label` (1 = beat the
    cross-sectional cutoff on that date) columns. Rows with no valid forward
    window, or on dates too thin to rank cross-sectionally, are dropped.

    label_mode:
      - "median" (default): rank on RAW forward return. What the shipped model
        was trained on; unchanged.
      - "vol_adjusted": rank on forward return divided by each name's trailing
        return vol (as of the row's date -- causal). This stops the
        cross-section from being dominated by whichever high-beta names simply
        moved most, so the target rewards genuine relative strength rather than
        raw amplitude. `fwd_ret` itself is left RAW either way, so the
        walk-forward IC is always measured against real returns.

    Which target actually produces a sharper OOS edge is decided by the
    training run's own purged walk-forward + gating, not asserted here -- run
    both and compare the reports."""
    empty_out = feat_df.iloc[0:0].assign(fwd_ret=pd.Series(dtype=float), label=pd.Series(dtype=float))
    if feat_df.empty:
        return empty_out
    if label_mode not in ("median", "vol_adjusted"):
        raise ValueError(f"unknown label_mode {label_mode!r} (use 'median' or 'vol_adjusted')")

    # 1. forward N-bar return per (code, date), from each symbol's OWN bars --
    #    shift(-horizon) is the deliberate, sole forward-looking operation here.
    #    trailing vol is as-of-date (past returns only) -- causal, safe as a
    #    target scaler even though it rides alongside the forward return.
    fwd_frames = []
    for code, bars in bars_by_code.items():
        if bars.empty or len(bars) <= horizon:
            continue
        close = bars["close"]
        fwd_ret = close.shift(-horizon) / close - 1.0  # NaN on the last `horizon` rows -- no valid label there
        trail_vol = close.pct_change().rolling(VOL_WINDOW).std()
        fwd_frames.append(pd.DataFrame({
            "code": code, "date": bars.index,
            "fwd_ret": fwd_ret.values, "trail_vol": trail_vol.values,
        }))
    if not fwd_frames:
        return empty_out
    fwd_df = pd.concat(fwd_frames, ignore_index=True)

    # 2. join onto the (already fully-built, causal) feature frame
    merged = feat_df.merge(fwd_df, on=["code", "date"], how="left")
    merged = merged.dropna(subset=["fwd_ret"])

    # 3. the value we rank cross-sectionally: raw return, or vol-adjusted.
    if label_mode == "vol_adjusted":
        merged = merged[merged["trail_vol"].notna() & (merged["trail_vol"] > 0)].copy()
        merged["_target"] = merged["fwd_ret"] / merged["trail_vol"]
    else:
        merged = merged.copy()
        merged["_target"] = merged["fwd_ret"]

    # 4. cross-sectional median-rank label; only on dates with enough symbols
    #    to make "the median" a meaningful, non-degenerate cutoff.
    counts = merged.groupby("date")["_target"].transform("count")
    merged = merged[counts >= min_symbols_per_date].copy()
    if merged.empty:
        return empty_out
    medians = merged.groupby("date")["_target"].transform("median")
    merged["label"] = (merged["_target"] > medians).astype(float)

    return merged.drop(columns=["_target", "trail_vol"]).reset_index(drop=True)


def subsample_non_overlapping(labeled: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Thin the labeled rows so each symbol's kept observations are >= `horizon`
    bars apart, i.e. their forward-return label windows never overlap.

    Why: with daily rows and an N-bar label, consecutive observations share
    ~N days of their forward window, so the ~thousands of rows are nowhere near
    independent -- which is exactly why the walk-forward CI on OOS AUC comes out
    so wide. Keeping one row per horizon per symbol trades raw row count for
    genuinely independent samples, giving an honest (usually tighter, sometimes
    just more truthful) read. Opt-in -- the default path is unchanged."""
    if labeled.empty or horizon <= 1:
        return labeled
    keep = []
    for _code, grp in labeled.sort_values(["code", "date"]).groupby("code", sort=False):
        # positional stride within each symbol's own chronological rows
        keep.append(grp.iloc[::horizon])
    return pd.concat(keep, ignore_index=True) if keep else labeled
