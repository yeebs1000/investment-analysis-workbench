"""Offline training CLI: build the historical store, train + validate the ML
signal, and (best-effort, separately) recalibrate the rule-based WEIGHTS and
Kelly slope from the same walk-forward evidence.

    python -m app.ml.train [--tf day] [--horizons 10,20] [--folds 6]

Writes `data_store/models/current.json` (activating a new ML component)
ONLY if gating thresholds are met -- otherwise prints why not and leaves any
previous model (or none) serving. `WEIGHTS`/`KELLY_SLOPE` are NEVER written
back into technical.py automatically; the report prints paste-ready suggested
values for a human to review and commit by hand.

Never scheduled -- this is a deliberate, manual, offline step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from app.analytics import technical
from app.ml import data_store, features, labels as labels_mod, models as ml_models, universe, validation

MIN_UNIVERSE_SIZE = 15
MIN_FOLDS = 3
RECAL_SAMPLE_CAP_DEFAULT = 3000
# Moomoo caps request_history_kline at 60 calls / 30s. Space fetches ~0.55s
# apart to stay safely under it (a broad universe like sp500 otherwise gets
# throttled after the first 60 symbols -- including the benchmark, which then
# aborts the whole run). ~0.55s * 500 symbols ≈ 4.5 min of fetch.
KLINE_MIN_INTERVAL_S = 0.55

COMPONENT_KEYS = ["trend", "momentum", "volatility", "volume", "levels", "quant"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tf", default="day")
    ap.add_argument("--horizons", default="10,20", help="comma-separated forward-bar horizons to try")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--label-mode", default="median", choices=["median", "vol_adjusted"],
                    help="cross-sectional target: raw forward return (median) or "
                         "vol-adjusted forward return (vol_adjusted). Run both and "
                         "compare the reports -- the walk-forward decides which is sharper.")
    ap.add_argument("--non-overlapping", action="store_true",
                    help="keep only rows >= horizon bars apart per symbol, so label "
                         "windows don't overlap. Fewer but independent samples -> an "
                         "honest (usually tighter) OOS confidence interval. May starve "
                         "folds on a small universe -- widen the universe if so.")
    ap.add_argument("--min-train-months", type=int, default=6)
    ap.add_argument("--groups", default=None, help="comma-separated watchlist group names (default: all)")
    ap.add_argument("--universe", default="holdings",
                    help="training universe: 'holdings' (your positions + watchlists, default), "
                         "'sp500' (the bundled S&P 500 list -- broad, fixes survivorship/sample "
                         "size but a long broker fetch), 'smallcap' (sector-stratified small/"
                         "micro-cap sample via financedatabase -- tests the small-cap/hot-sector "
                         "growth tilt out-of-sample), or a path to your own file of "
                         "MARKET.SYMBOL codes (one per line).")
    ap.add_argument("--recal-sample-cap", type=int, default=RECAL_SAMPLE_CAP_DEFAULT)
    ap.add_argument("--lookback-days", type=int, default=1095,
                    help="history window for the training snapshot (default 3y). Must be much "
                         "longer than the live 430-day default: sma200/beta_120 consume the "
                         "first ~200 bars of any window, and the walk-forward needs >6 months "
                         "of COMPLETE feature rows after that or it produces zero folds.")
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    # Progress must be visible while the run is live: without line buffering,
    # a redirected stdout holds everything until exit and the run looks hung.
    sys.stdout.reconfigure(line_buffering=True)

    from app.services.analysis_service import BENCHMARK_CODE, TIMEFRAMES, service
    ppy = TIMEFRAMES[args.tf]["ppy"]
    lookback_days = max(args.lookback_days, TIMEFRAMES[args.tf]["lookback_days"])

    print("== 1. Resolving universe ==")
    codes = universe.resolve_universe(service, source=args.universe)
    tradable = [c for c in codes if c != BENCHMARK_CODE]
    print(f"  source={args.universe}: {len(tradable)} tradable symbols + benchmark")
    if len(tradable) <= 60:
        print(f"  {tradable}")
    else:
        print(f"  (first 20: {tradable[:20]} ...)")

    print("\n== 2. Refreshing the bar store (full replace, see data_store.py docstring) ==")
    store = data_store.BarStore()
    bars_by_code: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []
    for i, code in enumerate(codes, 1):
        if i > 1:
            time.sleep(KLINE_MIN_INTERVAL_S)  # stay under Moomoo's 60-calls/30s cap
        try:
            b = store.update(code, service._client, service._lock, lookback_days=lookback_days)
        except Exception as e:  # noqa: BLE001 - e.g. no quote permission for a market/index
            skipped.append((code, str(e)))
            print(f"  [{i}/{len(codes)}] {code}: SKIPPED -- {e}")
            continue
        if b.empty:
            skipped.append((code, "no bars returned"))
            print(f"  [{i}/{len(codes)}] {code}: SKIPPED -- no bars returned")
            continue
        bars_by_code[code] = b
        print(f"  [{i}/{len(codes)}] {code}: {len(b)} bars")

    if BENCHMARK_CODE not in bars_by_code:
        print(f"\nFATAL: benchmark {BENCHMARK_CODE} bars unavailable -- beta/relative-strength "
              f"features would all be NaN and every training row would be dropped. Aborting.")
        return

    # Drop skipped symbols from every downstream step -- crucially, this also
    # prevents build_universe_features from silently loading a STALE parquet
    # snapshot (from an older run) for a symbol we couldn't refresh today.
    codes = [c for c in codes if c in bars_by_code]
    tradable = [c for c in tradable if c in bars_by_code]
    if skipped:
        print(f"  NOTE: skipped {len(skipped)} symbol(s) (no quote permission / no data): "
              + ", ".join(c for c, _ in skipped))
    if len(tradable) < MIN_UNIVERSE_SIZE:
        print(f"  WARNING: universe ({len(tradable)}) is below the {MIN_UNIVERSE_SIZE}-symbol "
              f"gating threshold -- training will proceed for the report, but no model will "
              f"be activated regardless of validation results.")
    store.manifest(codes)

    print("\n== 3. Building point-in-time features ==")
    feat_df = features.build_universe_features(store, codes, bench_code=BENCHMARK_CODE, ppy=ppy)
    print(f"  {len(feat_df)} raw feature rows across {feat_df['code'].nunique() if not feat_df.empty else 0} symbols")

    horizon_reports = []
    for horizon in horizons:
        print(f"\n{'='*70}\nHORIZON = {horizon} bars\n{'='*70}")
        labeled = labels_mod.add_forward_labels(
            feat_df, bars_by_code, horizon, label_mode=args.label_mode,
        )
        print(f"  {len(labeled)} labeled rows after horizon-drop + cross-sectional threshold")
        if args.non_overlapping:
            labeled = labels_mod.subsample_non_overlapping(labeled, horizon)
            print(f"  {len(labeled)} rows after non-overlapping subsample (>= {horizon} bars apart per symbol)")
        if labeled.empty:
            print("  no labeled rows -- skipping this horizon")
            continue
        report = _run_horizon(labeled, bars_by_code, horizon, args.folds, args.min_train_months, ppy, args.recal_sample_cap)
        report["horizon"] = horizon
        report["universe_size"] = len(tradable)
        report["label_mode"] = args.label_mode
        horizon_reports.append(report)

    if not horizon_reports:
        print("\nNo horizon produced any labeled data -- nothing to report or activate.")
        return

    _write_report(horizon_reports, tradable, feat_df, args)
    _maybe_activate(horizon_reports)


def _run_horizon(labeled, bars_by_code, horizon, n_folds, min_train_months, ppy, recal_sample_cap) -> dict:
    feature_cols = features.FEATURE_COLUMNS

    print("\n-- ML model validation (purged walk-forward) --")
    lin_folds = validation.run_walkforward(labeled, feature_cols, ml_models.make_linear_model, horizon, n_folds, min_train_months)
    print(f"  linear:   {len(lin_folds)} folds -- " + _fold_summary(lin_folds))

    lgb_folds = None
    if ml_models.LIGHTGBM_AVAILABLE and not lin_folds.empty:
        lgb_folds = validation.run_walkforward(labeled, feature_cols, ml_models.make_lgb_model, horizon, n_folds, min_train_months)
        print(f"  lightgbm: {len(lgb_folds)} folds -- " + _fold_summary(lgb_folds))

    selected_name, selected_agg = ml_models.select_model(lin_folds, lgb_folds)
    selected_folds = lgb_folds if selected_name == "lightgbm" else lin_folds
    print(f"  -> selected: {selected_name} ({_fmt_agg(selected_agg)})")

    ci = validation.block_bootstrap_ci(selected_folds, "auc") if not selected_folds.empty else None
    print(f"  90% CI on mean OOS AUC (fold-level block bootstrap): {ci}")

    print("\n-- Shuffled-label control (should collapse to ~0.50 AUC) --")
    shuffled = validation.shuffle_labels_within_date(labeled)
    shuf_factory = ml_models.make_linear_model
    shuf_folds = validation.run_walkforward(shuffled, feature_cols, shuf_factory, horizon, n_folds, min_train_months)
    shuf_auc = float(shuf_folds["auc"].mean()) if not shuf_folds.empty and shuf_folds["auc"].notna().any() else None
    print(f"  shuffled-label mean OOS AUC: {_fmt(shuf_auc)} (expected ~0.50)")

    print("\n-- Sub-period breakdown (first half vs second half of test folds) --")
    sub = _sub_period_breakdown(selected_folds)
    print(f"  {sub}")

    reliability = _reliability(selected_agg, len(bars_by_code) - 1)  # -1 excludes the benchmark

    n_folds_achieved = selected_agg.get("n_folds", 0) or 0
    gate_ok = (
        (len(bars_by_code) - 1) >= MIN_UNIVERSE_SIZE
        and n_folds_achieved >= MIN_FOLDS
        and ci is not None and ci[0] > 0.5
    )

    fitted_model = None
    if gate_ok:
        fitted_model = ml_models.make_lgb_model() if selected_name == "lightgbm" else ml_models.make_linear_model()
        train_full = labeled.dropna(subset=[*feature_cols, "label"])
        fitted_model.fit(train_full[feature_cols], train_full["label"])

    print("\n-- Rule-based weight + Kelly slope recalibration (best-effort, separate analysis) --")
    recal = _recalibrate_rule_based(labeled, bars_by_code, horizon, n_folds, min_train_months, ppy, recal_sample_cap)

    return {
        "model_name": selected_name, "agg": selected_agg, "ci_auc": ci,
        "shuffled_auc": shuf_auc, "sub_period": sub, "reliability": reliability,
        "gate_ok": gate_ok, "fitted_model": fitted_model,
        "date_range": _date_range(labeled), "recal": recal,
    }


def _reliability(agg: dict, n_symbols: int) -> float:
    n_folds = agg.get("n_folds") or 0
    auc_std = agg.get("auc_std")
    fold_term = min(n_folds / 8.0, 1.0)
    noise_term = 1.0 - min((auc_std or 0.15) / 0.15, 1.0)
    universe_term = min(n_symbols / 30.0, 1.0)
    return round(max(0.0, min(1.0, fold_term * noise_term * universe_term)), 3)


def _fold_summary(folds: pd.DataFrame) -> str:
    if folds.empty:
        return "no valid folds"
    return f"AUC {_fmt(folds['auc'].mean())}±{_fmt(folds['auc'].std())}, IC {_fmt(folds['ic'].mean())}"


def _fmt_agg(agg: dict) -> str:
    return f"AUC {_fmt(agg.get('auc_mean'))}±{_fmt(agg.get('auc_std'))}, IC {_fmt(agg.get('ic_mean'))}, n_folds={agg.get('n_folds')}"


def _fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def _date_range(df: pd.DataFrame) -> str:
    if df.empty:
        return "n/a"
    return f"{df['date'].min().date()} to {df['date'].max().date()}"


def _sub_period_breakdown(folds: pd.DataFrame) -> dict:
    if folds.empty or len(folds) < 2:
        return {"first_half_auc": None, "second_half_auc": None}
    half = len(folds) // 2
    return {
        "first_half_auc": float(folds.iloc[:half]["auc"].mean()) if folds.iloc[:half]["auc"].notna().any() else None,
        "second_half_auc": float(folds.iloc[half:]["auc"].mean()) if folds.iloc[half:]["auc"].notna().any() else None,
    }


def _recalibrate_rule_based(labeled, bars_by_code, horizon, n_folds, min_train_months, ppy, sample_cap) -> dict:
    """Replays technical.analyze() point-in-time on a (capped) random sample
    of labeled rows to get the 6 existing component scores, then fits
    non-negative weights + a Kelly hit-rate slope from purged walk-forward
    OOS evidence. Best-effort: failures here never block the ML result."""
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return {"ok": False, "reason": "scikit-learn not installed"}

    sample = labeled if len(labeled) <= sample_cap else labeled.sample(sample_cap, random_state=42)
    rows = []
    for _, r in sample.iterrows():
        code, dt = r["code"], r["date"]
        bars = bars_by_code.get(code)
        if bars is None or dt not in bars.index:
            continue
        pos = bars.index.get_loc(dt)
        sliced = bars.iloc[: pos + 1]
        if len(sliced) < technical.MIN_BARS:
            continue
        result = technical.analyze(code, code, sliced, ppy=ppy)
        if result.error:
            continue
        comp_scores = {c.name.lower(): c.score for c in result.components}
        row = {"date": dt, "code": code, "fwd_ret": r["fwd_ret"], "score": result.score}
        row.update({k: comp_scores.get(k, 0.0) for k in COMPONENT_KEYS})
        rows.append(row)

    recal_df = pd.DataFrame(rows)
    if recal_df.empty:
        return {"ok": False, "reason": "no rows survived the historical replay"}

    splits = validation.purged_walkforward_splits(recal_df["date"], horizon, n_folds, min_train_months)
    fold_r2, fold_ic = [], []
    for train_dates, test_dates in splits:
        train = recal_df[recal_df["date"].isin(train_dates)]
        test = recal_df[recal_df["date"].isin(test_dates)]
        if len(train) < 10 or len(test) < 5:
            continue
        lr = LinearRegression(positive=True, fit_intercept=False)
        lr.fit(train[COMPONENT_KEYS], train["fwd_ret"])
        pred = lr.predict(test[COMPONENT_KEYS])
        ss_res = float(((test["fwd_ret"] - pred) ** 2).sum())
        ss_tot = float(((test["fwd_ret"] - test["fwd_ret"].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
        if r2 is not None:
            fold_r2.append(r2)
        from scipy.stats import spearmanr
        ic, _p = spearmanr(pred, test["fwd_ret"])
        if not pd.isna(ic):
            fold_ic.append(float(ic))

    trustworthy = bool(fold_r2) and (sum(1 for x in fold_r2 if x > 0) / len(fold_r2)) > 0.5

    suggested_weights = None
    if trustworthy:
        lr_full = LinearRegression(positive=True, fit_intercept=False)
        lr_full.fit(recal_df[COMPONENT_KEYS], recal_df["fwd_ret"])
        coefs = np.clip(lr_full.coef_, 0.0, None)
        total = coefs.sum()
        if total > 0:
            suggested_weights = {k: round(float(c / total), 3) for k, c in zip(COMPONENT_KEYS, coefs)}

    # Kelly slope: bucket by score decile, fit hit-rate ~ (score-50)/100, OOS test rows only
    kelly_slope = None
    oos_rows = []
    for train_dates, test_dates in splits:
        oos_rows.append(recal_df[recal_df["date"].isin(test_dates)])
    if oos_rows:
        oos = pd.concat(oos_rows, ignore_index=True)
        if len(oos) >= 20:
            oos = oos.copy()
            oos["decile"] = pd.qcut(oos["score"], q=min(10, oos["score"].nunique()), duplicates="drop")
            bucket = oos.groupby("decile", observed=True).agg(
                hit_rate=("fwd_ret", lambda s: float((s > 0).mean())),
                score_mid=("score", "mean"),
            ).dropna()
            if len(bucket) >= 3:
                x = (bucket["score_mid"] - 50.0) / 100.0
                y = bucket["hit_rate"] - 0.5  # centered, so slope is comparable to the current hardcoded 0.7
                denom = float((x * x).sum())
                if denom > 0:
                    kelly_slope = round(float((x * y).sum() / denom), 3)

    return {
        "ok": True, "trustworthy": trustworthy,
        "fold_r2_mean": float(np.mean(fold_r2)) if fold_r2 else None,
        "fold_ic_mean": float(np.mean(fold_ic)) if fold_ic else None,
        "n_folds": len(fold_r2), "n_sampled_rows": len(recal_df),
        "suggested_weights": suggested_weights, "suggested_kelly_slope": kelly_slope,
        "current_weights": dict(technical.WEIGHTS), "current_kelly_slope": technical.KELLY_SLOPE,
    }


def _write_report(horizon_reports: list[dict], universe_codes: list[str], feat_df: pd.DataFrame, args) -> None:
    d = data_store.store_dir() / "reports"
    d.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    path = d / f"calibration_{stamp}.md"

    uni_str = ", ".join(universe_codes) if len(universe_codes) <= 60 else \
        f"{', '.join(universe_codes[:60])}, ... (+{len(universe_codes) - 60} more)"
    survivorship = (
        "- **Survivorship bias**: trained only on symbols the user already chose to hold/watch --\n"
        "  this validates existing judgment, it does not generalize to \"what to buy next.\""
        if args.universe == "holdings" else
        "- **Survivorship bias (reduced)**: a broad list is used, but it's a CURRENT index\n"
        "  snapshot -- names dropped/delisted are absent, so some bias remains (point-in-time\n"
        "  membership would need a paid data source)."
    )
    lines = [
        f"# ML Calibration Report -- {stamp}",
        "",
        f"Universe source: **{args.universe}** ({len(universe_codes)} symbols): {uni_str}",
        f"Timeframe: {args.tf}  |  Folds requested: {args.folds}  |  Min train months: {args.min_train_months}",
        f"Label target: {args.label_mode}  |  Non-overlapping sampling: {args.non_overlapping}",
        "",
        "## Limitations (read before trusting anything below)",
        survivorship,
        "- **Regime narrowness**: the whole date range is one market regime (see date ranges below);",
        "  any apparent edge may be specific to that period, not a durable effect.",
        "- **Small effective sample size**: overlapping labels and cross-sectional correlation mean",
        "  the true independent sample size is far smaller than the row count suggests.",
        "- **QFQ/dividend adjustment**: forward returns near ex-dividend dates for high-yield names",
        "  may be quietly distorted by Moomoo's adjustment convention.",
        "",
    ]
    for r in horizon_reports:
        h = r["horizon"]
        lines += [
            f"## Horizon = {h} bars",
            f"Date range: {r['date_range']}",
            f"Selected model: **{r['model_name']}**  ({_fmt_agg(r['agg'])})",
            f"90% CI on mean OOS AUC: {r['ci_auc']}",
            f"Shuffled-label control (expect ~0.50): {_fmt(r['shuffled_auc'])}",
            f"Sub-period breakdown: {r['sub_period']}",
            f"Reliability score: {r['reliability']}",
            f"**Gating: {'PASSED -- eligible for activation' if r['gate_ok'] else 'NOT MET -- will not activate'}**",
            "",
            "### Rule-based weight / Kelly-slope recalibration",
        ]
        recal = r["recal"]
        if not recal.get("ok"):
            lines.append(f"Skipped: {recal.get('reason')}")
        else:
            lines += [
                f"Sampled rows: {recal['n_sampled_rows']}, OOS folds: {recal['n_folds']}",
                f"OOS R² mean: {_fmt(recal['fold_r2_mean'])}, OOS IC mean: {_fmt(recal['fold_ic_mean'])}",
                f"Trustworthy (majority of folds R² > 0): **{recal['trustworthy']}**",
                "",
                f"Current WEIGHTS: `{recal['current_weights']}`",
            ]
            if recal["suggested_weights"]:
                lines.append(f"Suggested WEIGHTS (paste into technical.py after review):\n```python\nWEIGHTS = {recal['suggested_weights']}\n```")
            else:
                lines.append("Suggested WEIGHTS: none (recalibration not trustworthy enough -- keep current)")
            lines.append(f"Current KELLY_SLOPE: {recal['current_kelly_slope']}")
            lines.append(f"Suggested KELLY_SLOPE: {recal['suggested_kelly_slope']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written: {path}")


def _maybe_activate(horizon_reports: list[dict]) -> None:
    gated = [r for r in horizon_reports if r["gate_ok"] and r["fitted_model"] is not None]
    if not gated:
        print("\nNo horizon met the activation gating thresholds -- current.json left unchanged.")
        return
    best = max(gated, key=lambda r: r["agg"].get("auc_mean") or 0.0)

    models_dir = data_store.store_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    model_file = f"ml_model_{stamp}.joblib"
    joblib.dump(best["fitted_model"], models_dir / model_file)

    meta = {
        "model_file": model_file,
        "feature_cols": features.FEATURE_COLUMNS,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": best["model_name"],
        "horizon": best["horizon"],
        "label_mode": best.get("label_mode", "median"),
        "universe_size": best["universe_size"],
        "date_range": best["date_range"],
        "n_folds": best["agg"].get("n_folds"),
        "oos_auc": best["agg"].get("auc_mean"),
        "oos_auc_std": best["agg"].get("auc_std"),
        "oos_ic": best["agg"].get("ic_mean"),
        "reliability": best["reliability"],
    }
    (models_dir / "current.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"\nActivated model: {model_file} (horizon={best['horizon']}, {_fmt_agg(best['agg'])}, "
          f"reliability={best['reliability']}). Restart the backend to pick it up.")


if __name__ == "__main__":
    main()
    # The moomoo SDK and the IBKR asyncio loop leave non-daemon threads running;
    # without a hard exit the interpreter hangs forever in threading._shutdown
    # after main() returns (observed: "stuck" runs that had actually finished
    # minutes earlier). Everything is already written to disk by this point.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
