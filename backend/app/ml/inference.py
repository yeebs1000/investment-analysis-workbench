"""Request-time ML inference -- the ONLY file in app/ml touched by the live
request path (via `AnalysisService.analyze_symbol`).

Must never raise into a caller. Any failure -- missing package, missing or
corrupt model artifact, incomplete feature row, whatever -- returns None, and
`technical.analyze(ml_signal=None)` already degrades to exactly the pre-ML
6-component behavior. No hot-reload: a retrain requires an app restart to
pick up the new model, matching the "human reviews, then deploys" philosophy
(see app/ml/train.py) -- so the model is loaded and cached once per process.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.data.models import MLSignal
from app.ml import data_store, features

_bundle_cache: dict | None = None
_cache_loaded = False


def _current_json_path() -> Path:
    return data_store.store_dir() / "models" / "current.json"


def load_active_model() -> dict | None:
    global _bundle_cache, _cache_loaded
    if _cache_loaded:
        return _bundle_cache
    _cache_loaded = True
    try:
        meta_path = _current_json_path()
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        import joblib  # lazy -- only touched if a model was actually trained+activated
        model_path = data_store.store_dir() / "models" / meta["model_file"]
        model = joblib.load(model_path)
        _bundle_cache = {**meta, "model": model}
    except Exception:  # noqa: BLE001 - any failure -> no ML signal, ever
        _bundle_cache = None
    return _bundle_cache


def score_symbol(
    code: str, bars: pd.DataFrame, bench_bars: pd.DataFrame | None, ppy: float,
) -> MLSignal | None:
    bundle = load_active_model()
    if bundle is None:
        return None
    try:
        # last_row_only: inference only reads the final row -- skip computing
        # the hundreds of historical rows the training path needs.
        feat = features.build_feature_frame(code, bars, bench_bars, ppy, last_row_only=True)
        if feat.empty:
            return None
        cols = bundle["feature_cols"]
        row = feat.iloc[[-1]]
        if row[cols].isna().any(axis=1).iloc[0]:
            return None  # incomplete feature row (e.g. too little history) -- don't guess
        proba = float(bundle["model"].predict_proba(row[cols])[0, 1])
        score = 2.0 * proba - 1.0
        reliability = float(bundle.get("reliability", 0.0))
        weak = " -- weak/tentative edge" if reliability < 0.5 else ""
        reason = (
            f"ML forecast: {proba*100:.0f}% prob. of beating the peer median forward return "
            f"(trained on {bundle.get('universe_size', '?')} symbols, {bundle.get('date_range', '?')}, "
            f"OOS AUC {_fmt(bundle.get('oos_auc'))}±{_fmt(bundle.get('oos_auc_std'))} "
            f"across {bundle.get('n_folds', '?')} folds{weak})."
        )
        return MLSignal(score=score, probability=proba, reliability=reliability, reasons=[reason])
    except Exception:  # noqa: BLE001 - ML is a bonus, never fatal to the request
        return None


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"
