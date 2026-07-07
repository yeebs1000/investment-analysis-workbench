"""Candidate model factories + the pre-registered model-selection rule.

Lazy-imported (same pattern as `app/llm/router.py`'s LLM providers) so the
rest of the app runs fine without scikit-learn/lightgbm installed -- only
`app/ml/train.py` (offline) and `app/ml/inference.py` (guarded) touch these.
"""
from __future__ import annotations

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    LIGHTGBM_AVAILABLE = False

# Pre-registered comparison rule (decided before looking at any results, so we
# can't rationalize picking the fancier model after the fact): LightGBM must
# beat the linear baseline by this margin, in this fraction of folds, or the
# linear model wins by default.
LGB_MIN_AUC_EDGE = 0.02
LGB_MIN_IC_EDGE = 0.02
LGB_MIN_WIN_FRACTION = 0.60


def make_linear_model():
    """L1-regularized ('Lasso') logistic regression -- the default model.
    Small C = strong regularization, appropriate for a modest sample size.

    Uses solver='saga' + l1_ratio=1.0 (pure L1) rather than the older
    penalty='l1'/liblinear form: `penalty` was deprecated in scikit-learn 1.8
    and is removed in 1.10, so the old form would break a fresh install. saga
    needs scaled inputs (the StandardScaler ahead of it) and a higher iter cap
    to converge on this feature count."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required: pip install scikit-learn")
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(l1_ratio=1.0, solver="saga", C=0.3,
                                   class_weight="balanced", max_iter=2000)),
    ])


def make_lgb_model():
    """Shallow, heavily-regularized gradient boosting -- only shipped if it
    clearly beats the linear baseline OOS (see `select_model`)."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("lightgbm is required: pip install lightgbm")
    return lgb.LGBMClassifier(
        n_estimators=200, max_depth=3, num_leaves=7, learning_rate=0.03,
        min_child_samples=30, reg_alpha=1.0, reg_lambda=1.0,
        subsample=0.7, colsample_bytree=0.7, verbose=-1,
    )


def select_model(linear_folds, lgb_folds) -> tuple[str, "object"]:
    """Apply the pre-registered comparison rule. `linear_folds`/`lgb_folds` are
    per-fold metrics DataFrames from `validation.run_walkforward` (lgb_folds
    may be None if LightGBM wasn't run or produced no valid folds).
    Returns (name, aggregate_metrics_dict)."""
    lin_agg = _aggregate(linear_folds)
    if lgb_folds is None or lgb_folds.empty:
        return "linear", lin_agg

    lgb_agg = _aggregate(lgb_folds)
    joined = linear_folds.merge(lgb_folds, on="fold", suffixes=("_lin", "_lgb"))
    if joined.empty:
        return "linear", lin_agg

    auc_wins = ((joined["auc_lgb"] - joined["auc_lin"]) >= LGB_MIN_AUC_EDGE).mean()
    ic_wins = ((joined["ic_lgb"] - joined["ic_lin"]) >= LGB_MIN_IC_EDGE).mean()
    if auc_wins >= LGB_MIN_WIN_FRACTION and ic_wins >= LGB_MIN_WIN_FRACTION:
        return "lightgbm", lgb_agg
    return "linear", lin_agg


def _aggregate(folds) -> dict:
    if folds is None or folds.empty:
        return {"auc_mean": None, "auc_std": None, "ic_mean": None, "ic_std": None,
                "accuracy_mean": None, "n_folds": 0}
    return {
        "auc_mean": float(folds["auc"].mean()) if folds["auc"].notna().any() else None,
        "auc_std": float(folds["auc"].std()) if folds["auc"].notna().sum() > 1 else None,
        "ic_mean": float(folds["ic"].mean()) if folds["ic"].notna().any() else None,
        "ic_std": float(folds["ic"].std()) if folds["ic"].notna().sum() > 1 else None,
        "accuracy_mean": float(folds["accuracy"].mean()) if folds["accuracy"].notna().any() else None,
        "n_folds": int(len(folds)),
    }
