"""Fit the logistic regression on every feature set.
Run via `uv run python -m logreg.train`.
"""

from __future__ import annotations

import json
import sys
import time
import warnings

from sklearn.metrics import roc_auc_score

from compas_scoring.config import CONFIG, FEATURE_SET_LABELS
from compas_scoring.data import train_test
from compas_scoring.models import FittedModel, model_path, save
from compas_scoring.stability import repeated_cv
from logreg.model import build_logistic

NAME = "logistic"
FEATURE_SETS = tuple(CONFIG.feature_sets)


def fit_one(feature_set: str, train, force: bool = False) -> FittedModel | None:
    """Fit on one feature set. Returns None when a cached fit is reused."""
    if not force and model_path(f"{NAME}__{feature_set}").exists():
        print(f"  {NAME} cached, skipping")
        return None

    start = time.perf_counter()
    estimator = build_logistic(random_state=CONFIG.random_state)
    estimator.fit(train.X, train.y)
    folds = repeated_cv(estimator, train.X, train.y, n_repeats=1)
    cv_auc, cv_sd = float(folds["auc"].mean()), float(folds["auc"].std(ddof=1))
    elapsed = time.perf_counter() - start

    fitted = FittedModel(
        name=NAME,
        feature_set=feature_set,
        estimator=estimator,
        features=list(train.X.columns),
        fit_seconds=elapsed,
        cv_auc=cv_auc,
    )

    train_auc = roc_auc_score(train.y, fitted.predict_proba(train.X))
    gap = train_auc - cv_auc
    flag = "  <-- overfit?" if gap > 0.05 else ""
    print(
        f"  {NAME} fit {elapsed:5.1f}s   train AUC {train_auc:.4f}   "
        f"CV AUC {cv_auc:.4f} +- {cv_sd:.4f}   gap {gap:+.4f}{flag}"
    )

    save(fitted)
    return fitted


def main() -> int:
    force = "--force" in sys.argv
    summary: dict[str, dict[str, float]] = {}

    for feature_set in FEATURE_SETS:
        train, _ = train_test(feature_set)
        label = FEATURE_SET_LABELS.get(feature_set, feature_set)
        print(f"\n{label} {feature_set}  ({train.X.shape[1]} features, {len(train):,} train rows)")
        fitted = fit_one(feature_set, train, force=force)
        if fitted is not None:
            summary[fitted.key] = {"cv_auc": fitted.cv_auc, "fit_seconds": fitted.fit_seconds}

    out = CONFIG.path("artifacts", "fit_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():  # merge, so teammates' entries are kept
        summary = {**json.loads(out.read_text()), **summary}
    out.write_text(json.dumps(summary, indent=2))

    present = len(list(CONFIG.path("models").glob(f"{NAME}__*.joblib")))
    print(f"\nmodels/ holds {present} of {len(FEATURE_SETS)} logistic fits")
    return 0 if present == len(FEATURE_SETS) else 1


if __name__ == "__main__":
    sys.exit(main())
