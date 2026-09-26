import json
import sys
import time

from sklearn.metrics import roc_auc_score

from compas_scoring.config import CONFIG, FEATURE_SET_LABELS
from compas_scoring.data import train_test
from compas_scoring.models import (
    MODEL_NAMES,
    TUNED_MODELS,
    FittedModel,
    build_model,
    model_path,
    save,
    tune,
)
from compas_scoring.stability import repeated_cv

FEATURE_SETS = tuple(CONFIG.feature_sets)
model_log_only = ("logistic")

def fit_one(name: str, feature_set: str, train, force: bool = False) -> FittedModel | None:
    """Fit one model. Returns None when a cached fit is reused."""
    key = f"{name}__{feature_set}"
    if not force and model_path(key).exists():
        print(f"  {name:13s} cached, skipping")
        return None

    start = time.perf_counter()
    best_params = None

    if name in TUNED_MODELS:
        search = tune(name, train.X, train.y, random_state=CONFIG.random_state)
        estimator = search.best_estimator_
        best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        cv_auc, cv_sd = float(search.best_score_), float("nan")
    else:
        estimator = build_model(name, random_state=CONFIG.random_state)
        estimator.fit(train.X, train.y)
        folds = repeated_cv(estimator, train.X, train.y, n_repeats=1)
        cv_auc, cv_sd = float(folds["auc"].mean()), float(folds["auc"].std(ddof=1))

    elapsed = time.perf_counter() - start
    fitted = FittedModel(
        name=name,
        feature_set=feature_set,
        estimator=estimator,
        features=list(train.X.columns),
        fit_seconds=elapsed,
        best_params=best_params,
        cv_auc=cv_auc,
    )

    train_auc = roc_auc_score(train.y, fitted.predict_proba(train.X))
    gap = train_auc - cv_auc
    flag = "  <-- overfit?" if gap > 0.05 else ""
    spread = "" if cv_sd != cv_sd else f" +- {cv_sd:.4f}"
    print(
        f"  {name:13s} fit {elapsed:7.1f}s   train AUC {train_auc:.4f}   "
        f"CV AUC {cv_auc:.4f}{spread}   gap {gap:+.4f}{flag}"
    )
    if best_params:
        print(f"                tuned: {json.dumps(best_params, sort_keys=True)}")

    save(fitted)
    return fitted


def main() -> int:
    force = "--force" in sys.argv
    summary: dict[str, dict[str, float]] = {}

    for feature_set in FEATURE_SETS:
        train, _ = train_test(feature_set)
        label = FEATURE_SET_LABELS.get(feature_set, feature_set)
        print(f"\n{label} {feature_set}  ({train.X.shape[1]} features, {len(train):,} train rows)")
        print("-" * 82)

        name = model_log_only[0]
        fitted = fit_one(name, feature_set, train, force=force)
        if fitted is not None:
            summary[fitted.key] = {
                "cv_auc": fitted.cv_auc,
                "fit_seconds": fitted.fit_seconds,
            }

    out = CONFIG.path("artifacts", "fit_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():  # merge, so a resumed run does not lose earlier entries
        summary = {**json.loads(out.read_text()), **summary}
    out.write_text(json.dumps(summary, indent=2))

    expected = len(FEATURE_SETS) * len(model_log_only)
    present = len(list(CONFIG.path("models").glob("*.joblib")))
    print(f"\nmodels/ holds {present} of {expected} expected fits; summary -> {out.name}")
    return 0 if present == expected else 1