"""XGBoost on the COMPAS two-year recidivism cohort: tuning, training and saving.

Uses the shared train/test split from ``compas_scoring.data`` so the numbers are comparable
with every other model in the project. Hyperparameters are chosen by a small randomized
search with stratified 5-fold CV on the training set only. The fitted model is saved to
``xgboost/models/`` so each evaluation dimension (performance, interpretability, stability,
fairness) scores the same model without refitting.

Run from the repository root:  ``python xgboost/xgb_model.py``
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from compas_scoring.config import CONFIG
from compas_scoring.data import Dataset, train_test

MODEL_DIR = Path(__file__).resolve().parent / "models"

# Shallow trees and strong regularization: with ~4,300 training rows and 10 low-cardinality
# features, XGBoost overfits quickly at its default depth.
SEARCH_SPACE = {
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 300, 500],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 5, 10, 20],
    "reg_lambda": [0.5, 1.0, 5.0, 10.0],
    "gamma": [0.0, 0.1, 0.5],
}


def load_data(feature_set: str = "race_aware") -> tuple[Dataset, Dataset]:
    """The project's shared stratified train/test split for one feature set."""
    return train_test(feature_set)


def tune_xgboost(X, y, n_iter: int = 40) -> RandomizedSearchCV:
    """Randomized search over SEARCH_SPACE, scored by cross-validated AUC on the training set."""
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=CONFIG.random_state,
        n_jobs=1,
    )
    cv = StratifiedKFold(
        n_splits=CONFIG.iterations.cv_folds, shuffle=True, random_state=CONFIG.random_state
    )
    search = RandomizedSearchCV(
        model,
        SEARCH_SPACE,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=CONFIG.random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search


def model_paths(feature_set: str = "race_aware") -> tuple[Path, Path]:
    """Where the booster and its metadata live for one feature set."""
    stem = MODEL_DIR / f"xgb_{feature_set}"
    return stem.with_suffix(".json"), stem.with_suffix(".meta.json")


def save_model(search: RandomizedSearchCV, feature_set: str = "race_aware") -> Path:
    """Save the best estimator in XGBoost's native format, plus how it was obtained."""
    model_path, meta_path = model_paths(feature_set)
    MODEL_DIR.mkdir(exist_ok=True)
    search.best_estimator_.save_model(model_path)
    meta = {
        "feature_set": feature_set,
        "features": CONFIG.features(feature_set),
        "best_params": search.best_params_,
        "cv_auc": search.best_score_,
        "cv_folds": CONFIG.iterations.cv_folds,
        "test_size": CONFIG.test_size,
        "random_state": CONFIG.random_state,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return model_path


def load_model(feature_set: str = "race_aware") -> XGBClassifier:
    """The saved, fitted model for one feature set. Run this module first to create it."""
    model_path, _ = model_paths(feature_set)
    if not model_path.is_file():
        raise FileNotFoundError(f"{model_path} not found - run `python xgboost/xgb_model.py` first")
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def load_metadata(feature_set: str = "race_aware") -> dict:
    _, meta_path = model_paths(feature_set)
    return json.loads(meta_path.read_text())


def main(feature_set: str = "race_aware") -> None:
    train, test = load_data(feature_set)
    print(f"Feature set {feature_set}: {len(train)} train rows, {len(test)} test rows")

    search = tune_xgboost(train.X, train.y)
    print(f"Best CV AUC: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")

    path = save_model(search, feature_set)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    main()
