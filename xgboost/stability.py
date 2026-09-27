"""Stability of the XGBoost architecture under resampling and temporal splits.

Each comparison below trains two fresh models (same tuned hyperparameters as the saved
model, read from its ``.meta.json``, so only the training data varies) and compares their
predicted probabilities via the L2 norm. Every function returns data; plotting is left to
the notebook.

Comparisons:
    1. Random split: two models each trained on their own random, non-overlapping 40% of
       the cohort, both evaluated on a common held-out 20% -- so the L2 norm is a genuine
       row-matched comparison (same test rows, same order).
    2. Temporal split: model A trained on the first 40% (by time), tested on the next 20%
       (rows 40-60%); model B trained on the first 80%, tested on the last 20% (rows
       80-100%). The two test windows are the same SIZE but cover DIFFERENT rows, so the
       L2 norm here compares two same-length prediction arrays position-by-position -- it
       reflects how similarly-shaped the model's predicted-probability distribution is
       across two points in time, not a per-individual matched comparison.

Note: the modelling table has no explicit timestamp column, so "temporal" order falls back
to row index as a stand-in for chronology. Pass a real date/id column via ``time_column``
if one becomes available.

Run from the repository root:  ``python xgboost/stability.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from compas_scoring.config import CONFIG
from compas_scoring.data import Dataset, build_dataset
from xgb_model import load_metadata


def _fit_xgb(train: Dataset, feature_set: str = "race_aware") -> XGBClassifier:
    """A fresh XGBClassifier with the saved model's tuned hyperparameters, fit on ``train``.

    Hyperparameters are held fixed (read from the original run's metadata) so that only
    the training data varies between models -- otherwise a difference in predictions could
    come from re-tuning landing on different hyperparameters, not from the data split.
    """
    params = load_metadata(feature_set)["best_params"]
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=CONFIG.random_state,
        n_jobs=1,
        **params,
    )
    model.fit(train.X, train.y)
    return model


def predict_proba_positive(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """P(recidivism) for every row in X."""
    return model.predict_proba(X)[:, 1]


def l2_norm(p1: np.ndarray, p2: np.ndarray) -> float:
    """L2 norm between two same-length prediction arrays, position-by-position."""
    p1, p2 = np.asarray(p1), np.asarray(p2)
    if len(p1) != len(p2):
        raise ValueError(f"Prediction arrays differ in length: {len(p1)} vs {len(p2)}")
    return float(np.linalg.norm(p1 - p2))


def model_split(
    feature_set: str = "race_aware", seed: int = 42
) -> tuple[XGBClassifier, XGBClassifier, Dataset, Dataset, Dataset]:
    """Comparison 1: two models on their own random 40%, evaluated on a common 20% test set.

    Returns ``(model_a, model_b, train_a, train_b, test)``.
    """
    full = build_dataset(feature_set)
    idx_a, rest_idx = train_test_split(
        full.X.index, train_size=0.4, stratify=full.y, random_state=seed
    )
    rest = full.subset(rest_idx)
    # Of the remaining 60%, 2/3 -> another 40% of the full cohort (train_b), 1/3 -> 20% (test).
    idx_b, idx_test = train_test_split(
        rest_idx, train_size=2 / 3, stratify=rest.y, random_state=seed
    )

    train_a, train_b, test = full.subset(idx_a), full.subset(idx_b), full.subset(idx_test)
    model_a = _fit_xgb(train_a, feature_set)
    model_b = _fit_xgb(train_b, feature_set)
    return model_a, model_b, train_a, train_b, test


def model_split2(
    feature_set: str = "race_aware", time_column: str | None = None
) -> tuple[XGBClassifier, XGBClassifier, Dataset, Dataset, Dataset, Dataset]:
    """Comparison 2: temporal split, two differently-sized training windows.

    Model A: trained on the first 40% (by time), tested on the next 20% (rows 40-60%).
    Model B: trained on the first 80%, tested on the last 20% (rows 80-100%).
    The two test sets are trimmed to the same length (they can differ by a row or two from
    rounding) so the L2 norm below is well-defined.

    Returns ``(model_a, model_b, train_a, test_a, train_b, test_b)``.
    """
    full = build_dataset(feature_set)
    order = full.X.index if time_column is None else full.groups[time_column].sort_values().index
    n = len(order)

    train_a_idx = order[: int(0.4 * n)]
    test_a_idx = order[int(0.4 * n) : int(0.6 * n)]
    train_b_idx = order[: int(0.8 * n)]
    test_b_idx = order[int(0.8 * n) :]

    min_len = min(len(test_a_idx), len(test_b_idx))
    test_a_idx, test_b_idx = test_a_idx[:min_len], test_b_idx[:min_len]

    train_a, test_a = full.subset(train_a_idx), full.subset(test_a_idx)
    train_b, test_b = full.subset(train_b_idx), full.subset(test_b_idx)

    model_a = _fit_xgb(train_a, feature_set)
    model_b = _fit_xgb(train_b, feature_set)
    return model_a, model_b, train_a, test_a, train_b, test_b


def model_metrics(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
) -> dict[str, float]:
    """Accuracy, F1, precision, recall, and ROC AUC of ``model`` on (X_test, y_test)."""
    y_score = predict_proba_positive(model, X_test)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


def main(feature_set: str = "race_aware") -> None:
    print("Comparison 1: random 40/40, common 20% test set")
    model_a, model_b, train_a, train_b, test = model_split(feature_set)
    preds_a = predict_proba_positive(model_a, test.X)
    preds_b = predict_proba_positive(model_b, test.X)
    print(f"  L2 norm: {l2_norm(preds_a, preds_b):.4f}")

    print("\nComparison 2: temporal, 40%->next 20% vs 80%->last 20%")
    model_a2, model_b2, train_a2, test_a2, train_b2, test_b2 = model_split2(feature_set)
    preds_a2 = predict_proba_positive(model_a2, test_a2.X)
    preds_b2 = predict_proba_positive(model_b2, test_b2.X)
    print(f"  L2 norm: {l2_norm(preds_a2, preds_b2):.4f}")

    print("\nModel metrics, comparison 1, model_a on its test set:")
    print(model_metrics(model_a, test.X, test.y))


if __name__ == "__main__":
    main()