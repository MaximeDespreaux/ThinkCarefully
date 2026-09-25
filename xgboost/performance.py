"""Predictive performance of the saved XGBoost model on the shared test set.

Every function returns data (a DataFrame or arrays); plotting is left to the notebook.
The model is loaded from ``xgboost/models/``, so run ``xgb_model.py`` first.

Run from the repository root:  ``python xgboost/performance.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from compas_scoring.data import Dataset
from xgb_model import load_data, load_model


def scores(model, data: Dataset) -> np.ndarray:
    """Predicted probability of two-year recidivism."""
    return model.predict_proba(data.X)[:, 1]


def performance(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    """Performance metrics for predicted probabilities; y_score may also be a 0/1 label."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_score),
    }


def performance_table(model, test: Dataset, threshold: float = 0.5) -> pd.DataFrame:
    """XGBoost next to the incumbent COMPAS score (score_factor) on the same test rows.

    COMPAS only gives a high/low label here, so its AUC is that of a single cut-off and its
    Brier score is that of a 0/1 prediction -- read them as a reference point, not a rival
    probability model.
    """
    rows = {
        "XGBoost": performance(test.y, scores(model, test), threshold),
        "COMPAS (incumbent)": performance(test.y, test.incumbent, threshold),
    }
    return pd.DataFrame(rows).T.round(4)


def confusion(model, test: Dataset, threshold: float = 0.5) -> pd.DataFrame:
    y_pred = (scores(model, test) >= threshold).astype(int)
    return pd.DataFrame(
        confusion_matrix(test.y, y_pred),
        index=["actual: no reoffence", "actual: reoffence"],
        columns=["predicted: no", "predicted: yes"],
    )


def roc_data(y_true, y_score) -> pd.DataFrame:
    """ROC curve points; works for the model's probabilities and for the COMPAS label."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def main(feature_set: str = "race_aware") -> None:
    _, test = load_data(feature_set)
    model = load_model(feature_set)

    print(f"Test-set performance, feature set {feature_set} (threshold 0.5):")
    print(performance_table(model, test))
    print("\nConfusion matrix:")
    print(confusion(model, test))


if __name__ == "__main__":
    main()
