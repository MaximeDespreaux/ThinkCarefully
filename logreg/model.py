"""The logistic regression pipeline: engineered terms, standardisation, L2 logistic."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from compas_scoring.config import CONFIG
from logreg.features import engineered


def build_logistic(random_state: int = CONFIG.random_state) -> Pipeline:
    return Pipeline(
        [
            ("engineer", FunctionTransformer(engineered, validate=False)),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


@dataclass
class FittedModel:
    """A fitted estimator plus the provenance needed to reproduce it."""

    name: str
    feature_set: str
    estimator: BaseEstimator
    features: list[str]
    fit_seconds: float
    best_params: dict | None = None
    # Train-only cross-validated AUC. Recorded at fit time so model quality can be
    # discussed without reading the holdout.
    cv_auc: float | None = None

    @property
    def key(self) -> str:
        return f"{self.name}__{self.feature_set}"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probability, with the feature contract enforced."""
        if list(X.columns) != self.features:
            raise ValueError(f"{self.key} expects {self.features}, received {list(X.columns)}")
        return self.estimator.predict_proba(X)[:, 1]


def model_path(key: str):
    return CONFIG.path("models", f"{key}.joblib")


def save(fitted: FittedModel) -> None:
    path = model_path(fitted.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, path)


def load(name: str, feature_set: str = "race_aware") -> FittedModel:
    path = model_path(f"{name}__{feature_set}")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `make models` first")
    return joblib.load(path)
