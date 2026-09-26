"""The logistic regression pipeline: engineered terms, standardisation, L2 logistic."""

from __future__ import annotations

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
