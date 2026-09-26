from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from compas_scoring.config import CONFIG
from compas_scoring.models import BUILDERS
from logreg.features import engineered


def build_model(name: str, random_state: int = CONFIG.random_state) -> BaseEstimator:
    try:
        return BUILDERS[name](random_state)
    except KeyError:
        raise KeyError(f"Unknown model {name!r}; available: {list(BUILDERS)}") from None

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
