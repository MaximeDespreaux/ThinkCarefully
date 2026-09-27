"""The three models, fitted the same way on any shared split, for the cross-model comparison.

Each model keeps its own group's definition:
    LogReg   logreg.model.build_logistic (engineered priors terms, L2, C = 1)
    XGBoost  the tuned hyperparameters saved by xgboost/xgb_model.py, held fixed across refits
             (the race-aware set's for every variant without its own tuned model)
    TabPFN   tabpfn/pfn_model.build_model; its committed predictions are reused wherever they
             exist, since a TabPFN fit costs minutes rather than milliseconds
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from compas_scoring.config import CONFIG
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
# The repo root (for the logreg package) and tabpfn/, a folder of scripts imported by name.
for path in (ROOT, ROOT / "tabpfn"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from logreg.model import build_logistic  # noqa: E402

MODELS = ("LogReg", "XGBoost", "TabPFN")
TABPFN_PREDICTIONS = ROOT / "tabpfn" / "artifacts" / "predictions"


def xgb_params(feature_set: str) -> dict:
    """The saved tuned hyperparameters: the set's own if it was tuned, else the race-aware ones."""
    path = ROOT / "xgboost" / "models" / f"xgb_{feature_set}.meta.json"
    if not path.exists():
        path = ROOT / "xgboost" / "models" / "xgb_race_aware.meta.json"
    return json.loads(path.read_text())["best_params"]


def build(model: str, feature_set: str = "race_aware"):
    """A fresh, unfitted estimator."""
    if model == "LogReg":
        return build_logistic()
    if model == "XGBoost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=CONFIG.random_state,
            n_jobs=1,
            **xgb_params(feature_set),
        )
    if model == "TabPFN":
        from pfn_model import build_model

        return build_model()
    raise ValueError(f"unknown model {model!r}")


def saved_xgboost(feature_set: str):
    """The XGBoost group's saved model for this feature set, or None if they did not save one."""
    path = ROOT / "xgboost" / "models" / f"xgb_{feature_set}.json"
    if not path.exists():
        return None
    model = XGBClassifier()
    model.load_model(path)
    return model


def fit(model: str, X: pd.DataFrame, y: pd.Series, feature_set: str = "race_aware"):
    return build(model, feature_set).fit(X, y)


def score(estimator, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(estimator.predict_proba(X))[:, 1]


def tabpfn_scores(run: str, feature_set: str) -> pd.Series:
    """TabPFN's committed test-set scores for one run, indexed by cohort row."""
    frame = pd.read_csv(TABPFN_PREDICTIONS / f"{run}__{feature_set}.csv", index_col="row")
    return frame["tabpfn"]
