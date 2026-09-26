"""How much race a feature set leaks, even with the race columns removed.

The EDA shows the pairwise links (association matrix, figure 05; priors by race, figure 09).
What it cannot show is the joint effect: race may be predictable from several features
together. This measures it as the 5-fold cross-validated AUC of a logistic model separating
African-American from Caucasian defendants (the two groups every fairness comparison here
uses), fitted on the training split. 0.5 means the feature set carries no race information.
"""

from __future__ import annotations

import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from compas_scoring.config import CONFIG
from compas_scoring.data import build_dataset, split_index


def race_leakage(feature_set: str) -> float:
    """CV AUC of predicting African-American vs Caucasian from a feature set's non-race columns."""
    train_idx, _ = split_index()
    data = build_dataset(feature_set).subset(train_idx)
    race = data.groups["race"]
    keep = race.isin(["African-American", "Caucasian"])
    X = data.X[keep].drop(columns=list(CONFIG.race_dummies), errors="ignore")
    target = (race[keep] == "African-American").astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=CONFIG.random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn's lbfgs "iprint" option warning
        score = cross_val_predict(
            LogisticRegression(max_iter=1000), X, target, cv=cv, method="predict_proba"
        )[:, 1]
    return float(roc_auc_score(target, score))
