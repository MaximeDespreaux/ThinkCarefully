"""Race proxies: which features carry race information, and how much a feature set leaks.

Dropping the race columns ("fairness through unawareness") does not remove race if other
features encode it. Two measures, both on the training split so the holdout stays unseen:

* **Pairwise association**: bias-corrected Cramér's V between each race dummy and each other
  feature. A feature is a *race proxy* if its V with any race dummy is at least
  PROXY_THRESHOLD (0.1, Cohen's "small effect"). Priors is discretised into the same bands
  as the EDA, so the numbers match its association matrix.
* **Joint leakage**: how well race can be predicted from a whole feature set: the 5-fold
  cross-validated AUC of a logistic model separating African-American from Caucasian
  defendants (the two groups every fairness comparison here uses). 0.5 means the set carries
  no race information; pairwise measures can miss proxies that only work in combination.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from compas_scoring.config import CONFIG
from compas_scoring.data import build_dataset, split_index

PROXY_THRESHOLD = 0.1
# Same bands as scripts/run_eda.py, so V is comparable with the EDA's association matrix.
PRIORS_BANDS = ([-1, 0, 1, 3, 6, 100], ["0", "1", "2-3", "4-6", "7+"])


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramér's V between two categorical series (Bergsma 2013)."""
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    k_corrected = k - (k - 1) ** 2 / (n - 1)
    denominator = min(k_corrected - 1, r_corrected - 1)
    return float(np.sqrt(phi2 / denominator)) if denominator > 0 else float("nan")


def _training_rows() -> pd.DataFrame:
    train_idx, _ = split_index()
    return build_dataset("race_aware").X.loc[train_idx]


def race_associations() -> pd.DataFrame:
    """Cramér's V of every race dummy (rows) with every non-race feature (columns)."""
    X = _training_rows()
    others = [c for c in X.columns if c not in CONFIG.race_dummies]
    discrete = X.copy()
    edges, labels = PRIORS_BANDS
    discrete["Number_of_Priors"] = pd.cut(discrete["Number_of_Priors"], edges, labels=labels)
    return pd.DataFrame(
        {col: [cramers_v(discrete[race], discrete[col]) for race in CONFIG.race_dummies]
         for col in others},
        index=CONFIG.race_dummies,
    )  # fmt: skip


def race_proxies(threshold: float = PROXY_THRESHOLD) -> list[str]:
    """Non-race features whose association with some race dummy reaches the threshold."""
    strongest = race_associations().max(axis=0)
    return [col for col in strongest.index if strongest[col] >= threshold]


def race_leakage(feature_set: str) -> float:
    """CV AUC of predicting African-American vs Caucasian from a feature set (0.5 = none)."""
    train_idx, _ = split_index()
    data = build_dataset(feature_set).subset(train_idx)
    race = data.groups["race"]
    keep = race.isin(["African-American", "Caucasian"])
    X = data.X[keep].drop(columns=list(CONFIG.race_dummies), errors="ignore")
    target = (race[keep] == "African-American").astype(int)
    if X.shape[1] == 0:
        return 0.5
    cv = StratifiedKFold(5, shuffle=True, random_state=CONFIG.random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn's lbfgs "iprint" option warning
        score = cross_val_predict(
            LogisticRegression(max_iter=1000), X, target, cv=cv, method="predict_proba"
        )[:, 1]
    return float(roc_auc_score(target, score))
