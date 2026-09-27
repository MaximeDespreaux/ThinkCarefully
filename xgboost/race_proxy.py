"""How well can race be predicted from the *non-race* features?

Companion to fairness.py's finding that dropping race from the model's inputs barely moves
the racial disparity: if the remaining features can recover race on their
own, "fairness through unawareness" was never going to work, because the model can still act
on race indirectly, through whichever feature proxies for it.

Here the target is flipped: instead of predicting recidivism, a fresh XGBoost model predicts
*race* (African-American vs everyone else in the test set) from the race-blind features. Two
variants are compared:

- all race-blind features (Number_of_Priors, both age dummies, Female, Misdemeanor)
- Number_of_Priors alone

An AUC well above 0.5 means race is recoverable from that feature set; an AUC near 0.5 means
it is not. Neither model is saved: both fit in a few seconds, so the notebook recomputes them.

Run from the repository root:  ``python xgboost/race_proxy.py``
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from compas_scoring.data import Dataset
from xgb_model import load_data, tune_xgboost

FEATURE_VARIANTS = {
    "all_features": None,  # every column of the race_blind feature set
    "priors_only": ["Number_of_Priors"],
}


def race_target(data: Dataset) -> pd.Series:
    """1 if the defendant is African-American, 0 for every other race in the test set."""
    return (data.groups["race"] == "African-American").astype(int).rename("is_african_american")


def fit_proxy(X: pd.DataFrame, y: pd.Series, n_iter: int = 40):
    """A model whose target is race, not recidivism; same tuning as the real models get."""
    return tune_xgboost(X, y, n_iter=n_iter)


def evaluate_proxy(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    y_score = model.predict_proba(X)[:, 1]
    return {
        "auc": roc_auc_score(y, y_score),
        "accuracy": accuracy_score(y, (y_score >= 0.5).astype(int)),
    }


def race_proxy_table(feature_set: str = "race_blind", n_iter: int = 40) -> pd.DataFrame:
    """One row per feature variant: how well it recovers race on the held-out test set."""
    train, test = load_data(feature_set)
    y_train, y_test = race_target(train), race_target(test)

    rows = {}
    for name, columns in FEATURE_VARIANTS.items():
        cols = train.X.columns if columns is None else columns
        search = fit_proxy(train.X[cols], y_train, n_iter=n_iter)
        metrics = evaluate_proxy(search.best_estimator_, test.X[cols], y_test)
        rows[name] = {"n_features": len(cols), "cv_auc": search.best_score_, **metrics}
    return pd.DataFrame(rows).T


def main() -> None:
    table = race_proxy_table()
    print("Predicting African-American from the race-blind features (test set):")
    print(table.round(4))


if __name__ == "__main__":
    main()
