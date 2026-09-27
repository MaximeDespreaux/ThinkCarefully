
"""Interpretability of the saved XGBoost model on the shared test set.
 
Covers marginal effects, impurity-based feature importance (weight/gain/cover), partial
dependence (PDP), individual conditional expectation (ICE), LIME, and SHAP.
 
Every function returns data (a Series, DataFrame, or array); plotting is left to the
notebook. The model is loaded from ``xgboost/models/``, so run ``xgb_model.py`` first.
 
Run from the repository root:  ``python xgboost/interpretability.py``
"""
 
from __future__ import annotations
 
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path
from sklearn.inspection import partial_dependence

from compas_scoring.data import Dataset
from xgb_model import MODEL_DIR, load_data, load_model
 
 
def marginal_effects(model, data: Dataset, epsilon: float = 1e-3) -> pd.Series:
    """Average marginal effect of each feature on P(recidivism).
 
    Continuous features are nudged by +/- ``epsilon`` standard deviations (central finite
    difference); binary/dummy features are swapped 0 -> 1 instead, since a fractional step
    is meaningless for them. All other features are held fixed at their observed values.
    """
    effects = {}
    for col in data.X.columns:
        x_plus = data.X.copy()
        x_minus = data.X.copy()
        is_binary = data.X[col].dropna().isin([0, 1]).all()
        if is_binary:
            x_plus[col] = 1
            x_minus[col] = 0
            denom = 1.0
        else:
            step = epsilon * data.X[col].std()
            x_plus[col] = data.X[col] + step
            x_minus[col] = data.X[col] - step
            denom = 2 * step
        p_plus = model.predict_proba(x_plus)[:, 1]
        p_minus = model.predict_proba(x_minus)[:, 1]
        effects[col] = float(np.mean(p_plus - p_minus) / denom)
    return pd.Series(effects, name="marginal_effect").sort_values(key=np.abs, ascending=False)
 
 
def impurity_importance(model) -> pd.DataFrame:
    """XGBoost's three impurity-based importance measures: weight, gain, cover.
 
    - weight: how many times a feature is used to split, across all trees.
    - gain:   average loss reduction from splits on that feature (the usual "importance").
    - cover:  average number of samples affected by splits on that feature.
    """
    booster = model.get_booster()
    measures = ["weight", "gain", "cover"]
    table = pd.DataFrame(
        {measure: pd.Series(booster.get_score(importance_type=measure)) for measure in measures}
    ).fillna(0.0)
    return table.sort_values("gain", ascending=False)
 
 
def pdp_data(
    model, data: Dataset, features: list[str] | None = None, grid_resolution: int = 50
) -> dict[str, pd.DataFrame]:
    """Partial dependence of P(recidivism) on each feature (or the given subset)."""
    features = features or list(data.X.columns)
    result = {}
    for col in features:
        pd_result = partial_dependence(
            model, data.X, [col], kind="average", grid_resolution=grid_resolution
        )
        result[col] = pd.DataFrame(
            {col: pd_result["grid_values"][0], "partial_dependence": pd_result["average"][0]}
        )
    return result
 
 
def ice_data(model, data: Dataset, feature: str, grid_resolution: int = 50) -> pd.DataFrame:
    """ICE curves for one feature: one row per test individual, one column per grid point."""
    ice_result = partial_dependence(
        model, data.X, [feature], kind="individual", grid_resolution=grid_resolution
    )
    grid = ice_result["grid_values"][0]
    curves = ice_result["individual"][0]  # shape: (n_samples, grid_resolution)
    return pd.DataFrame(curves, columns=grid, index=data.X.index)
 
 
def lime_explanation(model, train: Dataset, instance: pd.Series, num_features: int = 10) -> pd.DataFrame:
    """LIME local explanation for one row (e.g. ``test.X.loc[some_index]``)."""
    explainer = LimeTabularExplainer(
        train.X.to_numpy(),
        feature_names=list(train.X.columns),
        class_names=["no_recidivism", "recidivism"],
        mode="classification",
        discretize_continuous=True,
    )
    explanation = explainer.explain_instance(
        instance.to_numpy(), model.predict_proba, num_features=num_features
    )
    return pd.DataFrame(explanation.as_list(), columns=["rule", "weight"])
 
 
def shap_values(model, data: Dataset) -> tuple[np.ndarray, shap.Explanation]:
    """SHAP values for every row in ``data``, via XGBoost's native (exact) TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    explanation = explainer(data.X)
    return explanation.values, explanation
 
 
def interpretability_paths(feature_set: str = "race_aware") -> dict[str, Path]:
    stem = MODEL_DIR / f"interp_{feature_set}"
    return {
        "marginal": stem.with_name(stem.name + "_marginal.csv"),
        "impurity": stem.with_name(stem.name + "_impurity.csv"),
        "shap": stem.with_name(stem.name + "_shap.csv"),
    }
 
 
def save_interpretability(
    marginal: pd.Series,
    impurity: pd.DataFrame,
    shap_vals: np.ndarray,
    data: Dataset,
    feature_set: str = "race_aware",
) -> None:
    paths = interpretability_paths(feature_set)
    paths["marginal"].parent.mkdir(parents=True, exist_ok=True)
    marginal.to_csv(paths["marginal"])
    impurity.to_csv(paths["impurity"])
    pd.DataFrame(shap_vals, columns=data.X.columns, index=data.X.index).to_csv(paths["shap"])
 
 
def load_interpretability(feature_set: str = "race_aware") -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    paths = interpretability_paths(feature_set)
    marginal = pd.read_csv(paths["marginal"], index_col=0).squeeze("columns")
    impurity = pd.read_csv(paths["impurity"], index_col=0)
    shap_df = pd.read_csv(paths["shap"], index_col=0)
    return marginal, impurity, shap_df
 
 
def main(feature_set: str = "race_aware") -> None:
    train, test = load_data(feature_set)
    model = load_model(feature_set)
 
    marginal = marginal_effects(model, test)
    impurity = impurity_importance(model)
    shap_vals, _ = shap_values(model, test)
 
    save_interpretability(marginal, impurity, shap_vals, test, feature_set)
 
    print(f"Marginal effects, feature set {feature_set}:")
    print(marginal)
    print("\nImpurity-based importance (weight, gain, cover):")
    print(impurity)
    print(f"\nSHAP values saved for {len(test)} test rows.")
    print("PDP, ICE, and LIME are computed on demand (per-feature / per-row) — call")
    print("pdp_data(), ice_data(), or lime_explanation() directly from a notebook.")
 
 
if __name__ == "__main__":
    main()


