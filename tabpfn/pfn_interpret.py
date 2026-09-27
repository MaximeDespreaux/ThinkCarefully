"""Post-hoc explanations of TabPFN: marginal effects, surrogate trees, PDP/ICE, SHAP, LIME.

TabPFN has no coefficients and no splits, so every method here reads the model from the
outside, by asking it questions. Each ``predict_proba`` call re-reads the training context, so
the functions build their queries into as few calls as possible.

Ported from the earlier study (compas-data, src/compas_scoring/interpret.py). New here:
``marginal_effects`` and the three impurity measures in ``surrogate_impurity``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

from compas_scoring.config import CONFIG

# Dummy groups that encode one categorical variable each: at most one of them is 1, and
# "all zero" is the reference category (Caucasian, 25 - 45). Flipping one dummy to 1 has to
# clear the others, or the counterfactual row is a person who cannot exist.
AGE_DUMMIES = ["Age_Above_FourtyFive", "Age_Below_TwentyFive"]
PRIORS = "Number_of_Priors"


def _positive(estimator, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(estimator.predict_proba(X))[:, 1]


# ----------------------------------------------------------------------- marginal effects


def marginal_effects(estimator, X: pd.DataFrame) -> pd.DataFrame:
    """Average change in predicted risk from a one-unit change in each feature.

    TabPFN has no coefficients; this is the model-agnostic stand-in for them. For a dummy:
    everybody set to 1 minus everybody set to 0, other features as observed (for race and age,
    0 means the reference category). For priors: +1 prior against the observed count. All
    counterfactual frames go to the model in a single call.
    """
    one_hot = {c: list(CONFIG.race_dummies) for c in CONFIG.race_dummies}
    one_hot.update({c: AGE_DUMMIES for c in AGE_DUMMIES})

    frames, labels = [], []
    for feature in X.columns:
        if feature == PRIORS:
            low, high = X.copy(), X.copy()
            high[PRIORS] = high[PRIORS] + 1
            change = "observed -> +1"
        else:
            siblings = [c for c in one_hot.get(feature, [feature]) if c in X.columns]
            low, high = X.copy(), X.copy()
            low[siblings] = 0.0
            high[siblings] = 0.0
            high[feature] = 1.0
            change = "0 -> 1"
        frames += [low, high]
        labels.append((feature, change))

    scores = _positive(estimator, pd.concat(frames, ignore_index=True))
    scores = scores.reshape(len(frames), len(X))

    rows = []
    for i, (feature, change) in enumerate(labels):
        effect = scores[2 * i + 1] - scores[2 * i]
        rows.append(
            {
                "feature": feature,
                "change": change,
                "marginal_effect": float(effect.mean()),
                "sd": float(effect.std(ddof=1)),
                # Standard error of the mean over explained rows: the spread of the effect
                # across people, not model uncertainty.
                "se": float(effect.std(ddof=1) / np.sqrt(len(effect))),
                "share_positive": float((effect > 0).mean()),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.reindex(frame["marginal_effect"].abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------------- surrogate trees


def _impurity(p: np.ndarray, criterion: str) -> np.ndarray:
    """Node impurity from class proportions, one row per node."""
    if criterion == "gini":
        return 1.0 - (p**2).sum(axis=1)
    if criterion == "entropy":
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.where(p > 0, np.log2(p), 0.0)
        return -(p * logs).sum(axis=1)
    if criterion == "misclassification":
        return 1.0 - p.max(axis=1)
    raise ValueError(f"unknown criterion {criterion!r}")


CRITERIA = ("gini", "entropy", "misclassification")


def impurity_decrease(tree, feature_names, criterion: str) -> pd.Series:
    """Weighted impurity decrease per feature on a fitted classification tree, normalised.

    Computed from the tree's node class proportions, so the same tree can be scored on every
    criterion (sklearn only grows with gini or entropy, and reports only the one it grew with).
    """
    t = tree.tree_
    counts = t.value[:, 0, :]
    p = counts / counts.sum(axis=1, keepdims=True)
    impurity = _impurity(p, criterion)
    weight = t.weighted_n_node_samples

    decrease = np.zeros(len(feature_names))
    for node in range(t.node_count):
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:  # leaf
            continue
        decrease[t.feature[node]] += (
            weight[node] * impurity[node]
            - weight[left] * impurity[left]
            - weight[right] * impurity[right]
        )
    total = decrease.sum()
    return pd.Series(decrease / total if total > 0 else decrease, index=list(feature_names))


def surrogate_impurity(
    black_box,
    X: pd.DataFrame,
    threshold: float,
    max_depth: int = 4,
    seed: int = CONFIG.random_state,
) -> dict:
    """Two readable trees fitted on TabPFN's own outputs, with their fidelity.

    * a regression tree on TabPFN's **scores**: fidelity R^2, and the rules;
    * one classification tree per growing criterion (gini, entropy) on TabPFN's **decisions**
      at ``threshold``: fidelity accuracy, and every feature's impurity decrease under gini,
      entropy and misclassification error.

    The target is the model's output, not the truth: a surrogate explains the model, not the
    world, and its impurity decreases are only worth reading next to its fidelity.
    """
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

    scores = _positive(black_box, X)
    decisions = (scores >= threshold).astype(int)

    regressor = DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=50, random_state=seed
    ).fit(X, scores)

    importance, fidelity = [], {}
    for grown_with in ("gini", "entropy"):
        tree = DecisionTreeClassifier(
            criterion=grown_with, max_depth=max_depth, min_samples_leaf=50, random_state=seed
        ).fit(X, decisions)
        fidelity[grown_with] = float(accuracy_score(decisions, tree.predict(X)))
        for criterion in CRITERIA:
            for feature, value in impurity_decrease(tree, X.columns, criterion).items():
                importance.append(
                    {
                        "tree_grown_with": grown_with,
                        "criterion": criterion,
                        "feature": feature,
                        "impurity_decrease_share": float(value),
                    }
                )

    return {
        "fidelity_r2": float(r2_score(scores, regressor.predict(X))),
        "fidelity_accuracy_gini": fidelity["gini"],
        "fidelity_accuracy_entropy": fidelity["entropy"],
        "decision_rate": float(decisions.mean()),
        "n_leaves": int(regressor.get_n_leaves()),
        "max_depth": max_depth,
        "rules": export_text(regressor, feature_names=list(X.columns), decimals=3),
        "impurity": pd.DataFrame(importance),
    }


# --------------------------------------------------------------------------- PDP and ICE


def partial_dependence_curve(estimator, X, feature: str, grid_resolution: int = 20):
    """Average predicted risk as one feature varies."""
    result = partial_dependence(
        estimator,
        X,
        features=[list(X.columns).index(feature)],
        grid_resolution=grid_resolution,
        kind="average",
    )
    return pd.DataFrame({feature: result["grid_values"][0], "predicted_risk": result["average"][0]})


def ice_curves(
    estimator,
    X: pd.DataFrame,
    feature: str,
    n_curves: int = 100,
    grid_resolution: int = 20,
    seed: int = CONFIG.random_state,
) -> pd.DataFrame:
    """One predicted-risk curve per individual as the feature sweeps its range.

    The PDP is the average of these curves; if they fan out, the single PDP line describes
    nobody in particular.
    """
    rows = X if len(X) <= n_curves else X.sample(n_curves, random_state=seed)
    result = partial_dependence(
        estimator,
        rows,
        features=[list(X.columns).index(feature)],
        grid_resolution=grid_resolution,
        kind="individual",
    )
    return pd.DataFrame(result["individual"][0], columns=result["grid_values"][0], index=rows.index)


def center_ice(curves: pd.DataFrame, anchor=None) -> pd.DataFrame:
    """Every curve re-expressed as a difference from its own value at the anchor."""
    anchor = curves.columns.min() if anchor is None else anchor
    return curves.sub(curves[anchor], axis=0)


def ice_heterogeneity(curves: pd.DataFrame) -> dict[str, float]:
    """How much the individual curves disagree about the total effect of the feature."""
    final = center_ice(curves)[curves.columns.max()]
    return {
        "n_curves": int(len(curves)),
        "mean_total_effect": float(final.mean()),
        "sd_total_effect": float(final.std(ddof=1)),
        "min_total_effect": float(final.min()),
        "max_total_effect": float(final.max()),
        # Some curves up and others down: heterogeneity a PDP cannot show.
        "sign_disagreement": bool((final.min() < 0) and (final.max() > 0)),
    }


# ----------------------------------------------------------------------------------- SHAP


def kernel_shap(
    estimator,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    n_background: int | None = None,
    n_explain: int | None = None,
    n_samples: int | None = None,
) -> tuple[np.ndarray, float, pd.DataFrame, float]:
    """KernelSHAP (there is no TreeSHAP for TabPFN). Returns (values, base, rows, seconds).

    Budgets default to [tool.compas_scoring.iterations]. The background is summarised by
    k-means, and the explained rows are the first ``n_explain`` of ``X_explain``.
    """
    import shap

    n_background = n_background or CONFIG.iterations.shap_background
    n_explain = n_explain or CONFIG.iterations.shap_explain
    n_samples = n_samples or CONFIG.iterations.shap_nsamples
    columns = list(X_explain.columns)

    background = shap.kmeans(X_background, min(n_background, len(X_background)))
    explain = X_explain.iloc[: min(n_explain, len(X_explain))]

    def predict(data):
        return _positive(estimator, pd.DataFrame(data, columns=columns))

    start = time.perf_counter()
    explainer = shap.KernelExplainer(predict, background)
    values = explainer.shap_values(explain, nsamples=n_samples, silent=True)
    elapsed = time.perf_counter() - start
    return np.asarray(values), float(explainer.expected_value), explain, elapsed


def global_importance(shap_values: np.ndarray, feature_names) -> pd.Series:
    """Mean |SHAP| per feature."""
    return pd.Series(
        np.abs(np.asarray(shap_values)).mean(axis=0), index=list(feature_names)
    ).sort_values(ascending=False)


def shap_efficiency_check(
    shap_values: np.ndarray, base_value: float, predictions: np.ndarray
) -> dict[str, float]:
    """Efficiency axiom: the SHAP values add up to prediction minus base value.

    KernelSHAP only approximates it, so the residual measures how far to trust the values.
    """
    residual = np.asarray(predictions, dtype=float) - (
        np.asarray(shap_values).sum(axis=1) + base_value
    )
    return {
        "max_abs_residual": float(np.abs(residual).max()),
        "mean_abs_residual": float(np.abs(residual).mean()),
        "n_rows": int(len(residual)),
        "holds_to_1e-5": bool(np.abs(residual).max() < 1e-5),
    }


# ----------------------------------------------------------------------------------- LIME


def lime_explanation(
    estimator,
    X_train: pd.DataFrame,
    row: pd.Series,
    num_samples: int = 5000,
    seed: int = CONFIG.random_state,
) -> pd.Series:
    """LIME's local surrogate weights for one row, one value per feature."""
    from lime.lime_tabular import LimeTabularExplainer

    columns = list(X_train.columns)
    explainer = LimeTabularExplainer(
        X_train.to_numpy(dtype=float),
        feature_names=columns,
        class_names=["no re-offence", "re-offence"],
        discretize_continuous=True,
        random_state=seed,
    )
    explanation = explainer.explain_instance(
        row.to_numpy(dtype=float),
        lambda array: estimator.predict_proba(pd.DataFrame(array, columns=columns)),
        num_features=len(columns),
        num_samples=num_samples,
    )
    # LIME labels a weight by its discretised condition ("Number_of_Priors > 4.00"), so the
    # owning column is recovered by name. Longest names first, so "Other" cannot claim a
    # label that belongs to a longer column containing it.
    weights = pd.Series(0.0, index=columns)
    by_length = sorted(columns, key=len, reverse=True)
    for label, weight in explanation.as_list():
        owner = next(c for c in by_length if c in label)
        weights[owner] += weight
    return weights


# ------------------------------------------------------------------------------ agreement


def importance_agreement(rankings: dict[str, pd.Series]) -> pd.DataFrame:
    """Spearman correlation between importance vectors (e.g. two runs, or two methods)."""
    features = sorted(set().union(*(set(r.index) for r in rankings.values())))
    matrix = pd.DataFrame({name: r.reindex(features) for name, r in rankings.items()})
    return matrix.corr(method="spearman")
