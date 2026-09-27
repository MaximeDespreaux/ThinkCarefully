"""Interpretability: what drives the model, and what drove *this* prediction.

The four models sit at very different points on the explainability curve, and the
comparison is the deliverable:

* Logistic regression is transparent by construction -- coefficients *are* the model,
  and they convert into a points scorecard a caseworker can apply on paper.
* XGBoost is opaque but cheaply explained: TreeSHAP gives exact Shapley values in
  milliseconds.
* The foundation models have no internal structure to read and no fast explainer, so
  KernelSHAP has to approximate from the outside -- orders of magnitude slower, and
  itself only an estimate. That cost is measured here, not asserted.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.inspection import partial_dependence, permutation_importance

from compas_scoring.config import CONFIG

# Points-scorecard convention, borrowed from credit scoring: `pdo` points double the odds.
SCORECARD_BASE_POINTS = 600
SCORECARD_BASE_ODDS = 1.0
SCORECARD_PDO = 20


# ------------------------------------------------------------------ white-box internals


def odds_ratios(X: pd.DataFrame, y, add_constant: bool = True) -> pd.DataFrame:
    """Logistic coefficients as odds ratios with Wald confidence intervals.

    Fitted through statsmodels rather than read off the sklearn pipeline because the
    client needs the standard errors and p-values, which sklearn does not expose.
    """
    design = sm.add_constant(X, has_constant="add") if add_constant else X
    result = sm.Logit(np.asarray(y), design).fit(disp=False)

    conf = result.conf_int()
    table = pd.DataFrame(
        {
            "coefficient": result.params,
            "std_error": result.bse,
            "p_value": result.pvalues,
            "odds_ratio": np.exp(result.params),
            "or_lower": np.exp(conf[0]),
            "or_upper": np.exp(conf[1]),
        }
    )
    table["significant"] = table["p_value"] < 0.05
    return table.sort_values("odds_ratio", ascending=False)


def points_scorecard(
    coefficients: pd.Series,
    intercept: float,
    n_features: int,
    pdo: int = SCORECARD_PDO,
    base_points: int = SCORECARD_BASE_POINTS,
    base_odds: float = SCORECARD_BASE_ODDS,
) -> pd.DataFrame:
    """Convert log-odds into integer points -- the deployable form of a white-box model.

    A scorecard is auditable without running anything: a caseworker adds up points and a
    defendant can be told exactly which factors cost them. None of the other three models
    can be delivered in this form at all.
    """
    factor = pdo / np.log(2)
    offset = base_points - factor * np.log(base_odds)

    rows = []
    for feature, coefficient in coefficients.items():
        rows.append(
            {
                "feature": feature,
                "coefficient": float(coefficient),
                "points_per_unit": float(-factor * coefficient),
            }
        )
    table = pd.DataFrame(rows)
    table["base_points"] = float(offset - factor * intercept) / max(n_features, 1)
    return table.sort_values("points_per_unit")


def average_marginal_effects(X: pd.DataFrame, y, add_constant: bool = True) -> pd.DataFrame:
    """Average marginal effects: the change in P(re-arrest) per unit of each feature.

    Odds ratios are what a statistician reads; marginal effects are what a client reads.
    "This multiplies the odds by 1.18" is not a sentence a caseworker can act on, whereas
    "each additional prior adds 3 percentage points of predicted risk" is.
    """
    design = sm.add_constant(X, has_constant="add") if add_constant else X
    result = sm.Logit(np.asarray(y), design).fit(disp=False)
    effects = result.get_margeff(at="overall", method="dydx")

    table = pd.DataFrame(
        {
            "marginal_effect": effects.margeff,
            "std_error": effects.margeff_se,
            "p_value": effects.pvalues,
        },
        index=[c for c in design.columns if c != "const"],
    )
    table["ci_lower"] = table["marginal_effect"] - 1.96 * table["std_error"]
    table["ci_upper"] = table["marginal_effect"] + 1.96 * table["std_error"]
    table["significant"] = table["p_value"] < 0.05
    return table.sort_values("marginal_effect", ascending=False)


def lasso_path(X: pd.DataFrame, y, n_steps: int = 40) -> pd.DataFrame:
    """L1 coefficients across the regularisation path, on standardised features.

    Shows the order in which predictors survive shrinkage -- the clearest single view of
    which features are load-bearing and which are along for the ride.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for c in np.logspace(-3, 1.5, n_steps):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l1",
                        C=float(c),
                        solver="liblinear",
                        max_iter=5000,
                        random_state=CONFIG.random_state,
                    ),
                ),
            ]
        ).fit(X, y)
        coefficients = model.named_steps["clf"].coef_[0]
        rows.append(
            {
                "C": float(c),
                "n_selected": int((np.abs(coefficients) > 1e-6).sum()),
                **{n: float(v) for n, v in zip(X.columns, coefficients, strict=True)},
            }
        )
    return pd.DataFrame(rows)


def stepwise_log(
    X: pd.DataFrame, y, direction: str = "forward", max_steps: int | None = None
) -> pd.DataFrame:
    """Greedy feature selection, logging every step it considered and why.

    The log is the deliverable, not the final subset: it records which features were tried,
    what each one bought in cross-validated AUC, and where the search stopped. That is what
    makes the feature-set choice auditable instead of asserted.

    Scored with train-only cross-validation -- the holdout is never consulted.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cv = StratifiedKFold(
        n_splits=CONFIG.iterations.cv_folds, shuffle=True, random_state=CONFIG.random_state
    )

    def score(columns: list[str]) -> tuple[float, float]:
        if not columns:
            return 0.5, 0.0
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=CONFIG.random_state)),
            ]
        )
        scores = cross_val_score(model, X[columns], y, scoring="roc_auc", cv=cv)
        return float(scores.mean()), float(scores.std(ddof=1))

    available = list(X.columns)
    current = [] if direction == "forward" else list(available)
    best_auc, best_sd = score(current)
    max_steps = max_steps or len(available)

    rows = [
        {
            "step": 0,
            "direction": direction,
            "candidate": "(start)",
            "action": "start",
            "n_features": len(current),
            "cv_auc": best_auc,
            "cv_sd": best_sd,
            "delta_auc": 0.0,
            "accepted": True,
            "features": ", ".join(current),
        }
    ]

    for step in range(1, max_steps + 1):
        candidates = (
            [f for f in available if f not in current] if direction == "forward" else list(current)
        )
        if not candidates:
            break

        trials = []
        for feature in candidates:
            trial = (
                [*current, feature]
                if direction == "forward"
                else [f for f in current if f != feature]
            )
            auc, sd = score([c for c in X.columns if c in trial])
            trials.append((feature, trial, auc, sd))

        feature, trial, auc, sd = max(trials, key=lambda t: t[2])
        accepted = auc > best_auc
        rows.append(
            {
                "step": step,
                "direction": direction,
                "candidate": feature,
                "action": "add" if direction == "forward" else "remove",
                "n_features": len(trial),
                "cv_auc": auc,
                "cv_sd": sd,
                "delta_auc": auc - best_auc,
                "accepted": bool(accepted),
                "features": ", ".join(c for c in X.columns if c in trial),
            }
        )
        if not accepted:
            break
        current, best_auc, best_sd = [c for c in X.columns if c in trial], auc, sd

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- SHAP


def tree_shap(estimator, X: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """Exact Shapley values for the tree model. Returns (values, base_value, seconds)."""
    import shap

    model = estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator
    start = time.perf_counter()
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    elapsed = time.perf_counter() - start

    base = explainer.expected_value
    base = float(np.ravel(base)[0])
    if isinstance(values, list):  # older shap returns one array per class
        values = values[1]
    return np.asarray(values), base, elapsed


def kernel_shap(
    estimator,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    n_background: int | None = None,
    n_explain: int | None = None,
) -> tuple[np.ndarray, float, float]:
    """Model-agnostic Shapley approximation, for models with no internal structure to read.

    Subsampled hard on both sides because cost grows with background x explained rows, and
    each evaluation is a full foundation-model forward pass. The wall-clock figure this
    returns is a headline result: it is what post-hoc explanation of a TFM actually costs.
    """
    import shap

    n_background = n_background or CONFIG.iterations.shap_background
    n_explain = n_explain or CONFIG.iterations.shap_explain

    background = shap.kmeans(X_background, min(n_background, len(X_background)))
    explain = X_explain.iloc[: min(n_explain, len(X_explain))]

    start = time.perf_counter()
    explainer = shap.KernelExplainer(lambda data: estimator.predict_proba(data)[:, 1], background)
    values = explainer.shap_values(explain, silent=True)
    elapsed = time.perf_counter() - start

    return np.asarray(values), float(explainer.expected_value), elapsed


def global_importance(shap_values: np.ndarray, feature_names) -> pd.Series:
    """Mean |SHAP| per feature: the standard global-importance summary."""
    return pd.Series(
        np.abs(np.asarray(shap_values)).mean(axis=0), index=list(feature_names)
    ).sort_values(ascending=False)


def local_explanation(
    shap_values: np.ndarray, base_value: float, row: pd.Series, index: int
) -> pd.DataFrame:
    """One defendant's prediction decomposed into per-feature contributions."""
    contributions = np.asarray(shap_values)[index]
    table = pd.DataFrame(
        {
            "feature": list(row.index),
            "value": row.to_numpy(),
            "shap_value": contributions,
        }
    )
    table["abs_shap"] = table["shap_value"].abs()
    table = table.sort_values("abs_shap", ascending=False).drop(columns="abs_shap")
    table.attrs["base_value"] = float(base_value)
    table.attrs["prediction"] = float(base_value + contributions.sum())
    return table


# ------------------------------------------------------- model-agnostic cross-comparison


def permutation_ranking(estimator, X, y, n_repeats: int = 10) -> pd.Series:
    """Permutation importance -- the one importance measure defined for all four models.

    Needed because coefficients, TreeSHAP and KernelSHAP are not on comparable scales;
    this puts every model on one axis so their rankings can actually be compared.
    """
    result = permutation_importance(
        estimator, X, y, n_repeats=n_repeats, random_state=CONFIG.random_state, scoring="roc_auc"
    )
    return pd.Series(result.importances_mean, index=list(X.columns)).sort_values(ascending=False)


def importance_agreement(rankings: dict[str, pd.Series]) -> pd.DataFrame:
    """Spearman correlation between models' importance rankings.

    Low agreement is the finding to surface: if two models with the same AUC disagree on
    *why*, at most one of the explanations is describing something real.
    """
    names = list(rankings)
    features = sorted(set().union(*(set(r.index) for r in rankings.values())))
    matrix = pd.DataFrame(
        {name: rankings[name].reindex(features) for name in names}, index=features
    )
    return matrix.corr(method="spearman")


def partial_dependence_curve(estimator, X, feature: str, grid_resolution: int = 20):
    """Average predicted risk as one feature varies -- comparable across all four models."""
    result = partial_dependence(
        estimator,
        X,
        features=[list(X.columns).index(feature)],
        grid_resolution=grid_resolution,
        kind="average",
    )
    return pd.DataFrame({feature: result["grid_values"][0], "predicted_risk": result["average"][0]})


def archetype_defendants(X: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Three concrete people to carry through every model's local explanation.

    Abstract importance plots do not persuade a client; three named cases where the models
    disagree about the same defendant do.
    """
    priors = X["Number_of_Priors"]
    picks = {
        "Low risk profile": X.index[(priors == 0) & (X["Age_Above_FourtyFive"] == 1)],
        "Borderline profile": X.index[
            (priors.between(2, 4))
            & (X["Age_Below_TwentyFive"] == 0)
            & (X["Age_Above_FourtyFive"] == 0)
        ],
        "High risk profile": X.index[(priors >= 8) & (X["Age_Below_TwentyFive"] == 1)],
    }
    rows = []
    for label, candidates in picks.items():
        if len(candidates) == 0:
            continue
        idx = candidates[0]
        rows.append(
            {"archetype": label, "index": idx, **X.loc[idx].to_dict(), **groups.loc[idx].to_dict()}
        )
    return pd.DataFrame(rows).set_index("archetype")


# ------------------------------------------------------------- ICE and global surrogate


def ice_curves(
    estimator,
    X: pd.DataFrame,
    feature: str,
    n_curves: int = 100,
    grid_resolution: int = 20,
    seed: int = CONFIG.random_state,
) -> pd.DataFrame:
    """One predicted-risk curve per individual, as the feature sweeps its range.

    The PDP is the average of these curves, so it can only ever show one shape. ICE is how
    you find out whether that average describes anybody: if the curves fan out, the model
    treats priors differently for different people and the single PDP line is a fiction.
    """
    rows = X if len(X) <= n_curves else X.sample(n_curves, random_state=seed)
    result = partial_dependence(
        estimator,
        rows,
        features=[list(X.columns).index(feature)],
        grid_resolution=grid_resolution,
        kind="individual",
    )
    grid = result["grid_values"][0]
    values = result["individual"][0]  # (n_curves, n_grid)
    return pd.DataFrame(values, columns=grid, index=rows.index)


def center_ice(curves: pd.DataFrame, anchor=None) -> pd.DataFrame:
    """Centered ICE: every curve re-expressed as a difference from the anchor point.

    Raw ICE curves start at different heights, so differences in *level* drown out
    differences in *slope* -- which is the thing worth seeing. Subtracting each curve's own
    value at the anchor pins them all to zero there, leaving only the shape.
    """
    anchor = curves.columns.min() if anchor is None else anchor
    return curves.sub(curves[anchor], axis=0)


def ice_heterogeneity(curves: pd.DataFrame) -> dict[str, float]:
    """How much the individual curves disagree -- the number that justifies showing ICE."""
    centered = center_ice(curves)
    final = centered[centered.columns.max()]
    return {
        "n_curves": int(len(curves)),
        "mean_total_effect": float(final.mean()),
        "sd_total_effect": float(final.std(ddof=1)),
        "min_total_effect": float(final.min()),
        "max_total_effect": float(final.max()),
        # A negative minimum with a positive maximum means the model moves some people up
        # and others down on the same feature -- heterogeneity a PDP cannot show.
        "sign_disagreement": bool((final.min() < 0) and (final.max() > 0)),
    }


def global_surrogate(
    black_box,
    X: pd.DataFrame,
    max_depth: int = 4,
    seed: int = CONFIG.random_state,
) -> dict:
    """Approximate an opaque model with a readable tree fitted on *its predictions*.

    The target is the black box's output, not the truth: a surrogate explains the model, not
    the world. Fidelity (R-squared against the black box's scores) is the number that decides
    whether the tree may be shown at all -- a low-fidelity surrogate is worse than no
    surrogate, because it looks like an explanation while describing a different model.
    """
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.tree import DecisionTreeRegressor, export_text

    black_box_scores = np.asarray(black_box.predict_proba(X))[:, 1]
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=50, random_state=seed).fit(
        X, black_box_scores
    )
    surrogate_scores = tree.predict(X)

    return {
        "fidelity_r2": float(r2_score(black_box_scores, surrogate_scores)),
        "fidelity_correlation": float(np.corrcoef(black_box_scores, surrogate_scores)[0, 1]),
        "surrogate_auc_vs_black_box": float(
            roc_auc_score(
                (black_box_scores >= np.median(black_box_scores)).astype(int), surrogate_scores
            )
        ),
        "n_leaves": int(tree.get_n_leaves()),
        "max_depth": max_depth,
        "rules": export_text(tree, feature_names=list(X.columns), decimals=3),
        "importance": pd.Series(tree.feature_importances_, index=X.columns).to_dict(),
    }


# ------------------------------------------------------------------------------- LIME


def lime_explanation(
    estimator,
    X_train: pd.DataFrame,
    row: pd.Series,
    num_features: int | None = None,
    num_samples: int = 5000,
    seed: int = CONFIG.random_state,
) -> pd.Series:
    """LIME's local surrogate weights for one row, aligned to the full feature index.

    LIME perturbs around the instance and fits a weighted sparse linear model to whatever
    the black box says nearby. Unlike SHAP it carries no axioms -- which is exactly why
    running both and comparing them is worth doing.
    """
    from lime.lime_tabular import LimeTabularExplainer

    columns = list(X_train.columns)
    explainer = LimeTabularExplainer(
        X_train.to_numpy(dtype=float),
        feature_names=columns,
        class_names=["no re-arrest", "re-arrest"],
        discretize_continuous=True,
        random_state=seed,
    )

    def predict_fn(array):
        return estimator.predict_proba(pd.DataFrame(array, columns=columns))

    explanation = explainer.explain_instance(
        row.to_numpy(dtype=float),
        predict_fn,
        num_features=num_features or len(columns),
        num_samples=num_samples,
    )
    # LIME labels a weight by its discretised condition ("Number_of_Priors > 4.00"), so the
    # owning column has to be recovered by matching names back to the feature list.
    weights = pd.Series(0.0, index=columns)
    for label, weight in explanation.as_list():
        for column in columns:
            if column in label:
                weights[column] += weight
                break
    return weights


# ---------------------------------------------------------------- disagreement problem


def _top_k(values: pd.Series, k: int) -> set[str]:
    return set(values.abs().sort_values(ascending=False).head(k).index)


def disagreement_metrics(a: pd.Series, b: pd.Series, k: int = 5) -> dict[str, float]:
    """Krishna et al. (2025) agreement measures between two explanations.

    A single rank correlation hides the failures that matter. Two explanations can correlate
    at 0.9 and still disagree about which feature is most important, or about whether a
    feature raises or lowers risk -- and a caseworker only ever sees the top few features
    and their direction.
    """
    features = a.index.intersection(b.index)
    a, b = a[features], b[features]

    top_a, top_b = _top_k(a, k), _top_k(b, k)
    rank_a = a.abs().rank(ascending=False)
    rank_b = b.abs().rank(ascending=False)
    shared = top_a & top_b

    pairs = [(i, j) for i in features for j in features if i < j]
    concordant = sum(1 for i, j in pairs if (rank_a[i] < rank_a[j]) == (rank_b[i] < rank_b[j]))

    return {
        "feature_agreement": len(shared) / k,
        "rank_agreement": (sum(1 for f in shared if rank_a[f] == rank_b[f]) / k if shared else 0.0),
        "sign_agreement": float((np.sign(a) == np.sign(b)).mean()),
        "signed_rank_agreement": (
            sum(1 for f in shared if rank_a[f] == rank_b[f] and np.sign(a[f]) == np.sign(b[f])) / k
            if shared
            else 0.0
        ),
        "pairwise_rank_agreement": concordant / len(pairs) if pairs else 1.0,
        "rank_correlation": float(a.abs().corr(b.abs(), method="spearman")),
        "top_feature_match": bool(a.abs().idxmax() == b.abs().idxmax()),
    }


def disagreement_panel(explanations: dict[str, pd.Series], k: int = 5) -> pd.DataFrame:
    """Every pair of explanation methods, scored on all the agreement measures."""
    names = list(explanations)
    rows = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            rows.append(
                {
                    "method_a": first,
                    "method_b": second,
                    **disagreement_metrics(explanations[first], explanations[second], k=k),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ SHAP axioms


def shap_efficiency_check(
    shap_values: np.ndarray, base_value: float, predictions: np.ndarray
) -> dict[str, float]:
    """Verify the efficiency axiom: sum of SHAP values == prediction minus base value.

    This is the property that makes Shapley values a fair allocation rather than just an
    attribution heuristic. TreeSHAP satisfies it exactly; KernelSHAP only approximates it,
    so the residual here is a direct measure of how much the foundation models' explanations
    can be trusted.
    """
    reconstructed = np.asarray(shap_values).sum(axis=1) + base_value
    residual = np.asarray(predictions, dtype=float) - reconstructed
    return {
        "max_abs_residual": float(np.abs(residual).max()),
        "mean_abs_residual": float(np.abs(residual).mean()),
        "n_rows": int(len(residual)),
        "exact": bool(np.abs(residual).max() < 1e-6),
    }


def paired_local_explanation(
    estimator,
    X_train: pd.DataFrame,
    row: pd.Series,
    n_background: int = 50,
    lime_samples: int = 2000,
    seed: int = CONFIG.random_state,
) -> pd.DataFrame:
    """One defendant, explained twice: LIME and KernelSHAP, on the same scale.

    The aggregate agreement table says explanations disagree. This is the version a client
    can argue with: a single person, two methods that are each defensible on their own, and
    whatever they say about why that person was scored the way they were. Where the bars
    point in opposite directions, one of the two explanations is telling the defendant
    something false about their own case -- and nothing in either method reveals which.

    Both methods are run on the same row with the same background, so the comparison is not
    confounded by sampling: only the method differs.
    """
    import shap

    columns = list(X_train.columns)
    frame = row.to_frame().T[columns].astype(float)

    lime_weights = lime_explanation(estimator, X_train, row, num_samples=lime_samples, seed=seed)

    background = shap.kmeans(X_train, min(n_background, len(X_train)))
    explainer = shap.KernelExplainer(
        lambda data: estimator.predict_proba(pd.DataFrame(data, columns=columns))[:, 1],
        background,
    )
    shap_values = np.asarray(explainer.shap_values(frame, silent=True)).ravel()

    return pd.DataFrame(
        {
            "feature": columns,
            "value": row[columns].to_numpy(dtype=float),
            "lime": lime_weights[columns].to_numpy(dtype=float),
            "kernelshap": shap_values,
        }
    )


def explanation_conflicts(paired: pd.DataFrame) -> dict[str, float]:
    """Where the two explanations of one prediction actually contradict each other.

    A feature counts as contested only when BOTH methods give it real weight. Without that
    floor, a feature one method scores at exactly zero and the other at 1e-4 registers as a
    sign conflict, and the "disagreement" reported is arithmetic noise on features neither
    explanation is actually using -- which would be a manufactured finding, not a measured one.
    """
    lime, shap_values = paired["lime"].to_numpy(), paired["kernelshap"].to_numpy()
    # 5% of each method's own largest contribution: scale-free, and it keeps the threshold
    # from depending on whether the model outputs probabilities or log-odds.
    floor_lime = 0.05 * np.abs(lime).max()
    floor_shap = 0.05 * np.abs(shap_values).max()
    active = (np.abs(lime) >= floor_lime) & (np.abs(shap_values) >= floor_shap)
    opposed = active & (np.sign(lime) != np.sign(shap_values))
    return {
        "n_features_contested": int(active.sum()),
        "n_opposite_sign": int(opposed.sum()),
        "opposite_sign_share": float(opposed.sum() / max(active.sum(), 1)),
        "conflicting_features": ", ".join(paired.loc[opposed, "feature"]),
        "lime_top": str(paired.loc[paired["lime"].abs().idxmax(), "feature"]),
        "shap_top": str(paired.loc[paired["kernelshap"].abs().idxmax(), "feature"]),
        "top_feature_agrees": bool(
            paired.loc[paired["lime"].abs().idxmax(), "feature"]
            == paired.loc[paired["kernelshap"].abs().idxmax(), "feature"]
        ),
    }
