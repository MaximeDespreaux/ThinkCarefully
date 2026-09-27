"""Stability: does the model say the same thing tomorrow, on slightly different data?

A model that scores well once but moves under resampling is not deployable -- two
defendants with identical files would get different answers depending on which Tuesday
the model was refitted. This module measures that from five angles:

* **Resampling** -- spread of CV performance across repeats.
* **Prediction stability** -- how much one individual's score moves across bootstrap refits.
  This is the one a client feels directly, and it is not visible in any average metric.
* **Explanation stability** -- whether the drivers stay in the same order across refits.
  An explanation that reshuffles is worse than no explanation, because it invites
  confident but unfounded reasoning.
* **Sample-size sensitivity** -- learning curves; where a foundation model should earn its keep.
* **Perturbation and drift** -- response to noisy inputs and to shifted populations (PSI).

Temporal stability is deliberately absent: the chosen table carries no screening dates,
so it cannot be measured here. Saying so is more useful than substituting a proxy.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from compas_scoring.config import CONFIG


def _seeded(estimator, seed: int):
    """Clone an estimator, re-seeding every random_state it exposes."""
    fresh = clone(estimator)
    params = {k: seed for k in fresh.get_params() if k.endswith("random_state")}
    if params:
        fresh.set_params(**params)
    return fresh


def _proba(estimator, X) -> np.ndarray:
    return estimator.predict_proba(X)[:, 1]


# ------------------------------------------------------------------- resampling spread


def repeated_cv(
    estimator, X, y, n_repeats: int | None = None, n_splits: int | None = None
) -> pd.DataFrame:
    """Repeated stratified k-fold. One row per fold, so the caller sees the spread."""
    n_repeats = n_repeats or CONFIG.iterations.cv_repeats
    n_splits = n_splits or CONFIG.iterations.cv_folds

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=CONFIG.random_state
    )
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)

    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fitted = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        scores = _proba(fitted, X.iloc[test_idx])
        rows.append(
            {
                "fold": fold,
                "repeat": fold // n_splits,
                "auc": roc_auc_score(y.iloc[test_idx], scores),
                "brier": brier_score_loss(y.iloc[test_idx], scores),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------- prediction / explanation stability


def bootstrap_predictions(
    estimator, X_train, y_train, X_eval, n_refits: int | None = None
) -> np.ndarray:
    """Refit on bootstrap resamples; return an (n_refits, n_eval) matrix of probabilities."""
    n_refits = n_refits or CONFIG.iterations.bootstrap_refits
    rng = np.random.default_rng(CONFIG.random_state)

    X_train = X_train.reset_index(drop=True)
    y_train = pd.Series(np.asarray(y_train)).reset_index(drop=True)
    n = len(X_train)

    out = np.empty((n_refits, len(X_eval)))
    for i in range(n_refits):
        idx = rng.integers(0, n, n)
        while y_train.iloc[idx].nunique() < 2:  # pragma: no cover - vanishingly rare
            idx = rng.integers(0, n, n)
        fitted = _seeded(estimator, CONFIG.random_state + i).fit(
            X_train.iloc[idx], y_train.iloc[idx]
        )
        out[i] = _proba(fitted, X_eval)
    return out


def prediction_stability(matrix: np.ndarray) -> dict[str, float]:
    """Summarise how far an individual's score wanders across refits."""
    per_individual_sd = matrix.std(axis=0, ddof=1)
    spread = np.percentile(matrix, 97.5, axis=0) - np.percentile(matrix, 2.5, axis=0)
    return {
        "mean_sd": float(per_individual_sd.mean()),
        "median_sd": float(np.median(per_individual_sd)),
        "p95_sd": float(np.percentile(per_individual_sd, 95)),
        "max_sd": float(per_individual_sd.max()),
        "mean_95pct_range": float(spread.mean()),
    }


def decision_flip_rate(matrix: np.ndarray, threshold: float) -> dict[str, float]:
    """Share of defendants whose *decision* -- not just score -- is unstable.

    The number to put in front of a client: at this threshold, this fraction of people
    would be classified differently depending only on which resample trained the model.
    """
    decisions = matrix >= threshold
    detain_share = decisions.mean(axis=0)
    unstable = (detain_share > 0.05) & (detain_share < 0.95)
    return {
        "threshold": float(threshold),
        "unstable_share": float(unstable.mean()),
        "mean_decision_entropy": float(
            np.mean(
                -(
                    np.clip(detain_share, 1e-9, 1) * np.log2(np.clip(detain_share, 1e-9, 1))
                    + np.clip(1 - detain_share, 1e-9, 1)
                    * np.log2(np.clip(1 - detain_share, 1e-9, 1))
                )
            )
        ),
    }


def importance_stability(
    estimator, X_train, y_train, importance_fn, n_refits: int = 25
) -> dict[str, float]:
    """Rank correlation of feature importances across bootstrap refits.

    ``importance_fn`` takes a fitted estimator and returns one importance per feature.
    """
    rng = np.random.default_rng(CONFIG.random_state)
    X_train = X_train.reset_index(drop=True)
    y_train = pd.Series(np.asarray(y_train)).reset_index(drop=True)
    n = len(X_train)

    importances = []
    for i in range(n_refits):
        idx = rng.integers(0, n, n)
        fitted = _seeded(estimator, CONFIG.random_state + i).fit(
            X_train.iloc[idx], y_train.iloc[idx]
        )
        importances.append(np.asarray(importance_fn(fitted), dtype=float))

    matrix = np.vstack(importances)
    correlations = [
        spearmanr(matrix[i], matrix[j]).statistic
        for i in range(len(matrix))
        for j in range(i + 1, len(matrix))
    ]
    ranks = np.argsort(np.argsort(-matrix, axis=1), axis=1)
    return {
        "mean_rank_correlation": float(np.nanmean(correlations)),
        "min_rank_correlation": float(np.nanmin(correlations)),
        "top_feature_consistency": float(
            pd.Series(np.argmax(matrix, axis=1)).value_counts(normalize=True).iloc[0]
        ),
        "mean_rank_sd": float(ranks.std(axis=0, ddof=1).mean()),
    }


# ------------------------------------------------------------- sample size, noise, drift


def learning_curve(
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    sizes: Sequence[int] | None = None,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Performance against training-set size.

    The scenario behind this: a client with one county's data, not fifty. If the
    foundation model holds its AUC at n=250 where the others collapse, that is a real
    deployment advantage and not a benchmark curiosity.
    """
    sizes = sizes or CONFIG.iterations.learning_curve_sizes
    rng = np.random.default_rng(CONFIG.random_state)
    X_train = X_train.reset_index(drop=True)
    y_train = pd.Series(np.asarray(y_train)).reset_index(drop=True)

    rows = []
    for size in sizes:
        if size > len(X_train):
            continue
        for repeat in range(n_repeats):
            idx = rng.choice(len(X_train), size=size, replace=False)
            if y_train.iloc[idx].nunique() < 2:
                continue
            start = time.perf_counter()
            fitted = _seeded(estimator, CONFIG.random_state + repeat).fit(
                X_train.iloc[idx], y_train.iloc[idx]
            )
            rows.append(
                {
                    "n_train": size,
                    "repeat": repeat,
                    "auc": roc_auc_score(y_test, _proba(fitted, X_test)),
                    "fit_seconds": time.perf_counter() - start,
                }
            )
    return pd.DataFrame(rows)


def perturbation_sensitivity(
    estimator, X, columns: Sequence[str] | None = None, n_trials: int = 20
) -> pd.DataFrame:
    """How far predictions move when an input is recorded slightly wrong.

    Priors counts get miskeyed and charge degrees get reclassified, so a model whose
    output swings on a one-unit change is fragile in the way that actually bites.
    """
    rng = np.random.default_rng(CONFIG.random_state)
    baseline = _proba(estimator, X)
    columns = list(columns) if columns else list(X.columns)

    rows = []
    for column in columns:
        is_binary = set(np.unique(X[column])) <= {0.0, 1.0}
        for trial in range(n_trials if not is_binary else 1):
            perturbed = X.copy()
            if is_binary:
                perturbed[column] = 1.0 - perturbed[column]
                label = "flip"
            else:
                noise = rng.integers(-1, 2, len(X))
                perturbed[column] = np.clip(perturbed[column] + noise, 0, None)
                label = "+-1 count"
            shift = np.abs(_proba(estimator, perturbed) - baseline)
            rows.append(
                {
                    "feature": column,
                    "perturbation": label,
                    "trial": trial,
                    "mean_abs_shift": float(shift.mean()),
                    "p95_abs_shift": float(np.percentile(shift, 95)),
                }
            )
    return pd.DataFrame(rows)


def population_stability_index(expected, actual, n_bins: int = 10) -> float:
    """PSI between two score distributions. The standard deployment drift alarm.

    Convention: <0.10 stable, 0.10-0.25 watch, >0.25 investigate.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def seed_sensitivity(estimator, X_train, y_train, X_test, y_test, n_seeds: int | None = None):
    """Refit under different seeds on identical data. Any spread here is pure arbitrariness."""
    n_seeds = n_seeds or CONFIG.iterations.n_seeds
    rows = []
    for seed in range(n_seeds):
        fitted = _seeded(estimator, CONFIG.random_state + seed).fit(X_train, y_train)
        scores = _proba(fitted, X_test)
        rows.append(
            {
                "seed": CONFIG.random_state + seed,
                "auc": roc_auc_score(y_test, scores),
                "brier": brier_score_loss(y_test, scores),
                "mean_score": float(scores.mean()),
            }
        )
    return pd.DataFrame(rows)


def subgroup_performance(y_true, y_score, groups: pd.DataFrame) -> pd.DataFrame:
    """Per-subgroup AUC/Brier: an aggregate metric can hide a subgroup the model fails."""
    rows = []
    for attribute in groups.columns:
        for level, chunk in pd.DataFrame(
            {"y": np.asarray(y_true), "s": np.asarray(y_score), "g": groups[attribute].to_numpy()}
        ).groupby("g"):
            if chunk["y"].nunique() < 2 or len(chunk) < 30:
                continue
            rows.append(
                {
                    "attribute": attribute,
                    "group": level,
                    "n": len(chunk),
                    "auc": float(roc_auc_score(chunk["y"], chunk["s"])),
                    "brier": float(brier_score_loss(chunk["y"], chunk["s"])),
                }
            )
    return pd.DataFrame(rows)


# ====================================================== stability as distance between models
#
# Everything above measures how much the *predictions* move. The course defines stability
# one level deeper: two samples drawn from the same population should induce approximately
# the same *model*. A model whose predictions are steady only because its coefficients
# happen to offset each other is not stable in that sense, and the distinction matters when
# the deliverable is a scorecard a caseworker reads rather than a score they consume.


def _coefficient_vector(estimator) -> pd.Series:
    """Named coefficient vector for any of the linear members of the roster.

    Named rather than positional because PLTR's design is *learned*: it harvests its
    threshold terms from the training sample, so two fits on different halves do not
    produce the same columns -- or even the same number of them. Comparing them
    positionally would silently compare unrelated coefficients, and comparing raw arrays
    raises a shape error. Both are wrong; aligning by name is the fix.
    """
    step = estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator
    if hasattr(step, "coefficients_"):  # PLTR exposes the adaptive-lasso parameterisation
        return step.coefficients_.astype(float)
    if hasattr(step, "coef_"):
        values = np.asarray(step.coef_, dtype=float).ravel()
        names = getattr(step, "feature_names_in_", None)
        index = list(names) if names is not None else [f"x{i}" for i in range(len(values))]
        return pd.Series(values, index=index)
    raise AttributeError(f"{type(step).__name__} has no coefficients to compare")


def coefficient_distance(
    estimator, X: pd.DataFrame, y, n_pairs: int = 25, seed: int = CONFIG.random_state
) -> dict[str, float]:
    """||theta_1 - theta_2||_2 between models fitted on independent halves of the data.

    Splitting into disjoint halves rather than bootstrapping is deliberate: the definition
    is about two datasets from the same population, and bootstrap resamples share rows.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n = len(X)

    distances, norms, overlaps = [], [], []
    for i in range(n_pairs):
        order = rng.permutation(n)
        half = n // 2
        first, second = order[:half], order[half : 2 * half]
        theta_1 = _coefficient_vector(
            _seeded(estimator, CONFIG.random_state + i).fit(X.iloc[first], y.iloc[first])
        )
        theta_2 = _coefficient_vector(
            _seeded(estimator, CONFIG.random_state + i).fit(X.iloc[second], y.iloc[second])
        )

        # A term present in one fit and absent from the other has an implicit coefficient
        # of zero there, so the union is the shared parameter space. For a fixed design
        # the union is just the feature list and this is the plain difference; for PLTR it
        # is what makes the comparison meaningful at all.
        union = theta_1.index.union(theta_2.index)
        aligned_1 = theta_1.reindex(union, fill_value=0.0)
        aligned_2 = theta_2.reindex(union, fill_value=0.0)

        distances.append(float(np.linalg.norm(aligned_1 - aligned_2)))
        norms.append(float(np.linalg.norm(np.concatenate([aligned_1, aligned_2])) / np.sqrt(2)))
        intersection = len(theta_1.index.intersection(theta_2.index))
        overlaps.append(intersection / len(union) if len(union) else 1.0)

    distances = np.array(distances)
    return {
        "mean_distance": float(distances.mean()),
        "sd_distance": float(distances.std(ddof=1)),
        "max_distance": float(distances.max()),
        # Scale-free version: the distance as a fraction of the coefficients' own size, so
        # models with different numbers of parameters can be compared.
        "relative_distance": float(distances.mean() / np.mean(norms)),
        # 1.0 when both fits chose the same terms. Below 1.0 the *structure* moved, not just
        # the weights -- a second kind of instability that a coefficient norm alone hides.
        "term_overlap": float(np.mean(overlaps)),
        "n_pairs": n_pairs,
    }


def importance_distance(
    estimator,
    X: pd.DataFrame,
    y,
    importance_fn,
    n_pairs: int = 25,
    seed: int = CONFIG.random_state,
) -> dict[str, float]:
    """||phi(f_1) - phi(f_2)||_2 between importance vectors from independent halves.

    Defined for every family, including the foundation models that have no coefficients at
    all, so this is the stability metric the whole roster can be ranked on. Importances are
    normalised to sum to one first -- otherwise the comparison measures scale, not shape.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n = len(X)

    def normalised(fitted) -> np.ndarray:
        values = np.abs(np.asarray(importance_fn(fitted), dtype=float))
        total = values.sum()
        return values / total if total > 0 else values

    distances = []
    for i in range(n_pairs):
        order = rng.permutation(n)
        half = n // 2
        first, second = order[:half], order[half : 2 * half]
        phi_1 = normalised(
            _seeded(estimator, CONFIG.random_state + i).fit(X.iloc[first], y.iloc[first])
        )
        phi_2 = normalised(
            _seeded(estimator, CONFIG.random_state + i).fit(X.iloc[second], y.iloc[second])
        )
        distances.append(float(np.linalg.norm(phi_1 - phi_2)))

    distances = np.array(distances)
    return {
        "mean_importance_distance": float(distances.mean()),
        "sd_importance_distance": float(distances.std(ddof=1)),
        "max_importance_distance": float(distances.max()),
        "n_pairs": n_pairs,
    }


def stability_constrained_fit(
    X_train: pd.DataFrame,
    y_train,
    X_new: pd.DataFrame,
    y_new,
    X_test: pd.DataFrame,
    y_test,
    lambdas=None,
) -> pd.DataFrame:
    """Refit on new data while penalising departure from the incumbent's coefficients.

    The objective is ``-loglik(theta) + lambda * ||theta - theta_incumbent||^2``. At
    lambda = 0 this is an ordinary refit, free to move anywhere; as lambda grows the new
    model is pulled back toward the model already in production. Sweeping lambda traces the
    trade-off the client actually faces: how much predictive performance does it cost to
    keep the scorecard recognisable between releases?

    The penalised likelihood is minimised directly with L-BFGS rather than bent into a
    library call: the objective is eleven parameters and twenty lines, and a stability
    constraint the client is asked to accept should be legible in the code that produced it.
    """
    from scipy.optimize import minimize
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    lambdas = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0] if lambdas is None else list(lambdas)

    scaler = StandardScaler().fit(X_train)
    incumbent = LogisticRegression(max_iter=5000, random_state=CONFIG.random_state).fit(
        scaler.transform(X_train), np.asarray(y_train)
    )
    theta_prior = incumbent.coef_[0]
    intercept_prior = float(incumbent.intercept_[0])

    design = scaler.transform(X_new)
    y_new = np.asarray(y_new, dtype=float)
    scaled_test = scaler.transform(X_test)
    y_test = np.asarray(y_test)
    n, p = design.shape

    def objective(params, lam):
        theta, intercept = params[:p], params[p]
        z = design @ theta + intercept
        # Mean negative log-likelihood, written through logaddexp so large |z| cannot
        # overflow the way log(sigmoid(z)) does.
        nll = float(np.mean(np.logaddexp(0.0, z) - y_new * z))
        penalty = lam * float(np.sum((theta - theta_prior) ** 2))
        residual = expit(z) - y_new
        grad_theta = design.T @ residual / n + 2 * lam * (theta - theta_prior)
        grad_intercept = float(residual.mean())
        return nll + penalty, np.append(grad_theta, grad_intercept)

    rows = []
    for lam in lambdas:
        start = np.append(theta_prior, intercept_prior)
        result = minimize(objective, start, args=(lam,), jac=True, method="L-BFGS-B")
        theta, intercept = result.x[:p], result.x[p]
        scores = expit(scaled_test @ theta + intercept)
        distance = float(np.linalg.norm(theta - theta_prior))
        rows.append(
            {
                "lambda": lam,
                "auc": float(roc_auc_score(y_test, scores)),
                "coefficient_distance": distance,
                "converged": bool(result.success),
            }
        )

    table = pd.DataFrame(rows)
    # Everything the client cares about is relative to the unconstrained refit: how much
    # AUC was given up, and how much coefficient movement was bought with it.
    free = table.loc[table["lambda"] == 0].iloc[0]
    table["auc_change_pct"] = 100 * (table["auc"] - free["auc"]) / free["auc"]
    table["distance_reduction_pct"] = (
        100
        * (free["coefficient_distance"] - table["coefficient_distance"])
        / free["coefficient_distance"]
    )
    return table
