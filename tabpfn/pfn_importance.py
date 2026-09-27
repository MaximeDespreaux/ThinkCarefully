"""Permutation importance for TabPFN, on two performance metrics, with intervals.

Ported from the earlier study (compas-data, studies/tabpfn/importance.py). It keeps:

* **per-repeat values**, so each feature's importance carries a confidence interval;
* **an economic metric**: permutation importance is defined against any performance measure,
  so it is also run on cost per defendant ("which feature saves the money");
* **the normalised share** %dPM_j = dPM_j / sum_j |dPM_j|, which puts the two metrics on one
  axis.

Permutation importance shuffles one column at a time, so it ignores interactions and
under-credits features that share information with another one (race and priors). XPER
(pfn_xper) attributes that shared credit; the two are expected to disagree.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from compas_scoring.config import CONFIG


def cost_scorer(threshold: float):
    """A scikit-learn scorer returning NEGATIVE cost per defendant.

    Scorers are maximised, and cost is to be minimised, so the sign is flipped: a feature whose
    permutation makes the model more expensive shows up as a positive importance, exactly as it
    does for AUC.

    The threshold is held fixed at the model's deployed operating point. Re-optimising it after
    each shuffle would measure "the cost, plus whatever retuning recovers", which is not the
    quantity being attributed.
    """
    c_fn, c_fp = CONFIG.costs.c_fn, CONFIG.costs.c_fp

    def scorer(estimator, X, y) -> float:
        scores = estimator.predict_proba(X)[:, 1]
        predicted = (scores >= threshold).astype(int)
        y = np.asarray(y)
        false_negatives = int(((predicted == 0) & (y == 1)).sum())
        false_positives = int(((predicted == 1) & (y == 0)).sum())
        return -(false_negatives * c_fn + false_positives * c_fp) / len(y)

    return scorer


def auc_scorer(estimator, X, y) -> float:
    return float(roc_auc_score(np.asarray(y), estimator.predict_proba(X)[:, 1]))


def detailed(
    estimator,
    X: pd.DataFrame,
    y,
    scoring,
    n_repeats: int = 10,
    seed: int = CONFIG.random_state,
) -> pd.DataFrame:
    """Permutation importance keeping every repeat, so the spread is reportable.

    ``sklearn.inspection.permutation_importance`` already returns the full
    (n_features, n_repeats) matrix; keep it instead of averaging it away.
    """
    started = time.perf_counter()
    result = permutation_importance(
        estimator, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    elapsed = time.perf_counter() - started

    matrix = result.importances  # (n_features, n_repeats)
    frame = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance": matrix.mean(axis=1),
            "sd": matrix.std(axis=1, ddof=1),
            "ci_lower": np.quantile(matrix, 0.025, axis=1),
            "ci_upper": np.quantile(matrix, 0.975, axis=1),
            "min": matrix.min(axis=1),
            "max": matrix.max(axis=1),
        }
    )
    # A feature whose interval spans zero has not been shown to matter at all, however large
    # its mean looks. With two repeats that distinction cannot be made.
    frame["interval_excludes_zero"] = (frame["ci_lower"] > 0) | (frame["ci_upper"] < 0)
    frame["share"] = normalised_share(frame["importance"])

    frame.attrs["seconds"] = elapsed
    frame.attrs["n_repeats"] = n_repeats
    # Each shuffled column is a fresh question put to the model, and on a prior-fitted
    # transformer every question re-encodes the whole training context. This is what makes
    # permutation importance expensive here and cheap everywhere else.
    frame.attrs["model_calls"] = len(X.columns) * n_repeats + 1
    frame.attrs["per_repeat"] = matrix
    return frame


def negative_cost(threshold: float):
    """metric(y, score): minus cost per defendant at a fixed threshold (higher is better)."""
    c_fn, c_fp = CONFIG.costs.c_fn, CONFIG.costs.c_fp

    def metric(y, score) -> float:
        y = np.asarray(y)
        predicted = np.asarray(score) >= threshold
        return -(
            ((~predicted) & (y == 1)).sum() * c_fn + (predicted & (y == 0)).sum() * c_fp
        ) / len(y)

    return metric


def permutation_multi(
    estimator,
    X: pd.DataFrame,
    y,
    metrics: dict,
    n_repeats: int = 10,
    seed: int = CONFIG.random_state,
) -> dict[str, pd.DataFrame]:
    """Permutation importance on several metrics from the SAME shuffles, in few model calls.

    ``metrics`` maps a name to ``metric(y, score)`` (higher is better). Each repeat shuffles
    every column once and scores all the shuffled copies in a single ``predict_proba`` call,
    then every metric reads the same scores. That is 1 + n_repeats calls instead of
    1 + n_features * n_repeats per metric -- the difference that matters on TabPFN, where each
    call re-reads the training context. Returns one ``detailed``-shaped frame per metric.
    """
    y = np.asarray(y)
    columns = list(X.columns)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    base = np.asarray(estimator.predict_proba(X))[:, 1]
    baseline = {name: metric(y, base) for name, metric in metrics.items()}
    drops = {name: np.zeros((len(columns), n_repeats)) for name in metrics}
    for r in range(n_repeats):
        shuffled = []
        for column in columns:
            frame = X.copy()
            frame[column] = frame[column].to_numpy()[rng.permutation(len(X))]
            shuffled.append(frame)
        scores = np.asarray(estimator.predict_proba(pd.concat(shuffled, ignore_index=True)))[:, 1]
        scores = scores.reshape(len(columns), len(X))
        for name, metric in metrics.items():
            for j in range(len(columns)):
                drops[name][j, r] = baseline[name] - metric(y, scores[j])
    elapsed = time.perf_counter() - started

    out = {}
    for name, matrix in drops.items():
        frame = pd.DataFrame(
            {
                "feature": columns,
                "importance": matrix.mean(axis=1),
                "sd": matrix.std(axis=1, ddof=1) if n_repeats > 1 else np.nan,
                "ci_lower": np.quantile(matrix, 0.025, axis=1),
                "ci_upper": np.quantile(matrix, 0.975, axis=1),
            }
        )
        frame["interval_excludes_zero"] = (frame["ci_lower"] > 0) | (frame["ci_upper"] < 0)
        frame["share"] = normalised_share(frame["importance"])
        frame["baseline"] = baseline[name]
        frame.attrs.update(seconds=elapsed, n_repeats=n_repeats, model_calls=1 + n_repeats)
        out[name] = frame
    return out


def normalised_share(importance: pd.Series) -> pd.Series:
    """%dPM_j from slide 207: each feature's share of the total attributed performance.

    Absolute values in the denominator, so a feature whose permutation *helps* the model does
    not cancel out one that hurts it.
    """
    total = np.abs(importance).sum()
    return importance / total if total > 0 else importance * 0.0


def compare_to_shap(permutation: pd.DataFrame, shap: pd.DataFrame) -> pd.DataFrame:
    """Permutation importance against KernelSHAP, on one table.

    They answer different questions -- permutation importance asks what the model's
    *performance* owes to a feature, SHAP asks what its *predictions* owe to it -- so
    disagreement is expected rather than alarming. What matters is whether the two orderings
    would lead a reader to different conclusions about which features drive the model.
    """
    merged = permutation[["feature", "importance", "share"]].merge(
        shap.rename(columns={shap.columns[1]: "shap"}), on="feature", how="outer"
    )
    merged["shap_share"] = normalised_share(merged["shap"].abs())
    merged["rank_permutation"] = merged["importance"].abs().rank(ascending=False)
    merged["rank_shap"] = merged["shap"].abs().rank(ascending=False)
    merged["rank_gap"] = (merged["rank_permutation"] - merged["rank_shap"]).abs()
    return merged.sort_values("importance", ascending=False).reset_index(drop=True)


def summary(frame: pd.DataFrame, label: str) -> dict:
    """The numbers the write-up quotes, and the cost of having produced them."""
    ordered = frame.sort_values("importance", ascending=False)
    informative = frame[frame["interval_excludes_zero"]]
    return {
        "metric": label,
        "n_repeats": frame.attrs["n_repeats"],
        "seconds": frame.attrs["seconds"],
        "model_calls": frame.attrs["model_calls"],
        "seconds_per_call": frame.attrs["seconds"] / frame.attrs["model_calls"],
        "top_feature": str(ordered["feature"].iloc[0]),
        "top_importance": float(ordered["importance"].iloc[0]),
        "top_share": float(ordered["share"].iloc[0]),
        "n_features_distinguishable_from_zero": int(len(informative)),
        "n_features": int(len(frame)),
        # Concentration: how much of the attributed performance sits in the leading feature.
        "share_in_top_feature": float(
            np.abs(ordered["importance"].iloc[0]) / np.abs(frame["importance"]).sum()
        ),
    }
