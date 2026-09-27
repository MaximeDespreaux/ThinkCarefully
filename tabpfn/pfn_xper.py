"""XPER: eXplainable PERformance (Hue, Hurlin, Perignon & Saurin, 2026).

SHAP decomposes a **prediction**. XPER decomposes a **performance metric**::

    PM = phi_0 + sum_j phi_j

where ``phi_0`` is the metric when no feature is informative and ``phi_j`` is feature j's
contribution to the metric actually achieved. It applies to any metric, so it also answers
"which feature saves the money" (``cost_metric``).

With p <= 16 features every coalition is enumerated, so the efficiency property holds to
floating point, and ``check_efficiency`` asserts it.

Ported from the earlier study (compas-data, src/compas_scoring/xper.py), with one change for
TabPFN: every ``predict_fn`` call pays a fixed cost (it re-reads the training context), so the
coalitions are **batched** into calls of about ``batch_rows`` rows instead of one call per
coalition. The scores are the same; only the number of calls changes.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compas_scoring.config import CONFIG

# Rows per predict_fn call. 24k rows ~ 8 coalitions of 100 x 30, a call size TabPFN handled
# in ~30 s in the earlier study's timings, without exhausting memory.
BATCH_ROWS = 24_000


def auc_metric(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def cost_metric(threshold: float):
    """Cost per defendant at a fixed operating point, with the configured costs.

    The threshold stays fixed across coalitions: re-optimising it inside each coalition would
    decompose "the metric plus the retuning", which is not the quantity being attributed.
    """
    c_fn, c_fp = CONFIG.costs.c_fn, CONFIG.costs.c_fp

    def metric(y_true, y_score) -> float:
        y_true = np.asarray(y_true)
        predicted = (np.asarray(y_score) >= threshold).astype(int)
        false_negatives = int(((predicted == 0) & (y_true == 1)).sum())
        false_positives = int(((predicted == 1) & (y_true == 0)).sum())
        return (false_negatives * c_fn + false_positives * c_fp) / len(y_true)

    return metric


def _shapley_weights(p: int) -> dict[int, float]:
    """|S|! (p-1-|S|)! / p!: the fraction of arrival orders with S already seated."""
    return {size: factorial(size) * factorial(p - 1 - size) / factorial(p) for size in range(p)}


def _coalition_rows(explain: np.ndarray, background: np.ndarray, coalition) -> np.ndarray:
    """The rows whose mean score is E_background f(x_S, X_Sbar), for each explained row.

    Coalition columns keep the explained row's values; the others come from a background row
    (the interventional expectation SHAP and XPER both use).
    """
    n_background = len(background)
    stacked = np.tile(background, (len(explain), 1))
    if coalition:
        index = np.array(coalition, dtype=int)
        stacked[:, index] = np.repeat(explain, n_background, axis=0)[:, index]
    return stacked


def decompose(
    predict_fn,
    X_explain: pd.DataFrame,
    y_explain,
    X_background: pd.DataFrame,
    metric=auc_metric,
    batch_rows: int = BATCH_ROWS,
    progress=None,
) -> pd.Series:
    """Exact XPER decomposition. Returns phi_0 ("benchmark") plus one value per feature.

    ``predict_fn`` takes a DataFrame and returns positive-class probabilities. ``progress``, if
    given, is called with the number of coalitions just scored.
    """
    features = list(X_explain.columns)
    p = len(features)
    if p > 16:
        raise ValueError(f"exact enumeration is 2^{p} coalitions; use fewer features")

    y_explain = np.asarray(y_explain)
    explain = X_explain.to_numpy(dtype=float)
    background = X_background.to_numpy(dtype=float)
    n_explain, n_background = len(explain), len(background)
    rows_per_coalition = n_explain * n_background
    per_call = max(1, batch_rows // rows_per_coalition)

    coalitions = [c for size in range(p + 1) for c in combinations(range(p), size)]
    values: dict[frozenset[int], float] = {}
    for start in range(0, len(coalitions), per_call):
        chunk = coalitions[start : start + per_call]
        stacked = np.vstack([_coalition_rows(explain, background, c) for c in chunk])
        scores = np.asarray(predict_fn(pd.DataFrame(stacked, columns=features)), dtype=float)
        for i, coalition in enumerate(chunk):
            block = scores[i * rows_per_coalition : (i + 1) * rows_per_coalition]
            averaged = block.reshape(n_explain, n_background).mean(axis=1)
            values[frozenset(coalition)] = float(metric(y_explain, averaged))
        if progress is not None:
            progress(len(chunk))

    weights = _shapley_weights(p)
    contributions = {}
    for j, name in enumerate(features):
        others = [k for k in range(p) if k != j]
        contributions[name] = sum(
            weights[size] * (values[frozenset(s) | {j}] - values[frozenset(s)])
            for size in range(p)
            for s in combinations(others, size)
        )

    result = pd.Series({"benchmark": values[frozenset()], **contributions})
    result.attrs["metric_value"] = values[frozenset(range(p))]
    result.attrs["model_calls"] = -(-len(coalitions) // per_call)
    return result


def check_efficiency(decomposition: pd.Series, tolerance: float = 1e-8) -> float:
    """phi_0 + sum phi_j must equal the metric on the full feature set. Returns the error."""
    reconstructed = float(decomposition.sum())
    error = abs(reconstructed - decomposition.attrs["metric_value"])
    if error > tolerance:
        raise AssertionError(
            f"XPER efficiency violated: phi_0 + sum(phi_j) = {reconstructed:.12f} "
            f"but PM = {decomposition.attrs['metric_value']:.12f} (error {error:.2e})"
        )
    return error


def sample(X: pd.DataFrame, y, n: int, seed: int = CONFIG.random_state):
    """A reproducible subsample, or the whole frame when it is already small enough."""
    if len(X) <= n:
        return X, np.asarray(y)
    rows = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
    return X.iloc[rows], np.asarray(y)[rows]


def as_table(decomposition: pd.Series) -> pd.DataFrame:
    """One decomposition as a long table, with each feature's share of the explained part."""
    table = decomposition.rename("contribution").rename_axis("feature").reset_index()
    table["metric_value"] = decomposition.attrs["metric_value"]
    total = float(decomposition.drop("benchmark").abs().sum())
    table["share_of_explained"] = table["contribution"].abs() / total if total else np.nan
    table.loc[table["feature"] == "benchmark", "share_of_explained"] = np.nan
    return table
