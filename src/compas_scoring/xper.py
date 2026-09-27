"""XPER: eXplainable PERformance.

Hue, Hurlin, Perignon & Saurin (2026), *Measuring the Driving Forces of Predictive
Performance: Application to Credit Scoring*.

SHAP decomposes a **prediction**. XPER decomposes a **performance metric**::

    PM = phi_0 + sum_j phi_j

where ``phi_0`` is the benchmark value of the metric when no feature is informative and
``phi_j`` is feature j's contribution to the metric actually achieved. The two answer
different questions -- "what moved this defendant's score?" versus "which feature earns the
performance?" -- and they are expected to disagree.

Two things make this worth the compute here:

* It applies to **any** metric, statistical or economic. The brief asks for economic
  performance, and nothing else in this analysis answers "which feature saves the money".
  ``cost_metric`` decomposes dollars per head under the same ``CostModel`` used everywhere.
* Differencing a train decomposition against a test one attributes the train-test gap to
  individual features, which localises overfitting rather than just measuring it.

**Exactness.** With p = 10 features all 2^10 coalitions are enumerated, so the efficiency
property holds to floating-point tolerance rather than approximately -- ``check_efficiency``
asserts it. That is the reason this is implemented here rather than taken from the ``XPER``
PyPI package: a Monte Carlo approximation could not be checked this way.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compas_scoring.config import CONFIG
from compas_scoring.evaluate import CostModel

# Kept small on purpose: the work is n_explain * n_background * 2^p model evaluations, so
# these two numbers are the whole compute budget. 200 x 30 x 1024 = 6.1M rows per model.
N_EXPLAIN = 200
N_BACKGROUND = 30


def auc_metric(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def cost_metric(threshold: float, costs: CostModel | None = None):
    """Cost per head at a fixed operating point -- the economic performance metric.

    The threshold is held fixed across coalitions on purpose: re-optimising it inside each
    coalition would decompose "the metric plus the retuning", which is not the quantity the
    decomposition claims to explain.
    """
    costs = costs or CostModel()

    def metric(y_true, y_score) -> float:
        y_true = np.asarray(y_true)
        predicted = (np.asarray(y_score) >= threshold).astype(int)
        false_negatives = int(((predicted == 0) & (y_true == 1)).sum())
        false_positives = int(((predicted == 1) & (y_true == 0)).sum())
        return (false_negatives * costs.c_fn + false_positives * costs.c_fp) / len(y_true)

    return metric


def _shapley_weights(p: int) -> dict[int, float]:
    """|S|! (p-1-|S|)! / p! -- the fraction of arrival orders with S already seated."""
    return {size: factorial(size) * factorial(p - 1 - size) / factorial(p) for size in range(p)}


def _coalition_scores(
    predict_fn,
    X_explain: pd.DataFrame,
    X_background: pd.DataFrame,
    coalition: tuple[int, ...],
) -> np.ndarray:
    """E over the background of f(x_S, X_Sbar), one averaged score per explained row.

    Features inside the coalition keep the explained row's own values; features outside it
    are replaced by a background row, which is the interventional expectation SHAP and XPER
    both use.
    """
    n_explain, n_background = len(X_explain), len(X_background)
    explain = X_explain.to_numpy(dtype=float)
    background = X_background.to_numpy(dtype=float)

    # One block per background row: tile the explained rows, then overwrite the coalition
    # columns. Built as a single array so the model is called once, not n_explain times.
    stacked = np.tile(background, (n_explain, 1))
    if coalition:
        repeated = np.repeat(explain, n_background, axis=0)
        index = np.array(coalition, dtype=int)
        stacked[:, index] = repeated[:, index]

    frame = pd.DataFrame(stacked, columns=X_explain.columns)
    scores = np.asarray(predict_fn(frame), dtype=float)
    return scores.reshape(n_explain, n_background).mean(axis=1)


def decompose(
    predict_fn,
    X_explain: pd.DataFrame,
    y_explain,
    X_background: pd.DataFrame,
    metric=auc_metric,
    verbose: bool = False,
) -> pd.Series:
    """Exact XPER decomposition. Returns phi_0 plus one contribution per feature.

    ``predict_fn`` takes a DataFrame and returns positive-class probabilities.
    """
    features = list(X_explain.columns)
    p = len(features)
    if p > 16:
        raise ValueError(f"exact enumeration is 2^{p} coalitions; use fewer features")

    y_explain = np.asarray(y_explain)
    weights = _shapley_weights(p)

    # Every coalition's metric value, computed once and reused by the p features whose
    # marginal contribution it appears in.
    values: dict[frozenset[int], float] = {}
    for size in range(p + 1):
        for coalition in combinations(range(p), size):
            scores = _coalition_scores(predict_fn, X_explain, X_background, coalition)
            values[frozenset(coalition)] = float(metric(y_explain, scores))
        if verbose:
            print(f"      coalitions of size {size}: done", flush=True)

    contributions = {}
    for j, name in enumerate(features):
        others = [k for k in range(p) if k != j]
        total = 0.0
        for size in range(p):
            for subset in combinations(others, size):
                key = frozenset(subset)
                total += weights[size] * (values[key | {j}] - values[key])
        contributions[name] = total

    result = pd.Series({"benchmark": values[frozenset()], **contributions})
    result.attrs["metric_value"] = values[frozenset(range(p))]
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
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(X), size=n, replace=False)
    return X.iloc[rows], np.asarray(y)[rows]


def as_table(decompositions: dict[str, pd.Series]) -> pd.DataFrame:
    """Stack per-model decompositions into one long table, with shares of the total."""
    frames = []
    for model, values in decompositions.items():
        table = values.rename("contribution").rename_axis("feature").reset_index()
        table["model"] = model
        table["metric_value"] = values.attrs["metric_value"]
        explained = values.drop("benchmark")
        total = float(np.abs(explained).sum())
        table["share_of_explained"] = table["contribution"].abs() / total
        table.loc[table["feature"] == "benchmark", "share_of_explained"] = np.nan
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def overfitting_decomposition(train: pd.Series, test: pd.Series) -> pd.DataFrame:
    """Per-feature attribution of the train-minus-test gap (slide 201).

    A feature with a large positive gap is one whose apparent contribution does not survive
    out of sample -- that is where the overfitting lives.
    """
    gap = (train - test).rename("gap")
    table = pd.concat([train.rename("train"), test.rename("test"), gap], axis=1)
    return table.rename_axis("feature").reset_index()
