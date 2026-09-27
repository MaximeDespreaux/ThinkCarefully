"""Predictive performance: statistical and economic.

Two rules this module exists to enforce:

* **Never report a bare AUC difference.** On 1,852 test rows the standard error of an AUC
  is roughly 0.013, so a 0.005 gap between two models is noise. Every headline AUC carries
  a bootstrap confidence interval, and every pairwise claim goes through a DeLong test.
* **Never report a single threshold.** The operating point is a client decision driven by
  the cost of a missed re-offence versus an unnecessary detention, so performance is
  reported as a curve over thresholds and a surface over cost ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from compas_scoring.config import CONFIG

# --------------------------------------------------------------------------- statistical


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: the maximum separation between the two score distributions."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """Mean |predicted - observed| across equal-width probability bins, weighted by size.

    Calibration is the property a client actually consumes: "40% risk" has to mean 40%.
    A model can win on AUC and still be unusable if its probabilities are inflated.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_score, bins) - 1, 0, n_bins - 1)

    error = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        error += mask.mean() * abs(y_score[mask].mean() - y_true[mask].mean())
    return float(error)


def statistical_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    """The full statistical panel at one operating point."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_score)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-9, 1 - 1e-9))),
        "ece": expected_calibration_error(y_true, y_score),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "selection_rate": float(y_pred.mean()),
    }


def bootstrap_auc_ci(
    y_true,
    y_score,
    n_boot: int | None = None,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for AUC. Returns (point estimate, lower, upper)."""
    n_boot = n_boot or CONFIG.iterations.bootstrap_ci
    rng = np.random.default_rng(CONFIG.random_state if random_state is None else random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)

    stats_ = np.empty(n_boot)
    drawn = 0
    while drawn < n_boot:
        idx = rng.integers(0, n, n)
        sample = y_true[idx]
        if sample.min() == sample.max():  # degenerate resample, redraw
            continue
        stats_[drawn] = roc_auc_score(sample, y_score[idx])
        drawn += 1

    lower, upper = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(roc_auc_score(y_true, y_score)), float(lower), float(upper)


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(len(x), dtype=float)
    out[order] = ranks
    return out


def delong_test(y_true, score_a, score_b) -> dict[str, float]:
    """DeLong's test for two correlated ROC curves (same rows, two models).

    Correlated is the operative word: the models see identical defendants, so an unpaired
    comparison would badly overstate the uncertainty of their difference.
    """
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)

    pos = y_true == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong needs both classes present")

    # The estimator below indexes the first m entries as the positives, so the scores
    # must be reordered into [positives, negatives] rather than left in row order.
    scores = np.vstack([np.concatenate([s[pos], s[neg]]) for s in (score_a, score_b)])

    tx = np.array([_midrank(s[:m]) for s in scores])
    ty = np.array([_midrank(s[m:]) for s in scores])
    tz = np.array([_midrank(s) for s in scores])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    cov = np.atleast_2d(cov)

    delta = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {
            "auc_a": float(aucs[0]),
            "auc_b": float(aucs[1]),
            "delta": float(delta),
            "z": 0.0,
            "p_value": 1.0,
        }

    z = delta / np.sqrt(var)
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta": float(delta),
        "z": float(z),
        "p_value": float(2 * stats.norm.sf(abs(z))),
    }


# ------------------------------------------------------------------------------ economic


@dataclass(frozen=True)
class CostModel:
    """Expected cost of a pretrial screening decision.

    These are assumptions, not measurements. Monetising detention and re-offence is
    contested, so the deliverable reports the whole sensitivity surface rather than
    presenting one ratio as the answer.
    """

    c_fn: float = CONFIG.costs.c_fn
    c_fp: float = CONFIG.costs.c_fp
    c_tp: float = 0.0
    c_tn: float = 0.0

    @property
    def ratio(self) -> float:
        return self.c_fn / self.c_fp

    def total(self, y_true, y_pred) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        return fn * self.c_fn + fp * self.c_fp + tp * self.c_tp + tn * self.c_tn

    def per_capita(self, y_true, y_pred) -> float:
        return self.total(y_true, y_pred) / len(y_true)

    @property
    def break_even_threshold(self) -> float:
        """The probability at which detaining and releasing cost the same.

        Detain when p*c_tp + (1-p)*c_fp < p*c_fn + (1-p)*c_tn, i.e. above this point.
        With the default 40k/13.5k this sits at 0.252 -- far below the 0.5 a naive
        accuracy-optimising threshold would pick, which is exactly why the economic
        view changes the recommendation.
        """
        numerator = self.c_fp - self.c_tn
        denominator = (self.c_fn - self.c_tp) + (self.c_fp - self.c_tn)
        return float(numerator / denominator)


@dataclass
class CostCurve:
    thresholds: np.ndarray
    cost_per_capita: np.ndarray
    optimal_threshold: float
    optimal_cost: float
    metrics_at_optimum: dict[str, float] = field(default_factory=dict)


def cost_curve(y_true, y_score, costs: CostModel | None = None, n_points: int = 201) -> CostCurve:
    """Expected cost per defendant across the full threshold range."""
    costs = costs or CostModel()
    thresholds = np.linspace(0.0, 1.0, n_points)
    per_capita = np.array(
        [costs.per_capita(y_true, (np.asarray(y_score) >= t).astype(int)) for t in thresholds]
    )
    best = int(np.argmin(per_capita))
    return CostCurve(
        thresholds=thresholds,
        cost_per_capita=per_capita,
        optimal_threshold=float(thresholds[best]),
        optimal_cost=float(per_capita[best]),
        metrics_at_optimum=statistical_metrics(y_true, y_score, float(thresholds[best])),
    )


def baseline_costs(y_true, costs: CostModel | None = None) -> dict[str, float]:
    """Cost floors any model has to beat to be worth deploying at all.

    ``best_trivial`` is the one that matters. Measuring savings against release-all
    alone flatters every model, and at a 3:1 cost ratio it is not even the tougher
    baseline -- detaining everyone is cheaper. A model that cannot beat the better of
    the two is not earning its deployment, whatever its AUC says.
    """
    costs = costs or CostModel()
    y_true = np.asarray(y_true)
    release_all = costs.per_capita(y_true, np.zeros_like(y_true))
    detain_all = costs.per_capita(y_true, np.ones_like(y_true))
    return {
        "release_all": release_all,
        "detain_all": detain_all,
        "best_trivial": min(release_all, detain_all),
        "best_trivial_policy": "detain_all" if detain_all < release_all else "release_all",
    }


def cost_sensitivity(y_true, y_score, ratios=None, c_fp: float | None = None) -> pd.DataFrame:
    """Re-optimise the threshold across cost ratios.

    The question this answers is not "what is the optimal threshold" but "does the model
    ranking survive the client disagreeing with our cost assumptions".
    """
    ratios = np.arange(1.0, 10.5, 0.5) if ratios is None else np.asarray(ratios)
    c_fp = CONFIG.costs.c_fp if c_fp is None else c_fp

    rows = []
    for ratio in ratios:
        costs = CostModel(c_fn=ratio * c_fp, c_fp=c_fp)
        curve = cost_curve(y_true, y_score, costs)
        floors = baseline_costs(y_true, costs)
        rows.append(
            {
                "cost_ratio": float(ratio),
                "optimal_threshold": curve.optimal_threshold,
                "cost_per_capita": curve.optimal_cost,
                "saving_vs_release_all": floors["release_all"] - curve.optimal_cost,
                "saving_vs_best_trivial": floors["best_trivial"] - curve.optimal_cost,
                "best_trivial_policy": floors["best_trivial_policy"],
                "selection_rate": curve.metrics_at_optimum["selection_rate"],
                "recall": curve.metrics_at_optimum["recall"],
            }
        )
    return pd.DataFrame(rows)
