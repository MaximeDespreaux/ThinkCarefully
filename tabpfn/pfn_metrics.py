"""Predictive performance of a scored test set: AUC, accuracy, F1, recall, Type I/II errors.

Reading the confusion matrix as a hypothesis test, per defendant:

    H0: the defendant will NOT re-offend within two years    (flagged = H0 rejected)

    Type I error  (alpha) = false positive rate = FP / (FP + TN)
                   flagging, and possibly detaining, someone who would not have re-offended
    Type II error (beta)  = false negative rate = FN / (FN + TP)
                   releasing someone who then re-offends
    Power         (1 - beta) = recall = true positive rate

Every threshold-dependent number is reported at two operating points:

* ``0.5``        -- the accuracy-maximising default for a calibrated model;
* ``break_even`` -- c_fp / (c_fp + c_fn) from the cost assumptions in pyproject.toml
                   (0.252 with the default $13.5k / $40k). Above it, detaining is cheaper in
                   expectation than releasing. Fixed in advance, so it never looks at the test set.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from compas_scoring.config import CONFIG


def break_even_threshold() -> float:
    costs = CONFIG.costs
    return float(costs.c_fp / (costs.c_fp + costs.c_fn))


def thresholds() -> dict[str, float]:
    return {"0.5": 0.5, "break_even": break_even_threshold()}


def bootstrap_auc_ci(
    y_true, y_score, n_boot: int | None = None, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the AUC."""
    n_boot = n_boot or CONFIG.iterations.bootstrap_ci
    rng = np.random.default_rng(CONFIG.random_state)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)

    draws = []
    while len(draws) < n_boot:
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():  # one class only: redraw
            continue
        draws.append(roc_auc_score(y_true[idx], y_score[idx]))
    lower, upper = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lower), float(upper)


def confusion(y_true, y_pred) -> dict[str, int]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def at_threshold(y_true, y_score, threshold: float) -> dict[str, float]:
    """Every threshold-dependent metric at one operating point."""
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    c = confusion(y_true, y_pred)
    n = len(y_pred)
    precision = _ratio(c["tp"], c["tp"] + c["fp"])
    recall = _ratio(c["tp"], c["tp"] + c["fn"])
    return {
        "threshold": float(threshold),
        "accuracy": (c["tp"] + c["tn"]) / n,
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "specificity": _ratio(c["tn"], c["tn"] + c["fp"]),
        "type_i_error": _ratio(c["fp"], c["fp"] + c["tn"]),
        "type_ii_error": _ratio(c["fn"], c["fn"] + c["tp"]),
        "power": recall,
        "selection_rate": float(y_pred.mean()),
        "cost_per_defendant": (c["fn"] * CONFIG.costs.c_fn + c["fp"] * CONFIG.costs.c_fp) / n,
        **c,
    }


def performance(y_true, y_score, with_ci: bool = True) -> list[dict[str, float]]:
    """One row per operating point; the threshold-free metrics repeat on each row."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    common = {
        "n": len(y_true),
        "base_rate": float(y_true.mean()),
        "auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }
    if with_ci:
        common["auc_ci_low"], common["auc_ci_high"] = bootstrap_auc_ci(y_true, y_score)
    return [
        {**common, "operating_point": name, **at_threshold(y_true, y_score, t)}
        for name, t in thresholds().items()
    ]
