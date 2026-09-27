"""Stability of TabPFN between two fits, measured on what the model outputs.

TabPFN has no estimated parameters: its weights are pre-trained and frozen, and "fitting" only
stores the training rows. So the distance between two fits is measured on their outputs:

* **predictions** on the same defendants: L2 norm and mean |delta| between the score vectors,
  and the share of defendants whose decision flips at each threshold;
* **performance**: delta of every metric between two runs;
* **explanations**: L2 distance between normalised importance vectors, and rank agreement;
* **score distribution** (when the defendants differ, as in the time-ordered design): PSI.

Ported in part from the earlier study (compas-data, src/compas_scoring/stability.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def score_distance(a, b) -> dict[str, float]:
    """Distance between two score vectors for the same defendants, in the same order."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"score vectors differ in shape: {a.shape} vs {b.shape}")
    delta = a - b
    return {
        "n": int(len(a)),
        "l2": float(np.linalg.norm(delta)),
        # L2 divided by sqrt(n): the same distance on a per-defendant scale, comparable
        # across test sets of different sizes.
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mean_abs_delta": float(np.abs(delta).mean()),
        "max_abs_delta": float(np.abs(delta).max()),
        "correlation": float(np.corrcoef(a, b)[0, 1]),
    }


def decision_flips(a, b, threshold: float) -> dict[str, float]:
    """Share of defendants whose decision differs between two score vectors at one threshold."""
    da = np.asarray(a, dtype=float) >= threshold
    db = np.asarray(b, dtype=float) >= threshold
    return {
        "threshold": float(threshold),
        "flip_rate": float((da != db).mean()),
        "n_flips": int((da != db).sum()),
        "flagged_by_a_only": int((da & ~db).sum()),
        "flagged_by_b_only": int((~da & db).sum()),
    }


def metric_deltas(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """b - a for every numeric metric the two rows share."""
    shared = [c for c in a.index if c in b.index and pd.api.types.is_number(a[c])]
    rows = [
        {"metric": c, "a": float(a[c]), "b": float(b[c]), "delta": float(b[c]) - float(a[c])}
        for c in shared
    ]
    return pd.DataFrame(rows)


def normalised(importance: pd.Series) -> pd.Series:
    """|importance| scaled to sum to one, so two vectors are compared on shape, not scale."""
    values = importance.abs()
    total = values.sum()
    return values / total if total > 0 else values


def importance_distance(a: pd.Series, b: pd.Series, k: int = 3) -> dict[str, float]:
    """||phi_a - phi_b||_2 on normalised importances, plus rank agreement.

    Features are aligned by name; a feature missing from one vector counts as zero there.
    """
    features = sorted(set(a.index) | set(b.index))
    pa = normalised(a.reindex(features).fillna(0.0))
    pb = normalised(b.reindex(features).fillna(0.0))
    top_a = set(pa.sort_values(ascending=False).head(k).index)
    top_b = set(pb.sort_values(ascending=False).head(k).index)
    rho = spearmanr(pa, pb).statistic if len(features) > 1 else float("nan")
    return {
        "n_features": len(features),
        "l2": float(np.linalg.norm(pa - pb)),
        "spearman": float(rho),
        f"top{k}_overlap": len(top_a & top_b) / k,
        "top_feature_a": str(pa.idxmax()),
        "top_feature_b": str(pb.idxmax()),
        "same_top_feature": bool(pa.idxmax() == pb.idxmax()),
    }


def population_stability_index(expected, actual, n_bins: int = 10) -> float:
    """PSI between two score distributions: < 0.10 stable, 0.10-0.25 watch, > 0.25 shift."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
