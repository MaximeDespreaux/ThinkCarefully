"""pfn_stability: identical fits must measure as identical, and known gaps as known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pfn_stability import (
    decision_flips,
    importance_distance,
    metric_deltas,
    population_stability_index,
    score_distance,
)


def test_identical_scores_are_at_distance_zero_and_never_flip():
    s = np.random.default_rng(0).uniform(size=200)
    d = score_distance(s, s)
    assert d["l2"] == 0.0 and d["mean_abs_delta"] == 0.0 and d["correlation"] == pytest.approx(1)
    assert decision_flips(s, s, 0.252)["flip_rate"] == 0.0


def test_score_distance_on_a_known_shift():
    a = np.arange(4, dtype=float)
    d = score_distance(a, a + 0.1)
    assert d["l2"] == pytest.approx(0.2)
    assert d["mean_abs_delta"] == pytest.approx(0.1)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        score_distance(np.zeros(3), np.zeros(4))


def test_decision_flips_count_each_direction():
    a = np.array([0.1, 0.3, 0.6, 0.9])
    b = np.array([0.3, 0.1, 0.6, 0.9])
    flips = decision_flips(a, b, 0.252)
    assert flips["n_flips"] == 2
    assert flips["flagged_by_a_only"] == 1 and flips["flagged_by_b_only"] == 1


def test_metric_deltas_are_b_minus_a_on_numeric_columns():
    a = pd.Series({"auc": 0.712, "run": "X1_to_X3"})
    b = pd.Series({"auc": 0.714, "run": "X2_to_X3"})
    frame = metric_deltas(a, b).set_index("metric")
    assert list(frame.index) == ["auc"]
    assert frame.loc["auc", "delta"] == pytest.approx(0.002)


def test_importance_distance_ignores_scale_and_sign():
    a = pd.Series({"priors": 0.2, "age": 0.1, "race": 0.05})
    same = importance_distance(a, -3 * a)
    assert same["l2"] == pytest.approx(0.0) and same["spearman"] == pytest.approx(1.0)
    swapped = importance_distance(a, pd.Series({"priors": 0.05, "age": 0.1, "race": 0.2}))
    assert swapped["l2"] > 0 and swapped["spearman"] == pytest.approx(-1.0)
    assert not swapped["same_top_feature"]


def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    s = rng.uniform(size=5000)
    assert population_stability_index(s, s) == pytest.approx(0.0, abs=1e-9)
    assert population_stability_index(s, np.clip(s + 0.3, 0, 1)) > 0.25
