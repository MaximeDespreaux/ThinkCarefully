"""Guardrails on the modelling table. These encode the invariants the whole analysis rests on."""

from __future__ import annotations

import pytest

from compas_scoring.config import CONFIG
from compas_scoring.data import build_dataset, group_frame, split_index, train_test


def test_cohort_is_the_standard_propublica_table():
    data = build_dataset("race_aware")
    assert len(data) == CONFIG.expected_rows
    assert data.base_rate == pytest.approx(CONFIG.expected_base_rate, abs=5e-4)


@pytest.mark.parametrize("feature_set", ["race_aware", "race_blind"])
def test_incumbent_never_leaks_into_features(feature_set):
    """score_factor is COMPAS's own output. As a predictor it would leak the benchmark."""
    data = build_dataset(feature_set)
    assert CONFIG.incumbent not in data.X.columns
    assert CONFIG.target not in data.X.columns
    assert list(data.X.columns) == CONFIG.features(feature_set)


def test_race_blind_carries_no_race_column():
    data = build_dataset("race_blind")
    assert not set(CONFIG.race_dummies) & set(data.X.columns)


def test_protected_attributes_stay_outside_the_feature_matrix():
    """A race-blind model must not be reachable from its own feature matrix."""
    data = build_dataset("race_blind")
    assert "race" in data.groups.columns
    assert "race" not in data.X.columns


def test_split_is_disjoint_stratified_and_shared_across_feature_sets():
    train_idx, test_idx = split_index()
    assert not set(train_idx) & set(test_idx)
    assert len(train_idx) + len(test_idx) == CONFIG.expected_rows

    aware_train, aware_test = train_test("race_aware")
    blind_train, blind_test = train_test("race_blind")
    # Identical defendants in both, or the two feature sets are not comparable.
    assert list(aware_train.X.index) == list(blind_train.X.index)
    assert list(aware_test.X.index) == list(blind_test.X.index)
    assert aware_train.base_rate == pytest.approx(CONFIG.expected_base_rate, abs=0.01)


def test_split_is_deterministic():
    assert split_index() == split_index()


def test_race_labels_reconstruct_from_dummies():
    groups = group_frame()
    counts = groups["race"].value_counts()
    # Published composition of the ProPublica cohort.
    assert counts["African-American"] == 3175
    assert counts["Caucasian"] == 2103
    assert counts.sum() == CONFIG.expected_rows
