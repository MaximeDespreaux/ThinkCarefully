"""The feature sets are read from pyproject.toml; none may leak the incumbent or target."""

from __future__ import annotations

import pytest

from compas_scoring.config import CONFIG, FEATURE_SET_LABELS
from compas_scoring.data import build_dataset


def test_feature_sets_are_discovered_from_toml():
    assert {"race_aware", "race_blind"} <= set(CONFIG.feature_sets)
    for name in CONFIG.feature_sets:
        assert name in FEATURE_SET_LABELS


def test_adding_a_feature_set_needs_no_code_change():
    """config reads every key under [features] except the race_dummies helper."""
    assert "race_dummies" not in CONFIG.feature_sets
    assert CONFIG.race_dummies


@pytest.mark.parametrize("feature_set", sorted(CONFIG.feature_sets))
def test_no_feature_set_leaks_the_incumbent_or_target(feature_set):
    data = build_dataset(feature_set)
    assert CONFIG.incumbent not in data.X.columns
    assert CONFIG.target not in data.X.columns
