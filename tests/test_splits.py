"""The two stability designs: the X1/X2/X3 partition and the time-ordered windows."""

from __future__ import annotations

import pytest

from compas_scoring.config import CONFIG
from compas_scoring.data import (
    FOLLOW_UP_CUTOFF,
    build_dataset,
    load_dated,
    partition,
    partition_index,
    temporal_windows,
)


def test_partition_is_disjoint_and_covers_the_cohort():
    x1, x2, x3 = (set(idx) for idx in partition_index())
    assert not (x1 & x2) and not (x1 & x3) and not (x2 & x3)
    assert len(x1 | x2 | x3) == CONFIG.expected_rows


def test_partition_sizes_and_base_rates_match_the_configuration():
    parts = partition("race_aware")
    for name, fraction in zip(("X1", "X2", "X3"), CONFIG.splits.partition):
        assert len(parts[name]) == pytest.approx(fraction * CONFIG.expected_rows, abs=2)
        assert parts[name].base_rate == pytest.approx(CONFIG.expected_base_rate, abs=0.005)


@pytest.mark.parametrize("attribute", ["race", "sex", "age_band"])
def test_partition_keeps_the_cohorts_protected_group_shares(attribute):
    """Each of X1 / X2 / X3 is a representative sample: every group within 0.5 points."""
    parts = partition("race_aware")
    cohort = build_dataset().groups[attribute].value_counts(normalize=True)
    for name in ("X1", "X2", "X3"):
        share = parts[name].groups[attribute].value_counts(normalize=True)
        gap = (share.reindex(cohort.index, fill_value=0) - cohort).abs().max()
        assert gap < 0.005, f"{name} is {gap:.1%} off the cohort's {attribute} mix"


def test_partition_is_deterministic_and_shared_across_feature_sets():
    assert partition_index() == partition_index()
    aware, blind = partition("race_aware"), partition("race_blind")
    for name in ("X1", "X2", "X3"):
        assert list(aware[name].X.index) == list(blind[name].X.index)


def test_dated_cohort_is_sorted_and_speaks_the_modelling_table_columns():
    df = load_dated()
    assert df["screening_date"].is_monotonic_increasing
    assert set(CONFIG.features("race_aware")) <= set(df.columns)
    # Comparable to, not identical with, the modelling table (see load_dated).
    assert 5000 < len(df) < 6500
    assert 0.28 < df[CONFIG.target].mean() < 0.40
    assert df["screening_date"].max() <= FOLLOW_UP_CUTOFF
    race = df[list(CONFIG.race_dummies)].sum(axis=1)
    assert race.max() <= 1


def test_temporal_windows_train_strictly_on_the_past():
    windows = temporal_windows("race_aware")
    assert len(windows) == len(CONFIG.splits.temporal_windows)
    n = len(load_dated())
    for window, (train_end, test_end) in zip(windows, CONFIG.splits.temporal_windows):
        assert window.train.X.index.max() < window.test.X.index.min()
        assert window.train_period[1] <= window.test_period[0]
        assert len(window.train) == round(train_end * n)
        assert len(window.test) == round(test_end * n) - round(train_end * n)
        assert CONFIG.incumbent not in window.train.X.columns


def test_base_rate_does_not_drift_across_the_time_windows():
    """A censoring artefact (see load_dated) shows up as a base rate climbing over time."""
    for window in temporal_windows("race_aware"):
        assert abs(window.train.base_rate - window.test.base_rate) < 0.05
