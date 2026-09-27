"""TabPFN's adapter to the shared fairness protocol, on the committed predictions (no fit)."""

from __future__ import annotations

import pytest
from pfn_fairness import cohort, feature_matrix, load_predictions, protocol_table

from compas_scoring.config import CONFIG
from compas_scoring.fairness import thresholds


@pytest.fixture(scope="module")
def holdout():
    return protocol_table("holdout", "race_aware")


def test_tabpfn_and_the_compas_tool_get_the_identical_protocol(holdout):
    per_model = holdout.groupby("model").size()
    assert set(per_model.index) == {"TabPFN", "COMPAS tool"}
    assert per_model.nunique() == 1  # same rows for both
    expected = len(thresholds()) * len(CONFIG.fairness.comparisons) * 5
    assert per_model.iloc[0] == expected


def test_primary_gap_reads_protected_minus_reference(holdout):
    row = holdout.query(
        "model == 'TabPFN' and primary and metric == 'statistical_parity' "
        "and threshold_name == '0.5'"
    ).iloc[0]
    assert row["comparison"] == "African-American vs Caucasian"
    assert row["gap"] == pytest.approx(row["rate_protected"] - row["rate_reference"])


def test_time_ordered_runs_are_conditioned_on_the_dated_cohort():
    assert cohort("temporal_2") == "dated" and cohort("holdout") == "modelling"
    table = protocol_table("temporal_2", "race_blind")
    assert table["p_value"].notna().any()


def test_fpdp_perturbs_exactly_the_features_the_model_saw():
    rows = load_predictions("holdout", "race_proxy_blind").index[:5]
    columns = feature_matrix("holdout", "race_proxy_blind", rows).columns
    assert list(columns) == CONFIG.features("race_proxy_blind")
