"""Fairness metrics and tests (pfn_fairness), checked on hand-built cases with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pfn_fairness import (
    SEXES,
    chi2_statistical_parity,
    cmh_test,
    disparity_summary,
    equivalence_test,
    false_negative_rate,
    false_positive_rate,
    group_metrics,
    is_degenerate,
    positive_predictive_value,
)


def test_rates_on_a_hand_worked_confusion_matrix():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    assert false_positive_rate(y_true, y_pred) == pytest.approx(0.5)  # 2 of 4 negatives
    assert false_negative_rate(y_true, y_pred) == pytest.approx(0.75)  # 3 of 4 positives
    assert positive_predictive_value(y_true, y_pred) == pytest.approx(1 / 3)


def test_degenerate_operating_points_are_flagged():
    """A model that flags everyone has a zero FPR gap by construction, not by merit."""
    assert is_degenerate(np.ones(100))
    assert is_degenerate(np.zeros(100))
    assert not is_degenerate(np.r_[np.ones(50), np.zeros(50)])


def test_detaining_everyone_looks_perfectly_fair_and_is_flagged():
    """The exact failure mode that made XGBoost appear unbiased at a shared threshold."""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.45, 400)
    sensitive = np.where(np.arange(400) < 200, "African-American", "Caucasian")
    everyone = np.ones(400, dtype=int)

    summary = disparity_summary(y, everyone, sensitive)
    assert summary["fpr_difference"] == pytest.approx(0.0)
    assert summary["degenerate_operating_point"] is True


def test_disparity_summary_signs_follow_the_named_group_order():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    sensitive = np.array(["African-American"] * 4 + ["Caucasian"] * 4)
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0])  # 75% vs 0% false positives
    summary = disparity_summary(y, y_pred, sensitive)
    assert summary["fpr_difference"] == pytest.approx(0.75)
    assert summary["group_a"] == "African-American"


def test_group_metrics_covers_every_group_present():
    y = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    sensitive = np.array(["a", "a", "b", "b", "c", "c"])
    frame = group_metrics(y, y_pred, sensitive)
    assert set(frame.index) == {"a", "b", "c"}
    assert frame["count"].sum() == 6


def test_missing_group_raises_rather_than_silently_returning_nan():
    y = np.array([0, 1, 0, 1])
    sensitive = np.array(["a", "a", "b", "b"])
    with pytest.raises(KeyError):
        disparity_summary(y, y, sensitive, groups=("a", "nonexistent"))


def test_group_metrics_index_keeps_the_attribute_name():
    """Fairlearn labels it `sensitive_feature_0`, which breaks callers that reset_index()."""
    y = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    named = pd.Series(["a", "a", "b", "b", "c", "c"], name="race")
    assert group_metrics(y, y_pred, named).index.name == "race"
    assert "race" in group_metrics(y, y_pred, named).reset_index().columns
    assert group_metrics(y, y_pred, named.to_numpy()).index.name == "group"


def test_chi2_rejects_parity_only_when_selection_rates_differ():
    sensitive = np.array(["African-American"] * 200 + ["Caucasian"] * 200)
    equal = np.tile([1, 0], 200)
    unequal = np.r_[np.ones(150), np.zeros(50), np.ones(50), np.zeros(150)].astype(int)
    assert not chi2_statistical_parity(equal, sensitive)["reject_fairness"]
    assert chi2_statistical_parity(unequal, sensitive)["reject_fairness"]


def test_chi2_works_for_sex_as_well_as_race():
    sensitive = np.array(["Female"] * 100 + ["Male"] * 100)
    y_pred = np.r_[np.zeros(80), np.ones(20), np.ones(80), np.zeros(20)].astype(int)
    assert chi2_statistical_parity(y_pred, sensitive, groups=SEXES)["reject_fairness"]


def test_cmh_sees_through_a_disparity_explained_by_the_strata():
    """Simpson's case: each stratum is fair, the pooled table is not."""

    # Stratum "high": both groups flagged 80%; stratum "low": both 20%. Group a sits mostly
    # in "high", so pooled selection rates differ, but conditional parity holds.
    def block(group, stratum, n, flagged):
        return [(group, stratum, int(i < flagged)) for i in range(n)]

    rows = (
        block("African-American", "high", 100, 80) + block("African-American", "low", 20, 4)
        + block("Caucasian", "high", 20, 16) + block("Caucasian", "low", 100, 20)
    )  # fmt: skip
    frame = pd.DataFrame(rows, columns=["group", "stratum", "flag"])
    assert chi2_statistical_parity(frame["flag"], frame["group"])["reject_fairness"]
    conditional = cmh_test(frame["flag"], frame["group"], frame["stratum"])
    assert not conditional["reject_fairness"]
    assert conditional["n_strata"] == 2


def test_equivalence_certifies_only_inside_the_tolerance():
    rng = np.random.default_rng(0)
    tight = rng.normal(0.0, 0.005, 2000)
    assert equivalence_test(tight, delta=0.05)["certified_fair"]
    wide = rng.normal(0.19, 0.02, 2000)
    result = equivalence_test(wide, delta=0.05)
    assert not result["certified_fair"]
    # The tightest certifiable tolerance covers the whole (1 - 2 alpha) interval.
    assert result["minimum_delta"] == pytest.approx(np.quantile(wide, 0.95))
