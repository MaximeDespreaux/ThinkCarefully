"""The shared fairness protocol (compas_scoring.fairness), on cases with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from compas_scoring.config import CONFIG
from compas_scoring.data import build_dataset, split_index
from compas_scoring.fairness import (
    candidate_variables,
    cmh_test,
    comparison_rows,
    fairness_table,
    hurlin_test,
    proxy_strata,
    rate_gap,
    thresholds,
)


def _data(table):
    """y_pred and protected for a 2x2 table [[n(0,ref), n(0,prot)], [n(1,ref), n(1,prot)]]."""
    y, d = [], []
    for y_value, row in enumerate(table):
        for d_value, count in zip([False, True], row):
            y += [y_value] * count
            d += [d_value] * count
    return pd.Series(y), pd.Series(d)


# --- the Hurlin likelihood-ratio test (ported from the XGBoost group's tests) --------------


def test_hurlin_matches_scipy_likelihood_ratio():
    table = [[40, 25], [15, 30]]
    y_pred, protected = _data(table)
    reference = chi2_contingency(np.array(table), correction=False, lambda_="log-likelihood")
    result = hurlin_test(y_pred, protected)
    assert result["statistic"] == pytest.approx(reference.statistic)
    assert result["p_value"] == pytest.approx(reference.pvalue)
    assert result["df"] == 1


def test_hurlin_sums_strata_and_drops_uninformative_ones():
    y1, d1 = _data([[40, 25], [15, 30]])
    y2, d2 = _data([[10, 12], [9, 3]])
    y3, d3 = _data([[10, 12], [0, 0]])  # nobody flagged: no information about the null
    y_pred = pd.concat([y1, y2, y3], ignore_index=True)
    protected = pd.concat([d1, d2, d3], ignore_index=True)
    strata = pd.Series(["a"] * len(y1) + ["b"] * len(y2) + ["c"] * len(y3))
    result = hurlin_test(y_pred, protected, strata)
    expected = hurlin_test(y1, d1)["statistic"] + hurlin_test(y2, d2)["statistic"]
    assert result["statistic"] == pytest.approx(expected)
    assert result["df"] == result["strata_used"] == 2


def test_constant_predictions_pass_trivially():
    y_pred, protected = _data([[30, 20], [0, 0]])
    assert hurlin_test(y_pred, protected) == {
        "statistic": 0.0,
        "df": 1,
        "p_value": 1.0,
        "strata_used": 0,
    }


# --- z-test, confidence interval and TOST ---------------------------------------------------


def test_z_test_is_pearson_chi2_without_continuity_correction():
    """z^2 = Pearson chi2: the two tests are the same, so the protocol reports one."""
    table = [[40, 25], [15, 30]]
    y_pred, protected = _data(table)
    result = rate_gap(y_pred, protected)
    pearson = chi2_contingency(np.array(table), correction=False)
    assert result["z"] ** 2 == pytest.approx(pearson.statistic)
    assert result["p_value"] == pytest.approx(pearson.pvalue)
    assert result["gap"] == pytest.approx(30 / 55 - 15 / 55)


def test_tost_certifies_only_a_gap_well_inside_delta():
    rng = np.random.default_rng(0)
    protected = pd.Series(np.r_[np.ones(5000), np.zeros(5000)].astype(bool))
    same = pd.Series(rng.binomial(1, 0.4, 10_000))
    assert rate_gap(same, protected, delta=0.05)["certified_fair"]
    apart = pd.Series(np.r_[rng.binomial(1, 0.6, 5000), rng.binomial(1, 0.4, 5000)])
    result = rate_gap(apart, protected, delta=0.05)
    assert not result["certified_fair"]
    # The tightest certifiable tolerance sits just beyond the observed gap.
    assert result["minimum_delta"] > abs(result["gap"])
    assert rate_gap(apart, protected, delta=result["minimum_delta"] + 1e-9)["certified_fair"]


# --- conditional parity (Simpson's case, ported from the TabPFN group's tests) --------------


def test_conditioning_sees_through_a_disparity_explained_by_the_strata():
    """Each stratum is fair; the pooled table is not, because group mix differs by stratum."""

    def block(group, stratum, n, flagged):
        return [(group, stratum, int(i < flagged)) for i in range(n)]

    rows = (
        block(True, "high", 100, 80) + block(True, "low", 20, 4)
        + block(False, "high", 20, 16) + block(False, "low", 100, 20)
    )  # fmt: skip
    frame = pd.DataFrame(rows, columns=["protected", "stratum", "flag"])
    assert hurlin_test(frame["flag"], frame["protected"])["p_value"] < 0.05
    assert hurlin_test(frame["flag"], frame["protected"], frame["stratum"])["p_value"] > 0.05
    cmh = cmh_test(frame["flag"], frame["protected"], frame["stratum"])
    assert cmh["p_value"] > 0.05
    assert cmh["odds_ratio"] == pytest.approx(1.0)


# --- candidate variables ---------------------------------------------------------------------


def test_candidates_require_fairness_to_be_rejected_on_the_real_data():
    curve = pd.DataFrame({"feature": "f", "value": [0, 1], "p_value": [0.001, 0.6],
                          "degenerate": [False, False]})  # fmt: skip
    assert candidate_variables(curve, baseline_p=0.001)["is_candidate"].item()
    assert not candidate_variables(curve, baseline_p=0.3)["is_candidate"].item()


def test_candidates_ignore_values_where_everyone_gets_the_same_decision():
    curve = pd.DataFrame({"feature": "f", "value": [0, 1], "p_value": [0.001, 1.0],
                          "degenerate": [False, True]})  # fmt: skip
    assert not candidate_variables(curve, baseline_p=0.001)["is_candidate"].item()


# --- the protocol on real data ----------------------------------------------------------------


def test_proxy_strata_are_the_race_proxies_and_ignore_the_feature_set():
    _, test_idx = split_index()
    strata = proxy_strata(test_idx)
    assert strata.str.count(r"\|").eq(2).all()  # priors band | age band | charge degree
    assert set(strata.str.split(" priors").str[0]) == {"0", "1-3", "4+"}
    # Built from the record, not the model's features: identical for any feature set.
    assert strata.equals(proxy_strata(build_dataset("race_proxy_blind").X.loc[test_idx].index))


def test_excluded_groups_never_enter_a_comparison():
    groups = build_dataset().groups
    for comparison in CONFIG.fairness.comparisons:
        rows, _ = comparison_rows(groups, comparison)
        assert not groups.loc[rows, comparison[0]].isin(CONFIG.fairness.excluded_groups).any()


def test_fairness_table_covers_every_threshold_comparison_and_metric():
    _, test_idx = split_index()
    data = build_dataset().subset(test_idx)
    score = data.X["Number_of_Priors"].clip(upper=10) / 10  # any score will do
    table = fairness_table(data.y, score, data.groups, proxy_strata(test_idx))
    metrics = {"statistical_parity", "conditional_statistical_parity", "fpr", "fnr",
               "equalized_odds"}  # fmt: skip
    assert set(table["metric"]) == metrics
    assert set(table["threshold_name"]) == set(thresholds())
    assert len(table) == len(thresholds()) * len(CONFIG.fairness.comparisons) * len(metrics)
    assert table["primary"].sum() == len(thresholds()) * len(metrics)
    assert (table["p_holm"].dropna() >= table.loc[table["p_holm"].notna(), "p_value"]).all()
