"""Test logreg/fairness.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.proportion import proportions_ztest

from logreg import fairness

AA, CA, HI = "African-American", "Caucasian", "Hispanic"


def sample(rows):
    """Expand (race, stratum, decision, outcome, count) rows into aligned arrays."""
    race, strata, y_pred, y_true = [], [], [], []
    for r, s, d, o, n in rows:
        race += [r] * n
        strata += [s] * n
        y_pred += [d] * n
        y_true += [o] * n
    return np.array(race), np.array(strata), np.array(y_pred), np.array(y_true)


@pytest.fixture
def confounded():
    """Flag rates differ by race overall, but are identical inside each stratum.

    African-American defendants sit mostly in the high stratum, where everyone is flagged
    more often -- a disparity that conditioning on the stratum should dissolve.
    """
    rows = [
        (AA, "high", 1, 1, 64), (AA, "high", 0, 0, 16),
        (AA, "low", 1, 1, 4), (AA, "low", 0, 0, 16),
        (CA, "high", 1, 1, 16), (CA, "high", 0, 0, 4),
        (CA, "low", 1, 1, 16), (CA, "low", 0, 0, 64),
    ]  # fmt: skip
    return sample(rows)


@pytest.fixture
def biased():
    """African-American defendants flagged more within every stratum."""
    rows = [
        (AA, "high", 1, 1, 45), (AA, "high", 0, 0, 5),
        (AA, "low", 1, 0, 30), (AA, "low", 0, 0, 20),
        (CA, "high", 1, 1, 25), (CA, "high", 0, 0, 25),
        (CA, "low", 1, 0, 5), (CA, "low", 0, 0, 45),
        (HI, "high", 1, 1, 10), (HI, "high", 0, 0, 10),
    ]  # fmt: skip
    return sample(rows)


class TestComparisonGroups:
    def test_aa_vs_others_keeps_every_row(self):
        in_scope, is_protected = fairness.comparison_groups([AA, CA, HI], "aa_vs_others")
        assert in_scope.tolist() == [True, True, True]
        assert is_protected.tolist() == [True, False, False]

    def test_aa_vs_caucasian_drops_other_groups(self):
        in_scope, is_protected = fairness.comparison_groups([AA, CA, HI], "aa_vs_caucasian")
        assert in_scope.tolist() == [True, True, False]
        assert is_protected.tolist() == [True, False, False]

    def test_unknown_comparison_raises(self):
        with pytest.raises(KeyError, match="Unknown comparison"):
            fairness.comparison_groups([AA], "aa_vs_everyone")

    def test_every_listed_comparison_is_supported(self):
        for comparison in fairness.COMPARISONS:
            fairness.comparison_groups([AA, CA], comparison)


class TestComparisonLabels:
    def test_aa_vs_others_pools_every_other_group(self):
        labels, groups = fairness.comparison_labels([AA, CA, HI], "aa_vs_others")
        assert labels.tolist() == [AA, "all other groups", "all other groups"]
        assert groups == (AA, "all other groups")

    def test_aa_vs_caucasian_keeps_the_original_labels(self):
        labels, groups = fairness.comparison_labels([AA, CA, HI], "aa_vs_caucasian")
        assert labels.tolist() == [AA, CA, HI]
        assert groups == (AA, CA)

    def test_both_groups_are_present_in_the_labels(self):
        for comparison in fairness.COMPARISONS:
            labels, groups = fairness.comparison_labels([AA, CA, HI], comparison)
            assert set(groups) <= set(labels)

    def test_unknown_comparison_raises(self):
        with pytest.raises(KeyError, match="Unknown comparison"):
            fairness.comparison_labels([AA], "aa_vs_everyone")


class TestPriorsBand:
    def test_bands_match_the_shared_analysis(self):
        bands = fairness.priors_band([0, 1, 2, 3, 4, 6, 7, 38])
        assert bands.astype(str).tolist() == ["0", "1", "2-3", "2-3", "4-6", "4-6", "7+", "7+"]

    def test_every_count_lands_in_a_band(self):
        assert fairness.priors_band(np.arange(0, 60)).notna().all()


class TestTwoProportionZtest:
    def test_matches_statsmodels(self):
        result = fairness.two_proportion_ztest(90, 120, 60, 110)
        z, p = proportions_ztest([90, 60], [120, 110])
        assert result["z"] == pytest.approx(z)
        assert result["p_value"] == pytest.approx(p)

    def test_z_squared_is_the_uncorrected_chi_squared(self):
        result = fairness.two_proportion_ztest(90, 120, 60, 110)
        statistic, *_ = chi2_contingency([[90, 30], [60, 50]], correction=False)
        assert result["z"] ** 2 == pytest.approx(statistic)

    def test_reports_the_rates_and_their_signed_difference(self):
        result = fairness.two_proportion_ztest(30, 100, 50, 100)
        assert result["rate_a"] == 0.3 and result["rate_b"] == 0.5
        assert result["difference"] == pytest.approx(-0.2)
        assert result["z"] < 0

    def test_interval_brackets_the_difference_and_widens_with_confidence(self):
        narrow = fairness.two_proportion_ztest(30, 100, 50, 100, alpha=0.10)
        wide = fairness.two_proportion_ztest(30, 100, 50, 100, alpha=0.01)
        assert narrow["ci_lower"] < narrow["difference"] < narrow["ci_upper"]
        assert wide["ci_lower"] < narrow["ci_lower"] and wide["ci_upper"] > narrow["ci_upper"]

    def test_equal_rates_are_not_rejected(self):
        result = fairness.two_proportion_ztest(40, 100, 80, 200)
        assert result["difference"] == 0
        assert result["p_value"] == pytest.approx(1.0)
        assert not result["reject_fairness"]

    def test_large_gap_is_rejected(self):
        assert fairness.two_proportion_ztest(90, 100, 30, 100)["reject_fairness"]

    def test_identical_degenerate_rates_do_not_divide_by_zero(self):
        result = fairness.two_proportion_ztest(50, 50, 20, 20)
        assert result["z"] == 0.0 and result["p_value"] == 1.0

    def test_empty_group_raises(self):
        with pytest.raises(ValueError, match="at least one observation"):
            fairness.two_proportion_ztest(0, 0, 5, 10)


class TestStatisticalParityZtest:
    def test_compares_selection_rates(self, biased):
        race, _, y_pred, _ = biased
        result = fairness.statistical_parity_ztest(y_pred, race, "aa_vs_caucasian")
        assert result["criterion"] == "statistical_parity"
        assert result["metric"] == "selection_rate"
        assert result["rate_a"] == pytest.approx(75 / 100)
        assert result["rate_b"] == pytest.approx(30 / 100)
        assert result["n_a"] == 100 and result["n_b"] == 100

    def test_reference_group_depends_on_the_comparison(self, biased):
        race, _, y_pred, _ = biased
        others = fairness.statistical_parity_ztest(y_pred, race, "aa_vs_others")
        assert others["n_b"] == 120
        assert others["rate_b"] == pytest.approx(40 / 120)


class TestEqualizedOddsZtests:
    def test_returns_the_fpr_then_the_fnr_test(self, biased):
        race, _, y_pred, y_true = biased
        fpr, fnr = fairness.equalized_odds_ztests(y_true, y_pred, race, "aa_vs_caucasian")
        assert (fpr["criterion"], fpr["metric"]) == ("equalized_odds", "fpr")
        assert (fnr["criterion"], fnr["metric"]) == ("equalized_odds", "fnr")

    def test_fpr_is_the_flag_rate_among_non_reoffenders(self, biased):
        race, _, y_pred, y_true = biased
        fpr, _ = fairness.equalized_odds_ztests(y_true, y_pred, race, "aa_vs_caucasian")
        # African-American non-re-offenders: 5 + 30 + 20 = 55, of whom 30 flagged.
        assert fpr["rate_a"] == pytest.approx(30 / 55)
        assert fpr["n_a"] == 55
        # Caucasian non-re-offenders: 25 + 5 + 45 = 75, of whom 5 flagged.
        assert fpr["rate_b"] == pytest.approx(5 / 75)

    def test_fnr_is_the_miss_rate_among_reoffenders(self):
        race, _, y_pred, y_true = sample(
            [
                (AA, "s", 1, 1, 9), (AA, "s", 0, 1, 1), (AA, "s", 0, 0, 5),
                (CA, "s", 1, 1, 6), (CA, "s", 0, 1, 4), (CA, "s", 0, 0, 5),
            ]
        )  # fmt: skip
        _, fnr = fairness.equalized_odds_ztests(y_true, y_pred, race, "aa_vs_caucasian")
        assert fnr["rate_a"] == pytest.approx(0.1)
        assert fnr["rate_b"] == pytest.approx(0.4)

    def test_equal_error_rates_are_not_rejected(self):
        race, _, y_pred, y_true = sample(
            [
                (AA, "s", 1, 0, 2), (AA, "s", 0, 0, 8), (AA, "s", 1, 1, 7), (AA, "s", 0, 1, 3),
                (CA, "s", 1, 0, 4), (CA, "s", 0, 0, 16), (CA, "s", 1, 1, 14), (CA, "s", 0, 1, 6),
            ]
        )  # fmt: skip
        for row in fairness.equalized_odds_ztests(y_true, y_pred, race, "aa_vs_others"):
            assert row["difference"] == pytest.approx(0)
            assert not row["reject_fairness"]


class TestStratumTables:
    def test_orientation_is_protected_first_and_flagged_first(self, biased):
        race, strata, y_pred, _ = biased
        tables, labels = fairness.stratum_tables(y_pred, race, strata, "aa_vs_caucasian")
        high = tables[labels.index("high")]
        assert high.tolist() == [[45, 5], [25, 25]]

    def test_one_table_per_informative_stratum(self, biased):
        race, strata, y_pred, _ = biased
        tables, labels = fairness.stratum_tables(y_pred, race, strata, "aa_vs_others")
        assert sorted(labels) == ["high", "low"]
        assert all(table.shape == (2, 2) for table in tables)

    def test_reference_rows_follow_the_comparison(self, biased):
        race, strata, y_pred, _ = biased
        tables, labels = fairness.stratum_tables(y_pred, race, strata, "aa_vs_others")
        assert tables[labels.index("high")][1].tolist() == [35, 35]  # Caucasian + Hispanic

    def test_strata_missing_a_group_or_a_decision_are_dropped(self):
        race, strata, y_pred, _ = sample(
            [
                (AA, "mixed", 1, 1, 3), (AA, "mixed", 0, 0, 2),
                (CA, "mixed", 1, 1, 1), (CA, "mixed", 0, 0, 4),
                (AA, "one_group", 1, 1, 5), (AA, "one_group", 0, 0, 5),
                (AA, "all_flagged", 1, 1, 5), (CA, "all_flagged", 1, 1, 5),
            ]
        )  # fmt: skip
        _, labels = fairness.stratum_tables(y_pred, race, strata, "aa_vs_caucasian")
        assert labels == ["mixed"]

    def test_a_zero_cell_alone_does_not_drop_the_stratum(self):
        race, strata, y_pred, _ = sample(
            [(AA, "s", 1, 1, 5), (CA, "s", 1, 1, 2), (CA, "s", 0, 0, 3)]
        )
        tables, _ = fairness.stratum_tables(y_pred, race, strata, "aa_vs_caucasian")
        assert tables[0].tolist() == [[5, 0], [2, 3]]


class TestGStatistic:
    def test_matches_scipys_log_likelihood_ratio(self):
        table = np.array([[45, 5], [25, 25]])
        expected, *_ = chi2_contingency(table, correction=False, lambda_="log-likelihood")
        assert fairness.g_statistic(table) == pytest.approx(expected)

    def test_is_zero_under_exact_independence(self):
        assert fairness.g_statistic(np.array([[10, 20], [30, 60]])) == pytest.approx(0)

    def test_handles_a_zero_cell(self):
        value = fairness.g_statistic(np.array([[5, 0], [2, 3]]))
        assert np.isfinite(value) and value > 0


class TestSummedLrTest:
    def test_is_the_sum_of_per_stratum_g_statistics(self, biased):
        race, strata, y_pred, _ = biased
        tables, _ = fairness.stratum_tables(y_pred, race, strata, "aa_vs_caucasian")
        result = fairness.summed_lr_test(y_pred, race, strata, "aa_vs_caucasian")
        assert result["lr_statistic"] == pytest.approx(sum(map(fairness.g_statistic, tables)))
        assert result["lr_dof"] == 2

    def test_rejects_a_disparity_present_in_every_stratum(self, biased):
        race, strata, y_pred, _ = biased
        result = fairness.summed_lr_test(y_pred, race, strata, "aa_vs_caucasian")
        assert result["lr_rejects"] and result["lr_p_value"] < 1e-6

    def test_does_not_reject_a_purely_confounded_disparity(self, confounded):
        race, strata, y_pred, _ = confounded
        assert fairness.statistical_parity_ztest(y_pred, race, "aa_vs_caucasian")["reject_fairness"]
        result = fairness.summed_lr_test(y_pred, race, strata, "aa_vs_caucasian")
        assert result["lr_statistic"] == pytest.approx(0)
        assert not result["lr_rejects"]

    def test_opposite_gaps_add_up_instead_of_cancelling(self):
        """Where CMH sees nothing, the summed LR still sees both disparities."""
        race, strata, y_pred, _ = sample(
            [
                (AA, "a", 1, 1, 40), (AA, "a", 0, 0, 10), (CA, "a", 1, 1, 10), (CA, "a", 0, 0, 40),
                (AA, "b", 1, 1, 10), (AA, "b", 0, 0, 40), (CA, "b", 1, 1, 40), (CA, "b", 0, 0, 10),
            ]
        )  # fmt: skip
        lr = fairness.summed_lr_test(y_pred, race, strata, "aa_vs_caucasian")
        mh = fairness.mantel_haenszel(y_pred, race, strata, "aa_vs_caucasian")
        assert lr["lr_rejects"]
        assert not mh["cmh_rejects"]

    def test_no_informative_stratum_gives_an_empty_result(self):
        race, strata, y_pred, _ = sample([(AA, "s", 1, 1, 5), (CA, "s", 1, 1, 5)])
        result = fairness.summed_lr_test(y_pred, race, strata, "aa_vs_caucasian")
        assert result["lr_dof"] == 0
        assert np.isnan(result["lr_statistic"]) and np.isnan(result["lr_p_value"])
        assert not result["lr_rejects"]


class TestMantelHaenszel:
    def test_matches_statsmodels(self, biased):
        race, strata, y_pred, _ = biased
        tables, _ = fairness.stratum_tables(y_pred, race, strata, "aa_vs_caucasian")
        reference = StratifiedTable(np.dstack(tables))
        result = fairness.mantel_haenszel(y_pred, race, strata, "aa_vs_caucasian")
        assert result["mh_odds_ratio"] == pytest.approx(reference.oddsratio_pooled)
        assert result["cmh_p_value"] == pytest.approx(
            reference.test_null_odds(correction=True).pvalue
        )
        assert result["breslow_day_p_value"] == pytest.approx(reference.test_equal_odds().pvalue)
        assert result["n_strata"] == 2

    def test_odds_ratio_above_one_means_the_protected_group_is_flagged_more(self, biased):
        race, strata, y_pred, _ = biased
        result = fairness.mantel_haenszel(y_pred, race, strata, "aa_vs_caucasian")
        assert result["mh_odds_ratio"] > 1
        assert result["mh_or_lower"] < result["mh_odds_ratio"] < result["mh_or_upper"]
        assert result["cmh_rejects"]

    def test_purely_confounded_disparity_gives_an_odds_ratio_of_one(self, confounded):
        race, strata, y_pred, _ = confounded
        result = fairness.mantel_haenszel(y_pred, race, strata, "aa_vs_caucasian")
        assert result["mh_odds_ratio"] == pytest.approx(1.0)
        assert not result["cmh_rejects"]

    def test_breslow_day_flags_odds_ratios_that_differ_across_strata(self):
        # Odds ratio 16 in stratum a, about 0.67 in b. (Not mirror images: statsmodels'
        # Breslow-Day returns NaN when the pooled odds ratio is exactly 1.)
        race, strata, y_pred, _ = sample(
            [
                (AA, "a", 1, 1, 40), (AA, "a", 0, 0, 10), (CA, "a", 1, 1, 10), (CA, "a", 0, 0, 40),
                (AA, "b", 1, 1, 20), (AA, "b", 0, 0, 30), (CA, "b", 1, 1, 25), (CA, "b", 0, 0, 25),
            ]
        )  # fmt: skip
        result = fairness.mantel_haenszel(y_pred, race, strata, "aa_vs_caucasian")
        assert result["breslow_day_p_value"] < 0.05

    def test_single_stratum_has_no_homogeneity_test(self):
        race, strata, y_pred, _ = sample(
            [(AA, "s", 1, 1, 8), (AA, "s", 0, 0, 2), (CA, "s", 1, 1, 4), (CA, "s", 0, 0, 6)]
        )
        result = fairness.mantel_haenszel(y_pred, race, strata, "aa_vs_caucasian")
        assert result["n_strata"] == 1
        assert result["mh_odds_ratio"] == pytest.approx((8 * 6) / (2 * 4))
        assert np.isnan(result["breslow_day_p_value"])

    def test_no_informative_stratum_gives_an_empty_result(self):
        race, strata, y_pred, _ = sample([(AA, "s", 1, 1, 5), (CA, "s", 1, 1, 5)])
        result = fairness.mantel_haenszel(y_pred, race, strata, "aa_vs_caucasian")
        assert result["n_strata"] == 0
        assert np.isnan(result["mh_odds_ratio"]) and np.isnan(result["cmh_p_value"])
        assert not result["cmh_rejects"]


class TestStratumOddsRatios:
    def test_one_row_per_informative_stratum(self, biased):
        race, strata, y_pred, _ = biased
        table = fairness.stratum_odds_ratios(y_pred, race, strata, "aa_vs_caucasian")
        assert sorted(table["stratum"]) == ["high", "low"]

    def test_rates_and_odds_ratio_match_the_table(self, biased):
        race, strata, y_pred, _ = biased
        table = fairness.stratum_odds_ratios(y_pred, race, strata, "aa_vs_caucasian")
        high = table.set_index("stratum").loc["high"]
        assert high["n_protected"] == 50 and high["n_reference"] == 50
        assert high["flag_rate_protected"] == pytest.approx(0.9)
        assert high["flag_rate_reference"] == pytest.approx(0.5)
        assert high["odds_ratio"] == pytest.approx((45 * 25) / (5 * 25))

    def test_zero_cell_gets_a_finite_corrected_odds_ratio(self):
        race, strata, y_pred, _ = sample(
            [(AA, "s", 1, 1, 5), (CA, "s", 1, 1, 2), (CA, "s", 0, 0, 3)]
        )
        table = fairness.stratum_odds_ratios(y_pred, race, strata, "aa_vs_caucasian")
        assert table["odds_ratio"].iloc[0] == pytest.approx((5.5 * 3.5) / (0.5 * 2.5))

    def test_empty_when_no_stratum_is_informative(self):
        race, strata, y_pred, _ = sample([(AA, "s", 1, 1, 5), (CA, "s", 1, 1, 5)])
        table = fairness.stratum_odds_ratios(y_pred, race, strata, "aa_vs_caucasian")
        assert isinstance(table, pd.DataFrame) and table.empty
