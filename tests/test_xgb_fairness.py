"""Fairness tests, FPDP and mitigation in xgboost/fairness.py (African-American vs Rest)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import StratifiedTable
# Aliased: a bare `test_proportions_2indep` import would be collected by pytest as a test.
from statsmodels.stats.proportion import confint_proportions_2indep
from statsmodels.stats.proportion import test_proportions_2indep as sm_two_proportion_test

import fairness
from fairness import (
    PROTECTED_GROUP,
    apply_fixed_value,
    candidate_variables,
    conditional_statistical_parity,
    equalized_odds_test,
    evaluate_fairness,
    fpdp,
    fpdp_all,
    group_rates,
    make_strata,
    mitigate_by_reestimation,
    mitigate_by_substitution,
    predict_labels,
    predict_scores,
    race_group,
    refit_without_feature,
    statistical_parity_test,
)


class PriorsModel:
    """Flags everyone with 3+ priors and nobody else: easy to reason about by hand, and it
    works for every feature set since all of them contain Number_of_Priors."""

    def predict_proba(self, X):
        p = np.where(X["Number_of_Priors"].to_numpy() >= 3, 0.9, 0.1)
        return np.column_stack([1 - p, p])


@pytest.fixture(scope="module")
def race_aware(split):
    _, test = split
    return test


@pytest.fixture(scope="module")
def strata(race_aware):
    return make_strata(race_aware)


@pytest.fixture(scope="module")
def priors_pred(race_aware):
    return predict_labels(PriorsModel(), race_aware)


# --- race_group() / group_rates() -----------------------------------------------------------


def test_race_group_matches_label(race_aware):
    protected = race_group(race_aware)

    assert protected.sum() == (race_aware.groups["race"] == PROTECTED_GROUP).sum()
    assert (~protected).sum() == (race_aware.groups["race"] != PROTECTED_GROUP).sum()
    assert protected.index.equals(race_aware.X.index)


def test_group_rates_covers_everyone(race_aware, priors_pred):
    rates = group_rates(priors_pred, race_aware)

    assert list(rates.index) == [PROTECTED_GROUP, "Rest"]
    assert rates["n"].sum() == len(race_aware)


def test_group_rates_values(race_aware, priors_pred):
    rates = group_rates(priors_pred, race_aware)
    protected = race_group(race_aware)

    assert rates.loc[PROTECTED_GROUP, "flag_rate"] == pytest.approx(priors_pred[protected].mean())
    assert rates.loc["Rest", "base_rate"] == pytest.approx(race_aware.y[~protected].mean())


# --- statistical_parity_test(): two-proportion z-test ----------------------------------------


def test_sp_matches_statsmodels_directly():
    y_pred = pd.Series([1] * 40 + [0] * 60 + [1] * 20 + [0] * 80)
    protected = pd.Series([True] * 100 + [False] * 100)
    result = statistical_parity_test(y_pred, protected)

    reference = sm_two_proportion_test(40, 100, 20, 100, compare="diff")
    ci = confint_proportions_2indep(40, 100, 20, 100, compare="diff")
    assert result["z_statistic"] == pytest.approx(reference.statistic)
    assert result["p_value"] == pytest.approx(reference.pvalue)
    assert result["difference"] == pytest.approx(0.2)
    assert result["ci_low"] == pytest.approx(ci[0])
    assert result["ci_high"] == pytest.approx(ci[1])


def test_sp_equal_rates_are_not_significant():
    y_pred = pd.Series([1] * 30 + [0] * 70 + [1] * 30 + [0] * 70)
    protected = pd.Series([True] * 100 + [False] * 100)
    result = statistical_parity_test(y_pred, protected)

    assert result["difference"] == pytest.approx(0)
    assert result["p_value"] == pytest.approx(1, abs=1e-9)
    assert result["ci_low"] < 0 < result["ci_high"]


# --- equalized_odds_test(): FPR and FNR z-tests -----------------------------------------------


def test_equalized_odds_isolates_fpr_and_fnr():
    """4 protected, 4 rest: y=0 rows split for FPR, y=1 rows split for FNR, independently."""
    y_true = pd.Series([0, 0, 1, 1] * 2)
    protected = pd.Series([True] * 4 + [False] * 4)
    # Protected: flagged on both negatives (FPR=1) and on neither positive (FNR=1).
    # Rest: flagged on neither negative (FPR=0) and on both positives (FNR=0).
    y_pred = pd.Series([1, 1, 0, 0, 0, 0, 1, 1])
    result = equalized_odds_test(y_pred, y_true, protected)

    assert result["fpr"]["rate_protected"] == pytest.approx(1.0)
    assert result["fpr"]["rate_rest"] == pytest.approx(0.0)
    assert result["fnr"]["rate_protected"] == pytest.approx(1.0)
    assert result["fnr"]["rate_rest"] == pytest.approx(0.0)


def test_equalized_odds_matches_manual_rates(race_aware, priors_pred):
    result = equalized_odds_test(priors_pred, race_aware.y, race_group(race_aware))
    protected = race_group(race_aware)
    negative = race_aware.y == 0

    manual_fpr_protected = priors_pred[negative & protected].mean()
    assert result["fpr"]["rate_protected"] == pytest.approx(manual_fpr_protected)


# --- conditional_statistical_parity(): Hurlin / CMH / MH -------------------------------------


def test_csp_matches_manual_cmh_and_stratified_table():
    """Regression test for the stratum-table orientation: rows = (protected, rest), columns =
    (flagged, not) -- get this backwards and CMH/MH silently give a wrong but plausible-looking
    number."""
    rng = np.random.default_rng(0)
    n = 300
    stratum = pd.Series(rng.choice(["a", "b"], n))
    protected = pd.Series(rng.random(n) < 0.4)
    base = np.where(stratum == "a", 0.3, 0.6)
    p_flag = base + np.where(protected, 0.25, 0.0)
    y_pred = pd.Series((rng.random(n) < p_flag).astype(int))

    result = conditional_statistical_parity(y_pred, protected, stratum)

    tables = []
    for name in ["a", "b"]:
        mask = stratum == name
        n1, n0 = int((mask & protected).sum()), int((mask & ~protected).sum())
        c1, c0 = int(y_pred[mask & protected].sum()), int(y_pred[mask & ~protected].sum())
        tables.append(np.array([[c1, n1 - c1], [c0, n0 - c0]], dtype=float))
    reference = StratifiedTable(tables)

    assert result["strata_used"] == 2
    assert result["cmh_statistic"] == pytest.approx(reference.test_null_odds().statistic)
    assert result["mh_odds_ratio"] == pytest.approx(reference.oddsratio_pooled)
    # Protected group is flagged more in both strata: odds ratio must be > 1, not < 1.
    assert result["mh_odds_ratio"] > 1


def test_csp_drops_degenerate_strata():
    """A stratum where the model flags nobody (or everybody) carries no information about
    the null and must be dropped from both the Hurlin sum and the CMH/MH tables."""
    rng = np.random.default_rng(1)
    n = 200
    protected = pd.Series(rng.random(n) < 0.4)
    stratum = pd.Series(["informative"] * 100 + ["degenerate"] * 100)
    y_pred = pd.Series([int(rng.random() < 0.5) for _ in range(100)] + [0] * 100)

    result = conditional_statistical_parity(y_pred, protected, stratum)

    assert result["strata_used"] == 1
    assert result["strata"] == ["informative"]
    assert result["hurlin_df"] == 1


def test_csp_drops_stratum_missing_one_group():
    """A stratum with no protected (or no Rest) members can't say anything about H0 either,
    distinct from the all-flagged/all-unflagged case above."""
    protected = pd.Series([True, True, False, False, False, False])
    stratum = pd.Series(["mixed", "mixed", "mixed", "mixed", "protected_only", "protected_only"])
    y_pred = pd.Series([1, 0, 1, 0, 1, 1])
    protected.iloc[4:] = True  # "protected_only" stratum: both rows are protected, no Rest

    result = conditional_statistical_parity(y_pred, protected, stratum)

    assert result["strata_used"] == 1
    assert result["strata"] == ["mixed"]


def test_csp_no_usable_strata_returns_nan():
    protected = pd.Series([True, False, True, False])
    stratum = pd.Series(["only"] * 4)
    y_pred = pd.Series([0, 0, 0, 0])  # nobody flagged anywhere
    result = conditional_statistical_parity(y_pred, protected, stratum)

    assert result["strata_used"] == 0
    assert result["hurlin_statistic"] == 0
    assert np.isnan(result["cmh_statistic"])
    assert np.isnan(result["mh_odds_ratio"])


def test_csp_single_stratum_hurlin_equals_sp_chi2():
    """With one stratum, Hurlin's LR statistic is an ordinary 2x2 independence test: its
    p-value should closely track the (different but related) SP z-test's p-value in sign."""
    y_pred, protected = pd.Series([1] * 40 + [0] * 60 + [1] * 15 + [0] * 85), pd.Series(
        [True] * 100 + [False] * 100
    )
    stratum = pd.Series(["only"] * 200)
    csp = conditional_statistical_parity(y_pred, protected, stratum)
    sp = statistical_parity_test(y_pred, protected)

    assert csp["hurlin_df"] == 1
    assert (csp["hurlin_p_value"] < 0.05) == (sp["p_value"] < 0.05)


def test_breslow_day_runs_without_warning_on_sparse_tables(race_aware, priors_pred, strata):
    """shift_zeros must be on: some real strata have very few flagged/unflagged cells."""
    result = conditional_statistical_parity(priors_pred, race_group(race_aware), strata)

    assert not np.isnan(result["breslow_day_p_value"])
    assert 0 <= result["breslow_day_p_value"] <= 1


# --- FPDP and candidate_variables() ----------------------------------------------------------


def test_fpdp_grid_covers_all_observed_values(race_aware):
    curve = fpdp(PriorsModel(), race_aware, "Hispanic")

    assert list(curve["value"]) == [0.0, 1.0]


def test_fpdp_does_not_modify_data(race_aware):
    before = race_aware.X.copy()
    fpdp(PriorsModel(), race_aware, "Number_of_Priors", grid=[0, 5])

    pd.testing.assert_frame_equal(race_aware.X, before)


def test_fpdp_flags_degenerate_values(race_aware):
    """Both groups get the same constant prediction at these values, so the test must not
    reject -- though with unequal group sizes (956 vs 896), the Agresti-Caffo boundary
    adjustment means the p-value lands very high rather than exactly 1.0."""
    curve = fpdp(PriorsModel(), race_aware, "Number_of_Priors", grid=[0, 5])

    assert curve["degenerate"].all()
    assert (curve["p_value"] > 0.9).all()


def test_apply_fixed_value_matches_fpdp_row(race_aware):
    protected = race_group(race_aware)
    y_pred = apply_fixed_value(PriorsModel(), race_aware, "Number_of_Priors", 5)
    curve = fpdp(PriorsModel(), race_aware, "Number_of_Priors", grid=[5])

    assert y_pred[protected].mean() == pytest.approx(curve.loc[0, "flag_rate_protected"])
    assert y_pred[~protected].mean() == pytest.approx(curve.loc[0, "flag_rate_rest"])


def test_fpdp_all_covers_every_feature(race_aware):
    curves = fpdp_all(PriorsModel(), race_aware)

    assert list(curves) == list(race_aware.X.columns)


def test_candidate_when_some_value_passes():
    curve_driver = pd.DataFrame(
        {"value": [0, 1, 2], "p_value": [0.001, 0.30, 0.02], "degenerate": [False, False, False]}
    )
    curve_bystander = pd.DataFrame(
        {"value": [0, 1], "p_value": [0.001, 0.002], "degenerate": [False, False]}
    )
    result = candidate_variables({"driver": curve_driver, "bystander": curve_bystander})

    assert result.loc["driver", "candidate"]
    assert result.loc["driver", "at_value"] == 1
    assert not result.loc["bystander", "candidate"]


def test_degenerate_values_are_not_evidence():
    curve = pd.DataFrame({"value": [0, 1, 9], "p_value": [1.0, 0.006, 1.0], "degenerate": [True, False, True]})

    assert not candidate_variables({"priors": curve}).loc["priors", "candidate"]
    assert candidate_variables({"priors": curve}, include_degenerate=True).loc["priors", "candidate"]


# --- Step 3: mitigation -----------------------------------------------------------------------


def test_refit_without_feature_drops_the_column(small_train):
    model = refit_without_feature(small_train, "Number_of_Priors", n_iter=3)

    assert "Number_of_Priors" not in model.get_booster().feature_names
    assert model.n_features_in_ == small_train.X.shape[1] - 1


@pytest.mark.slow
def test_mitigate_by_reestimation_uses_dropped_feature_model(small_train, small_test):
    strata = make_strata(small_test)
    result = mitigate_by_reestimation(small_train, small_test, "Number_of_Priors", strata, n_iter=3)

    assert result["method"] == "re-estimation"
    assert result["feature"] == "Number_of_Priors"
    assert set(result) >= {"sp", "csp", "eo"}


def test_evaluate_fairness_auc_is_optional(race_aware, priors_pred, strata):
    without = evaluate_fairness(priors_pred, race_aware, strata)
    assert "auc" not in without

    y_score = predict_scores(PriorsModel(), race_aware)
    with_auc = evaluate_fairness(priors_pred, race_aware, strata, y_score=y_score)
    assert with_auc["auc"] == pytest.approx(roc_auc_score(race_aware.y, y_score))


def test_mitigate_by_reestimation_reports_auc(small_train, small_test):
    strata = make_strata(small_test)
    result = mitigate_by_reestimation(small_train, small_test, "Number_of_Priors", strata, n_iter=3)

    assert 0 <= result["auc"] <= 1


def test_mitigate_by_substitution_reports_auc(small_model, small_test):
    strata = make_strata(small_test)
    result = mitigate_by_substitution(small_model, small_test, "Number_of_Priors", 5, strata)

    y_score = fairness._fixed_value_scores(small_model, small_test, "Number_of_Priors", 5)
    assert result["auc"] == pytest.approx(roc_auc_score(small_test.y, y_score))


def test_mitigate_by_substitution_matches_fpdp(small_model, small_test):
    strata = make_strata(small_test)
    result = mitigate_by_substitution(small_model, small_test, "Number_of_Priors", 5, strata)

    y_pred = apply_fixed_value(small_model, small_test, "Number_of_Priors", 5)
    expected = evaluate_fairness(y_pred, small_test, strata)
    assert result["sp"]["p_value"] == pytest.approx(expected["sp"]["p_value"])
    assert result["value"] == 5


# --- main() ------------------------------------------------------------------------------------


class ConstantModel:
    """Ignores its input entirely: safe to call with any column set, including one missing a
    column another fake model would have required (Step 3's re-estimation drops a column)."""

    def predict_proba(self, X):
        p = np.full(len(X), 0.3)
        return np.column_stack([1 - p, p])


def test_main_runs_all_three_steps(monkeypatch, capsys):
    monkeypatch.setattr(fairness, "load_model", lambda feature_set: ConstantModel())
    monkeypatch.setattr(
        fairness, "tune_xgboost", lambda X, y, n_iter=40: type("S", (), {"best_estimator_": ConstantModel()})()
    )

    fairness.main()

    out = capsys.readouterr().out
    assert "=== race_aware ===" in out
    assert "=== race_blind ===" in out
    assert "FPDP candidate variables" in out
