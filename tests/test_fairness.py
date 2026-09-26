"""Fairness tests and FPDP in xgboost/fairness.py (Hurlin, Pérignon & Saurin method)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

import fairness
from fairness import (
    COMPARISONS,
    EXCLUDED_GROUPS,
    candidate_variables,
    conditional_parity_detail,
    fairness_test,
    fpdp,
    fpdp_all,
    group_rates,
    make_strata,
    parity_table,
    predict_labels,
)


class PriorsModel:
    """Flags everyone with 3+ priors and nobody else: easy to reason about, and it works for
    every feature set because all of them contain Number_of_Priors."""

    def predict_proba(self, X):
        p = np.where(X["Number_of_Priors"].to_numpy() >= 3, 0.9, 0.1)
        return np.column_stack([1 - p, p])


def _data(table):
    """y_pred and protected Series reproducing a 2x2 table [[n(0,F), n(0,T)], [n(1,F), n(1,T)]]."""
    y, d = [], []
    for y_value, row in enumerate(table):
        for d_value, count in zip([False, True], row):
            y += [y_value] * count
            d += [d_value] * count
    return pd.Series(y), pd.Series(d)


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


# --- fairness_test(): the paper's LR test --------------------------------------------------


@pytest.mark.parametrize("stat, lambda_", [("lr", "log-likelihood"), ("pearson", None)])
def test_matches_scipy_chi2_contingency(stat, lambda_):
    table = [[40, 25], [15, 30]]
    y_pred, protected = _data(table)
    reference = chi2_contingency(np.array(table), correction=False, lambda_=lambda_)
    result = fairness_test(y_pred, protected, stat=stat)

    assert result["statistic"] == pytest.approx(reference.statistic)
    assert result["p_value"] == pytest.approx(reference.pvalue)
    assert result["df"] == 1


def test_equal_rates_give_zero_statistic():
    y_pred, protected = _data([[20, 10], [20, 10]])
    result = fairness_test(y_pred, protected)

    assert result["statistic"] == pytest.approx(0)
    assert result["p_value"] == pytest.approx(1)


def test_single_stratum_equals_statistical_parity():
    y_pred, protected = _data([[40, 25], [15, 30]])
    one_stratum = pd.Series("only", index=y_pred.index)

    assert fairness_test(y_pred, protected, one_stratum) == pytest.approx(
        fairness_test(y_pred, protected)
    )


def test_conditional_statistic_sums_strata():
    y1, d1 = _data([[40, 25], [15, 30]])
    y2, d2 = _data([[10, 12], [9, 3]])
    y_pred = pd.concat([y1, y2], ignore_index=True)
    protected = pd.concat([d1, d2], ignore_index=True)
    strata = pd.Series(["a"] * len(y1) + ["b"] * len(y2))
    result = fairness_test(y_pred, protected, strata)

    expected = fairness_test(y1, d1)["statistic"] + fairness_test(y2, d2)["statistic"]
    assert result["statistic"] == pytest.approx(expected)
    assert result["df"] == 2
    assert result["strata_used"] == 2


def test_uninformative_stratum_is_dropped():
    """Nobody flagged in stratum b: its table has an empty row, so it adds nothing to q."""
    y1, d1 = _data([[40, 25], [15, 30]])
    y2, d2 = _data([[10, 12], [0, 0]])
    y_pred = pd.concat([y1, y2], ignore_index=True)
    protected = pd.concat([d1, d2], ignore_index=True)
    strata = pd.Series(["a"] * len(y1) + ["b"] * len(y2))
    result = fairness_test(y_pred, protected, strata)

    assert result["strata_used"] == 1
    assert result["df"] == 1
    assert result["statistic"] == pytest.approx(fairness_test(y1, d1)["statistic"])


def test_constant_predictions_pass_trivially():
    y_pred, protected = _data([[30, 20], [0, 0]])
    result = fairness_test(y_pred, protected)

    assert result == {"statistic": 0.0, "df": 1, "p_value": 1.0, "strata_used": 0}


def test_unknown_statistic_rejected():
    y_pred, protected = _data([[40, 25], [15, 30]])

    with pytest.raises(ValueError, match="lr"):
        fairness_test(y_pred, protected, stat="wald")


# --- strata and group rates ----------------------------------------------------------------


def test_priors_bands(race_aware, strata):
    priors = race_aware.X["Number_of_Priors"]
    band = strata.str.split(" priors").str[0]

    assert (band[priors == 0] == "0").all()
    assert (band[(priors >= 1) & (priors <= 3)] == "1-3").all()
    assert (band[priors >= 4] == "4+").all()


def test_strata_combine_band_and_charge(race_aware, strata):
    assert strata.nunique() == 6
    assert (strata.str.split(" \\| ").str[1] == race_aware.groups["charge_degree"]).all()
    assert strata.index.equals(race_aware.X.index)


def test_group_rates_leave_out_tiny_groups(race_aware, priors_pred):
    rates = group_rates(priors_pred, race_aware, "race")

    assert not set(rates.index) & EXCLUDED_GROUPS
    assert rates["n"].sum() == (~race_aware.groups["race"].isin(EXCLUDED_GROUPS)).sum()


def test_group_rates_values(race_aware, priors_pred):
    rates = group_rates(priors_pred, race_aware, "sex")
    female = race_aware.groups["sex"] == "Female"

    assert rates.loc["Female", "flag_rate"] == pytest.approx(priors_pred[female].mean())
    assert rates.loc["Female", "base_rate"] == pytest.approx(race_aware.y[female].mean())


def test_predict_labels_threshold(race_aware):
    y_pred = predict_labels(PriorsModel(), race_aware)

    assert y_pred.index.equals(race_aware.X.index)
    assert set(y_pred.unique()) <= {0, 1}
    assert (y_pred == (race_aware.X["Number_of_Priors"] >= 3)).all()


# --- parity_table() and the per-stratum detail ---------------------------------------------


def test_comparisons_never_use_excluded_groups():
    for _, protected, reference in COMPARISONS.values():
        assert protected not in EXCLUDED_GROUPS
        assert reference not in EXCLUDED_GROUPS


def test_parity_table_rows_and_rates(race_aware, priors_pred, strata):
    table = parity_table(priors_pred, race_aware, strata)
    race = race_aware.groups["race"]
    row = table.loc["African-American vs Caucasian"]

    assert list(table.index) == list(COMPARISONS)
    assert row["n_protected"] == (race == "African-American").sum()
    assert row["flag_rate_reference"] == pytest.approx(priors_pred[race == "Caucasian"].mean())
    assert row["sp_difference"] == pytest.approx(
        row["flag_rate_protected"] - row["flag_rate_reference"]
    )
    assert row["sp_ratio"] == pytest.approx(
        row["flag_rate_protected"] / row["flag_rate_reference"]
    )


def test_priors_only_model_keeps_only_mixed_strata(race_aware, priors_pred, strata):
    """A model that looks only at priors (3+) treats everyone in a 0 or 4+ band identically,
    so those strata are dropped; only the 1-3 bands (one per charge degree) remain. Its
    unconditional flag rates still differ because the groups have different priors."""
    table = parity_table(priors_pred, race_aware, strata)
    row = table.loc["African-American vs Caucasian"]

    assert row["sp_p_value"] < 0.05
    assert row["csp_df"] <= 2


def test_detail_adds_up_to_conditional_test(race_aware, priors_pred, strata):
    comparison = "Female vs Male"
    detail = conditional_parity_detail(priors_pred, race_aware, strata, comparison)
    row = parity_table(priors_pred, race_aware, strata).loc[comparison]

    assert detail["statistic"].sum() == pytest.approx(row["csp_statistic"])
    assert detail["n_protected"].sum() == row["n_protected"]
    assert detail["p_value"].notna().sum() == row["csp_df"]


# --- FPDP ----------------------------------------------------------------------------------


def test_fpdp_grid_covers_values_outside_compared_groups(race_aware):
    """No African-American or Caucasian defendant has Hispanic = 1, but it must be tried."""
    curve = fpdp(PriorsModel(), race_aware, "Hispanic", "African-American vs Caucasian")

    assert list(curve["value"]) == [0.0, 1.0]


def test_fpdp_row_matches_manual_computation(race_aware):
    comparison = "African-American vs Caucasian"
    curve = fpdp(PriorsModel(), race_aware, "Female", comparison, grid=[1])
    race = race_aware.groups["race"]
    rows = race.isin(["African-American", "Caucasian"])
    y_pred = predict_labels(PriorsModel(), race_aware.subset(race_aware.X.index[rows]))
    expected = fairness_test(y_pred, race[rows] == "African-American")

    assert curve.loc[0, "p_value"] == pytest.approx(expected["p_value"])
    assert curve.loc[0, "flag_rate_protected"] == pytest.approx(
        y_pred[race[rows] == "African-American"].mean()
    )


def test_fpdp_does_not_modify_data(race_aware):
    before = race_aware.X.copy()
    fpdp(PriorsModel(), race_aware, "Number_of_Priors", "Female vs Male", grid=[0, 5])

    pd.testing.assert_frame_equal(race_aware.X, before)


def test_fpdp_flags_degenerate_values(race_aware):
    """Fixing priors fixes the PriorsModel's prediction for everyone: trivially 'fair'."""
    curve = fpdp(PriorsModel(), race_aware, "Number_of_Priors", "Female vs Male", grid=[0, 5])

    assert curve["degenerate"].all()
    assert (curve["p_value"] == 1.0).all()
    assert list(curve["flag_rate_protected"]) == [0.0, 1.0]


def test_fpdp_conditional_uses_strata(race_aware, strata):
    comparison = "African-American vs Caucasian"
    sp = fpdp(PriorsModel(), race_aware, "Female", comparison, grid=[0])
    csp = fpdp(PriorsModel(), race_aware, "Female", comparison, grid=[0], strata=strata)

    assert sp.loc[0, "statistic"] != pytest.approx(csp.loc[0, "statistic"])


def test_fpdp_all_covers_every_feature(race_aware):
    curves = fpdp_all(PriorsModel(), race_aware, "Female vs Male")

    assert list(curves) == list(race_aware.X.columns)


# --- candidate_variables(): Definition 6 ---------------------------------------------------


def _curve(values, p_values, degenerate):
    return pd.DataFrame({"value": values, "p_value": p_values, "degenerate": degenerate})


def test_candidate_when_some_value_passes():
    curves = {
        "driver": _curve([0, 1, 2], [0.001, 0.30, 0.02], [False, False, False]),
        "bystander": _curve([0, 1], [0.001, 0.002], [False, False]),
    }
    result = candidate_variables(curves)

    assert result.loc["driver", "candidate"]
    assert result.loc["driver", "at_value"] == 1
    assert not result.loc["bystander", "candidate"]
    assert list(result.index) == ["driver", "bystander"]


def test_degenerate_values_are_not_evidence():
    curves = {"priors": _curve([0, 1, 9], [1.0, 0.006, 1.0], [True, False, True])}

    assert not candidate_variables(curves).loc["priors", "candidate"]
    assert candidate_variables(curves, include_degenerate=True).loc["priors", "candidate"]


def test_all_degenerate_curve():
    curves = {"priors": _curve([0, 9], [1.0, 1.0], [True, True])}
    result = candidate_variables(curves)

    assert np.isnan(result.loc["priors", "max_p_value"])
    assert not result.loc["priors", "candidate"]


def test_alpha_threshold():
    curves = {"x": _curve([0, 1], [0.001, 0.07], [False, False])}

    assert candidate_variables(curves, alpha=0.05).loc["x", "candidate"]
    assert not candidate_variables(curves, alpha=0.10).loc["x", "candidate"]


# --- main() --------------------------------------------------------------------------------


def test_main_reports_both_models(monkeypatch, capsys):
    monkeypatch.setattr(fairness, "load_model", lambda feature_set: PriorsModel())

    fairness.main()

    out = capsys.readouterr().out
    assert "=== race_aware ===" in out
    assert "=== race_blind ===" in out
    assert out.count("FPDP candidate variables") == 4
