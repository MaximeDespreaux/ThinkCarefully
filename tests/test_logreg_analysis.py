"""Test for logreg/analysis.py"""

from __future__ import annotations

import json
import runpy
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import compas_scoring.config as config_module
from compas_scoring.config import CONFIG
from compas_scoring.data import train_test
from compas_scoring.evaluate import CostModel, baseline_costs
from logreg import analysis, fairness
from logreg import train as train_module
from logreg.features import engineered
from logreg.model import load

FEATURE_SETS = analysis.FEATURE_SETS  # the sets the analysis runs on, not every configured one
STAGE_NAMES = [
    "performance",
    "interpretability",
    "explanations",
    "whitebox",
    "stability",
    "fairness",
    "fairness_groups",
    "fairness_inference",
    "fpdp",
    "xper",
    "xper_force",
]
COMPARISONS = tuple(fairness.COMPARISONS)

# Enough draws for every statistic to be defined; few enough for the suite to take seconds.
SMALL_BUDGET = {
    "cv_repeats": 1,
    "bootstrap_refits": 3,
    "fairness_bootstrap": 20,
    "distance_pairs": 2,
    "distance_rows": 100,
    "permutation_repeats": 2,
    "n_seeds": 2,
    "perturbation_trials": 2,
    "learning_curve_repeats": 1,
    "ice_curves": 5,
    "lime_rows": 3,
    "lime_samples": 200,
    "shap_background": 5,
    "shap_explain": 5,
    "xper_explain": 40,
    "xper_background": 5,
    "xper_force_sample": 40,
    "xper_force_coalitions": 10,
}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A scratch project root: models/ and outputs redirected there, tiny budgets, one fitted
    model per analysed feature set. Undone when the module finishes."""
    root = tmp_path_factory.mktemp("root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config_module, "project_root", lambda: root)
        mp.setattr(analysis, "ART", root / "logreg" / "artifacts" / "analysis")
        for key, value in SMALL_BUDGET.items():
            mp.setitem(analysis.BUDGET, key, value)
        for feature_set in analysis.FEATURE_SETS:
            train_module.fit_one(feature_set, train_test(feature_set)[0])
        yield root


def run_stage(workspace, stage):
    stage()
    return workspace / "logreg" / "artifacts" / "analysis"


@pytest.fixture(scope="module")
def performance(workspace):
    return run_stage(workspace, analysis.stage_performance)


@pytest.fixture(scope="module")
def interpretability(workspace):
    return run_stage(workspace, analysis.stage_interpretability)


@pytest.fixture(scope="module")
def whitebox(workspace):
    return run_stage(workspace, analysis.stage_whitebox)


@pytest.fixture(scope="module")
def stability_dir(workspace):
    return run_stage(workspace, analysis.stage_stability)


@pytest.fixture(scope="module")
def fairness_dir(workspace):
    return run_stage(workspace, analysis.stage_fairness)


@pytest.fixture(scope="module")
def explanations(workspace):
    return run_stage(workspace, analysis.stage_explanations)


@pytest.fixture(scope="module")
def fairness_groups(workspace):
    return run_stage(workspace, analysis.stage_fairness_groups)


@pytest.fixture(scope="module")
def fairness_inference(workspace):
    return run_stage(workspace, analysis.stage_fairness_inference)


@pytest.fixture(scope="module")
def fpdp(workspace):
    return run_stage(workspace, analysis.stage_fpdp)


@pytest.fixture(scope="module")
def xper_dir(workspace):
    return run_stage(workspace, analysis.stage_xper)


@pytest.fixture(scope="module")
def xper_force(workspace):
    return run_stage(workspace, analysis.stage_xper_force)


@pytest.fixture(scope="module")
def primary(workspace):
    """The race-aware train split, fitted pipeline, and engineered design matrix."""
    train, _ = train_test(analysis.PRIMARY)
    pipeline = load(analysis.NAME, analysis.PRIMARY).estimator
    return train, pipeline, engineered(train.X)


class TestConstants:
    def test_analyses_the_logistic_registry_entry(self):
        assert analysis.NAME == "logistic"

    def test_primary_feature_set_is_race_aware(self):
        assert analysis.PRIMARY == "race_aware"
        assert analysis.PRIMARY in analysis.FEATURE_SETS

    def test_covers_every_configured_feature_set(self):
        assert analysis.FEATURE_SETS == FEATURE_SETS

    def test_writes_under_logreg_artifacts_analysis(self):
        assert analysis.ART == CONFIG.path("logreg", "artifacts", "analysis")

    def test_stage_registry_order(self):
        assert list(analysis.STAGES) == STAGE_NAMES

    def test_one_tost_delta_for_every_metric(self):
        assert analysis.TOST_DELTA == 0.05

    def test_budget_defaults_follow_the_shared_configuration(self):
        assert set(analysis.BUDGET) == set(SMALL_BUDGET)
        assert analysis.BUDGET["cv_repeats"] == CONFIG.iterations.cv_repeats
        assert analysis.BUDGET["bootstrap_refits"] == CONFIG.iterations.bootstrap_refits
        assert analysis.BUDGET["n_seeds"] == CONFIG.iterations.n_seeds
        assert analysis.BUDGET["shap_explain"] == CONFIG.iterations.shap_explain


class TestScored:
    def test_returns_the_splits_model_scores_and_own_threshold(self, workspace):
        from compas_scoring.evaluate import cost_curve

        feature_set = FEATURE_SETS[-1]
        train, test, model, score, threshold = analysis.scored(feature_set)
        assert model.feature_set == feature_set
        assert len(train) + len(test) == CONFIG.expected_rows
        np.testing.assert_allclose(score, model.predict_proba(test.X))
        assert threshold == cost_curve(test.y, score, CostModel()).optimal_threshold


class TestFrameInput:
    def test_accepts_numpy_and_matches_dataframe_predictions(self, primary):
        train, pipeline, _ = primary
        wrapped = analysis.FrameInput(pipeline, train.X.columns)
        rows = train.X.head(20)
        np.testing.assert_allclose(
            wrapped.predict_proba(rows.to_numpy()), pipeline.predict_proba(rows)
        )

    def test_keeps_the_column_order(self, primary):
        train, pipeline, _ = primary
        assert analysis.FrameInput(pipeline, train.X.columns).columns == list(train.X.columns)

    def test_the_bare_pipeline_rejects_numpy(self, primary):
        """The reason the wrapper exists."""
        train, pipeline, _ = primary
        # engineered() selects columns by name; a bare array has none to select.
        with pytest.raises((AttributeError, IndexError)):
            pipeline.predict_proba(train.X.head(5).to_numpy())


class TestLocalContributions:
    def test_contributions_plus_intercept_are_the_log_odds(self, primary):
        train, pipeline, _ = primary
        rows = train.X.head(4)
        table = analysis.local_contributions(pipeline, rows)
        totals = table.groupby("defendant", sort=False)["contribution"].sum()
        log_odds = pipeline.decision_function(rows)
        np.testing.assert_allclose(totals.to_numpy() + table["intercept"].iloc[0], log_odds)

    def test_one_row_per_term_per_defendant(self, primary):
        train, pipeline, design = primary
        table = analysis.local_contributions(pipeline, train.X.head(3))
        assert len(table) == 3 * design.shape[1]
        assert list(table["term"].iloc[: design.shape[1]]) == list(design.columns)

    def test_values_are_the_engineered_design(self, primary):
        train, pipeline, design = primary
        table = analysis.local_contributions(pipeline, train.X.head(1))
        np.testing.assert_allclose(table["value"], design.iloc[0].to_numpy(dtype=float))


class TestWrite:
    @pytest.fixture(autouse=True)
    def scratch(self, tmp_path, monkeypatch):
        self.dir = tmp_path / "nested" / "logreg"
        monkeypatch.setattr(analysis, "ART", self.dir)

    def test_creates_the_directory(self):
        analysis.write({"a": 1}, "x.json")
        assert self.dir.is_dir()

    def test_dataframe_goes_to_csv_without_the_index(self):
        frame = pd.DataFrame({"a": [1, 2]}, index=[10, 20])
        analysis.write(frame, "t.csv")
        pd.testing.assert_frame_equal(pd.read_csv(self.dir / "t.csv"), frame.reset_index(drop=True))

    def test_dict_goes_to_json_with_numpy_scalars_as_floats(self):
        analysis.write({"auc": np.float64(0.73), "n": 5}, "t.json")
        assert json.loads((self.dir / "t.json").read_text()) == {"auc": 0.73, "n": 5}

    def test_overwrites_an_existing_file(self):
        analysis.write({"v": 1}, "t.json")
        analysis.write({"v": 2}, "t.json")
        assert json.loads((self.dir / "t.json").read_text()) == {"v": 2}

    def test_announces_the_file(self, capsys):
        analysis.write({"a": 1}, "t.json")
        assert "-> t.json" in capsys.readouterr().out


class TestBanner:
    def test_prints_the_title_on_its_own_line(self, capsys):
        analysis.banner("Stage title")
        assert capsys.readouterr().out == "\nStage title\n"


class TestStagePerformance:
    def test_writes_every_artifact(self, performance):
        for name in (
            "performance.csv",
            "cost_curves.csv",
            "cost_sensitivity.csv",
            "cost_baselines.json",
            "delong_tests.csv",
        ):
            assert (performance / name).exists(), name

    def test_one_row_per_feature_set_plus_the_incumbent(self, performance):
        # Two rows per feature set: the cost-optimal threshold and the fixed 0.5 one.
        table = pd.read_csv(performance / "performance.csv")
        assert table["model"].tolist() == ["logistic"] * 2 * len(FEATURE_SETS) + ["compas"]
        assert table["feature_set"].tolist()[:-1] == [fs for fs in FEATURE_SETS for _ in (0, 1)]

    def test_auc_sits_inside_its_bootstrap_interval(self, performance):
        table = pd.read_csv(performance / "performance.csv").query(
            "model == 'logistic' and threshold_policy == 'optimal'"
        )
        assert (table["auc_lower"] <= table["auc"]).all()
        assert (table["auc"] <= table["auc_upper"]).all()
        assert (table["auc"] > 0.5).all()

    def test_incumbent_has_no_interval_and_a_fixed_cutoff(self, performance):
        row = pd.read_csv(performance / "performance.csv").query("model == 'compas'").iloc[0]
        assert np.isnan(row["auc_lower"]) and np.isnan(row["auc_upper"])
        assert row["cost_optimal_threshold"] == 0.5

    def test_savings_are_measured_against_the_best_trivial_policy(self, performance):
        table = pd.read_csv(performance / "performance.csv")
        floors = json.loads((performance / "cost_baselines.json").read_text())
        np.testing.assert_allclose(
            table["saving_vs_best_trivial"], floors["best_trivial"] - table["cost_per_capita"]
        )
        np.testing.assert_allclose(table["saving_per_1000"], table["saving_vs_best_trivial"] * 1000)

    def test_baselines_match_the_test_split(self, performance):
        _, test = train_test(analysis.PRIMARY)
        floors = json.loads((performance / "cost_baselines.json").read_text())
        assert floors["n_test"] == len(test)
        expected = baseline_costs(test.y, CostModel())
        assert floors["best_trivial"] == pytest.approx(expected["best_trivial"])

    def test_cost_optimum_is_the_minimum_of_the_curve(self, performance):
        table = pd.read_csv(performance / "performance.csv").query(
            "model == 'logistic' and threshold_policy == 'optimal'"
        )
        curves = pd.read_csv(performance / "cost_curves.csv")
        for _, row in table.iterrows():
            curve = curves[curves["feature_set"] == row["feature_set"]]
            assert row["cost_per_capita"] == pytest.approx(curve["cost_per_capita"].min())

    def test_curves_and_sensitivity_cover_every_feature_set(self, performance):
        for name in ("cost_curves.csv", "cost_sensitivity.csv"):
            frame = pd.read_csv(performance / name)
            assert set(frame["feature_set"]) == set(FEATURE_SETS), name

    def test_delong_compares_every_pair_and_each_set_against_compas(self, performance):
        pairs = pd.read_csv(performance / "delong_tests.csv")
        expected = [(a, b) for a, b in combinations(FEATURE_SETS, 2)]
        expected += [(fs, "compas") for fs in FEATURE_SETS]
        assert list(zip(pairs["model_a"], pairs["model_b"], strict=True)) == expected
        assert pairs["p_value"].between(0, 1).all()


class TestStageInterpretability:
    def test_writes_odds_ratios_and_scorecard(self, interpretability):
        assert (interpretability / "odds_ratios.csv").exists()
        assert (interpretability / "scorecard.csv").exists()

    def test_odds_ratios_use_the_engineered_design(self, interpretability, primary):
        """Raw X would describe a different model from the one deployed."""
        _, _, design = primary
        table = pd.read_csv(interpretability / "odds_ratios.csv")
        assert set(table["feature"]) == {"const", *design.columns}

    def test_odds_ratio_is_the_exponentiated_coefficient_inside_its_interval(
        self, interpretability
    ):
        table = pd.read_csv(interpretability / "odds_ratios.csv")
        np.testing.assert_allclose(table["odds_ratio"], np.exp(table["coefficient"]))
        assert (table["or_lower"] <= table["odds_ratio"]).all()
        assert (table["odds_ratio"] <= table["or_upper"]).all()

    def test_scorecard_is_built_from_the_pipeline_coefficients(self, interpretability, primary):
        _, pipeline, design = primary
        card = pd.read_csv(interpretability / "scorecard.csv").set_index("feature")
        fitted = pd.Series(pipeline.named_steps["clf"].coef_[0], index=design.columns)
        assert set(card.index) == set(design.columns)
        np.testing.assert_allclose(card.loc[fitted.index, "coefficient"], fitted)

    def test_risk_increasing_terms_cost_points(self, interpretability):
        card = pd.read_csv(interpretability / "scorecard.csv")
        assert (np.sign(card["points_per_unit"]) == -np.sign(card["coefficient"])).all()


class TestStageWhitebox:
    def test_writes_every_artifact(self, whitebox):
        for name in ("marginal_effects.csv", "lasso_path.csv", "experiment_log.csv"):
            assert (whitebox / name).exists(), name

    def test_marginal_effects_describe_the_engineered_design(self, whitebox, primary):
        _, _, design = primary
        table = pd.read_csv(whitebox / "marginal_effects.csv")
        assert set(table["feature"]) == set(design.columns)

    def test_marginal_effects_sit_inside_their_intervals(self, whitebox):
        table = pd.read_csv(whitebox / "marginal_effects.csv")
        assert (table["ci_lower"] <= table["marginal_effect"]).all()
        assert (table["marginal_effect"] <= table["ci_upper"]).all()

    def test_lasso_path_runs_over_the_raw_features(self, whitebox, primary):
        train, _, _ = primary
        path = pd.read_csv(whitebox / "lasso_path.csv")
        assert list(path.columns) == ["C", "n_selected", *train.X.columns]
        assert path["C"].is_monotonic_increasing

    def test_lasso_keeps_more_features_as_the_penalty_relaxes(self, whitebox):
        path = pd.read_csv(whitebox / "lasso_path.csv")
        assert path["n_selected"].iloc[0] <= path["n_selected"].iloc[-1]

    def test_experiment_log_holds_both_directions_from_a_start_row(self, whitebox):
        log = pd.read_csv(whitebox / "experiment_log.csv")
        assert set(log["direction"]) == {"forward", "backward"}
        starts = log[log["step"] == 0]
        assert len(starts) == 2
        assert (starts["action"] == "start").all()

    def test_every_accepted_step_improves_cv_auc(self, whitebox):
        log = pd.read_csv(whitebox / "experiment_log.csv")
        accepted = log[(log["step"] > 0) & log["accepted"]]
        assert (accepted["delta_auc"] > 0).all()

    def test_reports_the_forward_selection(self, workspace, capsys):
        analysis.stage_whitebox()
        assert "forward selection accepted" in capsys.readouterr().out


class TestStageStability:
    def test_writes_the_stability_summary(self, stability_dir):
        result = json.loads((stability_dir / "coefficient_stability.json").read_text())
        assert set(result) == {
            "mean_rank_correlation",
            "min_rank_correlation",
            "top_feature_consistency",
            "mean_rank_sd",
        }

    def test_values_are_in_range(self, stability_dir):
        result = json.loads((stability_dir / "coefficient_stability.json").read_text())
        assert -1 <= result["min_rank_correlation"] <= result["mean_rank_correlation"] <= 1
        assert 0 < result["top_feature_consistency"] <= 1
        assert result["mean_rank_sd"] >= 0

    def test_ranks_coefficients_by_absolute_size(self, workspace, monkeypatch, tmp_path):
        """The importance handed to importance_stability is |coef|, one per engineered column."""
        seen = {}

        def spy(estimator, X, y, importance_fn, n_refits):
            fitted = estimator.fit(X, y)
            seen["importance"] = importance_fn(fitted)
            seen["coef"] = fitted.named_steps["clf"].coef_[0]
            seen["n_refits"] = n_refits
            return {"mean_rank_correlation": 1.0}

        monkeypatch.setattr(analysis.stability, "importance_stability", spy)
        monkeypatch.setattr(analysis, "ART", tmp_path)  # keep the shared artifacts intact
        analysis.stage_stability()
        np.testing.assert_allclose(seen["importance"], np.abs(seen["coef"]))
        assert (seen["importance"] >= 0).all()
        assert seen["n_refits"] == analysis.BUDGET["bootstrap_refits"]

    def test_writes_every_artifact(self, stability_dir):
        for name in (
            "stability_cv.csv",
            "stability_summary.json",
            "learning_curves.csv",
            "perturbation.csv",
            "seed_sensitivity.csv",
            "subgroup_performance.csv",
            "model_distance.json",
            "stability_frontier.csv",
        ):
            assert (stability_dir / name).exists(), name

    def test_repeated_cv_runs_the_budgeted_folds(self, stability_dir):
        cv = pd.read_csv(stability_dir / "stability_cv.csv")
        assert len(cv) == SMALL_BUDGET["cv_repeats"] * CONFIG.iterations.cv_folds
        assert cv["auc"].between(0.5, 1).all()

    def test_summary_is_taken_at_the_cost_optimal_threshold(self, stability_dir):
        summary = json.loads((stability_dir / "stability_summary.json").read_text())
        _, _, _, score, threshold = analysis.scored(analysis.PRIMARY)
        assert summary["threshold"] == pytest.approx(threshold)
        assert summary["selection_rate_at_threshold"] == pytest.approx((score >= threshold).mean())
        assert summary["n_bootstrap_refits"] == SMALL_BUDGET["bootstrap_refits"]
        assert 0 <= summary["unstable_share"] <= 1
        assert summary["mean_sd"] >= 0 and summary["psi_train_vs_test"] >= 0

    def test_resampling_tables_follow_the_budget(self, stability_dir):
        seeds = pd.read_csv(stability_dir / "seed_sensitivity.csv")
        assert len(seeds) == SMALL_BUDGET["n_seeds"]
        perturbation = pd.read_csv(stability_dir / "perturbation.csv")
        assert set(perturbation["feature"]) == {"Number_of_Priors", "Misdemeanor"}
        curve = pd.read_csv(stability_dir / "learning_curves.csv")
        assert curve["n_train"].is_monotonic_increasing

    def test_subgroups_cover_every_protected_attribute(self, stability_dir):
        table = pd.read_csv(stability_dir / "subgroup_performance.csv")
        assert {"race", "sex", "age_band"} <= set(table["attribute"])

    def test_model_distance_has_importance_and_coefficient_parts(self, stability_dir):
        distance = json.loads((stability_dir / "model_distance.json").read_text())
        assert distance["mean_importance_distance"] >= 0
        assert distance["mean_distance"] >= 0
        assert distance["n_pairs"] == SMALL_BUDGET["distance_pairs"]

    def test_stability_frontier_sweeps_lambda(self, stability_dir):
        frontier = pd.read_csv(stability_dir / "stability_frontier.csv")
        assert frontier["lambda"].is_monotonic_increasing
        assert frontier["lambda"].iloc[0] == 0


class TestFairnessStrata:
    @pytest.fixture(scope="class")
    def test_split(self):
        return train_test(analysis.PRIMARY)[1]

    def test_one_stratifier_per_legitimate_risk_factor(self, test_split):
        strata = analysis.fairness_strata(test_split)
        assert list(strata) == ["priors_band", "age_band", "sex", "charge_degree"]

    def test_every_stratifier_labels_every_test_row_positionally(self, test_split):
        for name, values in analysis.fairness_strata(test_split).items():
            assert len(values) == len(test_split), name
            assert list(values.index) == list(range(len(test_split))), name
            assert values.notna().all(), name

    def test_group_labels_are_carried_over_unchanged(self, test_split):
        strata = analysis.fairness_strata(test_split)
        for name in ("age_band", "sex", "charge_degree"):
            assert strata[name].tolist() == test_split.groups[name].astype(str).tolist()

    def test_priors_use_the_shared_bands(self, test_split):
        bands = analysis.fairness_strata(test_split)["priors_band"]
        assert set(bands) <= {"0", "1", "2-3", "4-6", "7+"}


class TestStageFairness:
    COMPARISONS = ("aa_vs_others", "aa_vs_caucasian")

    def test_writes_every_artifact(self, fairness_dir):
        for name in (
            "fairness_parity_ztests.csv",
            "fairness_conditional_parity.csv",
            "fairness_stratum_odds_ratios.csv",
        ):
            assert (fairness_dir / name).exists(), name

    def test_ztests_cover_every_feature_set_comparison_and_metric(self, fairness_dir):
        table = pd.read_csv(fairness_dir / "fairness_parity_ztests.csv")
        expected = {
            (fs, comparison, metric)
            for fs in FEATURE_SETS
            for comparison in self.COMPARISONS
            for metric in ("selection_rate", "fpr", "fnr")
        }
        assert set(zip(table["feature_set"], table["comparison"], table["metric"])) == expected
        assert len(table) == len(expected)

    def test_criteria_are_labelled(self, fairness_dir):
        table = pd.read_csv(fairness_dir / "fairness_parity_ztests.csv")
        criteria = table.set_index("metric")["criterion"]
        assert set(criteria["selection_rate"]) == {"statistical_parity"}
        assert set(criteria["fpr"]) == set(criteria["fnr"]) == {"equalized_odds"}

    def test_each_model_is_judged_at_its_own_cost_optimal_threshold(self, fairness_dir):
        from compas_scoring.evaluate import cost_curve

        table = pd.read_csv(fairness_dir / "fairness_parity_ztests.csv")
        for feature_set in FEATURE_SETS:
            _, test = train_test(feature_set)
            score = load("logistic", feature_set).predict_proba(test.X)
            expected = cost_curve(test.y, score, CostModel()).optimal_threshold
            rows = table[table["feature_set"] == feature_set]
            assert rows["threshold"].unique().tolist() == pytest.approx([expected])
            assert rows["selection_rate"].iloc[0] == pytest.approx((score >= expected).mean())

    def test_reference_group_is_named(self, fairness_dir):
        table = pd.read_csv(fairness_dir / "fairness_parity_ztests.csv")
        names = dict(zip(table["comparison"], table["reference_group"], strict=False))
        assert names == {"aa_vs_others": "all other groups", "aa_vs_caucasian": "Caucasian"}

    def test_intervals_bracket_the_gaps(self, fairness_dir):
        table = pd.read_csv(fairness_dir / "fairness_parity_ztests.csv")
        assert (table["ci_lower"] <= table["difference"]).all()
        assert (table["difference"] <= table["ci_upper"]).all()
        assert table["p_value"].between(0, 1).all()

    def test_conditional_tests_cover_every_stratifier(self, fairness_dir):
        table = pd.read_csv(fairness_dir / "fairness_conditional_parity.csv")
        expected = {
            (fs, comparison, stratifier)
            for fs in FEATURE_SETS
            for comparison in self.COMPARISONS
            for stratifier in ("priors_band", "age_band", "sex", "charge_degree")
        }
        found = zip(table["feature_set"], table["comparison"], table["stratifier"])
        assert set(found) == expected

    def test_conditional_rows_carry_lr_cmh_and_mh_results(self, fairness_dir):
        table = pd.read_csv(fairness_dir / "fairness_conditional_parity.csv")
        for column in ("lr_statistic", "lr_p_value", "cmh_p_value", "mh_odds_ratio"):
            assert table[column].notna().all(), column
        assert (table["lr_dof"] == table["n_strata"]).all()
        assert (table["n_strata"] <= table["n_levels"]).all()
        assert (table["mh_or_lower"] <= table["mh_odds_ratio"]).all()
        assert (table["mh_odds_ratio"] <= table["mh_or_upper"]).all()

    def test_stratum_detail_matches_the_conditional_summary(self, fairness_dir):
        detail = pd.read_csv(fairness_dir / "fairness_stratum_odds_ratios.csv")
        summary = pd.read_csv(fairness_dir / "fairness_conditional_parity.csv")
        counts = detail.groupby(["feature_set", "comparison", "stratifier"]).size()
        for _, row in summary.iterrows():
            key = (row["feature_set"], row["comparison"], row["stratifier"])
            assert counts[key] == row["n_strata"], key

    def test_prints_the_equal_priors_odds_ratio(self, workspace, capsys):
        analysis.stage_fairness()
        out = capsys.readouterr().out
        assert out.count("at equal priors: MH odds ratio") == len(FEATURE_SETS)


class TestStageExplanations:
    def test_writes_every_artifact(self, explanations):
        for name in (
            "permutation_importance.csv",
            "pdp_Number_of_Priors.csv",
            "ice_curves.csv",
            "ice_heterogeneity.json",
            "shap_importance.csv",
            "shap_axioms.json",
            "lime_weights.csv",
            "disagreement_panel.csv",
            "archetypes.csv",
            "local_contributions.csv",
            "local_explanations.csv",
            "explanation_conflicts.csv",
        ):
            assert (explanations / name).exists(), name

    def test_global_importances_cover_every_raw_feature(self, explanations, primary):
        train, _, _ = primary
        for name in ("permutation_importance.csv", "shap_importance.csv"):
            table = pd.read_csv(explanations / name)
            assert sorted(table["feature"]) == sorted(train.X.columns), name

    def test_ice_has_raw_and_centered_curves_for_the_budgeted_rows(self, explanations):
        ice = pd.read_csv(explanations / "ice_curves.csv")
        assert set(ice["kind"]) == {"raw", "centered"}
        assert ice.query("kind == 'raw'")["row"].nunique() == SMALL_BUDGET["ice_curves"]

    def test_kernelshap_satisfies_efficiency_in_probability_space(self, explanations):
        axioms = json.loads((explanations / "shap_axioms.json").read_text())
        assert axioms["kernelshap"]["exact"]
        assert axioms["kernelshap"]["space"] == "probability"
        assert axioms["kernelshap_rows"] == SMALL_BUDGET["shap_explain"]

    def test_lime_explains_the_budgeted_rows(self, explanations):
        assert len(pd.read_csv(explanations / "lime_weights.csv")) == SMALL_BUDGET["lime_rows"]

    def test_disagreement_compares_all_three_methods(self, explanations):
        panel = pd.read_csv(explanations / "disagreement_panel.csv")
        pairs = {frozenset(pair) for pair in zip(panel["method_a"], panel["method_b"], strict=True)}
        assert pairs == {
            frozenset(("lime", "permutation")),
            frozenset(("lime", "kernelshap")),
            frozenset(("permutation", "kernelshap")),
        }

    def test_archetypes_carry_a_prediction_and_an_exact_explanation(self, explanations):
        archetypes = pd.read_csv(explanations / "archetypes.csv")
        assert archetypes["predicted_risk"].between(0, 1).all()
        contributions = pd.read_csv(explanations / "local_contributions.csv")
        assert set(contributions["archetype"]) == set(archetypes["archetype"])

    def test_paired_explanations_cover_every_archetype(self, explanations):
        archetypes = set(pd.read_csv(explanations / "archetypes.csv")["archetype"])
        assert set(pd.read_csv(explanations / "local_explanations.csv")["archetype"]) == archetypes
        conflicts = pd.read_csv(explanations / "explanation_conflicts.csv")
        assert set(conflicts["archetype"]) == archetypes

    def test_reports_the_top_driver(self, workspace, capsys, tmp_path, monkeypatch):
        monkeypatch.setattr(analysis, "ART", tmp_path)
        analysis.stage_explanations()
        assert "top driver by permutation importance" in capsys.readouterr().out


class TestStageFairnessGroups:
    def test_writes_every_artifact(self, fairness_groups):
        for name in (
            "fairness_by_group.csv",
            "fairness_disparities.csv",
            "calibration_by_group.csv",
            "impossibility.csv",
            "fairness_frontier.csv",
            "mitigation.csv",
            "group_thresholds.json",
        ):
            assert (fairness_groups / name).exists(), name

    def test_rates_cover_every_attribute_and_feature_set(self, fairness_groups):
        rates = pd.read_csv(fairness_groups / "fairness_by_group.csv")
        found = set(zip(rates["feature_set"], rates["attribute"], strict=True))
        assert found == {(fs, a) for fs in FEATURE_SETS for a in ("race", "sex", "age_band")}

    def test_one_disparity_row_per_feature_set_and_comparison(self, fairness_groups):
        table = pd.read_csv(fairness_groups / "fairness_disparities.csv")
        found = set(zip(table["feature_set"], table["comparison"], strict=True))
        assert found == {(fs, c) for fs in FEATURE_SETS for c in COMPARISONS}
        assert len(table) == len(found)

    def test_sweeps_keep_their_own_threshold_column(self, fairness_groups):
        """Regression: the operating threshold once overwrote the swept one."""
        for name in ("impossibility.csv", "fairness_frontier.csv"):
            table = pd.read_csv(fairness_groups / name)
            for _, block in table.groupby(["feature_set", "comparison"]):
                assert block["threshold"].nunique() > 1, name
                assert block["operating_threshold"].nunique() == 1, name

    def test_mitigation_compares_both_strategies_everywhere(self, fairness_groups):
        table = pd.read_csv(fairness_groups / "mitigation.csv")
        counts = table.groupby(["feature_set", "comparison"])["strategy"].apply(set)
        assert all(s == {"uniform threshold", "per-group thresholds"} for s in counts)

    def test_group_thresholds_are_keyed_by_feature_set_and_comparison(self, fairness_groups):
        thresholds = json.loads((fairness_groups / "group_thresholds.json").read_text())
        assert set(thresholds) == {f"{fs}__{c}" for fs in FEATURE_SETS for c in COMPARISONS}
        assert set(thresholds["race_aware__aa_vs_others"]) == {
            "African-American",
            "all other groups",
        }


class TestStageFairnessInference:
    METRICS = {
        "demographic_parity_difference",
        "fpr_difference",
        "fnr_difference",
        "equalized_odds_difference",
    }

    def test_writes_every_artifact(self, fairness_inference):
        assert (fairness_inference / "fairness_inference.csv").exists()
        assert (fairness_inference / "fairness_equivalence.csv").exists()

    def test_tost_uses_the_one_shared_delta(self, fairness_inference):
        table = pd.read_csv(fairness_inference / "fairness_equivalence.csv")
        assert (table["delta"] == analysis.TOST_DELTA).all()

    def test_tost_covers_every_cell(self, fairness_inference):
        table = pd.read_csv(fairness_inference / "fairness_equivalence.csv")
        found = set(zip(table["feature_set"], table["comparison"], table["metric"], strict=True))
        expected = {(fs, c, m) for fs in FEATURE_SETS for c in COMPARISONS for m in self.METRICS}
        assert found == expected

    def test_certified_means_the_interval_is_inside_the_zone(self, fairness_inference):
        table = pd.read_csv(fairness_inference / "fairness_equivalence.csv")
        inside = (table["tost_lower"] > -table["delta"]) & (table["tost_upper"] < table["delta"])
        assert (table["certified_fair"] == inside).all()

    def test_inference_has_intervals_for_every_comparison(self, fairness_inference):
        table = pd.read_csv(fairness_inference / "fairness_inference.csv")
        assert set(table["comparison"]) == set(COMPARISONS)
        assert self.METRICS <= set(table["metric"])
        assert table["p_value"].between(0, 1).all()


class TestStageFpdp:
    def test_writes_every_artifact(self, fpdp):
        for name in ("fpdp_curves.csv", "fpdp_candidates.csv", "fpdp_mitigation.csv"):
            assert (fpdp / name).exists(), name

    def test_curves_sweep_every_feature_of_each_model(self, fpdp):
        curves = pd.read_csv(fpdp / "fpdp_curves.csv")
        for feature_set in FEATURE_SETS:
            _, test = train_test(feature_set)
            block = curves[curves["feature_set"] == feature_set]
            assert set(block["feature"]) == set(test.X.columns)
            assert set(block["comparison"]) == set(COMPARISONS)

    def test_one_candidate_verdict_per_feature(self, fpdp):
        candidates = pd.read_csv(fpdp / "fpdp_candidates.csv")
        counts = candidates.groupby(["feature_set", "comparison"])["feature"].nunique()
        assert (counts == candidates.groupby(["feature_set", "comparison"]).size()).all()

    def test_baseline_mitigation_row_for_every_model(self, fpdp):
        table = pd.read_csv(fpdp / "fpdp_mitigation.csv")
        baseline = table[table["strategy"] == "uniform threshold"]
        assert len(baseline) == len(FEATURE_SETS) * len(COMPARISONS)

    def test_a_candidate_variable_is_neutralised(self, workspace, monkeypatch, tmp_path):
        """Force one candidate so the mitigation branch runs regardless of the data."""
        from compas_scoring import fairness_legacy as shared

        real = shared.candidate_variables

        def one_candidate(curves, alpha=0.05):
            found = real(curves, alpha)
            found.loc[found.index[0], "is_candidate"] = True
            return found

        monkeypatch.setattr(analysis.shared_fairness, "candidate_variables", one_candidate)
        monkeypatch.setattr(analysis, "ART", tmp_path)
        analysis.stage_fpdp()
        table = pd.read_csv(tmp_path / "fpdp_mitigation.csv")
        neutralised = table[table["strategy"] == "neutralise candidate variable"]
        assert len(neutralised) == len(FEATURE_SETS) * len(COMPARISONS)
        assert neutralised["variable"].str.contains(" = ").all()


class TestStageXper:
    def test_writes_every_artifact(self, xper_dir):
        for name in ("xper_auc.csv", "xper_cost.csv", "xper_overfitting.csv"):
            assert (xper_dir / name).exists(), name

    def test_decomposition_is_efficient(self, xper_dir):
        table = pd.read_csv(xper_dir / "xper_auc.csv")
        for _, block in table.groupby("feature_set"):
            assert block["contribution"].sum() == pytest.approx(block["metric_value"].iloc[0])

    def test_benchmark_is_the_uninformative_auc(self, xper_dir):
        table = pd.read_csv(xper_dir / "xper_auc.csv").query("feature == 'benchmark'")
        assert set(table["feature_set"]) == set(FEATURE_SETS)
        np.testing.assert_allclose(table["contribution"], 0.5)

    def test_overfitting_gap_is_train_minus_test(self, xper_dir):
        table = pd.read_csv(xper_dir / "xper_overfitting.csv")
        np.testing.assert_allclose(table["gap"], table["train"] - table["test"])
        assert set(table["feature_set"]) == set(FEATURE_SETS)


class TestGridModel:
    @pytest.fixture(scope="class")
    def grid(self, primary):
        train, pipeline, _ = primary
        return analysis.GridModel(pipeline, train.X), train, pipeline

    def test_one_score_per_possible_row(self, grid):
        model, train, _ = grid
        expected = np.prod([train.X[c].nunique() for c in train.X.columns])
        assert len(model.scores) == expected

    def test_lookup_equals_the_pipeline(self, grid):
        model, train, pipeline = grid
        rows = train.X.head(200)
        np.testing.assert_allclose(
            model.predict_proba(rows.to_numpy()), pipeline.predict_proba(rows)
        )

    def test_predict_thresholds_at_one_half(self, grid):
        model, train, _ = grid
        rows = train.X.head(200).to_numpy()
        expected = (model.predict_proba(rows)[:, 1] >= 0.5).astype(int)
        np.testing.assert_array_equal(model.predict(rows), expected)

    def test_rejects_a_value_outside_the_grid(self, grid):
        model, train, _ = grid
        row = train.X.head(1).to_numpy(dtype=float)
        row[0, 0] = 0.5  # half a prior
        with pytest.raises(ValueError, match="outside the lookup grid"):
            model.predict_proba(row)


class TestXperForceSample:
    def test_draws_the_budgeted_stratified_sample(self, workspace):
        _, test = train_test(analysis.PRIMARY)
        sample = analysis.xper_force_sample(test)
        assert len(sample) == SMALL_BUDGET["xper_force_sample"]
        assert sample.base_rate == pytest.approx(test.base_rate, abs=0.03)
        assert list(sample.X.index) == sorted(sample.X.index)

    def test_is_reproducible(self, workspace):
        _, test = train_test(analysis.PRIMARY)
        first = analysis.xper_force_sample(test).X.index
        assert list(first) == list(analysis.xper_force_sample(test).X.index)


class TestStageXperForce:
    def test_writes_every_artifact(self, xper_force):
        for name in ("xper_package_auc.csv", "xper_auc_individual.csv", "xper_package_check.json"):
            assert (xper_force / name).exists(), name

    def test_global_values_have_a_benchmark_and_every_feature(self, xper_force, primary):
        train, _, _ = primary
        phi = pd.read_csv(xper_force / "xper_package_auc.csv")
        assert list(phi["feature"]) == ["benchmark", *train.X.columns]

    def test_one_row_per_sampled_defendant(self, xper_force):
        phi_i = pd.read_csv(xper_force / "xper_auc_individual.csv")
        assert len(phi_i) == SMALL_BUDGET["xper_force_sample"]
        assert phi_i["row"].is_unique

    def test_check_records_the_protocol(self, xper_force):
        check = json.loads((xper_force / "xper_package_check.json").read_text())
        assert check["seed"] == analysis.XPER_FORCE_SEED
        assert check["n_coalitions"] == SMALL_BUDGET["xper_force_coalitions"]
        assert check["lookup_max_abs_diff"] < 1e-9
        assert 0.5 < check["auc_on_sample"] <= 1

    def test_refuses_a_lookup_table_that_disagrees(self, workspace, monkeypatch, tmp_path):
        real = analysis.GridModel.predict_proba
        monkeypatch.setattr(
            analysis.GridModel, "predict_proba", lambda self, X: real(self, X) + 0.1
        )
        monkeypatch.setattr(analysis, "ART", tmp_path)
        with pytest.raises(AssertionError, match="lookup table disagrees"):
            analysis.stage_xper_force()


class TestMain:
    @pytest.fixture
    def calls(self, monkeypatch, tmp_path):
        """Replace every stage with a recorder so main is tested on its own."""
        monkeypatch.setattr(analysis, "ART", tmp_path)
        record = []
        for name in analysis.STAGES:
            monkeypatch.setitem(analysis.STAGES, name, lambda n=name: record.append(n))
        return record

    def test_runs_every_stage_in_order_by_default(self, calls, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["logreg.analysis"])
        assert analysis.main() == 0
        assert calls == list(analysis.STAGES)

    def test_all_is_the_same_as_no_argument(self, calls, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["logreg.analysis", "--stage", "all"])
        analysis.main()
        assert calls == list(analysis.STAGES)

    @pytest.mark.parametrize("stage", STAGE_NAMES)
    def test_runs_only_the_selected_stage(self, calls, monkeypatch, stage):
        monkeypatch.setattr(sys, "argv", ["logreg.analysis", "--stage", stage])
        analysis.main()
        assert calls == [stage]

    def test_rejects_an_unknown_stage(self, calls, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["logreg.analysis", "--stage", "nope"])
        with pytest.raises(SystemExit):
            analysis.main()
        assert calls == []

    def test_reports_where_the_artifacts_went(self, calls, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(sys, "argv", ["logreg.analysis"])
        analysis.main()
        assert f"Artifacts in {tmp_path}" in capsys.readouterr().out


class TestEntryPoint:
    def test_running_the_module_exits_with_mains_status(self, workspace, monkeypatch):
        """`python -m logreg.analysis` calls main() and exits with its return value.

        The module is re-executed from scratch, so its ART is recomputed -- under the scratch
        root, because `workspace` keeps CONFIG.path redirected for the whole module.
        """
        monkeypatch.setattr(sys, "argv", ["logreg.analysis", "--stage", "interpretability"])
        with pytest.warns(RuntimeWarning, match="found in sys.modules"):
            with pytest.raises(SystemExit) as exit_info:
                runpy.run_module("logreg.analysis", run_name="__main__")
        assert exit_info.value.code == 0
        assert (workspace / "logreg" / "artifacts" / "analysis" / "scorecard.csv").exists()
