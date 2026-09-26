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
from compas_scoring.models import load
from logreg import analysis
from logreg import train as train_module
from logreg.features import engineered

FEATURE_SETS = tuple(CONFIG.feature_sets)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A scratch project root holding one fitted logistic model per feature set."""
    root = tmp_path_factory.mktemp("root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config_module, "project_root", lambda: root)
        mp.setattr(analysis, "ART", root / "artifacts" / "logreg")
        for feature_set in FEATURE_SETS:
            train_module.fit_one(feature_set, train_test(feature_set)[0])
        yield root


def run_stage(workspace, stage):
    stage()
    return workspace / "artifacts" / "logreg"


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

    def test_writes_under_artifacts_logreg(self):
        assert analysis.ART == CONFIG.path("artifacts", "logreg")

    def test_stage_registry_order(self):
        assert list(analysis.STAGES) == ["performance", "interpretability", "whitebox", "stability"]


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
        assert "-> logreg/t.json" in capsys.readouterr().out


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
        table = pd.read_csv(performance / "performance.csv")
        assert table["model"].tolist() == ["logistic"] * len(FEATURE_SETS) + ["compas"]
        assert table["feature_set"].tolist()[:-1] == list(FEATURE_SETS)

    def test_auc_sits_inside_its_bootstrap_interval(self, performance):
        table = pd.read_csv(performance / "performance.csv").query("model == 'logistic'")
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
        table = pd.read_csv(performance / "performance.csv").query("model == 'logistic'")
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

    def test_ranks_coefficients_by_absolute_size(self, workspace, monkeypatch):
        """The importance handed to importance_stability is |coef|, one per engineered column."""
        seen = {}

        def spy(estimator, X, y, importance_fn, n_refits):
            fitted = estimator.fit(X, y)
            seen["importance"] = importance_fn(fitted)
            seen["coef"] = fitted.named_steps["clf"].coef_[0]
            seen["n_refits"] = n_refits
            return {"mean_rank_correlation": 1.0}

        monkeypatch.setattr(analysis.stability, "importance_stability", spy)
        analysis.stage_stability()
        np.testing.assert_allclose(seen["importance"], np.abs(seen["coef"]))
        assert (seen["importance"] >= 0).all()
        assert seen["n_refits"] == CONFIG.iterations.bootstrap_refits


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

    @pytest.mark.parametrize("stage", ["performance", "interpretability", "whitebox", "stability"])
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
        assert (workspace / "artifacts" / "logreg" / "scorecard.csv").exists()
