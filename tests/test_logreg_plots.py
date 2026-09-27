"""Test logreg/plots.py."""

from __future__ import annotations

import runpy
import sys

import matplotlib.pyplot as plt
import pytest

import compas_scoring.config as config_module
from compas_scoring.data import train_test
from logreg import analysis, plots
from logreg import train as train_module

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
        mp.setattr(plots, "ART", root / "logreg" / "artifacts" / "analysis")
        mp.setattr(plots, "FIG", root / "logreg" / "artifacts" / "figures")
        for key, value in SMALL_BUDGET.items():
            mp.setitem(analysis.BUDGET, key, value)
        for feature_set in analysis.FEATURE_SETS:
            train_module.fit_one(feature_set, train_test(feature_set)[0])
        for stage in analysis.STAGES.values():
            stage()
        yield root


def assert_png(path):
    assert path.exists() and path.suffix == ".png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert path.parent == plots.FIG


@pytest.fixture(autouse=True)
def no_open_figures():
    """Every plot function must close its figure, or a long run leaks memory."""
    yield
    assert plt.get_fignums() == []


class TestRead:
    def test_reads_a_table_from_the_artifacts(self, workspace):
        assert not plots.read("performance.csv").empty


class TestColorAndLabel:
    def test_each_feature_set_has_a_fixed_colour_and_label(self):
        assert plots.color("race_aware") == "#2a78d6"
        assert plots.color("race_blind") == "#eb6834"
        assert plots.label("race_aware") == "Race-aware (FS1)"

    def test_unknown_feature_sets_fall_back_to_neutral(self):
        assert plots.color("selected") == plots.NEUTRAL
        assert plots.label("selected") == "selected"


class TestDollars:
    def test_sign_goes_before_the_currency(self):
        assert plots.dollars(-350) == "-$350"
        assert plots.dollars(12500) == "$12,500"
        assert plots.dollars(0) == "$0"


class TestStyle:
    def test_hides_top_and_right_spines(self):
        fig, ax = plt.subplots()
        plots.style(ax)
        assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
        plt.close(fig)


class TestTitle:
    def test_sets_a_left_aligned_title_and_optional_subtitle(self):
        fig, ax = plt.subplots()
        plots.title(ax, "Heading", "note")
        assert ax.get_title(loc="left") == "Heading"
        assert any(text.get_text() == "note" for text in ax.texts)
        plots.title(ax, "Heading only")
        assert ax.get_title(loc="left") == "Heading only"
        plt.close(fig)


class TestLogAxis:
    def test_plain_number_ticks_inside_the_data_range(self):
        fig, ax = plt.subplots()
        plots.log_axis(ax, [1.8, 4.5])
        assert ax.get_xscale() == "log"
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == ["2", "3", "5"]
        plt.close(fig)

    def test_falls_back_to_one_when_no_candidate_fits(self):
        fig, ax = plt.subplots()
        plots.log_axis(ax, [100, 200])
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["1"]
        plt.close(fig)


class TestFigureLegend:
    def test_one_legend_above_the_panels(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], label="a")
        ax.plot([1, 0], label="b")
        plots.figure_legend(fig, ax)
        assert len(fig.legends) == 1
        assert [t.get_text() for t in fig.legends[0].get_texts()] == ["a", "b"]
        plt.close(fig)


class TestDotWhisker:
    def test_draws_one_whisker_and_one_dot(self):
        fig, ax = plt.subplots()
        plots.dot_whisker(ax, 0, 1.0, 0.5, 1.5, "#2a78d6", "series")
        assert len(ax.collections) == 2
        assert ax.get_legend_handles_labels()[1] == ["series"]
        plt.close(fig)


class TestSave:
    def test_creates_the_directory_and_closes_the_figure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plots, "FIG", tmp_path / "new" / "dir")
        fig, _ = plt.subplots()
        path = plots.save(fig, "x.png")
        assert_png(path)
        assert not plt.fignum_exists(fig.number)


@pytest.mark.parametrize(
    ("function", "filename"),
    [
        ("plot_auc_intervals", "auc_intervals.png"),
        ("plot_cost_curves", "cost_curves.png"),
        ("plot_odds_ratios", "odds_ratios.png"),
        ("plot_marginal_effects", "marginal_effects.png"),
        ("plot_lasso_path", "lasso_path.png"),
        ("plot_pdp_ice", "pdp_ice_priors.png"),
        ("plot_local_contributions", "local_contributions.png"),
        ("plot_learning_curve", "learning_curve.png"),
        ("plot_fairness_ztests", "fairness_ztests.png"),
        ("plot_mh_odds_ratios", "mh_odds_ratios.png"),
        ("plot_tost", "tost_equivalence.png"),
        ("plot_fpdp", "fpdp_priors.png"),
        ("plot_fairness_frontier", "fairness_frontier.png"),
        ("plot_calibration_by_group", "calibration_by_group.png"),
        ("plot_xper", "xper_auc.png"),
        ("plot_scorecard", "scorecard.png"),
        ("plot_stepwise", "stepwise_log.png"),
        ("plot_importance_comparison", "importance_comparison.png"),
        ("plot_cost_sensitivity", "cost_sensitivity.png"),
        ("plot_subgroup_performance", "subgroup_performance.png"),
        ("plot_stability_frontier", "stability_frontier.png"),
        ("plot_fairness_by_group", "fairness_by_group.png"),
        ("plot_impossibility", "impossibility.png"),
        ("plot_mitigation", "mitigation.png"),
        ("plot_stratum_flag_rates", "stratum_flag_rates.png"),
        ("plot_xper_cost", "xper_cost.png"),
        ("plot_xper_force", "xper_auc_force.png"),
    ],
)
class TestEveryFigure:
    def test_draws_its_png(self, workspace, function, filename):
        path = getattr(plots, function)()
        assert path.name == filename
        assert_png(path)

    def test_is_registered(self, function, filename):
        assert getattr(plots, function) in plots.FIGURES.values()


class TestPlotLassoPath:
    def test_highlights_the_first_features_to_enter(self, workspace, monkeypatch):
        seen = {}
        real_legend = plt.Axes.legend

        def spy(self, *args, **kwargs):
            seen["labels"] = list(args[1]) if len(args) > 1 else None
            return real_legend(self, *args, **kwargs)

        monkeypatch.setattr(plt.Axes, "legend", spy)
        plots.plot_lasso_path(highlight=2)
        assert len(seen["labels"]) == 2


class TestMain:
    def test_every_figure_function_is_registered_once(self):
        functions = [name for name in dir(plots) if name.startswith("plot_")]
        assert len(plots.FIGURES) == len(functions) == 27

    def test_draws_every_registered_figure(self, workspace, capsys):
        assert plots.main() == 0
        assert len(list(plots.FIG.glob("*.png"))) >= len(plots.FIGURES)
        assert f"{len(plots.FIGURES)} figures" in capsys.readouterr().out

    def test_refuses_to_run_before_the_analysis(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(plots, "ART", tmp_path / "missing")
        assert plots.main() == 1
        assert "run `uv run python -m logreg.analysis` first" in capsys.readouterr().out


class TestEntryPoint:
    def test_running_the_module_exits_with_mains_status(self, workspace, monkeypatch):
        """`python -m logreg.plots` calls main() and exits with its return value.

        The re-executed module recomputes ART and FIG from CONFIG, which the workspace
        keeps pointed at the scratch root.
        """
        monkeypatch.setattr(sys, "argv", ["logreg.plots"])
        with pytest.warns(RuntimeWarning, match="found in sys.modules"):
            with pytest.raises(SystemExit) as exit_info:
                runpy.run_module("logreg.plots", run_name="__main__")
        assert exit_info.value.code == 0
        assert (workspace / "logreg" / "artifacts" / "figures" / "xper_auc.png").exists()
