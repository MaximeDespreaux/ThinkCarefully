"""Test logreg/train.py"""

from __future__ import annotations

import json
import runpy
import sys

import numpy as np
import pytest

import compas_scoring.config as config_module
from compas_scoring.config import CONFIG
from compas_scoring.data import train_test
from compas_scoring.models import FittedModel, load, model_path
from compas_scoring.stability import repeated_cv
from logreg import train as train_module
from logreg.model import build_logistic


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Redirect CONFIG.path -- and so models/ and artifacts/ -- to a scratch directory."""
    monkeypatch.setattr(config_module, "project_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def race_aware_train():
    return train_test("race_aware")[0]


class TestConstants:
    def test_name_matches_the_shared_registry_key(self):
        assert train_module.NAME == "logistic"

    def test_fits_every_configured_feature_set(self):
        assert train_module.FEATURE_SETS == tuple(CONFIG.feature_sets)


class TestFitOne:
    def test_returns_a_fitted_model_with_provenance(self, root, race_aware_train):
        fitted = train_module.fit_one("race_aware", race_aware_train)
        assert isinstance(fitted, FittedModel)
        assert fitted.name == "logistic"
        assert fitted.feature_set == "race_aware"
        assert fitted.key == "logistic__race_aware"
        assert fitted.features == list(race_aware_train.X.columns)
        assert fitted.best_params is None
        assert fitted.fit_seconds >= 0

    def test_saves_the_model_where_load_finds_it(self, root, race_aware_train):
        fitted = train_module.fit_one("race_aware", race_aware_train)
        path = model_path("logistic__race_aware")
        assert path.parent == root / "models"
        assert path.exists()
        reloaded = load("logistic", "race_aware")
        np.testing.assert_array_equal(
            reloaded.predict_proba(race_aware_train.X), fitted.predict_proba(race_aware_train.X)
        )

    def test_estimator_is_the_logreg_pipeline_and_is_fitted(self, root, race_aware_train):
        estimator = train_module.fit_one("race_aware", race_aware_train).estimator
        assert list(estimator.named_steps) == list(build_logistic().named_steps)
        assert hasattr(estimator.named_steps["clf"], "coef_")

    def test_cv_auc_is_single_repeat_train_only_cross_validation(self, root, race_aware_train):
        fitted = train_module.fit_one("race_aware", race_aware_train)
        expected = repeated_cv(
            build_logistic(), race_aware_train.X, race_aware_train.y, n_repeats=1
        )["auc"].mean()
        assert fitted.cv_auc == pytest.approx(expected)
        assert 0.5 < fitted.cv_auc < 1.0

    def test_cached_fit_is_skipped(self, root, race_aware_train, capsys):
        train_module.fit_one("race_aware", race_aware_train)
        stamp = model_path("logistic__race_aware").stat().st_mtime_ns
        capsys.readouterr()

        assert train_module.fit_one("race_aware", race_aware_train) is None
        assert "cached, skipping" in capsys.readouterr().out
        assert model_path("logistic__race_aware").stat().st_mtime_ns == stamp

    def test_force_refits_over_a_cached_fit(self, root, race_aware_train):
        train_module.fit_one("race_aware", race_aware_train)
        assert train_module.fit_one("race_aware", race_aware_train, force=True) is not None

    def test_reports_train_and_cv_auc_without_an_overfit_flag(self, root, race_aware_train, capsys):
        train_module.fit_one("race_aware", race_aware_train)
        out = capsys.readouterr().out
        assert "train AUC" in out and "CV AUC" in out
        assert "overfit?" not in out

    def test_flags_a_train_cv_gap_above_five_points(
        self, root, race_aware_train, capsys, monkeypatch
    ):
        monkeypatch.setattr(train_module, "roc_auc_score", lambda *_: 0.99)
        train_module.fit_one("race_aware", race_aware_train)
        assert "<-- overfit?" in capsys.readouterr().out


class TestMain:
    @pytest.fixture(autouse=True)
    def no_cli_args(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["logreg.train"])

    def test_fits_every_feature_set_and_returns_zero(self, root):
        assert train_module.main() == 0
        for feature_set in CONFIG.feature_sets:
            assert model_path(f"logistic__{feature_set}").exists()

    def test_writes_the_fit_summary(self, root):
        train_module.main()
        summary = json.loads((root / "artifacts" / "fit_summary.json").read_text())
        assert set(summary) == {f"logistic__{fs}" for fs in CONFIG.feature_sets}
        for entry in summary.values():
            assert set(entry) == {"cv_auc", "fit_seconds"}

    def test_merges_into_an_existing_summary(self, root):
        """Teammates' entries in the shared summary must survive a logistic run."""
        out = root / "artifacts" / "fit_summary.json"
        out.parent.mkdir(parents=True)
        out.write_text(json.dumps({"xgboost__race_aware": {"cv_auc": 0.7, "fit_seconds": 1.0}}))

        train_module.main()
        summary = json.loads(out.read_text())
        assert summary["xgboost__race_aware"] == {"cv_auc": 0.7, "fit_seconds": 1.0}
        assert "logistic__race_aware" in summary

    def test_second_run_reuses_cached_fits(self, root, capsys):
        train_module.main()
        capsys.readouterr()
        assert train_module.main() == 0
        assert capsys.readouterr().out.count("cached, skipping") == len(CONFIG.feature_sets)

    def test_force_flag_refits_everything(self, root, monkeypatch, capsys):
        train_module.main()
        capsys.readouterr()
        monkeypatch.setattr(sys, "argv", ["logreg.train", "--force"])
        train_module.main()
        assert "cached, skipping" not in capsys.readouterr().out

    def test_counts_only_logistic_fits(self, root):
        """Other families' files in models/ must not inflate the logistic count."""
        (root / "models").mkdir()
        (root / "models" / "xgboost__race_aware.joblib").write_bytes(b"")
        assert train_module.main() == 0

    def test_returns_one_when_a_fit_is_missing(self, root, monkeypatch):
        monkeypatch.setattr(train_module, "save", lambda fitted: None)
        assert train_module.main() == 1

    def test_never_reads_the_holdout(self, root, monkeypatch):
        """The test split handed back by train_test must never be touched during fitting."""

        class Holdout:
            def __getattr__(self, name):
                raise AssertionError(f"train.py read the holdout (.{name})")

            def __len__(self):
                raise AssertionError("train.py read the holdout (len)")

        monkeypatch.setattr(train_module, "train_test", lambda fs: (train_test(fs)[0], Holdout()))
        assert train_module.main() == 0


class TestEntryPoint:
    def test_running_the_module_exits_with_mains_status(self, root, monkeypatch):
        """`python -m logreg.train` calls main() and exits with its return value."""
        monkeypatch.setattr(sys, "argv", ["logreg.train"])
        with pytest.warns(RuntimeWarning, match="found in sys.modules"):
            with pytest.raises(SystemExit) as exit_info:
                runpy.run_module("logreg.train", run_name="__main__")
        assert exit_info.value.code == 0
        assert model_path("logistic__race_aware").exists()
