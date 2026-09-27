"""Test logreg/model.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import compas_scoring.config as config_module
from compas_scoring.config import CONFIG
from compas_scoring.data import train_test
from logreg.features import engineered
from logreg.model import FittedModel, build_logistic, load, model_path, save

FEATURE_SETS = sorted(CONFIG.feature_sets)


@pytest.fixture(scope="module")
def splits():
    return {fs: train_test(fs)[0] for fs in FEATURE_SETS}


class TestBuildLogistic:
    def test_is_an_engineer_scale_classify_pipeline(self):
        pipeline = build_logistic()
        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == ["engineer", "scale", "clf"]
        assert isinstance(pipeline.named_steps["scale"], StandardScaler)
        assert isinstance(pipeline.named_steps["clf"], LogisticRegression)

    def test_engineer_step_is_logreg_features_engineered(self):
        assert build_logistic().named_steps["engineer"].func is engineered

    def test_classifier_hyperparameters(self):
        clf = build_logistic().named_steps["clf"]
        assert clf.penalty == "l2"
        assert clf.C == 1.0
        assert clf.solver == "lbfgs"
        assert clf.max_iter == 2000

    def test_random_state_defaults_to_config_and_is_passed_through(self):
        assert build_logistic().named_steps["clf"].random_state == CONFIG.random_state
        assert build_logistic(7).named_steps["clf"].random_state == 7

    def test_returns_a_fresh_unfitted_pipeline_each_call(self):
        a, b = build_logistic(), build_logistic()
        assert a is not b
        assert not hasattr(a.named_steps["clf"], "coef_")

    def test_is_cloneable(self):
        """repeated_cv and importance_stability clone the pipeline; that must round-trip."""
        cloned = clone(build_logistic())
        assert list(cloned.named_steps) == ["engineer", "scale", "clf"]

    @pytest.mark.parametrize("feature_set", FEATURE_SETS)
    def test_one_coefficient_per_engineered_column(self, splits, feature_set):
        train = splits[feature_set]
        pipeline = build_logistic().fit(train.X, train.y)
        assert pipeline.named_steps["clf"].coef_.shape == (1, engineered(train.X).shape[1])

    @pytest.mark.parametrize("feature_set", FEATURE_SETS)
    def test_predicts_valid_probabilities(self, splits, feature_set):
        train = splits[feature_set]
        proba = build_logistic().fit(train.X, train.y).predict_proba(train.X)
        assert proba.shape == (len(train), 2)
        assert ((proba >= 0) & (proba <= 1)).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_scale_step_standardises_the_engineered_design(self, splits):
        train = splits["race_aware"]
        pipeline = build_logistic().fit(train.X, train.y)
        design = pipeline.named_steps["engineer"].transform(train.X)
        scaled = pipeline.named_steps["scale"].transform(design)
        np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-9)

    def test_same_seed_gives_identical_coefficients(self, splits):
        train = splits["race_aware"]
        a = build_logistic(0).fit(train.X, train.y).named_steps["clf"].coef_
        b = build_logistic(0).fit(train.X, train.y).named_steps["clf"].coef_
        np.testing.assert_array_equal(a, b)

    def test_priors_raise_predicted_risk(self, splits):
        """Sanity check on direction: more priors, all else equal, never lowers the risk."""
        train = splits["race_aware"]
        pipeline = build_logistic().fit(train.X, train.y)
        base = train.X.iloc[[0]].copy()
        risks = []
        for priors in (0, 2, 5, 10, 20):
            row = base.copy()
            row["Number_of_Priors"] = priors
            risks.append(pipeline.predict_proba(row)[0, 1])
        assert risks == sorted(risks)


@pytest.fixture
def fitted(splits):
    train = splits["race_aware"]
    return FittedModel(
        name="logistic",
        feature_set="race_aware",
        estimator=build_logistic().fit(train.X, train.y),
        features=list(train.X.columns),
        fit_seconds=0.1,
        cv_auc=0.72,
    )


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "project_root", lambda: tmp_path)
    return tmp_path


class TestFittedModel:
    def test_key_joins_name_and_feature_set(self, fitted):
        assert fitted.key == "logistic__race_aware"

    def test_predict_proba_returns_the_positive_class(self, fitted, splits):
        X = splits["race_aware"].X.head(10)
        np.testing.assert_allclose(fitted.predict_proba(X), fitted.estimator.predict_proba(X)[:, 1])

    def test_predict_proba_enforces_the_feature_contract(self, fitted, splits):
        X = splits["race_aware"].X.head(10)
        with pytest.raises(ValueError, match="expects"):
            fitted.predict_proba(X[list(reversed(X.columns))])

    def test_optional_fields_default_to_none(self, splits):
        model = FittedModel("logistic", "race_aware", build_logistic(), [], 0.0)
        assert model.best_params is None and model.cv_auc is None


class TestModelPath:
    def test_lives_in_models_with_a_joblib_suffix(self, root):
        assert model_path("logistic__race_blind") == root / "models" / "logistic__race_blind.joblib"


class TestSave:
    def test_creates_the_directory_and_writes_the_file(self, root, fitted):
        save(fitted)
        assert (root / "models" / "logistic__race_aware.joblib").exists()


class TestLoad:
    def test_round_trips_a_saved_model(self, root, fitted, splits):
        save(fitted)
        loaded = load("logistic", "race_aware")
        X = splits["race_aware"].X.head(10)
        np.testing.assert_allclose(loaded.predict_proba(X), fitted.predict_proba(X))
        assert loaded.cv_auc == fitted.cv_auc

    def test_missing_model_says_how_to_build_it(self, root):
        with pytest.raises(FileNotFoundError, match="not found"):
            load("logistic", "race_blind")
