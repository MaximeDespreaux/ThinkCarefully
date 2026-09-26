"""Test logreg/model.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from compas_scoring.config import CONFIG
from compas_scoring.data import train_test
from logreg.features import engineered
from logreg.model import build_logistic

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
