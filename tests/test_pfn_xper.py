"""pfn_xper: the efficiency axiom is the whole claim, so it is asserted rather than printed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pfn_xper as xper
import pytest
from sklearn.linear_model import LogisticRegression

from compas_scoring.config import CONFIG


@pytest.fixture(scope="module")
def toy():
    """A small, fully controlled problem: x0 informative, x1 weaker, x2 pure noise."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(
        {
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    logit = 2.0 * X["x0"] + 0.5 * X["x1"]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = LogisticRegression(max_iter=2000).fit(X, y)
    return X, y, model


def test_efficiency_holds_exactly(toy):
    """phi_0 + sum(phi_j) == PM. Exact enumeration means this is not an approximation."""
    X, y, model = toy
    decomposition = xper.decompose(
        lambda f: model.predict_proba(f)[:, 1], X.iloc[:100], y[:100], X.iloc[100:130]
    )
    error = xper.check_efficiency(decomposition)
    assert error < 1e-10


def test_efficiency_failure_is_raised_not_swallowed():
    broken = pd.Series({"benchmark": 0.5, "a": 0.1})
    broken.attrs["metric_value"] = 0.9  # 0.6 != 0.9
    with pytest.raises(AssertionError, match="efficiency violated"):
        xper.check_efficiency(broken)


def test_benchmark_is_the_uninformative_auc(toy):
    """With no features revealed every row gets the same score, so AUC must be 0.5."""
    X, y, model = toy
    decomposition = xper.decompose(
        lambda f: model.predict_proba(f)[:, 1], X.iloc[:100], y[:100], X.iloc[100:130]
    )
    assert decomposition["benchmark"] == pytest.approx(0.5, abs=1e-9)


def test_noise_feature_contributes_about_nothing(toy):
    """x2 never entered the data-generating process; its XPER share should be negligible."""
    X, y, model = toy
    decomposition = xper.decompose(
        lambda f: model.predict_proba(f)[:, 1], X.iloc[:150], y[:150], X.iloc[150:190]
    )
    assert abs(decomposition["x2"]) < abs(decomposition["x0"]) / 10
    assert decomposition["x0"] > decomposition["x1"] > 0


def test_cost_metric_matches_a_direct_computation():
    costs = CONFIG.costs
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.1, 0.9, 0.1])  # one FN, one FP at t=0.5
    metric = xper.cost_metric(0.5)
    assert metric(y_true, y_score) == pytest.approx((costs.c_fn + costs.c_fp) / 4)


def test_exact_enumeration_refuses_an_infeasible_width():
    X = pd.DataFrame(np.zeros((5, 17)), columns=[f"f{i}" for i in range(17)])
    with pytest.raises(ValueError, match="2\\^17"):
        xper.decompose(lambda f: np.zeros(len(f)), X, np.zeros(5), X)


def test_batching_changes_the_number_of_calls_not_the_answer(toy):
    """TabPFN pays a fixed cost per call, so coalitions are batched. The values must not move."""
    X, y, model = toy
    args = (lambda f: model.predict_proba(f)[:, 1], X.iloc[:60], y[:60], X.iloc[60:80])
    one_per_call = xper.decompose(*args, batch_rows=1)
    batched = xper.decompose(*args, batch_rows=60 * 20 * 3)
    assert one_per_call.attrs["model_calls"] == 8
    assert batched.attrs["model_calls"] == 3
    pd.testing.assert_series_equal(one_per_call, batched)
