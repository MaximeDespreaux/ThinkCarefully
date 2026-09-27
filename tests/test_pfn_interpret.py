"""pfn_interpret and pfn_importance on toy models whose answer is known (no TabPFN fit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pfn_importance import cost_scorer
from pfn_interpret import (
    impurity_decrease,
    marginal_effects,
    shap_efficiency_check,
    surrogate_impurity,
)
from sklearn.tree import DecisionTreeClassifier

from compas_scoring.config import CONFIG


class Additive:
    """A 'model' whose risk is 0.1 + 0.05 * priors + 0.2 * African_American + 0.1 * Hispanic."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = 0.1 + 0.05 * X["Number_of_Priors"] + 0.2 * X["African_American"] + 0.1 * X["Hispanic"]
        p = np.clip(np.asarray(p, dtype=float), 0, 1)
        return np.column_stack([1 - p, p])


@pytest.fixture
def people():
    return pd.DataFrame(
        {
            "Number_of_Priors": [0.0, 2.0, 5.0, 1.0],
            "African_American": [1.0, 0.0, 0.0, 0.0],
            "Hispanic": [0.0, 1.0, 0.0, 0.0],
        }
    )


def test_marginal_effects_recover_an_additive_model(people):
    effects = marginal_effects(Additive(), people).set_index("feature")["marginal_effect"]
    assert effects["Number_of_Priors"] == pytest.approx(0.05)
    # Against the reference category (all race dummies 0), not against "some other race".
    assert effects["African_American"] == pytest.approx(0.2)
    assert effects["Hispanic"] == pytest.approx(0.1)


def test_flipping_one_race_dummy_clears_the_others(people):
    """Setting Hispanic = 1 for an African-American row must not create a two-race person."""
    seen = []

    class Recorder(Additive):
        def predict_proba(self, X):
            seen.append(X.copy())
            return super().predict_proba(X)

    marginal_effects(Recorder(), people)
    frame = seen[0]
    race = [c for c in CONFIG.race_dummies if c in frame.columns]
    assert frame[race].sum(axis=1).max() == 1


def test_impurity_decrease_all_on_the_only_informative_feature():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"signal": rng.integers(0, 2, 500), "noise": rng.integers(0, 2, 500)})
    y = X["signal"].to_numpy()
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
    for criterion in ("gini", "entropy", "misclassification"):
        shares = impurity_decrease(tree, X.columns, criterion)
        assert shares["signal"] == pytest.approx(1.0)
        assert shares.sum() == pytest.approx(1.0)


def test_surrogate_is_faithful_to_a_model_it_can_represent(people):
    X = pd.concat([people] * 50, ignore_index=True)
    result = surrogate_impurity(Additive(), X, threshold=0.25, max_depth=4)
    assert result["fidelity_accuracy_gini"] == pytest.approx(1.0)
    assert set(result["impurity"]["criterion"]) == {"gini", "entropy", "misclassification"}


def test_shap_efficiency_check_measures_the_residual():
    values = np.array([[0.1, 0.2], [0.0, -0.1]])
    check = shap_efficiency_check(values, 0.5, np.array([0.8, 0.4]))
    assert check["max_abs_residual"] == pytest.approx(0.0)
    assert check["holds_to_1e-5"]
    assert not shap_efficiency_check(values, 0.5, np.array([0.9, 0.4]))["holds_to_1e-5"]


def test_cost_scorer_is_negative_cost_with_the_configured_costs(people):
    y = np.array([1, 0, 1, 0])  # scores 0.3, 0.3, 0.35, 0.15 -> at 0.32: FN, TN, TP, TN
    score = cost_scorer(0.32)(Additive(), people, y)
    assert score == pytest.approx(-(1 * CONFIG.costs.c_fn) / 4)


def test_permutation_multi_scores_every_metric_from_one_call_per_repeat(people):
    from pfn_importance import negative_cost, permutation_multi
    from sklearn.metrics import roc_auc_score

    calls = []

    class Counting(Additive):
        def predict_proba(self, X):
            calls.append(len(X))
            return super().predict_proba(X)

    rng = np.random.default_rng(0)
    race = rng.integers(0, 3, 400)  # 0 reference, 1 African-American, 2 Hispanic
    X = pd.DataFrame(
        {
            "Number_of_Priors": rng.integers(0, 12, 400).astype(float),
            "African_American": (race == 1).astype(float),
            "Hispanic": (race == 2).astype(float),
        }
    )
    y = (rng.uniform(size=400) < Additive().predict_proba(X)[:, 1]).astype(int)
    result = permutation_multi(
        Counting(), X, y, {"auc": roc_auc_score, "cost": negative_cost(0.252)}, n_repeats=4
    )
    assert len(calls) == 1 + 4
    auc = result["auc"].set_index("feature")["importance"]
    # Priors carries most of the signal; shuffling it must cost the most AUC.
    assert auc.idxmax() == "Number_of_Priors"
    assert set(result) == {"auc", "cost"}
