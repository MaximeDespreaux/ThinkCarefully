"""Race-recoverability diagnostic in xgboost/race_proxy.py."""

from __future__ import annotations

import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

from race_proxy import FEATURE_VARIANTS, evaluate_proxy, fit_proxy, race_proxy_table, race_target


def test_race_target_matches_group_label(small_test):
    target = race_target(small_test)

    assert set(target.unique()) <= {0, 1}
    assert target.index.equals(small_test.X.index)
    assert (target == 1).sum() == (small_test.groups["race"] == "African-American").sum()


def test_race_target_name(small_test):
    assert race_target(small_test).name == "is_african_american"


def test_feature_variants_defined_on_race_blind_columns(split):
    train, _ = split
    for columns in FEATURE_VARIANTS.values():
        if columns is not None:
            assert set(columns) <= set(train.X.columns)


def test_evaluate_proxy_matches_sklearn(small_model, small_test):
    """small_model predicts recidivism, not race, but evaluate_proxy is metric-agnostic --
    it only needs a model, an X and a binary y."""
    y = race_target(small_test)
    result = evaluate_proxy(small_model, small_test.X, y)

    y_score = small_model.predict_proba(small_test.X)[:, 1]
    assert result["auc"] == pytest.approx(roc_auc_score(y, y_score))
    assert result["accuracy"] == pytest.approx(accuracy_score(y, (y_score >= 0.5).astype(int)))


@pytest.mark.slow
def test_priors_only_model_never_sees_other_columns(small_train):
    """Regression guard: race_proxy_table must slice X down to the variant's columns, not
    fit on the full frame regardless of the fake column list."""
    y = race_target(small_train)
    search = fit_proxy(small_train.X[["Number_of_Priors"]], y, n_iter=3)

    assert search.best_estimator_.n_features_in_ == 1


@pytest.mark.slow
def test_race_proxy_table_shape_and_ranking():
    table = race_proxy_table(n_iter=5)

    assert set(table.index) == set(FEATURE_VARIANTS)
    assert table.loc["all_features", "n_features"] == 5
    assert table.loc["priors_only", "n_features"] == 1
    # All race-blind features can only help relative to priors alone.
    assert table.loc["all_features", "auc"] >= table.loc["priors_only", "auc"]


@pytest.mark.slow
def test_race_is_recoverable_above_chance():
    """The headline finding: race-blind features still predict race well above the AUC=0.5
    floor, which is why removing race from the model barely changes the fairness gap."""
    table = race_proxy_table(n_iter=10)

    assert table.loc["all_features", "auc"] > 0.6
    assert table.loc["priors_only", "auc"] > 0.55


def test_main_prints_table(monkeypatch, capsys):
    import race_proxy

    original = race_proxy.race_proxy_table
    monkeypatch.setattr(race_proxy, "race_proxy_table", lambda: original(n_iter=3))
    race_proxy.main()

    out = capsys.readouterr().out
    assert "all_features" in out
    assert "priors_only" in out
