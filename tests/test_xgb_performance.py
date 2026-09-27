"""Metric logic in xgboost/performance.py."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from performance import confusion, performance, performance_table, roc_data, scores

METRICS = ["auc", "accuracy", "precision", "recall", "f1", "brier"]


# --- performance() -------------------------------------------------------------------------


def test_performance_hand_computed():
    # Positive/negative pairs: 0.4>0.1, 0.4<0.6, 0.9>0.1, 0.9>0.6 -> AUC 3/4.
    # Predictions at 0.5: [0, 1, 0, 1] -> one each of TN, FP, FN, TP.
    result = performance([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])

    assert result["auc"] == pytest.approx(0.75)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["brier"] == pytest.approx((0.01 + 0.36 + 0.36 + 0.01) / 4)


def test_performance_perfect_scorer():
    result = performance([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0])

    for metric in ["auc", "accuracy", "precision", "recall", "f1"]:
        assert result[metric] == pytest.approx(1.0)
    assert result["brier"] == pytest.approx(0.0)


def test_threshold_is_inclusive():
    """A score exactly at the threshold counts as a positive prediction."""
    assert performance([0, 1], [0.2, 0.5])["recall"] == pytest.approx(1.0)
    assert performance([0, 1], [0.2, 0.5], threshold=0.6)["recall"] == pytest.approx(0.0)


def test_binary_labels_as_scores():
    """The COMPAS path: with a 0/1 score, AUC is the balanced accuracy of that one cut-off."""
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    label = np.array([0, 1, 0, 1, 1, 0, 1, 0])

    assert performance(y, label)["auc"] == pytest.approx(balanced_accuracy_score(y, label))


def test_no_positive_predictions_is_quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = performance([0, 1, 1, 0], [0.1, 0.2, 0.3, 0.4])

    assert result["precision"] == 0
    assert result["recall"] == 0
    assert result["f1"] == 0


def test_input_types_agree():
    y = [0, 1, 0, 1, 1]
    s = [0.3, 0.7, 0.55, 0.45, 0.9]
    as_list = performance(y, s)
    as_array = performance(np.array(y), np.array(s))
    as_series = performance(pd.Series(y), pd.Series(s, index=[10, 11, 12, 13, 14]))

    assert as_list == pytest.approx(as_array)
    assert as_list == pytest.approx(as_series)


# --- performance_table() -------------------------------------------------------------------


def test_performance_table_layout(small_model, small_test):
    table = performance_table(small_model, small_test)

    assert list(table.index) == ["XGBoost", "COMPAS (incumbent)"]
    assert list(table.columns) == METRICS
    assert table.equals(table.round(4))


def test_performance_table_matches_performance(small_model, small_test):
    table = performance_table(small_model, small_test)
    expected = performance(small_test.y, scores(small_model, small_test))

    for metric in METRICS:
        assert table.loc["XGBoost", metric] == pytest.approx(round(expected[metric], 4))


# --- confusion() ---------------------------------------------------------------------------


def test_confusion_counts_everyone(small_model, small_test):
    assert confusion(small_model, small_test).to_numpy().sum() == len(small_test)


def test_confusion_orientation(small_model, small_test):
    """Rows are actual outcomes, columns are predictions -- a transposed matrix would pass
    the sum check but swap false positives and false negatives."""
    y = small_test.y.to_numpy()
    y_pred = (scores(small_model, small_test) >= 0.5).astype(int)
    matrix = confusion(small_model, small_test)

    assert matrix.loc["actual: no reoffence", "predicted: no"] == np.sum((y == 0) & (y_pred == 0))
    assert matrix.loc["actual: no reoffence", "predicted: yes"] == np.sum((y == 0) & (y_pred == 1))
    assert matrix.loc["actual: reoffence", "predicted: no"] == np.sum((y == 1) & (y_pred == 0))
    assert matrix.loc["actual: reoffence", "predicted: yes"] == np.sum((y == 1) & (y_pred == 1))


def test_higher_threshold_predicts_fewer_positives(small_model, small_test):
    predicted_yes = [
        confusion(small_model, small_test, threshold)["predicted: yes"].sum()
        for threshold in [0.3, 0.5, 0.7]
    ]

    assert predicted_yes == sorted(predicted_yes, reverse=True)


# --- roc_data() ----------------------------------------------------------------------------


def test_roc_curve_shape(small_model, small_test):
    roc = roc_data(small_test.y, scores(small_model, small_test))

    assert (roc.fpr.iloc[0], roc.tpr.iloc[0]) == (0.0, 0.0)
    assert (roc.fpr.iloc[-1], roc.tpr.iloc[-1]) == (1.0, 1.0)
    assert roc.fpr.is_monotonic_increasing
    assert roc.tpr.is_monotonic_increasing


def test_roc_area_is_auc(small_model, small_test):
    y_score = scores(small_model, small_test)
    roc = roc_data(small_test.y, y_score)

    assert np.trapezoid(roc.tpr, roc.fpr) == pytest.approx(roc_auc_score(small_test.y, y_score))


# --- scores() ------------------------------------------------------------------------------


def test_scores_are_positive_class_probabilities(small_model, small_test):
    y_score = scores(small_model, small_test)

    assert y_score.shape == (len(small_test),)
    assert ((y_score >= 0) & (y_score <= 1)).all()
    np.testing.assert_array_equal(y_score, small_model.predict_proba(small_test.X)[:, 1])
