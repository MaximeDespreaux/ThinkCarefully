"""The performance metrics and the model adapter, on toy inputs (no TabPFN fit needed)."""

from __future__ import annotations

import numpy as np
import pytest
from pfn_metrics import (
    at_threshold,
    break_even_threshold,
    confusion,
    expected_calibration_error,
    performance,
)
from pfn_model import TabPFNModel
from sklearn.base import is_classifier

from compas_scoring.config import CONFIG

Y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
SCORE = np.array([0.9, 0.6, 0.3, 0.7, 0.2, 0.1, 0.4, 0.8])


def test_confusion_counts():
    pred = (SCORE >= 0.5).astype(int)
    assert confusion(Y, pred) == {"tp": 3, "fp": 1, "tn": 3, "fn": 1}


def test_error_rates_read_as_a_hypothesis_test():
    m = at_threshold(Y, SCORE, 0.5)
    assert m["type_i_error"] == pytest.approx(1 / 4)  # FP / (FP + TN)
    assert m["type_ii_error"] == pytest.approx(1 / 4)  # FN / (FN + TP)
    assert m["power"] == m["recall"] == pytest.approx(1 - m["type_ii_error"])
    assert m["specificity"] == pytest.approx(1 - m["type_i_error"])
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["balanced_accuracy"] == pytest.approx((m["recall"] + m["specificity"]) / 2)


def test_cost_per_defendant_uses_the_configured_costs():
    m = at_threshold(Y, SCORE, 0.5)
    expected = (1 * CONFIG.costs.c_fn + 1 * CONFIG.costs.c_fp) / len(Y)
    assert m["cost_per_defendant"] == pytest.approx(expected)


def test_break_even_threshold_is_where_detaining_starts_to_pay():
    t = break_even_threshold()
    # At p = t, the expected cost of releasing (p * c_fn) equals detaining ((1 - p) * c_fp).
    assert t * CONFIG.costs.c_fn == pytest.approx((1 - t) * CONFIG.costs.c_fp)


def test_performance_reports_both_operating_points():
    rows = performance(Y, SCORE, with_ci=False)
    assert [r["operating_point"] for r in rows] == ["0.5", "break_even"]
    assert rows[0]["auc"] == rows[1]["auc"]


def test_perfect_scores_give_perfect_metrics():
    rows = performance(Y, Y.astype(float), with_ci=False)
    assert rows[0]["auc"] == 1.0 and rows[0]["f1"] == 1.0 and rows[0]["type_i_error"] == 0.0


def test_calibration_error_is_zero_when_scores_match_observed_rates():
    y = np.array([0, 0, 0, 1] * 25)  # 25% re-offend
    assert expected_calibration_error(y, np.full(100, 0.25)) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(100, 0.75)) == pytest.approx(0.5)


def test_adapter_is_a_classifier_for_sklearn():
    """Needed by permutation_importance, PDP and the sklearn scorers."""
    assert is_classifier(TabPFNModel())
