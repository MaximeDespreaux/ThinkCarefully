"""Tests for pfn_significance, the statistical machinery.

Each test checks a case where the right answer is known in advance, rather than checking that
the function runs. Two of them exist because the correct test and the obvious test disagree,
and the obvious one is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pfn_significance as sig

# ------------------------------------------------- the correction that makes the t-test valid


def test_self_comparison_is_never_significant():
    """A model compared against itself must return p = 1, not divide by zero."""
    scores = np.random.default_rng(0).normal(0.73, 0.02, 150)
    result = sig.corrected_resampled_ttest(scores, scores, n_train=4320, n_test=1080)
    assert result["p_corrected"] == 1.0
    assert result["mean_difference"] == 0.0


def test_correction_is_strictly_more_conservative():
    """The Nadeau-Bengio correction inflates the variance, so p can only grow.

    This is the whole point: repeated-CV folds share training rows, so the naive test treats
    correlated measurements as independent and reports significance it has not earned.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(0.735, 0.016, 150)
    b = rng.normal(0.729, 0.016, 150)
    result = sig.corrected_resampled_ttest(a, b, n_train=4320, n_test=1080)

    assert result["p_corrected"] > result["p_naive"]
    assert abs(result["t_corrected"]) < abs(result["t_naive"])


def test_correction_scales_with_test_fold_size():
    """A larger held-out fraction means more overlap between training sets, so more inflation."""
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.74, 0.02, 100), rng.normal(0.73, 0.02, 100)
    small = sig.corrected_resampled_ttest(a, b, n_train=4800, n_test=200)
    large = sig.corrected_resampled_ttest(a, b, n_train=3000, n_test=2000)
    assert large["p_corrected"] > small["p_corrected"]


def test_paired_test_picks_wilcoxon_when_differences_are_not_normal():
    rng = np.random.default_rng(3)
    normal_diff = rng.normal(0.01, 0.005, 60)
    skewed_diff = rng.exponential(0.01, 60)
    base = rng.normal(0.73, 0.01, 60)

    assert sig.paired_difference_test(base + normal_diff, base, "a", "b")["test"] == "paired t-test"
    assert (
        sig.paired_difference_test(base + skewed_diff, base, "a", "b")["test"]
        == "Wilcoxon signed-rank"
    )


# ----------------------------------------------------------------------------- McNemar


def test_mcnemar_detects_a_one_sided_error_pattern():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    good, bad = y.copy(), y.copy()
    good[rng.choice(400, 40, replace=False)] ^= 1
    bad[rng.choice(400, 140, replace=False)] ^= 1

    result = sig.mcnemar_decisions(y, good, bad, "good", "bad")
    assert result["p_value"] < 1e-6
    assert result["only_a_correct"] > result["only_b_correct"]


def test_mcnemar_on_identical_decisions_does_not_crash():
    """Zero discordant pairs is a degenerate table, not an error."""
    y = np.random.default_rng(2).integers(0, 2, 100)
    result = sig.mcnemar_decisions(y, y, y, "a", "b")
    assert result["p_value"] == 1.0
    assert "nothing to test" in result["note"]


def test_mcnemar_uses_the_exact_test_when_discordance_is_small():
    """Below ~25 discordant pairs the chi-squared approximation is unreliable."""
    y = np.zeros(200, dtype=int)
    a, b = y.copy(), y.copy()
    a[[1, 2, 3]] = 1
    b[[4, 5]] = 1
    assert "exact" in sig.mcnemar_decisions(y, a, b, "a", "b")["test"]


# --------------------------------------------------------------------------- equivalence


def test_tost_certifies_a_genuinely_small_auc_difference():
    draws = np.random.default_rng(0).normal(0.001, 0.002, 2000)
    assert sig.auc_equivalence(draws, delta=0.01)["equivalent"]


def test_tost_refuses_a_difference_that_is_small_but_imprecise():
    """Centred on zero, far too wide to certify -- the case a conventional test waves through."""
    draws = np.random.default_rng(0).normal(0.0, 0.05, 2000)
    assert not sig.auc_equivalence(draws, delta=0.01)["equivalent"]


# ------------------------------------------------------------------------------ spread


def test_zero_spread_is_reported_as_zero():
    """A conventional model permuted row-wise produces exactly this: no variation at all."""
    result = sig.spread_against_zero(np.full(20, 0.7355), "AUC")
    assert result["p_value"] == 1.0
    assert result["sd"] == 0.0


def test_real_spread_is_detected():
    values = np.random.default_rng(0).normal(0.735, 0.004, 20)
    assert sig.spread_against_zero(values, "AUC")["p_value"] < 0.05


def test_variance_comparison_finds_the_wider_sample():
    rng = np.random.default_rng(0)
    wide, narrow = rng.normal(0, 0.09, 500), rng.normal(0, 0.02, 500)
    result = sig.variance_comparison(wide, narrow, "wide", "narrow")
    assert result["p_value"] < 1e-10
    assert result["variance_ratio"] > 1


# ------------------------------------------------------------------------- calibration


def test_calibration_tests_pass_on_well_calibrated_scores():
    """Labels drawn from the predicted probabilities: calibrated by construction."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < p).astype(int)
    assert sig.hosmer_lemeshow(y, p)["p_value"] > 0.05
    assert sig.spiegelhalter_z(y, p)["p_value"] > 0.05


def test_calibration_tests_catch_a_shifted_score():
    """Same ranking, systematically overstated risk -- invisible to AUC, caught here."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, 4000)
    y = (rng.uniform(size=4000) < p).astype(int)
    inflated = np.clip(p + 0.25, 0.01, 0.99)
    assert sig.hosmer_lemeshow(y, inflated)["p_value"] < 0.01
    assert sig.spiegelhalter_z(y, inflated)["p_value"] < 0.01


# ------------------------------------------------------------------------ multiplicity


def test_holm_is_more_conservative_than_raw_but_never_reverses_order():
    frame = pd.DataFrame({"p_value": [0.001, 0.02, 0.04, 0.3, 0.9]})
    out = sig.holm_adjust(frame, "family")
    assert (out["p_adjusted"] >= out["p_value"]).all()
    assert out["significant_adjusted"].sum() <= out["significant_raw"].sum()
    assert out["p_adjusted"].is_monotonic_increasing


def test_holm_tolerates_missing_p_values():
    """TOST rows carry an interval rather than a p-value; they must not break the family."""
    frame = pd.DataFrame({"p_value": [0.01, np.nan, 0.6]})
    out = sig.holm_adjust(frame, "family")
    assert out["p_adjusted"].isna().sum() == 1
    assert out["significant_adjusted"].iloc[0]


def test_assemble_puts_the_claim_first():
    frame = sig.holm_adjust(pd.DataFrame({"claim": ["x"], "test": ["t"], "p_value": [0.01]}), "f")
    out = sig.assemble(frame)
    assert list(out.columns)[:2] == ["family", "claim"]
