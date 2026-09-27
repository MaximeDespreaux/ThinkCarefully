"""Statistical tests behind the comparative claims (ported from compas-data, studies/tabpfn).

The study says things like "TabPFN beats XGBoost on small data" and "TabPFN's scores are less
stable". Each of those is a hypothesis, and each gets a named test with an effect size, a
p-value and a multiplicity correction. Nothing here is exotic -- everything runs on scipy and
statsmodels, both already pinned by the parent project.

Two things are easy to get wrong, and both are handled explicitly.

**The obvious t-test is invalid.** Comparing two models across repeated k-fold AUCs with
``scipy.stats.ttest_rel`` is what everyone reaches for and it does not hold: the training sets
of overlapping folds share most of their rows, so the fold scores are not independent draws.
The variance estimate comes out biased downward and the Type I error rate is badly inflated
(Dietterich 1998; Nadeau & Bengio 2003). ``corrected_resampled_ttest`` applies the
Nadeau-Bengio variance correction and reports the naive statistic alongside it, so the reader
can see what the correction costs rather than taking it on faith.

**Running many tests manufactures significance.** Comparing one model against five others at
six training sizes is thirty tests; at alpha = 0.05 about 1.5 come back significant by chance.
``holm_adjust`` corrects each family and publishes raw and adjusted p-values together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

ALPHA = 0.05


def _verdict(p: float, alpha: float = ALPHA, claim: str = "difference") -> str:
    if np.isnan(p):
        return "not testable"
    return f"{claim} supported (p<{alpha})" if p < alpha else f"no evidence of {claim}"


# ------------------------------------------------------------------ comparing two models


def corrected_resampled_ttest(
    scores_a, scores_b, n_train: int, n_test: int, n_splits: int = 5
) -> dict[str, float]:
    """Nadeau & Bengio corrected paired t-test for repeated cross-validation.

    The correction inflates the variance of the mean difference by ``1/k + n_test/n_train``
    to account for the fact that resampled training sets overlap. Without it the test treats
    k*r correlated fold scores as if they were k*r independent measurements.

    Returns the corrected statistic and p-value, plus the naive uncorrected pair so the
    difference between them is visible rather than asserted.
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired scores must align: {a.shape} vs {b.shape}")

    differences = a - b
    n = len(differences)
    mean = float(differences.mean())
    variance = float(differences.var(ddof=1))

    if variance == 0 or n < 2:
        # Identical models: the difference is exactly zero everywhere, so there is nothing
        # to test. Returning p=1 is the honest answer; a t-statistic would be 0/0.
        return {
            "mean_difference": mean,
            "t_corrected": 0.0,
            "p_corrected": 1.0,
            "t_naive": 0.0,
            "p_naive": 1.0,
            "df": max(n - 1, 0),
            "correction_factor": float("nan"),
        }

    # The naive test any tutorial would reach for -- reported only for contrast.
    t_naive = mean / np.sqrt(variance / n)
    p_naive = float(2 * stats.t.sf(abs(t_naive), df=n - 1))

    inflation = 1.0 / n_splits + n_test / n_train
    t_corrected = mean / np.sqrt(variance * inflation)
    p_corrected = float(2 * stats.t.sf(abs(t_corrected), df=n - 1))

    return {
        "mean_difference": mean,
        "t_corrected": float(t_corrected),
        "p_corrected": p_corrected,
        "t_naive": float(t_naive),
        "p_naive": p_naive,
        "df": n - 1,
        "correction_factor": float(inflation * n),
    }


def paired_difference_test(scores_a, scores_b, label_a: str, label_b: str) -> dict:
    """Paired comparison that picks its own test: t if the differences look normal, else Wilcoxon.

    Shapiro-Wilk decides, so the choice is justified by the data rather than by habit.
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    differences = a - b

    if np.allclose(differences, 0):
        return {
            "comparison": f"{label_a} vs {label_b}",
            "test": "none (identical)",
            "statistic": 0.0,
            "p_value": 1.0,
            "normal": True,
            "effect_size": 0.0,
            "mean_difference": 0.0,
        }

    normal = bool(stats.shapiro(differences).pvalue > 0.05) if len(differences) >= 3 else True
    if normal:
        result = stats.ttest_rel(a, b)
        name = "paired t-test"
    else:
        result = stats.wilcoxon(a, b)
        name = "Wilcoxon signed-rank"

    spread = differences.std(ddof=1)
    return {
        "comparison": f"{label_a} vs {label_b}",
        "test": name,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "normal": normal,
        # Cohen's d for paired samples: the mean difference in units of its own spread.
        "effect_size": float(differences.mean() / spread) if spread > 0 else float("nan"),
        "mean_difference": float(differences.mean()),
    }


def mcnemar_decisions(y_true, pred_a, pred_b, label_a: str, label_b: str) -> dict:
    """McNemar's test on the decisions two models make about the same defendants.

    DeLong asks whether the two models *rank* people differently. McNemar asks a different
    question: of the people they disagree about, is the disagreement one-sided? Two models can
    have indistinguishable AUCs while systematically misclassifying different people, and only
    this test sees that. Dietterich (1998) recommends it for exactly this setting.
    """
    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true

    only_a = int((correct_a & ~correct_b).sum())  # a right, b wrong
    only_b = int((~correct_a & correct_b).sum())  # b right, a wrong

    if only_a + only_b == 0:
        return {
            "comparison": f"{label_a} vs {label_b}",
            "test": "McNemar",
            "only_a_correct": 0,
            "only_b_correct": 0,
            "statistic": float("nan"),
            "p_value": 1.0,
            "note": "identical decisions -- nothing to test",
        }

    table = [[0, only_a], [only_b, 0]]
    # exact=True below ~25 discordant pairs, where the chi-squared approximation is poor.
    exact = (only_a + only_b) < 25
    result = mcnemar(table, exact=exact, correction=not exact)

    return {
        "comparison": f"{label_a} vs {label_b}",
        "test": "McNemar (exact)" if exact else "McNemar (chi-sq, continuity-corrected)",
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "note": "",
    }


def matched_rate_decisions(score_a, score_b, selection_rate: float) -> tuple:
    """Threshold two score vectors so both detain the same share of defendants.

    Comparing decisions at each model's own cost-optimal threshold conflates two different
    things. TabPFN's optimum detains 86.6% of defendants and XGBoost's detains 69.0%, so a
    McNemar test between them is dominated by the fact that one policy is far more
    aggressive -- the more aggressive one simply makes more false positives. That comparison
    is worth reporting, but it answers "do these deployed configurations differ", not "is one
    model better at ranking people".

    Fixing the selection rate removes the policy difference and leaves the ranking, which is
    what a model-quality claim needs.
    """
    score_a, score_b = np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    cut = 100 * (1 - selection_rate)
    return (
        (score_a >= np.percentile(score_a, cut)).astype(int),
        (score_b >= np.percentile(score_b, cut)).astype(int),
    )


def auc_equivalence(differences, delta: float, alpha: float = ALPHA) -> dict:
    """TOST on an AUC difference: can two models be certified *equivalent*?

    The same argument the fairness section makes. Failing to reject "the models differ" is not
    evidence that they perform the same -- it is usually just a small sample. Reversing the
    hypotheses asks the question the recommendation actually rests on: is the difference small
    enough to be treated as no difference at all?
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[~np.isnan(differences)]
    lower = float(np.quantile(differences, alpha))
    upper = float(np.quantile(differences, 1 - alpha))
    equivalent = bool(lower > -delta and upper < delta)
    return {
        "delta": float(delta),
        "mean_difference": float(differences.mean()),
        "tost_lower": lower,
        "tost_upper": upper,
        "equivalent": equivalent,
        "minimum_delta": float(max(abs(lower), abs(upper))),
        "verdict": "equivalent" if equivalent else "cannot certify equivalence",
    }


# ------------------------------------------------------------------------- variance tests


def variance_comparison(samples_a, samples_b, label_a: str, label_b: str) -> dict:
    """Brown-Forsythe test: do two models' per-defendant score variances differ?

    "Less stable" is a claim about spread, so it needs a test of spread. Brown-Forsythe
    (Levene centred on the median) rather than Bartlett, because prediction variances are
    strongly right-skewed and Bartlett assumes normality.
    """
    a, b = np.asarray(samples_a, dtype=float), np.asarray(samples_b, dtype=float)
    result = stats.levene(a, b, center="median")
    return {
        "comparison": f"{label_a} vs {label_b}",
        "test": "Brown-Forsythe (Levene, median-centred)",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "sd_a": float(a.std(ddof=1)),
        "sd_b": float(b.std(ddof=1)),
        "variance_ratio": float(a.var(ddof=1) / b.var(ddof=1)) if b.var(ddof=1) > 0 else np.inf,
    }


def spread_against_zero(values, label: str) -> dict:
    """Is an observed spread distinguishable from zero?

    Used for the row-order probe: for any conventional model, permuting the training rows
    changes nothing at all, so the null here is not a formality -- it is the behaviour every
    other model in the roster is guaranteed to show.
    """
    values = np.asarray(values, dtype=float)
    centred = values - values.mean()
    if np.allclose(centred, 0):
        return {
            "quantity": label,
            "test": "one-sample t-test on deviations",
            "statistic": 0.0,
            "p_value": 1.0,
            "sd": 0.0,
            "range": 0.0,
        }
    result = stats.ttest_1samp(np.abs(centred), 0.0)
    return {
        "quantity": label,
        "test": "one-sample t-test on |deviation from mean|",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "sd": float(values.std(ddof=1)),
        "range": float(values.max() - values.min()),
    }


def groups_differ(*samples, label: str) -> dict:
    """Kruskal-Wallis across several groups -- used for the seed sweep.

    Non-parametric because ten AUCs per group is far too few to lean on normality.
    """
    result = stats.kruskal(*samples)
    return {
        "quantity": label,
        "test": "Kruskal-Wallis",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_groups": len(samples),
    }


# ------------------------------------------------------------------------ calibration


def hosmer_lemeshow(y_true, y_score, n_bins: int = 10) -> dict:
    """Hosmer-Lemeshow goodness-of-fit: are predicted risks borne out within score bands?

    The project reports expected calibration error everywhere but never tests it. A large
    statistic means the predicted probabilities are wrong as probabilities, which matters more
    than AUC to anyone who reads the number itself rather than the ranking.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"], n_bins, labels=False, duplicates="drop")

    statistic, used = 0.0, 0
    for _, chunk in frame.groupby("bin"):
        observed, expected = chunk["y"].sum(), chunk["p"].sum()
        n = len(chunk)
        denominator = expected * (1 - expected / n)
        if denominator <= 0:
            continue
        statistic += (observed - expected) ** 2 / denominator
        used += 1

    df = max(used - 2, 1)
    p_value = float(stats.chi2.sf(statistic, df))
    return {
        "test": "Hosmer-Lemeshow",
        "statistic": float(statistic),
        "df": df,
        "p_value": p_value,
        # A *large* p-value is the good outcome here: no evidence of miscalibration.
        "verdict": "no evidence of miscalibration" if p_value > ALPHA else "miscalibrated",
    }


def spiegelhalter_z(y_true, y_score) -> dict:
    """Spiegelhalter's z-test for calibration -- a complement to Hosmer-Lemeshow.

    Hosmer-Lemeshow depends on an arbitrary choice of bins; this one does not.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_score, dtype=float), 1e-9, 1 - 1e-9)

    numerator = np.sum((y - p) * (1 - 2 * p))
    denominator = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    z = float(numerator / denominator) if denominator > 0 else float("nan")
    p_value = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return {
        "test": "Spiegelhalter z",
        "statistic": z,
        "p_value": p_value,
        "verdict": "no evidence of miscalibration" if p_value > ALPHA else "miscalibrated",
    }


# --------------------------------------------------------------------------- proportions


def selection_rate_test(y_pred, sensitive, group_a: str, group_b: str) -> dict:
    """Two-proportion z-test on selection rates between two groups.

    The third test named alongside chi-squared and Cochran-Mantel-Haenszel in the course
    material. It answers the plainest version of the fairness question: are these two groups
    detained at different rates?
    """
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)
    mask_a, mask_b = sensitive == group_a, sensitive == group_b

    counts = np.array([y_pred[mask_a].sum(), y_pred[mask_b].sum()])
    nobs = np.array([mask_a.sum(), mask_b.sum()])
    statistic, p_value = proportions_ztest(counts, nobs)

    rate_a, rate_b = counts[0] / nobs[0], counts[1] / nobs[1]
    return {
        "comparison": f"{group_a} vs {group_b}",
        "test": "two-proportion z-test",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "rate_a": float(rate_a),
        "rate_b": float(rate_b),
        "difference": float(rate_a - rate_b),
    }


# ------------------------------------------------------------------------- multiplicity


def holm_adjust(frame: pd.DataFrame, family: str, column: str = "p_value") -> pd.DataFrame:
    """Holm-Bonferroni within one family of comparisons.

    Holm rather than plain Bonferroni: it controls the same family-wise error rate but is
    uniformly more powerful, so it costs less to be honest about multiplicity.
    """
    out = frame.copy()
    valid = out[column].notna()
    out["family"] = family
    out["p_adjusted"] = np.nan
    out["significant_raw"] = out[column] < ALPHA
    out["significant_adjusted"] = False

    if valid.sum():
        reject, adjusted, _, _ = multipletests(
            out.loc[valid, column].to_numpy(), alpha=ALPHA, method="holm"
        )
        out.loc[valid, "p_adjusted"] = adjusted
        out.loc[valid, "significant_adjusted"] = reject
    return out


def assemble(*frames: pd.DataFrame) -> pd.DataFrame:
    """Stack every tested family into the one table the write-up is allowed to cite."""
    table = pd.concat(frames, ignore_index=True)
    ordered = [
        c
        for c in (
            "family",
            "claim",
            "comparison",
            "quantity",
            "test",
            "statistic",
            "p_value",
            "p_adjusted",
            "significant_raw",
            "significant_adjusted",
            "effect_size",
            "mean_difference",
        )
        if c in table.columns
    ]
    return table[ordered + [c for c in table.columns if c not in ordered]]
