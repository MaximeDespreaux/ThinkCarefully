"""Fairness measurement and inference for TabPFN's decisions.

Ported from the earlier single-model study (compas-data, src/compas_scoring/fairness.py);
the post-processing mitigations are left out, since they are not in this analysis's scope.

The COMPAS cohort is the canonical worked example of why a single fairness number is
never enough, so this module is built around the tension rather than around one metric:

* ProPublica's charge was **error-rate imbalance** -- Black defendants who did not
  re-offend were flagged high-risk far more often (FPR gap).
* Northpointe's defence was **calibration** -- a given score meant the same re-offence
  rate for both groups.

Both were arithmetically correct. With unequal base rates (52.3% vs 39.1% here),
calibration and equalised odds cannot both hold; ``impossibility_evidence`` demonstrates
that on the client's own data rather than asserting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import brier_score_loss, roc_auc_score

PRIMARY_RACES = ("African-American", "Caucasian")
# Sex comparison, in the same (unprivileged-first) order convention: gaps read Female - Male.
SEXES = ("Female", "Male")
FOUR_FIFTHS = 0.8  # the 80% rule used by US disparate-impact guidance


def _rate(y_true, y_pred, true_value: int, pred_value: int) -> float:
    mask = np.asarray(y_true) == true_value
    if not mask.any():
        return float("nan")
    return float((np.asarray(y_pred)[mask] == pred_value).mean())


def false_positive_rate(y_true, y_pred) -> float:
    """Flagged high-risk among those who did not re-offend. ProPublica's central metric."""
    return _rate(y_true, y_pred, true_value=0, pred_value=1)


def false_negative_rate(y_true, y_pred) -> float:
    return _rate(y_true, y_pred, true_value=1, pred_value=0)


def true_positive_rate(y_true, y_pred) -> float:
    return _rate(y_true, y_pred, true_value=1, pred_value=1)


def positive_predictive_value(y_true, y_pred) -> float:
    """Of those flagged high-risk, the share who re-offended. Northpointe's metric."""
    mask = np.asarray(y_pred) == 1
    if not mask.any():
        return float("nan")
    return float((np.asarray(y_true)[mask] == 1).mean())


GROUP_METRICS = {
    "count": lambda yt, yp: len(yt),
    "base_rate": lambda yt, yp: float(np.mean(yt)),
    "selection_rate": lambda yt, yp: float(np.mean(yp)),
    "accuracy": lambda yt, yp: float(np.mean(np.asarray(yt) == np.asarray(yp))),
    "tpr": true_positive_rate,
    "fpr": false_positive_rate,
    "fnr": false_negative_rate,
    "ppv": positive_predictive_value,
}


def group_metrics(y_true, y_pred, sensitive) -> pd.DataFrame:
    """Per-group confusion-derived rates, via fairlearn's MetricFrame.

    The index keeps the name of the attribute it groups by ("race", "sex"). Fairlearn
    would otherwise label it `sensitive_feature_0`, which silently breaks any caller
    that reset_index()es and then refers to the column by its real name.
    """
    frame = MetricFrame(
        metrics=GROUP_METRICS,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        sensitive_features=np.asarray(sensitive),
    )
    by_group = frame.by_group
    by_group.index.name = getattr(sensitive, "name", None) or "group"
    return by_group


def score_metrics(y_true, y_score, sensitive) -> pd.DataFrame:
    """Threshold-free per-group metrics: discrimination and calibration quality."""
    df = pd.DataFrame(
        {"y": np.asarray(y_true), "s": np.asarray(y_score), "g": np.asarray(sensitive)}
    )
    rows = {}
    for group, chunk in df.groupby("g"):
        auc = roc_auc_score(chunk["y"], chunk["s"]) if chunk["y"].nunique() > 1 else float("nan")
        rows[group] = {
            "n": len(chunk),
            "auc": float(auc),
            "brier": float(brier_score_loss(chunk["y"], chunk["s"]))
            if chunk["y"].nunique() > 1
            else float("nan"),
            "mean_score": float(chunk["s"].mean()),
            "observed_rate": float(chunk["y"].mean()),
        }
    return pd.DataFrame(rows).T


def is_degenerate(y_pred) -> bool:
    """True when the operating point detains (or releases) essentially everyone.

    Fairness metrics are meaningless there: a model that flags all defendants has a
    zero FPR gap by construction. This has to be checked, not assumed away -- a
    regularised model whose scores never fall below the threshold will otherwise look
    perfectly fair while being useless.
    """
    rate = float(np.mean(np.asarray(y_pred)))
    return rate >= 0.99 or rate <= 0.01


# A *claimed mitigation* is held to a stricter standard than a deployed operating point.
# 0.99 asks "is this model doing anything at all"; 0.95 asks "did this intervention buy
# fairness honestly, or by detaining nearly everyone". A counterfactual that detains 98%
# of defendants scores a near-zero gap on every group metric, and reporting that as a
# successful mitigation would be worse than reporting none.
MITIGATION_DEGENERATE_RATE = 0.95


def is_degenerate_rate(rate: float, bar: float = MITIGATION_DEGENERATE_RATE) -> bool:
    """Degeneracy test applied to an already-computed selection rate."""
    return rate >= bar or rate <= 1 - bar


def disparity_summary(y_true, y_pred, sensitive, groups=PRIMARY_RACES) -> dict[str, float]:
    """The headline gaps, computed between two named groups.

    Reported as differences between a specific pair rather than fairlearn's max-min
    difference, because with n=11 Native American defendants a max-min statistic is
    driven entirely by sampling noise in the smallest group.
    """
    by_group = group_metrics(y_true, y_pred, sensitive)
    a, b = groups
    missing = [g for g in groups if g not in by_group.index]
    if missing:
        raise KeyError(f"Groups absent from the data: {missing}")

    unprivileged, privileged = by_group.loc[a], by_group.loc[b]
    selection_ratio = (
        unprivileged["selection_rate"] / privileged["selection_rate"]
        if privileged["selection_rate"] > 0
        else float("nan")
    )
    return {
        "group_a": a,
        "group_b": b,
        "selection_rate": float(np.mean(np.asarray(y_pred))),
        "degenerate_operating_point": is_degenerate(y_pred),
        "demographic_parity_difference": float(
            unprivileged["selection_rate"] - privileged["selection_rate"]
        ),
        "disparate_impact_ratio": float(selection_ratio),
        "passes_four_fifths_rule": bool(selection_ratio >= FOUR_FIFTHS),
        "fpr_difference": float(unprivileged["fpr"] - privileged["fpr"]),
        "fnr_difference": float(unprivileged["fnr"] - privileged["fnr"]),
        "equal_opportunity_difference": float(unprivileged["tpr"] - privileged["tpr"]),
        "equalized_odds_difference": float(
            max(
                abs(unprivileged["tpr"] - privileged["tpr"]),
                abs(unprivileged["fpr"] - privileged["fpr"]),
            )
        ),
        "ppv_difference": float(unprivileged["ppv"] - privileged["ppv"]),
        "base_rate_difference": float(unprivileged["base_rate"] - privileged["base_rate"]),
    }


def calibration_by_group(
    y_true, y_score, sensitive, n_bins: int = 5, min_bin: int = 20
) -> pd.DataFrame:
    """Observed re-offence rate per score band per group.

    This is the table that settles the ProPublica/Northpointe argument: if the observed
    rates line up across groups within a band, the score is calibrated, whatever the
    error-rate gaps look like.
    """
    df = pd.DataFrame(
        {"y": np.asarray(y_true), "s": np.asarray(y_score), "g": np.asarray(sensitive)}
    )
    df["band"] = pd.cut(df["s"], np.linspace(0, 1, n_bins + 1), include_lowest=True)

    out = (
        df.groupby(["band", "g"], observed=True)
        .agg(n=("y", "size"), predicted=("s", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    return out[out["n"] >= min_bin]


def impossibility_evidence(y_true, y_score, sensitive, groups=PRIMARY_RACES) -> pd.DataFrame:
    """Sweep thresholds and show calibration and equalised odds pulling apart.

    At no threshold can both the FPR gap and the PPV gap be driven to zero while base
    rates differ. Showing this empirically is more convincing to a client than citing
    Chouldechova (2017).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    sensitive = np.asarray(sensitive)

    rows = []
    for threshold in np.linspace(0.05, 0.95, 91):
        y_pred = (y_score >= threshold).astype(int)
        try:
            summary = disparity_summary(y_true, y_pred, sensitive, groups)
        except (KeyError, ZeroDivisionError):
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "fpr_gap": summary["fpr_difference"],
                "fnr_gap": summary["fnr_difference"],
                "ppv_gap": summary["ppv_difference"],
                "selection_rate_gap": summary["demographic_parity_difference"],
                "accuracy": float((y_true == y_pred).mean()),
            }
        )
    return pd.DataFrame(rows)


# ============================================================================ inference
#
# Everything above returns a point estimate. A gap of +0.177 with no standard error cannot
# support a deployment decision: the reader has no way to tell it from +0.02 measured on a
# small group. The functions below supply the missing inference, in the three forms the
# course specifies -- a formal test, a confidence interval, and an equivalence test.


def chi2_statistical_parity(y_pred, sensitive, groups=PRIMARY_RACES) -> dict[str, float]:
    """Chi-squared test of independence between the decision and the protected attribute.

    This is the test behind statistical parity: under H0 the selection rate is the same in
    every group, so decisions and group membership are independent in the contingency table.
    """
    from scipy.stats import chi2_contingency

    sensitive = pd.Series(np.asarray(sensitive)).reset_index(drop=True)
    y_pred = pd.Series(np.asarray(y_pred)).reset_index(drop=True)
    mask = sensitive.isin(groups)

    table = pd.crosstab(sensitive[mask], y_pred[mask])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"statistic": np.nan, "p_value": np.nan, "dof": 0, "reject_fairness": False}

    statistic, p_value, dof, _ = chi2_contingency(table)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "dof": int(dof),
        # The slide's traffic light: red when the null of parity is rejected at 5%.
        "reject_fairness": bool(p_value < 0.05),
    }


def cmh_test(y_pred, sensitive, strata, groups=PRIMARY_RACES) -> dict[str, float]:
    """Cochran-Mantel-Haenszel: does the disparity survive conditioning on risk?

    This is *conditional* statistical parity. Stratifying by a legitimate risk factor asks
    a sharper question than the plain chi-squared test: not "are the groups treated
    differently?" but "are they treated differently *among defendants who look alike?*".
    A disparity that vanishes under stratification was confounded by the stratifier; one
    that survives it was not.
    """
    from scipy.stats import chi2

    sensitive = pd.Series(np.asarray(sensitive)).reset_index(drop=True)
    y_pred = pd.Series(np.asarray(y_pred)).reset_index(drop=True)
    strata = pd.Series(np.asarray(strata)).reset_index(drop=True)
    mask = sensitive.isin(groups)

    numerator, variance = 0.0, 0.0
    n_strata = 0
    for _, index in strata[mask].groupby(strata[mask]).groups.items():
        table = pd.crosstab(sensitive[index], y_pred[index])
        if table.shape != (2, 2) or table.to_numpy().sum() < 2:
            continue
        a = float(table.iloc[0, 1])
        row_totals = table.sum(axis=1).to_numpy(dtype=float)
        col_totals = table.sum(axis=0).to_numpy(dtype=float)
        total = float(table.to_numpy().sum())
        if total <= 1:
            continue
        expected = row_totals[0] * col_totals[1] / total
        stratum_variance = (row_totals[0] * row_totals[1] * col_totals[0] * col_totals[1]) / (
            total**2 * (total - 1)
        )
        numerator += a - expected
        variance += stratum_variance
        n_strata += 1

    if variance <= 0:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_strata": n_strata,
            "reject_fairness": False,
        }

    # Continuity-corrected Mantel-Haenszel statistic, 1 degree of freedom.
    statistic = (abs(numerator) - 0.5) ** 2 / variance
    p_value = float(chi2.sf(statistic, 1))
    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "n_strata": n_strata,
        "reject_fairness": bool(p_value < 0.05),
    }


def bootstrap_disparities(
    y_true,
    y_score,
    sensitive,
    threshold: float,
    n_boot: int = 1000,
    groups=PRIMARY_RACES,
    seed: int = 42,
) -> pd.DataFrame:
    """Resample the *scored* holdout to get a sampling distribution for every disparity.

    Resampling scores rather than refitting is the right unit here: the question is how
    precisely this deployed model's disparity is measured on a sample of this size, not how
    much the disparity would move if the model were retrained.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    sensitive = pd.Series(np.asarray(sensitive)).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws = []
    for _ in range(n_boot):
        rows = rng.integers(0, n, n)
        summary = disparity_summary(
            y_true[rows], (y_score[rows] >= threshold).astype(int), sensitive.iloc[rows], groups
        )
        draws.append(summary)
    return pd.DataFrame(draws)


def disparity_inference(
    point_estimates: dict[str, float],
    draws: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Standard error, confidence interval and p-value for each disparity measure."""
    rows = []
    for metric, estimate in point_estimates.items():
        if not isinstance(estimate, (int, float)) or isinstance(estimate, bool):
            continue
        if metric not in draws.columns:
            continue
        sample = draws[metric].dropna().to_numpy(dtype=float)
        if len(sample) < 2:
            continue
        # Null of no disparity: for a difference that is 0, for a ratio it is 1.
        null = 1.0 if "ratio" in metric else 0.0
        centred = sample - sample.mean()
        # Bootstrap p-value: how often a draw centred at the null is at least as extreme
        # as the observed estimate.
        p_value = float((np.abs(centred) >= abs(estimate - null)).mean())
        rows.append(
            {
                "metric": metric,
                "estimate": float(estimate),
                "std_error": float(sample.std(ddof=1)),
                "ci_lower": float(np.quantile(sample, alpha / 2)),
                "ci_upper": float(np.quantile(sample, 1 - alpha / 2)),
                "p_value": p_value,
                "significant": bool(p_value < alpha),
            }
        )
    return pd.DataFrame(rows)


def equivalence_test(draws: np.ndarray, delta: float, alpha: float = 0.05) -> dict[str, float]:
    """Schuirmann's two one-sided tests (TOST) for *fairness equivalence*.

    A conventional test has H0 = "fair", so failing to reject it never certifies fairness --
    it only says the evidence of unfairness was too weak, which is exactly what a small
    sample produces. And with a large sample any trivial gap becomes significant. TOST
    reverses the burden:

        H0: |theta| >= delta   (unfair)        H1: |theta| < delta   (fair)

    H0 is rejected -- the model is *certified* fair at tolerance delta -- only when the
    (1 - 2*alpha) interval lies entirely inside (-delta, delta). ``minimum_delta`` reports
    the tightest tolerance at which that would happen, which is more useful than any single
    delta because it does not depend on a number chosen after seeing the data.
    """
    draws = np.asarray(draws, dtype=float)
    draws = draws[~np.isnan(draws)]
    lower = float(np.quantile(draws, alpha))
    upper = float(np.quantile(draws, 1 - alpha))
    certified = bool(lower > -delta and upper < delta)
    return {
        "delta": float(delta),
        "estimate": float(draws.mean()),
        "tost_lower": lower,
        "tost_upper": upper,
        "certified_fair": certified,
        "minimum_delta": float(max(abs(lower), abs(upper))),
        "verdict": "fair (equivalent)" if certified else "cannot certify fair",
    }


# ------------------------------------------------------- fairness interpretability: FPDP


def fairness_pdp(
    predict_fn,
    X: pd.DataFrame,
    y_true,
    sensitive,
    feature: str,
    threshold: float,
    groups=PRIMARY_RACES,
    grid=None,
) -> pd.DataFrame:
    """Fairness Partial Dependence Plot: the fairness *test statistic* against a feature.

    An ordinary PDP sweeps a feature and plots the prediction. The FPDP sweeps a feature and
    plots the p-value of the fairness test, recomputed on a counterfactual population where
    everybody takes that value. The reading is the one from the lecture: a feature whose
    curve stays below alpha at every level cannot be the source of the disparity, while a
    feature whose curve crosses above alpha is a **candidate variable** -- fixing it would
    make the model statistically fair.
    """
    levels = sorted(X[feature].unique()) if grid is None else list(grid)
    rows = []
    for level in levels:
        counterfactual = X.copy()
        counterfactual[feature] = level
        scores = np.asarray(predict_fn(counterfactual), dtype=float)
        decisions = (scores >= threshold).astype(int)

        test = chi2_statistical_parity(decisions, sensitive, groups)
        summary = disparity_summary(y_true, decisions, sensitive, groups)
        rows.append(
            {
                "feature": feature,
                "level": float(level),
                "p_value": test["p_value"],
                "statistic": test["statistic"],
                "fpr_difference": summary["fpr_difference"],
                "demographic_parity_difference": summary["demographic_parity_difference"],
                "selection_rate": summary["selection_rate"],
            }
        )
    return pd.DataFrame(rows)


def candidate_variables(fpdp: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Which features can buy fairness, and at which level.

    A candidate variable is one where some level lifts the fairness test's p-value above
    alpha. The level with the largest p-value is the one worth trying in mitigation.
    """
    rows = []
    for feature, block in fpdp.groupby("feature"):
        best = block.loc[block["p_value"].idxmax()]
        # A level that reaches parity by detaining (or releasing) virtually everyone has
        # not made the model fair, it has made it useless -- the groups are treated alike
        # because nobody is being distinguished at all. Flagging it keeps a degenerate
        # "fix" from being reported as a mitigation.
        degenerate = is_degenerate_rate(float(best["selection_rate"]))
        rows.append(
            {
                "feature": feature,
                "best_level": float(best["level"]),
                "best_p_value": float(best["p_value"]),
                "baseline_p_value": float(block["p_value"].min()),
                "selection_rate_at_best_level": float(best["selection_rate"]),
                "degenerate_at_best_level": degenerate,
                "is_candidate": bool(best["p_value"] > alpha and not degenerate),
                "fpr_at_best_level": float(best["fpr_difference"]),
            }
        )
    return pd.DataFrame(rows).sort_values("best_p_value", ascending=False).reset_index(drop=True)
