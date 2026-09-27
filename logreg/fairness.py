"""Fairness tests for the logistic regression: z-tests, Hurlin's summed LR, and Mantel-Haenszel."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from statsmodels.stats.contingency_tables import StratifiedTable

PROTECTED = "African-American"
COMPARISONS = {
    "aa_vs_others": "all other groups",
    "aa_vs_caucasian": "Caucasian",
}
PRIORS_BINS = [-0.1, 0, 1, 3, 6, np.inf]
PRIORS_LABELS = ["0", "1", "2-3", "4-6", "7+"]


def comparison_groups(race, comparison: str) -> tuple[np.ndarray, np.ndarray]:
    """Rows in the comparison, and which of them belong to the protected group.

    Returns ``(in_scope, is_protected)`` as boolean arrays over every row. For
    ``aa_vs_caucasian`` the other groups fall out of scope; for ``aa_vs_others`` every row is
    in scope and the reference group is everybody who is not African-American.
    """
    race = np.asarray(race)
    is_protected = race == PROTECTED
    if comparison == "aa_vs_others":
        in_scope = np.ones(len(race), dtype=bool)
    elif comparison == "aa_vs_caucasian":
        in_scope = is_protected | (race == "Caucasian")
    else:
        raise KeyError(f"Unknown comparison {comparison!r}; available: {sorted(COMPARISONS)}")
    return in_scope, is_protected


def comparison_labels(race, comparison: str) -> tuple[np.ndarray, tuple[str, str]]:
    """Race labels and the (protected, reference) pair to hand to `compas_scoring.fairness`.

    The shared functions compare two *named* groups. For ``aa_vs_caucasian`` those names are
    already in the data; for ``aa_vs_others`` every non-African-American defendant is
    relabelled into one reference group so the same functions can be reused unchanged.
    """
    race = np.asarray(race, dtype=object)
    if comparison == "aa_vs_others":
        reference = COMPARISONS[comparison]
        return np.where(race == PROTECTED, PROTECTED, reference), (PROTECTED, reference)
    comparison_groups(race, comparison)  # raises on an unknown comparison
    return race, (PROTECTED, COMPARISONS[comparison])


def priors_band(priors) -> pd.Series:
    """Priors grouped so each stratum holds enough defendants for a 2x2 table to mean anything."""
    return pd.cut(pd.Series(np.asarray(priors)), bins=PRIORS_BINS, labels=PRIORS_LABELS)


def two_proportion_ztest(
    successes_a: int, n_a: int, successes_b: int, n_b: int, alpha: float = 0.05
) -> dict[str, float]:
    """z-test of H0: p_a = p_b, with a Wald confidence interval on p_a - p_b.

    The test uses the pooled standard error (the one implied by H0); the interval uses the
    unpooled one, since it describes the gap rather than testing it. On a 2x2 table z**2
    equals the uncorrected chi-squared statistic, so this is the same test as chi-squared,
    reported as a signed gap with an interval instead of a bare p-value.
    """
    if min(n_a, n_b) == 0:
        raise ValueError("both groups need at least one observation")
    p_a, p_b = successes_a / n_a, successes_b / n_b
    difference = p_a - p_b

    pooled = (successes_a + successes_b) / (n_a + n_b)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se_pooled == 0:
        z, p_value = 0.0, 1.0
    else:
        z = difference / se_pooled
        p_value = float(2 * norm.sf(abs(z)))

    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    margin = norm.ppf(1 - alpha / 2) * se
    return {
        "rate_a": float(p_a),
        "rate_b": float(p_b),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "difference": float(difference),
        "ci_lower": float(difference - margin),
        "ci_upper": float(difference + margin),
        "z": float(z),
        "p_value": p_value,
        "reject_fairness": bool(p_value < alpha),
    }


def _rate_test(outcome, member, eligible, alpha: float) -> dict[str, float]:
    """z-test on P(outcome) between protected and reference rows among the eligible ones."""
    outcome = np.asarray(outcome, dtype=int)[eligible]
    member = member[eligible]
    return two_proportion_ztest(
        int(outcome[member].sum()),
        int(member.sum()),
        int(outcome[~member].sum()),
        int((~member).sum()),
        alpha,
    )


def statistical_parity_ztest(y_pred, race, comparison: str, alpha: float = 0.05) -> dict:
    """Selection-rate gap between African-American defendants and the reference group."""
    in_scope, is_protected = comparison_groups(race, comparison)
    return {
        "criterion": "statistical_parity",
        "metric": "selection_rate",
        **_rate_test(y_pred, is_protected, in_scope, alpha),
    }


def equalized_odds_ztests(y_true, y_pred, race, comparison: str, alpha: float = 0.05) -> list:
    """False-positive-rate gap (among y = 0) and false-negative-rate gap (among y = 1).

    Equalized odds holds when both gaps are zero: people with the same true outcome are
    flagged at the same rate whatever their group.
    """
    in_scope, is_protected = comparison_groups(race, comparison)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return [
        {
            "criterion": "equalized_odds",
            "metric": "fpr",
            **_rate_test(y_pred, is_protected, in_scope & (y_true == 0), alpha),
        },
        {
            "criterion": "equalized_odds",
            "metric": "fnr",
            **_rate_test(1 - y_pred, is_protected, in_scope & (y_true == 1), alpha),
        },
    ]


def stratum_tables(y_pred, race, strata, comparison: str) -> tuple[list[np.ndarray], list]:
    """One 2x2 table per stratum: rows (protected, reference), columns (flagged, not flagged).

    With that orientation an odds ratio above 1 means the protected group is flagged more.
    Strata where either group or either decision is absent carry no information about the
    association and are dropped -- their odds ratio is undefined and their G statistic is 0.
    """
    in_scope, is_protected = comparison_groups(race, comparison)
    y_pred = np.asarray(y_pred, dtype=int)
    strata = pd.Series(np.asarray(strata, dtype=object))

    tables, labels = [], []
    for label in pd.unique(strata[in_scope]):
        rows = in_scope & (strata == label).to_numpy()
        member, flagged = is_protected[rows], y_pred[rows]
        table = np.array(
            [
                [(member & (flagged == 1)).sum(), (member & (flagged == 0)).sum()],
                [(~member & (flagged == 1)).sum(), (~member & (flagged == 0)).sum()],
            ],
            dtype=float,
        )
        if (table.sum(axis=0) > 0).all() and (table.sum(axis=1) > 0).all():
            tables.append(table)
            labels.append(label)
    return tables, labels


def g_statistic(table: np.ndarray) -> float:
    """Likelihood-ratio (G) statistic for independence in one 2x2 table; 0 * log 0 = 0."""
    table = np.asarray(table, dtype=float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    observed = table > 0
    return float(2 * (table[observed] * np.log(table[observed] / expected[observed])).sum())


def summed_lr_test(y_pred, race, strata, comparison: str, alpha: float = 0.05) -> dict:
    """Hurlin's conditional statistical parity test: the sum of per-stratum LR statistics.

    Under H0 -- decision independent of group *within* every stratum -- each stratum's G
    statistic is chi-squared with 1 degree of freedom, so their sum is chi-squared with one
    degree of freedom per informative stratum. Unlike CMH it does not assume the disparity
    points the same way in every stratum: gaps in opposite directions add up here instead
    of cancelling.
    """
    tables, _ = stratum_tables(y_pred, race, strata, comparison)
    if not tables:
        return {"lr_statistic": np.nan, "lr_dof": 0, "lr_p_value": np.nan, "lr_rejects": False}
    statistic = sum(g_statistic(table) for table in tables)
    dof = len(tables)
    p_value = float(chi2.sf(statistic, dof))
    return {
        "lr_statistic": float(statistic),
        "lr_dof": dof,
        "lr_p_value": p_value,
        "lr_rejects": bool(p_value < alpha),
    }


def mantel_haenszel(y_pred, race, strata, comparison: str, alpha: float = 0.05) -> dict:
    """CMH test, Mantel-Haenszel common odds ratio with its CI, and Breslow-Day homogeneity.

    The common odds ratio is the "x times the odds of being flagged, at equal <stratum>"
    number. Breslow-Day asks whether one number is a fair summary: if it rejects, the odds
    ratio differs across strata and the per-stratum ratios should be reported instead.
    """
    # stratum_tables keeps only strata with every margin positive, so each one contributes a
    # nonzero term to the MH numerator or denominator; a zero cell inside one is fine.
    tables, _ = stratum_tables(y_pred, race, strata, comparison)
    if not tables:
        return {
            "cmh_statistic": np.nan,
            "cmh_p_value": np.nan,
            "cmh_rejects": False,
            "mh_odds_ratio": np.nan,
            "mh_or_lower": np.nan,
            "mh_or_upper": np.nan,
            "breslow_day_p_value": np.nan,
            "n_strata": 0,
        }

    stratified = StratifiedTable(np.dstack(tables))
    cmh = stratified.test_null_odds(correction=True)
    lower, upper = stratified.oddsratio_pooled_confint(alpha=alpha)
    # Homogeneity needs at least two odds ratios to compare. statsmodels returns NaN when the
    # pooled odds ratio is exactly 1 (its quadratic degenerates); real samples never hit that.
    breslow_day = stratified.test_equal_odds() if len(tables) > 1 else None
    return {
        "cmh_statistic": float(cmh.statistic),
        "cmh_p_value": float(cmh.pvalue),
        "cmh_rejects": bool(cmh.pvalue < alpha),
        "mh_odds_ratio": float(stratified.oddsratio_pooled),
        "mh_or_lower": float(lower),
        "mh_or_upper": float(upper),
        "breslow_day_p_value": float(breslow_day.pvalue) if breslow_day else np.nan,
        "n_strata": len(tables),
    }


def stratum_odds_ratios(y_pred, race, strata, comparison: str) -> pd.DataFrame:
    """Per-stratum flag rates and odds ratios -- the detail behind the common odds ratio.

    A 0.5 continuity correction is applied only to strata with a zero cell, so every
    stratum gets a finite odds ratio to plot.
    """
    tables, labels = stratum_tables(y_pred, race, strata, comparison)
    rows = []
    for label, table in zip(labels, tables, strict=True):
        cells = table + 0.5 if (table == 0).any() else table
        rows.append(
            {
                "stratum": str(label),
                "n_protected": int(table[0].sum()),
                "n_reference": int(table[1].sum()),
                "flag_rate_protected": float(table[0, 0] / table[0].sum()),
                "flag_rate_reference": float(table[1, 0] / table[1].sum()),
                "odds_ratio": float((cells[0, 0] * cells[1, 1]) / (cells[0, 1] * cells[1, 0])),
            }
        )
    return pd.DataFrame(rows)
