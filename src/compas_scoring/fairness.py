"""The shared fairness protocol: every model is tested the same way, from its predictions.

Settings live in [tool.compas_scoring.fairness]. A model's fairness is fully determined by its
scores on a test set: pass ``y_true``, ``y_score``, the group labels (``Dataset.groups``, or the
same columns of a saved predictions file) and ``proxy_strata(index)`` to ``fairness_table``.

For each threshold (0.5 and the cost break-even) and each comparison (a protected group
against its reference; African-American vs Caucasian is primary):

=================================  ===================================================
metric                             test
=================================  ===================================================
statistical_parity                 two-proportion z-test on the flag-rate gap (equals
                                   Pearson chi2 without continuity correction), with the
                                   Hurlin likelihood-ratio test alongside; TOST
conditional_statistical_parity     Hurlin LR test summed over strata of the race proxies
                                   (priors band x age band x charge degree), plus CMH, the
                                   Mantel-Haenszel common odds ratio and Breslow-Day
fpr (equalized odds, negatives)    z-test on the false-positive-rate gap; TOST
fnr (equalized odds, positives)    z-test on the false-negative-rate gap; TOST
equalized_odds                     Hurlin LR test of Y_hat independent of D given Y
=================================  ===================================================

Conventions: Y_hat = 1 means flagged as likely to re-offend; every gap reads protected minus
reference; p-values are Holm-corrected across comparisons within each (threshold, metric).
Conditioning on the race proxies isolates race: a gap that survives among defendants with
the same priors, age band and charge degree cannot be explained by those proxies.

The likelihood-ratio test follows Hurlin, Perignon & Saurin, "The Fairness of Credit Scoring
Models" (as implemented by the XGBoost group); TOST follows the course (Schuirmann 1987, with
the unpooled variance of the difference).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

from compas_scoring.config import CONFIG
from compas_scoring.data import group_frame, load_dated, load_raw

DEGENERATE_RATE = 0.99  # flagging (or releasing) at least this share: nobody is distinguished


def thresholds() -> dict[str, float]:
    """The two operating points every model is assessed at."""
    return {"0.5": 0.5, "break_even": CONFIG.costs.break_even}


# ------------------------------------------------------------------------------ inputs


def proxy_strata(index, cohort: str = "modelling") -> pd.Series:
    """Conditioning strata for conditional statistical parity: the race proxies.

    Priors band x age band x charge degree, from each defendant's actual record, so the strata
    are the same whatever features a model was given (a race-blind model is conditioned on the
    same proxies as a race-aware one) and stay fixed when the FPDP overrides a feature.
    ``cohort`` is "modelling" (the 6,172-row table) or "dated" (the time-ordered designs).
    """
    raw = {"modelling": load_raw, "dated": load_dated}[cohort]().loc[index]
    groups = group_frame(raw)
    edges = [-np.inf, *CONFIG.fairness.priors_band_edges, np.inf]
    labels = _band_labels(CONFIG.fairness.priors_band_edges)
    band = pd.cut(raw["Number_of_Priors"], edges, labels=labels).astype(str)
    return (band + " priors | " + groups["age_band"] + " | " + groups["charge_degree"]).rename(
        "stratum"
    )


def _band_labels(edges) -> list[str]:
    """Readable labels for integer bands with these upper edges: [0, 3] -> 0 | 1-3 | 4+."""
    labels, low = [], None
    for high in edges:
        start = 0 if low is None else low + 1
        labels.append(str(high) if start == high else f"{start}-{high}")
        low = high
    return [*labels, f"{low + 1}+"]


def comparison_rows(groups: pd.DataFrame, comparison) -> tuple[pd.Index, pd.Series]:
    """Rows in either group of a comparison, and whether each is in the protected group."""
    attribute, protected, reference = comparison
    labels = groups[attribute]
    keep = labels.isin([protected, reference]) & ~labels.isin(CONFIG.fairness.excluded_groups)
    rows = labels.index[keep]
    return rows, (labels.loc[rows] == protected)


# ------------------------------------------------------------------------------- tests


def _g_statistic(table: np.ndarray) -> float:
    """Likelihood-ratio (G) statistic of independence for one 2x2 table."""
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    observed = table > 0  # 0 * log(0) = 0
    return float(2 * (table[observed] * np.log(table[observed] / expected[observed])).sum())


def _informative_tables(y_pred, protected, strata) -> list[np.ndarray]:
    """One 2x2 table (rows: flagged 0/1, columns: reference/protected) per informative stratum.

    A stratum where one group is absent, or where everyone gets the same decision, carries no
    information about the null and is dropped.
    """
    frame = pd.DataFrame({"y": np.asarray(y_pred), "d": np.asarray(protected, dtype=bool)})
    frame["s"] = "all" if strata is None else np.asarray(strata)
    tables = []
    for _, part in frame.groupby("s", sort=True):
        table = pd.crosstab(part["y"], part["d"]).reindex(
            index=[0, 1], columns=[False, True], fill_value=0
        )
        table = table.to_numpy(dtype=float)
        if (table.sum(axis=0) > 0).all() and (table.sum(axis=1) > 0).all():
            tables.append(table)
    return tables


def hurlin_test(y_pred, protected, strata=None) -> dict[str, float]:
    """H0: Y_hat independent of D (given strata). LR statistics summed over strata, chi2(q).

    Without strata this is statistical parity (1 df); with strata, conditional statistical
    parity (q = informative strata). Detects a difference in any stratum, in either direction.
    """
    tables = _informative_tables(y_pred, protected, strata)
    statistic = float(sum(_g_statistic(t) for t in tables))
    df = max(len(tables), 1)
    return {
        "statistic": statistic,
        "df": df,
        "p_value": float(stats.chi2.sf(statistic, df)) if tables else 1.0,
        "strata_used": len(tables),
    }


def cmh_test(y_pred, protected, strata) -> dict[str, float]:
    """Cochran-Mantel-Haenszel (1 df, no continuity correction) and the common odds ratio.

    More powerful than the summed test when the disparity points the same way in every
    stratum; the Mantel-Haenszel odds ratio is the effect size ("among defendants with the same
    proxies, the protected group has OR times the odds of being flagged"). Breslow-Day tests
    whether that single odds ratio fits every stratum.
    """
    # statsmodels wants rows = flagged (1, 0), columns = group (protected, reference).
    tables = [t[::-1, ::-1] for t in _informative_tables(y_pred, protected, strata)]
    empty = {k: np.nan for k in ("statistic", "p_value", "odds_ratio", "or_low", "or_high")}
    if not tables:
        return {**empty, "breslow_day_p": np.nan, "strata_used": 0}
    table = StratifiedTable(tables)
    test = table.test_null_odds(correction=False)
    low, high = table.oddsratio_pooled_confint()
    try:
        # Identical odds ratios in every stratum make Breslow-Day divide by zero: p is nan.
        with np.errstate(divide="ignore", invalid="ignore"):
            breslow_day = float(table.test_equal_odds().pvalue)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        breslow_day = np.nan
    return {
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "odds_ratio": float(table.oddsratio_pooled),
        "or_low": float(low),
        "or_high": float(high),
        "breslow_day_p": breslow_day,
        "strata_used": len(tables),
    }


def rate_gap(outcome, protected, alpha: float | None = None, delta: float | None = None) -> dict:
    """Gap in a rate (protected - reference): z-test, confidence interval and TOST.

    ``outcome`` is the 0/1 event whose rate is compared (flagged, or flagged among the
    negatives for FPR, or missed among the positives for FNR).

    * The z-test of H0: gap = 0 uses the pooled variance, so z^2 equals Pearson's chi2.
    * The confidence interval and TOST use the unpooled variance of the difference, as in the
      course: H0: |gap| >= delta (unfair) is rejected, and fairness certified, only when both
      z^L = (gap + delta) / s and z^U = (delta - gap) / s exceed t(1 - alpha, n - 2).
      ``minimum_delta`` is the tightest tolerance at which that would happen.
    """
    alpha = CONFIG.fairness.alpha if alpha is None else alpha
    delta = CONFIG.fairness.tost_delta if delta is None else delta
    outcome = np.asarray(outcome, dtype=float)
    protected = np.asarray(protected, dtype=bool)
    x1, x0 = outcome[protected], outcome[~protected]
    n1, n0 = len(x1), len(x0)
    nan = {"gap": np.nan, "ci_low": np.nan, "ci_high": np.nan, "z": np.nan, "p_value": np.nan,
           "tost_p": np.nan, "certified_fair": False, "minimum_delta": np.nan}  # fmt: skip
    if n1 == 0 or n0 == 0:
        return {**nan, "rate_protected": np.nan, "rate_reference": np.nan, "n_protected": n1,
                "n_reference": n0}  # fmt: skip
    p1, p0 = x1.mean(), x0.mean()
    gap = p1 - p0
    base = {"rate_protected": p1, "rate_reference": p0, "n_protected": n1, "n_reference": n0}

    pooled = (x1.sum() + x0.sum()) / (n1 + n0)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    if se_pooled == 0 or se == 0:  # everyone has the same outcome: no variation to test
        return {**nan, **base, "gap": gap, "p_value": 1.0 if gap == 0 else np.nan}

    z = gap / se_pooled
    z_crit = stats.norm.ppf(1 - alpha / 2)
    t_crit = stats.t.ppf(1 - alpha, n1 + n0 - 2)
    z_lower, z_upper = (gap + delta) / se, (delta - gap) / se
    tost_p = max(stats.t.sf(z_lower, n1 + n0 - 2), stats.t.sf(z_upper, n1 + n0 - 2))
    return {
        **base,
        "gap": gap,
        "ci_low": gap - z_crit * se,
        "ci_high": gap + z_crit * se,
        "z": z,
        "p_value": float(2 * stats.norm.sf(abs(z))),
        "tost_p": float(tost_p),
        "certified_fair": bool(z_lower > t_crit and z_upper > t_crit),
        "minimum_delta": float(abs(gap) + t_crit * se),
    }


# -------------------------------------------------------------------------- the table


def _is_degenerate(y_pred) -> bool:
    rate = float(np.mean(y_pred))
    return rate >= DEGENERATE_RATE or rate <= 1 - DEGENERATE_RATE


def fairness_table(
    y_true, y_score, groups: pd.DataFrame, strata: pd.Series, at: dict[str, float] | None = None
) -> pd.DataFrame:
    """Every protocol metric, for every threshold and comparison, as one long table.

    ``y_true``, ``y_score``, ``groups`` and ``strata`` must share an index (the defendants).
    ``at`` overrides the thresholds ({name: value}); by default the protocol's two.
    """
    y_true = pd.Series(np.asarray(y_true), index=groups.index)
    y_score = pd.Series(np.asarray(y_score, dtype=float), index=groups.index)
    strata = strata.reindex(groups.index)

    rows = []
    for t_name, t in (at or thresholds()).items():
        y_pred = (y_score >= t).astype(int)
        for comparison in CONFIG.fairness.comparisons:
            index, protected = comparison_rows(groups, comparison)
            pred, truth, strat = y_pred.loc[index], y_true.loc[index], strata.loc[index]
            common = {
                "threshold_name": t_name,
                "threshold": t,
                "comparison": f"{comparison[1]} vs {comparison[2]}",
                "attribute": comparison[0],
                "primary": comparison == CONFIG.fairness.comparisons[0],
                "selection_rate": float(pred.mean()),
                "degenerate": _is_degenerate(pred),
            }

            sp_z = rate_gap(pred, protected)
            sp_h = hurlin_test(pred, protected)
            rows.append({**common, "metric": "statistical_parity", "test": "z (pooled)", **sp_z,
                         "hurlin_statistic": sp_h["statistic"], "hurlin_df": sp_h["df"],
                         "hurlin_p": sp_h["p_value"]})  # fmt: skip

            csp = hurlin_test(pred, protected, strat)
            cmh = cmh_test(pred, protected, strat)
            rows.append({**common, "metric": "conditional_statistical_parity",
                         "test": "Hurlin LR over proxy strata",
                         "rate_protected": sp_z["rate_protected"],
                         "rate_reference": sp_z["rate_reference"],
                         "n_protected": sp_z["n_protected"], "n_reference": sp_z["n_reference"],
                         "p_value": csp["p_value"], "hurlin_statistic": csp["statistic"],
                         "hurlin_df": csp["df"], "hurlin_p": csp["p_value"],
                         "strata_used": csp["strata_used"], "cmh_statistic": cmh["statistic"],
                         "cmh_p": cmh["p_value"], "mh_odds_ratio": cmh["odds_ratio"],
                         "mh_or_low": cmh["or_low"], "mh_or_high": cmh["or_high"],
                         "breslow_day_p": cmh["breslow_day_p"]})  # fmt: skip

            for metric, mask, event in (
                ("fpr", truth == 0, pred == 1),  # flagged among those who did not re-offend
                ("fnr", truth == 1, pred == 0),  # released among those who did
            ):
                gap = rate_gap(event[mask].astype(int), protected[mask])
                rows.append({**common, "metric": metric, "test": "z (pooled)", **gap})

            eo = hurlin_test(pred, protected, truth)
            rows.append({**common, "metric": "equalized_odds", "test": "Hurlin LR given Y",
                         "p_value": eo["p_value"], "hurlin_statistic": eo["statistic"],
                         "hurlin_df": eo["df"], "hurlin_p": eo["p_value"],
                         "n_protected": int(protected.sum()),
                         "n_reference": int((~protected).sum())})  # fmt: skip

    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    for _, family in table.groupby(["threshold_name", "metric"]):
        valid = family["p_value"].notna()
        if valid.any():
            adjusted = multipletests(family.loc[valid, "p_value"], method="holm")[1]
            table.loc[family.index[valid], "p_holm"] = adjusted
    table["reject_fairness"] = table["p_holm"] < CONFIG.fairness.alpha
    return table


# ------------------------------------------------------------------ extras, any model


def calibration_by_group(y_true, y_score, sensitive, n_bins: int = 5, min_bin: int = 20):
    """Observed re-offence rate per score band per group (Northpointe's calibration argument).

    If the observed rates line up across groups within a band, the score is calibrated,
    whatever the error-rate gaps look like. From the TabPFN analysis (E. Franco-Tetu).
    """
    df = pd.DataFrame(
        {"y": np.asarray(y_true), "s": np.asarray(y_score, dtype=float), "g": np.asarray(sensitive)}
    )
    df["band"] = pd.cut(df["s"], np.linspace(0, 1, n_bins + 1), include_lowest=True)
    out = (
        df.groupby(["band", "g"], observed=True)
        .agg(n=("y", "size"), predicted=("s", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    return out[out["n"] >= min_bin]


def impossibility_sweep(y_true, y_score, groups: pd.DataFrame, comparison=None) -> pd.DataFrame:
    """FPR, FNR, PPV and flag-rate gaps across thresholds (Chouldechova's impossibility).

    With unequal base rates no threshold drives the FPR gap and the PPV gap to zero together;
    this shows it on the data. From the TabPFN analysis (E. Franco-Tetu).
    """
    comparison = comparison or CONFIG.fairness.comparisons[0]
    index, protected = comparison_rows(groups, comparison)
    y = np.asarray(pd.Series(np.asarray(y_true), index=groups.index).loc[index])
    s = np.asarray(pd.Series(np.asarray(y_score, dtype=float), index=groups.index).loc[index])
    d = protected.to_numpy()

    def rate(event, mask):
        return float(event[mask].mean()) if mask.any() else np.nan

    rows = []
    for t in np.linspace(0.05, 0.95, 91):
        f = s >= t
        gaps = {}
        for name, event, cond in (
            ("fpr", f, y == 0),
            ("fnr", ~f, y == 1),
            ("ppv", y == 1, f),
            ("selection_rate", f, np.ones_like(f)),
        ):
            gaps[f"{name}_gap"] = rate(event, cond & d) - rate(event, cond & ~d)
        rows.append({"threshold": float(t), **gaps, "accuracy": float((f == (y == 1)).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ FPDP


def fpdp(
    predict_fn,
    X: pd.DataFrame,
    groups: pd.DataFrame,
    strata: pd.Series,
    feature: str,
    threshold: float,
    comparison=None,
    grid=None,
    conditional: bool = False,
) -> pd.DataFrame:
    """Fairness Partial Dependence Plot data for one feature (course step 2).

    Every defendant's ``feature`` is set to each value in ``grid``, the model re-predicts, and
    the Hurlin test is re-run (``conditional=True``: on the proxy strata, which stay fixed at
    each defendant's actual record). ``predict_fn`` maps a feature frame to scores. Each grid
    value is one model call: pass a coarse grid for priors on a slow model.
    """
    comparison = comparison or CONFIG.fairness.comparisons[0]
    index, protected = comparison_rows(groups, comparison)
    X_rows = X.loc[index]
    grid = np.unique(X[feature]) if grid is None else grid
    rows = []
    for value in grid:
        counterfactual = X_rows.copy()
        counterfactual[feature] = value
        pred = (np.asarray(predict_fn(counterfactual), dtype=float) >= threshold).astype(int)
        test = hurlin_test(pred, protected, strata.loc[index] if conditional else None)
        rows.append({"feature": feature, "value": float(value), "p_value": test["p_value"],
                     "statistic": test["statistic"], "selection_rate": float(pred.mean()),
                     "degenerate": _is_degenerate(pred)})  # fmt: skip
    return pd.DataFrame(rows)


def candidate_variables(curves: pd.DataFrame, baseline_p: float, alpha: float | None = None):
    """Features whose FPDP lifts the fairness test's p-value above alpha (course step 2).

    Only defined when fairness is rejected on the actual data (``baseline_p`` < alpha), so the
    baseline is passed in explicitly. Values at which everyone gets the same decision are
    ignored: they pass the test trivially without saying anything about the feature.
    """
    alpha = CONFIG.fairness.alpha if alpha is None else alpha
    rows = []
    for feature, curve in curves.groupby("feature"):
        usable = curve[~curve["degenerate"]]
        best = usable.loc[usable["p_value"].idxmax()] if len(usable) else None
        rows.append(
            {
                "feature": feature,
                "baseline_p": baseline_p,
                "fairness_rejected": baseline_p < alpha,
                "best_value": np.nan if best is None else best["value"],
                "best_p": np.nan if best is None else best["p_value"],
                "is_candidate": bool(
                    baseline_p < alpha and best is not None and best["p_value"] > alpha
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("best_p", ascending=False).reset_index(drop=True)
