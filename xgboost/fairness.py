"""Fairness of the saved XGBoost models on the shared test set.

Follows Hurlin, Pérignon & Saurin, "The Fairness of Credit Scoring Models" (arXiv 2205.10200):

- Every fairness definition is a (conditional) independence hypothesis H0: Y_hat ⊥ D | C,
  tested with their likelihood-ratio statistic: one 2x2 table (prediction x group) per
  stratum of C, summed over strata, asymptotically chi2(q) with q = max(K, 1).
  Statistical parity has no strata; conditional statistical parity uses the strata below.
- D must be binary, so race is tested one group at a time against Caucasian defendants.
- The Fairness Partial Dependence Plot (FPDP) sets one feature to the same value for every
  defendant, re-predicts, and records the p-value of the test. A feature is a *candidate
  variable* (a driver of the unfairness) if some value lifts the p-value above alpha.

Every function returns data; plotting is left to the notebook. Here Y_hat = 1 means the
model flags the defendant as likely to re-offend.

Run from the repository root:  ``python xgboost/fairness.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from compas_scoring.data import Dataset
from xgb_model import load_data, load_model

# (attribute in Dataset.groups, protected group, reference group)
COMPARISONS = {
    "African-American vs Caucasian": ("race", "African-American", "Caucasian"),
    "Hispanic vs Caucasian": ("race", "Hispanic", "Caucasian"),
    "Other vs Caucasian": ("race", "Other", "Caucasian"),
    "Female vs Male": ("sex", "Female", "Male"),
}
# 11 Asian and 2 Native American defendants in the test set: far too few for the test's
# chi-squared approximation, so they are left out of every fairness result.
EXCLUDED_GROUPS = {"Asian", "Native American"}

PRIORS_BANDS = {"0": (-np.inf, 0), "1-3": (0, 3), "4+": (3, np.inf)}


def predict_labels(model, data: Dataset, threshold: float = 0.5) -> pd.Series:
    """Predicted class (1 = flagged as likely to re-offend), indexed like the data."""
    y_score = model.predict_proba(data.X)[:, 1]
    return pd.Series((y_score >= threshold).astype(int), index=data.X.index, name="y_pred")


def make_strata(data: Dataset) -> pd.Series:
    """Conditioning classes for conditional statistical parity: priors band x charge degree.

    Built from the defendants' actual records, so they stay fixed when the FPDP overrides a
    feature.
    """
    edges = [low for low, _ in PRIORS_BANDS.values()] + [np.inf]
    band = pd.cut(data.X["Number_of_Priors"], bins=edges, labels=list(PRIORS_BANDS))
    strata = band.astype(str) + " priors | " + data.groups["charge_degree"]
    return strata.rename("stratum")


def group_rates(y_pred: pd.Series, data: Dataset, attribute: str) -> pd.DataFrame:
    """Size, flag rate and actual re-offence rate of every (non-excluded) group."""
    group = data.groups[attribute]
    keep = ~group.isin(EXCLUDED_GROUPS)
    frame = pd.DataFrame({"group": group[keep], "y_pred": y_pred[keep], "y_true": data.y[keep]})
    return (
        frame.groupby("group")
        .agg(n=("y_pred", "size"), flag_rate=("y_pred", "mean"), base_rate=("y_true", "mean"))
        .sort_values("n", ascending=False)
    )


def _independence_statistic(table: np.ndarray, stat: str) -> float:
    """LR (G) or Pearson statistic of independence for one 2x2 table."""
    n = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    if stat == "pearson":
        return float(((table - expected) ** 2 / expected).sum())
    if stat == "lr":
        observed = table > 0  # 0 * log(0) = 0
        return float(2 * (table[observed] * np.log(table[observed] / expected[observed])).sum())
    raise ValueError(f"stat must be 'lr' or 'pearson', not {stat!r}")


def fairness_test(
    y_pred: pd.Series, protected: pd.Series, strata: pd.Series | None = None, stat: str = "lr"
) -> dict[str, float]:
    """The paper's fairness test of H0: y_pred ⊥ protected | strata.

    ``protected`` is a boolean Series (True = protected group) over the rows being compared.
    Without strata this is statistical parity (a 2x2 test of independence, 1 degree of
    freedom); with strata it is conditional statistical parity (q = number of strata).
    A stratum whose table has an empty row or column carries no information about the null
    -- e.g. everyone in it gets the same prediction -- and is dropped from the sum and from q.
    """
    frame = pd.DataFrame({"y_pred": y_pred, "d": protected.astype(bool)})
    frame["stratum"] = "all" if strata is None else strata

    statistic, used = 0.0, 0
    for _, part in frame.groupby("stratum", observed=True):
        table = pd.crosstab(part["y_pred"], part["d"]).reindex(
            index=[0, 1], columns=[False, True], fill_value=0
        )
        table = table.to_numpy()
        if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
            continue
        statistic += _independence_statistic(table, stat)
        used += 1

    df = max(used, 1)
    return {
        "statistic": statistic,
        "df": df,
        "p_value": float(chi2.sf(statistic, df)) if used else 1.0,
        "strata_used": used,
    }


def _comparison_rows(data: Dataset, comparison: str) -> tuple[pd.Index, pd.Series]:
    """Rows in either group of a comparison, and whether each is in the protected group."""
    attribute, protected, reference = COMPARISONS[comparison]
    group = data.groups[attribute]
    rows = group.index[group.isin([protected, reference])]
    return rows, (group.loc[rows] == protected)


def parity_table(
    y_pred: pd.Series, data: Dataset, strata: pd.Series, stat: str = "lr"
) -> pd.DataFrame:
    """Statistical parity and conditional statistical parity for every comparison."""
    rows = {}
    for comparison in COMPARISONS:
        index, is_protected = _comparison_rows(data, comparison)
        pred = y_pred.loc[index]
        sp = fairness_test(pred, is_protected, stat=stat)
        csp = fairness_test(pred, is_protected, strata.loc[index], stat=stat)
        rate_protected = pred[is_protected].mean()
        rate_reference = pred[~is_protected].mean()
        rows[comparison] = {
            "n_protected": int(is_protected.sum()),
            "n_reference": int((~is_protected).sum()),
            "flag_rate_protected": rate_protected,
            "flag_rate_reference": rate_reference,
            "sp_difference": rate_protected - rate_reference,
            "sp_ratio": rate_protected / rate_reference,
            "sp_statistic": sp["statistic"],
            "sp_p_value": sp["p_value"],
            "csp_statistic": csp["statistic"],
            "csp_df": csp["df"],
            "csp_p_value": csp["p_value"],
        }
    return pd.DataFrame(rows).T


def conditional_parity_detail(
    y_pred: pd.Series, data: Dataset, strata: pd.Series, comparison: str, stat: str = "lr"
) -> pd.DataFrame:
    """Per-stratum breakdown behind the conditional statistical parity test."""
    index, is_protected = _comparison_rows(data, comparison)
    rows = {}
    for stratum in sorted(strata.loc[index].unique()):
        in_stratum = strata.loc[index] == stratum
        pred, d = y_pred.loc[index][in_stratum], is_protected[in_stratum]
        test = fairness_test(pred, d, stat=stat)
        rows[stratum] = {
            "n_protected": int(d.sum()),
            "n_reference": int((~d).sum()),
            "flag_rate_protected": pred[d].mean(),
            "flag_rate_reference": pred[~d].mean(),
            "statistic": test["statistic"],
            "p_value": test["p_value"] if test["strata_used"] else np.nan,
        }
    return pd.DataFrame(rows).T


def fpdp(
    model,
    data: Dataset,
    feature: str,
    comparison: str,
    grid=None,
    strata: pd.Series | None = None,
    threshold: float = 0.5,
    stat: str = "lr",
) -> pd.DataFrame:
    """Fairness Partial Dependence Plot data for one feature (paper's Definitions 5-6).

    For each value in ``grid`` (default: every value observed in the data), the feature is set to that
    value for every defendant, the model re-predicts, and the fairness test is re-run. Pass
    ``strata`` for conditional statistical parity, leave it out for statistical parity.
    """
    index, is_protected = _comparison_rows(data, comparison)
    X = data.X.loc[index]
    # Values observed anywhere in the data, not just in the two compared groups: otherwise
    # e.g. the Hispanic dummy is always 0 among African-American and Caucasian defendants.
    grid = np.unique(data.X[feature]) if grid is None else grid
    rows = []
    for value in grid:
        X_fixed = X.copy()
        X_fixed[feature] = value
        pred = pd.Series(
            (model.predict_proba(X_fixed)[:, 1] >= threshold).astype(int), index=index
        )
        test = fairness_test(pred, is_protected, None if strata is None else strata.loc[index], stat)
        rows.append(
            {
                "value": value,
                "statistic": test["statistic"],
                "p_value": test["p_value"],
                "flag_rate_protected": pred[is_protected].mean(),
                "flag_rate_reference": pred[~is_protected].mean(),
                # Everyone (in every stratum) gets the same prediction: "fair" only trivially.
                "degenerate": test["strata_used"] == 0,
            }
        )
    return pd.DataFrame(rows)


def fpdp_all(
    model, data: Dataset, comparison: str, strata: pd.Series | None = None, **kwargs
) -> dict[str, pd.DataFrame]:
    """FPDP for every feature the model uses."""
    return {
        feature: fpdp(model, data, feature, comparison, strata=strata, **kwargs)
        for feature in data.X.columns
    }


def candidate_variables(
    curves: dict[str, pd.DataFrame], alpha: float = 0.05, include_degenerate: bool = False
) -> pd.DataFrame:
    """Features for which some fixed value stops the test from rejecting fairness.

    Only meaningful when the test rejects fairness on the actual data (Definition 6). By
    default, values at which the model gives everyone the same prediction are ignored: they
    pass the test trivially (p = 1) without saying anything about the feature's role.
    """
    rows = {}
    for feature, curve in curves.items():
        usable = curve if include_degenerate else curve[~curve["degenerate"]]
        if usable.empty:
            rows[feature] = {"max_p_value": np.nan, "at_value": np.nan, "candidate": False}
            continue
        best = usable.loc[usable["p_value"].idxmax()]
        rows[feature] = {
            "max_p_value": best["p_value"],
            "at_value": best["value"],
            "candidate": bool(best["p_value"] > alpha),
        }
    return pd.DataFrame(rows).T.sort_values("max_p_value", ascending=False)


def main() -> None:
    for feature_set in ("race_aware", "race_blind"):
        _, test = load_data(feature_set)
        model = load_model(feature_set)
        y_pred = predict_labels(model, test)
        strata = make_strata(test)

        print(f"\n=== {feature_set} ===")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(parity_table(y_pred, test, strata).round(4))
        for comparison in ("African-American vs Caucasian", "Female vs Male"):
            curves = fpdp_all(model, test, comparison)
            print(f"\nFPDP candidate variables (statistical parity), {comparison}:")
            print(candidate_variables(curves))


if __name__ == "__main__":
    main()
