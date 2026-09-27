"""Fairness of the saved XGBoost models on the shared test set: African-American defendants
vs every other defendant (race treated as a single binary protected attribute). No other
protected attribute (sex, individual non-African-American race groups) is compared here.

Three fairness definitions, three dedicated tests, following Hurlin, Pérignon & Saurin, "The
Fairness of Credit Scoring Models" (arXiv 2205.10200) plus standard stratified-table methods:

- Statistical parity (SP): H0: Y_hat ⊥ D. Tested with the two-proportion z-test (Agresti-Caffo)
  on the flag-rate gap, with a 95% confidence interval.
- Conditional statistical parity (CSP): H0: Y_hat ⊥ D | C, C = priors band x charge degree.
  Reported three ways: the paper's summed likelihood-ratio statistic (chi2, df = strata used,
  no assumption on how the effect varies by stratum), the Cochran-Mantel-Haenszel test (chi2,
  1 df, assumes a common odds ratio across strata -- generally more powerful when that holds),
  and the Mantel-Haenszel pooled odds ratio with its 95% CI (the size of that common effect).
  The Breslow-Day test is reported alongside as a check on the common-odds-ratio assumption
  CMH/MH rely on.
- Equalized odds (EO): H0: Y_hat ⊥ D | Y. Tested as two z-tests (again Agresti-Caffo), one on
  the false-positive-rate gap and one on the false-negative-rate gap.

The Fairness Partial Dependence Plot (FPDP) sets one feature to the same value for every
defendant, re-predicts, and re-runs the *statistical parity* test only (by design choice here;
CSP/EO could use the same mechanism but are not swept). A feature is a *candidate variable* if
some value lifts the SP p-value above alpha (paper, Definitions 5-6).

Two ways to act on a candidate variable, both re-checked against all three test families:
- re-estimation: drop the feature and retrain from scratch (mitigate_by_reestimation)
- substitution: keep the original model, fix the feature at one value (mitigate_by_substitution)

Every function returns data; plotting is left to the notebook. Y_hat = 1 means the model flags
the defendant as likely to re-offend.

Run from the repository root:  ``python xgboost/fairness.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.proportion import confint_proportions_2indep, test_proportions_2indep
from xgb_model import load_data, load_model, tune_xgboost

from compas_scoring.data import Dataset

PROTECTED_GROUP = "African-American"
PRIORS_BANDS = {"0": (-np.inf, 0), "1-3": (0, 3), "4+": (3, np.inf)}


def predict_scores(model, data: Dataset) -> np.ndarray:
    """Predicted probability of two-year recidivism -- the model's actual task, reported
    alongside its fairness metrics so a mitigation's performance cost is never invisible."""
    return model.predict_proba(data.X)[:, 1]


def predict_labels(model, data: Dataset, threshold: float = 0.5) -> pd.Series:
    """Predicted class (1 = flagged as likely to re-offend), indexed like the data."""
    y_score = predict_scores(model, data)
    return pd.Series((y_score >= threshold).astype(int), index=data.X.index, name="y_pred")


def race_group(data: Dataset) -> pd.Series:
    """True for African-American defendants, False for every other race combined."""
    return (data.groups["race"] == PROTECTED_GROUP).rename("protected")


def make_strata(data: Dataset) -> pd.Series:
    """Conditioning classes for conditional statistical parity: priors band x charge degree."""
    edges = [low for low, _ in PRIORS_BANDS.values()] + [np.inf]
    band = pd.cut(data.X["Number_of_Priors"], bins=edges, labels=list(PRIORS_BANDS))
    strata = band.astype(str) + " priors | " + data.groups["charge_degree"]
    return strata.rename("stratum")


def group_rates(y_pred: pd.Series, data: Dataset) -> pd.DataFrame:
    """Size, flag rate and actual re-offence rate of African-American defendants vs the rest."""
    protected = race_group(data)
    frame = pd.DataFrame(
        {
            "group": protected.map({True: PROTECTED_GROUP, False: "Rest"}),
            "y_pred": y_pred,
            "y_true": data.y,
        }
    )
    return (
        frame.groupby("group")
        .agg(n=("y_pred", "size"), flag_rate=("y_pred", "mean"), base_rate=("y_true", "mean"))
        .reindex([PROTECTED_GROUP, "Rest"])
    )


def _two_proportion_test(count1: int, n1: int, count0: int, n0: int, alpha: float) -> dict:
    """z-test (Agresti-Caffo) and 95% CI for the difference of two independent proportions."""
    result = test_proportions_2indep(count1, n1, count0, n0, compare="diff")
    ci_low, ci_high = confint_proportions_2indep(
        count1, n1, count0, n0, compare="diff", alpha=alpha
    )
    return {
        "n_protected": n1,
        "n_rest": n0,
        "rate_protected": count1 / n1,
        "rate_rest": count0 / n0,
        "difference": float(result.diff),
        "z_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def statistical_parity_test(y_pred: pd.Series, protected: pd.Series, alpha: float = 0.05) -> dict:
    """Two-proportion z-test of H0: flag rate is equal for African-American defendants and the
    rest, with a (1 - alpha) confidence interval on the flag-rate difference."""
    n1, n0 = int(protected.sum()), int((~protected).sum())
    count1, count0 = int(y_pred[protected].sum()), int(y_pred[~protected].sum())
    return _two_proportion_test(count1, n1, count0, n0, alpha)


def equalized_odds_test(
    y_pred: pd.Series, y_true: pd.Series, protected: pd.Series, alpha: float = 0.05
) -> dict:
    """Two independent z-tests of H0: Y_hat ⊥ D | Y: one on the false-positive-rate gap (among
    defendants who did not re-offend), one on the false-negative-rate gap (among those who did).
    """
    negative, positive = y_true == 0, y_true == 1

    def _rate_test(mask: pd.Series, error_is_flagged: bool) -> dict:
        pred, grp = y_pred[mask], protected[mask]
        error = (pred == 1) if error_is_flagged else (pred == 0)
        n1, n0 = int(grp.sum()), int((~grp).sum())
        count1, count0 = int(error[grp].sum()), int(error[~grp].sum())
        return _two_proportion_test(count1, n1, count0, n0, alpha)

    return {
        "fpr": _rate_test(negative, error_is_flagged=True),
        "fnr": _rate_test(positive, error_is_flagged=False),
    }


def _stratum_tables(
    y_pred: pd.Series, protected: pd.Series, strata: pd.Series
) -> tuple[list, list]:
    """One 2x2 table per usable stratum: rows = (protected, rest), columns = (flagged, not).

    A stratum is dropped if the model gives everyone in it the same prediction, or if it has
    no members of one of the two groups -- it then carries no information about H0.
    """
    frame = pd.DataFrame({"y_pred": y_pred, "protected": protected, "stratum": strata})
    tables, used = [], []
    for name, part in frame.groupby("stratum", observed=True):
        n1, n0 = int(part["protected"].sum()), int((~part["protected"]).sum())
        if n1 == 0 or n0 == 0:
            continue
        c1, c0 = (
            int(part.loc[part["protected"], "y_pred"].sum()),
            int(part.loc[~part["protected"], "y_pred"].sum()),
        )
        table = np.array([[c1, n1 - c1], [c0, n0 - c0]], dtype=float)
        if (table.sum(axis=0) == 0).any():  # nobody flagged, or everybody flagged, in this stratum
            continue
        tables.append(table)
        used.append(name)
    return tables, used


def _hurlin_lr_statistic(tables: list) -> dict:
    """The paper's likelihood-ratio statistic, summed over strata: chi2(q), q = len(tables)."""
    statistic = 0.0
    for table in tables:
        n = table.sum()
        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
        observed = table > 0  # 0 * log(0) = 0
        statistic += 2 * (table[observed] * np.log(table[observed] / expected[observed])).sum()
    df = max(len(tables), 1)
    return {"statistic": statistic, "df": df, "p_value": float(chi2.sf(statistic, df))}


def conditional_statistical_parity(
    y_pred: pd.Series, protected: pd.Series, strata: pd.Series, alpha: float = 0.05
) -> dict:
    """H0: Y_hat ⊥ D | strata, reported three ways: Hurlin's summed LR test, the
    Cochran-Mantel-Haenszel test, and the Mantel-Haenszel pooled odds ratio (+ CI). The
    Breslow-Day statistic checks the common-odds-ratio assumption CMH/MH rely on."""
    tables, used = _stratum_tables(y_pred, protected, strata)
    hurlin = _hurlin_lr_statistic(tables)
    result = {
        "strata_used": len(tables),
        "strata": used,
        **{f"hurlin_{k}": v for k, v in hurlin.items()},
    }
    if not tables:
        result.update(
            cmh_statistic=np.nan,
            cmh_p_value=np.nan,
            mh_odds_ratio=np.nan,
            mh_ci_low=np.nan,
            mh_ci_high=np.nan,
            breslow_day_statistic=np.nan,
            breslow_day_p_value=np.nan,
        )
        return result

    # shift_zeros: add 0.5 to a table's cells if any is exactly 0, so Breslow-Day's per-cell
    # variance terms don't divide by zero; CMH/MH barely move for the affected strata.
    table = StratifiedTable(tables, shift_zeros=True)
    cmh = table.test_null_odds()
    mh_ci_low, mh_ci_high = table.oddsratio_pooled_confint(alpha=alpha)
    breslow_day = table.test_equal_odds()
    result.update(
        cmh_statistic=float(cmh.statistic),
        cmh_p_value=float(cmh.pvalue),
        mh_odds_ratio=float(table.oddsratio_pooled),
        mh_ci_low=float(mh_ci_low),
        mh_ci_high=float(mh_ci_high),
        breslow_day_statistic=float(breslow_day.statistic),
        breslow_day_p_value=float(breslow_day.pvalue),
    )
    return result


def evaluate_fairness(
    y_pred: pd.Series,
    data: Dataset,
    strata: pd.Series,
    alpha: float = 0.05,
    y_score: np.ndarray | None = None,
) -> dict:
    """SP, CSP and EO bundled into one result, for baseline/mitigation comparisons. Pass
    `y_score` (predicted probabilities) to also report the recidivism-prediction AUC alongside
    the fairness metrics -- a mitigation's performance cost should never be invisible."""
    protected = race_group(data)
    result = {
        "sp": statistical_parity_test(y_pred, protected, alpha),
        "csp": conditional_statistical_parity(y_pred, protected, strata, alpha),
        "eo": equalized_odds_test(y_pred, data.y, protected, alpha),
    }
    if y_score is not None:
        result["auc"] = float(roc_auc_score(data.y, y_score))
    return result


# --- FPDP (statistical parity only) and candidate variables ---------------------------------


def _fixed_value_scores(model, data: Dataset, feature: str, value) -> np.ndarray:
    X_fixed = data.X.copy()
    X_fixed[feature] = value
    return model.predict_proba(X_fixed)[:, 1]


def apply_fixed_value(
    model, data: Dataset, feature: str, value, threshold: float = 0.5
) -> pd.Series:
    """Predictions with `feature` set to `value` for every defendant (paper, Definitions 5-6)."""
    y_score = _fixed_value_scores(model, data, feature, value)
    return pd.Series((y_score >= threshold).astype(int), index=data.X.index, name="y_pred")


def fpdp(model, data: Dataset, feature: str, grid=None, threshold: float = 0.5) -> pd.DataFrame:
    """FPDP data for one feature, against the statistical parity null. For each value in
    ``grid`` (default: every value observed in the data), the feature is fixed for everyone,
    the model re-predicts, and the SP z-test is re-run. Values at which the model gives
    everyone the same prediction are flagged ``degenerate``: the test then passes trivially.
    """
    protected = race_group(data)
    grid = np.unique(data.X[feature]) if grid is None else grid
    rows = []
    for value in grid:
        y_pred = apply_fixed_value(model, data, feature, value, threshold)
        sp = statistical_parity_test(y_pred, protected)
        rows.append(
            {
                "value": value,
                "z_statistic": sp["z_statistic"],
                "p_value": sp["p_value"],
                "flag_rate_protected": sp["rate_protected"],
                "flag_rate_rest": sp["rate_rest"],
                "degenerate": y_pred.nunique() == 1,
            }
        )
    return pd.DataFrame(rows)


def fpdp_all(model, data: Dataset, **kwargs) -> dict[str, pd.DataFrame]:
    """FPDP for every feature the model uses."""
    return {feature: fpdp(model, data, feature, **kwargs) for feature in data.X.columns}


def candidate_variables(
    curves: dict[str, pd.DataFrame], alpha: float = 0.05, include_degenerate: bool = False
) -> pd.DataFrame:
    """Features for which some fixed value stops the SP test from rejecting fairness.

    Only meaningful when SP rejects fairness on the actual data (Definition 6). By default,
    values at which the model gives everyone the same prediction are ignored: they pass the
    test trivially (p = 1) without saying anything about the feature's role.
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


# --- Step 3: mitigation ----------------------------------------------------------------------


def refit_without_feature(train: Dataset, feature: str, n_iter: int = 40):
    """Re-estimation: drop `feature` and retrain from scratch, tuned the same way as the real
    models. Returns the fitted estimator."""
    search = tune_xgboost(train.X.drop(columns=[feature]), train.y, n_iter=n_iter)
    return search.best_estimator_


def mitigate_by_reestimation(
    train: Dataset,
    test: Dataset,
    feature: str,
    strata: pd.Series,
    threshold: float = 0.5,
    n_iter: int = 40,
) -> dict:
    """Retrain without `feature`, then re-run every fairness test on the held-out test set."""
    model = refit_without_feature(train, feature, n_iter=n_iter)
    y_score = model.predict_proba(test.X.drop(columns=[feature]))[:, 1]
    y_pred = pd.Series((y_score >= threshold).astype(int), index=test.X.index, name="y_pred")
    return {
        "method": "re-estimation",
        "feature": feature,
        **evaluate_fairness(y_pred, test, strata, y_score=y_score),
    }


def mitigate_by_substitution(
    model, data: Dataset, feature: str, value, strata: pd.Series, threshold: float = 0.5
) -> dict:
    """Keep the original model, fix `feature` at `value` for everyone, then re-run every test."""
    y_score = _fixed_value_scores(model, data, feature, value)
    y_pred = pd.Series((y_score >= threshold).astype(int), index=data.X.index, name="y_pred")
    return {
        "method": "substitution",
        "feature": feature,
        "value": value,
        **evaluate_fairness(y_pred, data, strata, y_score=y_score),
    }


def _print_fairness(label: str, result: dict) -> None:
    sp, csp, eo = result["sp"], result["csp"], result["eo"]
    print(f"\n{label}")
    if "auc" in result:
        print(f"  AUC (recidivism prediction): {result['auc']:.4f}")
    print(
        f"  SP:  flag rate {sp['rate_protected']:.3f} vs {sp['rate_rest']:.3f}"
        f" (diff {sp['difference']:+.3f}, 95% CI [{sp['ci_low']:.3f}, {sp['ci_high']:.3f}]),"
        f" z={sp['z_statistic']:.2f}, p={sp['p_value']:.4f}"
    )
    print(
        f"  CSP: Hurlin chi2({csp['hurlin_df']})={csp['hurlin_statistic']:.2f} "
        f"p={csp['hurlin_p_value']:.4f} | "
        f"CMH chi2(1)={csp['cmh_statistic']:.2f} p={csp['cmh_p_value']:.4f} | "
        f"MH odds ratio={csp['mh_odds_ratio']:.2f} "
        f"[{csp['mh_ci_low']:.2f}, {csp['mh_ci_high']:.2f}] | "
        f"Breslow-Day p={csp['breslow_day_p_value']:.4f}"
    )
    print(
        f"  EO:  FPR gap {eo['fpr']['difference']:+.3f} (p={eo['fpr']['p_value']:.4f}), "
        f"FNR gap {eo['fnr']['difference']:+.3f} (p={eo['fnr']['p_value']:.4f})"
    )


def main() -> None:
    # Step 1: is either model unfair to African-American defendants?
    loaded = {}
    for feature_set in ("race_aware", "race_blind"):
        train, test = load_data(feature_set)
        model = load_model(feature_set)
        y_score = predict_scores(model, test)
        y_pred = predict_labels(model, test)
        strata = make_strata(test)
        loaded[feature_set] = {
            "train": train,
            "test": test,
            "model": model,
            "y_score": y_score,
            "y_pred": y_pred,
            "strata": strata,
        }
        _print_fairness(
            f"=== {feature_set} ===", evaluate_fairness(y_pred, test, strata, y_score=y_score)
        )

    # Step 2: if so, which features are driving it? (race_aware only, per FPDP scope choice)
    primary = loaded["race_aware"]
    curves = fpdp_all(primary["model"], primary["test"])
    candidates = candidate_variables(curves)
    print("\nFPDP candidate variables (statistical parity), race_aware:")
    print(candidates)

    # Step 3: for each candidate, try both mitigations and re-check every test. If no feature
    # alone crosses alpha, demonstrate the mechanism on the closest one instead, clearly
    # labelled: it is not expected to remove the rejection of fairness, only shrink it.
    to_mitigate = candidates[candidates["candidate"]]
    if to_mitigate.empty:
        closest = candidates.index[0]  # candidates is sorted by max_p_value, descending
        print(
            f"\nNo feature alone lifts the SP p-value above 0.05; demonstrating mitigation on "
            f"the closest, {closest} (max p = {candidates.loc[closest, 'max_p_value']:.4f})."
        )
        to_mitigate = candidates.loc[[closest]]

    for feature, row in to_mitigate.iterrows():
        print(f"\n--- mitigating {feature} ---")
        re_est = mitigate_by_reestimation(
            primary["train"], primary["test"], feature, primary["strata"]
        )
        _print_fairness(f"re-estimation (drop {feature})", re_est)
        sub = mitigate_by_substitution(
            primary["model"], primary["test"], feature, row["at_value"], primary["strata"]
        )
        _print_fairness(f"substitution ({feature} = {row['at_value']})", sub)


if __name__ == "__main__":
    main()
