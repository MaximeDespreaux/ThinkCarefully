"""Evaluate and interpret the logistic regression.
You can run this file as follows:
    uv run python -m logreg.analysis                      # every stage
    uv run python -m logreg.analysis --stage whitebox     # one stage
Note: the analysis can be found in artifacts/logreg/, and `python -m logreg.plots` turns it
into figures under reports/figures/logreg/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

import compas_scoring  # noqa: F401  (imported for its thread-pool pinning)
from compas_scoring import fairness as shared_fairness
from compas_scoring import interpret, stability, xper
from compas_scoring.config import CONFIG
from compas_scoring.data import train_test
from compas_scoring.evaluate import (
    CostModel,
    baseline_costs,
    bootstrap_auc_ci,
    cost_curve,
    cost_sensitivity,
    delong_test,
    statistical_metrics,
)
from logreg.model import load
from logreg import fairness

warnings.filterwarnings("ignore", message="Unknown solver options: iprint")
warnings.filterwarnings("ignore", category=FutureWarning)

NAME = "logistic"
PRIMARY = "race_aware"
FEATURE_SETS = tuple(CONFIG.feature_sets)
ART = CONFIG.path("artifacts", "logreg")

# The tolerance a disparity must sit inside for TOST to certify it fair. One value for every
# metric, and the same one the shared analysis uses, so the two can never disagree.
TOST_DELTA = 0.05

# Every resampling count in one place. The defaults match the shared analysis; the tests
# shrink them so the whole pipeline can be exercised in seconds.
BUDGET = {
    "cv_repeats": CONFIG.iterations.cv_repeats,
    "bootstrap_refits": CONFIG.iterations.bootstrap_refits,
    "fairness_bootstrap": 1000,
    "distance_pairs": 25,
    "distance_rows": 300,
    "permutation_repeats": 10,
    "n_seeds": CONFIG.iterations.n_seeds,
    "perturbation_trials": 20,
    "learning_curve_repeats": 5,
    "ice_curves": 100,
    "lime_rows": 30,
    "lime_samples": 5000,
    "shap_background": CONFIG.iterations.shap_background,
    "shap_explain": CONFIG.iterations.shap_explain,
    "xper_explain": xper.N_EXPLAIN,
    "xper_background": xper.N_BACKGROUND,
}


def write(obj, name: str) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(obj, indent=2, default=float))
    print(f"    -> logreg/{name}")


def banner(title: str) -> None:
    print(f"\n{title}")


def scored(feature_set: str):
    """Train split, test split, fitted model, holdout scores and the cost-optimal threshold.

    Every decision-level result is taken at the model's own cost-optimal threshold -- the
    operating point it would actually be deployed at.
    """
    train, test = train_test(feature_set)
    model = load(NAME, feature_set)
    score = model.predict_proba(test.X)
    threshold = cost_curve(test.y, score, CostModel()).optimal_threshold
    return train, test, model, score, threshold


class FrameInput:
    """Give the pipeline the DataFrame its `engineer` step needs, whatever the caller passes.

    KernelSHAP evaluates the model on bare numpy arrays; `engineered()` selects columns by
    name, so without this wrapper it fails on the first call.
    """

    def __init__(self, estimator, columns):
        self.estimator = estimator
        self.columns = list(columns)

    def predict_proba(self, X):
        return self.estimator.predict_proba(pd.DataFrame(np.asarray(X), columns=self.columns))


# performance


def stage_performance() -> None:
    """Holdout performance on every feature set, against the incumbent and the trivial policies."""
    banner("Predictive performance")
    costs = CostModel()
    print(
        f"    cost model: FN ${costs.c_fn:,.0f} / FP ${costs.c_fp:,.0f} "
        f"(ratio {costs.ratio:.1f}, break-even p={costs.break_even_threshold:.3f})"
    )

    rows, curves, sensitivity, scores = [], [], [], {}
    test = None
    for feature_set in FEATURE_SETS:
        _, test = train_test(feature_set)
        score = load(NAME, feature_set).predict_proba(test.X)
        scores[feature_set] = score

        curve = cost_curve(test.y, score, costs)
        auc, lo, hi = bootstrap_auc_ci(test.y, score)
        rows.append(
            {
                "model": NAME,
                "feature_set": feature_set,
                "auc_lower": lo,
                "auc_upper": hi,
                "cost_optimal_threshold": curve.optimal_threshold,
                "cost_per_capita": curve.optimal_cost,
                **statistical_metrics(test.y, score, curve.optimal_threshold),
            }
        )
        curves.append(
            pd.DataFrame(
                {
                    "feature_set": feature_set,
                    "threshold": curve.thresholds,
                    "cost_per_capita": curve.cost_per_capita,
                }
            )
        )
        sensitivity.append(cost_sensitivity(test.y, score).assign(feature_set=feature_set))
        print(f"    {feature_set:11s} AUC {auc:.4f} [{lo:.4f}, {hi:.4f}]")

    # Every feature set shares one split, so the last `test` carries the same rows and
    # the same incumbent decisions as all the others.
    incumbent = test.incumbent.to_numpy()
    rows.append(
        {
            "model": "compas",
            "feature_set": "-",
            "auc_lower": np.nan,
            "auc_upper": np.nan,
            "cost_optimal_threshold": 0.5,
            "cost_per_capita": costs.per_capita(test.y, incumbent),
            **statistical_metrics(test.y, incumbent.astype(float), 0.5),
        }
    )

    table = pd.DataFrame(rows)
    floors = baseline_costs(test.y, costs)
    table["saving_vs_best_trivial"] = floors["best_trivial"] - table["cost_per_capita"]
    table["saving_per_1000"] = table["saving_vs_best_trivial"] * 1000
    write(table, "performance.csv")
    write(pd.concat(curves), "cost_curves.csv")
    write(pd.concat(sensitivity), "cost_sensitivity.csv")
    write({**floors, "n_test": int(len(test))}, "cost_baselines.json")

    # Does removing race (or keeping only the selected features) cost any ranking power?
    pairs = [
        {"model_a": a, "model_b": b, **delong_test(test.y, scores[a], scores[b])}
        for a, b in combinations(FEATURE_SETS, 2)
    ]
    pairs += [
        {
            "model_a": fs,
            "model_b": "compas",
            **delong_test(test.y, scores[fs], incumbent.astype(float)),
        }
        for fs in FEATURE_SETS
    ]
    write(pd.DataFrame(pairs), "delong_tests.csv")


# Interpretability


def stage_interpretability() -> None:
    """Odds ratios and the points scorecard, read off the fitted pipeline."""
    banner("Interpretability")
    train, _ = train_test(PRIMARY)
    pipeline = load(NAME, PRIMARY).estimator

    # The pipeline engineers features internally, so the odds ratios must be fitted on
    # that same engineered design matrix.
    design = pipeline.named_steps["engineer"].transform(train.X)
    write(interpret.odds_ratios(design, train.y).reset_index(names="feature"), "odds_ratios.csv")

    clf = pipeline.named_steps["clf"]
    coefficients = pd.Series(clf.coef_[0], index=design.columns)
    write(
        interpret.points_scorecard(coefficients, float(clf.intercept_[0]), len(coefficients)),
        "scorecard.csv",
    )


# Explanations


def local_contributions(pipeline, rows: pd.DataFrame) -> pd.DataFrame:
    """Exact per-term log-odds contributions: coefficient times standardised value.

    This is what makes a logistic regression explainable without any approximation: the
    score of each defendant is the intercept plus these terms, exactly. Contributions are
    relative to the average defendant, because the design is standardised.
    """
    design = pipeline.named_steps["engineer"].transform(rows)
    scaled = pipeline.named_steps["scale"].transform(design)
    clf = pipeline.named_steps["clf"]
    frames = []
    for position, index in enumerate(rows.index):
        frames.append(
            pd.DataFrame(
                {
                    "defendant": index,
                    "term": design.columns,
                    "value": design.iloc[position].to_numpy(dtype=float),
                    "contribution": clf.coef_[0] * scaled[position],
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    table["intercept"] = float(clf.intercept_[0])
    return table


def stage_explanations() -> None:
    """Global and local explanations: permutation, PDP/ICE, KernelSHAP, LIME, archetypes."""
    banner("Explanations: permutation, PDP/ICE, KernelSHAP, LIME, archetypes")
    train, test, model, _, _ = scored(PRIMARY)
    pipeline = model.estimator
    columns = list(test.X.columns)

    permutation = interpret.permutation_ranking(
        pipeline, test.X, test.y, n_repeats=BUDGET["permutation_repeats"]
    )
    write(
        permutation.rename("importance").rename_axis("feature").reset_index(),
        "permutation_importance.csv",
    )

    feature = "Number_of_Priors"
    write(interpret.partial_dependence_curve(pipeline, test.X, feature), f"pdp_{feature}.csv")
    curves = interpret.ice_curves(pipeline, test.X, feature, n_curves=BUDGET["ice_curves"])
    ice = []
    for kind, frame in (("raw", curves), ("centered", interpret.center_ice(curves))):
        melted = frame.reset_index(names="row").melt(
            id_vars="row", var_name=feature, value_name="predicted_risk"
        )
        ice.append(melted.assign(kind=kind))
    write(pd.concat(ice, ignore_index=True), "ice_curves.csv")
    write(interpret.ice_heterogeneity(curves), "ice_heterogeneity.json")

    values, base, seconds = interpret.kernel_shap(
        FrameInput(pipeline, columns),
        train.X,
        test.X,
        n_background=BUDGET["shap_background"],
        n_explain=BUDGET["shap_explain"],
    )
    shap_importance = interpret.global_importance(values, columns)
    write(
        shap_importance.rename("mean_abs_shap").rename_axis("feature").reset_index(),
        "shap_importance.csv",
    )
    explained = test.X.iloc[: len(values)]
    write(
        {
            "kernelshap": {
                **interpret.shap_efficiency_check(
                    values, base, pipeline.predict_proba(explained)[:, 1]
                ),
                "space": "probability",
            },
            "kernelshap_seconds": seconds,
            "kernelshap_rows": int(len(values)),
        },
        "shap_axioms.json",
    )

    rows = test.X.sample(BUDGET["lime_rows"], random_state=CONFIG.random_state)
    lime = pd.DataFrame(
        {
            index: interpret.lime_explanation(
                pipeline, train.X, rows.loc[index], num_samples=BUDGET["lime_samples"]
            )
            for index in rows.index
        }
    ).T
    write(lime.reset_index(names="row"), "lime_weights.csv")
    write(
        interpret.disagreement_panel(
            {
                "lime": lime.abs().mean(),
                "permutation": permutation,
                "kernelshap": shap_importance,
            }
        ),
        "disagreement_panel.csv",
    )

    archetypes = interpret.archetype_defendants(test.X, test.groups)
    people = test.X.loc[archetypes["index"]]
    write(
        archetypes.assign(predicted_risk=pipeline.predict_proba(people)[:, 1]).reset_index(),
        "archetypes.csv",
    )
    contributions = local_contributions(pipeline, people)
    contributions["archetype"] = contributions["defendant"].map(
        dict(zip(archetypes["index"], archetypes.index, strict=True))
    )
    write(contributions, "local_contributions.csv")

    paired, conflicts = [], []
    for label, index in zip(archetypes.index, archetypes["index"], strict=True):
        frame = interpret.paired_local_explanation(pipeline, train.X, test.X.loc[index])
        paired.append(frame.assign(archetype=label, defendant=index))
        conflicts.append({"archetype": label, **interpret.explanation_conflicts(frame)})
    write(pd.concat(paired, ignore_index=True), "local_explanations.csv")
    write(pd.DataFrame(conflicts), "explanation_conflicts.csv")

    top = permutation.idxmax()
    print(f"    top driver by permutation importance: {top}; KernelSHAP took {seconds:.1f}s")


# Whitebox


def stage_whitebox() -> None:
    """Marginal effects, the L1 path, and the auditable stepwise selection log."""
    banner("White-box depth: marginal effects, lasso path, stepwise log")
    train, _ = train_test(PRIMARY)
    pipeline = load(NAME, PRIMARY).estimator

    design = pipeline.named_steps["engineer"].transform(train.X)
    write(
        interpret.average_marginal_effects(design, train.y).reset_index(names="feature"),
        "marginal_effects.csv",
    )

    write(interpret.lasso_path(train.X, train.y), "lasso_path.csv")

    forward = interpret.stepwise_log(train.X, train.y, direction="forward")
    backward = interpret.stepwise_log(train.X, train.y, direction="backward")
    write(pd.concat([forward, backward], ignore_index=True), "experiment_log.csv")

    accepted = forward[forward["accepted"] & (forward["step"] > 0)]
    print(
        f"    forward selection accepted {len(accepted)} features "
        f"(final CV AUC {forward['cv_auc'].max():.4f})"
    )


# Stability


def stage_stability() -> None:
    """Does the model -- its scores, decisions and coefficients -- hold still under resampling?

    A scorecard is only trustworthy if refitting it on a slightly different sample keeps the
    same drivers in the same order and gives the same people the same decisions.
    """
    banner("Stability: resampling, refits, perturbation, drift")
    train, test, model, score, threshold = scored(PRIMARY)
    pipeline = model.estimator

    def importance_fn(fitted):
        return np.abs(fitted.named_steps["clf"].coef_[0])

    result = stability.importance_stability(
        pipeline, train.X, train.y, importance_fn, n_refits=BUDGET["bootstrap_refits"]
    )
    write(result, "coefficient_stability.json")

    cv = stability.repeated_cv(pipeline, train.X, train.y, n_repeats=BUDGET["cv_repeats"])
    write(cv, "stability_cv.csv")
    matrix = stability.bootstrap_predictions(
        pipeline, train.X, train.y, test.X, n_refits=BUDGET["bootstrap_refits"]
    )
    summary = {
        "cv_auc_mean": float(cv["auc"].mean()),
        "cv_auc_sd": float(cv["auc"].std(ddof=1)),
        "cv_folds": int(len(cv)),
        **stability.prediction_stability(matrix),
        **stability.decision_flip_rate(matrix, threshold),
        "psi_train_vs_test": stability.population_stability_index(
            pipeline.predict_proba(train.X)[:, 1], score
        ),
        "n_bootstrap_refits": int(matrix.shape[0]),
        "threshold": threshold,
        "selection_rate_at_threshold": float((score >= threshold).mean()),
    }
    write(summary, "stability_summary.json")

    write(
        stability.learning_curve(
            pipeline, train.X, train.y, test.X, test.y, n_repeats=BUDGET["learning_curve_repeats"]
        ),
        "learning_curves.csv",
    )
    write(
        stability.perturbation_sensitivity(
            pipeline,
            test.X,
            columns=["Number_of_Priors", "Misdemeanor"],
            n_trials=BUDGET["perturbation_trials"],
        ),
        "perturbation.csv",
    )
    write(
        stability.seed_sensitivity(
            pipeline, train.X, train.y, test.X, test.y, n_seeds=BUDGET["n_seeds"]
        ),
        "seed_sensitivity.csv",
    )
    write(stability.subgroup_performance(test.y, score, test.groups), "subgroup_performance.csv")

    # Two samples from one population should induce approximately the same model: the
    # distance between coefficient vectors (and between permutation importances) fitted on
    # independent halves of the training data.
    rows = test.X.sample(BUDGET["distance_rows"], random_state=CONFIG.random_state)
    rows_y = test.y.loc[rows.index]

    def permutation_fn(fitted):
        return interpret.permutation_ranking(fitted, rows, rows_y, n_repeats=2).to_numpy()

    pairs = BUDGET["distance_pairs"]
    write(
        {
            **stability.importance_distance(
                pipeline, train.X, train.y, permutation_fn, n_pairs=pairs
            ),
            **stability.coefficient_distance(pipeline, train.X, train.y, n_pairs=pairs),
        },
        "model_distance.json",
    )

    half = len(train.X) // 2
    frontier = stability.stability_constrained_fit(
        train.X.iloc[:half],
        train.y.iloc[:half],
        train.X.iloc[half:],
        train.y.iloc[half:],
        test.X,
        test.y,
    )
    write(frontier, "stability_frontier.csv")
    print(
        f"    CV AUC {summary['cv_auc_mean']:.4f} +- {summary['cv_auc_sd']:.4f} | "
        f"score sd {summary['mean_sd']:.4f} | unstable decisions {summary['unstable_share']:.1%}"
    )


# Fairness


def fairness_strata(test) -> dict[str, pd.Series]:
    """The legitimate risk factors that conditional statistical parity is checked against.

    Each is tested on its own. Stratifying on all of them jointly is deliberately left out:
    the race-blind model's decision is a function of exactly these inputs, so inside a joint
    cell almost everyone gets the same decision and the test has nothing left to measure.
    """
    return {
        "priors_band": fairness.priors_band(test.X["Number_of_Priors"]).astype(str),
        "age_band": test.groups["age_band"].reset_index(drop=True).astype(str),
        "sex": test.groups["sex"].reset_index(drop=True).astype(str),
        "charge_degree": test.groups["charge_degree"].reset_index(drop=True).astype(str),
    }


def stage_fairness() -> None:
    """Statistical parity, conditional statistical parity and equalized odds, with tests.

    Each feature set's model is judged at its own cost-optimal threshold -- the operating
    point it would actually be deployed at -- and against both reference groups.
    """
    banner("Fairness: z-tests, Hurlin summed LR, CMH and Mantel-Haenszel odds ratio")
    costs = CostModel()
    parity, conditional, per_stratum = [], [], []

    for feature_set in FEATURE_SETS:
        _, test = train_test(feature_set)
        score = load(NAME, feature_set).predict_proba(test.X)
        threshold = cost_curve(test.y, score, costs).optimal_threshold
        y_pred = (score >= threshold).astype(int)
        race = test.groups["race"].to_numpy()
        strata = fairness_strata(test)
        context = {
            "feature_set": feature_set,
            "threshold": threshold,
            "selection_rate": float(y_pred.mean()),
        }

        for comparison, reference in fairness.COMPARISONS.items():
            labels = {**context, "comparison": comparison, "reference_group": reference}
            parity.append({**labels, **fairness.statistical_parity_ztest(y_pred, race, comparison)})
            parity += [
                {**labels, **row}
                for row in fairness.equalized_odds_ztests(test.y, y_pred, race, comparison)
            ]

            for stratifier, values in strata.items():
                conditional.append(
                    {
                        **labels,
                        "stratifier": stratifier,
                        # Strata where everyone gets the same decision drop out of the
                        # tests; n_levels against n_strata shows how many were left.
                        "n_levels": int(values.nunique()),
                        **fairness.summed_lr_test(y_pred, race, values, comparison),
                        **fairness.mantel_haenszel(y_pred, race, values, comparison),
                    }
                )
                per_stratum.append(
                    fairness.stratum_odds_ratios(y_pred, race, values, comparison).assign(
                        **labels, stratifier=stratifier
                    )
                )

    parity, conditional = pd.DataFrame(parity), pd.DataFrame(conditional)
    write(parity, "fairness_parity_ztests.csv")
    write(conditional, "fairness_conditional_parity.csv")
    write(pd.concat(per_stratum, ignore_index=True), "fairness_stratum_odds_ratios.csv")

    headline = conditional.query(
        "comparison == 'aa_vs_others' and stratifier == 'priors_band'"
    ).set_index("feature_set")
    for feature_set, row in headline.iterrows():
        print(
            f"    {feature_set:11s} at equal priors: MH odds ratio {row['mh_odds_ratio']:.2f} "
            f"[{row['mh_or_lower']:.2f}, {row['mh_or_upper']:.2f}]  "
            f"summed LR p={row['lr_p_value']:.2e}  CMH p={row['cmh_p_value']:.2e}"
        )


def stage_fairness_groups() -> None:
    """Per-group error rates, calibration, the impossibility result, and what mitigation costs."""
    banner("Fairness by group: rates, calibration, impossibility, frontier, mitigation")
    costs = CostModel()
    rates, disparities, calibration, impossibility, frontier, mitigation = [], [], [], [], [], []
    thresholds = {}

    for feature_set in FEATURE_SETS:
        _, test, _, score, threshold = scored(feature_set)
        y_pred = (score >= threshold).astype(int)
        # "operating_threshold", not "threshold": the impossibility and frontier tables sweep
        # their own threshold column, which a plain "threshold" key would overwrite.
        context = {"feature_set": feature_set, "operating_threshold": threshold}

        for attribute in ("race", "sex", "age_band"):
            by_group = shared_fairness.group_metrics(test.y, y_pred, test.groups[attribute])
            rates.append(by_group.reset_index(names="group").assign(**context, attribute=attribute))
        calibration.append(
            shared_fairness.calibration_by_group(test.y, score, test.groups["race"]).assign(
                **context
            )
        )

        for comparison in fairness.COMPARISONS:
            labels, groups = fairness.comparison_labels(test.groups["race"], comparison)
            tag = {**context, "comparison": comparison}
            disparities.append(
                {**tag, **shared_fairness.disparity_summary(test.y, y_pred, labels, groups)}
            )
            impossibility.append(
                shared_fairness.impossibility_evidence(test.y, score, labels, groups).assign(**tag)
            )
            frontier.append(
                shared_fairness.fairness_performance_frontier(
                    test.y, score, labels, costs, groups
                ).assign(**tag)
            )

            # Per-group thresholds equalising the FPR: the price of error-rate parity. It
            # conditions the decision on race, so it is evidence, not a deployable remedy.
            cutoffs = shared_fairness.group_thresholds(test.y, score, labels, "fpr")
            thresholds[f"{feature_set}__{comparison}"] = cutoffs
            adjusted = shared_fairness.apply_group_thresholds(score, labels, cutoffs)
            for strategy, decision in (
                ("uniform threshold", y_pred),
                ("per-group thresholds", adjusted),
            ):
                summary = shared_fairness.disparity_summary(test.y, decision, labels, groups)
                mitigation.append(
                    {
                        **tag,
                        "strategy": strategy,
                        "accuracy": float((test.y.to_numpy() == decision).mean()),
                        "cost_per_capita": costs.per_capita(test.y, decision),
                        "fpr_gap": summary["fpr_difference"],
                        "equalized_odds_difference": summary["equalized_odds_difference"],
                        "disparate_impact_ratio": summary["disparate_impact_ratio"],
                    }
                )

    write(pd.concat(rates, ignore_index=True), "fairness_by_group.csv")
    write(pd.DataFrame(disparities), "fairness_disparities.csv")
    write(pd.concat(calibration, ignore_index=True), "calibration_by_group.csv")
    write(pd.concat(impossibility, ignore_index=True), "impossibility.csv")
    write(pd.concat(frontier, ignore_index=True), "fairness_frontier.csv")
    write(pd.DataFrame(mitigation), "mitigation.csv")
    write(thresholds, "group_thresholds.json")

    for row in disparities:
        if row["comparison"] == "aa_vs_others":
            print(
                f"    {row['feature_set']:11s} detains {row['selection_rate']:.0%} | FPR gap "
                f"{row['fpr_difference']:+.3f} | 4/5 rule "
                f"{'pass' if row['passes_four_fifths_rule'] else 'FAIL'}"
            )


def stage_fairness_inference() -> None:
    """Bootstrap intervals for every disparity, and TOST equivalence at one shared delta."""
    banner(f"Fairness inference: bootstrap CIs and TOST at delta={TOST_DELTA}")
    inference, equivalence = [], []

    for feature_set in FEATURE_SETS:
        _, test, _, score, threshold = scored(feature_set)
        y_pred = (score >= threshold).astype(int)
        for comparison in fairness.COMPARISONS:
            labels, groups = fairness.comparison_labels(test.groups["race"], comparison)
            tag = {"feature_set": feature_set, "comparison": comparison, "threshold": threshold}
            draws = shared_fairness.bootstrap_disparities(
                test.y, score, labels, threshold, n_boot=BUDGET["fairness_bootstrap"], groups=groups
            )
            point = shared_fairness.disparity_summary(test.y, y_pred, labels, groups)
            inference.append(shared_fairness.disparity_inference(point, draws).assign(**tag))
            for metric in (
                "demographic_parity_difference",
                "fpr_difference",
                "fnr_difference",
                "equalized_odds_difference",
            ):
                equivalence.append(
                    {
                        **tag,
                        "metric": metric,
                        **shared_fairness.equivalence_test(draws[metric], delta=TOST_DELTA),
                    }
                )

    write(pd.concat(inference, ignore_index=True), "fairness_inference.csv")
    equivalence = pd.DataFrame(equivalence)
    write(equivalence, "fairness_equivalence.csv")
    print(
        f"    TOST at delta={TOST_DELTA}: {int(equivalence['certified_fair'].sum())} of "
        f"{len(equivalence)} feature-set/comparison/metric cells certified fair"
    )


def stage_fpdp() -> None:
    """Fairness PDP: which variables generate the disparity, and what neutralising them costs."""
    banner("FPDP: candidate variables and mitigation")
    costs = CostModel()
    curves, candidates, mitigation = [], [], []

    for feature_set in FEATURE_SETS:
        _, test, model, score, threshold = scored(feature_set)
        for comparison in fairness.COMPARISONS:
            labels, groups = fairness.comparison_labels(test.groups["race"], comparison)
            tag = {"feature_set": feature_set, "comparison": comparison, "threshold": threshold}
            found_curves = pd.concat(
                [
                    shared_fairness.fairness_pdp(
                        model.predict_proba, test.X, test.y, labels, column, threshold, groups
                    )
                    for column in test.X.columns
                ],
                ignore_index=True,
            )
            curves.append(found_curves.assign(**tag))
            found = shared_fairness.candidate_variables(found_curves)
            candidates.append(found.assign(**tag))

            def outcome(scores, strategy, variable, tag=tag, labels=labels, groups=groups):
                decision = (scores >= threshold).astype(int)
                summary = shared_fairness.disparity_summary(test.y, decision, labels, groups)
                return {
                    **tag,
                    "strategy": strategy,
                    "variable": variable,
                    "accuracy": float((test.y.to_numpy() == decision).mean()),
                    "auc": statistical_metrics(test.y, scores, threshold)["auc"],
                    "cost_per_capita": costs.per_capita(test.y, decision),
                    "sp_p_value": shared_fairness.chi2_statistical_parity(decision, labels, groups)[
                        "p_value"
                    ],
                    "fpr_difference": summary["fpr_difference"],
                    "disparate_impact_ratio": summary["disparate_impact_ratio"],
                }

            mitigation.append(outcome(score, "uniform threshold", ""))
            # Mitigation without re-estimation: fix the candidate variable, keep the model.
            for _, row in found[found["is_candidate"]].iterrows():
                neutralised = test.X.copy()
                neutralised[row["feature"]] = row["best_level"]
                mitigation.append(
                    outcome(
                        model.predict_proba(neutralised),
                        "neutralise candidate variable",
                        f"{row['feature']} = {row['best_level']:g}",
                    )
                )

    write(pd.concat(curves, ignore_index=True), "fpdp_curves.csv")
    candidates = pd.concat(candidates, ignore_index=True)
    write(candidates, "fpdp_candidates.csv")
    write(pd.DataFrame(mitigation), "fpdp_mitigation.csv")
    found = candidates[candidates["is_candidate"]]
    print(f"    candidate variables: {sorted(set(found['feature'])) or 'none'}")


def stage_xper() -> None:
    """XPER: which features earn the AUC and the dollars, and which ones overfit."""
    banner("XPER: decomposing AUC, cost and the train-test gap")
    costs = CostModel()
    auc_parts, cost_parts, overfitting = {}, {}, []

    for feature_set in FEATURE_SETS:
        train, test, model, _, threshold = scored(feature_set)
        estimator = model.estimator

        def predict(frame, estimator=estimator):
            return estimator.predict_proba(frame)[:, 1]

        X_explain, y_explain = xper.sample(test.X, test.y, BUDGET["xper_explain"])
        X_background, _ = xper.sample(train.X, train.y, BUDGET["xper_background"])
        X_train, y_train = xper.sample(train.X, train.y, BUDGET["xper_explain"], seed=7)

        auc = xper.decompose(predict, X_explain, y_explain, X_background)
        xper.check_efficiency(auc)
        cost = xper.decompose(
            predict, X_explain, y_explain, X_background, metric=xper.cost_metric(threshold, costs)
        )
        xper.check_efficiency(cost, tolerance=1e-6)
        on_train = xper.decompose(predict, X_train, y_train, X_background)

        auc_parts[feature_set], cost_parts[feature_set] = auc, cost
        overfitting.append(
            xper.overfitting_decomposition(on_train, auc).assign(feature_set=feature_set)
        )
        print(
            f"    {feature_set:11s} AUC={auc.attrs['metric_value']:.4f} "
            f"benchmark={auc['benchmark']:.3f} priors={auc['Number_of_Priors']:+.4f}"
        )

    write(xper.as_table(auc_parts).rename(columns={"model": "feature_set"}), "xper_auc.csv")
    write(xper.as_table(cost_parts).rename(columns={"model": "feature_set"}), "xper_cost.csv")
    write(pd.concat(overfitting, ignore_index=True), "xper_overfitting.csv")


STAGES = {
    "performance": stage_performance,
    "interpretability": stage_interpretability,
    "explanations": stage_explanations,
    "whitebox": stage_whitebox,
    "stability": stage_stability,
    "fairness": stage_fairness,
    "fairness_groups": stage_fairness_groups,
    "fairness_inference": stage_fairness_inference,
    "fpdp": stage_fpdp,
    "xper": stage_xper,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    args = parser.parse_args()

    started = time.perf_counter()
    for name, fn in STAGES.items():
        if args.stage in (name, "all"):
            fn()
    print(f"\nDone in {time.perf_counter() - started:.0f}s. Artifacts in {ART}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
