"""Evaluate and interpret the logistic regression.
You can run this file as follows:
    uv run python -m logreg.analysis                      # every stage
    uv run python -m logreg.analysis --stage whitebox     # one stage
Note: the analysis can be found in artifacts/logreg/
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
from compas_scoring import interpret, stability
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
from compas_scoring.models import load

warnings.filterwarnings("ignore", message="Unknown solver options: iprint")
warnings.filterwarnings("ignore", category=FutureWarning)

NAME = "logistic"
PRIMARY = "race_aware"
FEATURE_SETS = tuple(CONFIG.feature_sets)
ART = CONFIG.path("artifacts", "logreg")


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
    """Do the coefficients keep their ranking when the training sample changes?

    A scorecard is only trustworthy if refitting it on a slightly different sample keeps the
    same drivers in the same order. Train split only.
    """
    banner("Stability: coefficient ranking across bootstrap refits")
    train, _ = train_test(PRIMARY)
    pipeline = load(NAME, PRIMARY).estimator

    def importance_fn(fitted):
        return np.abs(fitted.named_steps["clf"].coef_[0])

    result = stability.importance_stability(
        pipeline, train.X, train.y, importance_fn, n_refits=CONFIG.iterations.bootstrap_refits
    )
    write(result, "coefficient_stability.json")


STAGES = {
    "performance": stage_performance,
    "interpretability": stage_interpretability,
    "whitebox": stage_whitebox,
    "stability": stage_stability,
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
