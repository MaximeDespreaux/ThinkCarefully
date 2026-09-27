"""T06. Fairness partial dependence (FPDP): which variable could make the model fair?

The 3-step approach: test -> identify -> mitigate. T01 is the test. This is the identify step.
For each feature X_A and each value v, everybody's X_A is set to v, the model rescores them,
and the chi-squared parity test is recomputed. X_A is a *candidate variable* if some v brings
the p-value above 0.05 -- without flagging (or releasing) nearly everyone, which would make
the groups alike only because nobody is being distinguished.

Setting one race (or age) dummy to 1 also clears its siblings, so no counterfactual person
has two races. Refits TabPFN once on holdout x race_aware (~30 s), then one predict_proba call
per feature. Outputs: fpdp_curves.csv, fpdp_candidates.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import numpy as np
import pandas as pd
from pfn_fairness import candidate_variables, chi2_statistical_parity, disparity_summary
from pfn_interpret import AGE_DUMMIES, PRIORS

from compas_scoring.config import CONFIG

RUN, FEATURE_SET = "holdout", "race_aware"
PRIORS_GRID = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]


def counterfactuals(X: pd.DataFrame, feature: str) -> list[tuple[float, pd.DataFrame]]:
    siblings = {c: CONFIG.race_dummies for c in CONFIG.race_dummies}
    siblings.update({c: AGE_DUMMIES for c in AGE_DUMMIES})
    levels = PRIORS_GRID if feature == PRIORS else [0.0, 1.0]
    frames = []
    for level in levels:
        frame = X.copy()
        if level == 1.0 and feature in siblings:
            frame[[s for s in siblings[feature] if s in X.columns]] = 0.0
        frame[feature] = float(level)
        frames.append((float(level), frame))
    return frames


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__)
    if c.cached(out / "fpdp_candidates.csv", args.force):
        return

    model, spec = c.fit(RUN, FEATURE_SET)
    X, y = spec.test.X, spec.test.y.to_numpy()
    groups = spec.test.groups

    rows = []
    for feature in X.columns:
        cases = counterfactuals(X, feature)
        scores = model.predict_proba(pd.concat([f for _, f in cases], ignore_index=True))[:, 1]
        scores = scores.reshape(len(cases), len(X))
        for (level, _), score in zip(cases, scores):
            for point, threshold in c.operating_points().items():
                flagged = (score >= threshold).astype(int)
                for attribute, pair in c.ATTRIBUTES.items():
                    sensitive = groups[attribute].to_numpy()
                    test = chi2_statistical_parity(flagged, sensitive, pair)
                    summary = disparity_summary(y, flagged, sensitive, pair)
                    rows.append(
                        {
                            "operating_point": point,
                            "threshold": threshold,
                            "attribute": attribute,
                            "feature": feature,
                            "level": level,
                            "p_value": test["p_value"],
                            "statistic": test["statistic"],
                            "fpr_difference": summary["fpr_difference"],
                            "demographic_parity_difference": summary[
                                "demographic_parity_difference"
                            ],
                            "selection_rate": float(np.mean(flagged)),
                        }
                    )
        print(f"  {feature}: {len(cases)} levels", flush=True)

    curves = pd.DataFrame(rows)
    curves.to_csv(out / "fpdp_curves.csv", index=False)

    candidates = []
    for (point, attribute), block in curves.groupby(["operating_point", "attribute"]):
        found = candidate_variables(block)
        found.insert(0, "attribute", attribute)
        found.insert(0, "operating_point", point)
        candidates.append(found)
    candidates = pd.concat(candidates, ignore_index=True)
    candidates.to_csv(out / "fpdp_candidates.csv", index=False)

    columns = ["operating_point", "attribute", "feature", "best_level", "best_p_value",
               "selection_rate_at_best_level", "is_candidate"]  # fmt: skip
    print(c.fmt(candidates[columns], 4))
    print(f"\n-> {out / 'fpdp_curves.csv'}\n-> {out / 'fpdp_candidates.csv'}")


if __name__ == "__main__":
    main()
