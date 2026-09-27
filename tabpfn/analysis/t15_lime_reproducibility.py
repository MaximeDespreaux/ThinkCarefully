"""T15. LIME: does it give the same explanation twice?

LIME explains one prediction by fitting a local linear model to the black box's answers on
random perturbations of the defendant. The earlier study found it was not reproducible on
TabPFN: 15 runs on one defendant named 4 different top features. This reruns the same
protocol on this repo's model: one borderline defendant (2-4 priors, aged 25-45, the first
such in the holdout test set), 3 sample sizes x 5 seeds.

Refits TabPFN once (holdout x race_aware). Outputs: lime.csv (one row per run), lime_summary.json.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import json

import pandas as pd
from pfn_interpret import lime_explanation

RUN, FEATURE_SET = "holdout", "race_aware"
SAMPLE_SIZES = (1000, 2000, 5000)
SEEDS = range(5)


def borderline(X: pd.DataFrame) -> int:
    mask = (
        X["Number_of_Priors"].between(2, 4)
        & (X["Age_Below_TwentyFive"] == 0)
        & (X["Age_Above_FourtyFive"] == 0)
    )
    return X.index[mask][0]


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__)
    if c.cached(out / "lime_summary.json", args.force):
        return

    model, spec = c.fit(RUN, FEATURE_SET)
    X = spec.test.X
    row_id = borderline(X)
    row = X.loc[row_id]

    rows = []
    for n in SAMPLE_SIZES:
        for seed in SEEDS:
            weights = lime_explanation(model, spec.train.X, row, num_samples=n, seed=seed)
            top = weights.abs().idxmax()
            rows.append(
                {
                    "num_samples": n,
                    "seed": seed,
                    "top_feature": top,
                    "top_weight": float(weights[top]),
                    # A feature the defendant does not have (value 0 on a dummy) named as the
                    # main reason for their score.
                    "top_feature_absent": bool(row[top] == 0 and top != "Number_of_Priors"),
                    **{f"w_{k}": float(v) for k, v in weights.items()},
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "lime.csv", index=False)

    counts = table["top_feature"].value_counts()
    summary = {
        "run": RUN,
        "feature_set": FEATURE_SET,
        "defendant_row": int(row_id),
        "defendant": {k: float(v) for k, v in row.items()},
        "prediction": float(model.predict_proba(row.to_frame().T)[0, 1]),
        "n_runs": int(len(table)),
        "n_distinct_top_features": int(counts.size),
        "top_features_named": {k: int(v) for k, v in counts.items()},
        "distinct_by_sample_size": {
            str(n): int(g["top_feature"].nunique()) for n, g in table.groupby("num_samples")
        },
        "runs_naming_an_absent_feature": int(table["top_feature_absent"].sum()),
        "earlier_study": {"n_distinct_top_features": 4, "runs_naming_an_absent_feature": 10},
    }
    (out / "lime_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
