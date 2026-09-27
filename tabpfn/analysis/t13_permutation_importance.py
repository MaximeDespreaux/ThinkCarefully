"""T13. Permutation importance, on AUC and on cost, with intervals.

Shuffle one feature, and measure how much performance drops: that is the feature's
importance. Run against two metrics, AUC (ranking) and cost per defendant at the 0.252
break-even threshold (the decisions), from the same shuffles, 10 repeats, so each importance
has a 95% interval and a feature whose interval spans zero is not shown to matter.

Refits TabPFN once per run in _common.INTERPRET_RUNS, on each run's full test set; 11
predict_proba calls per run (pfn_importance.permutation_multi). Output: permutation.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd
from pfn_importance import negative_cost, permutation_multi
from pfn_metrics import break_even_threshold
from sklearn.metrics import roc_auc_score

N_REPEATS = 10


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__) / "permutation.csv"
    if c.cached(out, args.force):
        return

    metrics = {"auc": roc_auc_score, "cost": negative_cost(break_even_threshold())}
    frames = []
    for run, feature_set in c.INTERPRET_RUNS:
        model, spec = c.fit(run, feature_set)
        result = permutation_multi(model, spec.test.X, spec.test.y, metrics, N_REPEATS)
        for metric, frame in result.items():
            frames.append(
                frame.assign(
                    run=run,
                    feature_set=feature_set,
                    metric=metric,
                    unit="AUC points" if metric == "auc" else "$ per defendant",
                    seconds=frame.attrs["seconds"],
                )
            )
        print(f"  {run} x {feature_set}: {result['auc'].attrs['seconds']:.0f}s", flush=True)

    table = pd.concat(frames, ignore_index=True)
    table["rank"] = table.groupby(["run", "feature_set", "metric"])["importance"].rank(
        ascending=False
    )
    first = ["run", "feature_set", "metric", "unit", "feature"]
    table = table[first + [col for col in table.columns if col not in first]]
    table.to_csv(out, index=False)

    shown = table[(table["run"] == "holdout") & (table["feature_set"] == "race_aware")]
    columns = ["metric", "feature", "importance", "ci_lower", "ci_upper", "share",
               "interval_excludes_zero"]  # fmt: skip
    print(c.fmt(shown.sort_values(["metric", "rank"])[columns], 4))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
