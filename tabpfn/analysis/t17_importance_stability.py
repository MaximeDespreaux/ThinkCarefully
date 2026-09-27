"""T17. Stability of the explanations: does TabPFN's story change with the training sample?

For each pair of fits (X1 vs X2 on the same X3; temporal_1 vs temporal_2), the distance
between their importance vectors, each normalised to sum to 1: the L2 norm of the difference
(0 = the same shares, sqrt(2) = disjoint), the Spearman rank correlation, the overlap of the
top 3, and whether the top feature is the same. For three importance measures: permutation
importance on AUC and on cost (T13) and mean |SHAP| (T14), plus the marginal effects (T10).

No refit. Output: importance_distance.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd
from pfn_stability import importance_distance

T10 = c.ANALYSIS / "t10_marginal_effects" / "marginal_effects.csv"
T13 = c.ANALYSIS / "t13_permutation_importance" / "permutation.csv"
T14 = c.ANALYSIS / "t14_kernel_shap" / "shap_importance.csv"


def vectors() -> dict[str, pd.DataFrame]:
    """Every importance measure as a long frame: run, feature, value."""
    perm = pd.read_csv(T13)
    measures = {
        f"permutation_{metric}": block.rename(columns={"importance": "value"})
        for metric, block in perm.groupby("metric")
    }
    measures["mean_abs_shap"] = pd.read_csv(T14).rename(columns={"mean_abs_shap": "value"})
    measures["marginal_effect"] = pd.read_csv(T10).rename(columns={"marginal_effect": "value"})
    return {
        name: frame[frame["feature_set"] == "race_aware"][["run", "feature", "value"]]
        for name, frame in measures.items()
    }


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__) / "importance_distance.csv"
    if c.cached(out, args.force):
        return

    rows = []
    for measure, frame in vectors().items():
        by_run = {run: block.set_index("feature")["value"] for run, block in frame.groupby("run")}
        for pair, (run_a, run_b) in c.PAIRS.items():
            rows.append(
                {
                    "pair": pair,
                    "measure": measure,
                    "run_a": run_a,
                    "run_b": run_b,
                    **importance_distance(by_run[run_a], by_run[run_b]),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)

    columns = ["pair", "measure", "l2", "spearman", "top3_overlap", "top_feature_a",
               "top_feature_b"]  # fmt: skip
    print(c.fmt(table[columns]))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
