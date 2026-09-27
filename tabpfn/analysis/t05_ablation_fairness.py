"""T05. Does removing a protected attribute reduce disparity? The ablation, on fairness.

For each design and operating point, one row per feature set: what it costs in performance
(AUC, cost per defendant, from performance.csv) against what it does to the race and sex gaps
(T03) and to the tightest tolerance at which they could be certified fair (T04).

"Fairness through unawareness" predicts race_blind closes the race gap. The race proxies
(priors above all) predict it does not. This table settles it on TabPFN. No refit.
Output: ablation.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd
from pfn_fairness import is_degenerate_rate

GAPS = c.ANALYSIS / "t03_equalized_odds" / "gaps.csv"
EQUIVALENCE = c.ANALYSIS / "t04_fairness_equivalence" / "equivalence.csv"
PERFORMANCE = c.ART / "performance.csv"

FEATURE_SET_ORDER = [
    "race_aware", "race_blind", "sex_blind", "age_blind",
    "protected_blind", "race_priors_blind", "race_proxy_blind",
]  # fmt: skip


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__) / "ablation.csv"
    if c.cached(out, args.force):
        return

    key = ["run", "feature_set", "operating_point"]
    gaps = pd.read_csv(GAPS)
    gaps = gaps[gaps["model"] == "TabPFN"]
    wide = gaps.pivot_table(
        index=key,
        columns="attribute",
        values=["fpr_difference", "demographic_parity_difference", "equalized_odds_difference"],
    )
    wide.columns = [f"{attribute}_{metric}" for metric, attribute in wide.columns]
    # A set that flags (or releases) nearly everyone has zero gaps by construction, not merit.
    wide["degenerate_operating_point"] = gaps[gaps["attribute"] == "race"].set_index(key)[
        "degenerate_operating_point"
    ]

    equivalence = pd.read_csv(EQUIVALENCE)
    equivalence = equivalence[
        (equivalence["model"] == "TabPFN") & (equivalence["metric"] == "fpr_difference")
    ]
    min_delta = equivalence.pivot_table(index=key, columns="attribute", values="minimum_delta")
    min_delta.columns = [f"{attribute}_fpr_minimum_delta" for attribute in min_delta.columns]

    perf = pd.read_csv(PERFORMANCE, dtype={"operating_point": str})
    perf = perf[perf["model"] == "TabPFN"].set_index(key)[
        ["auc", "cost_per_defendant", "selection_rate"]
    ]

    table = perf.join(wide).join(min_delta).reset_index()
    # The stricter bar the earlier study used for a claimed fix: a gap closed by flagging over
    # 95% (or under 5%) of defendants was bought by not distinguishing anyone.
    table["gap_closed_by_flagging_nearly_all"] = table["selection_rate"].map(is_degenerate_rate)
    table["feature_set"] = pd.Categorical(table["feature_set"], FEATURE_SET_ORDER, ordered=True)
    table = table.sort_values(["run", "operating_point", "feature_set"])

    # Change against the race-aware model of the same run and operating point.
    reference = table[table["feature_set"] == "race_aware"].set_index(["run", "operating_point"])
    for column in ["auc", "race_fpr_difference", "sex_fpr_difference"]:
        base = table.set_index(["run", "operating_point"]).index.map(reference[column])
        table[f"{column}_vs_race_aware"] = table[column].to_numpy() - base.to_numpy()
    table.to_csv(out, index=False)

    shown = table[(table["run"] == "holdout") & (table["operating_point"] == "break_even")]
    columns = [
        "feature_set", "auc", "cost_per_defendant", "selection_rate",
        "gap_closed_by_flagging_nearly_all", "race_fpr_difference",
        "race_fpr_minimum_delta", "sex_fpr_difference", "race_fpr_difference_vs_race_aware",
    ]  # fmt: skip
    print("holdout, break-even threshold (0.252):")
    print(c.fmt(shown[columns]))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
