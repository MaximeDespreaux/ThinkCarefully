"""T05. Does removing a protected attribute reduce disparity? The ablation, on fairness.

For each design and threshold, one row per feature set: what it costs in performance (AUC,
cost per defendant, from performance.csv) against what it does to the primary race gap and
the sex gap: FPR gap, tightest certifiable delta (TOST), and whether statistical parity and
conditional parity (proxies held fixed) are rejected. All from the shared protocol (T01).

"Fairness through unawareness" predicts race_blind closes the race gap; the race proxies
(priors above all) predict it does not. A set that flags nearly everyone (>= 95%) has small
gaps only because it distinguishes nobody, and is flagged as such. No refit.
Output: ablation.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd

PROTOCOL = c.ANALYSIS / "t01_statistical_parity" / "protocol.csv"
PERFORMANCE = c.ART / "performance.csv"
COMPARISONS = {"race": "African-American vs Caucasian", "sex": "Female vs Male"}
NEARLY_ALL = 0.95

FEATURE_SET_ORDER = [
    "race_aware", "race_blind", "sex_blind", "age_blind",
    "protected_blind", "race_priors_blind", "race_proxy_blind",
]  # fmt: skip


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__) / "ablation.csv"
    if c.cached(out, args.force):
        return

    key = ["run", "feature_set", "threshold_name"]
    protocol = pd.read_csv(PROTOCOL)
    protocol = protocol[protocol["model"] == "TabPFN"]

    columns = {}
    for label, comparison in COMPARISONS.items():
        block = protocol[protocol["comparison"] == comparison].set_index(key)
        by_metric = {m: block[block["metric"] == m] for m in block["metric"].unique()}
        columns[f"{label}_fpr_gap"] = by_metric["fpr"]["gap"]
        columns[f"{label}_fpr_minimum_delta"] = by_metric["fpr"]["minimum_delta"]
        columns[f"{label}_parity_rejected"] = by_metric["statistical_parity"]["reject_fairness"]
        columns[f"{label}_conditional_parity_rejected"] = by_metric[
            "conditional_statistical_parity"
        ]["reject_fairness"]
    fairness = pd.DataFrame(columns)

    perf = pd.read_csv(PERFORMANCE, dtype={"operating_point": str})
    perf = perf[perf["model"] == "TabPFN"].rename(columns={"operating_point": "threshold_name"})
    perf = perf.set_index(key)[["auc", "cost_per_defendant", "selection_rate", "specificity"]]

    table = perf.join(fairness).reset_index()
    table["gap_closed_by_flagging_nearly_all"] = (table["selection_rate"] >= NEARLY_ALL) | (
        table["selection_rate"] <= 1 - NEARLY_ALL
    )
    table["feature_set"] = pd.Categorical(table["feature_set"], FEATURE_SET_ORDER, ordered=True)
    table = table.sort_values(["run", "threshold_name", "feature_set"])

    # Change against the race-aware model of the same run and threshold.
    reference = table[table["feature_set"] == "race_aware"].set_index(["run", "threshold_name"])
    position = table.set_index(["run", "threshold_name"]).index
    for column in ["auc", "race_fpr_gap", "sex_fpr_gap"]:
        table[f"{column}_vs_race_aware"] = (
            table[column].to_numpy() - position.map(reference[column]).to_numpy()
        )
    table.to_csv(out, index=False)

    for threshold in ("0.5", "break_even"):
        shown = table[(table["run"] == "holdout") & (table["threshold_name"] == threshold)]
        print(f"\nholdout, threshold {threshold}:")
        print(c.fmt(shown[["feature_set", "auc", "cost_per_defendant", "selection_rate",
                           "gap_closed_by_flagging_nearly_all", "race_fpr_gap",
                           "race_fpr_minimum_delta", "race_conditional_parity_rejected",
                           "sex_fpr_gap"]]))  # fmt: skip
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
