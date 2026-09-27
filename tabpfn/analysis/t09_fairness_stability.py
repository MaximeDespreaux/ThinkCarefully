"""T09. Stability of fairness: do the gaps move between paired runs?

For each pair (X1 vs X2, temporal_1 vs temporal_2), feature set and operating point: the
delta of each fairness gap (T03) and of the tightest certifiable tolerance (T04), b - a.
For the time-ordered pair it adds the population stability index (PSI) of TabPFN's scores,
since those test sets differ: PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 shifted. It is
only meaningful when the scores take many values (not for the 2-feature sets).

No refit. Output: pairs_fairness.csv, psi.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd
from pfn_stability import population_stability_index

GAPS = c.ANALYSIS / "t03_equalized_odds" / "gaps.csv"
EQUIVALENCE = c.ANALYSIS / "t04_fairness_equivalence" / "equivalence.csv"
GAP_COLUMNS = [
    "fpr_difference", "fnr_difference", "equal_opportunity_difference",
    "equalized_odds_difference", "demographic_parity_difference", "ppv_difference",
]  # fmt: skip


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__)
    if c.cached(out / "pairs_fairness.csv", args.force):
        return

    key = ["feature_set", "model", "operating_point", "attribute"]
    gaps = pd.read_csv(GAPS)
    long = gaps.melt(id_vars=["run", *key], value_vars=GAP_COLUMNS, var_name="metric")
    equivalence = pd.read_csv(EQUIVALENCE)
    min_delta = equivalence.assign(metric=equivalence["metric"] + "_minimum_delta").rename(
        columns={"minimum_delta": "value"}
    )[["run", *key, "metric", "value"]]
    long = (
        pd.concat([long, min_delta], ignore_index=True)
        .set_index(["run", *key, "metric"])
        .sort_index()
    )

    frames = []
    for pair, (run_a, run_b) in c.PAIRS.items():
        a = long.loc[run_a, "value"].rename("a")
        b = long.loc[run_b, "value"].rename("b")
        joined = pd.concat([a, b], axis=1).reset_index()
        joined.insert(0, "pair", pair)
        frames.append(joined)
    table = pd.concat(frames, ignore_index=True)
    table["delta"] = table["b"] - table["a"]
    table.to_csv(out / "pairs_fairness.csv", index=False)

    psi = []
    run_a, run_b = c.PAIRS["temporal_1_vs_2"]
    for feature_set in sorted(gaps["feature_set"].unique()):
        a = c.read_predictions(run_a, feature_set)["tabpfn"]
        b = c.read_predictions(run_b, feature_set)["tabpfn"]
        psi.append(
            {
                "pair": "temporal_1_vs_2",
                "feature_set": feature_set,
                "psi_tabpfn_scores": population_stability_index(a, b),
                # With few features the scores take a handful of values, the decile bins
                # collapse and the PSI is not interpretable.
                "n_distinct_scores_a": int(a.round(6).nunique()),
            }
        )
    psi = pd.DataFrame(psi)
    psi.to_csv(out / "psi.csv", index=False)

    shown = table[
        (table["feature_set"] == "race_aware")
        & (table["model"] == "TabPFN")
        & (table["operating_point"] == "break_even")
        & table["metric"].isin(["fpr_difference", "fpr_difference_minimum_delta"])
    ]
    print("race_aware, TabPFN, break-even threshold (0.252):")
    print(c.fmt(shown[["pair", "attribute", "metric", "a", "b", "delta"]]))
    print("\n" + c.fmt(psi))
    print(f"\n-> {out / 'pairs_fairness.csv'}\n-> {out / 'psi.csv'}")


if __name__ == "__main__":
    main()
