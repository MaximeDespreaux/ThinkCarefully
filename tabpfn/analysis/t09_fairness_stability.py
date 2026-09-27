"""T09. Stability of fairness: do the gaps move between paired runs?

For each pair (X1 vs X2, temporal_1 vs temporal_2), feature set and operating point: the
delta, b - a, of each gap of the shared protocol (flag rate, FPR, FNR), of the tightest
certifiable tolerance (TOST) and of the Holm p-value of every test, read from T01's
protocol.csv.
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

PROTOCOL = c.ANALYSIS / "t01_statistical_parity" / "protocol.csv"
VALUES = ["gap", "minimum_delta", "p_holm"]  # the columns compared between runs


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__)
    if c.cached(out / "pairs_fairness.csv", args.force):
        return

    key = ["feature_set", "model", "threshold_name", "comparison", "metric"]
    protocol = pd.read_csv(PROTOCOL)
    long = (
        protocol.melt(id_vars=["run", *key], value_vars=VALUES, var_name="value_of")
        .dropna(subset="value")
        .set_index(["run", *key, "value_of"])
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
    for feature_set in sorted(protocol["feature_set"].unique()):
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
        & (table["threshold_name"] == "break_even")
        & (table["metric"] == "fpr")
        & table["comparison"].isin(["African-American vs Caucasian", "Female vs Male"])
    ]
    print("race_aware, TabPFN, FPR at the break-even threshold (0.252):")
    print(c.fmt(shown[["pair", "comparison", "value_of", "a", "b", "delta"]]))
    print("\n" + c.fmt(psi))
    print(f"\n-> {out / 'pairs_fairness.csv'}\n-> {out / 'psi.csv'}")


if __name__ == "__main__":
    main()
