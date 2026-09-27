"""T08. Stability of performance: delta of every performance.csv metric between paired runs.

Design 1, X1 -> X3 against X2 -> X3: same test defendants, different training sample.
Design 2, temporal_1 against temporal_2: the model trained on more, later data and tested on
newer defendants. There the test sets differ (and so does the base rate), so a delta mixes
the model with the population; runs.csv's base rates are reported next to it.

Delta = b - a, for TabPFN and for the COMPAS tool on the same rows (the tool never changes,
so its delta in design 2 measures the population drift alone). No refit.
Output: pairs_performance.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd
from pfn_stability import metric_deltas

METRICS = [
    "auc", "brier", "ece", "accuracy", "precision", "recall", "f1", "specificity",
    "balanced_accuracy", "type_i_error", "type_ii_error", "selection_rate",
    "cost_per_defendant", "base_rate",
]  # fmt: skip


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__) / "pairs_performance.csv"
    if c.cached(out, args.force):
        return

    perf = pd.read_csv(c.ART / "performance.csv", dtype={"operating_point": str})
    key = ["run", "feature_set", "model", "operating_point"]
    perf = perf.set_index(key)

    frames = []
    for pair, (run_a, run_b) in c.PAIRS.items():
        for feature_set, model, point in (
            perf.loc[run_a].index.unique()  # (feature_set, model, operating_point)
        ):
            a = perf.loc[(run_a, feature_set, model, point), METRICS]
            b = perf.loc[(run_b, feature_set, model, point), METRICS]
            deltas = metric_deltas(a, b)
            deltas.insert(0, "operating_point", point)
            deltas.insert(0, "model", model)
            deltas.insert(0, "feature_set", feature_set)
            deltas.insert(0, "pair", pair)
            frames.append(deltas)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(out, index=False)

    shown = table[
        (table["feature_set"] == "race_aware")
        & (table["operating_point"] == "break_even")
        & table["metric"].isin(["auc", "type_i_error", "type_ii_error", "cost_per_defendant",
                                "base_rate"])
    ]  # fmt: skip
    print("race_aware, break-even threshold (0.252), delta = b - a:")
    print(c.fmt(shown[["pair", "model", "metric", "a", "b", "delta"]], 4))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
