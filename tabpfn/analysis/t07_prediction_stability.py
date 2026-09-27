"""T07. Stability design 1: does the training sample change TabPFN's predictions?

TabPFN has no estimated parameters to compare (its weights are frozen), so the distance
between the X1 fit and the X2 fit is measured on their outputs, defendant by defendant, on the
shared test set X3: L2 norm and mean |delta| between the two score vectors, and the share of
defendants whose decision flips at each threshold.

The COMPAS tool line is a sanity check: it is the same score in both files, so 0.
No refit. Output: pairs_predictions.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd
from pfn_stability import decision_flips, score_distance


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__) / "pairs_predictions.csv"
    if c.cached(out, args.force):
        return

    run_a, run_b = c.PAIRS["X1_vs_X2"]
    rows = []
    for feature_set in sorted({fs for _, fs, _ in c.predictions()}):
        a = c.read_predictions(run_a, feature_set)
        b = c.read_predictions(run_b, feature_set).loc[a.index]  # align defendant by defendant
        if not (a["y"] == b["y"]).all():
            raise ValueError(f"{run_a} and {run_b} do not share their test rows")
        for model in ("tabpfn", "compas_tool"):
            distance = score_distance(a[model], b[model])
            for point, threshold in c.operating_points().items():
                rows.append(
                    {
                        "pair": "X1_vs_X2",
                        "run_a": run_a,
                        "run_b": run_b,
                        "feature_set": feature_set,
                        "model": "TabPFN" if model == "tabpfn" else "COMPAS tool",
                        "operating_point": point,
                        **distance,
                        **decision_flips(a[model], b[model], threshold),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)

    shown = table[table["model"] == "TabPFN"]
    columns = ["feature_set", "operating_point", "l2", "mean_abs_delta", "max_abs_delta",
               "correlation", "flip_rate", "n_flips"]  # fmt: skip
    print(f"X1 -> X3 against X2 -> X3, n = {int(shown['n'].iloc[0])} defendants in X3:")
    print(c.fmt(shown[columns], 4))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
