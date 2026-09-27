"""T18. Add the fairness and stability dimensions to the radar charts.

Writes tabpfn/artifacts/radar_extra.csv: one row per feature set (and one for the COMPAS
tool), one column per new spoke, each scaled to 0-1 with higher = better. plot_tabpfn.py
joins it onto the performance spokes as it is; `make tabpfn-plots` redraws the radars.

All read at the break-even threshold (0.252), on holdout for fairness and on X1 vs X2 for
stability, whichever panel of the radar they are drawn on:
    Race FPR parity        1 - |FPR gap|, African-American vs Caucasian           (T03)
    Sex FPR parity         1 - |FPR gap|, Female vs Male                          (T03)
    Certifiable fairness   1 - tightest delta at which the race FPR gap is TOST-certified (T04)
    Decision stability     1 - share of X3 decisions that flip between X1 and X2 fits (T07)

A feature set that flags nearly everyone scores well on the fairness spokes by construction;
its collapse shows on the specificity spoke (see T05, gap_closed_by_flagging_nearly_all).
No refit.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd

GAPS = c.ANALYSIS / "t03_equalized_odds" / "gaps.csv"
EQUIVALENCE = c.ANALYSIS / "t04_fairness_equivalence" / "equivalence.csv"
FLIPS = c.ANALYSIS / "t07_prediction_stability" / "pairs_predictions.csv"
OUT = c.ART / "radar_extra.csv"
TOOL = "COMPAS tool"


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    c.output_dir(__file__)  # keeps the one-folder-per-test layout, even if unused
    if c.cached(OUT, args.force):
        return

    gaps = pd.read_csv(GAPS)
    gaps = gaps[(gaps["run"] == "holdout") & gaps["operating_point"].isin(["break_even", "tool"])]
    equivalence = pd.read_csv(EQUIVALENCE)
    equivalence = equivalence[
        (equivalence["run"] == "holdout")
        & equivalence["operating_point"].isin(["break_even", "tool"])
        & (equivalence["attribute"] == "race")
        & (equivalence["metric"] == "fpr_difference")
    ]
    flips = pd.read_csv(FLIPS)
    flips = flips[flips["operating_point"] == "break_even"]

    def row_label(frame: pd.DataFrame) -> pd.Series:
        return frame["feature_set"].where(frame["model"] == "TabPFN", TOOL)

    spokes = {}
    for attribute, label in (("race", "Race FPR\nparity"), ("sex", "Sex FPR\nparity")):
        block = gaps[gaps["attribute"] == attribute]
        spokes[label] = pd.Series(
            1 - block["fpr_difference"].abs().to_numpy(), index=row_label(block)
        )
    spokes["Certifiable\nfairness"] = pd.Series(
        1 - equivalence["minimum_delta"].to_numpy(), index=row_label(equivalence)
    )
    spokes["Decision stability\n(X1 vs X2)"] = pd.Series(
        1 - flips["flip_rate"].to_numpy(), index=row_label(flips)
    )
    # The tool's rows repeat once per feature set file; they are identical, keep one.
    table = pd.DataFrame({k: v[~v.index.duplicated()] for k, v in spokes.items()})
    table = table.clip(0, 1).rename_axis("feature_set")
    table.to_csv(OUT)

    print(c.fmt(table.reset_index()))
    print(f"\n-> {OUT}\n   redraw with `make tabpfn-plots`")


if __name__ == "__main__":
    main()
