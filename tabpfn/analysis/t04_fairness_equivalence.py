"""T04. Fairness equivalence (TOST, Schuirmann 1987): can the model be *certified* fair?

A plain test has H0 = "fair", so not rejecting it certifies nothing. TOST reverses the burden:

    H0: |theta| >= delta  (unfair)      H1: -delta < theta < delta  (fair)

with theta the gap in flag rate (statistical parity), FPR or FNR, protected minus reference,
and delta = 0.05 fixed in [tool.compas_scoring.fairness] before any result was seen. Certified
when both one-sided tests reject (unpooled variance, as in the course). ``minimum_delta`` is
the tightest tolerance that could be certified, so no delta has to be picked after the fact.

From the shared protocol (T01's protocol.csv). No refit. Output: equivalence.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd

PROTOCOL = c.ANALYSIS / "t01_statistical_parity" / "protocol.csv"
COLUMNS = [
    "run", "feature_set", "model", "threshold_name", "threshold", "comparison", "attribute",
    "primary", "metric", "selection_rate", "degenerate", "gap", "ci_low", "ci_high", "tost_p",
    "certified_fair", "minimum_delta",
]  # fmt: skip


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__) / "equivalence.csv"
    if c.cached(out, args.force):
        return

    protocol = pd.read_csv(PROTOCOL)
    table = protocol[protocol["metric"].isin(["statistical_parity", "fpr", "fnr"])][COLUMNS]
    table.to_csv(out, index=False)

    shown = table[(table["run"] == "holdout") & (table["feature_set"] == "race_aware")]
    print(c.fmt(shown[["model", "threshold_name", "comparison", "metric", "gap",
                       "minimum_delta", "certified_fair"]]))  # fmt: skip
    certified = table[table["certified_fair"] & (table["model"] == "TabPFN")]
    print(
        f"\nTabPFN certified fair (delta = 0.05) in {len(certified)} cases, "
        f"of which degenerate (flags or releases >= 99%): {int(certified['degenerate'].sum())}"
        f"\n-> {out}"
    )


if __name__ == "__main__":
    main()
