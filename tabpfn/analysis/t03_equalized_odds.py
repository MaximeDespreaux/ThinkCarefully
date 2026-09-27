"""T03. Equalized odds: are the error rates the same across groups?

Equalized odds asks for equal error rates in every group: the FPR (flagged among those who did
not re-offend: ProPublica's finding) and the FNR (released among those who did). Each gap has a
z-test and a 95% interval; the Hurlin likelihood-ratio test of flagged independent of D given
the outcome Y tests both at once.

From the shared protocol (T01's protocol.csv). No refit. Output: equalized_odds.csv (long: one
row per metric fpr / fnr / equalized_odds) and error_rate_gaps.csv (wide: FPR and FNR side by
side, one row per case).
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd

PROTOCOL = c.ANALYSIS / "t01_statistical_parity" / "protocol.csv"
KEY = ["run", "feature_set", "model", "threshold_name", "comparison"]
COLUMNS = [
    *KEY, "metric", "threshold", "attribute", "primary", "selection_rate", "degenerate",
    "rate_protected", "rate_reference", "n_protected", "n_reference", "gap", "ci_low",
    "ci_high", "z", "p_value", "p_holm", "reject_fairness", "hurlin_statistic", "hurlin_p",
]  # fmt: skip


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__)
    if c.cached(out / "error_rate_gaps.csv", args.force):
        return

    protocol = pd.read_csv(PROTOCOL)
    long = protocol[protocol["metric"].isin(["fpr", "fnr", "equalized_odds"])][COLUMNS]
    long.to_csv(out / "equalized_odds.csv", index=False)

    rates = long[long["metric"].isin(["fpr", "fnr"])]
    wide = rates.pivot_table(
        index=[*KEY, "degenerate"],
        columns="metric",
        values=["rate_protected", "rate_reference", "gap", "p_holm"],
    )
    wide.columns = [f"{metric}_{value}" for value, metric in wide.columns]
    joint = long[long["metric"] == "equalized_odds"].set_index(KEY)["p_holm"]
    wide = wide.reset_index().join(joint.rename("equalized_odds_p_holm"), on=KEY)
    wide.to_csv(out / "error_rate_gaps.csv", index=False)

    shown = wide[(wide["run"] == "holdout") & (wide["feature_set"] == "race_aware")]
    columns = ["model", "threshold_name", "comparison", "fpr_rate_protected",
               "fpr_rate_reference", "fpr_gap", "fnr_gap", "equalized_odds_p_holm"]  # fmt: skip
    print(c.fmt(shown[columns]))
    print(f"\n-> {out / 'equalized_odds.csv'}\n-> {out / 'error_rate_gaps.csv'}")


if __name__ == "__main__":
    main()
