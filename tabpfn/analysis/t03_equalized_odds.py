"""T03. Equalized odds: are the error rates the same across groups?

Equalized odds asks for equal TPR and equal FPR across groups. The FPR gap is ProPublica's
finding: defendants who did not re-offend, flagged more often when Black. PPV (Northpointe's
calibration argument) is reported next to it, because with unequal base rates the two cannot
both be equal.

No refit. Outputs:
    by_group.csv   per group: count, base rate, selection rate, TPR, FPR, FNR, PPV
    gaps.csv       per pair: FPR, FNR, TPR, PPV, demographic parity and equalized-odds gaps
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd
from pfn_fairness import disparity_summary, group_metrics


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__)
    if c.cached(out / "gaps.csv", args.force):
        return

    by_group, gaps = [], []
    for run, feature_set, frame in c.predictions():
        y = frame["y"].to_numpy()
        for model, point, threshold, flagged in c.decision_cases(frame):
            meta = {
                "run": run,
                "feature_set": feature_set,
                "model": model,
                "operating_point": point,
                "threshold": threshold,
            }
            for attribute, groups in c.ATTRIBUTES.items():
                sensitive = frame[attribute].rename(attribute)
                table = group_metrics(y, flagged, sensitive).reset_index()
                table = table.rename(columns={attribute: "group"})
                by_group += [
                    {**meta, "attribute": attribute, **r} for r in table.to_dict("records")
                ]
                gaps.append(
                    {
                        **meta,
                        "attribute": attribute,
                        **disparity_summary(y, flagged, sensitive, groups),
                    }
                )

    pd.DataFrame(by_group).to_csv(out / "by_group.csv", index=False)
    gaps = pd.DataFrame(gaps)
    gaps.to_csv(out / "gaps.csv", index=False)

    shown = gaps[(gaps["run"] == "holdout") & (gaps["feature_set"] == "race_aware")]
    columns = [
        "model", "operating_point", "attribute", "fpr_difference", "fnr_difference",
        "equalized_odds_difference", "ppv_difference",
    ]  # fmt: skip
    print(c.fmt(shown[columns]))
    print(f"\n-> {out / 'by_group.csv'}\n-> {out / 'gaps.csv'}")


if __name__ == "__main__":
    main()
