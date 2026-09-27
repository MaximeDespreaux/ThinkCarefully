"""T01. Statistical parity: is the decision independent of race, and of sex?

H0: P(flagged | D = 1) = P(flagged | D = 0), tested with a chi-squared test of the decision
against D, for D = African-American vs Caucasian and D = Female vs Male.

Runs on every committed predictions file (run x feature set), for TabPFN at both operating
points (0.5 and the 0.252 break-even) and for the COMPAS tool on the same defendants.
No refit. Output: tests.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd
from pfn_fairness import chi2_statistical_parity, is_degenerate


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__) / "tests.csv"
    if c.cached(out, args.force):
        return

    rows = []
    for run, feature_set, frame in c.predictions():
        for model, point, threshold, flagged in c.decision_cases(frame):
            for attribute, (a, b) in c.ATTRIBUTES.items():
                group = frame[attribute].to_numpy()
                test = chi2_statistical_parity(flagged, group, (a, b))
                rows.append(
                    {
                        "run": run,
                        "feature_set": feature_set,
                        "model": model,
                        "operating_point": point,
                        "threshold": threshold,
                        "attribute": attribute,
                        "group_a": a,
                        "group_b": b,
                        "n_a": int((group == a).sum()),
                        "n_b": int((group == b).sum()),
                        "selection_rate_a": float(flagged[group == a].mean()),
                        "selection_rate_b": float(flagged[group == b].mean()),
                        "chi2": test["statistic"],
                        "p_value": test["p_value"],
                        "rejects_parity": test["reject_fairness"],
                        "degenerate_operating_point": is_degenerate(flagged),
                    }
                )
    table = pd.DataFrame(rows)
    table["parity_gap"] = table["selection_rate_a"] - table["selection_rate_b"]
    table.to_csv(out, index=False)

    shown = table[(table["run"] == "holdout") & (table["feature_set"] == "race_aware")]
    columns = ["model", "operating_point", "attribute", "parity_gap", "chi2", "p_value"]
    print(c.fmt(shown[columns], 4))
    print(
        f"\nparity rejected in {int(table['rejects_parity'].sum())} of {len(table)} tests\n-> {out}"
    )


if __name__ == "__main__":
    main()
