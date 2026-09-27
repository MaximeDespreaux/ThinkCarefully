"""T02. Conditional statistical parity: does the disparity survive among similar defendants?

H0: flagged is independent of D given X_c, tested with a Cochran-Mantel-Haenszel test, where
X_c = priors band (0 / 1-3 / 4+) x charge degree (felony / misdemeanour): six strata of
"legitimate" risk factors. If the gap were only "their records are longer", it would vanish
within strata.

The priors count is read back from the cohort (predictions files keep only the group labels),
so the strata are the same whatever the feature set. No refit. Output: cmh.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd
from pfn_fairness import cmh_test

PRIORS_BANDS = [-0.5, 0.5, 3.5, float("inf")]
PRIORS_LABELS = ["0", "1-3", "4+"]


def strata(run: str, frame: pd.DataFrame) -> pd.Series:
    priors = c.cohort_rows(run, frame.index)["Number_of_Priors"]
    band = pd.cut(priors, PRIORS_BANDS, labels=PRIORS_LABELS).astype(str)
    return band.to_numpy() + " | " + frame["charge_degree"].to_numpy()


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__) / "cmh.csv"
    if c.cached(out, args.force):
        return

    rows = []
    for run, feature_set, frame in c.predictions():
        stratum = strata(run, frame)
        for model, point, threshold, flagged in c.decision_cases(frame):
            for attribute, groups in c.ATTRIBUTES.items():
                test = cmh_test(flagged, frame[attribute].to_numpy(), stratum, groups)
                rows.append(
                    {
                        "run": run,
                        "feature_set": feature_set,
                        "model": model,
                        "operating_point": point,
                        "threshold": threshold,
                        "attribute": attribute,
                        "strata": "priors band x charge degree",
                        "n_strata_used": test["n_strata"],
                        "cmh": test["statistic"],
                        "p_value": test["p_value"],
                        "rejects_conditional_parity": test["reject_fairness"],
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)

    shown = table[(table["run"] == "holdout") & (table["feature_set"] == "race_aware")]
    columns = ["model", "operating_point", "attribute", "n_strata_used", "cmh", "p_value"]
    print(c.fmt(shown[columns], 4))
    print(
        f"\nconditional parity rejected in {int(table['rejects_conditional_parity'].sum())} "
        f"of {len(table)} tests\n-> {out}"
    )


if __name__ == "__main__":
    main()
