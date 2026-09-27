"""T04. Fairness equivalence (TOST, Schuirmann 1987): can the model be *certified* fair?

A plain test has H0 = "fair", so not rejecting it certifies nothing. TOST reverses the burden:

    H0: |theta| >= delta  (unfair)      H1: -delta < theta < delta  (fair)

with theta = p_1 - p_0 for three gaps (FPR, demographic parity, equalized odds). Certified at
delta when the 90% bootstrap interval lies inside (-delta, delta). ``minimum_delta`` is the
tightest tolerance that could be certified, so no delta has to be picked after the fact.

1,000 bootstrap resamples of the scored test set per case (resampling defendants, not
refitting: the question is how precisely this model's gap is measured). Vectorised, since the
earlier study's per-draw loop would take ~30 min over all 42 prediction files. No refit.
Output: equivalence.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import numpy as np
import pandas as pd
from pfn_fairness import equivalence_test

from compas_scoring.config import CONFIG

N_BOOT = 1000
DELTA = 0.05  # the tolerance to certify at; minimum_delta does not depend on it


def bootstrap_gaps(y, flagged, group, groups, n_boot: int = N_BOOT) -> pd.DataFrame:
    """One row per bootstrap draw: FPR, TPR and selection-rate gaps (group a minus group b)."""
    rng = np.random.default_rng(CONFIG.random_state)
    n = len(y)
    # Row k of `weight` counts how often each defendant appears in draw k.
    weight = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)])

    def rate(numerator, denominator):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (weight @ numerator) / (weight @ denominator)

    rates = {}
    for label, name in zip("ab", groups):
        g = (group == name).astype(float)
        rates[label] = {
            "fpr": rate(g * (y == 0) * flagged, g * (y == 0)),
            "tpr": rate(g * (y == 1) * flagged, g * (y == 1)),
            "sel": rate(g * flagged, g),
        }
    fpr = rates["a"]["fpr"] - rates["b"]["fpr"]
    tpr = rates["a"]["tpr"] - rates["b"]["tpr"]
    return pd.DataFrame(
        {
            "fpr_difference": fpr,
            "demographic_parity_difference": rates["a"]["sel"] - rates["b"]["sel"],
            # Signed like the larger of the two gaps, so TOST sees which way it points.
            "equalized_odds_difference": np.where(np.abs(fpr) >= np.abs(tpr), fpr, tpr),
        }
    )


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__) / "equivalence.csv"
    if c.cached(out, args.force):
        return

    rows = []
    for run, feature_set, frame in c.predictions():
        y = frame["y"].to_numpy()
        for model, point, threshold, flagged in c.decision_cases(frame):
            for attribute, groups in c.ATTRIBUTES.items():
                draws = bootstrap_gaps(y, flagged, frame[attribute].to_numpy(), groups)
                for metric in draws.columns:
                    sample = draws[metric].to_numpy()
                    rows.append(
                        {
                            "run": run,
                            "feature_set": feature_set,
                            "model": model,
                            "operating_point": point,
                            "threshold": threshold,
                            "attribute": attribute,
                            "metric": metric,
                            "ci95_lower": float(np.nanquantile(sample, 0.025)),
                            "ci95_upper": float(np.nanquantile(sample, 0.975)),
                            **equivalence_test(sample, DELTA),
                        }
                    )
    table = pd.DataFrame(rows).rename(columns={"estimate": "bootstrap_mean"})
    table.to_csv(out, index=False)

    shown = table[(table["run"] == "holdout") & (table["feature_set"] == "race_aware")]
    columns = ["model", "operating_point", "attribute", "metric", "bootstrap_mean",
               "minimum_delta", "certified_fair"]  # fmt: skip
    print(c.fmt(shown[columns]))
    print(
        f"\ncertified fair at delta = {DELTA}: {int(table['certified_fair'].sum())} "
        f"of {len(table)}\n-> {out}"
    )


if __name__ == "__main__":
    main()
