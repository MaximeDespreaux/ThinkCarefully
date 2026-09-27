"""T06. Fairness partial dependence (FPDP): which variable could make the model fair?

The course's 3-step approach: test -> identify -> mitigate. T01-T02 are the test; this is the
identify step, with the shared protocol's FPDP (compas_scoring.fairness.fpdp through
pfn_fairness.fpdp_candidates). For each feature X_A and each value v, every defendant's X_A is
set to v, TabPFN rescores them, and the Hurlin test is rerun. X_A is a *candidate variable* if
fairness is rejected on the real data and some v lifts the p-value above 0.05 without
everyone getting the same decision.

Run on holdout x race_aware, for the primary race comparison and for sex, at both thresholds,
for statistical parity and for conditional parity (proxies held fixed). Binary features take
their two values (as in the shared protocol, a race dummy set to 1 does not clear the others);
priors uses a coarse grid. The counterfactual frames repeat across thresholds and tests, so
their scores are cached: ~30 model calls in all. Refits TabPFN once.
Outputs: fpdp_curves.csv, fpdp_candidates.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import hashlib

import numpy as np
import pandas as pd
from pfn_fairness import fpdp_candidates

from compas_scoring.config import CONFIG

RUN, FEATURE_SET = "holdout", "race_aware"
PRIORS_GRID = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]
COMPARISONS = [CONFIG.fairness.comparisons[0], ("sex", "Female", "Male")]


class Cached:
    """Wraps the fitted model so an identical feature frame is scored only once."""

    def __init__(self, model):
        self.model, self.cache, self.calls = model, {}, 0

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        values = X.to_numpy(dtype=float)
        key = hashlib.sha1(values.tobytes() + X.index.to_numpy().tobytes()).hexdigest()
        if key not in self.cache:
            self.calls += 1
            self.cache[key] = self.model.predict_proba(X)
        return self.cache[key]


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__)
    if c.cached(out / "fpdp_candidates.csv", args.force):
        return

    model, _ = c.fit(RUN, FEATURE_SET)
    cached = Cached(model)

    curves, candidates = [], []
    for comparison in COMPARISONS:
        for threshold_name, threshold in c.operating_points().items():
            for conditional in (False, True):
                curve, found = fpdp_candidates(
                    cached, RUN, FEATURE_SET, threshold, conditional,
                    {"Number_of_Priors": PRIORS_GRID}, comparison,
                )  # fmt: skip
                meta = {
                    "comparison": f"{comparison[1]} vs {comparison[2]}",
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                    "test": "conditional parity" if conditional else "statistical parity",
                }
                curves.append(curve.assign(**meta))
                candidates.append(found.assign(**meta))
                print(f"  {meta['comparison']}, {threshold_name}, {meta['test']}: "
                      f"{cached.calls} model calls so far", flush=True)  # fmt: skip

    first = ["comparison", "threshold_name", "threshold", "test"]
    curves = pd.concat(curves, ignore_index=True)
    curves = curves[first + [col for col in curves.columns if col not in first]]
    curves.to_csv(out / "fpdp_curves.csv", index=False)
    candidates = pd.concat(candidates, ignore_index=True)
    candidates = candidates[first + [col for col in candidates.columns if col not in first]]
    candidates.to_csv(out / "fpdp_candidates.csv", index=False)

    found = candidates[candidates["is_candidate"]]
    columns = ["comparison", "threshold_name", "test", "feature", "best_value", "best_p",
               "baseline_p"]  # fmt: skip
    print(c.fmt(found[columns], 4) if len(found) else "no candidate variable")
    print(f"\n-> {out / 'fpdp_curves.csv'}\n-> {out / 'fpdp_candidates.csv'}")


if __name__ == "__main__":
    main()
