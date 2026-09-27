"""T11. Impurity (Gini, Shannon entropy, misclassification error) through a global surrogate.

Impurity measures are tree criteria and TabPFN has no splits. So a shallow tree (depth 4) is
fitted on TabPFN's *own outputs* and its impurity decreases are read instead:
    * a regression tree on TabPFN's scores gives the fidelity R^2 and readable rules;
    * a classification tree on TabPFN's decisions at the break-even threshold (grown with gini,
      and again with entropy) gives each feature's share of the impurity decrease under all
      three criteria, with its fidelity (accuracy against TabPFN's decisions).
A surrogate explains the model, not the data, and only as far as its fidelity goes.

No refit: TabPFN's scores on each test set are already in predictions/. Runs on every
committed predictions file. Outputs: surrogate.csv (fidelity), impurity.csv, rules/<run>.txt.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import numpy as np
import pandas as pd
from pfn_interpret import surrogate_impurity
from pfn_metrics import break_even_threshold

from compas_scoring.config import CONFIG


class StoredScores:
    """Answers predict_proba from TabPFN's saved scores, looked up by row index."""

    def __init__(self, scores: pd.Series):
        self.scores = scores

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.scores.loc[X.index].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__)
    if c.cached(out / "surrogate.csv", args.force):
        return
    (out / "rules").mkdir(exist_ok=True)

    fidelity, impurity = [], []
    for run, feature_set, frame in c.predictions():
        X = c.cohort_rows(run, frame.index)[CONFIG.features(feature_set)].astype(float)
        result = surrogate_impurity(StoredScores(frame["tabpfn"]), X, break_even_threshold())
        meta = {"run": run, "feature_set": feature_set}
        fidelity.append(
            {**meta, **{k: v for k, v in result.items() if k not in ("rules", "impurity")}}
        )
        impurity.append(result["impurity"].assign(**meta))
        (out / "rules" / f"{run}__{feature_set}.txt").write_text(
            f"Depth-4 regression tree on TabPFN's scores, {run} x {feature_set}, "
            f"fidelity R^2 = {result['fidelity_r2']:.3f}\n\n{result['rules']}"
        )

    fidelity = pd.DataFrame(fidelity)
    fidelity.to_csv(out / "surrogate.csv", index=False)
    impurity = pd.concat(impurity, ignore_index=True)
    impurity = impurity[["run", "feature_set", "tree_grown_with", "criterion", "feature",
                         "impurity_decrease_share"]]  # fmt: skip
    impurity.to_csv(out / "impurity.csv", index=False)

    print(c.fmt(fidelity[fidelity["feature_set"] == "race_aware"].drop(columns="max_depth")))
    holdout = impurity[
        (impurity["run"] == "holdout")
        & (impurity["feature_set"] == "race_aware")
        & (impurity["tree_grown_with"] == "gini")
    ].pivot_table(index="feature", columns="criterion", values="impurity_decrease_share")
    print("\nholdout x race_aware, gini-grown tree, share of impurity decrease:")
    print(c.fmt(holdout.sort_values("gini", ascending=False).reset_index()))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
