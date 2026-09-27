"""T14. SHAP values with KernelSHAP, and the efficiency check.

There is no TreeSHAP for TabPFN, so KernelSHAP approximates the Shapley values from the model's
answers. Efficiency is checked: for each explained defendant the SHAP values must add up to
prediction minus base value; the residual measures how far to trust the approximation.

Budget from [tool.compas_scoring.iterations]: 25 explained rows, 25 k-means background rows
from the training set, 256 coalition samples. Refits TabPFN once per run in
_common.INTERPRET_RUNS. Outputs: shap_importance.csv (mean |SHAP| per feature), shap_values.csv
(one row per explained defendant), efficiency.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd
from pfn_interpret import global_importance, kernel_shap, shap_efficiency_check


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__)
    if c.cached(out / "efficiency.csv", args.force):
        return

    importance, values, efficiency = [], [], []
    for run, feature_set in c.INTERPRET_RUNS:
        model, spec = c.fit(run, feature_set)
        explain = c.explain_sample(spec.test.X)
        shap_values, base, rows, seconds = kernel_shap(model, spec.train.X, explain)
        predictions = model.predict_proba(rows)[:, 1]
        meta = {"run": run, "feature_set": feature_set}

        mean_abs = global_importance(shap_values, rows.columns)
        importance.append(
            pd.DataFrame(
                {**meta, "feature": mean_abs.index, "mean_abs_shap": mean_abs.to_numpy()}
            ).assign(rank=lambda f: f["mean_abs_shap"].rank(ascending=False))
        )
        per_row = pd.DataFrame(shap_values, columns=rows.columns, index=rows.index)
        values.append(
            per_row.assign(**meta, base_value=base, prediction=predictions).reset_index(names="row")
        )
        efficiency.append(
            {
                **meta,
                "base_value": base,
                "seconds": seconds,
                "seconds_per_explained_row": seconds / len(rows),
                **shap_efficiency_check(shap_values, base, predictions),
            }
        )
        print(f"  {run} x {feature_set}: {seconds:.0f}s", flush=True)

    pd.concat(importance, ignore_index=True).to_csv(out / "shap_importance.csv", index=False)
    pd.concat(values, ignore_index=True).to_csv(out / "shap_values.csv", index=False)
    efficiency = pd.DataFrame(efficiency)
    efficiency.to_csv(out / "efficiency.csv", index=False)

    print(c.fmt(efficiency, 6))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
