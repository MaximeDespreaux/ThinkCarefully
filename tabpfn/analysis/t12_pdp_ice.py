"""T12. PDP and ICE on the number of priors, the only non-binary feature.

The PDP is the average predicted risk as priors sweeps its range; the ICE curves are the same
sweep for individual defendants. If the ICE curves fan out, or run in opposite directions, the
PDP line describes nobody in particular (ice_heterogeneity measures it).

Refits TabPFN once per run in _common.INTERPRET_RUNS. PDP on 500 test rows, ICE on 100 of
them, 20 grid points each. Outputs: pdp.csv, ice.csv (centred at 0 priors), ice_summary.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd
from pfn_interpret import PRIORS, center_ice, ice_curves, ice_heterogeneity

N_ICE = 100


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__)
    if c.cached(out / "ice_summary.csv", args.force):
        return

    pdp, ice, summary = [], [], []
    for run, feature_set in c.INTERPRET_RUNS:
        model, spec = c.fit(run, feature_set)
        X = c.explain_sample(spec.test.X)
        curves = ice_curves(model, X, PRIORS, n_curves=len(X))  # every explained row
        meta = {"run": run, "feature_set": feature_set}

        # The PDP is the mean of the ICE curves: one sweep serves both.
        mean = curves.mean(axis=0)
        pdp.append(pd.DataFrame({**meta, PRIORS: mean.index, "predicted_risk": mean.to_numpy()}))

        shown = center_ice(curves.iloc[:N_ICE])
        long = shown.reset_index(names="row").melt(
            id_vars="row", var_name=PRIORS, value_name="risk_change_from_min"
        )
        ice.append(long.assign(**meta))
        summary.append({**meta, **ice_heterogeneity(curves)})

    pd.concat(pdp, ignore_index=True).to_csv(out / "pdp.csv", index=False)
    pd.concat(ice, ignore_index=True).to_csv(out / "ice.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "ice_summary.csv", index=False)

    print(c.fmt(summary))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
