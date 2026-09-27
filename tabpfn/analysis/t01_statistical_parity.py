"""T01. The shared fairness protocol on every run, and statistical parity.

Runs compas_scoring.fairness (through pfn_fairness.protocol_table) on every committed
predictions file, for TabPFN and for the COMPAS tool on the same defendants, so TabPFN is
tested exactly like the logistic regression and XGBoost. Settings are in
[tool.compas_scoring.fairness]: thresholds 0.5 and 0.252; comparisons African-American (primary),
Hispanic and Other vs Caucasian, and Female vs Male; gaps read protected minus reference;
p-values Holm-corrected across comparisons within each threshold and metric.

protocol.csv holds every metric and is read by T02-T05, T09 and T18. This test reports its
first metric, **statistical parity**: H0 P(flagged | D = 1) = P(flagged | D = 0), a
two-proportion z-test on the flag-rate gap (= Pearson chi2), with the Hurlin likelihood-ratio
test alongside. No refit. Outputs: protocol.csv, statistical_parity.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd
from pfn_fairness import protocol_table

COLUMNS = [
    "run", "feature_set", "model", "threshold_name", "threshold", "comparison", "attribute",
    "primary", "selection_rate", "degenerate", "rate_protected", "rate_reference",
    "n_protected", "n_reference", "gap", "ci_low", "ci_high", "z", "p_value", "p_holm",
    "reject_fairness", "hurlin_statistic", "hurlin_p",
]  # fmt: skip


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__)
    if c.cached(out / "protocol.csv", args.force):
        return

    tables = [protocol_table(run, feature_set) for run, feature_set, _ in c.predictions()]
    protocol = pd.concat(tables, ignore_index=True)
    first = ["run", "feature_set", "model", "threshold_name", "comparison", "metric"]
    protocol = protocol[first + [col for col in protocol.columns if col not in first]]
    protocol.to_csv(out / "protocol.csv", index=False)

    parity = protocol[protocol["metric"] == "statistical_parity"][COLUMNS]
    parity.to_csv(out / "statistical_parity.csv", index=False)

    shown = parity[(parity["run"] == "holdout") & (parity["feature_set"] == "race_aware")]
    print(c.fmt(shown[["model", "threshold_name", "comparison", "gap", "p_holm", "hurlin_p"]], 4))
    tabpfn = parity[parity["model"] == "TabPFN"]
    print(
        f"\nTabPFN: parity rejected in {int(tabpfn['reject_fairness'].sum())} of {len(tabpfn)}"
        f" run x feature set x threshold x comparison cases\n-> {out / 'protocol.csv'}"
        f"\n-> {out / 'statistical_parity.csv'}"
    )


if __name__ == "__main__":
    main()
