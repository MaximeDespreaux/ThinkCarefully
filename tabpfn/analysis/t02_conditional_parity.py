"""T02. Conditional statistical parity: does the disparity survive among similar defendants?

H0: flagged independent of D given X_c, with X_c = the race proxies from the EDA held fixed:
priors band (0 / 1-3 / 4+) x age band x charge degree, read from each defendant's record, so
the strata are the same whatever features the model saw. Hurlin likelihood-ratio test summed
over the informative strata, plus Cochran-Mantel-Haenszel, the Mantel-Haenszel common odds
ratio (the effect size: "among defendants with the same proxies, the protected group has OR
times the odds of being flagged") and Breslow-Day (does one odds ratio fit every stratum).

From the shared protocol (T01's protocol.csv). No refit. Output: conditional_parity.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd

PROTOCOL = c.ANALYSIS / "t01_statistical_parity" / "protocol.csv"
COLUMNS = [
    "run", "feature_set", "model", "threshold_name", "threshold", "comparison", "attribute",
    "primary", "selection_rate", "degenerate", "rate_protected", "rate_reference",
    "strata_used", "hurlin_statistic", "hurlin_df", "p_value", "p_holm", "reject_fairness",
    "cmh_statistic", "cmh_p", "mh_odds_ratio", "mh_or_low", "mh_or_high", "breslow_day_p",
]  # fmt: skip


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__) / "conditional_parity.csv"
    if c.cached(out, args.force):
        return

    protocol = pd.read_csv(PROTOCOL)
    table = protocol[protocol["metric"] == "conditional_statistical_parity"][COLUMNS]
    table.to_csv(out, index=False)

    shown = table[(table["run"] == "holdout") & (table["feature_set"] == "race_aware")]
    print(c.fmt(shown[["model", "threshold_name", "comparison", "strata_used", "p_holm",
                       "mh_odds_ratio", "mh_or_low", "mh_or_high"]], 4))  # fmt: skip
    tabpfn = table[table["model"] == "TabPFN"]
    print(
        f"\nTabPFN: conditional parity rejected in {int(tabpfn['reject_fairness'].sum())} "
        f"of {len(tabpfn)} cases\n-> {out}"
    )


if __name__ == "__main__":
    main()
