"""Is every split a representative sample of the cohort? Run via `make split-balance`.

For each split and each protected attribute (race, sex, age band) plus charge degree, compares
the split's group shares with the full cohort's: the largest gap in percentage points, and a
chi-squared test of the split against the rows outside it. Writes artifacts/split_balance.csv.

    holdout      train70 / test30 against the modelling table
    partition    X1 / X2 / X3 against the modelling table
    temporal     each window's train and test against the dated cohort. These are cut by
                 date, not sampled, so a gap here is a real change in who was screened over
                 time: part of what the time-ordered design measures, not a sampling flaw.
"""

from __future__ import annotations

# isort: split
import compas_scoring  # noqa: F401  (pins thread pools before numpy loads)

# isort: split
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from compas_scoring.config import CONFIG
from compas_scoring.data import (
    build_dataset,
    group_frame,
    load_dated,
    partition_index,
    split_index,
    temporal_windows,
)

ATTRIBUTES = ["race", "sex", "age_band", "charge_degree"]


def balance(groups: pd.DataFrame, splits: dict[str, pd.Index], design: str) -> list[dict]:
    rows = []
    for attribute in ATTRIBUTES:
        column = groups[attribute]
        cohort = column.value_counts(normalize=True)
        for name, idx in splits.items():
            inside, outside = column.loc[idx], column.drop(idx)
            share = inside.value_counts(normalize=True).reindex(cohort.index, fill_value=0.0)
            label = np.r_[np.ones(len(inside)), np.zeros(len(outside))]
            table = pd.crosstab(label, pd.concat([inside, outside]).to_numpy())
            p_value = chi2_contingency(table)[1]
            gap = (share - cohort).abs()
            rows.append(
                {
                    "design": design,
                    "split": name,
                    "attribute": attribute,
                    "n": len(idx),
                    "max_gap_pp": 100 * gap.max(),
                    "worst_group": gap.idxmax(),
                    "p_value": p_value,
                }
            )
    return rows


def main() -> int:
    groups = build_dataset().groups
    train, test = split_index()
    x1, x2, x3 = partition_index()
    rows = balance(groups, {"train70": train, "test30": test}, "holdout")
    rows += balance(groups, {"X1": x1, "X2": x2, "X3": x3}, "partition")

    dated = load_dated()
    dated_groups = group_frame(dated)
    for i, window in enumerate(temporal_windows(), start=1):
        splits = {f"t{i}_train": window.train.X.index, f"t{i}_test": window.test.X.index}
        rows += balance(dated_groups, splits, "temporal")

    table = pd.DataFrame(rows)
    out = CONFIG.path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "split_balance.csv", index=False)

    wide = table.pivot_table(index=["design", "split"], columns="attribute", values="max_gap_pp")
    p = table.pivot_table(index=["design", "split"], columns="attribute", values="p_value")
    print("Largest gap from the cohort's group shares, in percentage points (p-value):\n")
    shown = wide.round(2).astype(str) + " (" + p.round(2).astype(str) + ")"
    print(shown[ATTRIBUTES].to_string())
    print("\nTemporal windows are cut by date, not sampled: gaps there are real drift over time.")
    print(f"-> {out / 'split_balance.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
