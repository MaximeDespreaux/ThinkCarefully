"""Build and validate the modelling table. Run via `make data`."""

from __future__ import annotations

import sys

import pandas as pd

import compas_scoring  # noqa: F401  (pins thread pools; import first)
from compas_scoring.config import CONFIG
from compas_scoring.data import build_dataset, group_frame, load_raw, train_test


def main() -> int:
    df = load_raw()
    print(f"Loaded {CONFIG.data_path.name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Target '{CONFIG.target}' base rate: {df[CONFIG.target].mean():.4f}")

    groups = group_frame(df)
    print("\nProtected attributes")
    for column in groups.columns:
        counts = groups[column].value_counts()
        rates = df.groupby(groups[column])[CONFIG.target].mean()
        summary = pd.DataFrame({"n": counts, "recid_rate": rates.round(4)})
        print(f"\n{column}:")
        print(summary.to_string())

    print("\nFeature sets")
    for name in CONFIG.feature_sets:
        data = build_dataset(name)
        print(f"  {name:11s} {data.X.shape[1]:2d} features: {', '.join(data.X.columns)}")
        assert CONFIG.incumbent not in data.X.columns
        assert CONFIG.target not in data.X.columns

    # Checking the split is stratified is a data-integrity check, not model evaluation:
    # no model output is scored here. Benchmarking the incumbent belongs in run_analysis,
    # which is the single place the holdout is evaluated.
    train, test = train_test("race_aware")
    print(f"\nSplit: {len(train):,} train / {len(test):,} test")
    print(f"  base rate  train {train.base_rate:.4f}   test {test.base_rate:.4f}")
    assert not set(train.X.index) & set(test.X.index), "train/test overlap"

    print("\nOK - modelling table validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
