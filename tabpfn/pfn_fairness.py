"""TabPFN under the shared fairness protocol (compas_scoring.fairness).

Every test, threshold, comparison and conditioning choice lives in the shared module, so
TabPFN, logistic regression and XGBoost are assessed identically. This file only feeds it
TabPFN's saved predictions (tabpfn/artifacts/predictions/), applies the same protocol to the
COMPAS tool's score as the benchmark, and runs the FPDP, which needs the fitted model.

Realigned from the port of the compas-data study (E. Franco-Tetu): its chi-squared (with
continuity correction), CMH, bootstrap inference and TOST are replaced by the shared protocol,
and its calibration-by-group and impossibility sweep moved to the shared module so every
model gets them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from compas_scoring.config import CONFIG
from compas_scoring.data import build_dataset, load_dated
from compas_scoring.fairness import (
    candidate_variables,
    comparison_rows,
    fairness_table,
    fpdp,
    hurlin_test,
    proxy_strata,
)

PREDICTIONS = Path(__file__).resolve().parent / "artifacts" / "predictions"
GROUP_COLUMNS = ["race", "sex", "age_band", "charge_degree"]
SCORES = {"TabPFN": "tabpfn", "COMPAS tool": "compas_tool"}


def cohort(run: str) -> str:
    """The time-ordered runs index the dated cohort; every other run, the modelling table."""
    return "dated" if run.startswith("temporal") else "modelling"


def load_predictions(run: str, feature_set: str) -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS / f"{run}__{feature_set}.csv", index_col="row")


def protocol_table(run: str, feature_set: str) -> pd.DataFrame:
    """The shared protocol on one saved run, for TabPFN and for the COMPAS tool.

    The COMPAS tool's decisions do not depend on the feature set, so compare its rows once
    (e.g. from race_aware) rather than once per TabPFN variant.
    """
    frame = load_predictions(run, feature_set)
    strata = proxy_strata(frame.index, cohort(run))
    tables = []
    for model, column in SCORES.items():
        table = fairness_table(frame["y"], frame[column], frame[GROUP_COLUMNS], strata)
        tables.append(table.assign(model=model, run=run, feature_set=feature_set))
    return pd.concat(tables, ignore_index=True)


def feature_matrix(run: str, feature_set: str, index) -> pd.DataFrame:
    """The feature matrix TabPFN saw for these test rows (what the FPDP perturbs)."""
    columns = CONFIG.features(feature_set)
    if cohort(run) == "dated":
        return load_dated().loc[index, columns].astype(float)
    return build_dataset(feature_set).X.loc[index]


def fpdp_candidates(
    model,
    run: str,
    feature_set: str,
    threshold: float,
    conditional: bool = True,
    grids: dict | None = None,
    comparison=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FPDP curves for every feature, and the candidate variables (course step 2).

    ``model`` is a TabPFN fitted on the run's training set. Each grid value is one call to the
    model, so pass a coarse grid for priors in ``grids`` (e.g. {"Number_of_Priors": [0, 1, 3,
    6, 10, 20]}); binary features default to their two values.
    """
    comparison = comparison or CONFIG.fairness.comparisons[0]
    frame = load_predictions(run, feature_set)
    groups, strata = frame[GROUP_COLUMNS], proxy_strata(frame.index, cohort(run))
    X = feature_matrix(run, feature_set, frame.index)

    index, protected = comparison_rows(groups, comparison)
    decisions = (frame.loc[index, "tabpfn"] >= threshold).astype(int)
    baseline = hurlin_test(decisions, protected, strata.loc[index] if conditional else None)

    def predict(features: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(features)[:, 1]

    grids = grids or {}
    curves = pd.concat(
        [
            fpdp(
                predict,
                X,
                groups,
                strata,
                feature,
                threshold,
                comparison,
                grids.get(feature),
                conditional,
            )
            for feature in X.columns
        ],
        ignore_index=True,
    )
    return curves, candidate_variables(curves, baseline["p_value"])
