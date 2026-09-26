"""Fit TabPFN on every design and score it. Run via `make tabpfn` (`--help` for options).

Designs, all defined in compas_scoring.data so the other models use the same cuts:

    holdout      the shared stratified 70/30 split                        (headline 1)
    partition    X1+X2 -> X3                                              (headline 2)
                 X1 -> X3, X2 -> X3              stability design 1: same test set X3
    temporal     first 40% -> next 20%, first 70% -> last 30%
                 on the dated cohort             stability design 2: does the story hold
                                                 once the model sees newer defendants?

Outputs, in tabpfn/artifacts/ (committed, so the analysis needs no refit; rerun after any
change to the model or the splits, then commit the new files):

    predictions/<run>__<feature_set>.csv   one row per test defendant: y, TabPFN score,
                                           COMPAS score_factor, race / sex / age band /
                                           charge degree -- everything the fairness and
                                           stability analyses need, without refitting
    performance.csv                        pfn_metrics.performance for TabPFN and for the
                                           COMPAS benchmark on the same test rows
    runs.csv                               sizes, base rates, periods and fit/predict seconds

Fitted models are not saved: TabPFN "fitting" only stores the training rows, so refitting
with pfn_model.build_model() on the same split reproduces a run exactly.

Predictions are cached: a run whose predictions file exists is not refitted (--force refits).
"""

from __future__ import annotations

# isort: split
import compas_scoring  # noqa: F401  (pins thread pools before numpy and torch load)

# isort: split
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pfn_metrics import performance
from pfn_model import build_model

from compas_scoring.config import CONFIG
from compas_scoring.data import Dataset, partition, temporal_windows, train_test

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
PRED = ART / "predictions"

DESIGNS = ("holdout", "partition", "temporal")
# The race-aware set plus the race-blind one, so the fairness analysis can compare the two.
DEFAULT_FEATURE_SETS = ("race_aware", "race_blind")


@dataclass(frozen=True)
class Run:
    name: str  # used in file names
    design: str
    train: Dataset
    test: Dataset
    cohort: str  # "modelling" (the 6,172-row table) or "dated" (see data.load_dated)
    note: str = ""


def build_runs(design: str, feature_set: str) -> list[Run]:
    if design == "holdout":
        train, test = train_test(feature_set)
        return [Run("holdout", design, train, test, "modelling")]

    if design == "partition":
        parts = partition(feature_set)
        x12 = Dataset(
            X=pd.concat([parts["X1"].X, parts["X2"].X]).sort_index(),
            y=pd.concat([parts["X1"].y, parts["X2"].y]).sort_index(),
            incumbent=pd.concat([parts["X1"].incumbent, parts["X2"].incumbent]).sort_index(),
            groups=pd.concat([parts["X1"].groups, parts["X2"].groups]).sort_index(),
            feature_set=feature_set,
        )
        return [
            Run("X1+X2_to_X3", design, x12, parts["X3"], "modelling"),
            Run("X1_to_X3", design, parts["X1"], parts["X3"], "modelling"),
            Run("X2_to_X3", design, parts["X2"], parts["X3"], "modelling"),
        ]

    if design == "temporal":
        runs = []
        for i, window in enumerate(temporal_windows(feature_set), start=1):
            period = (
                f"train {window.train_period[0]:%Y-%m-%d}..{window.train_period[1]:%Y-%m-%d}, "
                f"test {window.test_period[0]:%Y-%m-%d}..{window.test_period[1]:%Y-%m-%d}"
            )
            runs.append(Run(f"temporal_{i}", design, window.train, window.test, "dated", period))
        return runs

    raise ValueError(f"Unknown design {design!r}; choose from {DESIGNS}")


def predictions_path(run: Run) -> Path:
    return PRED / f"{run.name}__{run.train.feature_set}.csv"


def fit_and_predict(run: Run) -> tuple[pd.DataFrame, float, float]:
    model = build_model()
    start = time.perf_counter()
    model.fit(run.train.X, run.train.y)
    fit_s = time.perf_counter() - start

    start = time.perf_counter()
    score = model.predict_proba(run.test.X)[:, 1]
    predict_s = time.perf_counter() - start

    frame = pd.DataFrame(
        {"y": run.test.y, "tabpfn": score, "compas": run.test.incumbent},
        index=run.test.X.index.rename("row"),
    ).join(run.test.groups)
    return frame, fit_s, predict_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--designs", nargs="+", choices=DESIGNS, default=list(DESIGNS))
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=sorted(CONFIG.feature_sets),
        default=list(DEFAULT_FEATURE_SETS),
    )
    parser.add_argument("--force", action="store_true", help="refit even if cached")
    args = parser.parse_args()

    PRED.mkdir(parents=True, exist_ok=True)
    perf_rows, run_rows = [], []

    for design in args.designs:
        for feature_set in args.feature_sets:
            for run in build_runs(design, feature_set):
                path = predictions_path(run)
                label = f"{run.name:<12} {feature_set:<10}"
                if path.exists() and not args.force:
                    frame = pd.read_csv(path, index_col="row")
                    fit_s = predict_s = float("nan")
                    print(f"  {label} cached")
                else:
                    frame, fit_s, predict_s = fit_and_predict(run)
                    frame.to_csv(path)
                    print(f"  {label} fit {fit_s:5.1f}s  predict {predict_s:5.1f}s")

                meta = {"run": run.name, "design": design, "feature_set": feature_set}
                run_rows.append(
                    {
                        **meta,
                        "cohort": run.cohort,
                        "n_train": len(run.train),
                        "n_test": len(run.test),
                        "train_base_rate": run.train.base_rate,
                        "test_base_rate": run.test.base_rate,
                        "period": run.note,
                        "fit_seconds": fit_s,
                        "predict_seconds": predict_s,
                    }
                )
                for model, column in (("TabPFN", "tabpfn"), ("COMPAS", "compas")):
                    for row in performance(frame["y"], frame[column]):
                        perf_rows.append({**meta, "model": model, **row})

    runs = pd.DataFrame(run_rows)
    perf = pd.DataFrame(perf_rows)
    runs.to_csv(ART / "runs.csv", index=False)
    perf.to_csv(ART / "performance.csv", index=False)

    # COMPAS's score_factor is binary, so its two operating points coincide; show it once.
    shown = perf[(perf["model"] == "TabPFN") | (perf["operating_point"] == "0.5")]
    columns = [
        "run", "feature_set", "model", "operating_point", "auc", "auc_ci_low", "auc_ci_high",
        "accuracy", "f1", "recall", "type_i_error", "type_ii_error", "cost_per_defendant",
    ]  # fmt: skip
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n" + shown[columns].round(3).to_string(index=False))
    print(f"\n-> {ART / 'performance.csv'}\n-> {ART / 'runs.csv'}\n-> {PRED}/")

    # TODO(analysis): the predictions/ files are the input for the three analysis dimensions.
    # See tabpfn/README.md, "For the analysis", for what each dimension still needs.
    return 0


if __name__ == "__main__":
    sys.exit(main())
