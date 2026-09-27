"""Shared plumbing for the numbered analysis scripts (tNN_*.py) in this folder.

Every script is run as ``uv run python tabpfn/analysis/tNN_<name>.py`` (or ``make tabpfn-tNN``).
Importing this module first:

* pins the numeric thread pools (``import compas_scoring``) before numpy or torch load;
* puts ``tabpfn/`` on the path, so the scripts can import pfn_* and run_tabpfn.

Each script writes to ``tabpfn/artifacts/analysis/tNN_<name>/`` and reads only the committed
predictions, performance.csv, or the outputs of lower-numbered scripts.
"""

from __future__ import annotations

# isort: split
import compas_scoring  # noqa: F401  (pins thread pools before numpy and torch load)

# isort: split
import argparse
import sys
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import pandas as pd

TABPFN = Path(__file__).resolve().parents[1]
if str(TABPFN) not in sys.path:
    sys.path.insert(0, str(TABPFN))

from pfn_metrics import thresholds  # noqa: E402

from compas_scoring.data import load_dated, load_raw  # noqa: E402

ART = TABPFN / "artifacts"
PREDICTIONS = ART / "predictions"
ANALYSIS = ART / "analysis"

# Stability design 1 (same test set X3) and design 2 (successive time windows).
PAIRS = {
    "X1_vs_X2": ("X1_to_X3", "X2_to_X3"),
    "temporal_1_vs_2": ("temporal_1", "temporal_2"),
}


# The fits the interpretability scripts explain: every design with all features, plus the
# race-blind holdout model to see what the explanation does once race is removed.
INTERPRET_RUNS = [
    ("holdout", "race_aware"),
    ("X1+X2_to_X3", "race_aware"),
    ("X1_to_X3", "race_aware"),
    ("X2_to_X3", "race_aware"),
    ("temporal_1", "race_aware"),
    ("temporal_2", "race_aware"),
    ("holdout", "race_blind"),
]
# Test rows explained per run, drawn once with the project seed. Small on purpose: every
# explanation method multiplies it by its own number of model queries.
N_EXPLAIN = 500


def explain_sample(X: pd.DataFrame, n: int = N_EXPLAIN) -> pd.DataFrame:
    """A reproducible sample of test rows (the same rows for every run on the same test set)."""
    from compas_scoring.config import CONFIG

    return X if len(X) <= n else X.sample(n, random_state=CONFIG.random_state)


def output_dir(script_file: str) -> Path:
    """tabpfn/artifacts/analysis/<script name>/, created on first use."""
    path = ANALYSIS / Path(script_file).stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args(description: str, extra=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    if extra is not None:
        extra(parser)
    return parser.parse_args()


def cached(path: Path, force: bool) -> bool:
    """True when ``path`` exists and should be reused (the run_tabpfn.py convention)."""
    if path.exists() and not force:
        print(f"  cached: {path.relative_to(TABPFN.parent)}  (--force recomputes)")
        return True
    return False


def split_name(path: Path) -> tuple[str, str]:
    run, feature_set = path.stem.split("__")
    return run, feature_set


def read_predictions(run: str, feature_set: str) -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS / f"{run}__{feature_set}.csv", index_col="row")


def predictions() -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Every committed predictions file, as (run, feature_set, frame)."""
    for path in sorted(PREDICTIONS.glob("*.csv")):
        run, feature_set = split_name(path)
        yield run, feature_set, pd.read_csv(path, index_col="row")


def operating_points() -> dict[str, float]:
    """{"0.5": 0.5, "break_even": 0.252}, from pfn_metrics."""
    return thresholds()


@lru_cache(maxsize=2)
def _cohort(dated: bool) -> pd.DataFrame:
    return load_dated() if dated else load_raw()


def cohort_rows(run: str, rows) -> pd.DataFrame:
    """The full feature rows behind a predictions file (all features, whatever the set).

    The time-ordered runs index the dated cohort; the others index the modelling table.
    """
    return _cohort(run.startswith("temporal")).loc[rows]


def build_run(run: str, feature_set: str):
    """The run_tabpfn.Run (train and test Datasets) behind a predictions file."""
    from run_tabpfn import DESIGNS, build_runs

    for design in DESIGNS:
        for candidate in build_runs(design, feature_set):
            if candidate.name == run:
                return candidate
    raise KeyError(f"no run named {run!r}")


def fit(run: str, feature_set: str = "race_aware"):
    """(fitted TabPFN, Run). One fit (~1-3 min) per run: reuse the model, never refit in a loop."""
    import time

    from pfn_model import build_model

    spec = build_run(run, feature_set)
    start = time.perf_counter()
    model = build_model().fit(spec.train.X, spec.train.y)
    print(f"  fitted {run} x {feature_set} in {time.perf_counter() - start:.0f}s", flush=True)
    return model, spec


def fmt(frame: pd.DataFrame, digits: int = 3) -> str:
    with pd.option_context("display.width", 200, "display.max_columns", None):
        return frame.round(digits).to_string(index=False)
