"""Project configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def project_root() -> Path:
    """Walk up from this file until we find the directory holding pyproject.toml."""
    for candidate in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate pyproject.toml above {__file__}")


@dataclass(frozen=True)
class Costs:
    """Pretrial decision costs in USD. Client-configurable assumptions, not constants."""

    c_fn: float
    c_fp: float
    detention_days: int
    detention_cost_per_day: float

    @property
    def ratio(self) -> float:
        """Cost of a missed re-offence relative to an unnecessary detention."""
        return self.c_fn / self.c_fp


@dataclass(frozen=True)
class Iterations:
    bootstrap_ci: int
    bootstrap_refits: int
    cv_repeats: int
    cv_folds: int
    n_seeds: int
    shap_background: int
    shap_explain: int
    shap_nsamples: int
    learning_curve_sizes: list[int]
    slow_models: list[str]
    slow_divisor: int

    def budget(self, count: int, model: str) -> int:
        """Scale an iteration count down for the models that cost ~400x more per fit."""
        if model in self.slow_models:
            return max(1, count // self.slow_divisor)
        return count


@dataclass(frozen=True)
class Splits:
    """Fractions for the two stability designs. See [tool.compas_scoring.splits]."""

    partition: tuple[float, float, float]
    temporal_windows: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Config:
    random_state: int
    test_size: float
    data_path: Path
    target: str
    incumbent: str
    expected_rows: int
    expected_base_rate: float
    dated_data_path: Path
    splits: Splits
    costs: Costs
    iterations: Iterations
    feature_sets: dict[str, list[str]] = field(default_factory=dict)
    race_dummies: list[str] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return project_root()

    def path(self, *parts: str) -> Path:
        """Resolve a path relative to the project root, creating parent dirs on write paths."""
        return self.root.joinpath(*parts)

    def features(self, feature_set: str = "race_aware") -> list[str]:
        try:
            return list(self.feature_sets[feature_set])
        except KeyError:
            raise KeyError(
                f"Unknown feature set {feature_set!r}; available: {sorted(self.feature_sets)}"
            ) from None


@lru_cache(maxsize=1)
def load_config() -> Config:
    root = project_root()
    with (root / "pyproject.toml").open("rb") as fh:
        table = tomllib.load(fh)["tool"]["compas_scoring"]

    return Config(
        random_state=table["random_state"],
        test_size=table["test_size"],
        data_path=root / table["data_path"],
        target=table["target"],
        incumbent=table["incumbent"],
        expected_rows=table["expected_rows"],
        expected_base_rate=table["expected_base_rate"],
        dated_data_path=root / table["dated_data_path"],
        splits=Splits(
            partition=tuple(table["splits"]["partition"]),
            temporal_windows=tuple(tuple(w) for w in table["splits"]["temporal_windows"]),
        ),
        costs=Costs(**table["costs"]),
        iterations=Iterations(**table["iterations"]),
        # Every key under [tool.compas_scoring.features] is a feature set, except the
        # race_dummies helper list. Adding a set is a TOML edit, not a code change.
        feature_sets={
            name: columns for name, columns in table["features"].items() if name != "race_dummies"
        },
        race_dummies=table["features"]["race_dummies"],
    )


CONFIG = load_config()

# The course brief names the feature sets FS1/FS2/FS3; the code uses descriptive keys, so the
# deck and the app can print the brief's labels without any module hard-coding a set name.
FEATURE_SET_LABELS = {
    "race_aware": "FS1",
    "selected": "FS2",
    "race_blind": "FS3",
}
