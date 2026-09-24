"""Loading, validation and splitting of the COMPAS two-year recidivism cohort.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from compas_scoring.config import CONFIG

# Race is stored as five dummies with Caucasian as the implicit reference category
# (all five zero). Age is two dummies with 25-45 as the reference.
RACE_REFERENCE = "Caucasian"
AGE_REFERENCE = "25 - 45"


@dataclass(frozen=True)
class Dataset:
    """A feature matrix, its target, the incumbent benchmark, and the group labels.

    ``groups`` carries the protected attributes as readable labels (race, sex, age band)
    for the fairness analysis. They are kept alongside -- never inside -- the feature
    matrix, so a race-blind model cannot accidentally be handed race.
    """

    X: pd.DataFrame
    y: pd.Series
    incumbent: pd.Series
    groups: pd.DataFrame
    feature_set: str

    def __len__(self) -> int:
        return len(self.y)

    @property
    def base_rate(self) -> float:
        return float(self.y.mean())

    def subset(self, index) -> Dataset:
        return Dataset(
            X=self.X.loc[index],
            y=self.y.loc[index],
            incumbent=self.incumbent.loc[index],
            groups=self.groups.loc[index],
            feature_set=self.feature_set,
        )


def _derive_race(df: pd.DataFrame) -> pd.Series:
    """Collapse the five race dummies back into one readable label."""
    race = pd.Series(RACE_REFERENCE, index=df.index, name="race", dtype=object)
    labels = {
        "African_American": "African-American",
        "Asian": "Asian",
        "Hispanic": "Hispanic",
        "Native_American": "Native American",
        "Other": "Other",
    }
    for column, label in labels.items():
        race[df[column] == 1] = label

    overlapping = df[list(labels)].sum(axis=1) > 1
    if overlapping.any():
        raise ValueError(f"{int(overlapping.sum())} rows carry more than one race dummy")
    return race


def _derive_age_band(df: pd.DataFrame) -> pd.Series:
    band = pd.Series(AGE_REFERENCE, index=df.index, name="age_band", dtype=object)
    band[df["Age_Below_TwentyFive"] == 1] = "Less than 25"
    band[df["Age_Above_FourtyFive"] == 1] = "Greater than 45"

    both = (df["Age_Below_TwentyFive"] == 1) & (df["Age_Above_FourtyFive"] == 1)
    if both.any():
        raise ValueError(f"{int(both.sum())} rows are both under 25 and over 45")
    return band


@lru_cache(maxsize=1)
def load_raw() -> pd.DataFrame:
    """Read the CSV and assert it is the cohort we think it is."""
    df = pd.read_csv(CONFIG.data_path)

    if len(df) != CONFIG.expected_rows:
        raise ValueError(f"Expected {CONFIG.expected_rows} rows, found {len(df)}")
    if df.isna().any().any():
        raise ValueError("Unexpected missing values in the modelling table")

    base_rate = df[CONFIG.target].mean()
    if abs(base_rate - CONFIG.expected_base_rate) > 5e-4:
        raise ValueError(
            f"Base rate {base_rate:.4f} does not match the expected "
            f"{CONFIG.expected_base_rate:.4f}: this is not the standard ProPublica cohort"
        )

    expected_columns = {CONFIG.target, CONFIG.incumbent, *CONFIG.features("race_aware")}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    return df


def group_frame(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Protected attributes as readable labels, for the fairness analysis."""
    df = load_raw() if df is None else df
    return pd.DataFrame(
        {
            "race": _derive_race(df),
            "sex": np.where(df["Female"] == 1, "Female", "Male"),
            "age_band": _derive_age_band(df),
            "charge_degree": np.where(df["Misdemeanor"] == 1, "Misdemeanor", "Felony"),
        },
        index=df.index,
    )


def build_dataset(feature_set: str = "race_aware") -> Dataset:
    """Assemble the feature matrix for one feature set, with the guardrails applied."""
    df = load_raw()
    features = CONFIG.features(feature_set)

    leaked = {CONFIG.incumbent, CONFIG.target} & set(features)
    if leaked:
        raise ValueError(f"Feature set {feature_set!r} leaks {sorted(leaked)}")

    return Dataset(
        X=df[features].astype(float).copy(),
        y=df[CONFIG.target].astype(int).copy(),
        incumbent=df[CONFIG.incumbent].astype(int).copy(),
        groups=group_frame(df),
        feature_set=feature_set,
    )


@lru_cache(maxsize=1)
def split_index() -> tuple[pd.Index, pd.Index]:
    """The one stratified train/test split, shared by every model and every metric.

    Derived from the row index alone so it is identical across feature sets -- the
    race-aware and race-blind models are evaluated on exactly the same defendants.
    """
    df = load_raw()
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=CONFIG.test_size,
        stratify=df[CONFIG.target],
        random_state=CONFIG.random_state,
    )
    return pd.Index(sorted(train_idx)), pd.Index(sorted(test_idx))


def train_test(feature_set: str = "race_aware") -> tuple[Dataset, Dataset]:
    data = build_dataset(feature_set)
    train_idx, test_idx = split_index()
    return data.subset(train_idx), data.subset(test_idx)


def engineered(X: pd.DataFrame) -> pd.DataFrame:
    """Extra terms for the white-box model.

    Priors is the only genuinely continuous predictor and its effect on recidivism is
    strongly concave, so a raw linear term underfits the low end and overstates the tail.
    The log term and the age interaction give logistic regression a fair shot at the
    non-linearity the tree model gets for free.
    """
    out = X.copy()
    priors = out["Number_of_Priors"]
    out["log_priors"] = np.log1p(priors)
    out["priors_capped"] = priors.clip(upper=10)
    out["no_priors"] = (priors == 0).astype(float)
    if "Age_Below_TwentyFive" in out:
        out["young_x_log_priors"] = out["Age_Below_TwentyFive"] * out["log_priors"]
    return out
