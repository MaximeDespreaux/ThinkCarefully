"""Loading, validation and splitting of the COMPAS two-year recidivism cohort."""

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
    return _assemble(load_raw(), feature_set)


def _assemble(df: pd.DataFrame, feature_set: str) -> Dataset:
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


# --------------------------------------------------------------------------- stability designs


# Strata smaller than this are pooled by outcome alone, so every stratum can be cut 3 ways.
MIN_STRATUM = 10


def protected_strata(df: pd.DataFrame) -> pd.Series:
    """Stratification key: outcome x race x sex x age band.

    Stratifying on the outcome alone left X3 3.7 points off the cohort's age mix (p = 0.012):
    age is a strong predictor, and X3 is the shared test set, so that is not a detail. Races
    too small to stratify (Asian, Native American, Other: ~6% together) are pooled, and any
    stratum under MIN_STRATUM rows falls back to the outcome alone.
    """
    groups = group_frame(df)
    race = groups["race"].where(
        groups["race"].isin(["African-American", "Caucasian", "Hispanic"]), "small groups"
    )
    key = (
        df[CONFIG.target].astype(str)
        + "|" + race + "|" + groups["sex"] + "|" + groups["age_band"]
    )  # fmt: skip
    rare = key.map(key.value_counts()) < MIN_STRATUM
    return key.where(~rare, df[CONFIG.target].astype(str) + "|pooled")


@lru_cache(maxsize=1)
def partition_index() -> tuple[pd.Index, pd.Index, pd.Index]:
    """X1 / X2 / X3: the stratified random partition for stability design 1.

    Each part keeps the cohort's base rate *and* its mix of race, sex and age band
    (see protected_strata), so each part is a representative sample of the cohort.
    Models are trained on X1 and on X2 separately and both are tested on X3, so any
    difference between the two fits comes from the training sample alone. X1 + X2 -> X3 is
    also a headline design in its own right. Like split_index, it depends on the row index
    only, so it is shared by every feature set.
    """
    df = load_raw()
    f1, f2, f3 = CONFIG.splits.partition
    if abs(f1 + f2 + f3 - 1.0) > 1e-9:
        raise ValueError(f"Partition fractions must sum to 1, got {CONFIG.splits.partition}")

    strata = protected_strata(df)
    rest_idx, x3_idx = train_test_split(
        df.index, test_size=f3, stratify=strata, random_state=CONFIG.random_state
    )
    x1_idx, x2_idx = train_test_split(
        rest_idx,
        test_size=f2 / (f1 + f2),
        stratify=strata.loc[rest_idx],
        random_state=CONFIG.random_state,
    )
    return pd.Index(sorted(x1_idx)), pd.Index(sorted(x2_idx)), pd.Index(sorted(x3_idx))


def partition(feature_set: str = "race_aware") -> dict[str, Dataset]:
    """The modelling table cut into {"X1", "X2", "X3"}."""
    data = build_dataset(feature_set)
    return {name: data.subset(idx) for name, idx in zip(("X1", "X2", "X3"), partition_index())}


# The raw export's race labels, mapped onto the modelling table's dummy columns.
_DATED_RACE_DUMMIES = {
    "African-American": "African_American",
    "Asian": "Asian",
    "Hispanic": "Hispanic",
    "Native American": "Native_American",
    "Other": "Other",
}
TWO_YEARS = 730  # days
# ProPublica's screening cutoff: the last date that still leaves two years of follow-up.
FOLLOW_UP_CUTOFF = pd.Timestamp("2014-04-01")


@lru_cache(maxsize=1)
def load_dated() -> pd.DataFrame:
    """A dated cohort in the modelling table's columns, sorted by COMPAS screening date.

    The modelling table has no dates and its rows cannot be linked back to the raw export,
    so this rebuilds a comparable two-year cohort from ``cox-violent-parsed.csv`` with
    ProPublica's filters. It is used only for the time-ordered stability design.

    Rules, and why:

    * One row per person, keyed on (name, dob, screening date). The file's ``id`` column
      numbers rows, not people: a person with several custody episodes has several rows,
      and 7,315 rows carry no id at all. Every person-level column is constant within a key.
    * ProPublica's filters: arrest within 30 days of screening, ``is_recid != -1``, a
      felony or misdemeanour charge degree (``(F…)``/``(M…)``; anything else dropped), a
      COMPAS score present.
    * Target: re-offence within two years of screening. People who did not re-offend are
      kept only if observed for more than two years -- otherwise a "0" may just be a
      censored "1".
    * Screened on or before ProPublica's cutoff, 2014-04-01. After it almost nobody has two
      years of follow-up, so the rule above would keep only the late re-offenders and the
      base rate would climb from ~33% to over 60% across the time windows -- a drift made
      by the data collection, not by the defendants.
    * ``score_factor`` is COMPAS decile > 4 (Medium or High), as in the modelling table.

    The result is comparable to, but not the same as, the modelling table: about 5,690
    people screened Jan 2013 - Mar 2014 at a ~33% base rate, against 6,172 at 45.5%. The
    export's recidivism fields evidently differ from those behind the modelling table.
    Report that difference with any result that uses this cohort; the time-ordered design
    compares a model with itself, so it is unaffected.
    """
    raw = pd.read_csv(CONFIG.dated_data_path, low_memory=False)
    key = ["name", "dob", "compas_screening_date"]
    raw["observed_days"] = raw.groupby(key, dropna=False)["end"].transform("max")
    people = raw.drop_duplicates(key).copy()

    screened = pd.to_datetime(people["compas_screening_date"], dayfirst=True)
    reoffended = pd.to_datetime(people["r_offense_date"], dayfirst=True, errors="coerce")
    days_to_reoffence = (reoffended - screened).dt.days
    target = ((people["is_recid"] == 1) & (days_to_reoffence <= TWO_YEARS)).astype(int)

    degree = people["c_charge_degree"].fillna("")
    keep = (
        people["days_b_screening_arrest"].between(-30, 30)
        & (people["is_recid"] != -1)
        & (degree.str.startswith("(F") | degree.str.startswith("(M"))
        & people["score_text"].notna()
        & ((target == 1) | (people["observed_days"] > TWO_YEARS))
        & (screened <= FOLLOW_UP_CUTOFF)
    )
    people, screened, target, degree = people[keep], screened[keep], target[keep], degree[keep]

    out = pd.DataFrame(
        {
            CONFIG.target: target,
            "Number_of_Priors": people["priors_count"].astype(int),
            CONFIG.incumbent: (people["decile_score"] > 4).astype(int),
            "Age_Above_FourtyFive": (people["age_cat"] == "Greater than 45").astype(int),
            "Age_Below_TwentyFive": (people["age_cat"] == "Less than 25").astype(int),
            **{
                column: (people["race"] == label).astype(int)
                for label, column in _DATED_RACE_DUMMIES.items()
            },
            "Female": (people["sex"] == "Female").astype(int),
            "Misdemeanor": degree.str.startswith("(M").astype(int),
            "screening_date": screened,
        }
    )
    # Stable sort, so people screened on the same day keep the export's order.
    out = out.sort_values("screening_date", kind="stable").reset_index(drop=True)

    if out.drop(columns="screening_date").isna().any().any():
        raise ValueError("Unexpected missing values in the dated cohort")
    missing = {CONFIG.target, CONFIG.incumbent, *CONFIG.features("race_aware")} - set(out)
    if missing:
        raise ValueError(f"Dated cohort is missing columns: {sorted(missing)}")
    return out


@dataclass(frozen=True)
class TemporalWindow:
    """One time-ordered train/test pair from the dated cohort."""

    label: str
    train: Dataset
    test: Dataset
    train_period: tuple[pd.Timestamp, pd.Timestamp]
    test_period: tuple[pd.Timestamp, pd.Timestamp]


def temporal_windows(feature_set: str = "race_aware") -> list[TemporalWindow]:
    """Stability design 2: train on the earliest defendants, test on the ones screened next.

    With the default windows: train on the first 40% and test on the next 20%, then train
    on the first 70% and test on the last 30%. Cuts are by row position in date order, so
    people screened on the day of a cut can fall on either side of it.
    """
    df = load_dated()
    data = _assemble(df, feature_set)
    dates = df["screening_date"]
    n = len(df)

    windows = []
    for train_end, test_end in CONFIG.splits.temporal_windows:
        cut, stop = round(train_end * n), round(test_end * n)
        train_idx, test_idx = df.index[:cut], df.index[cut:stop]
        windows.append(
            TemporalWindow(
                label=f"first {train_end:.0%} -> next {test_end - train_end:.0%}",
                train=data.subset(train_idx),
                test=data.subset(test_idx),
                train_period=(dates[train_idx[0]], dates[train_idx[-1]]),
                test_period=(dates[test_idx[0]], dates[test_idx[-1]]),
            )
        )
    return windows
