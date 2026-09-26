"""Which features stand in for race, and how much race each feature set leaks. `make proxy-check`.

Dropping the race columns does not make a model race-blind if the remaining features encode
race. This prints, on the training split:

1. the bias-corrected Cramér's V between each race dummy and every other feature, and the
   features that count as race proxies (V >= 0.1 with any race dummy);
2. the share of African-American defendants by number of priors, the dominant proxy;
3. for every feature set in pyproject.toml, how well race can be predicted from it
   (cross-validated AUC, African-American vs Caucasian; 0.5 = no race information).

Writes artifacts/proxy_associations.csv and artifacts/race_leakage.csv (git-ignored).
"""

from __future__ import annotations

# isort: split
import compas_scoring  # noqa: F401  (pins thread pools before numpy loads)

# isort: split
import sys

import pandas as pd

from compas_scoring.config import CONFIG, FEATURE_SET_LABELS
from compas_scoring.data import build_dataset, split_index
from compas_scoring.proxies import (
    PROXY_THRESHOLD,
    race_associations,
    race_leakage,
    race_proxies,
)


def main() -> int:
    out = CONFIG.path("artifacts")
    out.mkdir(parents=True, exist_ok=True)

    associations = race_associations()
    associations.to_csv(out / "proxy_associations.csv", index_label="race_dummy")
    print("Cramér's V, race dummy x other feature (training split):\n")
    print(associations.round(3).to_string())
    print(f"\nRace proxies (V >= {PROXY_THRESHOLD} with any race dummy): {race_proxies()}")

    train_idx, _ = split_index()
    data = build_dataset("race_aware").subset(train_idx)
    race = data.groups["race"]
    keep = race.isin(["African-American", "Caucasian"])
    band = pd.cut(
        data.X.loc[keep, "Number_of_Priors"],
        [-1, 0, 1, 3, 6, 10, 100],
        labels=["0", "1", "2-3", "4-6", "7-10", "11+"],
    )
    share = (race[keep] == "African-American").groupby(band, observed=True).agg(["size", "mean"])
    share.columns = ["defendants", "share African-American"]
    print("\nAfrican-American share among African-American + Caucasian, by priors:\n")
    print(share.round(3).T.to_string())

    rows = [
        {
            "feature_set": fs,
            "label": FEATURE_SET_LABELS.get(fs, ""),
            "n_features": len(CONFIG.features(fs)),
            "race_leakage_auc": race_leakage(fs),
        }
        for fs in CONFIG.feature_sets
    ]
    leakage = pd.DataFrame(rows)
    leakage.to_csv(out / "race_leakage.csv", index=False)
    print("\nHow well race can be predicted from each feature set (CV AUC; 0.5 = none):\n")
    print(leakage.round(3).to_string(index=False))
    print(f"\n-> {out / 'proxy_associations.csv'}\n-> {out / 'race_leakage.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
