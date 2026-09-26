"""How much race each feature set leaks, race columns removed. Run via `make proxy-check`.

For every feature set in pyproject.toml: how well race can be predicted from its non-race
features (cross-validated AUC, African-American vs Caucasian; 0.5 = no race information).
The pairwise links behind it are in the EDA (association matrix, figure 09 for priors).
Writes artifacts/race_leakage.csv (git-ignored).
"""

from __future__ import annotations

# isort: split
import compas_scoring  # noqa: F401  (pins thread pools before numpy loads)

# isort: split
import sys

import pandas as pd

from compas_scoring.config import CONFIG, FEATURE_SET_LABELS
from compas_scoring.proxies import race_leakage


def main() -> int:
    leakage = pd.DataFrame(
        [
            {
                "feature_set": fs,
                "label": FEATURE_SET_LABELS.get(fs, ""),
                "n_features": len(CONFIG.features(fs)),
                "race_leakage_auc": race_leakage(fs),
            }
            for fs in CONFIG.feature_sets
        ]
    )
    out = CONFIG.path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    leakage.to_csv(out / "race_leakage.csv", index=False)
    print("How well race can be predicted from each feature set (CV AUC; 0.5 = none):\n")
    print(leakage.round(3).to_string(index=False))
    print(f"\n-> {out / 'race_leakage.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
