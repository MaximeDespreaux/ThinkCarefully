from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import compas_scoring  # noqa: F401  (imported for its thread-pool pinning)
from compas_scoring.config import CONFIG
from compas_scoring.data import build_dataset, group_frame

ART = CONFIG.path("artifacts")


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramer's V between two categorical series (Bergsma 2013)."""
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")

    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape

    phi2_corrected = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    k_corrected = k - (k - 1) ** 2 / (n - 1)
    denominator = min(k_corrected - 1, r_corrected - 1)
    if denominator <= 0:
        return float("nan")
    return float(np.sqrt(phi2_corrected / denominator))


def association_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(frame.columns)
    out = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for i, a in enumerate(columns):
        for b in columns[i:]:
            v = 1.0 if a == b else cramers_v(frame[a], frame[b])
            out.loc[a, b] = out.loc[b, a] = v
    return out


def main() -> int:
    data = build_dataset("race_aware")
    groups = group_frame()
    ART.mkdir(parents=True, exist_ok=True)

    print("WS1 exploratory analysis")
    print("-" * 74)

    # --- outcome rates per protected attribute -------------------------------------
    rows = []
    for attribute in groups.columns:
        for level, chunk in data.y.groupby(groups[attribute]):
            rows.append(
                {
                    "attribute": attribute,
                    "group": level,
                    "n": int(len(chunk)),
                    "share": float(len(chunk) / len(data)),
                    "recidivism_rate": float(chunk.mean()),
                }
            )
    outcomes = pd.DataFrame(rows)
    outcomes.to_csv(ART / "eda_group_rates.csv", index=False)
    print(f"  group rates       -> eda_group_rates.csv ({len(outcomes)} rows)")

    # --- univariate summary ---------------------------------------------------------
    summary = data.X.describe().T
    summary["prevalence"] = data.X.mean()
    summary["recid_rate_when_1"] = [
        float(data.y[data.X[c] == 1].mean()) if set(data.X[c].unique()) <= {0.0, 1.0} else np.nan
        for c in data.X.columns
    ]
    summary.reset_index(names="feature").to_csv(ART / "eda_univariate.csv", index=False)
    print(f"  univariate        -> eda_univariate.csv ({len(summary)} features)")

    # --- association ----------------------------------------------------------------
    discrete = data.X.copy()
    discrete["Number_of_Priors"] = pd.cut(
        discrete["Number_of_Priors"],
        [-1, 0, 1, 3, 6, 100],
        labels=["0", "1", "2-3", "4-6", "7+"],
    )
    associations = association_matrix(discrete)
    associations.reset_index(names="feature").to_csv(ART / "eda_association.csv", index=False)

    upper = associations.to_numpy()[np.triu_indices(len(associations), k=1)]
    strongest = (
        associations.where(np.triu(np.ones(associations.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    print(
        f"  Cramer's V        -> eda_association.csv "
        f"(median {np.nanmedian(upper):.3f}, max {np.nanmax(upper):.3f})"
    )
    for (a, b), v in strongest.head(3).items():
        print(f"      strongest: {a} <-> {b}  V={v:.3f}")

    # --- PCA -------------------------------------------------------------------------
    scaled = StandardScaler().fit_transform(data.X)
    pca = PCA().fit(scaled)
    explained = pd.DataFrame(
        {
            "component": np.arange(1, pca.n_components_ + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained.to_csv(ART / "eda_pca.csv", index=False)

    n_for_90 = int(np.searchsorted(explained["cumulative"], 0.90) + 1)
    first = float(pca.explained_variance_ratio_[0])
    print("  PCA               -> eda_pca.csv")
    print(
        f"      PC1 explains {first:.1%}; {n_for_90} of {data.X.shape[1]} components "
        f"needed for 90% variance"
    )

    verdict = (
        "No meaningful dimensionality reduction is available: the predictors are "
        f"near-orthogonal indicators, PC1 carries only {first:.1%} of the variance, and "
        f"{n_for_90} of {data.X.shape[1]} components are still required to reach 90%. "
        "Modelling therefore proceeds on the original features; a PCA rotation would cost "
        "all interpretability and buy no compression."
    )
    (ART / "eda_summary.json").write_text(
        json.dumps(
            {
                "n_rows": int(len(data)),
                "n_features": int(data.X.shape[1]),
                "base_rate": float(data.base_rate),
                "median_association": float(np.nanmedian(upper)),
                "max_association": float(np.nanmax(upper)),
                "pc1_explained": first,
                "components_for_90pct": n_for_90,
                "pca_verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"\n  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
