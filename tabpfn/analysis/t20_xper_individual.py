"""T20. XPER on AUC with the XPER package: global and per-defendant contributions, force plots.

Same method and settings as the XGBoost group (xgboost/performance.py), so the two models'
XPER values and force plots compare directly: the `XPER` package (Hue, Hurlin, Perignon &
Saurin), Kernel XPER with 300 sampled coalitions, seed 42, on a 400-defendant stratified
sample of the shared holdout test set. It returns phi (benchmark + one value per feature) and
phi_i (the same for each defendant): the force plot draws one defendant's phi_i.

**Why a lookup table.** The package queries the model defendant by defendant: 2 x 400 x 400
rows per coalition, ~96 M TabPFN predictions in all (~100 h). But TabPFN scores each test row
independently of the others (checked: identical in batches of 1, 40 or 600, in any order, to
2e-7), and the 10 features take few values (9 binary, priors 36 values): at most
2^9 x 36 = 18,432 distinct rows. So every possible row is scored once (~70 s) and the package
is answered from that table. The XPER values are the same as with direct calls; the table is
checked against direct predictions on the sample before use.

Refits TabPFN once (holdout x race_aware). Outputs: xper_auc.csv (phi), xper_auc_individual.csv
(phi_i, indexed by cohort row), lookup_check.json; figures in reports/figures/tabpfn/xper/.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import itertools
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

RUN, FEATURE_SET = "holdout", "race_aware"
N_SAMPLE, N_COALITIONS, SEED = 400, 300, 42  # as in xgboost/performance.py
FIGURES = c.TABPFN.parent / "reports" / "figures" / "tabpfn" / "xper"
# Defendants drawn as force plots: position 5 is the one the XGBoost notebook shows.
FORCE_PLOT_POSITIONS = [5]


class LookupModel:
    """Answers predict_proba from TabPFN's scores on every possible feature row."""

    def __init__(self, model, X: pd.DataFrame, values: dict[str, np.ndarray]):
        self.columns = list(X.columns)
        self.values = [np.asarray(values[col], dtype=float) for col in self.columns]
        self.radix = np.array([len(v) for v in self.values])
        grid = np.array(list(itertools.product(*self.values)), dtype=float)
        start = time.perf_counter()
        scores = model.predict_proba(pd.DataFrame(grid, columns=self.columns))[:, 1]
        self.seconds = time.perf_counter() - start
        self.scores = np.empty(len(grid))
        self.scores[self._keys(grid)] = scores
        self.n_rows = len(grid)

    def _keys(self, X: np.ndarray) -> np.ndarray:
        """Mixed-radix index of each row in the grid (raises if a value is off the grid)."""
        key = np.zeros(len(X), dtype=np.int64)
        for j, levels in enumerate(self.values):
            position = np.searchsorted(levels, X[:, j])
            position = np.clip(position, 0, len(levels) - 1)
            if not np.array_equal(levels[position], X[:, j]):
                raise ValueError(f"{self.columns[j]}: value outside the lookup grid")
            key = key * self.radix[j] + position
        return key

    def predict_proba(self, X) -> np.ndarray:
        X = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)
        p = self.scores[self._keys(X)]
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__)
    if c.cached(out / "xper_auc_individual.csv", args.force):
        # Figures are not committed: redraw them from the saved values, no refit.
        phi = pd.read_csv(out / "xper_auc.csv", index_col=0).squeeze("columns")
        phi_i = pd.read_csv(out / "xper_auc_individual.csv", index_col="row")
        X = c.cohort_rows(RUN, phi_i.index)[list(phi_i.columns[1:])].astype(float)
        draw(phi, phi_i, X)
        print(f"  figures redrawn -> {FIGURES}")
        return
    from XPER.compute.Performance import ModelPerformance

    model, spec = c.fit(RUN, FEATURE_SET)
    train, test = spec.train, spec.test
    index, _ = train_test_split(
        test.X.index, train_size=N_SAMPLE, stratify=test.y, random_state=SEED
    )
    sample = test.subset(index.sort_values())

    # Every value each feature takes in the train and test sets (the package only mixes
    # values between observed rows, so the grid covers every row it can ask for).
    both = pd.concat([train.X, test.X])
    values = {col: np.sort(both[col].unique()) for col in both.columns}
    lookup = LookupModel(model, test.X, values)
    direct = model.predict_proba(sample.X)[:, 1]
    check = {
        "grid_rows": lookup.n_rows,
        "grid_seconds": lookup.seconds,
        "max_abs_diff_vs_direct": float(
            np.abs(lookup.predict_proba(sample.X)[:, 1] - direct).max()
        ),
        "n_sample": len(sample),
        "n_coalitions": N_COALITIONS,
        "seed": SEED,
    }
    print(f"  lookup: {lookup.n_rows} rows in {lookup.seconds:.0f}s, max |diff| "
          f"{check['max_abs_diff_vs_direct']:.1e}", flush=True)  # fmt: skip
    if check["max_abs_diff_vs_direct"] > 1e-5:
        raise AssertionError("lookup table disagrees with direct predictions")

    start = time.perf_counter()
    xper = ModelPerformance(
        train.X.to_numpy(), train.y.to_numpy(), sample.X.to_numpy(), sample.y.to_numpy(),
        lookup, sample_size=len(sample),
    )  # fmt: skip
    phi, phi_i = xper.calculate_XPER_values(
        ["AUC"], kernel=True, seed=SEED, N_coalition_sampled=N_COALITIONS
    )
    check["xper_seconds"] = time.perf_counter() - start

    columns = ["benchmark", *sample.X.columns]
    phi = pd.Series(phi, index=columns, name="xper_auc")
    phi_i = pd.DataFrame(phi_i, index=sample.X.index.rename("row"), columns=columns)
    phi.to_csv(out / "xper_auc.csv")
    phi_i.to_csv(out / "xper_auc_individual.csv")
    check["auc_on_sample"] = float(roc_auc_score(sample.y, direct))
    check["benchmark_plus_contributions"] = float(phi.sum())
    (out / "lookup_check.json").write_text(json.dumps(check, indent=2))

    draw(phi, phi_i, sample.X)
    print(c.fmt(phi.rename("contribution").rename_axis("feature").reset_index(), 4))
    print(
        f"\nAUC on the sample {check['auc_on_sample']:.4f}; benchmark + contributions "
        f"{check['benchmark_plus_contributions']:.4f}; {check['xper_seconds']:.0f}s\n-> {out}"
        f"\n-> {FIGURES}"
    )


def draw(phi: pd.Series, phi_i: pd.DataFrame, X: pd.DataFrame) -> None:
    """The package's own bar plot and force plots, saved as PNG (make tabpfn-t20 redraws)."""
    from XPER.viz.Visualisation import visualizationClass as viz

    FIGURES.mkdir(parents=True, exist_ok=True)
    values = (phi.to_numpy(), phi_i.to_numpy())
    labels = list(X.columns)

    # The package's bar_plot is plotly and opens a browser; this is the same chart: each
    # feature's share of the total |contribution|, signed.
    contributions = phi.drop("benchmark")
    share = 100 * contributions / contributions.abs().sum()
    share = share.reindex(share.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(share.index, share.to_numpy(), color=np.where(share > 0, "#ff0051", "#008bfb"))
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_xlabel("Contribution to the AUC (% of total |contribution|)")
    ax.set_title(
        f"TabPFN, XPER on AUC: benchmark {phi['benchmark']:.3f}, "
        f"AUC {phi.sum():.3f} (holdout sample, n = {len(X)})",
        fontsize=10,
    )
    fig.savefig(FIGURES / "xper_auc_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    for position in FORCE_PLOT_POSITIONS:
        viz.force_plot(
            XPER_values=values, instance=position, X_test=X, variable_name=labels,
            figsize=(16, 4),
        )  # fmt: skip
        row = X.index[position]
        plt.gcf().savefig(
            FIGURES / f"xper_auc_force_{position}_row{row}.png", dpi=150, bbox_inches="tight"
        )
        plt.close("all")


if __name__ == "__main__":
    main()
