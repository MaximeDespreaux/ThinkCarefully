"""Figures for the logistic regression. Run via `uv run python -m logreg.plots`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter  # noqa: E402

from compas_scoring import fairness as shared_fairness  # noqa: E402
from compas_scoring.config import CONFIG  # noqa: E402
from logreg.analysis import FEATURE_SETS, PRIMARY, TOST_DELTA  # noqa: E402

ART = CONFIG.path("artifacts", "logreg")
FIG = CONFIG.path("reports", "figures", "logreg")

# --- design tokens (shared with scripts/plot_eda.py) ---------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#a3a29b"  # the incumbent, de-emphasised series and reference lines
# Categorical slots in fixed order: slot 1 = race-aware, slot 2 = race-blind.
COLORS = {"race_aware": "#2a78d6", "race_blind": "#eb6834"}
LABELS = {"race_aware": "Race-aware (FS1)", "race_blind": "Race-blind (FS3)"}
# Diverging poles for signed contributions: raises risk (red) vs lowers it (blue).
RAISES, LOWERS = "#d03b3b", "#2a78d6"
PCT = PercentFormatter(xmax=1.0, decimals=0)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "figure.dpi": 150,
    }
)


# --- helpers ---------------------------------------------------------------------------


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(ART / name)


def color(feature_set: str) -> str:
    return COLORS.get(feature_set, NEUTRAL)


def label(feature_set: str) -> str:
    return LABELS.get(feature_set, feature_set)


def dollars(value, _position=None) -> str:
    """Axis money format: -$350, not $-350."""
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def style(ax, *, grid: str = "y") -> None:
    """Recessive chrome: no top/right spines, grid on the value axis only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="both", visible=False)
    ax.grid(axis=grid, visible=True)
    ax.tick_params(length=0)


def title(ax, text: str, subtitle: str | None = None) -> None:
    ax.set_title(text, loc="left", fontsize=11.5, fontweight="bold", pad=24 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")


def log_axis(ax, values) -> None:
    """Log-scaled x axis with plain-number ticks (1, 2, 4 -- not 2x10^0) that never collide."""
    low, high = np.nanmin(values), np.nanmax(values)
    candidates = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20]
    ticks = [t for t in candidates if low / 1.5 <= t <= high * 1.5] or [1]
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])


def figure_legend(fig, ax) -> None:
    """One legend for a row of panels, above them, so it can never sit on a data point."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(1, 1.04))


def dot_whisker(ax, y, estimate, lower, upper, colour, name=None) -> None:
    """A point estimate with its interval: 2px whisker, 8px dot with a surface ring."""
    ax.hlines(y, lower, upper, color=colour, linewidth=2, zorder=2)
    ax.scatter(
        estimate, y, s=64, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3, label=name
    )


def save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> reports/figures/logreg/{name}")
    return path


# --- performance -----------------------------------------------------------------------


def plot_auc_intervals() -> Path:
    table = read("performance.csv").query("model == 'logistic'").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 2.4))
    for y, row in table.iterrows():
        fs = row["feature_set"]
        dot_whisker(ax, y, row["auc"], row["auc_lower"], row["auc_upper"], color(fs))
        ax.text(row["auc_upper"] + 0.002, y, f"{row['auc']:.3f}", va="center", color=INK_2)
    ax.set_yticks(range(len(table)), [label(fs) for fs in table["feature_set"]])
    ax.set_ylim(-0.7, len(table) - 0.3)
    ax.set_xlabel("Holdout AUC (95% bootstrap interval)")
    style(ax, grid="x")
    title(ax, "Removing race costs no ranking power", "overlapping intervals; see DeLong test")
    return save(fig, "auc_intervals.png")


def plot_cost_curves() -> Path:
    curves = read("cost_curves.csv")
    performance = read("performance.csv")
    floors = json.loads((ART / "cost_baselines.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for fs, block in curves.groupby("feature_set", sort=False):
        ax.plot(block["threshold"], block["cost_per_capita"], color=color(fs), label=label(fs))
        best = performance.query("feature_set == @fs").iloc[0]
        ax.scatter(
            best["cost_optimal_threshold"],
            best["cost_per_capita"],
            s=64,
            color=color(fs),
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
    ax.axhline(floors["best_trivial"], color=NEUTRAL, linestyle="--", linewidth=1)
    ax.text(
        0.99,
        floors["best_trivial"],
        " best trivial policy",
        color=MUTED,
        va="bottom",
        ha="right",
        transform=ax.get_yaxis_transform(),
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost per defendant ($)")
    ax.legend(loc="upper left")
    style(ax)
    title(ax, "Cost per defendant across thresholds", "dot = cost-optimal operating point")
    return save(fig, "cost_curves.png")


# --- white-box interpretation ----------------------------------------------------------


def _interval_panel(table, estimate, lower, upper, *, log=False, reference=0.0):
    table = table.sort_values(estimate).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(table) + 1.4))
    for y, row in table.iterrows():
        significant = row[lower] > reference or row[upper] < reference
        colour = COLORS[PRIMARY] if significant else NEUTRAL
        dot_whisker(ax, y, row[estimate], row[lower], row[upper], colour)
    ax.axvline(reference, color=AXIS, linewidth=1, zorder=1)
    ax.set_yticks(range(len(table)), table["feature"])
    ax.set_ylim(-0.7, len(table) - 0.3)
    if log:
        log_axis(ax, [*table[lower], *table[upper]])
    style(ax, grid="x")
    return fig, ax


def plot_odds_ratios() -> Path:
    table = read("odds_ratios.csv").query("feature != 'const'")
    fig, ax = _interval_panel(table, "odds_ratio", "or_lower", "or_upper", log=True, reference=1)
    ax.set_xlabel("Odds ratio (95% CI, log scale)  -  grey: interval crosses 1")
    title(ax, "Odds ratios of the fitted specification", "race-aware model, engineered design")
    return save(fig, "odds_ratios.png")


def plot_marginal_effects() -> Path:
    table = read("marginal_effects.csv")
    fig, ax = _interval_panel(table, "marginal_effect", "ci_lower", "ci_upper")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Change in predicted re-arrest probability per unit (95% CI)")
    title(ax, "Average marginal effects", "grey: not significant at 5%")
    return save(fig, "marginal_effects.png")


def plot_lasso_path(highlight: int = 3) -> Path:
    path = read("lasso_path.csv")
    features = [c for c in path.columns if c not in ("C", "n_selected")]
    # The features that enter first are the load-bearing ones: colour those, grey the rest.
    entry = {f: path.loc[path[f].abs() > 1e-6, "C"].min() for f in features}
    order = sorted(features, key=lambda f: (np.isnan(entry[f]), entry[f]))
    slots = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for rank, feature in enumerate(reversed(order)):
        position = len(order) - 1 - rank
        emphasised = position < highlight
        ax.plot(
            path["C"],
            path[feature],
            color=slots[position] if emphasised else GRID,
            linewidth=2 if emphasised else 1.2,
            label=feature if emphasised else None,
            zorder=3 if emphasised else 2,
        )
    ax.set_xscale("log")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("C (inverse L1 strength, log scale)  -  shrinkage relaxes to the right")
    ax.set_ylabel("Standardised coefficient")
    handles, names = ax.get_legend_handles_labels()
    ranked = sorted(zip(handles, names, strict=True), key=lambda pair: order.index(pair[1]))
    ax.legend(*zip(*ranked, strict=True), title="First to enter", loc="upper left")
    style(ax)
    title(ax, "L1 path: the order features survive shrinkage", "race-aware, raw features")
    return save(fig, "lasso_path.png")


def plot_pdp_ice() -> Path:
    ice = read("ice_curves.csv").query("kind == 'raw'")
    pdp = read("pdp_Number_of_Priors.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for _, block in ice.groupby("row"):
        ax.plot(block["Number_of_Priors"], block["predicted_risk"], color=GRID, linewidth=0.8)
    ax.plot(
        pdp["Number_of_Priors"],
        pdp["predicted_risk"],
        color=COLORS[PRIMARY],
        label="Average (partial dependence)",
    )
    ax.plot([], [], color=GRID, linewidth=0.8, label="Individual defendants (ICE)")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Number of priors")
    ax.set_ylabel("Predicted re-arrest probability")
    ax.legend(loc="lower right")
    style(ax)
    title(
        ax,
        "Risk rises with priors, concavely",
        "race-aware model; the average spans the central 90% of observed priors",
    )
    return save(fig, "pdp_ice_priors.png")


def plot_local_contributions() -> Path:
    table = read("local_contributions.csv")
    archetypes = list(dict.fromkeys(table["archetype"]))
    fig, axes = plt.subplots(1, len(archetypes), figsize=(4.2 * len(archetypes), 4), sharey=True)
    terms = list(dict.fromkeys(table["term"]))
    for ax, archetype in zip(np.atleast_1d(axes), archetypes, strict=True):
        block = table[table["archetype"] == archetype].set_index("term").loc[terms]
        values = block["contribution"].to_numpy()
        ax.barh(
            range(len(terms)),
            values,
            height=0.6,
            color=[RAISES if v > 0 else LOWERS for v in values],
        )
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(range(len(terms)), terms)
        ax.invert_yaxis()
        style(ax, grid="x")
        title(ax, archetype)
    np.atleast_1d(axes)[0].set_xlabel("Log-odds contribution vs the average defendant")
    fig.suptitle(
        "Exact local explanation: red raises risk, blue lowers it",
        x=0.01,
        ha="left",
        y=1.04,
        fontsize=12,
        fontweight="bold",
    )
    return save(fig, "local_contributions.png")


# --- stability -------------------------------------------------------------------------


def plot_learning_curve() -> Path:
    curve = read("learning_curves.csv").groupby("n_train")["auc"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colour = COLORS[PRIMARY]
    ax.fill_between(
        curve.index,
        curve["mean"] - curve["std"],
        curve["mean"] + curve["std"],
        color=colour,
        alpha=0.15,
        linewidth=0,
    )
    ax.plot(
        curve.index,
        curve["mean"],
        color=colour,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(curve.index))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(curve.index, [f"{int(n):,}" for n in curve.index])
    ax.set_xlabel("Training rows (log scale)")
    ax.set_ylabel("Holdout AUC (mean +- 1 sd)")
    style(ax)
    title(
        ax,
        "Learning curve",
        "gains level off at the full training size: more rows would add little",
    )
    return save(fig, "learning_curve.png")


# --- fairness --------------------------------------------------------------------------

COMPARISON_TITLES = {
    "aa_vs_others": "African-American vs all other groups",
    "aa_vs_caucasian": "African-American vs Caucasian",
}
METRIC_LABELS = {
    "selection_rate": "Flag rate\n(statistical parity)",
    "fpr": "False-positive rate\n(equalized odds)",
    "fnr": "False-negative rate\n(equalized odds)",
}


def _by_comparison(nrows_per_panel: int, height: float = 0.55):
    comparisons = list(COMPARISON_TITLES)
    fig, axes = plt.subplots(
        1, len(comparisons), figsize=(12, height * nrows_per_panel + 1.8), sharey=True
    )
    return fig, dict(zip(comparisons, np.atleast_1d(axes), strict=True))


def plot_fairness_ztests() -> Path:
    table = read("fairness_parity_ztests.csv")
    metrics = list(METRIC_LABELS)
    fig, axes = _by_comparison(len(metrics) * 1.4)
    offsets = dict(zip(FEATURE_SETS, np.linspace(-0.15, 0.15, len(FEATURE_SETS)), strict=True))
    for comparison, ax in axes.items():
        block = table[table["comparison"] == comparison]
        for fs in FEATURE_SETS:
            rows = block[block["feature_set"] == fs].set_index("metric").loc[metrics]
            for y, (_, row) in enumerate(rows.iterrows()):
                dot_whisker(
                    ax,
                    y + offsets[fs],
                    row["difference"],
                    row["ci_lower"],
                    row["ci_upper"],
                    color(fs),
                    label(fs) if y == 0 else None,
                )
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(range(len(metrics)), [METRIC_LABELS[m] for m in metrics])
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlabel("Gap: African-American minus reference (95% CI, z-test)")
        style(ax, grid="x")
        title(ax, COMPARISON_TITLES[comparison])
    figure_legend(fig, next(iter(axes.values())))
    fig.suptitle(
        "Parity and equalized-odds gaps at each model's cost-optimal threshold",
        x=0.01,
        ha="left",
        y=1.03,
        fontsize=12,
        fontweight="bold",
    )
    return save(fig, "fairness_ztests.png")


def plot_mh_odds_ratios() -> Path:
    table = read("fairness_conditional_parity.csv")
    stratifiers = list(dict.fromkeys(table["stratifier"]))
    fig, axes = _by_comparison(len(stratifiers))
    offsets = dict(zip(FEATURE_SETS, np.linspace(-0.15, 0.15, len(FEATURE_SETS)), strict=True))
    for comparison, ax in axes.items():
        block = table[table["comparison"] == comparison]
        for fs in FEATURE_SETS:
            rows = block[block["feature_set"] == fs].set_index("stratifier").loc[stratifiers]
            for y, (_, row) in enumerate(rows.iterrows()):
                dot_whisker(
                    ax,
                    y + offsets[fs],
                    row["mh_odds_ratio"],
                    row["mh_or_lower"],
                    row["mh_or_upper"],
                    color(fs),
                    label(fs) if y == 0 else None,
                )
        ax.axvline(1, color=AXIS, linewidth=1)
        log_axis(ax, [1, *table["mh_or_lower"], *table["mh_or_upper"]])
        ax.set_yticks(
            range(len(stratifiers)), [f"at equal {s.replace('_', ' ')}" for s in stratifiers]
        )
        ax.invert_yaxis()
        ax.set_xlabel("Mantel-Haenszel odds ratio of being flagged (95% CI, log scale)")
        style(ax, grid="x")
        title(ax, COMPARISON_TITLES[comparison])
    figure_legend(fig, next(iter(axes.values())))
    fig.suptitle(
        "Conditional statistical parity: the disparity survives every risk factor",
        x=0.01,
        ha="left",
        y=1.03,
        fontsize=12,
        fontweight="bold",
    )
    return save(fig, "mh_odds_ratios.png")


def plot_tost() -> Path:
    table = read("fairness_equivalence.csv")
    metrics = list(dict.fromkeys(table["metric"]))
    fig, axes = _by_comparison(len(metrics) * 1.4)
    offsets = dict(zip(FEATURE_SETS, np.linspace(-0.15, 0.15, len(FEATURE_SETS)), strict=True))
    for comparison, ax in axes.items():
        ax.axvspan(-TOST_DELTA, TOST_DELTA, color=GRID, alpha=0.6, linewidth=0, zorder=0)
        block = table[table["comparison"] == comparison]
        for fs in FEATURE_SETS:
            rows = block[block["feature_set"] == fs].set_index("metric").loc[metrics]
            for y, (_, row) in enumerate(rows.iterrows()):
                dot_whisker(
                    ax,
                    y + offsets[fs],
                    row["estimate"],
                    row["tost_lower"],
                    row["tost_upper"],
                    color(fs),
                    label(fs) if y == 0 else None,
                )
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(range(len(metrics)), [m.replace("_", " ") for m in metrics])
        ax.invert_yaxis()
        ax.set_xlabel(f"Disparity, 90% interval  -  shaded: equivalence zone +-{TOST_DELTA}")
        style(ax, grid="x")
        title(ax, COMPARISON_TITLES[comparison])
    figure_legend(fig, next(iter(axes.values())))
    fig.suptitle(
        "TOST: certified fair only if the interval sits inside the shaded zone",
        x=0.01,
        ha="left",
        y=1.03,
        fontsize=12,
        fontweight="bold",
    )
    return save(fig, "tost_equivalence.png")


def plot_fpdp(feature: str = "Number_of_Priors", comparison: str = "aa_vs_others") -> Path:
    curves = read("fpdp_curves.csv").query("feature == @feature and comparison == @comparison")
    floor = 1e-16  # p-values that underflow to 0 are drawn on the floor, not dropped
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for fs, block in curves.groupby("feature_set", sort=False):
        p_values = block["p_value"].clip(lower=floor)
        ax.plot(block["level"], p_values, color=color(fs), label=label(fs), zorder=2)
        # Hollow = the level detains (almost) everyone: parity there is bought by uselessness.
        degenerate = block["selection_rate"] >= shared_fairness.MITIGATION_DEGENERATE_RATE
        for hollow, marks in ((False, ~degenerate), (True, degenerate)):
            ax.scatter(
                block.loc[marks, "level"],
                p_values[marks],
                s=56,
                color=SURFACE if hollow else color(fs),
                edgecolor=color(fs),
                linewidth=2,
                zorder=3,
            )
    ax.axhline(0.05, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.text(
        0.01, 0.05, " alpha = 0.05", color=MUTED, va="bottom", transform=ax.get_yaxis_transform()
    )
    ax.set_yscale("log")
    ax.set_ylim(floor / 3, 3)
    ax.set_xlabel(f"{feature} imposed on every defendant")
    ax.set_ylabel("Statistical parity p-value (log)")
    ax.legend(loc="lower right")
    style(ax)
    title(
        ax,
        "Fairness PDP",
        f"{COMPARISON_TITLES[comparison]}; hollow = that level detains 95%+ of defendants",
    )
    return save(fig, "fpdp_priors.png")


def plot_fairness_frontier(comparison: str = "aa_vs_others") -> Path:
    frontier = read("fairness_frontier.csv").query("comparison == @comparison")
    # The operating point comes from the exact results, not the nearest threshold on the grid.
    chosen = read("fairness_disparities.csv").query("comparison == @comparison")
    costs = read("performance.csv").set_index("feature_set")["cost_per_capita"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for fs, block in frontier.groupby("feature_set", sort=False):
        ax.scatter(
            block["cost_per_capita"],
            block["fpr_gap"],
            s=14,
            color=color(fs),
            alpha=0.55,
            linewidth=0,
            label=f"{label(fs)}, one point per threshold",
        )
        point = chosen[chosen["feature_set"] == fs].iloc[0]
        ax.scatter(
            costs[fs],
            point["fpr_difference"],
            s=90,
            color=color(fs),
            edgecolor=INK,
            linewidth=1.5,
            zorder=3,
        )
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.xaxis.set_major_formatter(dollars)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Expected cost per defendant")
    ax.set_ylabel("FPR gap (African-American minus reference)")
    ax.legend(loc="upper right")
    style(ax)
    gaps = ", ".join(
        f"{label(fs).split(' (')[0].lower()} {row['fpr_difference']:+.0%}"
        for fs, row in chosen.set_index("feature_set").iterrows()
    )
    title(
        ax,
        "The price of error-rate parity",
        f"{COMPARISON_TITLES[comparison]}; ringed dot = cost-optimal point (FPR gap: {gaps})",
    )
    return save(fig, "fairness_frontier.png")


def plot_calibration_by_group(groups=("African-American", "Caucasian")) -> Path:
    table = read("calibration_by_group.csv").query("feature_set == @PRIMARY")
    styles = {groups[0]: "#2a78d6", groups[1]: "#eb6834"}
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--", label="Perfect calibration")
    for group in groups:
        block = table[table["g"] == group]
        ax.plot(
            block["predicted"],
            block["observed"],
            color=styles[group],
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=group,
        )
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted re-arrest probability")
    ax.set_ylabel("Observed re-arrest rate")
    ax.legend(loc="upper left")
    style(ax, grid="both")
    title(ax, "Calibration by race", "race-aware model; equal scores, equal risk?")
    return save(fig, "calibration_by_group.png")


# --- XPER ------------------------------------------------------------------------------


def plot_xper() -> Path:
    table = read("xper_auc.csv").query("feature != 'benchmark'")
    slots = int(table.groupby("feature_set").size().max())
    fig, axes = plt.subplots(
        1, len(FEATURE_SETS), figsize=(13, 0.4 * slots + 1.6), layout="constrained"
    )
    for ax, fs in zip(np.atleast_1d(axes), FEATURE_SETS, strict=True):
        block = table[table["feature_set"] == fs].sort_values("contribution")
        # Same band count in every panel, so bars have the same thickness side by side.
        positions = np.arange(len(block)) + (slots - len(block))
        ax.barh(positions, block["contribution"], height=0.6, color=color(fs))
        ax.set_yticks(positions, block["feature"])
        ax.set_ylim(-0.6, slots - 0.4)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_xlabel("Contribution to AUC")
        style(ax, grid="x")
        title(ax, label(fs))
    fig.suptitle(
        "XPER: which features earn the AUC", x=0.01, ha="left", fontsize=12, fontweight="bold"
    )
    return save(fig, "xper_auc.png")


# --- more white-box, performance and stability views -----------------------------------


def plot_scorecard() -> Path:
    card = read("scorecard.csv").sort_values("points_per_unit").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(card) + 1.4))
    colours = [RAISES if c > 0 else LOWERS for c in card["coefficient"]]
    ax.barh(range(len(card)), card["points_per_unit"], height=0.6, color=colours)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(range(len(card)), card["feature"])
    ax.set_xlabel("Points per unit (20 points halve the odds of re-arrest)")
    style(ax, grid="x")
    title(ax, "Points scorecard", "red: raises risk (costs points); blue: lowers it")
    return save(fig, "scorecard.png")


def plot_stepwise() -> Path:
    log = read("experiment_log.csv")
    directions = ["forward", "backward"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True, layout="constrained")
    for ax, direction in zip(axes, directions, strict=True):
        block = log[log["direction"] == direction].reset_index(drop=True)
        colour = COLORS[PRIMARY]
        ax.fill_between(
            block.index,
            block["cv_auc"] - block["cv_sd"],
            block["cv_auc"] + block["cv_sd"],
            color=colour,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(block.index, block["cv_auc"], color=colour, zorder=2)
        accepted = block["accepted"].astype(bool)
        for hollow, marks in ((False, accepted), (True, ~accepted)):
            ax.scatter(
                block.index[marks],
                block.loc[marks, "cv_auc"],
                s=56,
                zorder=3,
                color=SURFACE if hollow else colour,
                edgecolor=colour,
                linewidth=2,
            )
        action = "+" if direction == "forward" else "-"
        names = ["start"] + [f"{action} {c}" for c in block["candidate"].iloc[1:]]
        ax.set_xticks(block.index, names, rotation=35, ha="right")
        style(ax)
        title(ax, f"{direction.capitalize()} selection")
    axes[0].set_ylabel("Train-only CV AUC (+- 1 sd)")
    fig.suptitle(
        "Stepwise experiment log: filled = accepted, hollow = rejected (search stops)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return save(fig, "stepwise_log.png")


def plot_importance_comparison() -> Path:
    methods = {
        "Permutation (AUC drop)": read("permutation_importance.csv").set_index("feature")[
            "importance"
        ],
        "KernelSHAP (mean |value|)": read("shap_importance.csv").set_index("feature")[
            "mean_abs_shap"
        ],
        "LIME (mean |weight|)": read("lime_weights.csv").drop(columns="row").abs().mean(),
    }
    order = methods["Permutation (AUC drop)"].sort_values().index
    fig, axes = plt.subplots(
        1, len(methods), figsize=(13, 0.34 * len(order) + 1.8), sharey=True, layout="constrained"
    )
    for ax, (name, values) in zip(axes, methods.items(), strict=True):
        shares = (values / values.abs().sum()).reindex(order)
        ax.barh(range(len(order)), shares, height=0.6, color=COLORS[PRIMARY])
        ax.set_yticks(range(len(order)), order)
        ax.xaxis.set_major_formatter(PCT)
        ax.set_xlabel("Share of total importance")
        style(ax, grid="x")
        title(ax, name)
    fig.suptitle(
        "Three explanation methods, one model: do they agree on the drivers?",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return save(fig, "importance_comparison.png")


def plot_cost_sensitivity() -> Path:
    table = read("cost_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for fs, block in table.groupby("feature_set", sort=False):
        ax.plot(
            block["cost_ratio"],
            block["optimal_threshold"],
            color=color(fs),
            label=label(fs),
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
    configured = CONFIG.costs.c_fn / CONFIG.costs.c_fp
    ax.axvline(configured, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.text(
        configured,
        0.98,
        f" client's ratio {configured:.1f}",
        color=MUTED,
        va="top",
        transform=ax.get_xaxis_transform(),
    )
    ax.set_xlabel("Cost of a missed re-offence / cost of an unneeded detention")
    ax.set_ylabel("Cost-optimal threshold")
    ax.legend(loc="upper right")
    style(ax)
    title(ax, "The operating point is a client decision", "cost-optimal threshold per cost ratio")
    return save(fig, "cost_sensitivity.png")


def plot_subgroup_performance(min_n: int = 30) -> Path:
    table = read("subgroup_performance.csv").query("n >= @min_n").reset_index(drop=True)
    overall = read("performance.csv").query("feature_set == @PRIMARY")["auc"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(table) + 1.4))
    ax.scatter(
        table["auc"],
        range(len(table)),
        s=64,
        color=COLORS[PRIMARY],
        edgecolor=SURFACE,
        linewidth=2,
        zorder=3,
    )
    ax.axvline(overall, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.text(
        overall,
        0.01,
        f" overall AUC {overall:.3f}",
        color=MUTED,
        va="bottom",
        transform=ax.get_xaxis_transform(),
    )
    ax.set_yticks(
        range(len(table)),
        [
            f"{a}: {g} (n={n})"
            for a, g, n in zip(table["attribute"], table["group"], table["n"], strict=True)
        ],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Holdout AUC within the subgroup")
    style(ax, grid="x")
    title(
        ax,
        "Does the model rank equally well for everyone?",
        f"race-aware model; groups under {min_n} defendants omitted",
    )
    return save(fig, "subgroup_performance.png")


def plot_stability_frontier() -> Path:
    frontier = read("stability_frontier.csv")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    colour = COLORS[PRIMARY]
    ax.plot(
        frontier["distance_reduction_pct"],
        frontier["auc_change_pct"],
        color=colour,
        marker="o",
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    for _, row in frontier.iloc[[0, -1]].iterrows():
        ax.annotate(
            f"lambda = {row['lambda']:g}",
            (row["distance_reduction_pct"], row["auc_change_pct"]),
            xytext=(6, 6),
            textcoords="offset points",
            color=INK_2,
            fontsize=9,
        )
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("Reduction in coefficient movement at refit (%)")
    ax.set_ylabel("AUC change (%)")
    style(ax)
    title(
        ax,
        "Stability can be bought almost for free",
        "refit on new data while penalising departure from the old coefficients",
    )
    return save(fig, "stability_frontier.png")


# --- more fairness views ---------------------------------------------------------------


def plot_fairness_by_group(min_n: int = 30) -> Path:
    rates = read("fairness_by_group.csv").query("attribute == 'race' and count >= @min_n")
    groups = list(dict.fromkeys(rates.sort_values("count", ascending=False)["group"]))
    metrics = {
        "selection_rate": "Flagged",
        "fpr": "False-positive rate",
        "fnr": "False-negative rate",
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.8), sharey=True, layout="constrained")
    width = 0.36
    for ax, (metric, name) in zip(axes, metrics.items(), strict=True):
        for i, fs in enumerate(FEATURE_SETS):
            block = rates[rates["feature_set"] == fs].set_index("group").reindex(groups)
            ax.bar(
                np.arange(len(groups)) + (i - 0.5) * (width + 0.03),
                block[metric],
                width=width,
                color=color(fs),
                label=label(fs),
            )
        ax.set_xticks(range(len(groups)), groups, rotation=20, ha="right")
        ax.yaxis.set_major_formatter(PCT)
        style(ax)
        title(ax, name)
    figure_legend(fig, axes[0])
    fig.suptitle(
        f"Error rates by race at the cost-optimal threshold (groups of {min_n}+)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return save(fig, "fairness_by_group.png")


def plot_impossibility(comparison: str = "aa_vs_others") -> Path:
    table = read("impossibility.csv").query("comparison == @comparison")
    gaps = {"fpr_gap": "FPR gap", "fnr_gap": "FNR gap", "ppv_gap": "PPV gap (calibration)"}
    fig, axes = plt.subplots(1, len(gaps), figsize=(13, 3.8), sharey=True, layout="constrained")
    for ax, (column, name) in zip(axes, gaps.items(), strict=True):
        for fs, block in table.groupby("feature_set", sort=False):
            ax.plot(block["threshold"], block[column], color=color(fs), label=label(fs))
            ax.axvline(
                block["operating_threshold"].iloc[0], color=color(fs), linewidth=1, linestyle=":"
            )
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlabel("Decision threshold")
        style(ax)
        title(ax, name)
    figure_legend(fig, axes[0])
    fig.suptitle(
        "No threshold closes every gap at once (dotted: cost-optimal)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return save(fig, "impossibility.png")


def plot_mitigation(comparison: str = "aa_vs_others") -> Path:
    table = read("mitigation.csv").query("comparison == @comparison")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for fs, block in table.groupby("feature_set", sort=False):
        before = block[block["strategy"] == "uniform threshold"].iloc[0]
        after = block[block["strategy"] == "per-group thresholds"].iloc[0]
        ax.annotate(
            "",
            (after["cost_per_capita"], after["fpr_gap"]),
            (before["cost_per_capita"], before["fpr_gap"]),
            arrowprops={"arrowstyle": "->", "color": color(fs), "linewidth": 2},
        )
        ax.scatter(
            before["cost_per_capita"],
            before["fpr_gap"],
            s=64,
            color=color(fs),
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
            label=f"{label(fs)}: uniform",
        )
        ax.scatter(
            after["cost_per_capita"],
            after["fpr_gap"],
            s=64,
            color=SURFACE,
            edgecolor=color(fs),
            linewidth=2,
            zorder=3,
            label=f"{label(fs)}: per-group thresholds",
        )
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.xaxis.set_major_formatter(dollars)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Expected cost per defendant")
    ax.set_ylabel("FPR gap")
    ax.legend(loc="upper right", fontsize=8.5)
    style(ax)
    title(
        ax,
        "Closing the FPR gap with per-group thresholds, and what it costs",
        f"{COMPARISON_TITLES[comparison]}; conditions on race, so evidence, not a remedy",
    )
    return save(fig, "mitigation.png")


def plot_stratum_flag_rates(
    stratifier: str = "priors_band", comparison: str = "aa_vs_others"
) -> Path:
    table = read("fairness_stratum_odds_ratios.csv").query(
        "stratifier == @stratifier and comparison == @comparison"
    )
    fig, axes = plt.subplots(
        1, len(FEATURE_SETS), figsize=(12, 3.6), sharey=True, layout="constrained"
    )
    for ax, fs in zip(np.atleast_1d(axes), FEATURE_SETS, strict=True):
        block = table[table["feature_set"] == fs].reset_index(drop=True)
        y = np.arange(len(block))
        ax.hlines(
            y,
            block["flag_rate_reference"],
            block["flag_rate_protected"],
            color=GRID,
            linewidth=3,
            zorder=1,
        )
        ax.scatter(
            block["flag_rate_reference"],
            y,
            s=64,
            color=NEUTRAL,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
            label=COMPARISONS_REFERENCE[comparison],
        )
        ax.scatter(
            block["flag_rate_protected"],
            y,
            s=64,
            color=color(fs),
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
            label="African-American",
        )
        for yi, row in block.iterrows():
            ax.text(
                1.01,
                yi,
                f"OR {row['odds_ratio']:.1f}",
                va="center",
                color=INK_2,
                fontsize=9,
                transform=ax.get_yaxis_transform(),
            )
        ax.set_yticks(y, [f"{stratifier.split('_')[0]} {s}" for s in block["stratum"]])
        ax.xaxis.set_major_formatter(PCT)
        ax.set_xlabel("Share flagged high-risk")
        ax.legend(loc="upper left", fontsize=8.5)
        style(ax, grid="x")
        title(ax, label(fs))
    fig.suptitle(
        f"Within every {stratifier.replace('_', ' ')}, African-American defendants are flagged "
        "more (bands where everyone is flagged carry no comparison and are omitted)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return save(fig, "stratum_flag_rates.png")


COMPARISONS_REFERENCE = {"aa_vs_others": "All other groups", "aa_vs_caucasian": "Caucasian"}


def plot_xper_cost() -> Path:
    table = read("xper_cost.csv").query("feature != 'benchmark'")
    slots = int(table.groupby("feature_set").size().max())
    fig, axes = plt.subplots(
        1, len(FEATURE_SETS), figsize=(13, 0.4 * slots + 1.6), layout="constrained"
    )
    for ax, fs in zip(np.atleast_1d(axes), FEATURE_SETS, strict=True):
        block = table[table["feature_set"] == fs].sort_values("contribution", ascending=False)
        positions = np.arange(len(block)) + (slots - len(block))
        colours = [LOWERS if v < 0 else RAISES for v in block["contribution"]]
        ax.barh(positions, block["contribution"], height=0.6, color=colours)
        ax.set_yticks(positions, block["feature"])
        ax.set_ylim(-0.6, slots - 0.4)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.xaxis.set_major_formatter(dollars)
        ax.set_xlabel("Contribution to cost per defendant")
        style(ax, grid="x")
        title(ax, label(fs))
    fig.suptitle(
        "XPER on cost: blue features save money, red ones cost it",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return save(fig, "xper_cost.png")


FIGURES = {
    "auc_intervals": plot_auc_intervals,
    "cost_curves": plot_cost_curves,
    "odds_ratios": plot_odds_ratios,
    "marginal_effects": plot_marginal_effects,
    "lasso_path": plot_lasso_path,
    "pdp_ice": plot_pdp_ice,
    "local_contributions": plot_local_contributions,
    "learning_curve": plot_learning_curve,
    "fairness_ztests": plot_fairness_ztests,
    "mh_odds_ratios": plot_mh_odds_ratios,
    "tost": plot_tost,
    "fpdp": plot_fpdp,
    "fairness_frontier": plot_fairness_frontier,
    "calibration": plot_calibration_by_group,
    "xper": plot_xper,
    "scorecard": plot_scorecard,
    "stepwise": plot_stepwise,
    "importance_comparison": plot_importance_comparison,
    "cost_sensitivity": plot_cost_sensitivity,
    "subgroup_performance": plot_subgroup_performance,
    "stability_frontier": plot_stability_frontier,
    "fairness_by_group": plot_fairness_by_group,
    "impossibility": plot_impossibility,
    "mitigation": plot_mitigation,
    "stratum_flag_rates": plot_stratum_flag_rates,
    "xper_cost": plot_xper_cost,
}


def main() -> int:
    if not ART.exists():
        print(f"{ART} not found -- run `uv run python -m logreg.analysis` first")
        return 1
    for fn in FIGURES.values():
        fn()
    print(f"\n{len(FIGURES)} figures in {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
