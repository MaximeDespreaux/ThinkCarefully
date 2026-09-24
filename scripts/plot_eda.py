"""Exploratory analysis figures. Run via `make eda-plots`."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import compas_scoring  # noqa: F401  (imported for its thread-pool pinning)
from compas_scoring.config import CONFIG
from compas_scoring.data import RACE_REFERENCE, build_dataset, group_frame, load_raw

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

FIG = CONFIG.path("reports", "figures", "eda")
REPORT = CONFIG.path("reports", "EDA.md")

# --- design tokens -------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots, assigned in fixed order and never cycled.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
# Sequential: one hue, light to dark.
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Diverging: warm/cool poles, neutral gray midpoint.
DIV_LO, DIV_MID, DIV_HI = "#2a78d6", "#f0efec", "#d03b3b"

SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ)
DIV_CMAP = mpl.colors.LinearSegmentedColormap.from_list("div_br", [DIV_LO, DIV_MID, DIV_HI])

mpl.rcParams.update(
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
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "figure.dpi": 150,
    }
)

PCT = PercentFormatter(xmax=1.0, decimals=0)
THOUSANDS = FuncFormatter(lambda v, _: f"{int(v):,}")


def style(ax, *, xgrid: bool = False) -> None:
    """Hairline recessive chrome: no top/right spines, grid on the value axis only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="x" if xgrid else "y")
    ax.grid(axis="y" if xgrid else "x", visible=False)
    ax.tick_params(length=0)


BAR = 0.44  # bar thickness in band units; bands are sized so this renders thin


def title(ax, text: str, subtitle: str | None = None) -> None:
    ax.set_title(
        text,
        loc="left",
        fontsize=11.5,
        fontweight="600",
        color=INK,
        pad=26 if subtitle else 12,
    )
    if subtitle:
        ax.text(
            0.0,
            1.012,
            subtitle,
            transform=ax.transAxes,
            fontsize=9,
            color=MUTED,
            va="bottom",
        )


def bands(ax, n_levels: int, slots: int) -> np.ndarray:
    """Centre `n_levels` bars inside a fixed number of bands.

    Every panel of a small multiple then draws its bars at the same pixel thickness,
    however many categories it has - a two-category panel does not get bars three
    times the weight of a six-category one.
    """
    ax.set_ylim(-0.6, slots - 0.4)
    return np.arange(n_levels) + (slots - n_levels) / 2


def header(fig, heading: str, note: str) -> None:
    """Figure-level title and standfirst, clear of the panel titles beneath them."""
    fig.suptitle(heading, x=0.007, y=1.15, ha="left", fontsize=13, fontweight="600", color=INK)
    fig.text(0.007, 1.055, note, ha="left", fontsize=9.5, color=MUTED)


def save(fig, name: str) -> Path:
    path = FIG / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {path.relative_to(CONFIG.root)}")
    return path


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval. Correct at the tails, where priors counts are thin."""
    n = np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.zeros_like(n, dtype=float), where=n > 0)
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return centre - half, centre + half


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013), matching scripts/run_eda.py."""
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c = r - (r - 1) ** 2 / (n - 1)
    k_c = k - (k - 1) ** 2 / (n - 1)
    denominator = min(k_c - 1, r_c - 1)
    return float(np.sqrt(phi2 / denominator)) if denominator > 0 else float("nan")


PRIOR_EDGES = [-1, 0, 1, 3, 6, 100]
PRIOR_LABELS = ["0", "1", "2-3", "4-6", "7+"]


def priors_band(values: pd.Series) -> pd.Series:
    """Band priors the same way scripts/run_eda.py does, so the two agree."""
    return pd.cut(values, PRIOR_EDGES, labels=PRIOR_LABELS)


ATTRIBUTE_LABELS = {
    "race": "Race",
    "sex": "Sex",
    "age_band": "Age band",
    "charge_degree": "Charge degree",
}


# =====================================================================================
# 1. Cohort composition
# =====================================================================================
def fig_composition(groups: pd.DataFrame, n: int, facts: dict) -> None:
    slots = max(groups[a].nunique() for a in ATTRIBUTE_LABELS)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1))
    for ax, attribute in zip(axes, ATTRIBUTE_LABELS, strict=True):
        counts = groups[attribute].value_counts().sort_values()
        y = bands(ax, len(counts), slots)
        ax.barh(y, counts.to_numpy(), height=BAR, color=S1)
        ax.set_yticks(y, list(counts.index))
        for yi, value in zip(y, counts.to_numpy(), strict=True):
            ax.text(
                value + n * 0.02,
                yi,
                f"{value:,}  ({value / n:.1%})",
                va="center",
                fontsize=8.5,
                color=INK_2,
            )
        ax.set_xlim(0, counts.max() * 1.42)
        ax.xaxis.set_major_formatter(THOUSANDS)
        style(ax, xgrid=True)
        title(ax, ATTRIBUTE_LABELS[attribute])

    header(
        fig,
        f"Who is in the cohort  ·  {n:,} defendants, Broward County FL, 2013-2014",
        "Two groups carry the cohort: African-American and Caucasian defendants are "
        f"{facts['top_two_share']:.0%} of it. Asian (n=31) and Native American (n=11) "
        "are too small to support any subgroup metric.",
    )
    fig.tight_layout()
    save(fig, "01_cohort_composition.png")


# =====================================================================================
# 2. Outcome rate by protected attribute
# =====================================================================================
def fig_group_rates(y: pd.Series, groups: pd.DataFrame, base: float) -> pd.DataFrame:
    rows = []
    for attribute in ATTRIBUTE_LABELS:
        grouped = y.groupby(groups[attribute])
        for level, chunk in grouped:
            rows.append(
                {
                    "attribute": attribute,
                    "group": level,
                    "n": len(chunk),
                    "rate": float(chunk.mean()),
                }
            )
    table = pd.DataFrame(rows)

    # Values sit in a right-aligned column rather than at each bar tip, so the
    # base-rate rule can cross the plot without ever striking through a number.
    limit = 0.82
    slots = max(groups[a].nunique() for a in ATTRIBUTE_LABELS)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    for ax, attribute in zip(axes, ATTRIBUTE_LABELS, strict=True):
        chunk = table[table["attribute"] == attribute].sort_values("rate")
        y = bands(ax, len(chunk), slots)
        ax.barh(y, chunk["rate"].to_numpy(), height=BAR, color=S1)
        ax.set_yticks(y, list(chunk["group"]))
        for yi, rate, count in zip(y, chunk["rate"], chunk["n"], strict=True):
            ax.text(
                limit,
                yi,
                f"{rate:.1%}",
                va="center",
                ha="right",
                fontsize=9.5,
                color=INK,
                fontweight="600",
            )
            ax.text(0.012, yi, f"n={count:,}", va="center", fontsize=8, color="#ffffff")
        ax.axvline(base, color=DIV_HI, linewidth=1.4, zorder=1)
        ax.set_xlim(0, limit)
        ax.set_xticks([0, 0.2, 0.4, 0.6])
        ax.xaxis.set_major_formatter(PCT)
        style(ax, xgrid=True)
        title(ax, ATTRIBUTE_LABELS[attribute])
    axes[0].text(
        base + 0.015,
        slots - 0.5,
        f"base rate {base:.1%}",
        fontsize=8.5,
        color=DIV_HI,
        fontweight="600",
        va="top",
    )

    header(
        fig,
        "Two-year re-arrest rate by protected attribute",
        "The label is re-arrest, not re-offending. Arrest rates differ across exactly "
        "the groups being compared, so the target carries the bias the fairness work "
        "measures.",
    )
    fig.tight_layout()
    save(fig, "02_recidivism_by_group.png")
    return table


# =====================================================================================
# 3. Priors: distribution, and the same distribution split by outcome
# =====================================================================================
def fig_priors(df: pd.DataFrame, y: pd.Series, facts: dict) -> None:
    priors = df["Number_of_Priors"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6))

    counts = priors.value_counts().sort_index()
    axes[0].bar(counts.index, counts.to_numpy(), width=0.62, color=S1)
    axes[0].set_xlim(-0.8, 38.8)
    axes[0].set_xlabel("Prior offences")
    axes[0].set_yticks([0, 500, 1000, 1500, 2000])
    axes[0].yaxis.set_major_formatter(THOUSANDS)
    axes[0].annotate(
        f"{counts.loc[0]:,} defendants ({counts.loc[0] / len(df):.0%}) have none",
        xy=(0, counts.loc[0]),
        xytext=(5.2, counts.loc[0] * 0.93),
        fontsize=9,
        color=INK_2,
        arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 0.9},
    )
    axes[0].annotate(
        f"median {facts['priors_median']:.0f}  ·  mean {facts['priors_mean']:.2f}  ·  "
        f"max {facts['priors_max']:.0f}",
        xy=(0.42, 0.62),
        xycoords="axes fraction",
        fontsize=9,
        color=MUTED,
    )
    style(axes[0])
    title(
        axes[0],
        "Prior offences are the only continuous predictor",
        "and they are heavily right-skewed",
    )

    band = priors_band(priors)
    share = (
        pd.crosstab(band, y, normalize="columns")
        .rename(columns={0: "No re-arrest", 1: "Re-arrested"})
        .reindex(PRIOR_LABELS)
    )
    x = np.arange(len(share))
    width = 0.38
    axes[1].bar(x - width / 2 - 0.01, share["No re-arrest"], width, color=S1, label="No re-arrest")
    axes[1].bar(x + width / 2 + 0.01, share["Re-arrested"], width, color=S2, label="Re-arrested")
    axes[1].set_xticks(x, PRIOR_LABELS)
    axes[1].set_xlabel("Prior offences")
    axes[1].yaxis.set_major_formatter(PCT)
    axes[1].set_ylim(0, 0.62)
    axes[1].legend(loc="upper right", fontsize=9, labelcolor=INK_2)
    for xi, (a, b) in enumerate(zip(share["No re-arrest"], share["Re-arrested"], strict=True)):
        axes[1].text(
            xi - width / 2 - 0.01, a + 0.012, f"{a:.0%}", ha="center", fontsize=8.5, color=INK_2
        )
        axes[1].text(
            xi + width / 2 + 0.01, b + 0.012, f"{b:.0%}", ha="center", fontsize=8.5, color=INK_2
        )
    style(axes[1])
    title(
        axes[1],
        "Composition of each outcome group",
        "share of each outcome falling in a priors band",
    )

    fig.tight_layout()
    save(fig, "03_priors_distribution.png")


# =====================================================================================
# 4. The dose-response curve: re-arrest rate against priors
# =====================================================================================
def fig_priors_response(df: pd.DataFrame, y: pd.Series, base: float, facts: dict) -> None:
    priors = df["Number_of_Priors"]
    grouped = y.groupby(priors)
    rate = grouped.mean()
    count = grouped.size()
    keep = count.index[count.index <= 20]
    rate, count = rate.loc[keep], count.loc[keep]
    low, high = wilson(rate.to_numpy() * count.to_numpy(), count.to_numpy())

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.18}
    )

    axes[0].fill_between(keep, low, high, color=S1, alpha=0.10, linewidth=0)
    axes[0].plot(keep, rate, color=S1, linewidth=2, solid_capstyle="round", zorder=3)
    axes[0].scatter(keep, rate, s=34, color=S1, edgecolor=SURFACE, linewidth=2, zorder=4)
    axes[0].axhline(base, color=DIV_HI, linewidth=1.4)
    axes[0].text(
        0.35,
        base + 0.035,
        f"cohort base rate {base:.1%}",
        fontsize=8.5,
        color=DIV_HI,
        fontweight="600",
        va="bottom",
    )
    for point in (0, 5, 15):
        axes[0].annotate(
            f"{rate.loc[point]:.0%}",
            xy=(point, rate.loc[point]),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="600",
            color=INK,
        )
    axes[0].set_ylim(0.15, 1.02)
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("Re-arrested within two years")
    style(axes[0])
    title(
        axes[0],
        "Re-arrest rises steeply with priors, then flattens",
        "shaded band is a 95% Wilson interval; priors above 20 are pooled out (n < 30 each)",
    )

    axes[1].bar(keep, count.to_numpy(), width=0.62, color=MUTED)
    axes[1].set_xlabel("Prior offences")
    axes[1].set_ylabel("Defendants")
    axes[1].yaxis.set_major_formatter(THOUSANDS)
    axes[1].set_xlim(-0.8, 20.8)
    axes[1].set_xticks(range(0, 21, 5))
    style(axes[1])

    fig.text(
        0.007,
        -0.04,
        f"The first five priors move the rate {facts['priors_0']:.0%} -> {facts['priors_5']:.0%}; "
        f"the next ten move it only {facts['priors_5']:.0%} -> {facts['priors_15']:.0%}. "
        "That concavity is why the white-box model is given a log-priors term.",
        ha="left",
        fontsize=9.5,
        color=MUTED,
    )
    save(fig, "04_priors_response_curve.png")


# =====================================================================================
# 5. Association matrix (Cramer's V)
# =====================================================================================
def association_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Cramer's V over the predictors, with priors banded."""
    discrete = X.copy()
    discrete["Number_of_Priors"] = priors_band(discrete["Number_of_Priors"])

    columns = list(discrete.columns)
    matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for i, a in enumerate(columns):
        for b in columns[i:]:
            v = 1.0 if a == b else cramers_v(discrete[a], discrete[b])
            matrix.loc[a, b] = matrix.loc[b, a] = v
    return matrix


def fig_association(matrix: pd.DataFrame, facts: dict) -> None:
    # Upper triangle only, with the always-empty first column and last row trimmed off.
    names = list(matrix.columns)
    masked = matrix.to_numpy().copy()
    masked[np.tril_indices(len(names))] = np.nan
    masked = masked[:-1, 1:]
    rows, columns = names[:-1], names[1:]

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    image = ax.imshow(masked, cmap=SEQ_CMAP, vmin=0, vmax=0.35)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)), rows, fontsize=9)
    for i in range(len(rows)):
        for j in range(len(columns)):
            value = masked[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if value > 0.20 else INK_2,
            )
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    bar = fig.colorbar(image, ax=ax, shrink=0.55, pad=0.03)
    bar.set_label("Cramer's V", color=INK_2, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelcolor=INK_2, labelsize=8)

    title(
        ax,
        "Association between predictors (bias-corrected Cramer's V)",
        f"median {facts['median_assoc']:.3f}, max {facts['max_assoc']:.3f} - "
        "Pearson would understate this, because nine of ten predictors are dummies",
    )
    save(fig, "05_association_matrix.png")


# =====================================================================================
# 6. PCA - the negative result
# =====================================================================================
def fig_pca(ratio: np.ndarray, facts: dict) -> None:
    cumulative = np.cumsum(ratio)
    components = np.arange(1, len(ratio) + 1)
    uniform = 1 / len(ratio)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    axes[0].bar(components, ratio, width=0.62, color=S1)
    axes[0].axhline(uniform, color=DIV_HI, linewidth=1.4)
    axes[0].text(
        len(ratio) + 0.15,
        uniform,
        f" {uniform:.0%} = perfectly\n uncorrelated",
        va="center",
        fontsize=8.5,
        color=DIV_HI,
        fontweight="600",
    )
    axes[0].text(
        1,
        ratio[0] + 0.006,
        f"PC1 {ratio[0]:.1%}",
        ha="center",
        fontsize=9,
        fontweight="600",
        color=INK,
    )
    axes[0].set_xticks(components)
    axes[0].set_xlabel("Component")
    axes[0].set_ylim(0, 0.19)
    axes[0].set_yticks([0, 0.05, 0.10, 0.15])
    axes[0].yaxis.set_major_formatter(PCT)
    style(axes[0])
    title(axes[0], "Scree plot: no elbow", "explained variance per component")

    axes[1].plot(components, cumulative, color=S1, linewidth=2, solid_capstyle="round", zorder=3)
    axes[1].scatter(
        components, cumulative, s=34, color=S1, edgecolor=SURFACE, linewidth=2, zorder=4
    )
    axes[1].plot(components, components / len(ratio), color=MUTED, linewidth=1.4, zorder=2)
    axes[1].text(6.1, 6 / len(ratio) - 0.075, "perfectly uncorrelated", fontsize=8.5, color=MUTED)
    axes[1].axhline(0.90, color=DIV_HI, linewidth=1.4)
    n90 = facts["components_for_90"]
    axes[1].annotate(
        f"{n90} of {len(ratio)} components\nneeded for 90%",
        xy=(n90, cumulative[n90 - 1]),
        xytext=(-118, -6),
        textcoords="offset points",
        fontsize=9,
        fontweight="600",
        color=DIV_HI,
        arrowprops={"arrowstyle": "-", "color": DIV_HI, "linewidth": 0.9},
    )
    axes[1].set_xticks(components)
    axes[1].set_xlabel("Components retained")
    axes[1].set_ylim(0, 1.04)
    axes[1].set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    axes[1].yaxis.set_major_formatter(PCT)
    style(axes[1])
    title(
        axes[1],
        "Cumulative variance tracks the diagonal",
        "which is what a complete absence of structure looks like",
    )

    fig.tight_layout()
    save(fig, "06_pca_scree.png")


# =====================================================================================
# 7. Univariate lift: what each indicator moves the base rate to
# =====================================================================================
def fig_lift(X: pd.DataFrame, y: pd.Series, base: float) -> pd.DataFrame:
    rows = []
    for column in X.columns:
        if column == "Number_of_Priors":
            continue
        on = X[column] == 1
        rows.append(
            {
                "feature": column,
                "prevalence": float(on.mean()),
                "rate_on": float(y[on].mean()),
                "rate_off": float(y[~on].mean()),
                "n_on": int(on.sum()),
            }
        )
    lift = pd.DataFrame(rows)
    lift["gap"] = lift["rate_on"] - lift["rate_off"]
    lift = lift.sort_values("gap")

    limit = float(np.abs(lift["gap"]).max()) * 1.08
    norm = mpl.colors.Normalize(vmin=-limit, vmax=limit)
    colors = [DIV_CMAP(norm(g)) for g in lift["gap"]]

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})

    axes[0].barh(lift["feature"], lift["gap"], height=0.6, color=colors)
    axes[0].axvline(0, color=AXIS, linewidth=1)
    for i, (gap, n_on) in enumerate(zip(lift["gap"], lift["n_on"], strict=True)):
        offset = 0.007 if gap >= 0 else -0.007
        axes[0].text(
            gap + offset,
            i,
            f"{gap:+.1%}   n={n_on:,}",
            va="center",
            ha="left" if gap >= 0 else "right",
            fontsize=8.5,
            color=INK_2,
        )
    axes[0].set_xlim(-limit * 1.55, limit * 1.55)
    axes[0].xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    style(axes[0], xgrid=True)
    title(
        axes[0],
        "Each indicator's raw effect on the re-arrest rate",
        "rate when the flag is on, minus the rate when it is off - no controls",
    )

    # One series, one colour: the gap is already encoded on the left panel, and the
    # identity here is carried by the direct label beside each point.
    axes[1].scatter(
        lift["prevalence"],
        lift["rate_on"],
        s=110,
        color=S1,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=3,
    )
    axes[1].axhline(base, color=DIV_HI, linewidth=1.4, zorder=1)
    axes[1].text(
        0.60,
        base + 0.012,
        f"base rate {base:.1%}",
        fontsize=8.5,
        ha="right",
        color=DIV_HI,
        fontweight="600",
    )
    for _, row in lift.iterrows():
        # Labels go left or right of the point by where there is room, so the two
        # near-zero-prevalence markers do not print on top of the y-axis.
        right = row["prevalence"] < 0.30
        on_the_rule = abs(row["rate_on"] - base) < 0.02
        axes[1].annotate(
            row["feature"],
            xy=(row["prevalence"], row["rate_on"]),
            xytext=(11 if right else -11, -13 if on_the_rule else 0),
            textcoords="offset points",
            ha="left" if right else "right",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    axes[1].set_xlim(-0.03, 0.72)
    axes[1].set_ylim(0.20, 0.62)
    axes[1].xaxis.set_major_formatter(PCT)
    axes[1].yaxis.set_major_formatter(PCT)
    axes[1].set_xlabel("Prevalence (share of cohort with the flag on)")
    axes[1].set_ylabel("Re-arrest rate when on")
    style(axes[1])
    axes[1].grid(axis="x")
    title(
        axes[1],
        "Prevalence against effect",
        "a rare flag with a big gap moves few decisions; a common one moves many",
    )

    fig.tight_layout()
    save(fig, "07_univariate_lift.png")
    return lift


# =====================================================================================
# 8. The incumbent: what COMPAS's own decision already does
# =====================================================================================
def fig_incumbent(
    df: pd.DataFrame, y: pd.Series, groups: pd.DataFrame, facts: dict
) -> pd.DataFrame:
    incumbent = df[CONFIG.incumbent].astype(int)
    present = set(groups["race"])
    major = [r for r in ("African-American", RACE_REFERENCE, "Hispanic") if r in present]

    rows = []
    for race in major:
        mask = groups["race"] == race
        actual, flagged = y[mask], incumbent[mask]
        negatives, positives = actual == 0, actual == 1
        rows.append(
            {
                "race": race,
                "n": int(mask.sum()),
                "flag_rate": float(flagged.mean()),
                "base_rate": float(actual.mean()),
                "fpr": float(flagged[negatives].mean()),
                "fnr": float(1 - flagged[positives].mean()),
                "ppv": float(actual[flagged == 1].mean()),
            }
        )
    table = pd.DataFrame(rows)

    # One row order across all three panels. Re-sorting each panel by its own value
    # would make the rows stop lining up, which is the whole point of a small multiple.
    order = table.sort_values("n", ascending=False).iloc[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
    panels = [
        ("flag_rate", "Flagged high-risk by COMPAS", "share of the group scored decile > 4"),
        ("fpr", "False positive rate", "not re-arrested, but flagged high-risk"),
        ("fnr", "False negative rate", "re-arrested, but flagged low-risk"),
    ]
    for ax, (column, heading, subtitle) in zip(axes, panels, strict=True):
        y = bands(ax, len(order), len(order) + 0.4)
        ax.barh(y, order[column].to_numpy(), height=BAR, color=S1)
        ax.set_yticks(y, list(order["race"]))
        for yi, value in zip(y, order[column], strict=True):
            ax.text(
                value + 0.016,
                yi,
                f"{value:.1%}",
                va="center",
                fontsize=9.5,
                fontweight="600",
                color=INK,
            )
        ax.set_xlim(0, 0.76)
        ax.set_xticks([0, 0.2, 0.4, 0.6])
        ax.xaxis.set_major_formatter(PCT)
        style(ax, xgrid=True)
        title(ax, heading, subtitle)

    header(
        fig,
        "The incumbent instrument, before any model of ours is fitted",
        f"COMPAS flags Black defendants who were not re-arrested at {facts['fpr_aa']:.1%} against "
        f"{facts['fpr_ca']:.1%} for white defendants - a {facts['fpr_gap']:.1f}pp gap - while "
        f"missing {facts['fnr_ca']:.1%} of white re-offenders against {facts['fnr_aa']:.1%}. "
        "This is ProPublica's finding, reproduced on the modelling table.",
    )
    fig.tight_layout()
    save(fig, "08_incumbent_benchmark.png")
    return table


# =====================================================================================
# 9. Why race-blindness does not work: priors carries race
# =====================================================================================
def fig_proxy(df: pd.DataFrame, groups: pd.DataFrame, facts: dict) -> None:
    race = groups["race"]
    keys = ["African-American", RACE_REFERENCE, "All other groups"]
    series = {
        "African-American": df.loc[race == "African-American", "Number_of_Priors"],
        RACE_REFERENCE: df.loc[race == RACE_REFERENCE, "Number_of_Priors"],
        "All other groups": df.loc[
            ~race.isin(["African-American", RACE_REFERENCE]), "Number_of_Priors"
        ],
    }
    colors = {keys[0]: S1, keys[1]: S2, keys[2]: S3}

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 4.7))

    for key in keys:
        values = np.sort(series[key].to_numpy())
        ax_x = np.concatenate([[0], values])
        ax_y = np.concatenate([[0], np.arange(1, len(values) + 1) / len(values)])
        axes[0].step(ax_x, ax_y, where="post", color=colors[key], linewidth=2, label=key, zorder=3)
    # One annotation, on the group the finding is about. Labelling all three at x=0
    # stacks them on top of each other; the legend already carries the identities.
    share_none = {key: float((series[key] == 0).mean()) for key in keys}
    axes[0].annotate(
        f"{share_none[keys[0]]:.0%} of African-American defendants have no priors,\n"
        f"against {share_none[keys[1]]:.0%} of Caucasian and "
        f"{share_none[keys[2]]:.0%} of everyone else",
        xy=(0.0, share_none[keys[0]]),
        xytext=(5.0, 0.20),
        fontsize=8.5,
        color=INK_2,
        arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 0.9},
    )
    axes[0].set_xlim(-0.4, 20)
    axes[0].set_xticks(range(0, 21, 5))
    axes[0].set_ylim(0, 1.02)
    axes[0].set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_xlabel("Prior offences")
    axes[0].set_ylabel("Share of the group at or below")
    axes[0].legend(loc="center right", fontsize=9, labelcolor=INK_2)
    style(axes[0])
    title(
        axes[0],
        "Priors is distributed differently by race",
        "cumulative distribution - the curve further right carries more priors",
    )

    band = priors_band(df["Number_of_Priors"])
    grouped = pd.Series(
        np.where(
            race == "African-American",
            "African-American",
            np.where(race == RACE_REFERENCE, RACE_REFERENCE, "All other groups"),
        ),
        index=df.index,
    )
    share = pd.crosstab(grouped, band, normalize="index").reindex(keys)
    x = np.arange(len(PRIOR_LABELS))
    width = 0.26
    for offset, key in zip((-1, 0, 1), keys, strict=True):
        axes[1].bar(
            x + offset * (width + 0.015),
            share.loc[key].to_numpy(),
            width,
            color=colors[key],
            label=key,
        )
    axes[1].set_xticks(x, PRIOR_LABELS)
    axes[1].set_xlabel("Prior offences")
    axes[1].set_ylim(0, 0.52)
    axes[1].yaxis.set_major_formatter(PCT)
    axes[1].legend(loc="upper right", fontsize=9, labelcolor=INK_2)
    style(axes[1])
    title(
        axes[1],
        "Same fact, banded",
        "share of each group falling in a priors band",
    )

    fig.text(
        0.007,
        -0.045,
        f"Mean priors: {facts['priors_aa']:.2f} for African-American defendants against "
        f"{facts['priors_ca']:.2f} for Caucasian. Dropping the race column leaves this "
        "difference in the feature matrix, which is why FS3 does not buy fairness.",
        ha="left",
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout()
    save(fig, "09_priors_as_race_proxy.png")


# =====================================================================================
# 10. Where the risk actually sits: age band x priors band
# =====================================================================================
def fig_risk_grid(df: pd.DataFrame, y: pd.Series, groups: pd.DataFrame, base: float) -> None:
    band = priors_band(df["Number_of_Priors"])
    ages = ["Less than 25", "25 - 45", "Greater than 45"]

    rate = y.groupby([groups["age_band"], band]).mean().unstack().reindex(ages)[PRIOR_LABELS]
    count = y.groupby([groups["age_band"], band]).size().unstack().reindex(ages)[PRIOR_LABELS]

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    image = ax.imshow(rate.to_numpy(), cmap=SEQ_CMAP, vmin=0.15, vmax=0.85, aspect="auto")
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            value = rate.iat[i, j]
            ax.text(
                j,
                i - 0.09,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="600",
                color="#ffffff" if value > 0.52 else INK,
            )
            ax.text(
                j,
                i + 0.22,
                f"n={count.iat[i, j]:,}",
                ha="center",
                va="center",
                fontsize=8,
                color="#dbe6f4" if value > 0.52 else INK_2,
            )
    ax.set_xticks(range(rate.shape[1]), rate.columns, fontsize=9.5)
    ax.set_yticks(range(rate.shape[0]), rate.index, fontsize=9.5)
    ax.set_xlabel("Prior offences")
    ax.set_xticks(np.arange(-0.5, rate.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rate.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    bar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02)
    bar.set_label("Re-arrest rate", color=INK_2, fontsize=9)
    bar.ax.axhline(base, color=DIV_HI, linewidth=1.4)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelcolor=INK_2, labelsize=8)
    bar.ax.yaxis.set_major_formatter(PCT)

    title(
        ax,
        "Re-arrest rate by age band and priors band",
        "the two features the L1 path keeps - red line on the scale is the cohort base rate",
    )
    fig.tight_layout()
    save(fig, "10_risk_grid_age_priors.png")


# =====================================================================================
def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    data = build_dataset("race_aware")
    groups = group_frame(df)
    y, X = data.y, data.X
    base = data.base_rate
    n = len(df)

    race = groups["race"]
    priors = df["Number_of_Priors"]
    by_priors = y.groupby(priors).mean()
    incumbent = df[CONFIG.incumbent].astype(int)

    def rate_for(mask, column):
        return (
            float(incumbent[mask & (y == 0)].mean())
            if column == "fpr"
            else float(1 - incumbent[mask & (y == 1)].mean())
        )

    aa, ca = race == "African-American", race == RACE_REFERENCE
    facts = {
        "top_two_share": float((aa | ca).mean()),
        "priors_median": float(priors.median()),
        "priors_mean": float(priors.mean()),
        "priors_max": float(priors.max()),
        "priors_0": float(by_priors.loc[0]),
        "priors_5": float(by_priors.loc[5]),
        "priors_15": float(by_priors.loc[15]),
        "priors_aa": float(priors[aa].mean()),
        "priors_ca": float(priors[ca].mean()),
        "fpr_aa": rate_for(aa, "fpr"),
        "fpr_ca": rate_for(ca, "fpr"),
        "fnr_aa": rate_for(aa, "fnr"),
        "fnr_ca": rate_for(ca, "fnr"),
    }
    facts["fpr_gap"] = (facts["fpr_aa"] - facts["fpr_ca"]) * 100

    print(f"WS1 figures  ({n:,} rows x {df.shape[1]} columns, base rate {base:.2%})")
    print("-" * 74)

    # Headline numbers first: several figures print them in their own subtitles, so they
    # have to exist before anything is drawn.
    matrix = association_matrix(X)
    upper = matrix.to_numpy()[np.triu_indices(len(matrix), k=1)]
    facts["median_assoc"] = float(np.nanmedian(upper))
    facts["max_assoc"] = float(np.nanmax(upper))

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    ratio = PCA().fit(StandardScaler().fit_transform(X)).explained_variance_ratio_
    facts["pc1"] = float(ratio[0])
    facts["components_for_90"] = int(np.searchsorted(np.cumsum(ratio), 0.90) + 1)

    fig_composition(groups, n, facts)
    rates = fig_group_rates(y, groups, base)
    fig_priors(df, y, facts)
    fig_priors_response(df, y, base, facts)
    fig_association(matrix, facts)
    fig_pca(ratio, facts)
    lift = fig_lift(X, y, base)
    incumbent_table = fig_incumbent(df, y, groups, facts)
    fig_proxy(df, groups, facts)
    fig_risk_grid(df, y, groups, base)

    write_report(facts, rates, lift, incumbent_table, matrix, base, n)
    print(f"\n  -> {REPORT.relative_to(CONFIG.root)}")
    return 0


def write_report(
    facts: dict,
    rates: pd.DataFrame,
    lift: pd.DataFrame,
    incumbent: pd.DataFrame,
    matrix: pd.DataFrame,
    base: float,
    n: int,
) -> None:
    """Write the figures up, so each one is read with the number it rests on."""
    strongest = (
        matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    race_rates = rates[rates["attribute"] == "race"].sort_values("rate", ascending=False)

    def table(frame: pd.DataFrame, columns: dict[str, str], formats: dict) -> str:
        head = "| " + " | ".join(columns.values()) + " |"
        rule = "|" + "|".join(["---"] * len(columns)) + "|"
        body = []
        for _, row in frame.iterrows():
            body.append("| " + " | ".join(formats[c](row[c]) for c in columns) + " |")
        return "\n".join([head, rule, *body])

    spread = []
    for attribute, label in ATTRIBUTE_LABELS.items():
        chunk = rates[rates["attribute"] == attribute].sort_values("rate")
        low, high = chunk.iloc[0], chunk.iloc[-1]
        spread.append(
            f"- **{label}** — {low['group']} {low['rate']:.1%} → "
            f"{high['group']} {high['rate']:.1%}  ({(high['rate'] - low['rate']) * 100:.1f}pp)"
        )
    spread_lines = "\n".join(spread)

    by_race = rates[rates["attribute"] == "race"].set_index("group")["rate"]
    aa_rate, ca_rate = float(by_race["African-American"]), float(by_race["Caucasian"])
    major_gap = (aa_rate - ca_rate) * 100

    race_table = table(
        race_rates,
        {"group": "Race", "n": "n", "rate": "Re-arrest rate"},
        {"group": str, "n": lambda v: f"{int(v):,}", "rate": lambda v: f"{v:.2%}"},
    )
    lift_table = table(
        lift.sort_values("gap", ascending=False),
        {
            "feature": "Feature",
            "prevalence": "Prevalence",
            "rate_on": "Rate when on",
            "gap": "Gap (pp)",
        },
        {
            "feature": lambda v: f"`{v}`",
            "prevalence": lambda v: f"{v:.1%}",
            "rate_on": lambda v: f"{v:.1%}",
            "gap": lambda v: f"{v * 100:+.1f}",
        },
    )
    incumbent_table = table(
        incumbent,
        {
            "race": "Race",
            "n": "n",
            "flag_rate": "Flagged",
            "fpr": "FPR",
            "fnr": "FNR",
            "ppv": "PPV",
        },
        {
            "race": str,
            "n": lambda v: f"{int(v):,}",
            "flag_rate": lambda v: f"{v:.1%}",
            "fpr": lambda v: f"{v:.1%}",
            "fnr": lambda v: f"{v:.1%}",
            "ppv": lambda v: f"{v:.1%}",
        },
    )
    strongest_pairs = "\n".join(
        f"- `{a}` ↔ `{b}` — V = {v:.3f}" for (a, b), v in strongest.head(3).items()
    )

    text = f"""# Exploratory data analysis — the COMPAS cohort

Generated by `scripts/plot_eda.py`. Figures in [reports/figures/eda/](figures/eda/); the
underlying numbers are in `artifacts/eda_*.csv`, written by `scripts/run_eda.py`.

{n:,} defendants screened in Broward County, Florida during 2013–2014. {base:.2%} were
re-arrested within two years. Ten predictors, nine of them binary indicators, plus
`score_factor` — COMPAS's own decile>4 decision, held out as the benchmark and never used
as a feature.

---

## 1. Who is in the cohort — `01_cohort_composition.png`

Two groups carry it: African-American and Caucasian defendants are {facts["top_two_share"]:.0%}
of the table. Asian (n=31) and Native American (n=11) defendants are present but too small
to support any subgroup metric, which is why disparities in this project are computed
between named groups rather than as max–min statistics.

## 2. Re-arrest rate by protected attribute — `02_recidivism_by_group.png`

{race_table}

Every protected attribute separates the outcome, not just race:

{spread_lines}

Race shows the widest range, but its low end is the 31 Asian defendants, where the rate is
noise. Between the two groups large enough to carry a metric the gap is {major_gap:.1f}pp
(African-American {aa_rate:.1%} against Caucasian {ca_rate:.1%}) — which puts race behind
age band, not ahead of it.

**The label is re-arrest, not re-offending.** Arrest rates differ across exactly the groups
being compared, so the target carries the bias the fairness analysis sets out to measure.
Every rate above inherits that.

## 3–4. Priors — `03_priors_distribution.png`, `04_priors_response_curve.png`

`Number_of_Priors` is the only genuinely continuous predictor: median {facts["priors_median"]:.0f},
mean {facts["priors_mean"]:.2f}, max {facts["priors_max"]:.0f}, heavily right-skewed.

Its effect is strongly **concave**. The first five priors move the re-arrest rate
{facts["priors_0"]:.0%} → {facts["priors_5"]:.0%}; the next ten move it only
{facts["priors_5"]:.0%} → {facts["priors_15"]:.0%}. A raw linear term underfits the low end
and overstates the tail, which is exactly why `data.engineered()` hands logistic regression
a `log_priors` term, a capped term and a `no_priors` flag — the non-linearity a tree gets
for free.

## 5. Association — `05_association_matrix.png`

Bias-corrected Cramér's V, not Pearson: nine of the ten predictors are dummies, and a
Pearson correlation between two dummies is a phi coefficient bounded by the marginals, so
it understates association whenever prevalences differ — which here they always do.

Median association {facts["median_assoc"]:.3f}, max {facts["max_assoc"]:.3f}. The three
strongest pairs:

{strongest_pairs}

All three are mechanical: the race dummies are mutually exclusive, and so are the two age
dummies. There is no substantive collinearity to treat.

## 6. PCA — `06_pca_scree.png`

PC1 explains {facts["pc1"]:.1%} of the variance; {facts["components_for_90"]} of 10
components are needed for 90%. Perfectly uncorrelated features would put every component at
10.0% and track the diagonal exactly — which is nearly what the cumulative curve does.

**No dimensionality reduction is available, and that is the finding.** A scree plot with no
elbow is information: the predictors are near-orthogonal indicators, so a PCA rotation would
cost all interpretability and buy no compression.

## 7. Univariate lift — `07_univariate_lift.png`

Raw, uncontrolled gaps — the rate when a flag is on minus the rate when it is off:

{lift_table}

The second panel pairs each gap with its prevalence, because the two together decide how
many decisions a feature actually moves: a rare flag with a large gap touches few people.

## 8. The incumbent — `08_incumbent_benchmark.png`

COMPAS's own decision, before any model here is fitted:

{incumbent_table}

ProPublica's finding, reproduced on the modelling table: among defendants who were *not*
re-arrested, {facts["fpr_aa"]:.1%} of Black defendants were flagged high-risk against
{facts["fpr_ca"]:.1%} of white defendants — a {facts["fpr_gap"]:.1f}pp gap — while the
instrument missed {facts["fnr_ca"]:.1%} of white re-offenders against
{facts["fnr_aa"]:.1%} of Black ones. The errors fall in opposite directions by group.

## 9. Priors is a race proxy — `09_priors_as_race_proxy.png`

Mean priors: {facts["priors_aa"]:.2f} for African-American defendants against
{facts["priors_ca"]:.2f} for Caucasian defendants. The distributions separate across their
whole range, not just at the tail.

This is the mechanism behind the project's race-blind result. Dropping the race columns
(FS3) leaves this difference sitting in `Number_of_Priors`, so "fairness through
unawareness" removes the label and keeps the information.

## 10. Where the risk sits — `10_risk_grid_age_priors.png`

Re-arrest rate on the two dimensions the L1 path keeps. The grid is close to additive —
rate rises with priors within every age band, and with youth within every priors band, with
no cell where the ordering reverses.

That is the descriptive counterpart to two of the project's modelling findings: PLTR finds
no interaction structure to exploit, and every model family ties. There is not much here
for a flexible model to find that a linear one cannot.

---

## What the EDA implies for the modelling

1. **A thin, near-orthogonal feature set.** Ten predictors, no redundancy to compress, no
   collinearity to treat, and no interaction structure visible. Expect model families to
   converge — and they do.
2. **One non-linearity worth engineering.** Priors is concave in the outcome; everything
   else is binary and cannot be anything but linear in the log-odds.
3. **Disparity is in the data before it is in any model.** Every protected attribute splits
   the outcome, the incumbent's errors already split by race, and priors carries race
   information on its own. No choice of estimator changes that.
4. **The target is re-arrest.** Every fairness number downstream is a statement about a
   label that is itself produced by policing, and is reported as such.
"""
    REPORT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
