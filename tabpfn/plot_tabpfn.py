"""Figures for TabPFN's predictive performance. Run via `make tabpfn-plots`.

Reads only the committed outputs of run_tabpfn.py (tabpfn/artifacts/), so it needs no model
fit and runs in seconds. Writes PNGs to reports/figures/tabpfn/ (git-ignored, like the EDA
figures): the cross-feature-set figures at the top, and one folder per feature set with
its holdout figures. Performance only: fairness and stability figures belong to the analysis.

"COMPAS tool" is Northpointe's risk score, the dataset's ``score_factor`` column (1 = rated
medium or high risk): the incumbent that TabPFN is compared against on the same defendants.

Colour follows the entity in every figure: TabPFN race-aware is blue, TabPFN race-blind is
aqua, the COMPAS tool is orange. The palette is the validated reference palette (light mode).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pfn_metrics import at_threshold, break_even_threshold  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from compas_scoring.config import CONFIG, project_root  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
OUT = project_root() / "reports" / "figures" / "tabpfn"

# Slots 1-3 of the reference palette pass all-pairs; aqua is below 3:1 contrast on this
# surface, so every aqua series also carries a legend entry.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]  # blue ramp

TOOL = "COMPAS tool"  # model name in performance.csv
TOOL_DECISION = "COMPAS tool: medium/high risk"
AWARE = ("TabPFN", "race_aware")
BLIND = ("TabPFN", "race_blind")
SERIES = {
    AWARE: ("TabPFN, race-aware", BLUE),
    BLIND: ("TabPFN, race-blind", AQUA),
    (TOOL, "race_aware"): ("COMPAS tool score", ORANGE),
}
RUN_LABELS = {
    "holdout": "Holdout 70/30",
    "X1+X2_to_X3": "X1+X2 → X3",
    "X1_to_X3": "X1 → X3",
    "X2_to_X3": "X2 → X3",
    "temporal_1": "Time: first 40% → next 20%",
    "temporal_2": "Time: first 70% → last 30%",
}
BREAK_EVEN = break_even_threshold()
# Readable names for the feature sets, in ablation order (see pyproject.toml).
FS_NAMES = {
    "race_aware": "all features",
    "race_blind": "no race",
    "sex_blind": "no sex",
    "age_blind": "no age",
    "protected_blind": "no race, sex or age",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TEXT_2,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "text.color": TEXT,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def predictions(run: str, feature_set: str = "race_aware") -> pd.DataFrame:
    return pd.read_csv(ART / "predictions" / f"{run}__{feature_set}.csv", index_col="row")


def perf_row(perf: pd.DataFrame, run: str, feature_set: str, model: str, point: str):
    mask = (
        (perf["run"] == run)
        & (perf["feature_set"] == feature_set)
        & (perf["model"] == model)
        & (perf["operating_point"] == point)
    )
    return perf[mask].iloc[0]


def save(fig, name: str, feature_set: str | None = None) -> None:
    folder = OUT / feature_set if feature_set else OUT
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.relative_to(project_root())}")


def threshold_line(ax, x: float, label: str) -> None:
    ax.axvline(x, color=MUTED, lw=1, ls="--", zorder=1)
    ax.annotate(
        label,
        (x, 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        va="top",
        fontsize=8.5,
        color=TEXT_2,
    )


def point(ax, x, y, colour, marker="o") -> None:
    ax.plot(x, y, marker, ms=8, color=colour, mec=SURFACE, mew=2, zorder=4)


def note(ax, text, xy, offset) -> None:
    ax.annotate(text, xy, xytext=offset, textcoords="offset points", fontsize=8.5, color=TEXT_2)


# --------------------------------------------------------------------------------- figures


def fig_roc(perf: pd.DataFrame, df: pd.DataFrame, fs: str) -> None:
    """ROC on the holdout, with the two operating points and the COMPAS tool's one point."""
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(0.8, 0.72, "chance", color=MUTED, fontsize=8.5, rotation=40)

    label, colour = f"TabPFN, {FS_NAMES[fs]}", BLUE
    auc = perf_row(perf, "holdout", fs, "TabPFN", "0.5")["auc"]
    fpr, tpr, _ = roc_curve(df["y"], df["tabpfn"])
    ax.plot(fpr, tpr, color=colour, label=f"{label}  (AUC {auc:.3f})", zorder=3)
    for t in (0.5, BREAK_EVEN):
        m = at_threshold(df["y"], df["tabpfn"], t)
        point(ax, m["type_i_error"], m["recall"], colour)
        note(ax, f"t = {t:.3g}", (m["type_i_error"], m["recall"]), (8, -12))

    label, colour = SERIES[(TOOL, "race_aware")]
    auc = perf_row(perf, "holdout", fs, TOOL, "0.5")["auc"]
    m = at_threshold(df["y"], df["compas_tool"], 0.5)
    ax.plot(
        [0, m["type_i_error"], 1],
        [0, m["recall"], 1],
        color=colour,
        lw=1.4,
        zorder=2,
        label=f"{label}  (AUC {auc:.3f})",
    )
    point(ax, m["type_i_error"], m["recall"], colour, "s")
    note(ax, TOOL_DECISION, (m["type_i_error"], m["recall"]), (8, -4))

    ax.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
    ax.set_xlabel("False positive rate  (Type I error)")
    ax.set_ylabel("True positive rate  (power)")
    ax.set_title(f"ROC, holdout (n = 1,852), TabPFN on {FS_NAMES[fs]}")
    ax.legend(loc="lower right")
    save(fig, "01_roc_holdout.png", fs)


def fig_auc_by_run(perf: pd.DataFrame) -> None:
    """AUC with 95% bootstrap CIs across every run: TabPFN (both sets) and the COMPAS tool."""
    runs = list(RUN_LABELS)
    auc = perf[perf["operating_point"] == "0.5"]
    offsets = {AWARE: -0.22, BLIND: 0.0, (TOOL, "race_aware"): 0.22}
    fig, ax = plt.subplots(figsize=(8, 5))

    for (model, fs), (label, colour) in SERIES.items():
        rows = auc[(auc["model"] == model) & (auc["feature_set"] == fs)].set_index("run").loc[runs]
        ax.errorbar(
            rows["auc"],
            np.arange(len(runs)) + offsets[(model, fs)],
            xerr=[rows["auc"] - rows["auc_ci_low"], rows["auc_ci_high"] - rows["auc"]],
            fmt="s" if model == TOOL else "o",
            ms=7,
            color=colour,
            ecolor=colour,
            elinewidth=1.6,
            capsize=0,
            mec=SURFACE,
            mew=1.5,
            label=label,
            zorder=3,
        )

    # The two time-ordered runs use the dated cohort: a different population, set apart.
    ax.axhline(3.5, color=MUTED, lw=0.8)
    ax.text(0.585, 3.62, "dated cohort (33% base rate): compare only with each other",
            fontsize=8.5, color=TEXT_2, va="bottom")  # fmt: skip
    ax.text(0.585, -0.5, "modelling table (45.5% base rate)", fontsize=8.5, color=TEXT_2)
    ax.set_yticks(range(len(runs)), [RUN_LABELS[r] for r in runs])
    ax.set_ylim(len(runs) - 0.5, -0.75)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.58, 0.78)
    ax.set_xlabel("AUC  (bars: 95% bootstrap CI)")
    ax.set_title("AUC by train/test design")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    save(fig, "02_auc_by_run.png")


def fig_confusion(df: pd.DataFrame, fs: str) -> None:
    """Confusion matrices on the holdout: TabPFN at both thresholds, and the COMPAS tool."""
    panels = [
        ("TabPFN, t = 0.5", df["tabpfn"], 0.5),
        (f"TabPFN, t = {BREAK_EVEN:.3f} (break-even)", df["tabpfn"], BREAK_EVEN),
        (TOOL_DECISION, df["compas_tool"], 0.5),
    ]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("blue", SEQ)
    names = [["TN", "FP  (Type I)"], ["FN  (Type II)", "TP"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, score, t) in zip(axes, panels):
        m = at_threshold(df["y"], score, t)
        cells = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        rates = cells / cells.sum(axis=1, keepdims=True)
        ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ink = SURFACE if rates[i, j] > 0.55 else TEXT
                text = f"{names[i][j]}\n{cells[i, j]:,}\n{rates[i, j]:.0%} of row"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=ink)
        ax.set_xticks([0, 1], ["released", "flagged"])
        ax.set_yticks([0, 1], ["did not\nre-offend", "re-offended"])
        ax.set_xlabel("decision")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=10.5)
    title = f"Confusion matrices, holdout (n = 1,852), TabPFN on {FS_NAMES[fs]}"
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "03_confusion_holdout.png", fs)


def fig_threshold_tradeoff(df: pd.DataFrame, fs: str) -> None:
    """Error rates and expected cost across thresholds: two charts, one y-axis each."""
    ts = np.linspace(0.01, 0.99, 197)
    rows = pd.DataFrame([at_threshold(df["y"], df["tabpfn"], t) for t in ts])
    tool = at_threshold(df["y"], df["compas_tool"], 0.5)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"hspace": 0.55}
    )
    top.plot(ts, rows["type_i_error"], color=VIOLET,
             label="Type I error (FPR): flagged, would not have re-offended")  # fmt: skip
    top.plot(ts, rows["type_ii_error"], color=RED,
             label="Type II error (FNR): released, then re-offended")  # fmt: skip
    top.set(ylim=(0, 1), ylabel="error rate")
    top.set_title(f"TabPFN error rates and cost by threshold, holdout ({FS_NAMES[fs]})")
    top.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1)
    top.tick_params(labelbottom=True)

    bottom.plot(ts, rows["cost_per_defendant"] / 1000, color=BLUE, label="TabPFN")
    bottom.axhline(tool["cost_per_defendant"] / 1000, color=ORANGE, lw=1.4, label=TOOL_DECISION)
    best = int(rows["cost_per_defendant"].idxmin())
    best_cost = rows["cost_per_defendant"].iat[best]
    point(bottom, ts[best], best_cost / 1000, BLUE)
    bottom.annotate(
        f"minimum \\${best_cost:,.0f} at t = {ts[best]:.2f}",
        (ts[best], best_cost / 1000),
        xytext=(0, 10),
        ha="center",
        textcoords="offset points",
        fontsize=8.5,
        color=TEXT_2,
    )
    bottom.set(xlim=(0, 1), ylabel="expected cost per defendant ($k)")
    bottom.set_xlabel("decision threshold t  (flag if score ≥ t)")
    bottom.legend(loc="upper left")
    c = CONFIG.costs
    bottom.text(0.99, 0.04, f"costs: FN \\${c.c_fn:,.0f}, FP \\${c.c_fp:,.0f} (assumptions)",
                transform=bottom.transAxes, ha="right", fontsize=8, color=TEXT_2)  # fmt: skip
    for ax in (top, bottom):
        threshold_line(ax, BREAK_EVEN, f"break-even {BREAK_EVEN:.3f}")
        threshold_line(ax, 0.5, "0.5")
    save(fig, "04_threshold_tradeoff.png", fs)


def fig_calibration(df: pd.DataFrame, fs: str) -> None:
    """Reliability diagram: do TabPFN's probabilities mean what they say?"""
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(df["tabpfn"], bins) - 1, 0, 9)
    table = (
        df.assign(bin=idx)
        .groupby("bin")
        .agg(p=("tabpfn", "mean"), obs=("y", "mean"), n=("y", "size"))
    )
    table = table[table["n"] >= 10]

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    ax.text(0.02, 0.07, "perfect calibration", color=MUTED, fontsize=8.5, rotation=43)
    ax.plot(table["p"], table["obs"], "-o", color=BLUE, ms=7, mec=SURFACE, mew=1.5,
            label=f"TabPFN, {FS_NAMES[fs]}")  # fmt: skip
    for _, r in table.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["p"], r["obs"]), xytext=(6, -10),
                    textcoords="offset points", fontsize=7.5, color=TEXT_2)  # fmt: skip
    brier = ((df["tabpfn"] - df["y"]) ** 2).mean()
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed re-offence rate")
    ax.set_title(f"Calibration, holdout, {FS_NAMES[fs]} (Brier {brier:.3f})")
    ax.legend(loc="upper left")
    save(fig, "05_calibration_holdout.png", fs)


def fig_score_distribution(df: pd.DataFrame, fs: str) -> None:
    """How well the scores separate the two outcomes, with both thresholds."""
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for outcome, colour, label in ((0, MUTED, "did not re-offend"), (1, RED, "re-offended")):
        scores = df.loc[df["y"] == outcome, "tabpfn"]
        ax.hist(scores, bins=bins, histtype="step", lw=2, color=colour,
                label=f"{label} (n = {len(scores):,})")  # fmt: skip
    threshold_line(ax, BREAK_EVEN, f"break-even {BREAK_EVEN:.3f}")
    threshold_line(ax, 0.5, "0.5")
    ax.set(xlim=(0, 1), ylabel="defendants")
    ax.set_xlabel("TabPFN score (predicted probability of re-offence)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"TabPFN score distribution by actual outcome, holdout ({FS_NAMES[fs]})")
    ax.legend(loc="upper right")
    save(fig, "06_score_distribution.png", fs)


def fig_metrics(perf: pd.DataFrame, fs: str) -> None:
    """Threshold metrics on the holdout: TabPFN at both operating points, the COMPAS tool."""
    series = [
        ("TabPFN, t = 0.5", perf_row(perf, "holdout", fs, "TabPFN", "0.5"), BLUE),
        (
            f"TabPFN, t = {BREAK_EVEN:.3f}",
            perf_row(perf, "holdout", fs, "TabPFN", "break_even"),
            VIOLET,
        ),
        (TOOL_DECISION, perf_row(perf, "holdout", fs, TOOL, "0.5"), ORANGE),
    ]
    metrics = ["accuracy", "precision", "recall", "f1", "specificity"]
    names = ["Accuracy", "Precision", "Recall\n(power)", "F1", "Specificity"]
    width = 0.26
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for k, (label, row, colour) in enumerate(series):
        values = row[metrics].to_numpy(float)
        bars = ax.bar(x + (k - 1) * width, values, width - 0.02, color=colour, label=label,
                      zorder=3)  # fmt: skip
        ax.bar_label(bars, [f"{v:.2f}" for v in values], padding=2, fontsize=7.5, color=TEXT_2)
    ax.set_xticks(x, names)
    ax.set(ylim=(0, 1.05), ylabel="score")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Threshold metrics, holdout, TabPFN on {FS_NAMES[fs]}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    save(fig, "07_metrics_holdout.png", fs)


def fig_ablation_auc(perf: pd.DataFrame) -> None:
    """AUC per feature set: what removing each protected attribute costs in ranking power."""
    designs = [
        ("holdout", "Holdout 70/30"),
        ("X1+X2_to_X3", "X1+X2 \u2192 X3"),
        ("temporal_2", "Time: first 70% \u2192 last 30%"),
    ]
    sets = feature_sets(perf)
    auc = perf[perf["operating_point"] == "0.5"]
    fig, axes = plt.subplots(1, len(designs), figsize=(12, 3.8), sharey=True)
    for ax, (run, title) in zip(axes, designs):
        rows = auc[(auc["run"] == run) & (auc["model"] == "TabPFN")]
        rows = rows.set_index("feature_set").loc[sets]
        y = np.arange(len(sets))
        ax.errorbar(
            rows["auc"],
            y,
            xerr=[rows["auc"] - rows["auc_ci_low"], rows["auc_ci_high"] - rows["auc"]],
            fmt="o",
            ms=7,
            color=BLUE,
            elinewidth=1.6,
            mec=SURFACE,
            mew=1.5,
            zorder=3,
            label="TabPFN (95% CI)",
        )
        for yi, v in zip(y, rows["auc"]):
            note(ax, f"{v:.3f}", (v, yi), (-10, 7))
        tool = auc[(auc["run"] == run) & (auc["model"] == TOOL)]["auc"].iat[0]
        ax.axvline(tool, color=ORANGE, lw=1.4, zorder=2, label=f"COMPAS tool ({tool:.3f})")
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("AUC")
    axes[0].set_yticks(range(len(sets)), [FS_NAMES[fs] for fs in sets])
    axes[0].set_ylim(len(sets) - 0.5, -0.6)
    fig.legend(
        [mpl.lines.Line2D([], [], color=ORANGE, lw=1.4), axes[0].containers[0]],
        ["COMPAS tool (AUC on the same test set)", "TabPFN (95% bootstrap CI)"],
        loc="lower center",
        ncol=2,
    )
    fig.suptitle("TabPFN AUC with protected attributes removed", x=0.02, ha="left",
                 fontweight="bold", fontsize=12)  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "08_ablation_auc.png")


def fig_ablation_metrics(perf: pd.DataFrame) -> None:
    """Threshold metrics and cost per feature set on the holdout, at both operating points."""
    sets = feature_sets(perf)
    hold = perf[perf["run"] == "holdout"]
    panels = [
        ("recall", "Recall (power)", 1),
        ("type_i_error", "Type I error (FPR)", 1),
        ("accuracy", "Accuracy", 1),
        ("cost_per_defendant", "Cost per defendant ($k)", 1000),
    ]
    points = [
        ("0.5", "TabPFN, t = 0.5", BLUE),
        ("break_even", f"TabPFN, t = {BREAK_EVEN:.3f}", VIOLET),
    ]
    y = np.arange(len(sets))
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.8), sharey=True)
    for ax, (metric, title, scale) in zip(axes, panels):
        for (op, label, colour), dy in zip(points, (-0.12, 0.12)):
            rows = hold[(hold["model"] == "TabPFN") & (hold["operating_point"] == op)]
            values = rows.set_index("feature_set").loc[sets, metric] / scale
            ax.plot(values, y + dy, "o", ms=7, color=colour, mec=SURFACE, mew=1.5, label=label)
        tool = hold[(hold["model"] == TOOL) & (hold["operating_point"] == "0.5")][metric].iat[0]
        ax.axvline(tool / scale, color=ORANGE, lw=1.4, zorder=1, label=TOOL_DECISION)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(sets)), [FS_NAMES[fs] for fs in sets])
    axes[0].set_ylim(len(sets) - 0.5, -0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Holdout operating points with protected attributes removed", x=0.02,
                 ha="left", fontweight="bold", fontsize=12)  # fmt: skip
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "09_ablation_metrics.png")


# Radar: one line per TabPFN variant. Five hues on crossing lines sit in the colour-blind
# warning band (validated: worst CVD dE 6.9), so each variant also gets its own marker and
# line style. The COMPAS tool is a neutral dashed reference, not a sixth hue.
VARIANT_STYLE = {
    "race_aware": (BLUE, "o", "-"),
    "race_blind": (AQUA, "s", "--"),
    "sex_blind": (RED, "^", "-."),
    "age_blind": ("#eda100", "D", ":"),
    "protected_blind": (VIOLET, "v", (0, (5, 1, 1, 1))),
}
# Spokes: (column in performance.csv, label, transform). All read "higher is better", 0-1.
RADAR_SPOKES = [
    ("auc", "Discrimination\n(AUC)", lambda v: v),
    ("balanced_accuracy", "Balanced\naccuracy", lambda v: v),
    ("recall", "Sensitivity\n(1 - FNR)", lambda v: v),
    ("specificity", "Specificity\n(1 - FPR)", lambda v: v),
    ("precision", "Precision", lambda v: v),
    ("f1", "F1", lambda v: v),
    ("ece", "Calibration\n(1 - ECE)", lambda v: 1 - v),
]
# TODO(analysis): add the interpretability, stability and fairness dimensions as spokes.
# Write tabpfn/artifacts/radar_extra.csv with one row per feature_set and one column per
# new dimension, each already scaled to 0-1 with higher = better (e.g. 1 - |FPR gap|,
# 1 - decision flip rate between X1 and X2). Every column in it becomes a spoke here.
RADAR_EXTRA = ART / "radar_extra.csv"


def radar_scores(perf: pd.DataFrame, point: str) -> pd.DataFrame:
    """Holdout scores per variant (rows) and spoke (columns), plus the COMPAS tool."""
    hold = perf[(perf["run"] == "holdout") & (perf["operating_point"] == point)]
    rows = {}
    for fs in feature_sets(perf):
        row = hold[(hold["feature_set"] == fs) & (hold["model"] == "TabPFN")].iloc[0]
        rows[fs] = {label: f(row[col]) for col, label, f in RADAR_SPOKES}
    tool = hold[(hold["feature_set"] == "race_aware") & (hold["model"] == TOOL)].iloc[0]
    rows[TOOL] = {label: f(tool[col]) for col, label, f in RADAR_SPOKES}
    table = pd.DataFrame(rows).T
    if RADAR_EXTRA.exists():
        extra = pd.read_csv(RADAR_EXTRA, index_col="feature_set")
        table = table.join(extra)
    return table


def fig_radar(perf: pd.DataFrame) -> None:
    """Every TabPFN variant on every metric, at both operating points (like the brief's radar)."""
    points = [("0.5", "t = 0.5"), ("break_even", f"t = {BREAK_EVEN:.3f} (break-even)")]
    tables = {op: radar_scores(perf, op) for op, _ in points}
    spokes = list(tables["0.5"].columns)
    angles = np.linspace(0, 2 * np.pi, len(spokes), endpoint=False)
    closed = np.r_[angles, angles[:1]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), subplot_kw={"projection": "polar"})
    for ax, (op, title) in zip(axes, points):
        table = tables[op]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"],
                      fontsize=7.5, color=MUTED)  # fmt: skip
        # Ring labels sit between the F1 and Precision spokes, where no line passes.
        ax.set_rlabel_position(np.degrees(angles[4] + angles[5]) / 2)
        ax.set_xticks(angles, spokes, fontsize=9)
        ax.tick_params(axis="x", pad=10)
        ax.grid(color=GRID, lw=0.6)
        ax.spines["polar"].set_color(MUTED)

        values = table.loc[TOOL].to_numpy(float)
        ax.plot(closed, np.r_[values, values[:1]], color=TEXT_2, lw=1.4, ls=(0, (4, 3)),
                label=TOOL_DECISION, zorder=2)  # fmt: skip
        for fs in [fs for fs in VARIANT_STYLE if fs in table.index]:
            colour, marker, style = VARIANT_STYLE[fs]
            values = table.loc[fs].to_numpy(float)
            ax.plot(closed, np.r_[values, values[:1]], color=colour, lw=1.8, ls=style,
                    marker=marker, ms=6, mec=SURFACE, mew=1, label=f"TabPFN, {FS_NAMES[fs]}",
                    zorder=3)  # fmt: skip
        ax.set_title(title, fontsize=11, pad=22, loc="center")

    handles, labels = axes[0].get_legend_handles_labels()
    order = list(range(1, len(handles))) + [0]  # variants first, the tool last
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.02))  # fmt: skip
    fig.suptitle("TabPFN with and without protected attributes, holdout (n = 1,852)", x=0.02,
                 ha="left", fontweight="bold", fontsize=12)  # fmt: skip
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    save(fig, "10_radar_holdout.png")

    # The table view of the same numbers (accessibility, and exact values for the report).
    long = pd.concat({op: t for op, t in tables.items()}, names=["operating_point", "model"])
    long.round(4).to_csv(OUT / "10_radar_holdout.csv")
    print(f"  -> {(OUT / '10_radar_holdout.csv').relative_to(project_root())}")


def feature_sets(perf: pd.DataFrame) -> list[str]:
    """The feature sets present in performance.csv, in ablation order."""
    present = set(perf["feature_set"])
    return [fs for fs in FS_NAMES if fs in present]


def main() -> int:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    perf = pd.read_csv(ART / "performance.csv")
    print("TabPFN performance figures")
    # Across feature sets, in reports/figures/tabpfn/.
    fig_auc_by_run(perf)
    fig_ablation_auc(perf)
    fig_ablation_metrics(perf)
    fig_radar(perf)
    # One folder per feature set: that set's holdout figures.
    for fs in feature_sets(perf):
        holdout = predictions("holdout", fs)
        fig_roc(perf, holdout, fs)
        fig_confusion(holdout, fs)
        fig_threshold_tradeoff(holdout, fs)
        fig_calibration(holdout, fs)
        fig_score_distribution(holdout, fs)
        fig_metrics(perf, fs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
