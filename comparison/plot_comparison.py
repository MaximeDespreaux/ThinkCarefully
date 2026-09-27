"""Slide figures for the cross-model comparison, from comparison/artifacts/ (run_comparison.py).

    uv run python comparison/plot_comparison.py   # -> reports/figures/comparison/*.png

One figure per slide idea, in story order: data, performance, interpretability, stability,
fairness (steps 1-3), trade-offs. Colours follow the model everywhere: LogReg blue, XGBoost
orange, TabPFN violet, and the COMPAS tool as a grey dashed reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from compas_scoring.config import CONFIG  # noqa: E402
from compas_scoring.data import build_dataset  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
OUT = HERE.parent / "reports" / "figures" / "comparison"

SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COLOUR = {"LogReg": "#2a78d6", "XGBoost": "#eb6834", "TabPFN": "#4a3aa7", "COMPAS tool": MUTED}
MARKER = {"LogReg": "o", "XGBoost": "s", "TabPFN": "D", "COMPAS tool": "^"}
MODELS = ["LogReg", "XGBoost", "TabPFN"]
AA, CA = "African-American", "Caucasian"
RACE_COLOUR = {AA: "#4a3aa7", CA: "#1baf7a", "Other groups": MUTED}
SET_NAME = {
    "race_aware": "All features",
    "race_blind": "No race",
    "race_priors_blind": "No race, no priors",
    "priors_blind": "Drop priors, retrain",
    "priors_set_1": "Priors set to 1",
}
NICE = {
    "Number_of_Priors": "Priors (+1)",
    "Age_Below_TwentyFive": "Under 25",
    "Age_Above_FourtyFive": "Over 45",
    "Female": "Female",
    "Misdemeanor": "Misdemeanour",
    "African_American": "African-American",
    "Hispanic": "Hispanic",
    "Asian": "Asian",
    "Native_American": "Native American",
    "Other": "Other race",
}
BREAK_EVEN = CONFIG.costs.break_even


def style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": TEXT_2, "axes.edgecolor": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "text.color": TEXT, "legend.frameon": False, "lines.linewidth": 2,
    })  # fmt: skip


def title(fig, text: str, sub: str | None = None) -> None:
    fig.text(0.01, 1.05, text, ha="left", va="bottom", fontweight="bold", fontsize=14)
    if sub:
        fig.text(0.01, 1.045, sub, ha="left", va="top", fontsize=10, color=TEXT_2)


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(ART / name)


def perf_row(perf, model, run="holdout", fs="race_aware", threshold=0.5):
    q = perf[(perf.model == model) & (perf.run == run) & (perf.feature_set == fs)]
    return q[np.isclose(q.threshold, threshold)].iloc[0]


# ================================================================================== data


def fig_data() -> None:
    """Priors bands by race, and the re-offence rate by race."""
    data = build_dataset("race_aware")
    race = data.groups["race"].where(data.groups["race"].isin([AA, CA]), "Other groups")
    band = pd.cut(data.X["Number_of_Priors"], [-1, 0, 1, 3, 6, 100],
                  labels=["0", "1", "2-3", "4-6", "7+"])  # fmt: skip
    share = pd.crosstab(band, race, normalize="columns")[[AA, CA, "Other groups"]]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(len(share))
    for k, g in enumerate(share.columns):
        bars = ax.bar(x + (k - 1) * 0.27, share[g], 0.25, color=RACE_COLOUR[g], label=g, zorder=3)
        ax.bar_label(bars, [f"{v:.0%}" for v in share[g]], padding=2, fontsize=9, color=TEXT_2)
    ax.set_xticks(x, share.index)
    ax.set(xlabel="number of prior offences", ylabel="share of the group")
    ax.set_title("Priors: the strongest predictor, and unevenly distributed by race")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center")
    mean = data.X["Number_of_Priors"].groupby(race).mean()
    ax.text(0.98, 0.62, f"mean priors\n{AA}: {mean[AA]:.1f}\n{CA}: {mean[CA]:.1f}",
            transform=ax.transAxes, ha="right", fontsize=10, color=TEXT_2)  # fmt: skip

    rate = data.y.groupby(race).mean()[[AA, CA, "Other groups"]]
    n = race.value_counts()[rate.index]
    bars = ax2.bar(range(3), rate, 0.6, color=[RACE_COLOUR[g] for g in rate.index], zorder=3)
    ax2.bar_label(bars, [f"{v:.0%}" for v in rate], padding=2, fontsize=10, color=TEXT_2)
    ax2.set_xticks(range(3), [f"{g}\n(n = {n[g]:,})" for g in rate.index], fontsize=9)
    ax2.set(ylim=(0, 0.7), ylabel="re-offended within two years")
    ax2.set_title("Base rates differ too")
    ax2.grid(axis="x", visible=False)
    title(
        fig,
        "The data carries race through priors",
        "COMPAS two-year cohort, 6,172 defendants (Broward County, 2013-14)",
    )
    fig.tight_layout()
    save(fig, "01_data_priors_by_race.png")


# ============================================================================ performance


def fig_radar() -> None:
    perf = read("performance.csv")
    spokes = [("auc", "AUC"), ("accuracy", "Accuracy"), ("precision", "Precision"),
              ("recall", "Recall"), ("f1", "F1")]  # fmt: skip
    angles = np.linspace(0, 2 * np.pi, len(spokes), endpoint=False)
    closed = np.r_[angles, angles[:1]]
    low, high = 0.5, 0.8
    fig, ax = plt.subplots(figsize=(7.5, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(low, high)
    ticks = np.arange(low, high + 1e-9, 0.1)
    ax.set_yticks(ticks, [f"{t:.1f}" for t in ticks], fontsize=9, color=TEXT_2)
    ax.set_rlabel_position(360 / len(spokes) / 2)
    ax.set_xticks(angles, [s[1] for s in spokes], fontsize=12)
    ax.tick_params(axis="x", pad=12)
    styles = {"LogReg": "-", "XGBoost": "-", "TabPFN": "-", "COMPAS tool": "--"}
    for model in [*MODELS, "COMPAS tool"]:
        row = perf_row(perf, model)
        values = np.array([row[k] for k, _ in spokes])
        ax.plot(closed, np.r_[values, values[:1]], styles[model], color=COLOUR[model],
                marker=MARKER[model], ms=7, mec=SURFACE, lw=2.2 if model != "COMPAS tool" else 1.6,
                label=f"{model}  (AUC {row['auc']:.3f})")  # fmt: skip
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=11)
    title(
        fig,
        "Predictive performance, all features",
        "Holdout test set (n = 1,852), threshold 0.5. Radial axis from 0.5 to 0.8.",
    )
    save(fig, "02_performance_radar.png")


def fig_xper_person() -> None:
    table = read("xper_person.csv").set_index("feature")
    row = int(table.loc["row", "LogReg"])
    scores = table.loc[["score"]].max()  # one score per model
    parts = table.drop(index=["row", "score", "benchmark"])
    order = parts.abs().max(axis=1).sort_values().index
    parts = parts.loc[order]
    person = build_dataset("race_aware").X.loc[row]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, model in zip(axes, MODELS):
        values = parts[model].astype(float)
        colours = ["#e34948" if v > 0 else "#2a78d6" for v in values]
        ax.barh(range(len(values)), values, color=colours, height=0.6, zorder=3)
        for i, v in enumerate(values):
            if abs(v) > 1e-4:
                ax.annotate(f"{v:+.4f}", (v, i), xytext=(4 if v > 0 else -4, -3),
                            textcoords="offset points", ha="left" if v > 0 else "right",
                            fontsize=8.5, color=TEXT_2)  # fmt: skip
        ax.axvline(0, color=TEXT_2, lw=1)
        ax.set_xlim(-0.16, 0.38)
        ax.set_title(f"{model}   (risk score {float(scores[model]):.2f})", color=COLOUR[model])
        ax.grid(axis="y", visible=False)
    labels = [f"{NICE.get(f, f).replace(' (+1)', '')} = {person[f]:g}" for f in parts.index]
    axes[0].set_yticks(range(len(parts)), labels, fontsize=10)
    fig.supxlabel("contribution of this defendant's feature to the model's AUC (XPER)",
                  fontsize=10, color=TEXT_2)  # fmt: skip
    title(
        fig,
        "Same defendant, three models: what drives each model's performance",
        f"XPER per defendant (test row {row}): African-American, 25-45, "
        f"{person['Number_of_Priors']:g} priors, near the 0.5 threshold. "
        "Red raises AUC, blue lowers it.",
    )
    fig.tight_layout()
    save(fig, "03_xper_same_defendant.png")


# ======================================================================= interpretability


def fig_marginal_effects() -> None:
    me = read("marginal_effects.csv").query("run == 'holdout'")
    wide = me.pivot_table(index="feature", columns="model", values="marginal_effect")[MODELS]
    small = ["Asian", "Native_American"]  # ~11-31 defendants: not interpretable
    top = wide.drop(index=small).abs().mean(axis=1).sort_values(ascending=False).index[:5]
    wide = wide.loc[top[::-1]]
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(wide))
    for k, model in enumerate(MODELS):
        bars = ax.barh(y + (k - 1) * 0.26, wide[model], 0.24, color=COLOUR[model],
                       label=model, zorder=3)  # fmt: skip
        ax.bar_label(bars, [f"{v:+.3f}" for v in wide[model]], padding=3, fontsize=8.5,
                     color=TEXT_2)  # fmt: skip
    ax.axvline(0, color=TEXT_2, lw=1)
    ax.set_yticks(y, [NICE[f] for f in wide.index])
    ax.set_xlabel("average change in predicted risk (0 → 1 for a yes/no feature, +1 prior)")
    ax.set_xlim(-0.2, 0.25)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    title(
        fig,
        "Marginal effects: age and priors drive all three models",
        "Top 5 features by average |effect|, holdout, all features, 500 test defendants",
    )
    fig.tight_layout()
    save(fig, "04_marginal_effects_top5.png")


def fig_pdp() -> None:
    pdp = read("pdp_priors.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for model in MODELS:
        g = pdp[pdp.model == model]
        ax.plot(g.Number_of_Priors, g.predicted_risk, color=COLOUR[model], marker=MARKER[model],
                ms=5, mec=SURFACE, label=model)  # fmt: skip
    ax.set(xlabel="number of prior offences (set for everyone)", ylabel="average predicted risk",
           ylim=(0, 1))  # fmt: skip
    ax.legend(loc="lower right")
    title(
        fig,
        "Partial dependence on priors: same shape, different curvature",
        "Holdout, all features, 500 test defendants; grid from the 5th to 95th percentile",
    )
    fig.tight_layout()
    save(fig, "05_pdp_priors.png")


# ============================================================================== stability


def fig_stability() -> None:
    st = read("stability.csv")
    metrics = [("delta_auc", "AUC"), ("delta_accuracy", "Accuracy"), ("delta_recall", "Recall"),
               ("delta_f1", "F1"), ("me_l2", "Marginal effects\n(L2 distance)")]  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, pair in zip(axes, ["X1 vs X2", "time 1 vs 2"]):
        x = np.arange(len(metrics))
        for k, model in enumerate(MODELS):
            row = st.query("model == @model and pair == @pair").iloc[0]
            vals = [row[c] for c, _ in metrics]
            bars = ax.bar(x + (k - 1) * 0.26, vals, 0.24, color=COLOUR[model], label=model,
                          zorder=3)  # fmt: skip
            ax.bar_label(bars, [f"{v:.3f}" for v in vals], padding=2, fontsize=8, color=TEXT_2,
                         rotation=90)  # fmt: skip
        ax.set_xticks(x, [name for _, name in metrics], fontsize=10)
        ax.grid(axis="x", visible=False)
        name = (
            "Two random 40% samples, same test set"
            if pair == "X1 vs X2"
            else "Two time windows (first 40% → next 20% vs first 70% → last 30%)"
        )
        ax.set_title(name, fontsize=11)
    axes[0].set_ylabel("absolute change between the two fits")
    axes[0].set_ylim(0, 0.19)
    axes[0].legend(loc="upper left")
    title(
        fig,
        "Stability: how much each model changes when the training data changes",
        "Threshold 0.5, all features. Lower = more stable. XGBoost's swing over time comes from "
        "its compressed scores (max 0.54 there), not from a change in ranking: "
        "its AUC moves 0.004.",
    )
    fig.tight_layout()
    save(fig, "06_stability.png")


def fig_auc_by_design() -> None:
    perf = read("performance.csv")
    runs = [("holdout", "Holdout 70/30"), ("X1_to_X3", "X1 → X3"), ("X2_to_X3", "X2 → X3"),
            ("temporal_1", "Time: 40% → 20%"), ("temporal_2", "Time: 70% → 30%")]  # fmt: skip
    fig, ax = plt.subplots(figsize=(10, 4.6))
    offsets = {m: (k - 1.5) * 0.18 for k, m in enumerate([*MODELS, "COMPAS tool"])}
    for model in [*MODELS, "COMPAS tool"]:
        rows = [perf_row(perf, model, run) for run, _ in runs]
        auc = np.array([r["auc"] for r in rows])
        err = [auc - [r["auc_low"] for r in rows], [r["auc_high"] for r in rows] - auc]
        ax.errorbar(np.arange(len(runs)) + offsets[model], auc, yerr=err, fmt=MARKER[model],
                    color=COLOUR[model], ms=7, mec=SURFACE, capsize=0, elinewidth=1.6,
                    label=model)  # fmt: skip
    ax.axvline(2.5, color=MUTED, lw=0.8, ls=":")
    ax.text(2.55, 0.585, "dated cohort (33% base rate)", fontsize=9, color=TEXT_2)
    ax.set_xticks(range(len(runs)), [name for _, name in runs])
    ax.set(ylabel="AUC (95% CI)", ylim=(0.58, 0.78))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    title(
        fig,
        "AUC holds across training samples and over time, for every model",
        "All features. Time designs use the dated cohort; compare them with each other.",
    )
    fig.tight_layout()
    save(fig, "07_auc_by_design.png")


# =============================================================================== fairness


def fairness_rows(fs_list) -> pd.DataFrame:
    fair = read("fairness.csv")
    return fair[(fair.comparison == f"{AA} vs {CA}") & fair.feature_set.isin(fs_list)]


def fig_fairness_step1() -> None:
    fair = fairness_rows(["race_aware", "race_blind", "race_priors_blind"])
    panels = [
        ("statistical_parity", "Flag-rate gap", "gap"),
        ("fpr", "False-positive-rate gap", "gap"),
        ("fnr", "False-negative-rate gap", "gap"),
        ("conditional_statistical_parity", "Same proxies: odds ratio", "mh_odds_ratio"),
    ]
    sets = ["race_aware", "race_blind", "race_priors_blind"]
    entries = [(m, fs) for m in MODELS for fs in sets] + [("COMPAS tool", "race_aware")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.6), sharey=True)
    y = np.arange(len(entries))[::-1]
    for ax, (metric, name, col) in zip(axes, panels):
        for yi, (model, fs) in zip(y, entries):
            r = fair.query("model == @model and feature_set == @fs and metric == @metric")
            if r.empty or pd.isna(r.iloc[0][col]):
                continue
            r = r.iloc[0]
            lo, hi = (r.ci_low, r.ci_high) if col == "gap" else (r.mh_or_low, r.mh_or_high)
            hollow = r.p_value >= 0.05
            ax.plot([lo, hi], [yi, yi], color=COLOUR[model], lw=1.8)
            ax.plot(r[col], yi, MARKER[model], color=COLOUR[model], ms=8,
                    mfc=SURFACE if hollow else COLOUR[model], mec=COLOUR[model])  # fmt: skip
            p = "p < 0.001" if r.p_value < 0.001 else f"p = {r.p_value:.2f}"
            ax.annotate(f"{r[col]:+.2f}  ({p})" if col == "gap" else f"{r[col]:.2f}  ({p})",
                        (hi, yi), xytext=(5, -3), textcoords="offset points", fontsize=8,
                        color=TEXT_2)  # fmt: skip
        ax.axvline(1 if col == "mh_odds_ratio" else 0, color=TEXT_2, lw=1, ls="--")
        ax.set_title(name, fontsize=11)
        ax.grid(axis="y", visible=False)
        if col == "mh_odds_ratio":
            ax.set_xscale("log")
            ax.set_xticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axes[0].set_yticks(y, [f"{m} · {SET_NAME[fs]}" if m != "COMPAS tool" else "COMPAS tool"
                           for m, fs in entries])  # fmt: skip
    for x0, x1, ax in ((-0.1, 0.45, axes[0]), (-0.05, 0.4, axes[1]), (-0.5, 0.15, axes[2])):
        ax.set_xlim(x0, x1)
    title(
        fig,
        "Step 1: all three models treat African-American defendants differently",
        "African-American vs Caucasian, holdout, t = 0.5, 95% CI. Hollow = not significant. "
        "Odds ratio: among defendants with the same priors band, age band and charge.",
    )
    fig.tight_layout()
    save(fig, "08_fairness_step1.png")


def fig_fpdp() -> None:
    fp = read("fpdp_priors.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharex=True)
    for model in MODELS:
        g = fp[fp.model == model].sort_values("value")
        deg = g.degenerate.astype(bool)
        for ax, col in ((axes[0], "p_value"), (axes[1], "selection_rate")):
            ax.plot(g.value, g[col], color=COLOUR[model], label=model)
            ax.plot(g.value[~deg], g[col][~deg], MARKER[model], color=COLOUR[model], ms=6)
            ax.plot(g.value[deg], g[col][deg], MARKER[model], mfc=SURFACE, mec=COLOUR[model],
                    ms=6)  # fmt: skip
    axes[0].axhline(0.05, color="#e34948", lw=1, ls="--")
    axes[0].text(20, 0.07, "5% line", color="#e34948", fontsize=9, ha="right")
    axes[0].set(ylabel="p-value, statistical parity", ylim=(-0.03, 1.03),
                xlabel="priors set to this value for everyone")  # fmt: skip
    axes[1].set(ylabel="share of defendants flagged", ylim=(0, 1.05),
                xlabel="priors set to this value for everyone")  # fmt: skip
    axes[0].set_title("Parity is only reached where…", fontsize=11)
    axes[1].set_title("…everyone gets the same decision (hollow)", fontsize=11)
    axes[0].legend(loc="upper center")
    for ax in axes:
        ax.set_xticks([0, 1, 2, 3, 4, 5, 7, 10, 15, 20])
    title(
        fig,
        "Step 2: no value of priors makes a model fair without treating everyone alike",
        "Fairness partial dependence (FPDP), holdout, t = 0.5, African-American vs Caucasian",
    )
    fig.tight_layout()
    save(fig, "09_fairness_step2_fpdp.png")


def fig_mitigation() -> None:
    perf = read("performance.csv")
    fair = fairness_rows(["race_aware", "priors_blind", "priors_set_1"])
    variants = [
        ("race_aware", "Baseline (all features)", "o"),
        ("priors_blind", "Drop priors, retrain", "s"),
        ("priors_set_1", "Set priors to 1", "^"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for model in MODELS:
        pts = []
        for fs, _, marker in variants:
            auc = perf_row(perf, model, fs=fs)["auc"]
            gap = fair.query("model == @model and feature_set == @fs and "
                             "metric == 'statistical_parity'").iloc[0].gap  # fmt: skip
            pts.append((auc, gap))
            ax.plot(gap, auc, marker, color=COLOUR[model], ms=10, mec=SURFACE, zorder=3)
        for (a0, g0), (a1, g1) in zip(pts[:1] * 2, pts[1:]):
            ax.annotate("", (g1, a1), (g0, a0), arrowprops={"arrowstyle": "->",
                        "color": COLOUR[model], "lw": 1.2, "alpha": 0.7})  # fmt: skip
        ax.plot([], [], "-", color=COLOUR[model], label=model)
    tool = perf_row(perf, "COMPAS tool")
    tool_gap = (
        fairness_rows(["race_aware"])
        .query("model == 'COMPAS tool' and metric == 'statistical_parity'")
        .iloc[0]
        .gap
    )
    ax.plot(tool_gap, tool["auc"], "^", color=MUTED, ms=10)
    ax.annotate("COMPAS tool", (tool_gap, tool["auc"]), xytext=(8, -12),
                textcoords="offset points", color=TEXT_2)  # fmt: skip
    for _, name, marker in variants:
        ax.plot([], [], marker, color=TEXT_2, ms=8, label=name)
    ax.axvline(0, color=TEXT_2, lw=1, ls="--")
    ax.set(xlabel="flag-rate gap, African-American − Caucasian (0 = parity)",
           ylabel="AUC")  # fmt: skip
    ax.legend(loc="center right", ncol=2)
    title(
        fig,
        "Step 3: fairness is bought with accuracy",
        "Two fixes on priors, holdout, t = 0.5. Arrows go from each model's baseline to its fix.",
    )
    fig.tight_layout()
    save(fig, "10_fairness_step3_mitigation.png")


# ============================================================================== trade-offs


def fig_cost() -> None:
    perf = read("performance.csv")
    variants = ["race_aware", "race_priors_blind", "priors_set_1"]
    test_y = build_dataset("race_aware").y.loc[
        read("predictions.csv").query("model == 'LogReg' and run == 'holdout' "
                                      "and feature_set == 'race_aware'").row]  # fmt: skip
    detain_all = (1 - test_y.mean()) * CONFIG.costs.c_fp
    release_all = test_y.mean() * CONFIG.costs.c_fn
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(variants))
    for k, model in enumerate(MODELS):
        vals = [perf_row(perf, model, fs=fs, threshold=BREAK_EVEN)["cost_per_defendant"]
                for fs in variants]  # fmt: skip
        bars = ax.bar(x + (k - 1) * 0.26, np.array(vals) / 1000, 0.24, color=COLOUR[model],
                      label=model, zorder=3)  # fmt: skip
        ax.bar_label(bars, [f"${v / 1000:.2f}k" for v in vals], padding=2, fontsize=8.5,
                     color=TEXT_2)  # fmt: skip
    tool = perf_row(perf, "COMPAS tool")["cost_per_defendant"]
    for value, name, ls in ((detain_all, "Detain everyone", "--"), (tool, "COMPAS tool", ":")):
        ax.axhline(value / 1000, color=TEXT_2, lw=1.2, ls=ls)
        ax.text(len(variants) - 0.5, value / 1000 + 0.08, f"{name}: ${value / 1000:.2f}k",
                ha="right", fontsize=9, color=TEXT_2)  # fmt: skip
    ax.set_xticks(x, ["All features", "No race, no priors", "Priors set to 1"])
    ax.set(ylabel="expected cost per defendant ($k)", ylim=(6, 10))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3)
    title(
        fig,
        "Trade-off: removing the race proxy costs money",
        f"Cost at the break-even threshold {BREAK_EVEN:.3f} ($40k per missed re-offence, "
        "$13.5k per unneeded detention; assumptions). "
        f"Release everyone: ${release_all / 1000:.1f}k.",
    )
    fig.tight_layout()
    save(fig, "11_tradeoff_cost.png")


def main() -> int:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    for fig in (fig_data, fig_radar, fig_xper_person, fig_marginal_effects, fig_pdp,
                fig_stability, fig_auc_by_design, fig_fairness_step1, fig_fpdp, fig_mitigation,
                fig_cost):  # fmt: skip
        try:
            fig()
        except FileNotFoundError as missing:  # run_comparison.py has not written it yet
            print(f"  skipped {fig.__name__}: {missing.filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
