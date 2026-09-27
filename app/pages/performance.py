import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for lib

import lib  # noqa: E402

lib.setup_page(
    "Performance",
    "How well each model predicts two-year recidivism, at any threshold, on any design. "
    "Every threshold-dependent number is recomputed live from the per-defendant test scores.",
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

# ----------------------------------------------------------------------------- constants

RADAR_METRICS = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
RATE_METRICS = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "FPR", "FNR", "Flag rate"]
PCT_COLS = RATE_METRICS
GRID_T = np.round(np.arange(0.02, 0.9801, 0.005), 3)
HOLDOUT_SETS = list(lib.FEATURE_SETS)
XPER_FILES = {
    "LogReg": (lib.LOGREG_ART / "xper_auc.csv", lib.LOGREG_ART / "xper_auc_individual.csv"),
    "XGBoost": (
        lib.XGBOOST_ART / "xper_race_aware.csv",
        lib.XGBOOST_ART / "xper_race_aware_individual.csv",
    ),
    "TabPFN": (
        lib.TABPFN_ART / "analysis" / "t20_xper_individual" / "xper_auc.csv",
        lib.TABPFN_ART / "analysis" / "t20_xper_individual" / "xper_auc_individual.csv",
    ),
}
DEFAULT_DEFENDANT = 5069


def valid_sets(run: str) -> list[str]:
    return HOLDOUT_SETS if run == "holdout" else ["race_aware"]


def ordered(models) -> list[str]:
    """Selected models in the canonical order, so colours and positions never shuffle."""
    return [m for m in lib.ALL_MODELS if m in models]


def line_style(model: str, width: float = 2.5) -> dict:
    return {"color": lib.COLOUR[model], "dash": lib.DASH[model], "width": width}


def hex_rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------- computations


@st.cache_data
def metric_table(run: str, feature_set: str, threshold: float) -> pd.DataFrame:
    """Every metric for every model (and the COMPAS tool) at one threshold."""
    rows = {}
    for m in lib.ALL_MODELS:
        s = lib.scores(m, run, feature_set)
        rows[m] = lib.metrics(s.y, s.score, threshold if m != "COMPAS tool" else 0.5)
    return pd.DataFrame(rows).T


@st.cache_data
def sweep(run: str, feature_set: str) -> pd.DataFrame:
    """Metrics on a fine threshold grid, vectorised, for the three score models."""
    frames = []
    for m in lib.MODELS:
        s = lib.scores(m, run, feature_set)
        y = s.y.to_numpy() == 1
        flag = s.score.to_numpy()[None, :] >= GRID_T[:, None]
        tp = (flag & y).sum(1)
        fp = (flag & ~y).sum(1)
        fn, tn = y.sum() - tp, (~y).sum() - fp
        n = len(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
            recall = tp / (tp + fn)
            f1 = np.where(tp > 0, 2 * precision * recall / (precision + recall), 0.0)
        frames.append(
            pd.DataFrame(
                {
                    "model": m,
                    "threshold": GRID_T,
                    "Accuracy": (tp + tn) / n,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "Specificity": tn / (tn + fp),
                    "FPR": fp / (fp + tn),
                    "FNR": fn / (fn + tp),
                    "Flag rate": (tp + fp) / n,
                    "Cost per defendant": (fn * lib.C_FN + fp * lib.C_FP) / n,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def roc_curves(run: str, feature_set: str) -> dict:
    out = {}
    for m in lib.MODELS:
        s = lib.scores(m, run, feature_set)
        out[m] = lib.roc(s.y, s.score)
    return out


@st.cache_data
def calibration(run: str, feature_set: str, bins: int = 10) -> pd.DataFrame:
    frames = []
    edges = np.linspace(0, 1, bins + 1)
    for m in lib.MODELS:
        s = lib.scores(m, run, feature_set)
        b = pd.cut(s.score, edges, include_lowest=True, labels=False)
        g = s.groupby(b).agg(predicted=("score", "mean"), observed=("y", "mean"), n=("y", "size"))
        frames.append(g.assign(model=m, bin=g.index))
    return pd.concat(frames, ignore_index=True)


def min_costs(sw: pd.DataFrame) -> dict:
    """Each model's cheapest threshold on the grid: {model: (threshold, cost)}."""
    out = {}
    for m in lib.MODELS:
        d = sw[sw.model == m]
        i = d["Cost per defendant"].idxmin()
        out[m] = (float(d.at[i, "threshold"]), float(d.at[i, "Cost per defendant"]))
    return out


def trio(values, fmt: str = "{:.1%}") -> str:
    """LogReg/XGBoost/TabPFN values joined with slashes."""
    return "/".join(fmt.format(v) for v in values)


@st.cache_data
def design_table(threshold: float, dimension: str) -> pd.DataFrame:
    """Live metrics for every design (race_aware) or every holdout feature set."""
    combos = (
        [(r, "race_aware") for r in lib.RUNS]
        if dimension == "run"
        else [("holdout", fs) for fs in HOLDOUT_SETS]
    )
    rows = []
    for run, fs in combos:
        for m in lib.ALL_MODELS:
            s = lib.scores(m, run, fs)
            met = lib.metrics(s.y, s.score, threshold if m != "COMPAS tool" else 0.5)
            rows.append(
                {"model": m, "run": run, "feature_set": fs, "n": len(s), "base_rate": s.y.mean()}
                | met
            )
    return pd.DataFrame(rows)


@st.cache_data
def xper_global() -> pd.DataFrame:
    frames = []
    for m, (path, _) in XPER_FILES.items():
        d = lib.read_csv(str(path))
        if "feature_set" in d.columns:
            d = d[d.feature_set == "race_aware"][["feature", "contribution"]]
        else:
            d = d.set_axis(["feature", "contribution"], axis=1)
        frames.append(d.assign(model=m))
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def xper_individual() -> dict[str, pd.DataFrame]:
    return {m: lib.read_csv(str(path), index_col=0) for m, (_, path) in XPER_FILES.items()}


@st.cache_data
def defendant_labels(rows: tuple) -> dict:
    g = lib.groups(rows)
    raw = lib.load_raw().loc[list(rows)]
    out = {}
    for r in rows:
        priors = int(raw.at[r, "Number_of_Priors"])
        outcome = "reoffended" if raw.at[r, "Two_yr_Recidivism"] == 1 else "did not reoffend"
        out[r] = (
            f"#{r} · {g.at[r, 'race']}, {g.at[r, 'sex']}, {g.at[r, 'age_band']}, "
            f"{priors} prior{'s' if priors != 1 else ''}, {g.at[r, 'charge_degree'].lower()}, "
            f"{outcome}"
        )
    return out


# -------------------------------------------------------------------------------- controls

with st.container(border=True):
    threshold = lib.threshold_control(
        "perf",
        help_text="Flag a defendant as high risk when the model's predicted risk is at or "
        "above this value. The COMPAS tool's decision is fixed (medium/high risk).",
    )
    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    with c1:
        run = st.selectbox(
            "Evaluation design",
            list(lib.RUNS),
            format_func=lib.RUNS.get,
            key="perf_run",
            help="Holdout 70/30 is the headline design (n = 1,852).",
        )
    with c2:
        sets = valid_sets(run)
        feature_set = st.selectbox(
            "Feature set",
            sets,
            format_func=lib.FEATURE_SETS.get,
            key=f"perf_fs_{run}",
            disabled=len(sets) == 1,
            help="Feature-set variants exist for the holdout design only.",
        )
    with c3:
        models = ordered(lib.model_picker("perf_models"))

if not models:
    st.info("Pick at least one model.")
    st.stop()

score_models = [m for m in models if m in lib.MODELS]
table = metric_table(run, feature_set, threshold)
y_ref = lib.scores("LogReg", run, feature_set).y
n_test, base_rate = len(y_ref), float(y_ref.mean())

st.markdown(
    f"<p class='small-note'><b>{lib.RUNS[run]}</b> · {lib.FEATURE_SETS[feature_set]} · "
    f"test set n = {n_test:,} · base rate {base_rate:.1%} · threshold {threshold:.3f} · "
    f"costs: missed reoffender ${lib.C_FN:,.0f}, needless detention ${lib.C_FP:,.0f}</p>",
    unsafe_allow_html=True,
)
if run.startswith("temporal"):
    lib.caveat(
        "Temporal designs use the dated cohort, whose base rate is about 33% (vs 45.5% "
        "elsewhere). Compare them only with each other, not with the holdout numbers."
    )
elif run != "holdout":
    lib.caveat(
        "X1 → X3 and X2 → X3 train on a 40% sample and share one 20% test set (X3, n = 1,235): "
        "smaller training data, so expect slightly lower numbers than the holdout."
    )

# ------------------------------------------------------------------------------ KPI row

tool = table.loc["COMPAS tool"]
kpi_cols = st.columns(len(models))
for col, m in zip(kpi_cols, models, strict=True):
    r = table.loc[m]
    is_tool = m == "COMPAS tool"
    with col:
        st.markdown(
            f"<div style='font-weight:650;margin-bottom:0.3rem'>"
            f"<span style='color:{lib.COLOUR[m]}'>●</span> {m}"
            f"{' <span class=small-note>(fixed decision)</span>' if is_tool else ''}</div>",
            unsafe_allow_html=True,
        )
        a, b = st.columns(2)
        a.metric(
            "AUC",
            f"{r['AUC']:.3f}",
            None if is_tool else f"{r['AUC'] - tool['AUC']:+.3f} vs tool",
        )
        b.metric(
            "Accuracy",
            lib.fmt_pct(r["Accuracy"]),
            None if is_tool else f"{(r['Accuracy'] - tool['Accuracy']) * 100:+.1f} pts",
        )
        a, b = st.columns(2)
        a.metric(
            "Recall",
            lib.fmt_pct(r["Recall"]),
            None if is_tool else f"{(r['Recall'] - tool['Recall']) * 100:+.1f} pts",
        )
        b.metric(
            "Cost / defendant",
            lib.fmt_money(r["Cost per defendant"]),
            None if is_tool else f"{r['Cost per defendant'] - tool['Cost per defendant']:+,.0f}",
            delta_color="inverse",
        )

aucs = table.loc[lib.MODELS, "AUC"]
sw_all = sweep(run, feature_set)
mins_all = min_costs(sw_all)
base_all = lib.baselines(y_ref)
cheapest = min(mins_all, key=lambda m: mins_all[m][1])
lib.tldr(
    [
        f"AUC: LogReg {aucs['LogReg']:.3f}, XGBoost {aucs['XGBoost']:.3f}, "
        f"TabPFN {aucs['TabPFN']:.3f} vs COMPAS tool {tool['AUC']:.3f}",
        f"Accuracy at {threshold:.3f} (LogReg/XGB/TabPFN): "
        f"{trio(table.loc[lib.MODELS, 'Accuracy'])} vs tool {tool['Accuracy']:.1%}",
        f"Recall at {threshold:.3f}: {trio(table.loc[lib.MODELS, 'Recall'])} "
        f"vs tool {tool['Recall']:.1%}",
        f"Cheapest: {cheapest} {lib.fmt_money(mins_all[cheapest][1])} @ "
        f"{mins_all[cheapest][0]:.3f}; detain all {lib.fmt_money(base_all['Detain everyone'])}; "
        f"tool {lib.fmt_money(tool['Cost per defendant'])}",
    ],
    f"All three beat the COMPAS tool by {aucs.min() - tool['AUC']:+.3f} AUC or more, "
    f"and are tied with each other (spread {aucs.max() - aucs.min():.3f}).",
)
if "XGBoost" in score_models:
    xs = lib.scores("XGBoost", run, feature_set).score
    at_be = xs.ge(lib.BREAK_EVEN).mean()
    lib.caveat(
        f"XGBoost's scores are compressed (here {xs.min():.2f}–{xs.max():.2f}), so its 0/1 "
        "metrics swing sharply with the threshold: at the 0.252 break-even it flags "
        f"{at_be:.0%} of defendants. Its AUC (ranking quality) is unaffected."
    )

# ---------------------------------------------------------------------------------- tabs

tab_over, tab_curves, tab_conf, tab_designs, tab_xper = st.tabs(
    ["Overview", "Curves", "Confusion & calibration", "Across designs", "XPER"]
)

# ---------------------------------------------------------------------------- Overview
with tab_over:
    perf_all = lib.comparison("performance.csv")
    ci_half = (
        perf_all[
            (perf_all.run == run)
            & (perf_all.feature_set == feature_set)
            & (perf_all.threshold == 0.5)
            & perf_all.model.isin(lib.MODELS)
        ]
        .eval("(auc_high - auc_low) / 2")
        .mean()
    )
    f1s = table.loc[lib.MODELS, "F1"].astype(float)
    accs = table.loc[lib.MODELS, "Accuracy"].astype(float)
    lib.tldr(
        [
            f"Radar and table: every metric at threshold {threshold:.3f}, plus AUC 95% CIs",
            f"Spread across the three models: AUC {aucs.max() - aucs.min():.3f}, "
            f"F1 {f1s.max() - f1s.min():.3f}, accuracy {(accs.max() - accs.min()) * 100:.1f} pts",
            f"Best F1: {f1s.idxmax()} {f1s.max():.3f}; COMPAS tool {tool['F1']:.3f}",
        ],
        f"Model differences are far inside the AUC confidence interval (±{ci_half:.3f}).",
    )
    left, right = st.columns([1.1, 1])
    with left:
        lo, hi = st.slider(
            "Radial range", 0.0, 1.0, (0.5, 0.8), 0.05, key="perf_radar_range",
            help="Values outside the range are pinned to its edge (hover shows the true value).",
        )  # fmt: skip
        fig = go.Figure()
        theta = [*RADAR_METRICS, RADAR_METRICS[0]]
        for m in models:
            vals = table.loc[m, RADAR_METRICS].astype(float).fillna(0).to_numpy()
            vals = np.append(vals, vals[0])
            fig.add_trace(
                go.Scatterpolar(
                    r=np.clip(vals, lo, hi),
                    theta=theta,
                    name=m,
                    customdata=vals,
                    mode="lines+markers",
                    line=line_style(m),
                    marker={"symbol": lib.SYMBOL[m], "size": 10, "color": lib.COLOUR[m]},
                    fill="toself",
                    fillcolor=hex_rgba(lib.COLOUR[m], 0.05),
                    hovertemplate=f"<b>{m}</b><br>%{{theta}}: %{{customdata:.3f}}<extra></extra>",
                )
            )
        fig.update_layout(
            polar={
                "radialaxis": {
                    "range": [lo, hi],
                    "dtick": 0.1,
                    "tickformat": ".1f",
                    "gridcolor": lib.GRID,
                    "angle": 90,
                    "tickangle": 90,
                },
                "angularaxis": {"gridcolor": lib.GRID, "rotation": 90, "direction": "clockwise"},
            }  # fmt: skip
        )
        lib.show(lib.style(fig, f"Five headline metrics at threshold {threshold:.3f}", 480))
    with right:
        best_f1 = table.loc[score_models or models, "F1"].astype(float).idxmax()
        st.markdown("&nbsp;")
        lib.takeaway(
            "The three model shapes almost coincide: same ranking power, similar trade-off "
            f"between catching reoffenders (recall) and being right when flagging (precision). "
            f"Best F1 here: <b>{best_f1}</b> ({table.loc[best_f1, 'F1']:.3f})."
        )
        lib.caveat(
            "The COMPAS tool is a fixed yes/no decision, so its AUC is computed from a single "
            "cut-off and does not move with the threshold."
        )

    st.subheader("All metrics")
    perf = lib.comparison("performance.csv")
    ci = (
        perf[(perf.run == run) & (perf.feature_set == feature_set) & (perf.threshold == 0.5)]
        .set_index("model")[["auc_low", "auc_high"]]
        .reindex(models)
    )
    full = table.loc[models].copy()
    full.insert(1, "AUC 95% CI", [f"{a:.3f}–{b:.3f}" for a, b in ci.to_numpy()])
    for c in ["TP", "FP", "TN", "FN"]:
        full[c] = full[c].astype(int)
    fmt = {"AUC": "{:.3f}", "Cost per defendant": "${:,.0f}"} | dict.fromkeys(PCT_COLS, "{:.1%}")
    st.dataframe(
        full.style.format(fmt, na_rep="–"),
        width="stretch",
    )
    st.caption(
        f"Models at threshold {threshold:.3f}; COMPAS tool = its own medium/high-risk flag. "
        "AUC 95% CI from 1,000 bootstrap resamples of the test set (comparison/artifacts). "
        "Cost per defendant = (FN × $40,000 + FP × $13,500) / n."
    )

# ------------------------------------------------------------------------------ Curves
with tab_curves:
    sw = sweep(run, feature_set)
    rocs = roc_curves(run, feature_set)
    at_tool = {m: float(np.interp(tool["FPR"], rocs[m].FPR, rocs[m].TPR)) for m in lib.MODELS}
    at_half = metric_table(run, feature_set, 0.5)
    saving = np.mean([at_half.loc[m, "Cost per defendant"] - mins_all[m][1] for m in lib.MODELS])
    t_opt = np.mean([mins_all[m][0] for m in lib.MODELS])
    lib.tldr(
        [
            f"At the tool's false-positive rate ({tool['FPR']:.0%}), models catch "
            f"{min(at_tool.values()):.0%}–{max(at_tool.values()):.0%} vs tool {tool['Recall']:.0%}",
            "Min cost: "
            + ", ".join(
                f"{m} {lib.fmt_money(c)} @{t:.3f}"
                for m, (t, c) in sorted(mins_all.items(), key=lambda kv: kv[1][1])
            ),
            f"Detain everyone {lib.fmt_money(base_all['Detain everyone'])}, release everyone "
            f"{lib.fmt_money(base_all['Release everyone'])}, tool "
            f"{lib.fmt_money(tool['Cost per defendant'])}",
        ],
        f"Threshold beats model choice: 0.5 → ~{t_opt:.2f} saves "
        f"{lib.fmt_money(saving)} per defendant on average.",
    )

    # ROC
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                   line={"color": lib.MUTED, "dash": "dot", "width": 1}, hoverinfo="skip")
    )  # fmt: skip
    for m in score_models:
        rc = rocs[m]
        fig.add_trace(
            go.Scatter(
                x=rc.FPR, y=rc.TPR, mode="lines", line=line_style(m, 2),
                name=f"{m} (AUC {table.loc[m, 'AUC']:.3f})", legendgroup=m,
                customdata=rc.threshold,
                hovertemplate=f"<b>{m}</b><br>FPR %{{x:.3f}} · TPR %{{y:.3f}}"
                "<br>threshold %{customdata:.3f}<extra></extra>",
            )
        )  # fmt: skip
        fig.add_trace(
            go.Scatter(
                x=[table.loc[m, "FPR"]], y=[table.loc[m, "Recall"]], mode="markers",
                marker={"symbol": lib.SYMBOL[m], "size": 13, "color": lib.COLOUR[m],
                        "line": {"color": "white", "width": 2}},
                legendgroup=m, showlegend=False,
                hovertemplate=f"<b>{m} at {threshold:.3f}</b><br>FPR %{{x:.3f}} · "
                "TPR %{y:.3f}<extra></extra>",
            )
        )  # fmt: skip
    if "COMPAS tool" in models:
        fig.add_trace(
            go.Scatter(
                x=[0, tool["FPR"], 1], y=[0, tool["Recall"], 1], mode="lines+markers",
                line=line_style("COMPAS tool", 2),
                marker={"symbol": ["circle", lib.SYMBOL["COMPAS tool"], "circle"],
                        "size": [0, 14, 0], "color": lib.COLOUR["COMPAS tool"]},
                name=f"COMPAS tool (AUC {tool['AUC']:.3f})",
                hovertemplate="<b>COMPAS tool</b><br>FPR %{x:.3f} · TPR %{y:.3f}<extra></extra>",
            )
        )  # fmt: skip
    fig.update_xaxes(title="False positive rate (non-reoffenders flagged)", range=[0, 1])
    fig.update_yaxes(title="True positive rate (reoffenders caught)", range=[0, 1.01])
    c_roc, c_txt = st.columns([1.5, 1])
    with c_roc:
        lib.show(lib.style(fig, "ROC curves, with each model's operating point", 500))
    with c_txt:
        st.markdown("&nbsp;")
        lib.takeaway(
            "The three ROC curves lie on top of each other, and all sit above the COMPAS "
            "tool's single point: at the tool's false-positive rate, each model catches more "
            "reoffenders."
        )
        st.markdown(
            "<p class='small-note'>Large markers = where each model operates at the chosen "
            "threshold. Moving the threshold slides the marker along the curve; the curve "
            "itself (and the AUC) does not change.</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Metric vs threshold
    metric = st.selectbox(
        "Metric vs threshold",
        ["F1", "Precision", "Recall", "Accuracy", "Specificity", "FPR", "FNR", "Flag rate"],
        key="perf_sweep_metric",
    )
    fig = go.Figure()
    for m in score_models:
        d = sw[sw.model == m]
        fig.add_trace(
            go.Scatter(
                x=d.threshold, y=d[metric], name=m, mode="lines", line=line_style(m),
                hovertemplate=f"<b>{m}</b><br>threshold %{{x:.3f}}<br>{metric} "
                "%{y:.1%}<extra></extra>",
            )
        )  # fmt: skip
    if "COMPAS tool" in models:
        fig.add_hline(
            y=float(tool[metric]), line=line_style("COMPAS tool", 2),
            annotation_text=f"COMPAS tool {tool[metric]:.1%}", annotation_position="top left",
        )  # fmt: skip
    fig.add_vline(x=threshold, line={"color": lib.INK_2, "width": 1, "dash": "dot"})
    if abs(threshold - lib.BREAK_EVEN) > 1e-3:
        fig.add_vline(
            x=lib.BREAK_EVEN, line={"color": lib.MUTED, "width": 1, "dash": "dash"},
            annotation_text="cost break-even", annotation_position="top right",
        )  # fmt: skip
    fig.update_xaxes(title="Decision threshold", range=[0, 1])
    fig.update_yaxes(title=metric, tickformat=".0%", rangemode="tozero")
    lib.show(lib.style(fig, f"{metric} as the threshold moves", 430))
    if "XGBoost" in score_models:
        lib.takeaway(
            "LogReg and TabPFN change gradually with the threshold; XGBoost's curve is "
            "steeper because its scores are packed into a narrow band, so a small threshold "
            "change flips many decisions at once."
        )
    else:
        lib.takeaway(
            "Raising the threshold trades recall for precision: fewer people are flagged, "
            "and those flagged are more likely to reoffend."
        )

    st.divider()

    # Cost vs threshold
    base = lib.baselines(y_ref)
    fig = go.Figure()
    mins = {}
    for m in score_models:
        d = sw[sw.model == m]
        i = d["Cost per defendant"].idxmin()
        mins[m] = (d.at[i, "threshold"], d.at[i, "Cost per defendant"])
        fig.add_trace(
            go.Scatter(
                x=d.threshold, y=d["Cost per defendant"], name=m, mode="lines",
                line=line_style(m), legendgroup=m,
                hovertemplate=f"<b>{m}</b><br>threshold %{{x:.3f}}<br>$%{{y:,.0f}} per "
                "defendant<extra></extra>",
            )
        )  # fmt: skip
        fig.add_trace(
            go.Scatter(
                x=[mins[m][0]], y=[mins[m][1]], mode="markers", legendgroup=m, showlegend=False,
                marker={"symbol": lib.SYMBOL[m], "size": 12, "color": lib.COLOUR[m],
                        "line": {"color": "white", "width": 2}},
                hovertemplate=f"<b>{m} minimum</b><br>threshold %{{x:.3f}}<br>$%{{y:,.0f}}"
                "<extra></extra>",
            )
        )  # fmt: skip
    if "COMPAS tool" in models:
        fig.add_hline(
            y=float(tool["Cost per defendant"]), line=line_style("COMPAS tool", 2),
            annotation_text=f"COMPAS tool {lib.fmt_money(tool['Cost per defendant'])}",
            annotation_position="bottom left",
        )  # fmt: skip
    for name, val in base.items():
        fig.add_hline(
            y=val, line={"color": lib.INK_2, "width": 1, "dash": "dot"},
            annotation_text=f"{name}: {lib.fmt_money(val)}", annotation_position="top right",
        )  # fmt: skip
    fig.add_vline(x=threshold, line={"color": lib.INK_2, "width": 1, "dash": "dot"})
    fig.add_vline(
        x=lib.BREAK_EVEN, line={"color": lib.MUTED, "width": 1, "dash": "dash"},
        annotation_text=f"break-even {lib.BREAK_EVEN:.3f}", annotation_position="bottom right",
    )  # fmt: skip
    fig.update_xaxes(title="Decision threshold", range=[0, 1])
    fig.update_yaxes(title="Expected cost per defendant", tickprefix="$", tickformat=",.0f")
    lib.show(lib.style(fig, "Expected cost per defendant vs threshold", 460))
    if mins:
        parts = ", ".join(
            f"{m} {lib.fmt_money(c)} at {t:.3f}"
            for m, (t, c) in sorted(mins.items(), key=lambda kv: kv[1][1])
        )
        lib.takeaway(
            f"Cheapest thresholds: {parts}. A missed reoffender costs about three times a "
            f"needless detention, so the optimum sits near the break-even "
            f"{lib.BREAK_EVEN:.3f}, far below 0.5."
        )

# ------------------------------------------------------------------ Confusion & calibration
with tab_conf:
    cal_all = calibration(run, feature_set)
    gap = {
        m: float(np.average((d.predicted - d.observed).abs(), weights=d.n))
        for m, d in cal_all.groupby("model")
    }
    qs = {m: lib.scores(m, run, feature_set).score.quantile([0.01, 0.99]) for m in lib.MODELS}
    lib.tldr(
        [
            f"At {threshold:.3f} (LogReg/XGB/TabPFN): recall "
            f"{trio(table.loc[lib.MODELS, 'Recall'], '{:.0%}')}, false-positive rate "
            f"{trio(table.loc[lib.MODELS, 'FPR'], '{:.0%}')}",
            f"Tool: recall {tool['Recall']:.0%}, false-positive rate {tool['FPR']:.0%}",
            f"Calibration gap: {trio([gap[m] * 100 for m in lib.MODELS], '{:.1f}')} pts "
            "(predicted vs observed)",
            f"Middle-98% scores: XGBoost {qs['XGBoost'].iloc[0]:.2f}–{qs['XGBoost'].iloc[1]:.2f}, "
            f"LogReg {qs['LogReg'].iloc[0]:.2f}–{qs['LogReg'].iloc[1]:.2f}",
        ],
        "LogReg and TabPFN probabilities are reliable; XGBoost's are squeezed "
        f"(off by {gap['XGBoost'] * 100:.1f} pts).",
    )
    st.subheader(f"Confusion matrices at threshold {threshold:.3f}")
    fig = make_subplots(rows=1, cols=len(models), subplot_titles=models, horizontal_spacing=0.06)
    for j, m in enumerate(models, start=1):
        r = table.loc[m]
        counts = np.array([[r["TP"], r["FN"]], [r["FP"], r["TN"]]], dtype=float)
        row_pct = counts / counts.sum(axis=1, keepdims=True)
        text = [[f"<b>{int(c):,}</b><br>{p:.0%}" for c, p in zip(cr, pr, strict=True)]
                for cr, pr in zip(counts, row_pct, strict=True)]  # fmt: skip
        fig.add_trace(
            go.Heatmap(
                z=row_pct, x=["Flagged", "Not flagged"], y=["Reoffended", "Did not"],
                text=text, texttemplate="%{text}", textfont={"size": 14},
                colorscale=[[0, "#ffffff"], [1, hex_rgba(lib.COLOUR[m], 0.85)]],
                zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
                hovertemplate="Actual %{y} · predicted %{x}<br>%{text}<extra>" + m + "</extra>",
            ),
            row=1, col=j,
        )  # fmt: skip
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom")
    fig.update_yaxes(title="Actual outcome", row=1, col=1)
    lib.show(lib.style(fig, None, 300))
    st.caption("Counts, with the percentage of each actual-outcome row (row %).")
    lib.takeaway(
        "Top-left = reoffenders caught; bottom-left = people flagged who did not reoffend. "
        f"At {threshold:.3f} the models catch "
        + ", ".join(f"{m} {table.loc[m, 'Recall']:.0%}" for m in models)
        + " of reoffenders."
    )

    st.divider()
    if not score_models:
        st.info("Calibration and score distributions need at least one of the three models.")
    else:
        c_cal, c_hist = st.columns([1, 1.25])
        with c_cal:
            cal = calibration(run, feature_set)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                           line={"color": lib.MUTED, "dash": "dot", "width": 1},
                           hoverinfo="skip")
            )  # fmt: skip
            for m in score_models:
                d = cal[cal.model == m]
                fig.add_trace(
                    go.Scatter(
                        x=d.predicted, y=d.observed, name=m, mode="lines+markers",
                        line=line_style(m, 2), customdata=d.n,
                        marker={"symbol": lib.SYMBOL[m], "size": 9, "color": lib.COLOUR[m]},
                        hovertemplate=f"<b>{m}</b><br>predicted %{{x:.2f}} · observed "
                        "%{y:.2f}<br>n = %{customdata}<extra></extra>",
                    )
                )  # fmt: skip
            fig.update_xaxes(title="Mean predicted risk (10 bins)", range=[0, 1])
            fig.update_yaxes(title="Observed reoffence rate", range=[0, 1])
            lib.show(lib.style(fig, "Calibration (reliability curve)", 460))
            lib.takeaway(
                "Points on the diagonal mean a predicted 60% risk really reoffends 60% of the "
                "time. LogReg and TabPFN track it closely; XGBoost's points cluster in the "
                "middle because it rarely predicts extreme risks."
            )
        with c_hist:
            fig = make_subplots(
                rows=len(score_models), cols=1, shared_xaxes=True, vertical_spacing=0.07,
                subplot_titles=score_models,
            )  # fmt: skip
            for i, m in enumerate(score_models, start=1):
                s = lib.scores(m, run, feature_set)
                for outcome, colour, name in [
                    (0, lib.MUTED, "Did not reoffend"),
                    (1, lib.COLOUR[m], "Reoffended"),
                ]:
                    fig.add_trace(
                        go.Histogram(
                            x=s.score[s.y == outcome], xbins={"start": 0, "end": 1, "size": 0.02},
                            name=f"{m}: {name}", marker_color=colour, opacity=0.6,
                            legendgroup=m,
                            hovertemplate=f"{name}<br>score %{{x}}<br>%{{y}} defendants"
                            "<extra>" + m + "</extra>",
                        ),
                        row=i, col=1,
                    )  # fmt: skip
                fig.add_vline(
                    x=threshold, line={"color": lib.INK_2, "width": 1, "dash": "dot"},
                    row=i, col=1,
                )  # fmt: skip
            fig.update_layout(barmode="overlay", bargap=0.05)
            fig.update_xaxes(range=[0, 1])
            fig.update_xaxes(title="Predicted risk", row=len(score_models), col=1)
            lib.show(lib.style(fig, "Score distributions by actual outcome", 460))
            spreads = {
                m: lib.scores(m, run, feature_set).score.quantile([0.01, 0.99]).to_numpy()
                for m in score_models
            }
            lib.takeaway(
                "Middle 98% of scores: "
                + ", ".join(f"{m} {a:.2f}–{b:.2f}" for m, (a, b) in spreads.items())
                + ". Grey = did not reoffend; the dotted line is the threshold."
            )

# ----------------------------------------------------------------------------- Across designs
with tab_designs:
    pa = lib.comparison("performance.csv").query("threshold == 0.5")
    by_run = pa[pa.feature_set == "race_aware"].pivot(index="run", columns="model", values="auc")
    edge = by_run[lib.MODELS].min(axis=1) - by_run["COMPAS tool"]
    hold = pa[pa.run == "holdout"].pivot(index="feature_set", columns="model", values="auc")

    def span(fs: str) -> str:
        return f"{hold.loc[fs, lib.MODELS].min():.3f}–{hold.loc[fs, lib.MODELS].max():.3f}"

    lib.tldr(
        [
            f"AUC over 5 designs: models {by_run[lib.MODELS].min().min():.3f}–"
            f"{by_run[lib.MODELS].max().max():.3f}, tool {by_run['COMPAS tool'].min():.3f}–"
            f"{by_run['COMPAS tool'].max():.3f}",
            f"Weakest model beats the tool by {edge.min():+.3f} to {edge.max():+.3f} AUC",
            f"Holdout AUC: all features {span('race_aware')}, no race {span('race_blind')}, "
            f"no priors {span('priors_blind')}",
        ],
        "The ranking holds in every design; priors, not race, carries the prediction.",
    )
    dim = st.radio(
        "Compare across",
        ["run", "feature_set"],
        format_func={"run": "Evaluation designs (all features)",
                     "feature_set": "Feature sets (holdout)"}.get,
        horizontal=True, key="perf_dim",
    )  # fmt: skip
    labels = lib.RUNS if dim == "run" else lib.FEATURE_SETS
    perf = lib.comparison("performance.csv")
    sel = (
        perf[(perf.feature_set == "race_aware") & (perf.threshold == 0.5)]
        if dim == "run"
        else perf[(perf.run == "holdout") & (perf.threshold == 0.5)]
    )
    fig = go.Figure()
    for m in models:
        d = sel[sel.model == m].set_index(dim).reindex(list(labels))
        fig.add_trace(
            go.Scatter(
                x=[labels[k] for k in d.index], y=d.auc, name=m, mode="markers",
                marker={"symbol": lib.SYMBOL[m], "size": 11, "color": lib.COLOUR[m]},
                error_y={"type": "data", "symmetric": False, "array": d.auc_high - d.auc,
                         "arrayminus": d.auc - d.auc_low, "color": lib.COLOUR[m],
                         "thickness": 1.5, "width": 4},
                customdata=np.c_[d.auc_low, d.auc_high],
                hovertemplate=f"<b>{m}</b> · %{{x}}<br>AUC %{{y:.3f}}"
                "<br>95% CI %{customdata[0]:.3f}–%{customdata[1]:.3f}<extra></extra>",
            )
        )  # fmt: skip
    fig.update_layout(scattermode="group", scattergap=0.5)
    fig.update_yaxes(title="AUC (95% bootstrap CI)", tickformat=".2f")
    what = "design" if dim == "run" else "feature set"
    lib.show(lib.style(fig, f"AUC by {what}", 440))
    if dim == "run":
        lib.takeaway(
            "In every design the three models' confidence intervals overlap each other and sit "
            "above the COMPAS tool. Temporal designs score lower for everyone: the later cohort "
            "is harder to predict and has a lower base rate (compare them only with each other)."
        )
    else:
        lib.takeaway(
            "Dropping race barely moves AUC; dropping or neutralising priors costs about 0.1 "
            "AUC and pushes all three models below the COMPAS tool. Priors carries the "
            "prediction."
        )

    st.divider()
    dt = design_table(threshold, dim)
    metric2 = st.selectbox(
        f"Metric at threshold {threshold:.3f}",
        ["Accuracy", "Precision", "Recall", "F1", "FPR", "Flag rate", "Cost per defendant", "AUC"],
        key="perf_design_metric",
    )
    fig = go.Figure()
    for m in models:
        d = dt[dt.model == m].set_index(dim).reindex(list(labels))
        is_money = metric2 == "Cost per defendant"
        fig.add_trace(
            go.Bar(
                x=[labels[k] for k in d.index], y=d[metric2], name=m,
                marker={"color": lib.COLOUR[m] if m != "COMPAS tool" else "white",
                        "line": {"color": lib.COLOUR[m], "width": 1.5}},
                marker_pattern_shape="/" if m == "COMPAS tool" else "",
                marker_pattern_fgcolor=lib.COLOUR[m],
                hovertemplate=f"<b>{m}</b> · %{{x}}<br>{metric2} "
                + ("$%{y:,.0f}" if is_money else "%{y:.3f}") + "<extra></extra>",
            )
        )  # fmt: skip
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    if metric2 == "Cost per defendant":
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    elif metric2 != "AUC":
        fig.update_yaxes(tickformat=".0%")
    fig.update_yaxes(title=metric2)
    lib.show(lib.style(fig, f"{metric2} by {what} (computed live)", 420))
    lib.takeaway(
        "The ranking between models changes little across "
        f"{'designs' if dim == 'run' else 'feature sets'}; differences here are mostly about "
        "where each model's scores sit relative to the threshold, not about ranking skill."
    )
    with st.expander("Table"):
        wide = dt.assign(label=dt[dim].map(labels)).pivot_table(
            index="label", columns="model", values=metric2, sort=False
        )[models]
        base_info = dt.groupby(dim, sort=False)[["n", "base_rate"]].first()
        base_info.index = base_info.index.map(labels)
        wide = base_info.join(wide)
        vfmt = "${:,.0f}" if metric2 == "Cost per defendant" else "{:.3f}"
        st.dataframe(
            wide.style.format({"n": "{:,}", "base_rate": "{:.1%}"} | dict.fromkeys(models, vfmt)),
            width="stretch",
        )

# -------------------------------------------------------------------------------------- XPER
with tab_xper:
    xg0 = xper_global().pivot_table(index="model", columns="feature", values="contribution")
    explained = xg0.drop(columns="benchmark").sum(axis=1)
    share = xg0["Number_of_Priors"] / explained
    lib.tldr(
        [
            "Each feature's contribution to AUC, globally and for one defendant",
            f"Priors: {share.min():.0%}–{share.max():.0%} of explained AUC in all three models",
            "African-American dummy: "
            + ", ".join(f"{m} {xg0.at[m, 'African_American']:+.3f}" for m in XPER_FILES)
            + " AUC",
        ],
        "All three lean on priors, then age; only LogReg visibly uses race.",
    )
    st.markdown(
        "<p class='small-note'>XPER splits each model's AUC into a benchmark (the AUC of a "
        "model that knows nothing, about 0.5) plus one contribution per feature. Holdout, all "
        "features. Computed on each group's own XPER sample, so totals differ slightly.</p>",
        unsafe_allow_html=True,
    )
    xg = xper_global()
    xm = [m for m in models if m in XPER_FILES] or list(XPER_FILES)
    feats = xg[xg.feature != "benchmark"]
    order = feats.groupby("feature").contribution.mean().sort_values(ascending=True).index.tolist()
    fig = go.Figure()
    for m in xm:
        d = feats[feats.model == m].set_index("feature").reindex(order)
        fig.add_trace(
            go.Bar(
                y=[lib.FEATURES[f] for f in order], x=d.contribution, name=m, orientation="h",
                marker_color=lib.COLOUR[m],
                hovertemplate=f"<b>{m}</b> · %{{y}}<br>%{{x:+.4f}} AUC<extra></extra>",
            )
        )  # fmt: skip
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05)
    fig.update_xaxes(title="Contribution to AUC (AUC points)", tickformat="+.2f")
    c_g, c_t = st.columns([1.6, 1])
    with c_g:
        lib.show(lib.style(fig, "Global XPER: what each feature adds to AUC", 520))
    with c_t:
        summary = (
            xg.pivot_table(index="model", columns="feature", values="contribution")
            .reindex(xm)
            .assign(**{"Features total": lambda t: t.drop(columns="benchmark").sum(axis=1)})
        )
        summary = pd.DataFrame(
            {
                "Benchmark": summary["benchmark"],
                "Priors": summary["Number_of_Priors"],
                "African-American": summary["African_American"],
                "All features": summary["Features total"],
                "AUC (XPER sample)": summary["benchmark"] + summary["Features total"],
            }
        )
        st.markdown("&nbsp;")
        st.dataframe(summary.style.format("{:.3f}"), width="stretch")
        pri = summary["Priors"] / summary["All features"]
        lib.takeaway(
            "Priors share of explained AUC: "
            + ", ".join(f"{m} {v:.0%}" for m, v in pri.items())
            + "."
        )
        if "LogReg" in summary.index:
            lib.caveat(
                "LogReg gives the African-American dummy a visible share of AUC "
                f"({summary.at['LogReg', 'African-American']:+.3f}), the tree and foundation "
                "models almost none. LogReg's XPER sample has a higher total, so compare "
                "shapes, not totals."
            )

    st.divider()
    st.subheader("One defendant")
    ind = xper_individual()
    rows = tuple(int(r) for r in ind["LogReg"].index)
    labels_d = defendant_labels(rows)
    row = st.selectbox(
        "Defendant (400 shared XPER defendants from the holdout test set)",
        rows,
        index=rows.index(DEFAULT_DEFENDANT) if DEFAULT_DEFENDANT in rows else 0,
        format_func=labels_d.get,
        key="perf_xper_row",
    )
    preds = lib.predictions()
    here = preds[
        (preds.row == row) & (preds.run == "holdout") & (preds.feature_set == "race_aware")
    ]
    risk = here.set_index("model").score
    tool_flag = int(here.compas_tool.iloc[0])
    cols = st.columns(len(XPER_FILES) + 1)
    for c, m in zip(cols, XPER_FILES, strict=False):
        c.metric(
            f"{m} predicted risk",
            f"{risk[m]:.3f}",
            "flagged" if risk[m] >= threshold else "not flagged",
            delta_color="off",
        )
    cols[-1].metric("COMPAS tool", "Flagged" if tool_flag else "Not flagged")

    fig = go.Figure()
    feat_cols = list(lib.FEATURES)
    for m in XPER_FILES:
        v = ind[m].loc[row, feat_cols]
        fig.add_trace(
            go.Bar(
                y=[lib.FEATURES[f] for f in feat_cols], x=v.to_numpy(), name=m, orientation="h",
                marker_color=lib.COLOUR[m],
                hovertemplate=f"<b>{m}</b> · %{{y}}<br>%{{x:+.4f}}<extra></extra>",
            )
        )  # fmt: skip
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Contribution to this defendant's share of AUC", tickformat="+.2f")
    lib.show(lib.style(fig, f"XPER for defendant #{row}", 480))
    agree = {m: ind[m].loc[row, feat_cols].abs().idxmax() for m in XPER_FILES}
    tops = ", ".join(f"{m}: {lib.FEATURES[f]}" for m, f in agree.items())
    lib.takeaway(
        f"Largest driver for this defendant — {tops}. Positive bars help the model rank this "
        "person correctly relative to others; negative bars push it the wrong way. Averaged over "
        "all defendants they give the global chart above."
    )
    bench = {m: ind[m].at[row, "benchmark"] for m in XPER_FILES}
    st.caption(
        "Benchmark term for this defendant: "
        + ", ".join(f"{m} {b:.3f}" for m, b in bench.items())
        + ". Always shows the three models, holdout, all features."
    )
