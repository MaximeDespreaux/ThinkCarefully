import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for lib

import lib  # noqa: E402

lib.setup_page(
    "Stability",
    "Refit each model on different data and measure how far the two fits drift apart: "
    "their decisions, their metrics, their score distributions and their explanations. "
    "All features (race-aware). Every threshold-dependent number is recomputed live.",
)

# ------------------------------------------------------------------------------ vocabulary

PAIRS = {"X1 vs X2": ("X1_to_X3", "X2_to_X3"), "time 1 vs 2": ("temporal_1", "temporal_2")}
PAIR_LABEL = {
    "X1 vs X2": "Design 1 · two random 40% samples, same test set",
    "time 1 vs 2": "Design 2 · two time windows",
}
PAIR_SHORT = {"X1 vs X2": "Random samples (X1 vs X2)", "time 1 vs 2": "Over time (window 1 vs 2)"}
FIT = {
    "X1_to_X3": "X1 fit",
    "X2_to_X3": "X2 fit",
    "temporal_1": "Window 1",
    "temporal_2": "Window 2",
    "holdout": "Holdout",
}
METRICS = ["AUC", "Accuracy", "Precision", "Recall", "F1", "FPR", "Flag rate"]
ME_BAR = "Marginal effects (L2)"
TINY = lib.SMALL_GROUPS
N_BINS = 10
EXTRA_FS = {
    **lib.FEATURE_SETS,
    "sex_blind": "No sex",
    "age_blind": "No age",
    "protected_blind": "No race, sex, age",
    "race_proxy_blind": "No race, no proxies",
}
TABPFN_DIR = lib.TABPFN_ART / "analysis"


def tint(colour: str, alpha: float) -> str:
    h = colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def pct(x: float, digits: int = 1) -> str:
    return "–" if pd.isna(x) else f"{x:.{digits}%}"


# ----------------------------------------------------------------------------- computation


@st.cache_data
def me_table() -> pd.DataFrame:
    me = lib.comparison("marginal_effects.csv")
    return me[me.feature_set == "race_aware"]


def me_distance(model: str, pair: str, drop_tiny: bool = False) -> dict:
    """L2 distance and Spearman rank agreement (|effect|) between the two fits' effects."""
    if model not in lib.MODELS:
        return {"ME L2": float("nan"), "ME Spearman": float("nan")}
    a, b = PAIRS[pair]
    me = me_table()
    ea = me[(me.model == model) & (me.run == a)].set_index("feature").marginal_effect
    eb = me[(me.model == model) & (me.run == b)].set_index("feature").marginal_effect
    if drop_tiny:
        ea = ea.drop([f for f in TINY if f in ea.index])
    eb = eb.loc[ea.index]
    return {
        "ME L2": float(np.linalg.norm(ea - eb)),
        "ME Spearman": float(spearmanr(ea.abs(), eb.abs()).statistic),
    }


def paired(model: str) -> pd.DataFrame:
    """X1-fit and X2-fit scores for the same 1,235 test defendants."""
    sa, sb = lib.scores(model, "X1_to_X3"), lib.scores(model, "X2_to_X3")
    return pd.DataFrame({"y": sa.y, "a": sa.score, "b": sb.score.loc[sa.index]})


@st.cache_data
def live_stability(models: tuple, threshold: float) -> pd.DataFrame:
    """Every stability number of both designs, at one threshold."""
    rows = []
    for model in models:
        for pair, (a, b) in PAIRS.items():
            sa, sb = lib.scores(model, a), lib.scores(model, b)
            ma = lib.metrics(sa.y, sa.score, threshold)
            mb = lib.metrics(sb.y, sb.score, threshold)
            row = {"model": model, "pair": pair}
            for k in METRICS:
                row[k] = abs(mb[k] - ma[k])
                row[f"{k} a"], row[f"{k} b"] = ma[k], mb[k]
            row.update(me_distance(model, pair))
            if pair == "X1 vs X2":
                p = paired(model)
                flips = (p.a >= threshold) != (p.b >= threshold)
                row["Flip rate"], row["Flips"] = float(flips.mean()), int(flips.sum())
                row["Score RMSE"] = float(np.sqrt(((p.a - p.b) ** 2).mean()))
            row["PSI"] = psi(sa.score.to_numpy(), sb.score.to_numpy())
            rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data
def flip_curve(models: tuple) -> pd.DataFrame:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 3)
    rows = []
    for model in models:
        p = paired(model)
        a, b = p.a.to_numpy()[:, None], p.b.to_numpy()[:, None]
        rates = ((a >= grid) != (b >= grid)).mean(axis=0)
        rows += [{"model": model, "threshold": t, "flip rate": r} for t, r in zip(grid, rates)]
    return pd.DataFrame(rows)


def psi(ref: np.ndarray, new: np.ndarray, bins: int = N_BINS) -> float:
    """Population stability index, bins = quantiles of the reference scores."""
    inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1])
    if len(inner) == 0:
        return float("nan")
    k = len(inner) + 1
    p = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=k) / len(ref)
    q = np.bincount(np.searchsorted(inner, new, side="right"), minlength=k) / len(new)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_label(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    return "stable" if v < 0.1 else "moderate shift" if v < 0.25 else "major shift"


# -------------------------------------------------------------------------------- controls

with st.expander("How the two stability designs work", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        "**Design 1 · random samples.** Two stratified 40% samples of the cohort (X1, X2) "
        "train two fits of the same model; both score the **same** 20% test set X3 "
        "(1,235 defendants). Differences are pure training-sample noise, so we can compare "
        "the two fits **defendant by defendant**: score distance and decision flips."
    )
    c2.markdown(
        "**Design 2 · time.** On the dated cohort (screening dates, ~33% base rate, *not* "
        "comparable with the 45.5% modelling table): window 1 trains on the first 40% by date "
        "and tests on the next 20%; window 2 trains on the first 70% and tests on the last 30%. "
        "Test defendants differ, so we compare **metrics, explanation shapes and score "
        "distributions (PSI)**, not people."
    )
    c3.markdown(
        "**Distance between estimators.** LogReg has coefficients, XGBoost trees, TabPFN no "
        "fitted parameters at all. The common yardstick is each fit's vector of **marginal "
        "effects** (average change in predicted risk when a feature goes 0 → 1, priors +1): "
        "the **L2 distance** between the two fits' vectors and their **Spearman** rank "
        "agreement. The 70/30 holdout is shown for reference."
    )

ctrl1, ctrl2 = st.columns([3, 2], gap="large")
with ctrl1:
    threshold = lib.threshold_control(
        "stab_thr",
        help_text="Every flip rate, recall, flag rate and |Δ| on this page is recomputed "
        "at this threshold. The COMPAS tool is a fixed 0/1 decision, so it ignores it.",
    )
with ctrl2:
    models = lib.model_picker("stab_models")
if not models:
    st.warning("Pick at least one model.")
    st.stop()
models = [m for m in lib.ALL_MODELS if m in models]  # keep the canonical order
fitted = [m for m in models if m in lib.MODELS]
live = live_stability(tuple(models), threshold)

tab_sum, tab_same, tab_time, tab_expl, tab_extra = st.tabs(
    ["Summary", "Same defendants (X1 vs X2)", "Over time", "Explanations", "Group extras"]
)

# ------------------------------------------------------------------------------- summary

with tab_sum:
    st.subheader(f"Decision flips between the X1 and X2 fits at threshold {threshold:.3f}")
    cols = st.columns(len(models))
    for col, model in zip(cols, models):
        r = live[(live.model == model) & (live.pair == "X1 vs X2")].iloc[0]
        col.metric(
            model,
            pct(r["Flip rate"]),
            help="Share of the 1,235 shared test defendants flagged by one fit but not the other.",
        )
        if model == "COMPAS tool":
            col.caption("Does not retrain: 0 flips by construction.")
        else:
            col.caption(f"{r['Flips']:,} of 1,235 defendants · score RMSE {r['Score RMSE']:.3f}")

    picked = st.pills(
        "Metrics",
        METRICS,
        default=["AUC", "Accuracy", "Recall", "F1", "FPR"],
        selection_mode="multi",
        key="stab_sum_metrics",
    )
    picked = [m for m in METRICS if m in (picked or [])]
    cats = [*picked, ME_BAR]
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=[PAIR_LABEL[p] for p in PAIRS],
    )
    for j, pair in enumerate(PAIRS, start=1):
        for model in models:
            r = live[(live.model == model) & (live.pair == pair)].iloc[0]
            vals = [r[k] for k in picked] + [r["ME L2"]]
            detail = [f"{r[f'{k} a']:.3f} → {r[f'{k} b']:.3f}" for k in picked] + [
                "n/a (no marginal effects)"
                if pd.isna(r["ME L2"])
                else f"Spearman rank agreement {r['ME Spearman']:.2f}"
            ]
            fig.add_bar(
                x=cats,
                y=vals,
                name=model,
                legendgroup=model,
                showlegend=j == 1,
                marker={"color": lib.COLOUR[model], "cornerradius": 4},
                customdata=detail,
                hovertemplate=f"<b>{model}</b> · %{{x}}<br>|Δ| = %{{y:.3f}}"
                "<br>%{customdata}<extra></extra>",
                row=1,
                col=j,
            )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title_text="absolute change between the two fits", row=1, col=1)
    fig.update_annotations(font_size=13, xanchor="left", x=0)
    fig.layout.annotations[1].update(x=0.52)
    lib.show(lib.style(fig, height=460))

    fits = live[live.model.isin(fitted)]
    if not fits.empty:
        worst_auc = fits.AUC.max()
        d1 = fits[fits.pair == "X1 vs X2"].set_index("model")
        flip_txt = ", ".join(f"{m} {pct(d1.loc[m, 'Flip rate'])}" for m in d1.index)
        lib.takeaway(
            f"Refitting on different data barely moves the <b>ranking</b>: AUC changes by at "
            f"most {worst_auc:.3f} across models and both designs. What moves is the "
            f"<b>decision</b> of defendants who sit close to the line — at threshold "
            f"{threshold:.3f}, a different random 40% sample flips {flip_txt} of the same "
            f"defendants. Explanations (marginal effects) move by an L2 distance of "
            f"{fits['ME L2'].min():.3f}–{fits['ME L2'].max():.3f}."
        )
    if "XGBoost" in models:
        x = live[(live.model == "XGBoost") & (live.pair == "time 1 vs 2")].iloc[0]
        lib.caveat(
            f"<b>XGBoost's large over-time swings in Recall/F1 are a calibration artefact, not "
            f"instability.</b> Its scores are compressed (0.27–0.73 on the holdout, max ≈ 0.54 "
            f"in the time windows), so at a fixed threshold its flag rate and recall react "
            f"sharply to the lower base rate (here recall {x['Recall a']:.2f} → "
            f"{x['Recall b']:.2f}) while its AUC barely moves ({x['AUC a']:.3f} → "
            f"{x['AUC b']:.3f}). It ranks defendants just as consistently; it just places them "
            f"below 0.5. See the <i>Over time</i> tab."
        )
    if "COMPAS tool" in models:
        st.caption(
            "COMPAS tool: its decision never changes, so on Design 1 every |Δ| is 0; on Design 2 "
            "its |Δ| measures pure population drift between the two test windows — a useful "
            "floor for reading the models' changes."
        )

    with st.expander("Table: every stability number at this threshold"):
        cols_show = ["model", "pair", *METRICS, "ME L2", "ME Spearman"]
        cols_show += ["Flip rate", "Score RMSE", "PSI"]
        tbl = live[cols_show].copy()
        tbl["pair"] = tbl.pair.map(PAIR_SHORT)
        st.dataframe(tbl.style.format(precision=3, na_rep="–"), hide_index=True, width="stretch")
        st.caption(
            "Metric columns are |Δ| between the two fits. PSI: score-distribution shift, "
            f"{N_BINS} quantile bins of the first fit's scores."
        )
        ref = lib.comparison("stability.csv")
        st.caption("Precomputed reference at threshold 0.5 (comparison/artifacts/stability.csv):")
        st.dataframe(ref.style.format(precision=3, na_rep="–"), hide_index=True, width="stretch")

# ----------------------------------------------------------------------- same defendants

with tab_same:
    if not fitted:
        st.info("Pick at least one of LogReg, XGBoost or TabPFN to compare the two fits.")
    else:
        model = st.segmented_control(
            "Model", fitted, default=fitted[0], key="stab_same_model", selection_mode="single"
        )
        model = model or fitted[0]
        p = paired(model)
        fa, fb = p.a >= threshold, p.b >= threshold
        flip = fa != fb
        near = ((p.a + p.b) / 2 - threshold).abs() <= 0.05

        left, right = st.columns([3, 2], gap="large")
        with left:
            fig = go.Figure()
            same = p[~flip]
            fig.add_scatter(
                x=same.a,
                y=same.b,
                mode="markers",
                name="Same decision",
                marker={"color": tint(lib.COLOUR[model], 0.22), "size": 6},
                customdata=np.c_[same.index, same.y],
                hovertemplate="Row %{customdata[0]} · reoffended %{customdata[1]}"
                "<br>X1 fit %{x:.3f} · X2 fit %{y:.3f}<extra>same decision</extra>",
            )
            fl = p[flip]
            fig.add_scatter(
                x=fl.a,
                y=fl.b,
                mode="markers",
                name=f"Decision flips ({int(flip.sum())})",
                marker={
                    "color": lib.COLOUR[model],
                    "size": 9,
                    "symbol": lib.SYMBOL[model],
                    "line": {"color": "white", "width": 1.5},
                },
                customdata=np.c_[fl.index, fl.y],
                hovertemplate="Row %{customdata[0]} · reoffended %{customdata[1]}"
                "<br>X1 fit %{x:.3f} · X2 fit %{y:.3f}<extra>flip</extra>",
            )
            fig.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1, line={"color": lib.MUTED, "dash": "dot"}
            )
            fig.add_vline(threshold, line={"color": lib.INK_2, "dash": "dash", "width": 1})
            fig.add_hline(threshold, line={"color": lib.INK_2, "dash": "dash", "width": 1})
            for x, y, text in [
                (0.03, 0.97, "flagged by X2 only"),
                (0.97, 0.03, "flagged by X1 only"),
            ]:
                fig.add_annotation(
                    x=x, y=y, text=text, showarrow=False, font={"color": lib.MUTED, "size": 11},
                    xanchor="left" if x < 0.5 else "right",
                )  # fmt: skip
            fig.update_xaxes(title_text="Risk score, fit trained on X1", range=[0, 1])
            fig.update_yaxes(
                title_text="Risk score, fit trained on X2",
                range=[0, 1],
                scaleanchor="x",
                scaleratio=1,
            )
            lib.show(lib.style(fig, f"{model}: the same 1,235 defendants scored by two fits", 540))
        with right:
            st.markdown(f"##### {model} at threshold {threshold:.3f}")
            k1, k2 = st.columns(2)
            k1.metric("Decision flips", pct(flip.mean()), help="Share of the 1,235 defendants.")
            k2.metric("Score RMSE", f"{np.sqrt(((p.a - p.b) ** 2).mean()):.3f}")
            k3, k4 = st.columns(2)
            k3.metric("Score correlation", f"{np.corrcoef(p.a, p.b)[0, 1]:.3f}")
            k4.metric("Largest score gap", f"{(p.a - p.b).abs().max():.3f}")
            k5, k6 = st.columns(2)
            k5.metric("Flagged by X1 only", f"{int((fa & ~fb).sum())}")
            k6.metric("Flagged by X2 only", f"{int((~fa & fb).sum())}")
            share_near = (flip & near).sum() / flip.sum() if flip.any() else float("nan")
            st.markdown(
                f"<p class='small-note'>{pct(share_near, 0)} of the flips concern defendants "
                f"whose average score is within ±0.05 of the threshold; only {pct(near.mean(), 0)} "
                f"of all defendants sit there.</p>",
                unsafe_allow_html=True,
            )
        lib.takeaway(
            f"Points on the diagonal get the same score from both fits. The two fits of "
            f"{model} disagree by {np.sqrt(((p.a - p.b) ** 2).mean()):.3f} on average (RMSE), "
            "and the decisions that flip sit in the band around the threshold lines. The "
            f"largest single score gap is {(p.a - p.b).abs().max():.2f}: no defendant jumps "
            "from clearly low to clearly high risk."
        )

        st.subheader("Where flips happen: flip rate across every threshold")
        curve = flip_curve(tuple(fitted))
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06
        )
        bins = np.arange(0, 1.0001, 0.02)
        for m in fitted:
            c = curve[curve.model == m]
            fig.add_scatter(
                x=c.threshold,
                y=c["flip rate"],
                name=m,
                legendgroup=m,
                mode="lines",
                line={"color": lib.COLOUR[m], "width": 2, "dash": lib.DASH[m]},
                hovertemplate=f"<b>{m}</b><br>threshold %{{x:.2f}}<br>flip rate "
                "%{y:.1%}<extra></extra>",
                row=1,
                col=1,
            )
            pm = paired(m)
            mid = ((pm.a + pm.b) / 2).to_numpy()
            hist = np.histogram(mid, bins)[0] / len(mid)
            fig.add_scatter(
                x=(bins[:-1] + bins[1:]) / 2,
                y=hist,
                name=m,
                legendgroup=m,
                showlegend=False,
                mode="lines",
                line={"color": lib.COLOUR[m], "width": 2, "shape": "hvh"},
                hovertemplate=f"<b>{m}</b><br>score ≈ %{{x:.2f}}<br>%{{y:.1%}} of "
                "defendants<extra></extra>",
                row=2,
                col=1,
            )
        fig.add_vline(threshold, line={"color": lib.INK_2, "dash": "dash", "width": 1})
        fig.update_yaxes(title_text="decision flip rate", tickformat=".0%", row=1, col=1)
        fig.update_yaxes(title_text="share of defendants", tickformat=".0%", row=2, col=1)
        fig.update_xaxes(title_text="decision threshold / average score", row=2, col=1)
        lib.show(lib.style(fig, height=520))
        peaks = curve.loc[curve.groupby("model")["flip rate"].idxmax()]
        peak_txt = ", ".join(
            f"{r.model} {r['flip rate']:.1%} at {r.threshold:.2f}" for _, r in peaks.iterrows()
        )
        lib.takeaway(
            f"Flip rates peak where many defendants' scores are packed together ({peak_txt}). "
            "The bottom panel shows why: a threshold that cuts through a dense region of scores "
            "puts many borderline defendants at the mercy of training-sample noise. At the "
            "default 0.5 the scores are thinner, so fewer decisions flip."
        )

# ------------------------------------------------------------------------------ over time

with tab_time:
    st.subheader("AUC with 95% confidence interval, every design")
    perf = lib.comparison("performance.csv")
    perf = perf[(perf.feature_set == "race_aware")].drop_duplicates(["model", "run"])
    runs = list(lib.RUNS)
    base = {r: lib.scores("LogReg", r).y.mean() for r in runs}
    fig = go.Figure()
    offsets = np.linspace(-0.24, 0.24, len(models)) if len(models) > 1 else [0.0]
    for off, m in zip(offsets, models):
        q = perf[perf.model == m].set_index("run").reindex(runs)
        fig.add_scatter(
            x=np.arange(len(runs)) + off,
            y=q.auc,
            mode="markers",
            name=m,
            marker={"color": lib.COLOUR[m], "symbol": lib.SYMBOL[m], "size": 10},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": q.auc_high - q.auc,
                "arrayminus": q.auc - q.auc_low,
                "color": lib.COLOUR[m],
                "thickness": 2,
                "width": 0,
            },
            customdata=np.c_[[lib.RUNS[r] for r in runs], q.auc_low, q.auc_high],
            hovertemplate=f"<b>{m}</b> · %{{customdata[0]}}<br>AUC %{{y:.3f}} "
            "[%{customdata[1]:.3f}, %{customdata[2]:.3f}]<extra></extra>",
        )
    for x0, x1, text in [(0.5, 2.5, "Design 1: random samples"), (2.5, 4.5, "Design 2: time")]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor="#f7f7f5", line_width=0, layer="below")
        fig.add_annotation(
            x=(x0 + x1) / 2, y=1.04, yref="paper", text=text, showarrow=False,
            font={"color": lib.INK_2, "size": 12},
        )  # fmt: skip
    fig.update_xaxes(
        tickvals=list(range(len(runs))),
        ticktext=[f"{lib.RUNS[r]}<br><sup>base rate {base[r]:.1%}</sup>" for r in runs],
        showgrid=False,
    )
    fig.update_yaxes(title_text="AUC (test set)")
    lib.show(lib.style(fig, height=440))
    st.caption(
        "The COMPAS tool's AUC is that of its 0/1 decision. The time windows use the dated "
        "cohort (lower base rate), so compare the time points with each other, not with the "
        "holdout."
    )

    st.subheader(f"Metrics at threshold {threshold:.3f}: window 1 → window 2")
    tmetrics = st.pills(
        "Metrics",
        METRICS,
        default=["AUC", "Recall", "Flag rate", "Accuracy"],
        selection_mode="multi",
        key="stab_time_metrics",
    )
    tmetrics = [m for m in METRICS if m in (tmetrics or [])]
    if not tmetrics:
        st.info("Pick at least one metric.")
    else:
        tl = live[live.pair == "time 1 vs 2"].set_index("model").reindex(models)
        fig = make_subplots(
            rows=1, cols=len(tmetrics), shared_yaxes=True, subplot_titles=tmetrics,
            horizontal_spacing=0.03,
        )  # fmt: skip
        for j, k in enumerate(tmetrics, start=1):
            for m in models:
                a, b = tl.loc[m, f"{k} a"], tl.loc[m, f"{k} b"]
                fig.add_scatter(
                    x=[a, b], y=[m, m], mode="lines", line={"color": tint(lib.COLOUR[m], 0.5),
                    "width": 3}, showlegend=False, hoverinfo="skip", row=1, col=j,
                )  # fmt: skip
                for val, name, filled in [(a, "Window 1", False), (b, "Window 2", True)]:
                    fig.add_scatter(
                        x=[val],
                        y=[m],
                        mode="markers",
                        name=name,
                        legendgroup=name,
                        showlegend=False,
                        marker={
                            "color": lib.COLOUR[m] if filled else "white",
                            "size": 11,
                            "symbol": lib.SYMBOL[m],
                            "line": {"color": lib.COLOUR[m], "width": 2},
                        },
                        hovertemplate=f"<b>{m}</b> · {k}<br>{name}: %{{x:.3f}}<extra></extra>",
                        row=1,
                        col=j,
                    )
        for name, filled in [("Window 1 (first 40% → next 20%)", False),
                             ("Window 2 (first 70% → last 30%)", True)]:  # fmt: skip
            fig.add_scatter(
                x=[None], y=[None], mode="markers", name=name,
                marker={"color": lib.MUTED if filled else "white", "size": 10,
                        "line": {"color": lib.MUTED, "width": 2}},
            )  # fmt: skip
        fig.update_yaxes(autorange="reversed")
        fig.update_annotations(font_size=13)
        lib.show(lib.style(fig, height=120 + 55 * len(models)))

    st.subheader("Score distributions: where each model puts defendants")
    dist_pair = st.segmented_control(
        "Compare",
        ["time 1 vs 2", "X1 vs X2"],
        default="time 1 vs 2",
        format_func=PAIR_SHORT.get,
        key="stab_dist_pair",
    )
    dist_pair = dist_pair or "time 1 vs 2"
    if not fitted:
        st.info("Pick at least one of LogReg, XGBoost or TabPFN.")
    else:
        ra, rb = PAIRS[dist_pair]
        fig = go.Figure()
        for m in fitted:
            for run, side, alpha in [(ra, "negative", 0.25), (rb, "positive", 0.65)]:
                s = lib.scores(m, run).score
                fig.add_violin(
                    x=[m] * len(s),
                    y=s,
                    side=side,
                    name=f"{m} · {FIT[run]}",
                    legendgroup=m,
                    showlegend=False,
                    line={"color": lib.COLOUR[m], "width": 1.5},
                    fillcolor=tint(lib.COLOUR[m], alpha),
                    meanline={"visible": True},
                    points=False,
                    spanmode="hard",
                    width=0.9,
                    scalemode="width",
                    hoveron="violins",
                )
        fig.add_hline(
            threshold,
            line={"color": lib.INK_2, "dash": "dash", "width": 1},
            annotation_text=f"threshold {threshold:.3f}",
            annotation_position="top right",
        )
        fig.update_layout(violinmode="overlay", violingap=0.2)
        fig.update_yaxes(title_text="risk score", range=[0, 1])
        lib.show(
            lib.style(
                fig,
                f"Left half (light): {FIT[ra]} · right half (dark): {FIT[rb]}",
                height=440,
            )
        )

        cols = st.columns(len(fitted))
        for col, m in zip(cols, fitted):
            sa, sb = lib.scores(m, ra).score, lib.scores(m, rb).score
            v = psi(sa.to_numpy(), sb.to_numpy())
            col.metric(
                f"{m} · PSI",
                f"{v:.3f}",
                help=f"Population stability index, {N_BINS} quantile bins of {FIT[ra]}'s scores."
                " < 0.1 stable, 0.1–0.25 moderate, > 0.25 major shift.",
            )
            col.caption(
                f"{psi_label(v)} · flagged at {threshold:.3f}: {pct((sa >= threshold).mean())} "
                f"→ {pct((sb >= threshold).mean())} · max score {sa.max():.2f} → {sb.max():.2f}"
            )
        if "XGBoost" in fitted:
            x1 = lib.scores("XGBoost", "temporal_1")
            x2 = lib.scores("XGBoost", "temporal_2")
            m1, m2 = lib.metrics(x1.y, x1.score, 0.5), lib.metrics(x2.y, x2.score, 0.5)
            lib.caveat(
                f"<b>XGBoost's 'temporal instability' is a calibration artefact.</b> Its scores "
                f"are compressed (0.27–0.73 on the holdout; max {x1.score.max():.2f} and "
                f"{x2.score.max():.2f} in the two windows), so at 0.5 it flags only "
                f"{pct(m1['Flag rate'])} and {pct(m2['Flag rate'])} of defendants and its "
                f"recall collapses to {m1['Recall']:.2f} / {m2['Recall']:.2f}, while its AUC "
                f"barely moves ({m1['AUC']:.3f} → {m2['AUC']:.3f}). The ranking is stable; the "
                f"fixed threshold simply sits above almost all of its scores. Lower the "
                f"threshold and its recall comes back."
            )
        lib.takeaway(
            "Over time the <b>ranking</b> (AUC) holds for every model; the <b>score "
            "distribution</b> is what moves. PSI measures that shift without needing the same "
            "defendants; it matters because a fixed threshold turns a shifted distribution "
            "into a different flag rate. (Both windows come from the dated cohort, base rate "
            "~33%, lower than the 45.5% the holdout models were built on.)"
        )

# --------------------------------------------------------------------------- explanations

with tab_expl:
    if not fitted:
        st.info("Marginal effects exist for LogReg, XGBoost and TabPFN; pick at least one.")
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            epair = st.segmented_control(
                "Design",
                list(PAIRS),
                default="X1 vs X2",
                format_func=PAIR_SHORT.get,
                key="stab_expl_pair",
            )
            epair = epair or "X1 vs X2"
        with c2:
            drop_tiny = st.toggle(
                "Leave out the two tiny groups (Asian, Native American)",
                key="stab_expl_tiny",
                help="~11–31 defendants each: their effects are mostly noise.",
            )
        ra, rb = PAIRS[epair]
        me = me_table()
        feats = me[me.model.isin(fitted) & me.run.isin([ra, rb])]
        if drop_tiny:
            feats = feats[~feats.feature.isin(TINY)]
        order = feats.groupby("feature").marginal_effect.apply(lambda s: s.abs().mean())
        order = order.sort_values(ascending=False).index.tolist()
        label = {f: lib.FEATURES.get(f, f) + (" *" if f in TINY else "") for f in order}

        cols = st.columns(len(fitted))
        for col, m in zip(cols, fitted):
            d = me_distance(m, epair, drop_tiny)
            col.metric(f"{m} · L2 distance", f"{d['ME L2']:.3f}")
            col.caption(f"Spearman rank agreement of |effects|: {d['ME Spearman']:.2f}")

        fig = make_subplots(
            rows=1, cols=len(fitted), shared_yaxes=True, subplot_titles=fitted,
            horizontal_spacing=0.03,
        )  # fmt: skip
        for j, m in enumerate(fitted, start=1):
            q = feats[feats.model == m]
            ea = q[q.run == ra].set_index("feature").reindex(order)
            eb = q[q.run == rb].set_index("feature").reindex(order)
            ylab = [label[f] for f in order]
            for f, a, b in zip(ylab, ea.marginal_effect, eb.marginal_effect):
                fig.add_scatter(
                    x=[a, b], y=[f, f], mode="lines", showlegend=False, hoverinfo="skip",
                    line={"color": tint(lib.COLOUR[m], 0.45), "width": 3}, row=1, col=j,
                )  # fmt: skip
            for e, run, filled in [(ea, ra, False), (eb, rb, True)]:
                fig.add_scatter(
                    x=e.marginal_effect,
                    y=ylab,
                    mode="markers",
                    showlegend=False,
                    marker={
                        "color": lib.COLOUR[m] if filled else "white",
                        "size": 10,
                        "symbol": lib.SYMBOL[m],
                        "line": {"color": lib.COLOUR[m], "width": 2},
                    },
                    customdata=np.c_[e.change.fillna(""), e.se.fillna(np.nan)],
                    hovertemplate=f"<b>{m}</b> · {FIT[run]}<br>%{{y}} (%{{customdata[0]}})"
                    "<br>marginal effect %{x:+.3f} (se %{customdata[1]:.3f})<extra></extra>",
                    row=1,
                    col=j,
                )
            fig.add_vline(0, line={"color": lib.MUTED, "width": 1}, row=1, col=j)
        for name, filled in [(FIT[ra], False), (FIT[rb], True)]:
            fig.add_scatter(
                x=[None], y=[None], mode="markers", name=name,
                marker={"color": lib.MUTED if filled else "white", "size": 10,
                        "line": {"color": lib.MUTED, "width": 2}},
            )  # fmt: skip
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title_text="Δ predicted risk", tickformat="+.2f")
        fig.update_annotations(font_size=13)
        lib.show(
            lib.style(
                fig,
                f"Marginal effects of each fit · {PAIR_SHORT[epair]}",
                height=150 + 38 * len(order),
            )
        )
        if any(f in TINY for f in order):
            st.caption("* Asian and Native American: 11–31 defendants, effects are noise.")

        top = {}
        for m in fitted:
            q = feats[feats.model == m]
            top[m] = [
                q[q.run == r].set_index("feature").marginal_effect.abs().idxmax() for r in (ra, rb)
            ]
        same_top = [m for m, (a, b) in top.items() if a == b]
        dists = {m: me_distance(m, epair, drop_tiny) for m in fitted}
        closest = min(dists, key=lambda m: dists[m]["ME L2"])
        movers = []
        for m in fitted:
            q = feats[feats.model == m]
            gap = (
                q[q.run == ra].set_index("feature").marginal_effect
                - q[q.run == rb].set_index("feature").marginal_effect
            ).abs()
            movers.append(f"{m}: {lib.FEATURES.get(gap.idxmax(), gap.idxmax())} ({gap.max():.3f})")
        lib.takeaway(
            (
                f"{', '.join(same_top)} keep{'s' if len(same_top) == 1 else ''} the same "
                "strongest feature in both fits"
                if same_top
                else "No model keeps the same strongest feature in both fits"
            )
            + f"; {closest} moves least overall (L2 {dists[closest]['ME L2']:.3f}). "
            f"The single effect that moves most — {'; '.join(movers)}. When it is a tiny "
            "group (Asian, Native American), the move is noise from a handful of defendants, "
            "not a change in what the model has learned."
        )
        st.caption(
            "Marginal effect: average change in predicted risk when the feature goes 0 → 1 "
            "(priors: +1), computed on the test defendants. L2 distance over all shown "
            "features; Spearman over their absolute sizes (does the importance ranking hold?)."
        )

# --------------------------------------------------------------------------- group extras

with tab_extra:
    st.markdown(
        "<p class='small-note'>Each group also ran its own stability analysis. These are "
        "<b>that group's methods and numbers</b>, shown for completeness; they are not the "
        "shared design above.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("TabPFN group: the same two designs across 7 feature sets", expanded=True):
        f_pred = TABPFN_DIR / "t07_prediction_stability" / "pairs_predictions.csv"
        f_perf = TABPFN_DIR / "t08_performance_stability" / "pairs_performance.csv"
        f_psi = TABPFN_DIR / "t09_fairness_stability" / "psi.csv"
        if not (f_pred.exists() and f_perf.exists()):
            st.info("TabPFN stability artifacts not found (tabpfn/artifacts/analysis/t07–t09).")
        else:
            pp = lib.read_csv(str(f_pred))
            pp = pp[pp.model == "TabPFN"].copy()
            pp["fs"] = pp.feature_set.map(lambda f: EXTRA_FS.get(f, f))
            order_fs = (
                pp[pp.operating_point == "0.5"].sort_values("rmse", ascending=False).fs.tolist()
            )
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                for op, alpha, name in [("0.5", 1.0, "threshold 0.5"),
                                        ("break_even", 0.45, "cost break-even")]:  # fmt: skip
                    q = pp[pp.operating_point == op].set_index("fs").reindex(order_fs)
                    fig.add_bar(
                        y=q.index, x=q.flip_rate, orientation="h", name=name,
                        marker={"color": tint(lib.COLOUR["TabPFN"], alpha), "cornerradius": 4},
                        customdata=q.n_flips,
                        hovertemplate="%{y}<br>flip rate %{x:.1%} (%{customdata} defendants)"
                        "<extra>" + name + "</extra>",
                    )  # fmt: skip
                fig.update_layout(barmode="group")
                fig.update_xaxes(tickformat=".0%", title_text="decision flip rate, X1 vs X2")
                fig.update_yaxes(autorange="reversed")
                lib.show(lib.style(fig, "Decision flips by feature set", 400))
            with col2:
                q = pp[pp.operating_point == "0.5"].set_index("fs").reindex(order_fs)
                fig = go.Figure(
                    go.Bar(
                        y=q.index, x=q.rmse, orientation="h", showlegend=False,
                        marker={"color": lib.COLOUR["TabPFN"], "cornerradius": 4},
                        hovertemplate="%{y}<br>score RMSE %{x:.3f}<extra></extra>",
                    )
                )  # fmt: skip
                fig.update_xaxes(title_text="score RMSE between the X1 and X2 fits")
                fig.update_yaxes(autorange="reversed")
                lib.show(lib.style(fig, "Score distance by feature set", 400))

            pf = lib.read_csv(str(f_perf))
            pf = pf[(pf.model == "TabPFN") & (pf.metric == "auc") & (pf.operating_point == "0.5")]
            col3, col4 = st.columns(2)
            with col3:
                fig = go.Figure()
                for pair, alpha in [("X1_vs_X2", 1.0), ("temporal_1_vs_2", 0.45)]:
                    q = pf[pf.pair == pair].copy()
                    q["fs"] = q.feature_set.map(lambda f: EXTRA_FS.get(f, f))
                    q = q.set_index("fs").reindex(order_fs)
                    fig.add_bar(
                        y=q.index, x=q.delta.abs(), orientation="h",
                        name="X1 vs X2" if pair == "X1_vs_X2" else "window 1 vs 2",
                        marker={"color": tint(lib.COLOUR["TabPFN"], alpha), "cornerradius": 4},
                        customdata=np.c_[q.a, q.b],
                        hovertemplate="%{y}<br>AUC %{customdata[0]:.3f} → %{customdata[1]:.3f}"
                        "<extra></extra>",
                    )  # fmt: skip
                fig.update_layout(barmode="group")
                fig.update_xaxes(title_text="|Δ AUC| between the two fits")
                fig.update_yaxes(autorange="reversed")
                lib.show(lib.style(fig, "Ranking stability by feature set", 400))
            with col4:
                if f_psi.exists():
                    ps = lib.read_csv(str(f_psi)).copy()
                    ps["fs"] = ps.feature_set.map(lambda f: EXTRA_FS.get(f, f))
                    ps = ps.set_index("fs").reindex(order_fs)
                    fig = go.Figure(
                        go.Bar(
                            y=ps.index, x=ps.psi_tabpfn_scores, orientation="h",
                            showlegend=False, customdata=ps.n_distinct_scores_a,
                            marker={"color": lib.COLOUR["TabPFN"], "cornerradius": 4},
                            hovertemplate="%{y}<br>PSI %{x:.3f}<br>%{customdata} distinct "
                            "scores<extra></extra>",
                        )
                    )  # fmt: skip
                    fig.add_vline(0.25, line={"color": lib.MUTED, "dash": "dot"})
                    fig.update_xaxes(title_text="PSI, window 1 → window 2 scores")
                    fig.update_yaxes(autorange="reversed")
                    lib.show(lib.style(fig, "Score shift over time by feature set", 400))
                else:
                    st.info("psi.csv not found.")
            lib.takeaway(
                "Dropping race <i>and</i> priors leaves TabPFN only a handful of distinct scores: "
                "the two random-sample fits then barely disagree, but PSI over time explodes. "
                "With so few distinct scores a large PSI comes from the coarse bins, not from "
                "the model drifting."
            )

    with st.expander("LogReg group: repeated cross-validation and bootstrap refits"):
        L = lib.LOGREG_ART
        f_sum, f_cv = L / "stability_summary.json", L / "stability_cv.csv"
        if not f_sum.exists():
            st.info("LogReg stability artifacts not found (logreg/artifacts/analysis).")
        else:
            summ = json.loads(f_sum.read_text())
            f_coef, f_dist = L / "coefficient_stability.json", L / "model_distance.json"
            coef = json.loads(f_coef.read_text()) if f_coef.exists() else {}
            dist = json.loads(f_dist.read_text()) if f_dist.exists() else {}
            k = st.columns(4)
            k[0].metric(
                "Repeated-CV AUC",
                f"{summ['cv_auc_mean']:.3f} ± {summ['cv_auc_sd']:.3f}",
                help=f"Mean ± sd over {summ['cv_folds']} folds.",
            )
            k[1].metric(
                "Per-defendant score sd",
                f"{summ['median_sd']:.3f}",
                help=f"Median over defendants across {summ['n_bootstrap_refits']} bootstrap "
                f"refits (95th pct {summ['p95_sd']:.3f}).",
            )
            k[2].metric(
                "Unstable decisions",
                pct(summ["unstable_share"]),
                help=f"Share of defendants whose decision at {summ['threshold']:.3f} changes "
                "across bootstrap refits.",
            )
            if coef:
                k[3].metric(
                    "Coefficient rank agreement",
                    f"{coef['mean_rank_correlation']:.2f}",
                    help=f"Mean rank correlation across refits; top feature identical in "
                    f"{coef['top_feature_consistency']:.0%} of refits.",
                )
            if f_cv.exists():
                cv = lib.read_csv(str(f_cv))
                fig = go.Figure(
                    go.Histogram(
                        x=cv.auc, nbinsx=30, showlegend=False,
                        marker={"color": lib.COLOUR["LogReg"], "line": {"color": "white",
                                                                        "width": 1}},
                        hovertemplate="AUC %{x}<br>%{y} folds<extra></extra>",
                    )
                )  # fmt: skip
                fig.add_vline(
                    cv.auc.mean(), line={"color": lib.INK, "dash": "dash"},
                    annotation_text=f"mean {cv.auc.mean():.3f}",
                )  # fmt: skip
                fig.update_xaxes(title_text="AUC on the held-out fold")
                fig.update_yaxes(title_text="folds")
                lib.show(lib.style(fig, f"LogReg AUC over {len(cv)} repeated-CV folds", 340))
            notes = [
                f"PSI train vs test {summ['psi_train_vs_test']:.4f}",
                f"threshold {summ['threshold']:.3f} flags "
                f"{pct(summ['selection_rate_at_threshold'])}",
            ]
            if dist:
                notes.append(
                    "importance distance between refits "
                    f"{dist['mean_importance_distance']:.3f} ± "
                    f"{dist['sd_importance_distance']:.3f} over {dist['n_pairs']} pairs"
                )
            f_seed = L / "seed_sensitivity.csv"
            if f_seed.exists():
                seeds = lib.read_csv(str(f_seed))
                notes.append(
                    f"{len(seeds)} random seeds give AUC spread {np.ptp(seeds.auc):.4f} "
                    "(LogReg is deterministic)"
                )
            st.markdown(
                "<p class='small-note'>" + " · ".join(notes) + "</p>", unsafe_allow_html=True
            )
            lib.takeaway(
                f"The LogReg group's own resampling tells the same story as the shared design: "
                f"AUC varies by ±{summ['cv_auc_sd']:.3f} (sd) between folds, while at their "
                f"threshold {summ['threshold']:.3f} {pct(summ['unstable_share'])} of decisions "
                "change across bootstrap refits — the ranking is stable, borderline decisions "
                "are not."
            )
