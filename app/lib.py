"""Shared data, metrics and styling for the Streamlit app.

Everything the pages show comes from committed results:
    comparison/artifacts/   the cross-model comparison (run_comparison.py)
    tabpfn/, logreg/, xgboost/ artifacts   each group's own detailed results

The per-defendant scores in comparison/artifacts/predictions.csv make every threshold-dependent
number live: pages recompute metrics, costs and fairness for whatever threshold is chosen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from compas_scoring.config import CONFIG  # noqa: E402
from compas_scoring.data import group_frame, load_raw  # noqa: E402
from compas_scoring.fairness import fairness_table, proxy_strata  # noqa: E402

COMPARISON = ROOT / "comparison" / "artifacts"
TABPFN_ART = ROOT / "tabpfn" / "artifacts"
LOGREG_ART = ROOT / "logreg" / "artifacts" / "analysis"
XGBOOST_ART = ROOT / "xgboost" / "models"

# ------------------------------------------------------------------------------ vocabulary

MODELS = ["LogReg", "XGBoost", "TabPFN"]
ALL_MODELS = [*MODELS, "COMPAS tool"]
COLOUR = {"LogReg": "#2a78d6", "XGBoost": "#eb6834", "TabPFN": "#4a3aa7", "COMPAS tool": "#8a8984"}
SYMBOL = {
    "LogReg": "circle",
    "XGBoost": "square",
    "TabPFN": "diamond",
    "COMPAS tool": "triangle-up",
}
DASH = {"LogReg": "solid", "XGBoost": "solid", "TabPFN": "solid", "COMPAS tool": "dash"}
RACE_COLOUR = {"African-American": "#4a3aa7", "Caucasian": "#1baf7a", "Other groups": "#8a8984"}

FEATURE_SETS = {
    "race_aware": "All features",
    "race_blind": "No race",
    "race_priors_blind": "No race, no priors",
    "priors_blind": "Drop priors, retrain",
    "priors_set_1": "Priors set to 1",
}
RUNS = {
    "holdout": "Holdout 70/30",
    "X1_to_X3": "X1 → X3",
    "X2_to_X3": "X2 → X3",
    "temporal_1": "Time: first 40% → next 20%",
    "temporal_2": "Time: first 70% → last 30%",
}
FEATURES = {
    "Number_of_Priors": "Priors",
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
SMALL_GROUPS = {"Asian", "Native_American"}  # ~11-31 defendants: effects are noise
BREAK_EVEN = float(CONFIG.costs.break_even)
C_FN, C_FP = float(CONFIG.costs.c_fn), float(CONFIG.costs.c_fp)
COMPARISONS = [f"{p} vs {r}" for _, p, r in CONFIG.fairness.comparisons]
PRIMARY = COMPARISONS[0]  # African-American vs Caucasian

# --------------------------------------------------------------------------------- styling

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1350px;}
h1 {font-weight: 700; letter-spacing: -0.02em;}
h2, h3 {font-weight: 650; letter-spacing: -0.01em;}
div[data-testid="stMetric"] {background: #f7f7f5; border: 1px solid #e4e3df;
    border-radius: 10px; padding: 0.8rem 1rem;}
div[data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #52514e;}
.takeaway {border-left: 4px solid #2a78d6; background: #f5f8fd; padding: 0.7rem 1rem;
    border-radius: 0 8px 8px 0; margin: 0.4rem 0 1rem 0; color: #0b0b0b;}
.caveat {border-left: 4px solid #eda100; background: #fdf8ee; padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0; margin: 0.4rem 0 1rem 0; font-size: 0.92rem;}
.small-note {color: #52514e; font-size: 0.85rem;}
.tldr {background: #0b2545; color: #ffffff; border-radius: 10px; padding: 0.75rem 1.1rem;
    margin: 0.2rem 0 1.1rem 0; font-size: 0.93rem; line-height: 1.45;}
.tldr .tag {font-weight: 700; letter-spacing: 0.06em; font-size: 0.75rem; color: #9ec5f4;}
.tldr ul {margin: 0.3rem 0 0.35rem 1.1rem; padding: 0;}
.tldr li {margin: 0.1rem 0;}
.tldr .so {font-weight: 650; border-top: 1px solid #2c4a70; padding-top: 0.35rem;}
</style>
"""


def setup_page(title: str, subtitle: str | None = None) -> None:
    """Page title, subtitle and the shared stylesheet. Call first on every page."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f"<p class='small-note'>{subtitle}</p>", unsafe_allow_html=True)


def takeaway(text: str) -> None:
    st.markdown(f"<div class='takeaway'>{text}</div>", unsafe_allow_html=True)


def tldr(points: list[str], conclusion: str) -> None:
    """A dense summary box: 2-4 short points and the one-line conclusion they lead to."""
    items = "".join(f"<li>{p}</li>" for p in points)
    st.markdown(
        f"<div class='tldr'><div class='tag'>TL;DR</div><ul>{items}</ul>"
        f"<div class='so'>→ {conclusion}</div></div>",
        unsafe_allow_html=True,
    )


def caveat(text: str) -> None:
    st.markdown(f"<div class='caveat'>{text}</div>", unsafe_allow_html=True)


def style(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    """The app's chart style: white, thin grid, left-aligned title, legend below."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 16}} if title else None,
        font={"family": "Inter, Segoe UI, sans-serif", "size": 13, "color": INK},
        margin={"l": 10, "r": 10, "t": 50 if title else 20, "b": 10},
        legend={"orientation": "h", "yanchor": "top", "y": -0.15, "x": 0, "title": None},
        hoverlabel={"font_size": 12},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


# ------------------------------------------------------------------------------------ data


@st.cache_data
def predictions() -> pd.DataFrame:
    """Test-set scores: model, run, feature_set, row, y, score, compas_tool."""
    return pd.read_csv(COMPARISON / "predictions.csv")


@st.cache_data
def comparison(name: str) -> pd.DataFrame:
    """A table from comparison/artifacts/ (performance, fairness, stability, ...)."""
    return pd.read_csv(COMPARISON / name)


@st.cache_data
def read_csv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def scores(model: str, run: str = "holdout", feature_set: str = "race_aware") -> pd.DataFrame:
    """One model's test scores (y, score) indexed by cohort row. "COMPAS tool" = its decision."""
    p = predictions()
    source = "LogReg" if model == "COMPAS tool" else model
    q = p[(p.model == source) & (p.run == run) & (p.feature_set == feature_set)]
    out = q.set_index("row")[["y", "score" if model != "COMPAS tool" else "compas_tool"]]
    return out.set_axis(["y", "score"], axis=1)


@st.cache_data
def groups(rows: tuple) -> pd.DataFrame:
    """Race, sex, age band and charge degree for modelling-table rows."""
    return group_frame(load_raw().loc[list(rows)])


# --------------------------------------------------------------------------------- metrics


def metrics(y, score, threshold: float) -> dict:
    """Every 0/1 metric at one threshold, plus AUC and expected cost per defendant."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    flag = score >= threshold
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "AUC": roc_auc_score(y, score) if len(set(y)) > 1 else float("nan"),
        "Accuracy": (tp + tn) / len(y),
        "Precision": precision,
        "Recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if tp else 0.0,
        "Specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "FPR": fp / (fp + tn) if fp + tn else float("nan"),
        "FNR": fn / (fn + tp) if fn + tp else float("nan"),
        "Flag rate": float(flag.mean()),
        "Cost per defendant": (fn * C_FN + fp * C_FP) / len(y),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,  # fmt: skip
    }


def roc(y, score) -> pd.DataFrame:
    fpr, tpr, thr = roc_curve(y, score)
    return pd.DataFrame({"FPR": fpr, "TPR": tpr, "threshold": thr})


def baselines(y) -> dict:
    """Cost per defendant of the two trivial policies."""
    rate = float(np.mean(y))
    return {"Detain everyone": (1 - rate) * C_FP, "Release everyone": rate * C_FN}


@st.cache_data
def fairness(model: str, feature_set: str, threshold: float) -> pd.DataFrame:
    """The shared fairness protocol at any threshold (holdout; every comparison)."""
    s = scores(model, "holdout", feature_set)
    table = fairness_table(
        s.y, s.score, groups(tuple(s.index)), proxy_strata(s.index), at={"chosen": threshold}
    )
    return table.assign(model=model, feature_set=feature_set)


# ------------------------------------------------------------------------------- controls


def threshold_control(key: str, default: float = 0.5, help_text: str | None = None) -> float:
    """The decision threshold: presets for 0.5 and the cost break-even, or a custom slider."""
    presets = {"0.5 (default)": 0.5, f"{BREAK_EVEN:.3f} (cost break-even)": BREAK_EVEN}
    col1, col2 = st.columns([3, 2])
    with col2:
        preset = st.radio(
            "Threshold preset",
            [*presets, "Custom"],
            key=f"{key}_preset",
            label_visibility="collapsed",
        )
    with col1:
        if preset in presets:
            threshold = presets[preset]
            st.markdown(
                f"**Decision threshold: {threshold:.3f}**<br>"
                "<span class='small-note'>Flag a defendant if predicted risk ≥ threshold. "
                "Choose Custom to set any value.</span>",
                unsafe_allow_html=True,
            )
        else:
            threshold = st.slider(
                "Decision threshold (flag if risk ≥ threshold)",
                0.05, 0.95, float(default), 0.005,
                key=f"{key}_slider",
                help=help_text or "Defendants at or above this predicted risk are flagged.",
            )  # fmt: skip
    return float(threshold)


def model_picker(key: str, default=None, include_tool: bool = True) -> list[str]:
    options = ALL_MODELS if include_tool else MODELS
    return st.multiselect("Models", options, default=default or options, key=key)


def fmt_pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.1%}"


def fmt_money(x: float) -> str:
    return "–" if pd.isna(x) else f"${x:,.0f}"
