import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for lib

import lib  # noqa: E402

lib.setup_page(
    "Interpretability",
    "What drives each model's risk score: across the whole test set, along priors, and for one "
    "defendant. The cross-model views use one shared definition for all three models; each "
    "group's own analyses are labelled with their units. Default: holdout 70/30, all features.",
)

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

if str(lib.ROOT) not in sys.path:  # the repo root, for the logreg package
    sys.path.insert(0, str(lib.ROOT))

# ----------------------------------------------------------------------------- constants

TAB = lib.TABPFN_ART / "analysis"
LOG = lib.LOGREG_ART
XGB_MODEL = lib.XGBOOST_ART / "xgb_race_aware.json"
PRIORS = "Number_of_Priors"
FEATS = list(lib.FEATURES)
RUN_LABEL = {**lib.RUNS, "X1+X2_to_X3": "X1+X2 → X3 (TabPFN only)"}
RUN_ORDER = ["holdout", "X1_to_X3", "X2_to_X3", "X1+X2_to_X3", "temporal_1", "temporal_2"]
SET_LABEL = {
    **lib.FEATURE_SETS,
    "age_blind": "No age",
    "sex_blind": "No sex",
    "protected_blind": "No race, sex or age",
    "race_proxy_blind": "No race, no race proxies",
}
ENGINEERED = {
    "log_priors": "log(1 + priors)",
    "priors_capped": "Priors capped at 10",
    "no_priors": "No priors (0/1)",
    "young_x_log_priors": "Under 25 × log(1 + priors)",
    "const": "Intercept",
}
RAISE, LOWER, ABSENT = "#d64545", "#3a9d8f", "#d64545"
GROUP_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#8a8984"]
SMALL_NOTE = "† Asian and Native American: 11–31 defendants, so their effects are noise."
SHAP_BACKGROUND = 25  # k-means summary of the training set, as the TabPFN group used


# -------------------------------------------------------------------------------- helpers


def nice(feature: str) -> str:
    return lib.FEATURES.get(feature, ENGINEERED.get(feature, feature))


def label(feature: str) -> str:
    """Pretty name, with a dagger on the two tiny groups."""
    return f"{nice(feature)} †" if feature in lib.SMALL_GROUPS else nice(feature)


def load(path: Path, **kwargs) -> pd.DataFrame | None:
    """A committed CSV, or None if it is missing, empty or unreadable."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        frame = lib.read_csv(str(path), **kwargs)
        return frame if len(frame) else None
    except Exception:
        return None


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def guarded(render, what: str) -> None:
    """Run one section; if its data is broken, say so instead of crashing the page."""
    try:
        render()
    except Exception as err:
        st.info(f"{what} could not be shown ({type(err).__name__}: {err}).")


def opacity(features) -> list[float]:
    return [0.35 if f in lib.SMALL_GROUPS else 1.0 for f in features]


def age_band(x: pd.Series) -> str:
    if x["Age_Below_TwentyFive"] == 1:
        return "Under 25"
    return "Over 45" if x["Age_Above_FourtyFive"] == 1 else "25–45"


# ----------------------------------------------------------------------------- live models


@st.cache_resource(show_spinner="Loading the test set and the saved models ...")
def live() -> dict:
    """Test set plus the LogReg (refitted, deterministic) and the saved XGBoost."""
    out: dict = {"models": {}}
    from compas_scoring.data import train_test

    out["train"], out["test"] = train_test("race_aware")
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(XGB_MODEL)
        out["models"]["XGBoost"] = model
    except Exception:
        pass
    try:
        from logreg.model import build_logistic

        out["models"]["LogReg"] = build_logistic().fit(out["train"].X, out["train"].y)
    except Exception:
        pass
    return out


def live_or_none() -> dict | None:
    try:
        return live()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def kernel_explainer(model: str):
    import shap

    data = live()
    est, cols = data["models"][model], list(data["train"].X.columns)
    background = shap.kmeans(data["train"].X, SHAP_BACKGROUND)
    return shap.KernelExplainer(
        lambda d: est.predict_proba(pd.DataFrame(d, columns=cols))[:, 1], background
    )


@st.cache_data(show_spinner="Computing Kernel SHAP ...")
def kernel_shap(model: str, rows: tuple) -> pd.DataFrame:
    """Kernel SHAP in probability units; 2,048 coalitions enumerate all 1,024 exactly."""
    explainer = kernel_explainer(model)
    X = live()["test"].X.loc[list(rows)]
    values = np.asarray(explainer.shap_values(X, nsamples=2048, silent=True))
    out = pd.DataFrame(values.reshape(len(rows), -1), index=list(rows), columns=X.columns)
    out["base_value"] = float(np.ravel(explainer.expected_value)[0])
    return out


@st.cache_data(show_spinner="Computing XGBoost TreeSHAP on the test set ...")
def xgb_tree_shap() -> pd.DataFrame:
    """Exact TreeSHAP for every test defendant, in log-odds (the booster's margin)."""
    import shap

    data = live()
    explainer = shap.TreeExplainer(data["models"]["XGBoost"])
    X = data["test"].X
    out = pd.DataFrame(np.asarray(explainer.shap_values(X)), index=X.index, columns=X.columns)
    out["base_value"] = float(np.ravel(explainer.expected_value)[0])
    return out


@st.cache_data(show_spinner="Permuting features for XGBoost ...")
def xgb_permutation() -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    data = live()
    result = permutation_importance(
        data["models"]["XGBoost"],
        data["test"].X,
        data["test"].y,
        n_repeats=10,
        random_state=int(lib.CONFIG.random_state),
        scoring="roc_auc",
    )
    return pd.DataFrame(
        {
            "feature": list(data["test"].X.columns),
            "value": result.importances_mean,
            "lo": result.importances_mean - result.importances_std,
            "hi": result.importances_mean + result.importances_std,
        }
    )


@st.cache_data
def xgb_impurity() -> pd.DataFrame:
    booster = live()["models"]["XGBoost"].get_booster()
    frame = pd.DataFrame(
        {kind: booster.get_score(importance_type=kind) for kind in ("gain", "weight", "cover")}
    )
    return frame.reindex(FEATS).fillna(0.0)


@st.cache_data(show_spinner="Computing XGBoost ICE curves ...")
def xgb_ice(grid: tuple) -> pd.DataFrame:
    data = live()
    model, X = data["models"]["XGBoost"], data["test"].X
    frames = [
        pd.DataFrame(
            {"row": X.index, "x": v, "y": model.predict_proba(X.assign(**{PRIORS: v}))[:, 1]}
        )
        for v in grid
    ]
    return pd.concat(frames, ignore_index=True)


def logreg_terms(row: int) -> tuple[pd.DataFrame, float]:
    """LogReg's exact log-odds decomposition: coefficient × standardised engineered term."""
    pipe = live()["models"]["LogReg"]
    design = pipe.named_steps["engineer"].transform(live()["test"].X.loc[[row]])
    scaled = pipe.named_steps["scale"].transform(design)
    clf = pipe.named_steps["clf"]
    out = pd.DataFrame(
        {
            "term": design.columns,
            "value": design.iloc[0].to_numpy(dtype=float),
            "contribution": clf.coef_[0] * scaled[0],
        }
    )
    return out, float(clf.intercept_[0])


# ----------------------------------------------------------------------------- importance


def importance_table() -> pd.DataFrame:
    """Every global importance available on holdout × all features, in long form.

    Columns: model, method, feature, value, lo, hi, unit, definition.
    """
    rows: list[pd.DataFrame] = []

    def add(model, method, frame, unit, definition):
        if frame is None or not len(frame):
            return
        frame = frame.copy()
        for col in ("lo", "hi"):
            if col not in frame:
                frame[col] = np.nan
        rows.append(
            frame[["feature", "value", "lo", "hi"]].assign(
                model=model, method=method, unit=unit, definition=definition
            )
        )

    # LogReg: the group's own artifacts
    perm = load(LOG / "permutation_importance.csv")
    if perm is not None:
        add(
            "LogReg",
            "Permutation (AUC drop)",
            perm.rename(columns={"importance": "value"}),
            "AUC points",
            "Drop in test AUC when the feature is shuffled (10 shuffles).",
        )
    shap_imp = load(LOG / "shap_importance.csv")
    if shap_imp is not None:
        add(
            "LogReg",
            "Mean |SHAP| (group's Kernel SHAP)",
            shap_imp.rename(columns={"mean_abs_shap": "value"}),
            "probability",
            "LogReg group's Kernel SHAP on a test sample: average absolute contribution.",
        )
    lime = load(LOG / "lime_weights.csv")
    if lime is not None:
        w = lime.drop(columns="row").abs().mean()
        add(
            "LogReg",
            "Mean |LIME weight|",
            w.rename("value").rename_axis("feature").reset_index(),
            "LIME weight",
            "Average absolute LIME weight over 30 test defendants.",
        )
    xper = load(LOG / "xper_auc.csv")
    if xper is not None:
        x = xper[
            (xper.feature != "benchmark") & (xper.get("feature_set", "race_aware") == "race_aware")
        ]
        x = x.rename(columns={"contribution": "value"})
        add("LogReg", "XPER (AUC)", x, "AUC points", "Each feature's share of the model's AUC.")

    # XGBoost: computed live from the saved model
    if "XGBoost" in (live_or_none() or {}).get("models", {}):
        add(
            "XGBoost",
            "Permutation (AUC drop)",
            xgb_permutation(),
            "AUC points",
            "Drop in test AUC when the feature is shuffled (10 shuffles, ± 1 sd; computed "
            "in the app from the saved model).",
        )
        tree = xgb_tree_shap().drop(columns="base_value").abs().mean()
        add(
            "XGBoost",
            "Mean |TreeSHAP| (log-odds)",
            tree.rename("value").rename_axis("feature").reset_index(),
            "log-odds",
            "Exact TreeSHAP over the whole test set, in log-odds (not probability).",
        )
        imp = xgb_impurity()
        for kind, text in {
            "gain": "Average loss reduction of the splits on the feature.",
            "weight": "Number of splits on the feature.",
            "cover": "Average number of defendants passing through its splits.",
        }.items():
            add(
                "XGBoost",
                f"Impurity: {kind}",
                imp[kind].rename("value").rename_axis("feature").reset_index(),
                kind,
                text,
            )
    xper = load(lib.XGBOOST_ART / "xper_race_aware.csv", index_col=0)
    if xper is not None:
        x = xper.drop(index="benchmark", errors="ignore").iloc[:, 0]
        add(
            "XGBoost",
            "XPER (AUC)",
            x.rename("value").rename_axis("feature").reset_index(),
            "AUC points",
            "Each feature's share of the model's AUC.",
        )

    # TabPFN: the group's artifacts (holdout, all features)
    rows.extend(tabpfn_importance("holdout", "race_aware"))

    # The one SHAP defined identically for all three: same 25 defendants, same background
    for model, frame in same_defendant_shap().items():
        add(
            model,
            "Kernel SHAP, same 25 defendants",
            frame[FEATS].abs().mean().rename("value").rename_axis("feature").reset_index(),
            "probability",
            "Kernel SHAP with one shared background (k-means 25 of the training "
            "set) on the 25 defendants TabPFN's group explained. Comparable across models.",
        )
    if not rows:
        return pd.DataFrame(
            columns=["model", "method", "feature", "value", "lo", "hi", "unit", "definition"]
        )
    return pd.concat(rows, ignore_index=True)


def tabpfn_importance(run: str, feature_set: str) -> list[pd.DataFrame]:
    out = []

    def frame(df, method, unit, definition):
        df = df.copy()
        for col in ("lo", "hi"):
            if col not in df:
                df[col] = np.nan
        out.append(
            df[["feature", "value", "lo", "hi"]].assign(
                model="TabPFN", method=method, unit=unit, definition=definition
            )
        )

    perm = load(TAB / "t13_permutation_importance" / "permutation.csv")
    if perm is not None:
        p = perm[(perm.run == run) & (perm.feature_set == feature_set)]
        p = p.rename(columns={"importance": "value", "ci_lower": "lo", "ci_upper": "hi"})
        if len(p[p.metric == "auc"]):
            frame(
                p[p.metric == "auc"],
                "Permutation (AUC drop)",
                "AUC points",
                "Drop in test AUC when the feature is shuffled (95% interval).",
            )
        if len(p[p.metric == "cost"]):
            frame(
                p[p.metric == "cost"],
                "Permutation (cost)",
                "$ per defendant",
                "Rise in expected cost per defendant when the feature is shuffled (95% interval).",
            )
    shap_imp = load(TAB / "t14_kernel_shap" / "shap_importance.csv")
    if shap_imp is not None:
        s = shap_imp[(shap_imp.run == run) & (shap_imp.feature_set == feature_set)]
        if len(s):
            frame(
                s.rename(columns={"mean_abs_shap": "value"}),
                "Mean |SHAP| (Kernel SHAP)",
                "probability",
                "Kernel SHAP on 25 test defendants: average absolute "
                "contribution to the predicted risk.",
            )
    imp = load(TAB / "t11_surrogate_impurity" / "impurity.csv")
    if imp is not None:
        i = imp[(imp.run == run) & (imp.feature_set == feature_set)]
        i = i[(i.tree_grown_with == "gini") & (i.criterion == "gini")]
        if len(i):
            frame(
                i.rename(columns={"impurity_decrease_share": "value"}),
                "Surrogate tree impurity (Gini)",
                "share of impurity decrease",
                "Depth-4 tree fitted to TabPFN's scores; share of Gini decrease per "
                "feature. See the Surrogate tree tab.",
            )
    if run == "holdout" and feature_set == "race_aware":
        xper = load(TAB / "t20_xper_individual" / "xper_auc.csv", index_col=0)
        if xper is not None:
            x = xper.drop(index="benchmark", errors="ignore").iloc[:, 0]
            frame(
                x.rename("value").rename_axis("feature").reset_index(),
                "XPER (AUC)",
                "AUC points",
                "Each feature's share of the model's AUC.",
            )
        cost = load(TAB / "t16_xper_cost" / "xper_cost.csv")
        if cost is not None:
            c = cost[
                (cost.feature != "benchmark")
                & (cost.get("feature_set", "race_aware") == "race_aware")
            ]
            c = c.rename(columns={"contribution": "value"})
            frame(
                c,
                "XPER (cost)",
                "$ per defendant",
                "Each feature's share of the cost reduction over the benchmark.",
            )
    return out


def tabpfn_shap_rows() -> pd.DataFrame | None:
    sv = load(TAB / "t14_kernel_shap" / "shap_values.csv")
    if sv is None:
        return None
    sv = sv[(sv.run == "holdout") & (sv.feature_set == "race_aware")]
    return sv.set_index("row") if len(sv) else None


def same_defendant_shap() -> dict[str, pd.DataFrame]:
    """Kernel SHAP (probability) for TabPFN's 25 explained defendants, all three models."""
    tab = tabpfn_shap_rows()
    if tab is None:
        return {}
    out = {}
    data = live_or_none()
    for model in ("LogReg", "XGBoost"):
        if data and model in data["models"]:
            out[model] = kernel_shap(model, tuple(int(r) for r in tab.index))
    out["TabPFN"] = tab
    return {m: out[m] for m in lib.MODELS if m in out}


# =============================================================================== the page

(tab_cross, tab_imp, tab_ice, tab_local, tab_tree, tab_extra) = st.tabs(
    [
        "Cross-model effects",
        "Feature importance",
        "PDP & ICE",
        "Local explanations",
        "Surrogate tree (TabPFN)",
        "Each group's extras",
    ]
)


# ------------------------------------------------------------------------ 1. cross-model


def render_marginal_effects() -> None:
    me = lib.comparison("marginal_effects.csv")
    me = me[me.feature_set == "race_aware"]
    runs = [r for r in RUN_ORDER if r in set(me.run)]
    c1, c2, c3, c4 = st.columns([2.2, 2, 1.4, 1.2])
    run = c1.selectbox("Design", runs, format_func=RUN_LABEL.get, key="ix_me_run")
    top_n = c2.slider("Top features", 1, len(FEATS), 5, key="ix_me_top")
    hide_small = c3.toggle(
        "Hide tiny groups",
        True,
        key="ix_me_hide",
        help="Asian and Native American have 11–31 defendants.",
    )
    show_ci = c4.toggle("95% CI", True, key="ix_me_ci")

    sub = me[me.run == run]
    if hide_small:
        sub = sub[~sub.feature.isin(lib.SMALL_GROUPS)]
    rank = sub.groupby("feature").marginal_effect.apply(lambda s: s.abs().max())
    feats = list(rank.sort_values(ascending=False).index[:top_n])
    names = [label(f) + (" (+1)" if f == PRIORS else "") for f in feats]
    fig = go.Figure()
    for model in lib.MODELS:
        d = sub[sub.model == model].set_index("feature").reindex(feats)
        if d.marginal_effect.isna().all():
            continue
        fig.add_bar(
            x=names,
            y=d.marginal_effect,
            name=model,
            marker={"color": lib.COLOUR[model], "opacity": opacity(feats)},
            error_y={
                "type": "data",
                "array": 1.96 * d.se,
                "visible": show_ci,
                "color": lib.INK_2,
                "thickness": 1,
                "width": 3,
            },
            customdata=np.c_[d.sd, d.share_positive],
            hovertemplate=f"<b>{model}</b> · %{{x}}<br>average change %{{y:+.1%}}<br>"
            "sd across defendants %{customdata[0]:.3f}<br>"
            "share of defendants pushed up %{customdata[1]:.0%}<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title="Average change in predicted risk", tickformat="+.0%")
    lib.show(lib.style(fig, f"Marginal effects · {RUN_LABEL[run]}", 440))
    note = "" if hide_small else f" {SMALL_NOTE}"
    if run == "X1+X2_to_X3":
        note += " This design was only run for TabPFN."
    st.caption(
        "Same definition for all three models: the average change in each test defendant's "
        "predicted probability when a yes/no feature is switched 0 → 1 (its sibling dummies "
        "cleared, e.g. Under 25 clears Over 45), or when priors go up by one." + note
    )
    lib.takeaway(
        "<b>Age and priors drive all three models.</b> On the holdout, being under 25 adds "
        "about 17 points of "
        "risk for LogReg and TabPFN, each extra prior about 7 points; the African-American "
        "dummy moves risk by about 1 point. XGBoost's effects point the same way but are "
        "roughly half the size, because its scores are compressed (sd 0.12 vs 0.20: 100 trees "
        "at learning rate 0.01)."
    )


def render_pdp() -> None:
    pdp = lib.comparison("pdp_priors.csv")
    centre = st.toggle("Centre at 0 priors (compare shapes, not levels)", False, key="ix_pdp_c")
    fig = go.Figure()
    for model in lib.MODELS:
        d = pdp[pdp.model == model].sort_values(PRIORS)
        if not len(d):
            continue
        y = d.predicted_risk - (d.predicted_risk.iloc[0] if centre else 0)
        fig.add_scatter(
            x=d[PRIORS],
            y=y,
            name=model,
            mode="lines+markers",
            line={"color": lib.COLOUR[model], "width": 2.5},
            marker={"symbol": lib.SYMBOL[model], "size": 6},
            hovertemplate=f"<b>{model}</b><br>priors %{{x:.1f}}<br>risk %{{y:.1%}}<extra></extra>",
        )
    data = live_or_none()
    if data is not None:
        p90 = float(np.percentile(data["test"].X[PRIORS], 90))
        fig.add_vline(x=p90, line={"dash": "dot", "color": lib.MUTED})
        fig.add_annotation(
            x=p90,
            y=1,
            yref="paper",
            text=f"90% of defendants ≤ {p90:g} priors",
            showarrow=False,
            xanchor="left",
            font={"color": lib.INK_2, "size": 11},
        )
    fig.update_xaxes(title="Number of priors (set for every test defendant)")
    fig.update_yaxes(
        title="Change in average risk" if centre else "Average predicted risk",
        tickformat="+.0%" if centre else ".0%",
    )
    lib.show(lib.style(fig, "Partial dependence on priors (holdout)", 420))
    lib.takeaway(
        "LogReg and TabPFN rise from about 28% at 0 priors to about 79% at 15, steepest over "
        "the first few priors. XGBoost has the same shape but only climbs from 35% to 62%: "
        "same ranking of defendants, much narrower range of scores."
    )


def render_effect_stability() -> None:
    me = lib.comparison("marginal_effects.csv")
    me = me[me.feature_set == "race_aware"]
    feats = [f for f in FEATS if f in set(me.feature)]
    feature = st.selectbox("Feature", feats, format_func=label, key="ix_stab_feat")
    d = me[me.feature == feature]
    runs = [r for r in RUN_ORDER if r in set(d.run)]
    fig = go.Figure()
    for model in lib.MODELS:
        m = d[d.model == model].set_index("run").reindex(runs)
        fig.add_scatter(
            x=[RUN_LABEL[r] for r in runs],
            y=m.marginal_effect,
            name=model,
            mode="markers",
            marker={"color": lib.COLOUR[model], "symbol": lib.SYMBOL[model], "size": 11},
            error_y={
                "type": "data",
                "array": 1.96 * m.se,
                "color": lib.COLOUR[model],
                "thickness": 1.2,
                "width": 0,
            },
            hovertemplate=f"<b>{model}</b> · %{{x}}<br>%{{y:+.1%}}<extra></extra>",
        )
    fig.add_hline(y=0, line={"color": lib.MUTED, "width": 1})
    fig.update_layout(scattermode="group", scattergap=0.55)
    fig.update_yaxes(title="Average change in predicted risk", tickformat="+.0%")
    lib.show(lib.style(fig, f"{label(feature)}: the same effect on every design", 400))
    if feature in lib.SMALL_GROUPS:
        lib.caveat(SMALL_NOTE[2:] + " Watch it jump between designs.")
    else:
        st.caption("Error bars: 95% interval of the average over test defendants.")


with tab_cross:
    guarded(render_marginal_effects, "The cross-model marginal effects")
    left, right = st.columns(2, gap="large")
    with left:
        guarded(render_pdp, "The partial dependence on priors")
    with right:
        guarded(render_effect_stability, "The design-by-design effects")


# ------------------------------------------------------------------------ 2. importance


def bar_ranked(frame: pd.DataFrame, model: str, title: str, unit: str) -> go.Figure:
    frame = frame.sort_values("value")
    has_ci = frame.lo.notna().any()
    fig = go.Figure(
        go.Bar(
            y=[label(f) for f in frame.feature],
            x=frame.value,
            orientation="h",
            marker={"color": lib.COLOUR[model], "opacity": opacity(frame.feature)},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": frame.hi - frame.value,
                "arrayminus": frame.value - frame.lo,
                "color": lib.INK_2,
                "thickness": 1,
                "visible": bool(has_ci),
            },
            hovertemplate="%{y}: %{x:.4g}<extra></extra>",
        )
    )
    fig.update_xaxes(title=unit)
    return lib.style(fig, title, 60 + 34 * len(frame))


def render_odds_ratios() -> None:
    orr = load(LOG / "odds_ratios.csv")
    if orr is None:
        st.info("The LogReg odds ratios are not available.")
        return
    orr = orr[orr.feature != "const"].sort_values("odds_ratio")
    colour = [lib.COLOUR["LogReg"] if s else lib.MUTED for s in orr.significant]
    fig = go.Figure(
        go.Scatter(
            x=orr.odds_ratio,
            y=[label(f) for f in orr.feature],
            mode="markers",
            marker={"color": colour, "size": 10, "symbol": "circle"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": orr.or_upper - orr.odds_ratio,
                "arrayminus": orr.odds_ratio - orr.or_lower,
                "color": lib.INK_2,
                "thickness": 1,
            },
            customdata=np.c_[orr.p_value, orr.or_lower, orr.or_upper],
            hovertemplate="%{y}<br>odds ratio %{x:.2f} [%{customdata[1]:.2f}, "
            "%{customdata[2]:.2f}]<br>p = %{customdata[0]:.3g}<extra></extra>",
        )
    )
    fig.add_vline(x=1, line={"color": lib.MUTED, "dash": "dot"})
    fig.update_xaxes(type="log", title="Odds ratio per unit (log scale, 95% CI)")
    lib.show(lib.style(fig, "LogReg odds ratios (engineered terms)", 90 + 30 * len(orr)))
    st.caption(
        "The LogReg group's statsmodels fit, on its engineered design: priors enter four times "
        "(raw, log(1 + priors), capped at 10, a no-priors flag) plus an Under 25 × log priors "
        "interaction, so no single priors odds ratio is the priors effect. Blue: p < 0.05. "
        + SMALL_NOTE
    )


def render_importance() -> None:
    table = importance_table()
    c1, c2, c3 = st.columns([1.4, 2.4, 2.2])
    models = [m for m in lib.MODELS if m in set(table.model)] or list(lib.MODELS)
    model = c1.radio("Model", models, key="ix_imp_model", horizontal=True)
    methods = list(dict.fromkeys(table[table.model == model].method))
    if model == "LogReg":
        methods.append("Odds ratios")
    if not methods:
        st.info(f"No importance results are available for {model}.")
        return
    method = c2.selectbox("Method", methods, key=f"ix_imp_method_{model}")
    run, fs = "holdout", "race_aware"
    if model == "TabPFN" and method not in (
        "XPER (AUC)",
        "XPER (cost)",
        "Kernel SHAP, same 25 defendants",
    ):
        source_file = {
            "Mean |SHAP| (Kernel SHAP)": TAB / "t14_kernel_shap" / "shap_importance.csv",
            "Surrogate tree impurity (Gini)": TAB / "t11_surrogate_impurity" / "impurity.csv",
        }.get(method, TAB / "t13_permutation_importance" / "permutation.csv")
        avail = load(source_file)
        combos = (
            set(zip(avail.run, avail.feature_set, strict=False))
            if avail is not None
            else {("holdout", "race_aware")}
        )
        runs = [r for r in RUN_ORDER if any(c[0] == r for c in combos)] or ["holdout"]
        with c3:
            a, b = st.columns(2)
            run = a.selectbox("Design", runs, format_func=RUN_LABEL.get, key="ix_imp_run")
            sets = [s for s in SET_LABEL if (run, s) in combos] or ["race_aware"]
            fs = b.selectbox("Features", sets, format_func=SET_LABEL.get, key=f"ix_imp_fs_{run}")
    if method == "Odds ratios":
        render_odds_ratios()
    else:
        if model == "TabPFN" and (run, fs) != ("holdout", "race_aware"):
            frames = tabpfn_importance(run, fs)
            source = pd.concat(frames, ignore_index=True) if frames else table.iloc[0:0]
        else:
            source = table[table.model == model]
        d = source[source.method == method]
        if not len(d):
            st.info("This method has no result for that design and feature set.")
        else:
            title = f"{model} · {method}"
            if model == "TabPFN" and (run, fs) != ("holdout", "race_aware"):
                title += f" · {RUN_LABEL[run]} · {SET_LABEL[fs]}"
            lib.show(bar_ranked(d, model, title, d.unit.iloc[0]))
            st.caption(d.definition.iloc[0] + " " + SMALL_NOTE)

    st.subheader("Do the methods agree?")
    st.markdown(
        "<p class='small-note'>Each column rescaled to shares of 100% (absolute values), so "
        "methods with different units line up. Holdout, all features.</p>",
        unsafe_allow_html=True,
    )
    hide = st.toggle("Hide tiny groups", True, key="ix_heat_hide")
    t = table.copy()
    if hide:
        t = t[~t.feature.isin(lib.SMALL_GROUPS)]
    t["share"] = t.value.abs() / t.groupby(["model", "method"]).value.transform(
        lambda s: s.abs().sum()
    )
    t["column"] = t.model + " · " + t.method
    order = [c for m in lib.MODELS for c in dict.fromkeys(t[t.model == m].column)]
    grid = t.pivot_table(index="feature", columns="column", values="share").reindex(columns=order)
    grid = grid.loc[grid.mean(axis=1).sort_values(ascending=False).index]
    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=grid.columns,
            y=[label(f) for f in grid.index],
            colorscale=[[0, "#f7f7f5"], [0.25, "#b9cdf0"], [1, "#1e4f99"]],
            zmin=0,
            zmax=1,
            texttemplate="%{z:.0%}",
            textfont={"size": 10},
            hovertemplate="%{x}<br>%{y}: %{z:.1%}<extra></extra>",
            colorbar={"tickformat": ".0%"},
        )
    )
    fig.update_xaxes(tickangle=-40, side="top", automargin=True)
    fig.update_yaxes(autorange="reversed")
    lib.show(lib.style(fig, None, 120 + 34 * len(grid) + 110))
    top = grid.idxmax()
    others = [f"{c} ({nice(f)})" for c, f in top.items() if f != PRIORS]
    race = grid.reindex(["African_American", "Hispanic", "Other"]).dropna(how="all")
    race_note = ""
    if len(race):
        col = race.max().idxmax()
        race_note = (
            f" The largest race share anywhere is {race.max().max():.0%} "
            f"({nice(race[col].idxmax())}, {col}); most columns give race a few percent."
        )
    lib.takeaway(
        f"<b>Priors is the top feature in {int((top == PRIORS).sum())} of {len(top)} "
        "method × model columns</b>, with the two age bands next."
        + (f" Where it is not first: {'; '.join(others)}." if others else "")
        + race_note
    )


with tab_imp:
    guarded(render_importance, "The feature importance")


# -------------------------------------------------------------------------- 3. PDP & ICE


def ice_frame(model: str, run: str) -> tuple[pd.DataFrame | None, bool]:
    """(row, x, y) ICE curves and whether they are already centred."""
    if model == "TabPFN":
        ice = load(TAB / "t12_pdp_ice" / "ice.csv")
        if ice is None:
            return None, True
        ice = ice[(ice.run == run) & (ice.feature_set == "race_aware")]
        return ice.rename(columns={PRIORS: "x", "risk_change_from_min": "y"}), True
    if model == "LogReg":
        ice = load(LOG / "ice_curves.csv")
        if ice is None:
            return None, False
        return ice[ice.kind == "raw"].rename(columns={PRIORS: "x", "predicted_risk": "y"}), False
    pdp = lib.comparison("pdp_priors.csv")
    grid = tuple(float(v) for v in pdp[pdp.model == "XGBoost"][PRIORS].sort_values())
    if not grid:
        grid = tuple(float(v) for v in np.linspace(0, 15, 21))
    return xgb_ice(grid), False


def lines(frame: pd.DataFrame) -> tuple[list, list, list]:
    xs, ys, ids = [], [], []
    for row, g in frame.sort_values(["row", "x"]).groupby("row", sort=False):
        xs += [*g.x, None]
        ys += [*g.y, None]
        ids += [row] * len(g) + [None]
    return xs, ys, ids


def render_ice() -> None:
    c1, c2, c3, c4, c5 = st.columns([2.2, 1.8, 1.8, 1.6, 1.2])
    model = c1.radio("Model", list(lib.MODELS), index=2, horizontal=True, key="ix_ice_model")
    run = "holdout"
    if model == "TabPFN":
        ice_all = load(TAB / "t12_pdp_ice" / "ice.csv")
        runs = [r for r in RUN_ORDER if ice_all is not None and r in set(ice_all.run)]
        run = c2.selectbox(
            "Design", runs or ["holdout"], format_func=RUN_LABEL.get, key="ix_ice_run"
        )
    else:
        c2.selectbox(
            "Design",
            ["holdout"],
            format_func=RUN_LABEL.get,
            disabled=True,
            key=f"ix_ice_run_{model}",
        )
    if model == "XGBoost" and "XGBoost" not in (live_or_none() or {}).get("models", {}):
        st.info("The saved XGBoost model could not be loaded.")
        return
    ice, centred_only = ice_frame(model, run)
    if ice is None or not len(ice):
        st.info(f"No ICE curves are available for {model}.")
        return
    all_rows = ice.row.unique()
    n = c3.slider(
        "Curves shown", 5, len(all_rows), min(60, len(all_rows)), 5, key=f"ix_ice_n_{model}"
    )
    colour_by = c4.selectbox("Colour by", ["None", "Age band", "Sex", "Race"], key="ix_ice_by")
    centre = c5.toggle(
        "Centre",
        True if centred_only else False,
        disabled=centred_only,
        key=f"ix_ice_centre_{model}",
        help="Subtract each defendant's risk at 0 priors.",
    )
    if centred_only:
        centre = True

    ice = ice.copy()
    if centre and not centred_only:
        start = ice.sort_values("x").groupby("row").y.transform("first")
        ice["y"] = ice.y - start
    rng = np.random.default_rng(int(st.session_state.get("ix_ice_seed", 0)))
    shown = rng.choice(all_rows, size=min(n, len(all_rows)), replace=False)
    sample = ice[ice.row.isin(shown)]

    fig = go.Figure()
    if colour_by == "None":
        groups = {model: sample}
        colours = {model: lib.COLOUR[model]}
    else:
        g = lib.groups(tuple(int(r) for r in shown))
        key = {"Age band": "age_band", "Sex": "sex", "Race": "race"}[colour_by]
        values = g[key]
        if colour_by == "Race":
            values = values.where(values.isin(["African-American", "Caucasian"]), "Other groups")
        cats = sorted(values.unique())
        groups = {c: sample[sample.row.isin(values.index[values == c])] for c in cats}
        colours = (
            {c: lib.RACE_COLOUR.get(c, lib.MUTED) for c in cats}
            if colour_by == "Race"
            else dict(zip(cats, GROUP_PALETTE, strict=False))
        )
    for name, part in groups.items():
        xs, ys, ids = lines(part)
        fig.add_scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"{name} ({part.row.nunique()})",
            line={"color": colours[name], "width": 1},
            opacity=0.35,
            customdata=ids,
            hovertemplate="defendant #%{customdata}<br>priors %{x:.1f}<br>%{y:.1%}<extra></extra>",
        )
    pdp = ice.groupby("x").y.mean().sort_index()
    fig.add_scatter(
        x=pdp.index,
        y=pdp.values,
        mode="lines",
        name=f"PDP (mean of all {len(all_rows)})",
        line={"color": lib.INK, "width": 4},
        hovertemplate="PDP<br>priors %{x:.1f}<br>%{y:.1%}<extra></extra>",
    )
    fig.update_xaxes(title="Number of priors")
    fig.update_yaxes(
        title="Change in risk from 0 priors" if centre else "Predicted risk",
        tickformat="+.0%" if centre else ".0%",
    )
    lib.show(lib.style(fig, f"{model}: ICE curves on priors · {RUN_LABEL.get(run, run)}", 480))
    if st.button("Draw another sample", key="ix_ice_resample"):
        st.session_state["ix_ice_seed"] = int(st.session_state.get("ix_ice_seed", 0)) + 1
        st.rerun()

    # Heterogeneity: total effect from the lowest to the highest grid point, per defendant
    ends = ice.sort_values("x").groupby("row").y.agg(["first", "last"])
    total = ends["last"] - ends["first"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean total effect", f"{total.mean():+.1%}")
    m2.metric("Spread (sd)", f"{total.std():.1%}")
    m3.metric("Range", f"{total.min():+.0%} to {total.max():+.0%}")
    m4.metric("Curves going down", f"{int((total < 0).sum())} of {len(total)}")
    source = {
        "TabPFN": "TabPFN group's ICE (500 test defendants; stored centred at 0 priors, so "
        "raw curves are not available).",
        "LogReg": "LogReg group's ICE (100 test defendants, on the priors values they observed).",
        "XGBoost": "Computed in the app from the saved XGBoost model, every test defendant.",
    }[model]
    st.caption(f"{source} Total effect = risk at the highest grid point minus risk at 0 priors.")
    lib.takeaway(
        (
            "<b>Every curve ends higher than it starts</b>: more priors never lowers anyone's "
            "risk overall. "
            if (total < 0).sum() == 0
            else f"{int((total < 0).sum())} curves end lower than they start. "
        )
        + f"The total effect varies little between defendants (sd {total.std():.1%} around "
        f"{total.mean():+.1%}), so the curves are close to parallel: the priors effect barely "
        "depends on the rest of the profile. XGBoost's are flat steps (tree splits) and "
        "shorter (compressed scores)."
    )


with tab_ice:
    guarded(render_ice, "The ICE curves")


# ------------------------------------------------------------------------- 4. local


def profile(row: int) -> str:
    data = live_or_none()
    if data is None or row not in data["test"].X.index:
        return f"defendant #{row}"
    x = data["test"].X.loc[row]
    try:
        race = str(lib.groups((int(row),)).iloc[0]["race"])
    except Exception:
        race = "race n/a"
    sex = "Female" if x["Female"] == 1 else "Male"
    charge = "Misdemeanour" if x["Misdemeanor"] == 1 else "Felony"
    return f"{int(x[PRIORS])} priors · {age_band(x)} · {sex} · {charge} · {race}"


def render_local_shap() -> None:
    data = live_or_none()
    if data is None:
        st.info("The test set could not be loaded, so local explanations are unavailable.")
        return
    tab = tabpfn_shap_rows()
    archetypes = load(LOG / "archetypes.csv")
    options: dict[int, str] = {}
    if tab is not None:
        for r in tab.index:
            options[int(r)] = ""
    if archetypes is not None:
        for _, a in archetypes.iterrows():
            options[int(a["index"])] = f" · LogReg '{a.archetype}'"
    options = {r: s for r, s in options.items() if r in data["test"].X.index}
    if not options:
        st.info("No explained defendants are available.")
        return
    row = st.selectbox(
        "Defendant (TabPFN's 25 Kernel-SHAP defendants, plus the LogReg group's 3 archetypes)",
        list(options),
        format_func=lambda r: f"#{r} · {profile(r)}{options[r]}",
        key="ix_local_row",
    )
    x = data["test"].X.loc[row]
    cols = st.columns(4)
    for c, model in zip(cols, lib.MODELS, strict=False):
        try:
            c.metric(f"{model} risk", f"{float(lib.scores(model).score.loc[row]):.0%}")
        except Exception:
            c.metric(f"{model} risk", "–")
    cols[3].metric("Reoffended within 2 years", "Yes" if int(data["test"].y.loc[row]) else "No")

    shap_by_model: dict[str, pd.Series] = {}
    base: dict[str, float] = {}
    for model in ("LogReg", "XGBoost"):
        if model in data["models"]:
            v = kernel_shap(model, (int(row),)).iloc[0]
            shap_by_model[model], base[model] = v[FEATS], float(v["base_value"])
    if tab is not None and row in tab.index:
        shap_by_model["TabPFN"] = tab.loc[row, FEATS].astype(float)
        base["TabPFN"] = float(tab.loc[row, "base_value"])
    frame = pd.DataFrame(shap_by_model).reindex(
        columns=[m for m in lib.MODELS if m in shap_by_model]
    )
    frame = frame.loc[frame.abs().max(axis=1).sort_values().index]
    ylabels = [f"{label(f)} = {x[f]:g}" for f in frame.index]
    fig = go.Figure()
    for model in frame.columns:
        fig.add_bar(
            y=ylabels,
            x=frame[model],
            name=model,
            orientation="h",
            marker={"color": lib.COLOUR[model]},
            hovertemplate=f"<b>{model}</b><br>%{{y}}<br>%{{x:+.1%}}<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.2)
    fig.add_vline(x=0, line={"color": lib.MUTED, "width": 1})
    fig.update_xaxes(title="Contribution to predicted risk (probability points)", tickformat="+.0%")
    lib.show(lib.style(fig, f"Why this risk score? Kernel SHAP · defendant #{row}", 520))
    bases = " · ".join(f"{m} {base[m]:.0%}" for m in frame.columns)
    missing_tab = (
        ""
        if "TabPFN" in frame.columns
        else (" TabPFN's group explained only 25 defendants, so TabPFN is missing here.")
    )
    st.caption(
        "Kernel SHAP in probability units for all three models: the bars add up to the risk "
        f"score minus the average score over the background ({bases}). LogReg and XGBoost "
        "computed in the app with TabPFN's settings (k-means 25 of the training set), all "
        "1,024 feature coalitions. A dummy the defendant does not have can still get a small "
        "bar: 'not being X' relative to the background." + missing_tab
    )

    with st.expander("XGBoost's native TreeSHAP for this defendant (log-odds)"):
        if "XGBoost" not in data["models"]:
            st.info("The saved XGBoost model could not be loaded.")
        else:
            tree = xgb_tree_shap()
            v = tree.loc[row, FEATS].sort_values(key=abs)
            f2 = go.Figure(
                go.Bar(
                    y=[f"{label(f)} = {x[f]:g}" for f in v.index],
                    x=v.values,
                    orientation="h",
                    marker={"color": [RAISE if s > 0 else LOWER for s in v.values]},
                    hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>",
                )
            )
            f2.update_xaxes(title="Contribution (log-odds, not probability)")
            lib.show(lib.style(f2, "XGBoost TreeSHAP", 380))
            st.caption(
                "Exact TreeSHAP on the booster's margin. Base value "
                f"{tree.base_value.iloc[0]:+.3f} log-odds; base + bars = the log-odds of the "
                "score. Red raises risk, green lowers it."
            )

    with st.expander(
        "LogReg's exact decomposition for this defendant (log-odds, engineered terms)"
    ):
        if "LogReg" not in data["models"]:
            st.info("The LogReg model could not be refitted.")
        else:
            terms, intercept = logreg_terms(row)
            terms = terms[terms.contribution.abs() > 1e-9].sort_values("contribution", key=abs)
            f3 = go.Figure(
                go.Bar(
                    y=[
                        f"{nice(t)} = {v:.2f}"
                        for t, v in zip(terms.term, terms.value, strict=False)
                    ],
                    x=terms.contribution,
                    orientation="h",
                    marker={"color": [RAISE if s > 0 else LOWER for s in terms.contribution]},
                    hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>",
                )
            )
            f3.update_xaxes(title="Coefficient × standardised value (log-odds)")
            lib.show(lib.style(f3, "LogReg: intercept + these terms = the score, exactly", 480))
            st.caption(
                f"Intercept {intercept:+.3f}. Relative to the average defendant (the design is "
                "standardised). Same formula as the LogReg group's local_contributions.csv; "
                "priors appears four times through the engineered terms."
            )
        lime = load(LOG / "lime_weights.csv")
        if lime is not None and row in set(lime.row):
            w = lime.set_index("row").loc[row, FEATS].astype(float).sort_values(key=abs)
            f4 = go.Figure(
                go.Bar(
                    y=[f"{label(f)} = {x[f]:g}" for f in w.index],
                    x=w.values,
                    orientation="h",
                    marker={"color": [RAISE if s > 0 else LOWER for s in w.values]},
                )
            )
            f4.update_xaxes(title="LIME weight")
            lib.show(lib.style(f4, "LogReg group's LIME weights for this defendant", 380))
            st.caption(
                "Compare with the SHAP bars above: LIME gives weight to dummies the "
                "defendant does not have, e.g. Asian."
            )


def render_lime() -> None:
    st.subheader("LIME is not reproducible (TabPFN)")
    lime = load(TAB / "t15_lime_reproducibility" / "lime.csv")
    summary = load_json(TAB / "t15_lime_reproducibility" / "lime_summary.json") or {}
    if lime is None:
        st.info("The TabPFN LIME reproducibility runs are not available.")
        return
    person = summary.get("defendant", {})
    row = summary.get("defendant_row", "?")
    counts = lime.groupby("top_feature").agg(
        n=("seed", "size"), absent=("top_feature_absent", "any")
    )
    counts = counts.sort_values("n")
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        fig = go.Figure(
            go.Bar(
                y=[
                    label(f) + (" (defendant has 0)" if a else "")
                    for f, a in zip(counts.index, counts.absent, strict=False)
                ],
                x=counts.n,
                orientation="h",
                text=counts.n,
                textposition="outside",
                marker={"color": [ABSENT if a else lib.COLOUR["TabPFN"] for a in counts.absent]},
                hovertemplate="%{y}: top feature in %{x} runs<extra></extra>",
            )
        )
        fig.update_xaxes(title=f"Runs naming it the top feature (of {len(lime)})")
        lib.show(lib.style(fig, "Which feature does LIME call most important?", 300))
    with right:
        wcols = [c for c in lime.columns if c.startswith("w_")]
        long = lime.melt(
            id_vars=["num_samples", "seed"],
            value_vars=wcols,
            var_name="feature",
            value_name="weight",
        )
        long["feature"] = long.feature.str[2:]
        fig = go.Figure()
        for i, (ns, part) in enumerate(long.groupby("num_samples")):
            fig.add_box(
                x=[label(f) for f in part.feature],
                y=part.weight,
                name=f"{ns:,} samples",
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                line={"width": 1},
                marker={"size": 5, "color": GROUP_PALETTE[i % len(GROUP_PALETTE)]},
                fillcolor="rgba(0,0,0,0)",
            )
        fig.update_layout(boxmode="group")
        fig.add_hline(y=0, line={"color": lib.MUTED, "width": 1})
        fig.update_xaxes(tickangle=-35)
        fig.update_yaxes(title="LIME weight")
        lib.show(lib.style(fig, "LIME weights across the 15 runs", 300))
    absent = [label(f) for f in lime.loc[lime.top_feature_absent, "top_feature"].unique()]
    n_absent = summary.get("runs_naming_an_absent_feature", int(lime.top_feature_absent.sum()))
    lib.takeaway(
        f"Same defendant (#{row}: {int(person.get(PRIORS, 0))} priors, female, misdemeanour), "
        f"same model, {len(lime)} LIME runs that differ only in seed and sample size: "
        f"<b>{lime.top_feature.nunique()} different 'most important' features</b>, and "
        f"{n_absent} of {len(lime)} runs name a feature she does not have "
        f"({', '.join(absent)}: all 0 for her). LIME's local "
        "linear fit on a model with only yes/no features and one count is unstable: we rely "
        "on SHAP instead."
    )


with tab_local:
    guarded(render_local_shap, "The local SHAP explanations")
    st.divider()
    guarded(render_lime, "The LIME reproducibility study")


# ---------------------------------------------------------------------- 5. surrogate tree


def render_surrogate() -> None:
    sur = load(TAB / "t11_surrogate_impurity" / "surrogate.csv")
    imp = load(TAB / "t11_surrogate_impurity" / "impurity.csv")
    if sur is None or imp is None:
        st.info("The TabPFN surrogate-tree results are not available.")
        return
    c1, c2, c3, c4 = st.columns([1.6, 1.8, 2.4, 1.2])
    runs = [r for r in RUN_ORDER if r in set(sur.run)]
    run = c1.selectbox("Design", runs, format_func=RUN_LABEL.get, key="ix_tree_run")
    sets = [s for s in SET_LABEL if s in set(sur.feature_set)]
    fs = c2.selectbox(
        "Features",
        sets,
        index=sets.index("race_aware") if "race_aware" in sets else 0,
        format_func=SET_LABEL.get,
        key="ix_tree_fs",
    )
    crit = c3.radio(
        "Impurity criterion",
        ["gini", "entropy", "misclassification", "all three"],
        horizontal=True,
        key="ix_tree_crit",
    )
    grown = c4.selectbox("Tree split on", ["gini", "entropy"], key="ix_tree_grown")

    s = sur[(sur.run == run) & (sur.feature_set == fs)]
    if len(s):
        s = s.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Fidelity R²",
            f"{s.fidelity_r2:.3f}",
            help="Share of the variance of TabPFN's scores the tree reproduces.",
        )
        acc = s.get(f"fidelity_accuracy_{grown}", np.nan)
        m2.metric("Same decision at 0.5", "–" if pd.isna(acc) else f"{acc:.1%}")
        m3.metric("Leaves", f"{int(s.n_leaves)}")
        m4.metric("Depth", f"{int(s.max_depth)}")

    d = imp[(imp.run == run) & (imp.feature_set == fs) & (imp.tree_grown_with == grown)]
    crits = ["gini", "entropy", "misclassification"] if crit == "all three" else [crit]
    d = d[d.criterion.isin(crits)]
    if not len(d):
        st.info("No impurity decomposition for this combination.")
    else:
        order = d.groupby("feature").impurity_decrease_share.mean().sort_values().index
        fig = go.Figure()
        shades = {
            "gini": lib.COLOUR["TabPFN"],
            "entropy": "#8f84d6",
            "misclassification": "#c9c3ee",
        }
        for c in crits:
            part = d[d.criterion == c].set_index("feature").reindex(order)
            fig.add_bar(
                y=[label(f) for f in order],
                x=part.impurity_decrease_share,
                orientation="h",
                name=c.capitalize(),
                marker={"color": shades[c], "opacity": opacity(order)},
                hovertemplate=f"{c}<br>%{{y}}: %{{x:.1%}}<extra></extra>",
            )
        fig.update_layout(barmode="group", bargap=0.2)
        fig.update_xaxes(title="Share of the tree's impurity decrease", tickformat=".0%")
        lib.show(
            lib.style(
                fig,
                f"What the surrogate tree splits on · {RUN_LABEL[run]} · {SET_LABEL.get(fs, fs)}",
                90 + 28 * len(order) * len(crits),
            )
        )
        st.caption(
            "A depth-4 regression tree fitted to TabPFN's predicted risks (not to the outcomes), "
            "then each feature's share of the total impurity decrease, measured three ways. "
            "Features the tree never splits on get 0."
        )

    fid = sur[sur.run == run].set_index("feature_set").reindex(sets)
    fig = go.Figure(
        go.Bar(
            x=[SET_LABEL.get(f, f) for f in fid.index],
            y=fid.fidelity_r2,
            marker={"color": [lib.COLOUR["TabPFN"] if f == fs else "#c9c3ee" for f in fid.index]},
            text=[f"{v:.2f}" for v in fid.fidelity_r2],
            textposition="outside",
        )
    )
    fig.update_yaxes(title="Fidelity R²", range=[0, 1.08])
    lib.show(lib.style(fig, f"How well a 4-level tree mimics TabPFN · {RUN_LABEL[run]}", 320))
    lib.takeaway(
        "A tree with about a dozen leaves reproduces around 90% of the variance of TabPFN's "
        "scores on all features (holdout), splitting mostly on priors and the two age bands, "
        "then misdemeanour. The black box is, to a first approximation, a small readable rule "
        "set; with fewer features to use, the fit gets closer still."
    )
    path = TAB / "t11_surrogate_impurity" / "rules" / f"{run}__{fs}.txt"
    with st.expander(f"The tree's rules ({RUN_LABEL[run]}, {SET_LABEL.get(fs, fs)})"):
        if path.exists():
            st.code(path.read_text(), language=None)
        else:
            st.info("No rules file for this combination.")


with tab_tree:
    guarded(render_surrogate, "The surrogate tree")


# ------------------------------------------------------------------ 6. each group's extras


def render_logreg_extras() -> None:
    st.subheader("LogReg")
    me = load(LOG / "marginal_effects.csv")
    left, right = st.columns(2, gap="large")
    with left:
        if me is None:
            st.info("The LogReg marginal effects are not available.")
        else:
            d = me.sort_values("marginal_effect")
            fig = go.Figure(
                go.Scatter(
                    x=d.marginal_effect,
                    y=[label(f) for f in d.feature],
                    mode="markers",
                    marker={
                        "color": [lib.COLOUR["LogReg"] if s else lib.MUTED for s in d.significant],
                        "size": 9,
                    },
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": d.ci_upper - d.marginal_effect,
                        "arrayminus": d.marginal_effect - d.ci_lower,
                        "color": lib.INK_2,
                        "thickness": 1,
                    },
                    hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
                )
            )
            fig.add_vline(x=0, line={"color": lib.MUTED, "dash": "dot"})
            fig.update_xaxes(title="Average marginal effect (probability per unit)")
            lib.show(lib.style(fig, "Marginal effects on the engineered terms", 460))
            st.caption(
                "The LogReg group's statsmodels average marginal effects, one per engineered "
                "term (log priors, capped priors, no-priors flag, Under 25 × log priors, ...). "
                "Not comparable with the cross-model tab, where priors is one +1 change. "
                "Blue: p < 0.05."
            )
    with right:
        path = load(LOG / "lasso_path.csv")
        if path is None:
            st.info("The lasso path is not available.")
        else:
            fig = go.Figure()
            for i, f in enumerate(FEATS):
                if f not in path:
                    continue
                fig.add_scatter(
                    x=path.C,
                    y=path[f],
                    mode="lines",
                    name=label(f),
                    line={
                        "width": 2.5 if f == PRIORS else 1.5,
                        "color": GROUP_PALETTE[i % len(GROUP_PALETTE)],
                        "dash": "dot" if f in lib.SMALL_GROUPS else "solid",
                    },
                    hovertemplate=f"{label(f)}<br>C %{{x:.3g}}<br>coef %{{y:.3f}}<extra></extra>",
                )
            fig.update_xaxes(type="log", title="Regularisation C (larger = weaker penalty)")
            fig.update_yaxes(title="Standardised coefficient")
            lib.show(lib.style(fig, "Lasso path: order in which features enter", 460))
            st.caption(
                "Priors enters first, then the two age bands, then misdemeanour; the "
                "race dummies come in last with small coefficients."
            )
    card = load(LOG / "scorecard.csv")
    if card is not None:
        with st.expander("LogReg points scorecard"):
            st.dataframe(
                card.assign(feature=card.feature.map(nice)).rename(
                    columns={
                        "feature": "Term",
                        "coefficient": "Coefficient",
                        "points_per_unit": "Points per unit",
                        "base_points": "Base points",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
    conflicts = load(LOG / "explanation_conflicts.csv")
    if conflicts is not None:
        with st.expander("LIME vs Kernel SHAP on the three LogReg archetypes"):
            st.dataframe(
                conflicts.drop(columns=["conflicting_features"], errors="ignore"),
                width="stretch",
                hide_index=True,
            )


def render_tabpfn_extras() -> None:
    st.subheader("TabPFN")
    left, right = st.columns(2, gap="large")
    with left:
        eff = load(TAB / "t14_kernel_shap" / "efficiency.csv")
        if eff is None:
            st.info("The Kernel SHAP timing results are not available.")
        else:
            e = (
                eff[eff.feature_set == "race_aware"]
                .set_index("run")
                .reindex([r for r in RUN_ORDER if r in set(eff.run)])
            )
            fig = go.Figure(
                go.Bar(
                    x=[RUN_LABEL[r] for r in e.index],
                    y=e.seconds_per_explained_row,
                    marker={"color": lib.COLOUR["TabPFN"]},
                    text=[f"{v:.1f}s" for v in e.seconds_per_explained_row],
                    textposition="outside",
                )
            )
            fig.update_yaxes(title="Seconds per explained defendant")
            lib.show(lib.style(fig, "The price of explaining TabPFN: Kernel SHAP time", 360))
            ok = bool(e["holds_to_1e-5"].all()) if "holds_to_1e-5" in e else None
            st.caption(
                "Kernel SHAP, 25 defendants per design. For XGBoost and LogReg the same "
                "computation takes well under a second for all 25. "
                + (
                    "Efficiency holds (bars add up to the score) to 1e-5 on every design."
                    if ok
                    else ""
                )
            )
    with right:
        stab = load(TAB / "t17_importance_stability" / "importance_distance.csv")
        if stab is None:
            st.info("The importance-stability results (t17) are not available.")
        else:
            fig = go.Figure()
            for i, (pair, part) in enumerate(stab.groupby("pair")):
                fig.add_bar(
                    x=part.measure.str.replace("_", " "),
                    y=part.spearman,
                    name=pair.replace("_vs_", " vs ").replace("_", " "),
                    marker={"color": GROUP_PALETTE[i]},
                    customdata=np.c_[part.top3_overlap, part.same_top_feature],
                    hovertemplate="%{x}<br>Spearman %{y:.2f}<br>top-3 overlap "
                    "%{customdata[0]:.0%}<br>same top feature: %{customdata[1]}"
                    "<extra></extra>",
                )
            fig.update_layout(barmode="group")
            fig.update_yaxes(title="Rank correlation of importances", range=[0, 1.05])
            lib.show(lib.style(fig, "Are TabPFN's importances stable across refits?", 360))
            st.caption(
                "Two refits on different data (X1 vs X2; two time windows): rank "
                "correlation of the feature importances. Hover for top-3 overlap."
            )
    summ = load(TAB / "t12_pdp_ice" / "ice_summary.csv")
    if summ is not None:
        with st.expander("ICE heterogeneity on priors, every design"):
            st.dataframe(
                summ.assign(run=summ.run.map(lambda r: RUN_LABEL.get(r, r))),
                width="stretch",
                hide_index=True,
            )


def render_xgb_extras() -> None:
    st.subheader("XGBoost")
    data = live_or_none()
    if data is None or "XGBoost" not in data["models"]:
        st.info("The saved XGBoost model could not be loaded.")
        return
    imp = xgb_impurity()
    share = imp / imp.sum()
    fig = go.Figure()
    for kind, colour in zip(
        ["gain", "weight", "cover"], ["#eb6834", "#f3a37f", "#f8cdb8"], strict=False
    ):
        fig.add_bar(
            y=[label(f) for f in share.index],
            x=share[kind],
            orientation="h",
            name=kind.capitalize(),
            marker={"color": colour},
            hovertemplate=f"{kind}<br>%{{y}}: %{{x:.1%}}<extra></extra>",
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Share of the booster's total", tickformat=".0%")
    lib.show(lib.style(fig, "Built-in impurity importance: gain, weight, cover", 460))
    unused = [label(f) for f in imp.index if imp.loc[f].sum() == 0]
    st.caption(
        "From the saved booster (100 trees, depth 4). Gain = average loss reduction per split; "
        "weight = number of splits; cover = defendants per split. "
        + (f"Never split on: {', '.join(unused)}. " if unused else "")
        + "The XGBoost group's own marginal-effect file uses a finite-difference derivative "
        "and is not comparable, so it is not shown; see the cross-model tab instead."
    )


with tab_extra:
    guarded(render_logreg_extras, "The LogReg group's results")
    st.divider()
    guarded(render_tabpfn_extras, "The TabPFN group's results")
    st.divider()
    guarded(render_xgb_extras, "The XGBoost group's results")
