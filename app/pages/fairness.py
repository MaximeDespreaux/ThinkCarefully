import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for lib

import lib  # noqa: E402

from compas_scoring.config import CONFIG  # noqa: E402
from compas_scoring.fairness import calibration_by_group, comparison_rows  # noqa: E402

lib.setup_page(
    "Fairness",
    "Is any model unfair to a group, why, and can it be fixed? The shared protocol (course "
    "steps 1-3), recomputed live on the holdout set at the threshold you choose. Gaps read "
    "protected group minus reference group.",
)

# ------------------------------------------------------------------------------ vocabulary

ALPHA = float(CONFIG.fairness.alpha)
TOST_DELTA = float(CONFIG.fairness.tost_delta)
BASE_SETS = ["race_aware", "race_blind", "race_priors_blind"]
FIX_SETS = ["priors_blind", "priors_set_1"]
COMPARISON_KEY = {f"{p} vs {r}": (a, p, r) for a, p, r in CONFIG.fairness.comparisons}
METRIC_LABEL = {
    "statistical_parity": "Flag rate (statistical parity)",
    "conditional_statistical_parity": "Same profile (conditional parity)",
    "fpr": "False-positive rate",
    "fnr": "False-negative rate",
    "equalized_odds": "Equalized odds (joint)",
}
PRIORS_BANDS = ([-0.5, 0.5, 1.5, 3.5, 6.5, np.inf], ["0", "1", "2-3", "4-6", "7+"])
ROLE_COLOUR = {"protected": lib.RACE_COLOUR["African-American"], "reference": "#1baf7a"}
VARIANT_SHORT = {
    "race_aware": "All features",
    "race_blind": "No race",
    "race_priors_blind": "No race, no priors",
    "priors_blind": "Drop priors",
    "priors_set_1": "Priors = 1",
}


def group_colour(label: str, role: str) -> str:
    """Race groups follow lib.RACE_COLOUR; other labels (sex) take their role's colour."""
    if label in lib.RACE_COLOUR:
        return lib.RACE_COLOUR[label]
    if label in ("Hispanic", "Other"):
        return lib.RACE_COLOUR["Other groups"]
    return ROLE_COLOUR[role]


def row_label(model: str, feature_set: str) -> str:
    return "COMPAS tool" if model == "COMPAS tool" else f"{model} · {VARIANT_SHORT[feature_set]}"


def fmt_p(p) -> str:
    if p is None or pd.isna(p):
        return "–"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def fmt_signed(x, digits: int = 2) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:+.{digits}f}"


def span(values, digits: int = 2, signed: bool = True) -> str:
    """'a – b' over the finite values, or 'n/a'."""
    v = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return "n/a"
    f = (lambda x: f"{x:+.{digits}f}") if signed else (lambda x: f"{x:.{digits}f}")
    return f(v.min()) if np.isclose(v.min(), v.max()) else f"{f(v.min())} – {f(v.max())}"


# ------------------------------------------------------------------------------ data


def protocol(pairs, threshold: float) -> pd.DataFrame:
    """The shared protocol table for (model, feature set) pairs; lib.fairness caches each."""
    frames = []
    for model, feature_set in pairs:
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)  # statsmodels on empty strata
                frames.append(lib.fairness(model, feature_set, float(threshold)))
        except Exception:  # noqa: BLE001 - a missing variant must not break the page
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["label"] = [row_label(m, f) for m, f in zip(out.model, out.feature_set, strict=True)]
    return out


def pairs_for(models, feature_sets) -> list[tuple[str, str]]:
    pairs = [(m, fs) for m in models if m != "COMPAS tool" for fs in feature_sets]
    if "COMPAS tool" in models:
        pairs.append(("COMPAS tool", "race_aware"))  # its decision ignores our feature sets
    return pairs


@st.cache_data(show_spinner=False)
def cohort() -> pd.DataFrame:
    """The whole modelling cohort: priors, outcome and race (three groups for the charts)."""
    from compas_scoring.data import build_dataset

    data = build_dataset("race_aware")
    race = data.groups["race"]
    return pd.DataFrame(
        {
            "priors": data.X["Number_of_Priors"].astype(int),
            "y": data.y,
            "race": race,
            "group": race.where(race.isin(["African-American", "Caucasian"]), "Other groups"),
        }
    )


@st.cache_data(show_spinner=False)
def leakage() -> pd.DataFrame:
    """Race leakage (CV AUC of predicting race from the non-race features) per feature set."""
    try:
        from compas_scoring.proxies import race_leakage

        rows = [{"feature_set": fs, "auc": race_leakage(fs)} for fs in BASE_SETS]
        return pd.DataFrame(rows)
    except Exception:  # noqa: BLE001 - optional panel
        return pd.DataFrame(columns=["feature_set", "auc"])


@st.cache_data(show_spinner=False)
def auc_of(model: str, feature_set: str) -> float:
    try:
        s = lib.scores(model, "holdout", feature_set)
        return float(roc_auc_score(s.y, s.score))
    except Exception:  # noqa: BLE001
        return float("nan")


def comparison_data(model: str, feature_set: str, comparison: str):
    """Outcome, score, protected flag and group label for the rows of one comparison."""
    s = lib.scores(model, "holdout", feature_set)
    g = lib.groups(tuple(s.index))
    attribute = COMPARISON_KEY[comparison][0]
    index, protected = comparison_rows(g, COMPARISON_KEY[comparison])
    return (
        s.y.loc[index].to_numpy(),
        s.score.loc[index].to_numpy(dtype=float),
        protected.to_numpy(),
        g.loc[index, attribute].to_numpy(),
    )


def _rates(y, flag) -> dict:
    def share(event, cond):
        return float(event[cond].mean()) if cond.any() else np.nan

    return {
        "Flag rate": share(flag, np.ones_like(flag, dtype=bool)),
        "False-positive rate": share(flag, y == 0),
        "False-negative rate": share(~flag, y == 1),
        "Precision (PPV)": share(y == 1, flag),
        "Re-offence rate": share(y == 1, np.ones_like(flag, dtype=bool)),
    }


@st.cache_data(show_spinner=False)
def group_rates(model: str, feature_set: str, comparison: str, threshold: float) -> pd.DataFrame:
    y, score, protected, _ = comparison_data(model, feature_set, comparison)
    flag = score >= threshold
    _, p, r = COMPARISON_KEY[comparison]
    rows = []
    for name, mask in ((p, protected), (r, ~protected)):
        for metric, value in _rates(y[mask], flag[mask]).items():
            rows.append({"group": name, "metric": metric, "value": value, "n": int(mask.sum())})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def sweep(model: str, feature_set: str, comparison: str) -> pd.DataFrame:
    """Rate gaps (protected - reference) across thresholds 0.05-0.80."""
    y, score, protected, _ = comparison_data(model, feature_set, comparison)
    rows = []
    for t in np.round(np.arange(0.05, 0.8001, 0.01), 2):
        flag = score >= t
        a, b = _rates(y[protected], flag[protected]), _rates(y[~protected], flag[~protected])
        for metric in (
            "Flag rate",
            "False-positive rate",
            "False-negative rate",
            "Precision (PPV)",
        ):
            rows.append({"threshold": t, "metric": metric, "gap": a[metric] - b[metric]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def calibration(model: str, feature_set: str, comparison: str) -> pd.DataFrame:
    y, score, _, labels = comparison_data(model, feature_set, comparison)
    return calibration_by_group(y, score, labels, n_bins=10, min_bin=20)


@st.cache_data(show_spinner=False)
def agreement(feature_set: str, threshold: float) -> dict:
    """Share of defendants on whom the models make the same decision: all three, and pairs."""
    try:
        flags = pd.concat(
            [
                lib.scores(m, "holdout", feature_set).score.ge(threshold).rename(m)
                for m in lib.MODELS
            ],
            axis=1,
        ).dropna()
    except Exception:  # noqa: BLE001
        return {}
    out = {"All three": float(flags.nunique(axis=1).eq(1).mean())}
    for i, a in enumerate(lib.MODELS):
        for b in lib.MODELS[i + 1 :]:
            out[f"{a} = {b}"] = float((flags[a] == flags[b]).mean())
    return out


def optional_csv(path: Path) -> pd.DataFrame | None:
    try:
        return lib.read_csv(str(path)) if path.exists() else None
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------ controls

with st.container(border=True):
    threshold = lib.threshold_control(
        "fair", help_text="Every gap, test and table below is recomputed at this threshold."
    )
    c1, c2 = st.columns([2, 3])
    with c1:
        comparison = st.selectbox(
            "Comparison (protected vs reference)",
            lib.COMPARISONS,
            index=lib.COMPARISONS.index(lib.PRIMARY),
            key="fair_comparison",
            help="African-American vs Caucasian is the primary comparison. Asian and Native "
            "American defendants are too few to test.",
        )
    with c2:
        models = lib.model_picker("fair_models")

if not models:
    st.info("Pick at least one model.")
    st.stop()

_, PROT, REF = COMPARISON_KEY[comparison]
IS_PRIMARY = comparison == lib.PRIMARY

with st.spinner("Running the fairness protocol at this threshold..."):
    # Every variant for every model: the takeaways speak for all three models whatever is picked.
    everything = protocol(pairs_for(lib.ALL_MODELS, BASE_SETS + FIX_SETS), threshold)

if everything.empty:
    st.error("The fairness protocol could not be computed.")
    st.stop()

here = everything[everything.comparison == comparison]


def pick(metric: str, model: str, feature_set: str = "race_aware") -> pd.Series | None:
    rows = here[(here.metric == metric) & (here.model == model) & (here.feature_set == feature_set)]
    return None if rows.empty else rows.iloc[0]


def values(metric: str, column: str, feature_set: str = "race_aware", models=lib.MODELS):
    out = []
    for m in models:
        row = pick(metric, m, feature_set)
        out.append(np.nan if row is None else row[column])
    return np.array(out, dtype=float)


degenerate = here[(here.metric == "statistical_parity") & here.degenerate]
degenerate = degenerate[degenerate.model.isin(models)]
if len(degenerate):
    lib.caveat(
        f"At threshold {threshold:.3f} some variants flag (or release) at least 99% of "
        "defendants, so their tests cannot distinguish anyone: "
        + ", ".join(sorted(degenerate.label.unique()))
        + ". Their gaps are close to zero for a trivial reason."
    )

tabs = st.tabs(
    [
        "Why priors",
        "Step 1 — Is it unfair?",
        "Step 2 — Why?",
        "Step 3 — Can we fix it?",
        "Impossibility",
    ]
)

# =========================================================================== Why priors

with tabs[0]:
    data = cohort()
    groups3 = ["African-American", "Caucasian", "Other groups"]
    by_group = data.groupby("group").agg(
        n=("y", "size"), priors=("priors", "mean"), base=("y", "mean")
    )
    aa, ca = by_group.loc["African-American"], by_group.loc["Caucasian"]
    many = data.assign(many=data.priors >= 4).groupby("group")["many"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean priors, African-American", f"{aa.priors:.1f}")
    m2.metric("Mean priors, Caucasian", f"{ca.priors:.1f}")
    m3.metric("Re-offence rate, African-American", lib.fmt_pct(aa.base))
    m4.metric("Re-offence rate, Caucasian", lib.fmt_pct(ca.base))

    left, right = st.columns(2)
    with left:
        banded = data.assign(band=pd.cut(data.priors, PRIORS_BANDS[0], labels=PRIORS_BANDS[1]))
        share = (
            (
                banded.groupby(["group", "band"], observed=False).size()
                / banded.groupby("group").size()
            )
            .rename("share")
            .reset_index()
        )
        fig = go.Figure()
        for g in groups3:
            part = share[share.group == g]
            fig.add_bar(
                x=part.band.astype(str),
                y=part.share,
                name=g,
                marker_color=lib.RACE_COLOUR[g],
                hovertemplate=f"{g}<br>%{{x}} priors: %{{y:.1%}} of the group<extra></extra>",
            )
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
        fig.update_xaxes(title="Number of prior offences")
        fig.update_yaxes(title="Share of the group", tickformat=".0%")
        lib.show(lib.style(fig, "Priors by race (whole cohort)"))
    with right:
        rate = (
            banded.groupby(["group", "band"], observed=False)
            .agg(rate=("y", "mean"), n=("y", "size"))
            .reset_index()
        )
        fig = go.Figure()
        for g in groups3:
            part = rate[(rate.group == g) & (rate.n >= 20)]
            fig.add_scatter(
                x=part.band.astype(str),
                y=part.rate,
                name=g,
                mode="lines+markers",
                line={"color": lib.RACE_COLOUR[g], "width": 2},
                marker={"size": 9},
                customdata=part.n,
                hovertemplate=f"{g}<br>%{{x}} priors: %{{y:.1%}} re-offend "
                "(n = %{customdata})<extra></extra>",
            )
        fig.update_xaxes(title="Number of prior offences")
        fig.update_yaxes(title="Two-year re-offence rate", tickformat=".0%", rangemode="tozero")
        lib.show(lib.style(fig, "Re-offence rate by priors, per race"))

    leak = leakage()
    if len(leak):
        l1, l2 = st.columns([3, 2])
        with l1:
            leak = leak.assign(name=leak.feature_set.map(lib.FEATURE_SETS))
            fig = go.Figure(
                go.Bar(
                    x=leak.auc,
                    y=leak.name,
                    orientation="h",
                    marker_color=lib.INK_2,
                    text=[f"{a:.3f}" for a in leak.auc],
                    textposition="outside",
                    hovertemplate="%{y}: race predictable with AUC %{x:.3f}<extra></extra>",
                )
            )
            fig.add_vline(x=0.5, line_dash="dot", line_color=lib.MUTED)
            fig.update_xaxes(
                range=[0.45, 0.72], title="AUC of predicting race (0.5 = no race signal)"
            )
            fig.update_yaxes(autorange="reversed")
            lib.show(lib.style(fig, "How much race each feature set leaks", height=260))
        with l2:
            st.markdown(
                "<p class='small-note'>A logistic model tries to tell African-American from "
                "Caucasian defendants using only the features left in each set (race columns "
                "always removed; 5-fold cross-validation on the training split). An AUC above "
                "0.5 means race can still be read off the other features.</p>",
                unsafe_allow_html=True,
            )
        leak_txt = (
            f" Even with race removed, race can be predicted from the remaining features with "
            f"AUC {leak.auc.iloc[1]:.2f}; removing priors too lowers it to {leak.auc.iloc[2]:.2f}."
        )
    else:
        leak_txt = ""

    lib.takeaway(
        f"African-American defendants have on average <b>{aa.priors:.1f}</b> prior offences "
        f"against <b>{ca.priors:.1f}</b> for Caucasian defendants ({many['African-American']:.0%} "
        f"vs {many['Caucasian']:.0%} have 4 or more), and priors is the strongest predictor of "
        f"re-offending. Their observed re-offence rate is also higher ({aa.base:.0%} vs "
        f"{ca.base:.0%}). So a model that never sees race still learns race through priors."
        + leak_txt
    )

# ============================================================================ Step 1

with tabs[1]:
    sets = st.multiselect(
        "Feature sets",
        BASE_SETS,
        default=BASE_SETS,
        format_func=lambda fs: lib.FEATURE_SETS[fs],
        key="fair_sets",
        help="The COMPAS tool's decision does not depend on our feature sets: it appears once.",
    )
    chosen = here[here.set_index(["model", "feature_set"]).index.isin(pairs_for(models, sets))]
    order = [row_label(m, fs) for m, fs in pairs_for(models, sets)]

    if chosen.empty:
        st.info("Pick at least one feature set (or include the COMPAS tool).")
    else:
        panels = [
            ("statistical_parity", "Flag-rate gap"),
            ("fpr", "False-positive-rate gap"),
            ("fnr", "False-negative-rate gap"),
            ("conditional_statistical_parity", "Same-profile odds ratio"),
        ]
        fig = make_subplots(
            rows=1,
            cols=4,
            shared_yaxes=True,
            horizontal_spacing=0.025,
            subplot_titles=[t for _, t in panels],
        )
        for col, (metric, _) in enumerate(panels, start=1):
            part = chosen[chosen.metric == metric]
            for model in [m for m in lib.ALL_MODELS if m in models]:
                q = part[part.model == model]
                if metric == "conditional_statistical_parity":
                    ok = np.isfinite(q.mh_odds_ratio) & (q.mh_odds_ratio > 0)
                    q = q[ok]
                    x = np.log(q.mh_odds_ratio)
                    lo = np.log(q.mh_or_low.where(q.mh_or_low > 0))
                    hi = np.log(q.mh_or_high.where(q.mh_or_high > 0))
                    p = q.cmh_p
                    shown = q.mh_odds_ratio
                    hover = "%{y}<br>odds ratio %{customdata[0]:.2f} [%{customdata[1]:.2f}, %{customdata[2]:.2f}]<br>CMH p = %{customdata[3]:.3g} · Hurlin p = %{customdata[4]:.3g}<extra></extra>"  # noqa: E501
                    custom = np.c_[shown, q.mh_or_low, q.mh_or_high, p, q.hurlin_p]
                else:
                    x, lo, hi, p = q.gap, q.ci_low, q.ci_high, q.p_value
                    hover = "%{y}<br>gap %{x:+.3f} [%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]<br>p = %{customdata[2]:.3g} · Holm p = %{customdata[3]:.3g}<extra></extra>"  # noqa: E501
                    custom = np.c_[lo, hi, p, q.p_holm]
                if q.empty:
                    continue
                significant = (p < ALPHA).to_numpy()
                symbol = [lib.SYMBOL[model] + ("" if s else "-open") for s in significant]
                fig.add_scatter(
                    x=x,
                    y=q.label,
                    mode="markers",
                    name=model,
                    legendgroup=model,
                    showlegend=col == 1,
                    marker={
                        "color": lib.COLOUR[model],
                        "symbol": symbol,
                        "size": 11,
                        "line": {"width": 2, "color": lib.COLOUR[model]},
                    },
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": (hi - x).to_numpy(),
                        "arrayminus": (x - lo).to_numpy(),
                        "color": lib.COLOUR[model],
                        "thickness": 1.5,
                        "width": 0,
                    },
                    customdata=custom,
                    hovertemplate=hover,
                    row=1,
                    col=col,
                )
            fig.add_vline(x=0, line_dash="dot", line_color=lib.MUTED, row=1, col=col)
        ticks = [0.25, 0.5, 1, 2, 4, 8, 16]
        fig.update_xaxes(tickvals=np.log(ticks), ticktext=[str(t) for t in ticks], row=1, col=4)
        fig.update_xaxes(tickformat="+.2f", row=1, col=1)
        fig.update_xaxes(tickformat="+.2f", row=1, col=2)
        fig.update_xaxes(tickformat="+.2f", row=1, col=3)
        fig.update_yaxes(categoryorder="array", categoryarray=order[::-1])
        fig = lib.style(fig, height=110 + 34 * len(order))
        fig.update_annotations(font_size=13, xanchor="left", x=None)
        for i, ann in enumerate(fig.layout.annotations):
            ann.x = fig.layout[f"xaxis{'' if i == 0 else i + 1}"].domain[0]
        lib.show(fig)
        st.markdown(
            f"<p class='small-note'>{PROT} minus {REF}, at threshold {threshold:.3f}, with 95% "
            "confidence intervals. <b>Filled</b> marker: significant (p &lt; 0.05); "
            "<b>hollow</b>: not significant. Flag-rate gap: how much more often the group is "
            "flagged. False-positive-rate gap: among people who did <i>not</i> re-offend, how "
            "much more often they were flagged anyway. False-negative-rate gap: among people "
            "who did re-offend, how much more often they were released (negative = flagged "
            "more). Same-profile odds ratio (log scale; 1 = no gap): the Mantel-Haenszel odds "
            "of being flagged for the group, among defendants with the same priors band, age "
            "band and charge degree (significance: CMH test).</p>",
            unsafe_allow_html=True,
        )

    # ---- takeaway (all three models, whatever is picked)
    sp = values("statistical_parity", "gap")
    sp_blind = values("statistical_parity", "gap", "race_blind")
    fpr = values("fpr", "gap")
    ors = values("conditional_statistical_parity", "mh_odds_ratio")
    csp_p = values("conditional_statistical_parity", "p_value")
    sp_p = values("statistical_parity", "p_value")
    tool_sp, tool_fpr = pick("statistical_parity", "COMPAS tool"), pick("fpr", "COMPAS tool")
    tool_csp = pick("conditional_statistical_parity", "COMPAS tool")
    n_sig_sp, n_sig_csp = int(np.nansum(sp_p < ALPHA)), int(np.nansum(csp_p < ALPHA))
    blind_shift = np.nanmax(np.abs(sp_blind - sp)) if np.isfinite(sp_blind - sp).any() else np.nan

    def direction(gaps) -> str | None:
        g = pd.Series(gaps, dtype=float).dropna()
        if g.empty:
            return None
        return "more" if (g > 0).all() else "less" if (g < 0).all() else None

    sp_dir, fpr_dir = direction(sp), direction(fpr)
    txt = (
        f"At threshold {threshold:.3f}, the three models' flag-rate gap ({PROT} minus {REF}) "
        f"is <b>{span(sp)}</b>"
        + (f": {PROT} defendants are flagged {sp_dir} often" if sp_dir else "")
        + f" ({n_sig_sp} of 3 significant; COMPAS tool "
        f"{fmt_signed(None if tool_sp is None else tool_sp.gap)}). "
        f"The false-positive-rate gap is {span(fpr)} (COMPAS tool "
        f"{fmt_signed(None if tool_fpr is None else tool_fpr.gap)})"
        + (
            f": people who never re-offended are flagged {fpr_dir} often if they are {PROT}. "
            if fpr_dir
            else ". "
        )
    )
    if np.isfinite(blind_shift):
        txt += f"Removing race changes the flag-rate gap by at most {blind_shift:.2f}" + (
            ": race is carried by the other features (mainly priors). "
            if IS_PRIMARY and blind_shift < 0.03
            else ". "
        )
    tool_or = "–" if tool_csp is None else f"{tool_csp.mh_odds_ratio:.2f}"
    tool_p = None if tool_csp is None else tool_csp.p_value
    tool_sig = tool_p is not None and tool_p < ALPHA
    txt += (
        f"<br>Among defendants with the <b>same priors, age and charge</b>, {n_sig_csp} of 3 "
        f"models keep a significant gap (odds ratios {span(ors, signed=False)}); the COMPAS "
        f"tool {'does' if tool_sig else 'does not'} (odds ratio {tool_or}, p = {fmt_p(tool_p)})."
    )
    if IS_PRIMARY and n_sig_csp == 0 and tool_sig:
        txt += (
            " The models' gap is explained by priors, age and charge; the COMPAS tool's is not "
            "entirely."
        )
    lib.takeaway(txt)

    # ---- verdict matrix
    st.subheader("Verdicts of the protocol")
    st.markdown(
        "<p class='small-note'>Each test asks whether the model treats the two groups the "
        'same. "Unfair" = rejected at 5% after the Holm correction across the four '
        'comparisons (the protocol\'s decision). "Certified fair" = the TOST equivalence '
        f"test shows the gap is within ±{TOST_DELTA:.2f}.</p>",
        unsafe_allow_html=True,
    )
    if not chosen.empty:

        def verdict(row) -> str:
            if pd.isna(row.p_holm):
                return "–"
            if row.p_holm < ALPHA:
                return f"Unfair (p = {fmt_p(row.p_holm)})"
            if row.get("certified_fair") == True:  # noqa: E712 - numpy bool or NaN
                return f"Certified fair (p = {fmt_p(row.p_holm)})"
            return f"Not rejected (p = {fmt_p(row.p_holm)})"

        vt = chosen.assign(verdict=chosen.apply(verdict, axis=1)).pivot_table(
            index="label", columns="metric", values="verdict", aggfunc="first"
        )
        vt = vt.reindex(index=order, columns=list(METRIC_LABEL)).rename(columns=METRIC_LABEL)
        vt.index.name = None

        def colour(v) -> str:
            if isinstance(v, str) and v.startswith("Unfair"):
                return "color: #b3261e; font-weight: 600"
            if isinstance(v, str) and v.startswith("Certified"):
                return "color: #0f7b52; font-weight: 600"
            return "color: #52514e"

        st.dataframe(vt.style.map(colour), width="stretch")

        with st.expander("Detailed results (every number of the protocol)"):
            detail = chosen.copy()
            detail["Variant"] = detail.label
            detail["Metric"] = detail.metric.map(METRIC_LABEL)
            detail = detail.set_index(["Variant", "Metric"])[
                [
                    "rate_protected",
                    "rate_reference",
                    "gap",
                    "ci_low",
                    "ci_high",
                    "p_value",
                    "p_holm",
                    "minimum_delta",
                    "certified_fair",
                    "mh_odds_ratio",
                    "mh_or_low",
                    "mh_or_high",
                    "cmh_p",
                    "hurlin_p",
                    "breslow_day_p",
                    "selection_rate",
                    "degenerate",
                ]
            ]
            detail = detail.reindex(
                pd.MultiIndex.from_product([order, list(METRIC_LABEL.values())])
            ).dropna(how="all")
            detail.columns = [
                f"Rate, {PROT}",
                f"Rate, {REF}",
                "Gap",
                "CI low",
                "CI high",
                "p",
                "p (Holm)",
                "TOST minimum Δ",
                "Certified fair",
                "MH odds ratio",
                "OR low",
                "OR high",
                "CMH p",
                "Hurlin p",
                "Breslow-Day p",
                "Overall flag rate",
                "Degenerate",
            ]
            num = {c: st.column_config.NumberColumn(format="%.3f") for c in detail.columns}
            for c in ("p", "p (Holm)", "CMH p", "Hurlin p", "Breslow-Day p"):
                num[c] = st.column_config.NumberColumn(format="%.2e")
            st.dataframe(detail.reset_index(), column_config=num, width="stretch")
            st.caption(
                "TOST minimum Δ: the smallest tolerance within which the gap could be "
                f"certified equal (the protocol uses {TOST_DELTA:.2f}). Breslow-Day p < 0.05 "
                "means the same-profile odds ratio differs across strata."
            )

    # ---- group rates
    st.subheader("Rates for each group")
    g1, g2 = st.columns(2)
    with g1:
        rate_model = st.selectbox("Model", models, key="fair_rate_model")
    with g2:
        rate_set = st.selectbox(
            "Feature set",
            BASE_SETS + FIX_SETS,
            format_func=lambda fs: lib.FEATURE_SETS[fs],
            key="fair_rate_set",
            disabled=rate_model == "COMPAS tool",
        )
    rate_set = "race_aware" if rate_model == "COMPAS tool" else rate_set
    try:
        gr = group_rates(rate_model, rate_set, comparison, float(threshold))
    except Exception:  # noqa: BLE001
        gr = pd.DataFrame()
    if gr.empty:
        st.info("No predictions for this variant.")
    else:
        fig = go.Figure()
        for role, name in (("protected", PROT), ("reference", REF)):
            part = gr[gr.group == name]
            fig.add_bar(
                x=part.metric,
                y=part.value,
                name=f"{name} (n = {part.n.iloc[0]})",
                marker_color=group_colour(name, role),
                text=[lib.fmt_pct(v) for v in part.value],
                textposition="outside",
                hovertemplate=f"{name}<br>%{{x}}: %{{y:.1%}}<extra></extra>",
            )
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.08], title=None)
        lib.show(
            lib.style(
                fig, f"{row_label(rate_model, rate_set)} at threshold {threshold:.3f}", height=380
            )
        )
        st.caption(
            "Re-offence rate is the observed base rate of each group: it does not depend on "
            "the model. Precision (PPV): of those flagged, the share who did re-offend."
        )

# ============================================================================ Step 2

with tabs[2]:
    st.markdown(
        "**Fairness partial dependence (FPDP) on priors.** Every defendant's number of priors "
        "is set to the same value, the model re-predicts, and the statistical-parity test is "
        "re-run. If the gap disappears when priors is equalised, priors is what drives it."
    )
    try:
        fp = lib.comparison("fpdp_priors.csv")
        fp = fp[fp.model.isin(models)]
    except Exception:  # noqa: BLE001
        fp = pd.DataFrame()
    if fp.empty:
        st.info("FPDP results are not available for the selected models.")
    else:
        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            for model in [m for m in lib.MODELS if m in set(fp.model)]:
                q = fp[fp.model == model].sort_values("value")
                symbol = [lib.SYMBOL[model] + ("-open" if d else "") for d in q.degenerate]
                fig.add_scatter(
                    x=q.value,
                    y=q.p_value.clip(lower=1e-16),
                    name=model,
                    mode="lines+markers",
                    line={"color": lib.COLOUR[model], "width": 2},
                    marker={
                        "symbol": symbol,
                        "size": 10,
                        "color": lib.COLOUR[model],
                        "line": {"width": 2, "color": lib.COLOUR[model]},
                    },
                    customdata=np.c_[q.selection_rate, q.degenerate],
                    hovertemplate=f"{model}<br>priors = %{{x:.0f}}: p = %{{y:.2g}}"
                    "<br>flag rate %{customdata[0]:.1%}<extra></extra>",
                )
            fig.add_hline(
                y=ALPHA,
                line_dash="dash",
                line_color="#b3261e",
                annotation_text="5% level",
                annotation_position="top left",
            )
            fig.update_yaxes(type="log", title="Parity test p-value (log)", exponentformat="e")
            fig.update_xaxes(title="Priors set to this value for everyone")
            lib.show(lib.style(fig, "Parity test p-value vs equalised priors"))
        with right:
            fig = go.Figure()
            for model in [m for m in lib.MODELS if m in set(fp.model)]:
                q = fp[fp.model == model].sort_values("value")
                symbol = [lib.SYMBOL[model] + ("-open" if d else "") for d in q.degenerate]
                fig.add_scatter(
                    x=q.value,
                    y=q.selection_rate,
                    name=model,
                    mode="lines+markers",
                    line={"color": lib.COLOUR[model], "width": 2},
                    marker={
                        "symbol": symbol,
                        "size": 10,
                        "color": lib.COLOUR[model],
                        "line": {"width": 2, "color": lib.COLOUR[model]},
                    },
                    hovertemplate=f"{model}<br>priors = %{{x:.0f}}: %{{y:.1%}} flagged"
                    "<extra></extra>",
                )
            fig.update_yaxes(title="Share of defendants flagged", tickformat=".0%", range=[0, 1.05])
            fig.update_xaxes(title="Priors set to this value for everyone")
            lib.show(lib.style(fig, "Flag rate vs equalised priors"))
        st.markdown(
            "<p class='small-note'>Primary comparison (African-American vs Caucasian) at "
            "threshold 0.5, precomputed: each point re-runs the model on counterfactual data. "
            "Hollow points are degenerate (at least 99% flagged or released): the test "
            "passes there for a trivial reason and is ignored.</p>",
            unsafe_allow_html=True,
        )

        # Baseline statistic (actual priors) from the precomputed protocol at 0.5.
        try:
            base = lib.comparison("fairness.csv")
            base = base[
                (base.threshold_name == "0.5")
                & (base.comparison == lib.PRIMARY)
                & (base.metric == "statistical_parity")
                & (base.feature_set == "race_aware")
            ].set_index("model")
        except Exception:  # noqa: BLE001
            base = pd.DataFrame()
        usable = fp[~fp.degenerate]
        best = usable.loc[usable.groupby("model").p_value.idxmax()].set_index("model")
        lines = []
        for model in best.index:
            b = best.loc[model]
            stat0 = base.hurlin_statistic.get(model, np.nan) if len(base) else np.nan
            lines.append(
                f"{model}: test statistic {stat0:.0f} with actual priors, "
                f"{b.statistic:.0f} when everyone has {b.value:.0f} prior(s) "
                f"(p {fmt_p(b.p_value)})"
            )
        any_candidate = bool((usable.p_value > ALPHA).any())
        lib.takeaway(
            "At threshold 0.5, African-American vs Caucasian: equalising priors removes most "
            "of the disparity. "
            + "; ".join(lines)
            + ". "
            + (
                "At some value the test is no longer rejected, so priors is a candidate "
                "variable in the course's sense."
                if any_candidate
                else "The p-value never climbs above 5% at a non-degenerate value, so by the "
                "strict rule priors alone is not a 'candidate variable': a small residual gap "
                "remains (age and charge also differ by race). But it is by far the variable "
                "that carries the race gap."
            )
        )

    # Live counterpart: the priors-set-to-1 model is FPDP at value 1, at the chosen threshold.
    live = here[(here.metric == "statistical_parity") & here.model.isin(lib.MODELS)]
    live = live.pivot_table(index="model", columns="feature_set", values="gap")
    if {"race_aware", "priors_set_1"} <= set(live.columns):
        live = live.reindex([m for m in lib.MODELS if m in live.index])
        st.markdown(
            f"**Live, at threshold {threshold:.3f} ({comparison}):** flag-rate gap with "
            "actual priors vs with everyone's priors set to 1."
        )
        cols = st.columns(len(live))
        for col, (model, row) in zip(cols, live.iterrows(), strict=True):
            col.metric(
                model,
                fmt_signed(row.priors_set_1),
                delta=f"{row.priors_set_1 - row.race_aware:+.2f} vs actual priors",
                delta_color="inverse" if row.race_aware > 0 else "normal",
            )

    t06 = optional_csv(lib.TABPFN_ART / "analysis" / "t06_fpdp" / "fpdp_candidates.csv")
    if t06 is not None:
        with st.expander("TabPFN group: FPDP on every feature (candidate variables)"):
            st.dataframe(t06, width="stretch", hide_index=True)
            st.caption(
                "is_candidate: fairness is rejected on the actual data and some non-degenerate "
                "value of the feature lifts the p-value above 5%."
            )

# ============================================================================ Step 3

with tabs[3]:
    variants = ["race_aware", *FIX_SETS]
    show_others = st.toggle("Also show the race-blind variants", value=False, key="fair_others")
    if show_others:
        variants = ["race_aware", "race_blind", "race_priors_blind", *FIX_SETS]
    rows = []
    for model in [m for m in lib.MODELS if m in models]:
        for fs in variants:
            sp_row, fpr_row = pick("statistical_parity", model, fs), pick("fpr", model, fs)
            csp_row, fnr_row = (
                pick("conditional_statistical_parity", model, fs),
                pick("fnr", model, fs),
            )
            if sp_row is None:
                continue
            rows.append(
                {
                    "model": model,
                    "feature_set": fs,
                    "auc": auc_of(model, fs),
                    "gap": sp_row.gap,
                    "gap_p": sp_row.p_value,
                    "fpr_gap": fpr_row.gap,
                    "fnr_gap": fnr_row.gap,
                    "csp_p": csp_row.p_value,
                    "flag_rate": sp_row.selection_rate,
                }
            )
    if "COMPAS tool" in models and pick("statistical_parity", "COMPAS tool") is not None:
        t_sp = pick("statistical_parity", "COMPAS tool")
        rows.append(
            {
                "model": "COMPAS tool",
                "feature_set": "race_aware",
                "auc": auc_of("COMPAS tool", "race_aware"),
                "gap": t_sp.gap,
                "gap_p": t_sp.p_value,
                "fpr_gap": pick("fpr", "COMPAS tool").gap,
                "fnr_gap": pick("fnr", "COMPAS tool").gap,
                "csp_p": pick("conditional_statistical_parity", "COMPAS tool").p_value,
                "flag_rate": t_sp.selection_rate,
            }
        )
    tradeoff = pd.DataFrame(rows)

    if tradeoff.empty:
        st.info("Nothing to show for this selection.")
    else:
        fig = go.Figure()
        for model in [m for m in lib.MODELS if m in set(tradeoff.model)]:
            q = (
                tradeoff[tradeoff.model == model]
                .set_index("feature_set")
                .reindex(variants)
                .dropna(subset=["auc"])
            )
            fig.add_scatter(
                x=q.gap,
                y=q.auc,
                mode="markers",
                name=model,
                marker={
                    "color": lib.COLOUR[model],
                    "symbol": [
                        lib.SYMBOL[model] + ("" if fs == "race_aware" else "-open")
                        for fs in q.index
                    ],
                    "size": 13,
                    "line": {"width": 2, "color": lib.COLOUR[model]},
                },
                customdata=np.c_[[lib.FEATURE_SETS[fs] for fs in q.index], q.fpr_gap, q.gap_p],
                hovertemplate=f"{model} · %{{customdata[0]}}<br>AUC %{{y:.3f}}<br>flag-rate gap "
                "%{x:+.3f} (p = %{customdata[2]:.2g})<br>FPR gap %{customdata[1]:+.3f}"
                "<extra></extra>",
            )
            if "race_aware" in q.index:
                b = q.loc["race_aware"]
                for fs in [v for v in q.index if v != "race_aware"]:
                    fig.add_annotation(
                        x=q.loc[fs, "gap"],
                        y=q.loc[fs, "auc"],
                        ax=b.gap,
                        ay=b.auc,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.2,
                        arrowwidth=1.5,
                        arrowcolor=lib.COLOUR[model],
                        opacity=0.55,
                        text="",
                    )
        # One label per variant cluster, placed at the cluster's centre (no per-point clutter).
        models_only = tradeoff[tradeoff.model != "COMPAS tool"]
        for fs, part in models_only.groupby("feature_set"):
            fig.add_annotation(
                x=part.gap.mean(),
                y=part.auc.max(),
                text=f"<b>{lib.FEATURE_SETS[fs]}</b>",
                showarrow=False,
                yshift=18,
                font={"size": 12, "color": lib.INK_2},
            )
        tool = tradeoff[tradeoff.model == "COMPAS tool"]
        if len(tool):
            fig.add_scatter(
                x=tool.gap,
                y=tool.auc,
                mode="markers+text",
                name="COMPAS tool",
                marker={
                    "color": lib.COLOUR["COMPAS tool"],
                    "symbol": lib.SYMBOL["COMPAS tool"],
                    "size": 14,
                },
                text=["COMPAS tool"],
                textposition="bottom center",
                textfont={"color": lib.INK_2},
                hovertemplate="COMPAS tool<br>AUC %{y:.3f}<br>flag-rate gap %{x:+.3f}"
                "<extra></extra>",
            )
            fig.add_hline(y=float(tool.auc.iloc[0]), line_dash="dot", line_color=lib.MUTED)
        fig.add_vline(x=0, line_dash="dot", line_color=lib.MUTED)
        fig.update_xaxes(title=f"Flag-rate gap, {PROT} minus {REF} (0 = parity)", tickformat="+.2f")
        fig.update_yaxes(title="AUC (ranking quality, threshold-free)")
        lib.show(lib.style(fig, f"Accuracy vs fairness at threshold {threshold:.3f}", height=500))
        st.caption(
            "Filled marker: the full model. Arrows lead to its fixes (hollow): drop priors and "
            "retrain (race kept), or keep the model and set everyone's priors to 1. Dotted "
            "line: the COMPAS tool's AUC."
        )

        table = tradeoff.assign(
            Model=tradeoff.model,
            Variant=[
                "Its own decision" if m == "COMPAS tool" else lib.FEATURE_SETS[fs]
                for m, fs in zip(tradeoff.model, tradeoff.feature_set, strict=True)
            ],
        )[["Model", "Variant", "auc", "flag_rate", "gap", "gap_p", "fpr_gap", "fnr_gap", "csp_p"]]
        table.columns = [
            "Model",
            "Variant",
            "AUC",
            "Flag rate",
            "Flag-rate gap",
            "p (gap)",
            "FPR gap",
            "FNR gap",
            "Same-profile p",
        ]
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "AUC": st.column_config.NumberColumn(format="%.3f"),
                "Flag rate": st.column_config.NumberColumn(format="%.3f"),
                "Flag-rate gap": st.column_config.NumberColumn(format="%+.3f"),
                "FPR gap": st.column_config.NumberColumn(format="%+.3f"),
                "FNR gap": st.column_config.NumberColumn(format="%+.3f"),
                "p (gap)": st.column_config.NumberColumn(format="%.2e"),
                "Same-profile p": st.column_config.NumberColumn(format="%.2e"),
            },
        )

    # ---- takeaway (all three models)
    gap_blind = values("statistical_parity", "gap", "priors_blind")
    gap_one = values("statistical_parity", "gap", "priors_set_1")
    auc_one = np.array([auc_of(m, "priors_set_1") for m in lib.MODELS])
    auc_base = np.array([auc_of(m, "race_aware") for m in lib.MODELS])
    auc_tool = auc_of("COMPAS tool", "race_aware")
    agree_one = agreement("priors_set_1", float(threshold))
    agree_base = agreement("race_aware", float(threshold))
    if agree_one and agree_base:
        with st.expander("Do the models still differ once priors is neutralised?"):
            agree = pd.DataFrame(
                {"All features": agree_base, "Priors set to 1": agree_one}
            ).rename_axis("Models making the same decision")
            st.dataframe(
                agree,
                width="stretch",
                column_config={
                    c: st.column_config.NumberColumn(format="percent") for c in agree.columns
                },
            )
            st.caption(
                f"Share of the {len(lib.scores('LogReg'))} holdout defendants on whom the models "
                f"agree (flag or release) at threshold {threshold:.3f}."
            )
    txt = (
        f"<b>Dropping priors</b> and retraining moves the flag-rate gap from {span(sp)} to "
        f"<b>{span(gap_blind)}</b>"
        + (
            ": without priors the models lean on race itself. "
            if IS_PRIMARY and np.nanmin(gap_blind - sp) > 0
            else ". "
        )
        + f"<b>Setting everyone's priors to 1</b> brings it to <b>{span(gap_one)}</b>, but AUC "
        f"falls from {span(auc_base, 3, False)} to <b>{span(auc_one, 3, False)}</b>, "
        f"{'below' if np.nanmax(auc_one) < auc_tool else 'around'} the COMPAS tool "
        f"({auc_tool:.3f}). "
    )
    if agree_one and agree_base:
        pairs_one = {k: v for k, v in agree_one.items() if k != "All three"}
        top = max(pairs_one, key=pairs_one.get)
        txt += (
            f"With priors neutralised the three models agree on {agree_one['All three']:.1%} of "
            f"defendants ({top}: {pairs_one[top]:.1%}), against "
            f"{agree_base['All three']:.1%} with the full models"
            + (
                ": the choice of model matters little, with or without the fix. "
                if agree_one["All three"] >= 0.95
                else ". "
            )
        )
    if np.nanmax(auc_one) < np.nanmin(auc_base):
        txt += "Fairness here is bought with accuracy: there is no free fix."
    lib.takeaway(txt)

# ======================================================================== Impossibility

with tabs[4]:
    i1, i2 = st.columns(2)
    with i1:
        imp_model = st.selectbox(
            "Model", [m for m in lib.MODELS if m in models] or lib.MODELS, key="fair_imp_model"
        )
    with i2:
        imp_set = st.selectbox(
            "Feature set",
            BASE_SETS + FIX_SETS,
            format_func=lambda fs: lib.FEATURE_SETS[fs],
            key="fair_imp_set",
        )
    st.caption("The COMPAS tool gives a yes/no decision, not a score, so it cannot be swept.")

    try:
        sw = sweep(imp_model, imp_set, comparison)
        rates = group_rates(imp_model, imp_set, comparison, float(threshold))
    except Exception:  # noqa: BLE001
        sw, rates = pd.DataFrame(), pd.DataFrame()

    if sw.empty:
        st.info("No predictions for this variant.")
    else:
        base_p = rates[(rates.group == PROT) & (rates.metric == "Re-offence rate")].value.iloc[0]
        base_r = rates[(rates.group == REF) & (rates.metric == "Re-offence rate")].value.iloc[0]
        dashes = {
            "False-positive rate": "solid",
            "False-negative rate": "dash",
            "Precision (PPV)": "dot",
            "Flag rate": "dashdot",
        }
        fig = go.Figure()
        for metric, dash in dashes.items():
            q = sw[sw.metric == metric]
            fig.add_scatter(
                x=q.threshold,
                y=q.gap,
                name=f"{metric} gap",
                mode="lines",
                line={"color": lib.COLOUR[imp_model], "dash": dash, "width": 2.5},
                hovertemplate=f"{metric} gap<br>threshold %{{x:.2f}}: %{{y:+.3f}}<extra></extra>",
            )
            end = q.dropna(subset=["gap"]).tail(1)
            if len(end):
                fig.add_annotation(
                    x=end.threshold.iloc[0],
                    y=end.gap.iloc[0],
                    text=metric.split(" (")[0],
                    showarrow=False,
                    xanchor="left",
                    xshift=6,
                    font={"size": 11, "color": lib.INK_2},
                )
        fig.add_hline(y=0, line_color=lib.MUTED)
        fig.add_vline(
            x=threshold,
            line_dash="dot",
            line_color=lib.INK_2,
            annotation_text=f"chosen {threshold:.3f}",
            annotation_position="top",
        )
        fig.update_xaxes(title="Decision threshold", range=[0.04, 0.92])
        fig.update_yaxes(title=f"Gap, {PROT} minus {REF}", tickformat="+.2f")
        lib.show(
            lib.style(
                fig,
                f"Error-rate and precision gaps across thresholds · "
                f"{row_label(imp_model, imp_set)}",
                height=460,
            )
        )

        wide = sw.pivot_table(index="threshold", columns="metric", values="gap")
        worst = wide[["False-positive rate", "Precision (PPV)"]].abs().max(axis=1).dropna()
        best_t = float(worst.idxmin()) if len(worst) else np.nan

        cal = calibration(imp_model, imp_set, comparison)
        fig = go.Figure()
        for role, name in (("protected", PROT), ("reference", REF)):
            q = cal[cal.g == name]
            fig.add_scatter(
                x=q.predicted,
                y=q.observed,
                name=name,
                mode="lines+markers",
                line={"color": group_colour(name, role), "width": 2},
                marker={"size": 9},
                customdata=np.c_[q.n, q.band.astype(str)],
                hovertemplate=f"{name}<br>score band %{{customdata[1]}}<br>mean score "
                "%{x:.2f} → observed %{y:.1%} (n = %{customdata[0]})<extra></extra>",
            )
        fig.add_scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"color": lib.MUTED, "dash": "dot", "width": 1.5},
            hoverinfo="skip",
        )
        fig.update_xaxes(title="Predicted risk (mean score in the band)", range=[0, 1])
        fig.update_yaxes(title="Observed re-offence rate", tickformat=".0%", range=[0, 1])
        c_left, c_right = st.columns([3, 2])
        with c_left:
            lib.show(
                lib.style(
                    fig, "Calibration by group: does a score mean the same thing?", height=440
                )
            )
        with c_right:
            both = cal.pivot_table(
                index="band", columns="g", values=["observed", "n"], observed=True
            ).dropna()
            if len(both) and PROT in both["observed"] and REF in both["observed"]:
                diff = (both["observed"][PROT] - both["observed"][REF]).abs()
                weight = both["n"][PROT] + both["n"][REF]
                cal_gap = float((diff * weight).sum() / weight.sum())
            else:
                cal_gap = np.nan
            st.metric(f"Re-offence rate, {PROT}", lib.fmt_pct(base_p))
            st.metric(f"Re-offence rate, {REF}", lib.fmt_pct(base_r))
            st.metric(
                "Calibration difference within a score band",
                "–" if np.isnan(cal_gap) else f"{cal_gap * 100:.1f} points",
                help="Average absolute difference in observed re-offence rate between the two "
                "groups, within the same score band (weighted by band size; bands with at "
                "least 20 defendants per group).",
            )

        fpr_at = wide.loc[best_t, "False-positive rate"] if np.isfinite(best_t) else np.nan
        ppv_at = wide.loc[best_t, "Precision (PPV)"] if np.isfinite(best_t) else np.nan
        lib.takeaway(
            f"The two groups re-offend at different rates ({base_p:.0%} vs {base_r:.0%}). "
            "Chouldechova's result: when base rates differ, a score cannot be calibrated "
            "(same score, same risk) <i>and</i> have equal false-positive and false-negative "
            "rates. "
            + (
                ""
                if np.isnan(cal_gap)
                else f"Here the score is {'close to' if cal_gap < 0.1 else 'not well'} "
                f"calibrated (about {cal_gap * 100:.0f} points apart within a score band). "
            )
            + (
                f"Across thresholds, the best compromise ({best_t:.2f}) still leaves an FPR "
                f"gap of {fmt_signed(fpr_at)} and a precision gap of {fmt_signed(ppv_at)}: no "
                "threshold closes both. "
                if np.isfinite(best_t) and worst.min() >= 0.02
                else f"At threshold {best_t:.2f} both the FPR and precision gaps are small "
                f"({fmt_signed(fpr_at)}, {fmt_signed(ppv_at)}). "
            )
            + "Which fairness to require is a policy choice, not a modelling one."
        )

# ------------------------------------------------------------------ TabPFN group extras

EXTRAS = {
    "Statistical parity (every design)": "t01_statistical_parity/statistical_parity.csv",
    "Conditional parity": "t02_conditional_parity/conditional_parity.csv",
    "Equalized odds": "t03_equalized_odds/equalized_odds.csv",
    "Error-rate gaps": "t03_equalized_odds/error_rate_gaps.csv",
    "Fairness equivalence (TOST)": "t04_fairness_equivalence/equivalence.csv",
    "Ablation: fairness by feature set": "t05_ablation_fairness/ablation.csv",
    "FPDP curves": "t06_fpdp/fpdp_curves.csv",
    "Fairness stability across splits": "t09_fairness_stability/pairs_fairness.csv",
}
available = {k: v for k, v in EXTRAS.items() if (lib.TABPFN_ART / "analysis" / v).exists()}
if available:
    with st.expander("TabPFN group's detailed fairness tables (all designs and thresholds)"):
        name = st.selectbox("Table", list(available), key="fair_extra")
        extra = optional_csv(lib.TABPFN_ART / "analysis" / available[name])
        if extra is None:
            st.info("This table could not be read.")
        else:
            st.dataframe(extra, width="stretch", hide_index=True)
