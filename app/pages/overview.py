import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for lib

import lib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from compas_scoring.data import build_dataset  # noqa: E402

lib.setup_page(
    "Can a better model make COMPAS fair?",
    "LogReg (white box), XGBoost (black box) and TabPFN (foundation model), trained on the same "
    "6,172 defendants and tested on the same 1,852, against the COMPAS tool's own decisions.",
)

# ---------------------------------------------------------------------------- headline KPIs

perf = lib.comparison("performance.csv")
hold = perf[(perf.run == "holdout") & (perf.feature_set == "race_aware")]
at_half = hold[np.isclose(hold.threshold, 0.5)].set_index("model")
fair = lib.comparison("fairness.csv")
sp = fair[(fair.comparison == lib.PRIMARY) & (fair.metric == "statistical_parity")]

c1, c2, c3, c4 = st.columns(4)
best = at_half.loc[lib.MODELS, "auc"]
c1.metric(
    "AUC, our three models",
    f"{best.min():.3f}–{best.max():.3f}",
    f"+{best.min() - at_half.loc['COMPAS tool', 'auc']:.3f} vs COMPAS tool",
)
gaps = sp[(sp.feature_set == "race_aware") & sp.model.isin(lib.MODELS)].gap
c2.metric(
    "Extra flag rate, African-American",
    f"+{gaps.min():.0%} to +{gaps.max():.0%}",
    "vs Caucasian, threshold 0.5",
    delta_color="off",
)
fixed = perf[
    (perf.run == "holdout") & (perf.feature_set == "priors_set_1") & perf.model.isin(lib.MODELS)
]
fix_auc = fixed[np.isclose(fixed.threshold, 0.5)].auc
c3.metric(
    "AUC once priors is neutralised",
    f"{fix_auc.min():.2f}–{fix_auc.max():.2f}",
    "below the COMPAS tool (0.66)",
    delta_color="off",
)
data = build_dataset("race_aware")
race = data.groups.race
mean = data.X.Number_of_Priors.groupby(race).mean()
c4.metric(
    "Mean priors, African-American vs Caucasian",
    f"{mean['African-American']:.1f} vs {mean['Caucasian']:.1f}",
    "priors carries race",
    delta_color="off",
)

tool_gap = sp[(sp.model == "COMPAS tool") & np.isclose(sp.threshold, 0.5)].gap.iloc[0]
blind = sp[(sp.feature_set == "race_blind") & np.isclose(sp.threshold, 0.5)].gap
fix_gap = sp[(sp.feature_set == "priors_set_1") & np.isclose(sp.threshold, 0.5)].gap
lib.tldr(
    [
        f"<b>Accuracy:</b> AUC {best.min():.3f}–{best.max():.3f} for all three; COMPAS tool "
        f"{at_half.loc['COMPAS tool', 'auc']:.3f}.",
        f"<b>Bias:</b> African-American flagged +{gaps.min():.0%} to +{gaps.max():.0%} more "
        f"(tool +{tool_gap:.0%}).",
        f"<b>Cause:</b> priors ({mean['African-American']:.1f} vs {mean['Caucasian']:.1f} on "
        f"average). Race-blind gap still +{blind.min():.0%} to +{blind.max():.0%}.",
        f"<b>Fix?</b> Priors set to 1: gap +{fix_gap.min():.0%} to +{fix_gap.max():.0%}, but AUC "
        f"{fix_auc.min():.2f}–{fix_auc.max():.2f}, below the tool.",
    ],
    "The model doesn't matter; priors drives both the accuracy and the bias.",
)

tabs = st.tabs(
    [
        "1 · Data",
        "2 · Performance",
        "3 · What drives the models",
        "4 · Stability",
        "5 · Fairness",
        "6 · Trade-off & cost",
        "7 · Recommendation",
    ]
)

# ------------------------------------------------------------------------------------ data

with tabs[0]:
    grouped = race.where(race.isin(["African-American", "Caucasian"]), "Other groups")
    band = pd.cut(
        data.X.Number_of_Priors, [-1, 0, 1, 3, 6, 100], labels=["0", "1", "2-3", "4-6", "7+"]
    )
    share = pd.crosstab(band, grouped, normalize="columns")
    rate = data.y.groupby(grouped).mean()
    priors = data.X.Number_of_Priors
    lib.tldr(
        [
            f"Mean priors: {mean['African-American']:.1f} African-American vs "
            f"{mean['Caucasian']:.1f} Caucasian.",
            f"7+ priors: {(priors[race == 'African-American'] >= 7).mean():.0%} vs "
            f"{(priors[race == 'Caucasian'] >= 7).mean():.0%}.",
            f"Re-offence rate: {rate['African-American']:.0%} vs {rate['Caucasian']:.0%}.",
        ],
        "Priors, the strongest predictor, is racially skewed: any model using it inherits that.",
    )
    left, right = st.columns([2, 1])
    fig = go.Figure()
    for g in ["African-American", "Caucasian", "Other groups"]:
        fig.add_bar(
            x=share.index.astype(str),
            y=share[g],
            name=g,
            marker_color=lib.RACE_COLOUR[g],
            text=[f"{v:.0%}" for v in share[g]],
            textposition="outside",
        )
    fig.update_layout(
        barmode="group",
        yaxis_tickformat=".0%",
        xaxis_title="prior offences",
        yaxis_title="share of the group",
    )
    left.plotly_chart(
        lib.style(fig, "Priors, the strongest predictor, differ by race"), width="stretch"
    )
    fig = go.Figure(
        go.Bar(
            x=rate.index,
            y=rate,
            marker_color=[lib.RACE_COLOUR[g] for g in rate.index],
            text=[f"{v:.0%}" for v in rate],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 0.7], showlegend=False)
    right.plotly_chart(lib.style(fig, "Two-year re-offence rate"), width="stretch")
    lib.takeaway(
        f"{(data.X.Number_of_Priors[race == 'African-American'] >= 7).mean():.0%} of "
        "African-American defendants have 7+ priors, against "
        f"{(data.X.Number_of_Priors[race == 'Caucasian'] >= 7).mean():.0%} of Caucasian "
        "defendants. "
        "Any model that uses priors inherits that imbalance."
    )

# ----------------------------------------------------------------------------- performance

with tabs[1]:
    t = lib.threshold_control("ov_perf")
    rows = {m: lib.metrics(*lib.scores(m).T.values, t) for m in lib.ALL_MODELS}
    ours = pd.DataFrame({m: rows[m] for m in lib.MODELS}).T
    lib.tldr(
        [
            f"AUC {ours.AUC.min():.3f}–{ours.AUC.max():.3f} for our models vs "
            f"{rows['COMPAS tool']['AUC']:.3f} for the COMPAS tool.",
            f"At {t:.3f}: accuracy {ours.Accuracy.min():.1%}–{ours.Accuracy.max():.1%} "
            f"(tool {rows['COMPAS tool']['Accuracy']:.1%}), recall "
            f"{ours.Recall.min():.1%}–{ours.Recall.max():.1%} (tool "
            f"{rows['COMPAS tool']['Recall']:.1%}).",
            "XGBoost's scores sit in 0.27–0.73: below 0.27 it flags everyone.",
        ],
        "The three models tie, and all rank defendants better than the COMPAS tool.",
    )
    spokes = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
    lo, hi = st.slider("Radial range", 0.0, 1.0, (0.5, 0.8), 0.05, key="ov_range")
    fig = go.Figure()
    for m in lib.ALL_MODELS:
        vals = [rows[m][k] for k in spokes]
        fig.add_scatterpolar(
            r=vals + vals[:1],
            theta=spokes + spokes[:1],
            name=f"{m} (AUC {rows[m]['AUC']:.3f})",
            line={"color": lib.COLOUR[m], "dash": lib.DASH[m], "width": 2.5},
            marker={"symbol": lib.SYMBOL[m], "size": 8},
        )
    fig.update_polars(radialaxis={"range": [lo, hi], "dtick": 0.1, "tickformat": ".1f"})
    st.plotly_chart(
        lib.style(fig, f"Performance at threshold {t:.3f}, all features", 520), width="stretch"
    )
    lib.takeaway(
        "LogReg, XGBoost and TabPFN are within 0.003 AUC of each other and all rank defendants "
        "better than the COMPAS tool. The tool's higher recall at 0.5 comes from "
        "flagging more people."
    )
    lib.caveat(
        "XGBoost's scores are compressed (0.27–0.73), so its 0/1 metrics move sharply with the "
        "threshold: at 0.252 it flags every defendant. Its ranking (AUC) is as good as the others'."
    )

# ------------------------------------------------------------------------ interpretability

with tabs[2]:
    me = lib.comparison("marginal_effects.csv").query("run == 'holdout'")
    wide = me.pivot_table(index="feature", columns="model", values="marginal_effect")[lib.MODELS]

    def span(feature: str) -> str:
        v = wide.loc[feature]
        return f"{v.min():+.3f} to {v.max():+.3f}"

    lib.tldr(
        [
            f"Under 25: {span('Age_Below_TwentyFive')} risk. "
            f"Over 45: {span('Age_Above_FourtyFive')}.",
            f"Each extra prior: {span('Number_of_Priors')}.",
            f"Being African-American, directly: {span('African_American')}.",
        ],
        "Same story in all three models: age and priors drive risk; race enters through priors.",
    )
    n = st.slider("Features shown", 3, 10, 5, key="ov_me_n")
    top = wide.drop(index=list(lib.SMALL_GROUPS)).abs().mean(axis=1).nlargest(n).index[::-1]
    fig = go.Figure()
    for m in lib.MODELS:
        fig.add_bar(
            y=[lib.FEATURES[f] for f in top],
            x=wide.loc[top, m],
            name=m,
            orientation="h",
            marker_color=lib.COLOUR[m],
            text=[f"{v:+.3f}" for v in wide.loc[top, m]],
            textposition="outside",
        )
    fig.update_layout(
        barmode="group",
        xaxis_title="average change in predicted risk (0 → 1 for a yes/no feature, +1 prior)",
    )
    st.plotly_chart(
        lib.style(fig, "Marginal effects, all features (holdout)", 460), width="stretch"
    )
    pdp = lib.comparison("pdp_priors.csv")
    fig = go.Figure()
    for m in lib.MODELS:
        g = pdp[pdp.model == m]
        fig.add_scatter(
            x=g.Number_of_Priors,
            y=g.predicted_risk,
            name=m,
            mode="lines+markers",
            line={"color": lib.COLOUR[m]},
            marker={"symbol": lib.SYMBOL[m]},
        )
    fig.update_layout(
        xaxis_title="prior offences (set for everyone)",
        yaxis_title="average risk",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(lib.style(fig, "Partial dependence on priors"), width="stretch")
    lib.takeaway(
        "All three models tell the same story: youth and prior offences raise predicted risk; "
        "being African-American adds about +0.01–0.02 directly. Race enters through priors, not "
        "the race column."
    )

# ------------------------------------------------------------------------------- stability

with tabs[3]:
    st_tab = lib.comparison("stability.csv")
    t = lib.threshold_control("ov_stab")
    records = []
    for m in lib.MODELS:
        for pair, (a, b) in {
            "X1 vs X2": ("X1_to_X3", "X2_to_X3"),
            "Time window 1 vs 2": ("temporal_1", "temporal_2"),
        }.items():
            ma, mb = (
                lib.metrics(*lib.scores(m, a).T.values, t),
                lib.metrics(*lib.scores(m, b).T.values, t),
            )
            for k in ["AUC", "Accuracy", "Recall", "F1"]:
                records.append(
                    {"model": m, "pair": pair, "metric": k, "change": abs(mb[k] - ma[k])}
                )
    frame = pd.DataFrame(records)
    flips = {
        m: ((lib.scores(m, "X1_to_X3").score >= t) != (lib.scores(m, "X2_to_X3").score >= t)).mean()
        for m in lib.MODELS
    }
    auc_moves = frame[frame.metric == "AUC"].change
    worst = frame[frame.metric != "AUC"].sort_values("change").iloc[-1]
    lib.tldr(
        [
            f"AUC moves ≤ {auc_moves.max():.3f} between fits, for every model.",
            f"X1 vs X2 fits flip {min(flips.values()):.1%}–{max(flips.values()):.1%} of the same "
            f"decisions at {t:.3f}.",
            f"Largest swing: {worst.model} {worst.metric} {worst.change:.3f} ({worst.pair}), "
            "from scores crossing the threshold, not re-ranking.",
        ],
        "Rankings are stable; borderline decisions are not.",
    )
    cols = st.columns(2)
    for col, pair in zip(cols, ["X1 vs X2", "Time window 1 vs 2"]):
        fig = go.Figure()
        for m in lib.MODELS:
            g = frame[(frame.model == m) & (frame.pair == pair)]
            fig.add_bar(
                x=g.metric,
                y=g.change,
                name=m,
                marker_color=lib.COLOUR[m],
                text=[f"{v:.3f}" for v in g.change],
                textposition="outside",
            )
        fig.update_layout(barmode="group", yaxis_title="absolute change between the two fits")
        col.plotly_chart(lib.style(fig, pair), width="stretch")
    lib.takeaway(
        "AUC barely moves for any model (≤ 0.004). Decisions near the threshold do: the X1 and X2 "
        "fits disagree on 2–5% of the same defendants. XGBoost's swing over time is its "
        "compressed scores crossing 0.5, not a change in how it ranks people."
    )

# -------------------------------------------------------------------------------- fairness

with tabs[4]:
    t = lib.threshold_control("ov_fair")
    sets = ["race_aware", "race_blind", "race_priors_blind"]
    rows = []
    for m in lib.MODELS:
        for fs in sets:
            tab = lib.fairness(m, fs, t)
            tab = tab[tab.comparison == lib.PRIMARY].set_index("metric")
            rows.append(
                {
                    "label": f"{m} · {lib.FEATURE_SETS[fs]}",
                    "model": m,
                    "fs": fs,
                    "gap": tab.loc["statistical_parity", "gap"],
                    "lo": tab.loc["statistical_parity", "ci_low"],
                    "hi": tab.loc["statistical_parity", "ci_high"],
                    "or": tab.loc["conditional_statistical_parity", "mh_odds_ratio"],
                    "or_p": tab.loc["conditional_statistical_parity", "p_value"],
                }
            )
    tool = lib.fairness("COMPAS tool", "race_aware", t)
    tool = tool[tool.comparison == lib.PRIMARY].set_index("metric")
    rows.append(
        {
            "label": "COMPAS tool",
            "model": "COMPAS tool",
            "fs": "race_aware",
            "gap": tool.loc["statistical_parity", "gap"],
            "lo": tool.loc["statistical_parity", "ci_low"],
            "hi": tool.loc["statistical_parity", "ci_high"],
            "or": tool.loc["conditional_statistical_parity", "mh_odds_ratio"],
            "or_p": tool.loc["conditional_statistical_parity", "p_value"],
        }
    )
    fr = pd.DataFrame(rows)
    by_set = {fs: fr[(fr.fs == fs) & fr.model.isin(lib.MODELS)] for fs in sets}
    tool_row = fr[fr.model == "COMPAS tool"].iloc[0]
    n_sig = int((by_set["race_aware"].or_p < 0.05).sum())

    def gap_span(fs: str) -> str:
        g = by_set[fs].gap
        return f"{g.min():+.2f} to {g.max():+.2f}"

    lib.tldr(
        [
            f"Flag-rate gap (African-American − Caucasian): {gap_span('race_aware')}; "
            f"tool {tool_row.gap:+.2f}.",
            f"Drop race: {gap_span('race_blind')}. Drop race and priors: "
            f"{gap_span('race_priors_blind')}.",
            f"Same priors, age and charge: {n_sig} of 3 models significant; tool odds ratio "
            f"{tool_row['or']:.2f} (p = {tool_row.or_p:.3f}).",
        ],
        "Removing race does nothing; the gap runs through priors. Once priors, age and charge "
        "are held equal, only the tool's gap is significant.",
    )
    left, right = st.columns(2)
    fig = go.Figure()
    for m in lib.ALL_MODELS:
        g = fr[fr.model == m]
        fig.add_scatter(
            x=g.gap,
            y=g.label,
            mode="markers",
            name=m,
            marker={"color": lib.COLOUR[m], "symbol": lib.SYMBOL[m], "size": 11},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": g.hi - g.gap,
                "arrayminus": g.gap - g.lo,
            },
        )
    fig.add_vline(x=0, line_dash="dash", line_color=lib.INK_2)
    fig.update_layout(
        xaxis_title="extra flag rate, African-American − Caucasian",
        showlegend=False,
        yaxis={"autorange": "reversed"},
    )
    left.plotly_chart(lib.style(fig, "Flag-rate gap (95% CI)", 460), width="stretch")
    fig = go.Figure()
    for m in lib.ALL_MODELS:
        g = fr[fr.model == m]
        fig.add_scatter(
            x=g["or"],
            y=g.label,
            mode="markers",
            name=m,
            marker={
                "color": lib.COLOUR[m],
                "symbol": lib.SYMBOL[m],
                "size": 11,
                "line": {"color": lib.COLOUR[m], "width": 2},
            },
            marker_color=[lib.COLOUR[m] if p < 0.05 else "white" for p in g.or_p],
            hovertext=[f"p = {p:.3f}" for p in g.or_p],
        )
    fig.add_vline(x=1, line_dash="dash", line_color=lib.INK_2)
    fig.update_layout(
        xaxis_type="log",
        xaxis_title="odds ratio of being flagged (same priors, age and charge)",
        showlegend=False,
        yaxis={"autorange": "reversed"},
    )
    right.plotly_chart(
        lib.style(fig, "Among comparable defendants (hollow = not significant)", 460),
        width="stretch",
    )
    lib.takeaway(
        "Every model flags African-American defendants about 25 points more often, as the COMPAS "
        "tool does, and dropping the race column changes nothing. Among defendants with the same "
        "priors, age and charge, our models' gap is not significant at 0.5; the COMPAS tool's is."
    )

# --------------------------------------------------------------------- trade-off and cost

with tabs[5]:
    tldr_box = st.container()  # filled once the sliders below have set the costs
    st.markdown("**Cost assumptions** (per defendant; the break-even threshold follows from them)")
    a, b = st.columns(2)
    c_fn = a.slider(
        "Cost of releasing someone who re-offends ($)", 5_000, 100_000, int(lib.C_FN), 500
    )
    c_fp = b.slider(
        "Cost of detaining someone who would not re-offend ($)", 1_000, 60_000, int(lib.C_FP), 500
    )
    be = c_fp / (c_fp + c_fn)
    st.markdown(f"Break-even threshold: detain when predicted risk ≥ **{be:.3f}**.")
    ts = np.linspace(0.05, 0.95, 91)
    variant = st.radio(
        "Model variant",
        ["race_aware", "race_priors_blind", "priors_set_1"],
        format_func=lambda k: lib.FEATURE_SETS[k],
        horizontal=True,
    )
    fig = go.Figure()
    y = lib.scores("LogReg", "holdout", variant).y.to_numpy()
    detain_all, release_all = (1 - y.mean()) * c_fp, y.mean() * c_fn
    for m in lib.MODELS:
        s = lib.scores(m, "holdout", variant)
        yy, ss = s.y.to_numpy(), s.score.to_numpy()
        cost = [
            ((~(ss >= t) & (yy == 1)).sum() * c_fn + ((ss >= t) & (yy == 0)).sum() * c_fp) / len(yy)
            for t in ts
        ]
        fig.add_scatter(x=ts, y=cost, name=m, line={"color": lib.COLOUR[m]})
    tool = lib.scores("COMPAS tool")
    tool_cost = (
        ((tool.score == 0) & (tool.y == 1)).sum() * c_fn
        + ((tool.score == 1) & (tool.y == 0)).sum() * c_fp
    ) / len(tool)
    for value, name, dash in (
        (detain_all, "Detain everyone", "dash"),
        (release_all, "Release everyone", "dot"),
        (tool_cost, "COMPAS tool", "dashdot"),
    ):
        fig.add_hline(
            y=value,
            line_dash=dash,
            line_color=lib.INK_2,
            annotation_text=f"{name}: ${value:,.0f}",
            annotation_position="top left",
        )
    fig.add_vline(x=be, line_dash="dot", line_color=lib.MUTED)
    fig.update_layout(
        xaxis_title="decision threshold",
        yaxis_title="expected cost per defendant ($)",
        yaxis_range=[min(detain_all, tool_cost) * 0.8, max(tool_cost, detain_all) * 1.25],
    )
    st.plotly_chart(
        lib.style(fig, f"Cost per defendant, {lib.FEATURE_SETS[variant].lower()}", 480),
        width="stretch",
    )
    best_rows = []
    for m in lib.MODELS:
        for fs in ["race_aware", "race_priors_blind", "priors_set_1"]:
            s = lib.scores(m, "holdout", fs)
            flag = s.score >= be
            cost = (((~flag) & (s.y == 1)).sum() * c_fn + (flag & (s.y == 0)).sum() * c_fp) / len(s)
            best_rows.append(
                {
                    "Model": m,
                    "Variant": lib.FEATURE_SETS[fs],
                    "Cost per defendant": cost,
                    "Flagged": flag.mean(),
                    "vs detain everyone, per 1,000 defendants": (detain_all - cost) * 1000,
                }
            )
    costs = pd.DataFrame(best_rows)
    full = costs[costs.Variant == lib.FEATURE_SETS["race_aware"]].set_index("Model")
    stripped = costs[costs.Variant == lib.FEATURE_SETS["race_priors_blind"]]
    cheapest = full["Cost per defendant"].idxmin()
    with tldr_box:
        lib.tldr(
            [
                f"At break-even {be:.3f}: best model {cheapest} "
                f"{lib.fmt_money(full.loc[cheapest, 'Cost per defendant'])} per defendant; "
                f"detain everyone {lib.fmt_money(detain_all)}; COMPAS tool "
                f"{lib.fmt_money(tool_cost)}.",
                f"Without race and priors: flags {stripped.Flagged.min():.0%}–"
                f"{stripped.Flagged.max():.0%}, cost "
                f"{lib.fmt_money(stripped['Cost per defendant'].min())}+.",
                "Costs are assumptions: move the sliders.",
            ],
            "Priors is what makes a model worth more than detaining everyone.",
        )
    st.dataframe(
        costs.style.format(
            {
                "Cost per defendant": "${:,.0f}",
                "Flagged": "{:.0%}",
                "vs detain everyone, per 1,000 defendants": "${:,.0f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    lib.caveat(
        "Costs are assumptions, not measurements (defaults: $40,000 per missed re-offence, "
        "$13,500 per unneeded detention). Move the sliders: the ranking of models can change."
    )
    lib.takeaway(
        "With all features, the models save money against detaining everyone. Without race and "
        "priors they flag almost everyone and save nothing: fairness through input removal costs "
        "the whole value of the model."
    )

# -------------------------------------------------------------------------- recommendation

with tabs[6]:
    lib.tldr(
        [
            "Deploy none as is: more accurate than the tool, same ~25-point racial gap.",
            "Dropping race: no effect. Neutralising priors: fair but less accurate than the tool.",
            "Priors records policing as much as behaviour.",
        ],
        "Fix the data (priors), not the model.",
    )
    st.markdown(
        """
### Recommendation

1. **Do not deploy any of the three models as they stand.** They are more accurate than the
   COMPAS tool but flag African-American defendants about 25 points more often.
2. **Removing race is not a fix.** Race is carried by prior offences; the race-blind models
   behave exactly like the race-aware ones.
3. **Neutralising priors is fair but not useful.** The gap closes, but accuracy falls below the
   COMPAS tool and, at the break-even threshold, the model detains almost everyone.
4. **Fix the data before the model.** Prior offences reflect policing as much as behaviour; a
   more balanced record (e.g. convictions rather than arrests, or group-specific calibration
   audited over time) is the precondition for a model that is both accurate and fair.
"""
    )
