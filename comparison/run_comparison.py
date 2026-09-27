"""Every number behind the comparison slides, computed the same way for all three models.

    uv run python comparison/run_comparison.py            # reuses cached TabPFN refits
    uv run python comparison/run_comparison.py --force    # refits TabPFN too (~15 min)

Conventions (the group's decisions): full feature set unless stated; headline threshold 0.5,
cost break-even 0.252 only for the trade-off; fairness = African-American vs Caucasian with the
shared protocol (compas_scoring.fairness). Writes comparison/artifacts/*.csv.
"""

from __future__ import annotations

# isort: split
import compas_scoring  # noqa: F401  (pins thread pools before numpy and torch load)

# isort: split
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import models as m  # noqa: E402
from pfn_interpret import marginal_effects  # noqa: E402  (generic: any predict_proba model)

from compas_scoring.config import CONFIG  # noqa: E402
from compas_scoring.data import partition, temporal_windows, train_test  # noqa: E402
from compas_scoring.fairness import (  # noqa: E402
    fairness_table,
    fpdp,
    proxy_strata,
)

ART = HERE / "artifacts"
PRIORS = "Number_of_Priors"
BREAK_EVEN = CONFIG.costs.break_even
PRIMARY = CONFIG.fairness.comparisons[0]  # African-American vs Caucasian
N_EXPLAIN = 500  # the TabPFN analysis's explained sample, drawn the same way
FPDP_GRID = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]
GROUPS = ["race", "sex", "age_band", "charge_degree"]
# Fairness and mitigation variants on the holdout, beyond the configured feature sets:
#   priors_blind   re-estimation: drop priors, keep race, retrain (XGBoost group's fix a)
#   priors_set_1   substitution: the full model, with everyone's priors set to 1 (fix b)
FAIRNESS_SETS = ("race_aware", "race_blind", "race_priors_blind")


# ------------------------------------------------------------------------------ designs


def drop_priors(data):
    return replace(data, X=data.X.drop(columns=PRIORS), feature_set="priors_blind")


def designs() -> dict[tuple[str, str], tuple]:
    """(run, feature_set) -> (train, test) for every fit the comparison needs."""
    out = {}
    for fs in (*FAIRNESS_SETS,):
        out[("holdout", fs)] = train_test(fs)
    train, test = train_test("race_aware")
    out[("holdout", "priors_blind")] = (drop_priors(train), drop_priors(test))
    parts = partition("race_aware")
    out[("X1_to_X3", "race_aware")] = (parts["X1"], parts["X3"])
    out[("X2_to_X3", "race_aware")] = (parts["X2"], parts["X3"])
    for i, window in enumerate(temporal_windows("race_aware"), start=1):
        out[(f"temporal_{i}", "race_aware")] = (window.train, window.test)
    return out


def explain_sample(X: pd.DataFrame) -> pd.DataFrame:
    return X if len(X) <= N_EXPLAIN else X.sample(N_EXPLAIN, random_state=CONFIG.random_state)


# ------------------------------------------------------------------------- predictions


def predictions(force: bool) -> pd.DataFrame:
    """Test-set scores of every model on every design; TabPFN's read back where committed."""
    cache = ART / "tabpfn_refits.csv"
    tabpfn_refits = pd.read_csv(cache) if cache.exists() and not force else None
    frames, refits, fitted = [], [], {}
    for (run, fs), (train, test) in designs().items():
        for model in m.MODELS:
            if model == "TabPFN" and fs != "priors_blind":
                score = m.tabpfn_scores(run, fs).loc[test.X.index].to_numpy()
            elif model == "TabPFN" and tabpfn_refits is not None:
                score = tabpfn_refits.query("run == @run and feature_set == @fs")["score"]
                score = score.to_numpy()
            else:
                # The XGBoost group's saved holdout models, so the deck quotes their numbers.
                saved = m.saved_xgboost(fs) if model == "XGBoost" and run == "holdout" else None
                est = saved if saved is not None else m.fit(model, train.X, train.y, fs)
                score = m.score(est, test.X)
                if model == "TabPFN":
                    refits.append(pd.DataFrame({"run": run, "feature_set": fs, "score": score}))
                if run == "holdout" and fs == "race_aware":
                    fitted[model] = est
            frames.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "run": run,
                        "feature_set": fs,
                        "row": test.X.index,
                        "y": test.y.to_numpy(),
                        "score": score,
                        "compas_tool": test.incumbent.to_numpy(),
                    }
                )  # fmt: skip
            )
    if refits:
        pd.concat(refits, ignore_index=True).to_csv(cache, index=False)
    return pd.concat(frames, ignore_index=True), fitted


def holdout_model(model: str, fitted: dict):
    """The race-aware holdout model (refitted once for TabPFN, only for the steps that need it)."""
    if model not in fitted:
        train, _ = train_test("race_aware")
        fitted[model] = m.fit(model, train.X, train.y)
    return fitted[model]


def substitution(fitted: dict, force: bool) -> pd.DataFrame:
    """Fix (b): the full model scores everyone with priors = 1."""
    cache = ART / "substitution.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    _, test = train_test("race_aware")
    X = test.X.assign(**{PRIORS: 1.0})
    frames = [
        pd.DataFrame(
            {
                "model": model,
                "run": "holdout",
                "feature_set": "priors_set_1",
                "row": test.X.index,
                "y": test.y.to_numpy(),
                "score": m.score(holdout_model(model, fitted), X),
                "compas_tool": test.incumbent.to_numpy(),
            }
        )  # fmt: skip
        for model in m.MODELS
    ]
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(cache, index=False)
    return table


# ----------------------------------------------------------------------------- metrics


def metrics(y, score, threshold: float) -> dict:
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    flag = score >= threshold
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn)
    return {
        "threshold": threshold,
        "auc": roc_auc_score(y, score),
        "accuracy": (tp + tn) / len(y),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if tp else 0.0,
        "fpr": fp / (fp + tn),
        "selection_rate": float(flag.mean()),
        "cost_per_defendant": (fn * CONFIG.costs.c_fn + fp * CONFIG.costs.c_fp) / len(y),
    }


def auc_ci(y, score, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(CONFIG.random_state)
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    draws = []
    while len(draws) < n_boot:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            draws.append(roc_auc_score(y[i], score[i]))
    return tuple(np.percentile(draws, [2.5, 97.5]))


def performance(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, run, fs), g in preds.groupby(["model", "run", "feature_set"], sort=False):
        lo, hi = auc_ci(g.y, g.score)
        for t in (0.5, BREAK_EVEN):
            rows.append({"model": model, "run": run, "feature_set": fs,
                         **metrics(g.y, g.score, t), "auc_low": lo, "auc_high": hi})  # fmt: skip
        if model == m.MODELS[0]:  # the COMPAS tool's decision, once per test set
            lo, hi = auc_ci(g.y, g.compas_tool)
            rows.append({"model": "COMPAS tool", "run": run, "feature_set": fs,
                         **metrics(g.y, g.compas_tool, 0.5), "auc_low": lo,
                         "auc_high": hi})  # fmt: skip
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- fairness


def fairness(preds: pd.DataFrame) -> pd.DataFrame:
    """The shared protocol on every holdout variant, for the three models and the COMPAS tool."""
    frames = []
    hold = preds[preds.run == "holdout"]
    for (model, fs), g in hold.groupby(["model", "feature_set"], sort=False):
        g = g.set_index("row")
        groups = _groups(g.index)
        strata = proxy_strata(g.index)
        table = fairness_table(g.y, g.score, groups, strata).assign(model=model, feature_set=fs)
        frames.append(table)
        if model == m.MODELS[0] and fs == "race_aware":
            tool = fairness_table(g.y, g.compas_tool, groups, strata)
            frames.append(tool.assign(model="COMPAS tool", feature_set="race_aware"))
    table = pd.concat(frames, ignore_index=True)
    return table[(table.threshold_name == "0.5") | (table.model == "COMPAS tool")]


def _groups(index) -> pd.DataFrame:
    from compas_scoring.data import group_frame, load_raw

    return group_frame(load_raw().loc[index])


def fpdp_priors(fitted: dict, force: bool) -> pd.DataFrame:
    """Step 2: statistical parity when everyone's priors is fixed to each value in turn."""
    cache = ART / "fpdp_priors.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    _, test = train_test("race_aware")
    groups, strata = _groups(test.X.index), proxy_strata(test.X.index)
    frames = []
    for model in m.MODELS:
        est = holdout_model(model, fitted)
        curve = fpdp(lambda X, e=est: m.score(e, X), test.X, groups, strata, PRIORS, 0.5,
                     PRIMARY, FPDP_GRID)  # fmt: skip
        frames.append(curve.assign(model=model))
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(cache, index=False)
    return table


# ------------------------------------------------------------------- interpretability


def interpretability(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Marginal effects on every race-aware run, and the PDP on priors (holdout)."""
    tab = ROOT_TABPFN_ANALYSIS
    me_frames = [pd.read_csv(tab / "t10_marginal_effects" / "marginal_effects.csv")
                 .query("feature_set == 'race_aware'").assign(model="TabPFN")]  # fmt: skip
    pdp_tab = pd.read_csv(tab / "t12_pdp_ice" / "pdp.csv").query(
        "run == 'holdout' and feature_set == 'race_aware'"
    )
    grid = pdp_tab[PRIORS].to_numpy()
    pdp_frames = [pd.DataFrame({"model": "TabPFN", PRIORS: grid,
                                "predicted_risk": pdp_tab.predicted_risk})]  # fmt: skip
    for (run, fs), (train, test) in designs().items():
        if fs != "race_aware":
            continue
        X = explain_sample(test.X)
        for model in ("LogReg", "XGBoost"):
            est = m.fit(model, train.X, train.y)
            me_frames.append(marginal_effects(est, X).assign(run=run, feature_set=fs, model=model))
            if run == "holdout":
                risk = [m.score(est, X.assign(**{PRIORS: v})).mean() for v in grid]
                pdp_frames.append(pd.DataFrame({"model": model, PRIORS: grid,
                                                "predicted_risk": risk}))  # fmt: skip
    return pd.concat(me_frames, ignore_index=True), pd.concat(pdp_frames, ignore_index=True)


ROOT_TABPFN_ANALYSIS = m.ROOT / "tabpfn" / "artifacts" / "analysis"


# ---------------------------------------------------------------------------- stability


PAIRS = {"X1 vs X2": ("X1_to_X3", "X2_to_X3"), "time 1 vs 2": ("temporal_1", "temporal_2")}


def stability(preds: pd.DataFrame, perf: pd.DataFrame, me: pd.DataFrame) -> pd.DataFrame:
    """Distance between the two fits of each pair: metrics, decisions, marginal effects."""
    rows = []
    at = perf[(perf.threshold == 0.5) & (perf.feature_set == "race_aware")]
    for model in m.MODELS:
        for pair, (a, b) in PAIRS.items():
            ma = at.query("model == @model and run == @a").iloc[0]
            mb = at.query("model == @model and run == @b").iloc[0]
            row = {"model": model, "pair": pair}
            for k in ("auc", "accuracy", "precision", "recall", "f1", "fpr"):
                row[f"delta_{k}"] = abs(mb[k] - ma[k])
            ea = me.query("model == @model and run == @a").set_index("feature").marginal_effect
            eb = me.query("model == @model and run == @b").set_index("feature").marginal_effect
            eb = eb.loc[ea.index]
            row["me_l2"] = float(np.linalg.norm(ea - eb))
            row["me_spearman"] = float(spearmanr(ea.abs(), eb.abs()).statistic)
            if pair == "X1 vs X2":  # same test defendants: compare them one by one
                sa = preds.query("model == @model and run == @a").set_index("row").score
                sb = preds.query("model == @model and run == @b").set_index("row").score
                sb = sb.loc[sa.index]
                row["score_rmse"] = float(np.sqrt(((sa - sb) ** 2).mean()))
                row["flip_rate"] = float(((sa >= 0.5) != (sb >= 0.5)).mean())
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- XPER


XPER_FILES = {
    "LogReg": m.ROOT / "logreg" / "artifacts" / "analysis" / "xper_auc_individual.csv",
    "XGBoost": m.ROOT / "xgboost" / "models" / "xper_race_aware_individual.csv",
    "TabPFN": ROOT_TABPFN_ANALYSIS / "t20_xper_individual" / "xper_auc_individual.csv",
}


def xper_person(preds: pd.DataFrame) -> pd.DataFrame:
    """One borderline African-American defendant, the same for all three models.

    From the 400 defendants all three XPER runs share: African-American, 25-45, 2-4 priors,
    and the smallest largest-distance of the three models' scores from the 0.5 threshold.
    """
    per_model = {}
    for model, path in XPER_FILES.items():
        frame = pd.read_csv(path, index_col=0)
        per_model[model] = frame
    shared = sorted(set.intersection(*(set(f.index) for f in per_model.values())))
    hold = preds.query("run == 'holdout' and feature_set == 'race_aware'")
    scores = hold.pivot(index="row", columns="model", values="score").loc[shared]
    X, groups = train_test("race_aware")[1].X.loc[shared], _groups(shared)
    eligible = (groups.race == "African-American") & (groups.age_band == "25 - 45")
    eligible &= X[PRIORS].between(2, 4)
    distance = (scores[list(m.MODELS)] - 0.5).abs().max(axis=1)[eligible]
    row = int(distance.idxmin())
    out = pd.concat({model: f.loc[row] for model, f in per_model.items()}, axis=1)
    out.index.name = "feature"
    out = out.reset_index()
    out.attrs["row"] = row
    out.loc[len(out)] = {"feature": "score", **{k: float(scores.loc[row, k]) for k in m.MODELS}}
    out.loc[len(out)] = {"feature": "row", **{model: row for model in m.MODELS}}
    return out


# ------------------------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="refit TabPFN variants too")
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)

    print("predictions ...", flush=True)
    preds, fitted = predictions(args.force)
    preds = pd.concat([preds, substitution(fitted, args.force)], ignore_index=True)
    preds.to_csv(ART / "predictions.csv", index=False)

    print("performance, fairness ...", flush=True)
    perf = performance(preds)
    perf.to_csv(ART / "performance.csv", index=False)
    fairness(preds).to_csv(ART / "fairness.csv", index=False)
    print("FPDP on priors ...", flush=True)
    fpdp_priors(fitted, args.force)

    print("interpretability, stability ...", flush=True)
    me, pdp = interpretability(preds)
    me.to_csv(ART / "marginal_effects.csv", index=False)
    pdp.to_csv(ART / "pdp_priors.csv", index=False)
    stability(preds, perf, me).to_csv(ART / "stability.csv", index=False)

    person = xper_person(preds)
    person.to_csv(ART / "xper_person.csv", index=False)
    print(f"XPER defendant: row {person.attrs['row']}")
    print(f"-> {ART}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
