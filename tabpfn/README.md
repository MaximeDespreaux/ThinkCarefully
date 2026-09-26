# TabPFN — the foundation model

TabPFN v2 (Hollmann et al., *Nature* 2025, `tabpfn==2.2.1`) on the COMPAS two-year cohort,
used with its defaults (no tuning). This folder fits the model and measures **predictive
performance**. It also saves the per-defendant predictions that the **interpretability**,
**stability** and **fairness** analyses start from. Those analyses are still to do: see
[For the analysis](#for-the-analysis) below.

```bash
make tabpfn-smoke   # once: checks TabPFN installs, downloads its weights, predicts
make tabpfn         # fits every design -> tabpfn/artifacts/ (committed; cached, --force refits)
make tabpfn-plots   # performance figures -> reports/figures/tabpfn/ (seconds, no refit)
make tabpfn-test    # unit tests for the metrics and the adapter
uv run python tabpfn/run_tabpfn.py --help     # pick designs / feature sets
```

| File | What it does |
|---|---|
| `pfn_model.py` | `TabPFNModel`, an sklearn-compatible adapter (works with PDP, permutation importance, scorers) |
| `pfn_metrics.py` | the performance panel: AUC + bootstrap CI, accuracy, F1, recall, Type I/II errors, cost |
| `run_tabpfn.py` | fits every design and feature set, and writes predictions and performance |
| `plot_tabpfn.py` | performance figures from the committed artifacts (git-ignored PNGs): per feature set, across feature sets (ablation), and a radar of every variant |
| `smoke_tabpfn.py` | the install/weights check |

## What "COMPAS tool" means

COMPAS is also the name of Northpointe's commercial risk-assessment tool, the one the
dataset was collected to audit. The dataset's `score_factor` column is **that tool's own
prediction** (1 = rated medium or high risk). It is never a feature. It is the benchmark
("incumbent" in `pyproject.toml`): the model already in use, which TabPFN has to beat on the
same test defendants. It is labelled "COMPAS tool" everywhere here.

## Designs

All splits live in `compas_scoring.data`, so the logreg and xgboost groups use the same cuts.

| Run | Train → test | Cohort | Question |
|---|---|---|---|
| `holdout` | stratified 70% → 30% | modelling table | headline 1 |
| `X1+X2_to_X3` | X1 ∪ X2 (80%) → X3 (20%) | modelling table | headline 2 |
| `X1_to_X3`, `X2_to_X3` | 40% → the same 20% | modelling table | stability 1: does the training sample change the model? |
| `temporal_1` | first 40% → next 20% by screening date | dated cohort | stability 2: does the story hold on newer defendants? |
| `temporal_2` | first 70% → last 30% by screening date | dated cohort | stability 2 |

Each run is fitted on every feature set in `pyproject.toml`. This is the
**protected-attribute ablation**: the same model with each protected attribute removed.

| Feature set | Features | Removes |
|---|---|---|
| `race_aware` (FS1) | all 10 | nothing |
| `race_blind` (FS3) | 5 | the five race dummies |
| `sex_blind` | 9 | `Female` |
| `age_blind` | 8 | both age dummies |
| `protected_blind` | 2 | race, sex and age: only priors and charge degree remain |

Comparing each blind set with `race_aware` shows what that attribute adds to performance.
Whether removing it reduces disparity is a fairness question, answered from the same
prediction files, which keep every group label whatever the feature set.

**Split balance.** X1/X2/X3 are stratified on outcome × race × sex × age band. Each
part matches the cohort's race, sex and age mix within 0.12 percentage points. With the
outcome alone, X3 was 3.7 points off on age. `make split-balance` reports the gap for
every split:
- The 70/30 split is within noise (p ≥ 0.18).
- The time windows show real drift in who was screened: the sex mix, and later the age
  mix (p ≤ 0.02). A change between time windows can therefore come from the population,
  not only from the model.

**Dated cohort caveat.** The modelling table has no dates, so the time-ordered runs use
`data.load_dated()`. It rebuilds the cohort from `cox-violent-parsed.csv`: 5,686 people
screened from Jan 2013 to Mar 2014, with a **33.4%** base rate, against 45.5% in the
modelling table. The docstring explains how it is built. Compare temporal runs with each
other, never with the headline numbers.

## Outputs (`tabpfn/artifacts/`, committed)

These are committed, so the analysis can start without refitting. After any change to the
model or the splits, rerun `uv run python tabpfn/run_tabpfn.py --force` (~25 min on 8 cores)
and commit the new files.

- `predictions/<run>__<feature_set>.csv` has one row per test defendant: `y`, `tabpfn`
  (score), `compas_tool` (the COMPAS tool's own `score_factor`: 1 = medium/high risk), `race`, `sex`, `age_band`, `charge_degree`.
  The index `row` is the row in the cohort, so `X1_to_X3` and `X2_to_X3` line up
  defendant by defendant.
- `performance.csv` has one row per run × feature set × model (TabPFN, and the COMPAS
  tool's score as the benchmark on the same rows) × operating point.
- `runs.csv` holds sizes, base rates, date ranges, and fit/predict seconds.

## Performance

At each operating point, the per-defendant decision is read as a hypothesis test:

| | |
|---|---|
| H0 | the defendant will **not** re-offend within two years (flagging = rejecting H0) |
| Type I error α = FPR | flag someone who would **not** have re-offended |
| Type II error β = FNR | release someone who **then re-offends** |
| Power 1 − β = recall | share of re-offenders the model catches |

Two operating points. `0.5` is the accuracy default. `break_even` = c_fp / (c_fp + c_fn) =
**0.252** under the cost assumptions in `pyproject.toml` ($13.5k per unneeded detention,
$40k per missed re-offence). Both are fixed in advance, so neither is tuned on the test set.

**Which error is worse? This is for the report to argue, and the answer depends on the lens:**

- *Economic lens (the configured costs):* a false negative costs about 3× a false positive.
  That pushes the threshold down to 0.252, so the model flags more people: β falls and α
  rises.
- *Rights lens:* a false positive means detaining someone who would not have re-offended.
  That is a liberty cost the money figures leave out. Blackstone's ratio says it is better
  that ten guilty persons escape than that one innocent suffer, which argues for the
  opposite trade-off. ProPublica's criticism of COMPAS was a Type I error criticism:
  unequal FPRs by race.
- A useful way to say it: the costs are assumptions, the fixed thresholds show where each
  lens puts the model, and the fairness analysis shows who pays for the Type I errors.

**Not done here:**
- TODO(analysis): **XPER** (on AUC, and possibly on cost). An exact decomposition over
  all 2¹⁰ coalitions took ~1–2 h per metric on a 100-row sample in the old repo, because
  every coalition is a TabPFN call. Reuse `compas-data/src/compas_scoring/xper.py`.
  Batch coalitions into single `predict_proba` calls to pay the per-call cost fewer times.

## For the analysis

**Radar chart (`10_radar_holdout.png`).** It has one line per TabPFN variant, with a spoke
per metric. For now the spokes are the performance metrics. To add a dimension, write
`tabpfn/artifacts/radar_extra.csv` with one row per `feature_set`. Give it one column per
new score, already scaled to 0–1 with higher = better (e.g. `1 - |FPR gap|`, or
`1 - decision flip rate X1 vs X2`). Each column becomes a spoke; `plot_tabpfn.py` needs no
change.

The runner saves predictions. Anything that needs the model itself (SHAP, LIME, PDP, ICE,
permutation importance) refits it: `build_model().fit(train.X, train.y)`, with the train
set from `compas_scoring.data`. A fit takes ~1–3 min (it builds the prediction cache), and
after that each `predict_proba` call is cheap. So **fit once per design and reuse the
model**, explain a fixed, small sample of test rows, and save the results (see Runtime
below).

### Interpretability

TODO(analysis):

- [ ] **Marginal effects.** TabPFN has no coefficients. Use the average change in predicted
      probability when a dummy flips 0→1 (or priors +1), holding the other features fixed.
- [ ] **Impurity (Gini, Shannon entropy, misclassification error).** These are tree
      criteria, and TabPFN has no splits. Fit a shallow **global surrogate** tree on
      TabPFN's own predictions and report its impurity decreases *with its fidelity (R²)*.
      A surrogate explains the model, not the data.
- [ ] **PDP / ICE** with `sklearn.inspection.partial_dependence` (the adapter is a proper
      classifier). Priors is the only non-binary feature, so its curves are the interesting
      ones.
- [ ] **SHAP.** Use KernelSHAP, since there is no TreeSHAP for TabPFN, on a small
      background and explain set. Check efficiency: the SHAP values should add up to
      prediction minus base value.
- [ ] **LIME.** Run it several times with different seeds on the same defendant. The old
      repo found LIME was **not reproducible** on TabPFN, naming 4 different top features
      across 15 runs.
- [ ] **Permutation importance.** Compute it on AUC and on cost, with ≥10 repeats so there
      are intervals.

### Stability

Design 1 (`X1_to_X3` vs `X2_to_X3`, same X3) and design 2 (`temporal_1` vs `temporal_2`).

TODO(analysis):

- [ ] **Distance between estimated parameters: not defined for TabPFN.** Its weights are
      pre-trained and frozen, and "fitting" only stores the rows. Measure the distance on
      what the model *outputs* instead:
  - [ ] predictions: L2 norm (and mean |Δ|) between the two score vectors on X3, plus the
        share of defendants whose decision flips at each threshold;
  - [ ] performance: Δ of every `performance.csv` metric between the two runs;
  - [ ] interpretability: distance between the two runs' importance vectors (SHAP /
        permutation), and rank agreement (Spearman) of the features;
  - [ ] fairness: Δ of each fairness metric below.
- [ ] Design 2: X3 is not shared, so compare the *story*, not the rows. Check whether the
      importance ranking and the fairness gaps stay the same after the model sees newer
      defendants.

### Fairness

`predictions/*.csv` already carries `race`, `sex`, `age_band` and `charge_degree` next to
`y`, the TabPFN score and the COMPAS tool's score. Use either operating point to turn scores into
decisions.

TODO(analysis):

- [ ] **Statistical parity:** P(Ŷ=1 | D=1) = P(Ŷ=1 | D=0). Use a χ² test of Ŷ ⟂ D
      (D = African-American vs Caucasian, and Female vs Male).
- [ ] **Conditional statistical parity:** Ŷ ⟂ D | X_c. Choose X_c, e.g. priors band ×
      charge degree, and run a χ² test within strata or a Cochran–Mantel–Haenszel test.
- [ ] **Equalized odds:** equal TPR and FPR across groups. The FPR gap is ProPublica's
      finding.
- [ ] **Fairness equivalence (TOST, Schuirmann 1987):** θ = |p₁ − p₀|. H0: θ ≥ δ (unfair)
      vs H1: −δ < θ < δ. Report the tightest δ at which fairness can be certified.
- [ ] **FPDP** (fairness partial dependence) to find *candidate variables*. Recompute the
      test statistic with feature X_A set to each value, and flag X_A as a candidate if
      some value brings χ² below the critical value. This is the 3-step approach:
      test → identify → mitigate.
- [ ] Compare `race_aware` with `race_blind`. Dropping race does not remove disparity if
      priors acts as a proxy for it.

## Runtime and threads

`import compas_scoring` limits every numeric library to one thread. That avoids a known
freeze when XGBoost and PyTorch run in the same process. On one thread TabPFN is very slow:
predicting 1,852 rows took ~530 s. So `TabPFNModel` raises PyTorch's thread count back to
all cores for its own calls. **Never load XGBoost in the same process as a TabPFN run.**

It also defaults to `fit_mode="fit_with_cache"`. Fitting encodes the training context once
(~150 s on 8 cores), and after that each `predict_proba` call is cheap (~11 s per 300 rows,
against ~170 s without the cache). Predictions are identical either way. For the
explanation methods, which call `predict_proba` hundreds of times, **fit once and reuse the
fitted model**. Never refit inside a loop. The full benchmark is in the `pfn_model.py`
docstring.
