# TabPFN — the foundation model

TabPFN v2 (Hollmann et al., *Nature* 2025, `tabpfn==2.2.1`) on the COMPAS two-year cohort,
used with its defaults (no tuning). This folder fits the model and measures **predictive
performance**. It also saves the per-defendant predictions that the **interpretability**,
**stability** and **fairness** analyses start from. Those analyses are numbered tests in
`analysis/`: see [The analysis, test by test](#the-analysis-test-by-test) below.

**Results so far, with the numbers to quote: [FINDINGS.md](FINDINGS.md).**

```bash
make tabpfn-smoke   # once: checks TabPFN installs, downloads its weights, predicts
make tabpfn         # fits every design -> tabpfn/artifacts/ (committed; cached, --force refits)
make tabpfn-plots   # performance figures -> reports/figures/tabpfn/ (seconds, no refit)
make tabpfn-test    # its code tests, in tests/test_pfn_*.py (run with the rest by make test)
uv run python tabpfn/run_tabpfn.py --help     # pick designs / feature sets
```

| File | What it does |
|---|---|
| `pfn_model.py` | `TabPFNModel`, an sklearn-compatible adapter (works with PDP, permutation importance, scorers) |
| `pfn_metrics.py` | the performance panel: AUC + bootstrap CI, accuracy, F1, recall, Type I/II errors, cost |
| `run_tabpfn.py` | fits every design and feature set, and writes predictions and performance |
| `plot_tabpfn.py` | performance figures from the committed artifacts (git-ignored PNGs): per feature set, across feature sets (ablation), and a radar of every variant |
| `smoke_tabpfn.py` | the install/weights check |
| `pfn_fairness.py`, `pfn_interpret.py`, `pfn_importance.py`, `pfn_xper.py`, `pfn_stability.py`, `pfn_significance.py` | the analysis library, ported from the earlier study (`compas-data/studies/tabpfn`) and adapted to this repo; each has a `test_pfn_*.py` |
| `analysis/tNN_*.py` | the analysis, one numbered test per script (see [The analysis, test by test](#the-analysis-test-by-test)) |
| `artifacts/legacy/` | threshold-free results imported from the earlier study, with their provenance |

```bash
make tabpfn-analysis        # every numbered test, T01 -> T18 (~3-4 h from scratch, cached after)
make tabpfn-t04             # one test; ARGS=--force recomputes it
```

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
| `race_priors_blind` | 4 | race and priors, the dominant race proxy |
| `race_proxy_blind` | 2 | race and every race proxy: under 25 and sex remain |

Comparing each blind set with `race_aware` shows what that attribute adds to performance.
Whether removing it reduces disparity is a fairness question, answered from the same
prediction files, which keep every group label whatever the feature set.

**Race proxies.** Dropping the race columns does not remove race. The EDA shows the
pairwise links (association matrix; figure 09 for priors). `make proxy-check` adds the
joint effect: how well race can be predicted from each feature set.
It uses a cross-validated AUC, African-American vs Caucasian, where 0.5 means no race
information:

| Feature set | Race leakage (AUC) |
|---|---|
| all features / no race | **0.671** / **0.671**: the race-blind set carries as much race as the full one |
| no race or priors | 0.615 |
| no race or proxies | 0.569: much less, not none |

A feature counts as a proxy if its Cramér's V with any race group is ≥ 0.1 in the EDA
association matrix. That flags priors (0.225), over 45 (0.157) and misdemeanour (0.103).
Under 25 (0.09) sits just below the cutoff and stays, which is why some leakage remains.

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

**XPER.** On AUC the exact decomposition does not depend on the threshold: it is imported
from the earlier study (`artifacts/legacy/xper_auc.csv`). On cost it is redone at 0.252 (T16),
with coalitions batched into few `predict_proba` calls (`pfn_xper.py`).

## The analysis, test by test

Every test is a script `analysis/tNN_<name>.py`, run with `make tabpfn-tNN`. It writes to
`artifacts/analysis/tNN_<name>/` (committed) and reads only the predictions,
`performance.csv`, or the outputs of lower-numbered tests. Results and numbers to quote are in
[FINDINGS.md](FINDINGS.md). Tests marked *refit* fit TabPFN again (~20 s per fit on 12 cores);
the others run in seconds from the saved predictions.

Every threshold-dependent test is run at both operating points (0.5 and 0.252).

### Fairness: the shared protocol

All three models are tested the same way, by `compas_scoring.fairness`, with the settings in
`[tool.compas_scoring.fairness]`. For TabPFN, `pfn_fairness.protocol_table(run, feature_set)`
runs it on the saved predictions, for TabPFN and for the COMPAS tool's score as a benchmark.

It covers both thresholds (0.5 and 0.252). There are four comparisons: African-American vs
Caucasian (primary), Hispanic and Other vs Caucasian, and Female vs Male. Asian and Native
American are excluded as too small. Gaps read protected minus reference. p-values are
Holm-corrected across the four comparisons within each metric and threshold. TOST uses a
fixed tolerance δ = 0.05, and every row also reports the tightest δ at which fairness could be
certified.

| # | Test | Output |
|---|---|---|
| T01 | The **whole protocol** on every run × feature set (`protocol.csv`, read by T02–T05, T09, T18), and **statistical parity**: two-proportion z-test on the flag-rate gap (= Pearson χ²), Hurlin LR test alongside | `protocol.csv`, `statistical_parity.csv` |
| T02 | **Conditional statistical parity**: Hurlin LR test over the race proxies held fixed (priors band × age band × charge degree), plus CMH, Mantel–Haenszel odds ratio and Breslow–Day | `conditional_parity.csv` |
| T03 | **Equalized odds**: z-tests on the FPR and FNR gaps, and the Hurlin test of Ŷ ⟂ D given Y | `equalized_odds.csv`, `error_rate_gaps.csv` |
| T04 | **Fairness equivalence (TOST)** on the flag-rate, FPR and FNR gaps, and the tightest certifiable δ | `equivalence.csv` |
| T05 | **race_aware vs race_blind** and the other ablations: performance against the race and sex gaps, flagging the sets that close a gap by flagging nearly everyone | `ablation.csv` |
| T06 | **FPDP** *(refit)*: candidate variables (`fpdp_candidates`), for race and sex, both thresholds, with and without the proxies held fixed; counted only when fairness is rejected on the real data | `fpdp_curves.csv`, `fpdp_candidates.csv` |

Conditioning on the proxies isolates race. A gap that survives among defendants with the same
priors, age band and charge degree is not explained by those proxies.

### Stability

TabPFN has no estimated parameters (its weights are frozen), so the distance between two fits
is measured on what they output. Design 1 = `X1_to_X3` vs `X2_to_X3` (same X3), design 2 =
`temporal_1` vs `temporal_2` (different test sets: compare the story, not the rows).

| # | Test | Output |
|---|---|---|
| T07 | **Predictions**, design 1: L2 norm, mean \|Δ\|, share of decisions that flip | `pairs_predictions.csv` |
| T08 | **Performance**, both designs: Δ of every `performance.csv` metric | `pairs_performance.csv` |
| T09 | **Fairness**, both designs: Δ of every protocol gap, certifiable δ and Holm p-value; PSI of the scores in design 2 | `pairs_fairness.csv`, `psi.csv` |
| T17 | **Interpretability**, both designs: L2 distance and Spearman between importance vectors (permutation, SHAP, marginal effects) | `importance_distance.csv` |

### Interpretability

T10 and T12–T14 explain every design with all features, plus the race-blind holdout model.

| # | Test | Output |
|---|---|---|
| T10 | **Marginal effects** *(refit)*: average Δ risk when a dummy flips 0 → 1 (against the reference category) or priors +1 | `marginal_effects.csv` |
| T11 | **Impurity (Gini, entropy, misclassification)** through depth-4 surrogate trees on TabPFN's own scores and decisions, with their fidelity | `surrogate.csv`, `impurity.csv`, `rules/` |
| T12 | **PDP / ICE** on priors *(refit)* | `pdp.csv`, `ice.csv`, `ice_summary.csv` |
| T13 | **Permutation importance** on AUC and on cost, 10 repeats, 95% intervals *(refit)* | `permutation.csv` |
| T14 | **KernelSHAP** and the efficiency check *(refit)* | `shap_importance.csv`, `shap_values.csv`, `efficiency.csv` |
| T15 | **LIME reproducibility**: 15 runs on one defendant *(refit)* | `lime.csv`, `lime_summary.json` |
| T16 | **XPER on cost** at 0.252, exact over 1,024 coalitions *(refit, ~1 h)* | `xper_cost.csv` |

### Radar

T18 writes `artifacts/radar_extra.csv`: four more spokes (race FPR parity, sex FPR parity,
certifiable fairness, decision stability X1 vs X2), each 0–1 with higher = better.
`plot_tabpfn.py` adds them to both radar charts; `make tabpfn-plots` redraws them.

### Imported from the earlier study

`artifacts/legacy/` holds the earlier single-model study's results that do not depend on the
threshold (SHAP on 100 rows, XPER on AUC, 100 bootstrap refits, row-order and seed
sensitivity, learning curve), with a README giving their provenance and how that model
differs from this one.

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
