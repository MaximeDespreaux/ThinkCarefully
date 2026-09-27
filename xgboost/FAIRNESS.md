# XGBoost: fairness

Is either saved XGBoost model (`race_aware`, `race_blind`) unfair to African-American defendants, relative to every other defendant combined? This file explains what was computed, why, how to rerun it, and what came out. Only race (African-American vs everyone else) is compared here — no other protected attribute.

Code: [`fairness.py`](fairness.py) · Notebook: section 4 of [`base-model.ipynb`](base-model.ipynb) · Tests: [`../tests/test_fairness.py`](../tests/test_fairness.py)

## Method

Built around Hurlin, Pérignon & Saurin, *The Fairness of Credit Scoring Models* ([arXiv 2205.10200](https://arxiv.org/abs/2205.10200)) — the same group wrote XPER — plus standard stratified-table methods for the parts the paper doesn't work out in full (CMH, Mantel-Haenszel). Three steps, matching the three questions a fairness audit actually needs answered:

1. **Is the model unfair?** Three fairness definitions, three dedicated tests.
2. **If so, what's driving it?** FPDP: which feature, when fixed, would remove the evidence of unfairness.
3. **Can it be fixed?** Two ways to act on whatever FPDP finds, each re-checked against every test from step 1.

**Setup**
- **Data:** the shared stratified test set (1,852 defendants: 956 African-American, 896 everyone else), not the XPER subsample.
- **Prediction:** ŷ = 1 means the model flags the defendant as likely to re-offend, at threshold 0.5, same as the performance section.
- **Models:** both saved models (`models/xgb_race_aware.json`, `models/xgb_race_blind.json`). Race is an input to `race_aware` only.
- **Comparison:** African-American vs **everyone else combined** (`race_group()`). Every defendant falls into one of the two groups, so there's no small-sample group to exclude — unlike an earlier version of this file, which compared each race individually against Caucasian defendants and had to drop Asian (n=11) and Native American (n=2) as too small to test. Sex and other within-race comparisons are out of scope here.

## Step 1: is the model unfair?

| Definition | Null hypothesis | Meaning | Test |
|---|---|---|---|
| Statistical parity (SP) | ŷ ⊥ D | Same flag rate in both groups. | two-proportion z-test (Agresti-Caffo) + 95% CI on the flag-rate gap |
| Conditional statistical parity (CSP) | ŷ ⊥ D \| C | Same flag rate for defendants with the same legitimate risk factors C. | three ways, below |
| Equalized odds (EO) | ŷ ⊥ D \| Y | Same error rates (FPR, FNR) in both groups, given the true outcome. | two z-tests: FPR gap, FNR gap |

D is "African-American" vs "Rest". C is 6 strata: **prior-offence band** (0 / 1–3 / 4+) × **charge degree** (felony / misdemeanour), built by `make_strata()`. Y is the actual two-year re-offence outcome. EO is the one definition here that looks at Y, not just ŷ — it's close to what ProPublica's original 2016 COMPAS investigation measured (unequal false-positive/false-negative rates), as opposed to SP/CSP, which is closer to what Northpointe's calibration defence was about.

**CSP is reported three ways, deliberately redundant:**
- **Hurlin's summed likelihood-ratio test** (the paper's own method): per stratum, build a 2×2 table (flagged × group) and compute G = 2·Σ O·log(O/E); sum over strata. Under H₀, χ²(q), q = strata used. Makes no assumption about how the effect varies across strata.
- **Cochran-Mantel-Haenszel (CMH) test**: χ²(1), via `statsmodels.stats.contingency_tables.StratifiedTable.test_null_odds()`. Assumes a single common odds ratio across strata — generally more powerful than Hurlin's test when that assumption holds.
- **Mantel-Haenszel pooled odds ratio + 95% CI**: the *size* of that common effect, not just whether it's zero.
- **Breslow-Day test** (bonus, printed alongside): checks whether the common-odds-ratio assumption CMH/MH rely on is actually reasonable here. A low p-value means CMH/MH shouldn't be trusted and Hurlin's more conservative test is the one to read.

**Dropped strata.** A stratum is dropped if it has no members of one group, or if the model gives everyone in it the same prediction (it then carries no information about H₀). This happens here: **neither model flags anyone with 0 prior offences**, so CSP uses 4 of the 6 strata.

**Verification.** Every formula was checked before trusting it on real data: the z-test and CI match `statsmodels.stats.proportion` calls directly; the stratified-table construction (which row/column is which) was checked against a hand-computed CMH statistic from the textbook formula, and against the sign of the resulting odds ratio on a synthetic example with a known direction of effect (all in `tests/test_fairness.py`).

## Step 2: what's driving it? (FPDP)

The Fairness Partial Dependence Plot sets one feature to a value *v* for **every** defendant, re-predicts, and re-runs the **statistical parity** test only (a deliberate scope choice: SP is the simplest test, matches the paper's own worked example, and keeps one curve per feature rather than three). Repeated for every value observed in the data, then plotted as p-value vs *v*.

**A feature is a candidate variable** (Definition 6) if some value lifts the SP p-value above 5% — i.e. fixing that feature at that value would remove the statistical evidence of unfairness in the model's *current* form.

**Degenerate values are excluded from candidacy.** At some values the model gives *everyone* the same prediction — e.g. `Number_of_Priors = 0` flags nobody, `≥ 9` flags everybody. The test then passes trivially (very high p-value, not always exactly 1 with unequal group sizes — see the note in the tests), but that says nothing about the feature's role. These points are marked `degenerate` and ignored by `candidate_variables()` unless `include_degenerate=True`.

Computed for `race_aware` only, since that's the model where race is an explicit input.

## Step 3: can it be fixed?

Two ways to act on a candidate variable (or, if none crosses 5%, on the closest one, clearly labelled as a demonstration rather than a fix), both re-checked against SP, CSP and EO:

- **Re-estimation** (`mitigate_by_reestimation`): drop the feature entirely and retrain from scratch, tuned the same way as the real models (`tune_xgboost`). The model has to re-learn from the remaining features.
- **Substitution** (`mitigate_by_substitution`): keep the original model, fix the feature at one value for everyone (reuses `apply_fixed_value`, the same mechanism FPDP sweeps over).

These are genuinely different interventions, and — as the results below show — they don't do the same thing.

## How to run
```bash
# 1. once: train and save both models
python xgboost/xgb_model.py                                   # race_aware
python -c "import sys; sys.path.insert(0, 'xgboost'); import xgb_model; xgb_model.main('race_blind')"

# 2. print Step 1 (both models), Step 2 and Step 3 (race_aware) -- a few seconds
python xgboost/fairness.py

# 3. tests (all fast)
pytest tests/test_fairness.py
```
The notebook's section 4 draws the charts. Nothing is cached, because every step runs in seconds.

### Main functions
| Function | Returns |
|---|---|
| `predict_labels(model, data)` | ŷ at threshold 0.5 |
| `race_group(data)` | True for African-American, False for Rest |
| `make_strata(data)` | the CSP stratum of each defendant |
| `group_rates(y_pred, data)` | n, flag rate and actual re-offence rate per group |
| `statistical_parity_test(y_pred, protected)` | z, p-value, 95% CI on the flag-rate gap |
| `conditional_statistical_parity(y_pred, protected, strata)` | Hurlin, CMH, MH odds ratio + CI, Breslow-Day |
| `equalized_odds_test(y_pred, y_true, protected)` | z-tests on the FPR gap and the FNR gap |
| `evaluate_fairness(y_pred, data, strata)` | SP + CSP + EO bundled, for before/after comparisons |
| `fpdp(model, data, feature)` / `fpdp_all(...)` | SP-based FPDP data for one feature / all features |
| `candidate_variables(curves)` | Definition 6, degenerate values ignored |
| `apply_fixed_value(model, data, feature, value)` | ŷ with the feature fixed for everyone |
| `refit_without_feature(train, feature)` | a model retrained without that column |
| `mitigate_by_reestimation` / `mitigate_by_substitution` | Step 3's two interventions, pre-evaluated |

## Results

### Step 1: SP, CSP, EO (threshold 0.5)
| Model | SP: flag rate (AA vs Rest) | SP diff (95% CI) | SP p | CSP Hurlin p (4 df) | CSP CMH p | MH odds ratio (95% CI) | Breslow-Day p | EO: FPR gap (p) | EO: FNR gap (p) |
|---|---|---|---|---|---|---|---|---|---|
| race_aware | 0.506 vs 0.239 | +0.267 [0.225, 0.309] | < 0.001 | < 0.001 | < 0.001 | 2.22 [1.65, 3.00] | 0.69 | +0.159 (< 0.001) | −0.287 (< 0.001) |
| race_blind | 0.501 vs 0.238 | +0.263 [0.220, 0.305] | < 0.001 | < 0.001 | < 0.001 | 2.13 [1.58, 2.87] | 0.64 | +0.159 (< 0.001) | −0.281 (< 0.001) |

Both models reject all three fairness definitions decisively. The Breslow-Day p-values (0.69, 0.64) are high, meaning CMH and the MH odds ratio are trustworthy here — the common-odds-ratio assumption they rely on isn't contradicted by the data.

**Reading the EO row:** African-American defendants who did *not* re-offend are flagged about 16 points more often than everyone else who didn't (FPR gap), and African-American defendants who *did* re-offend are flagged about 29 points *less* often (FNR gap is negative: Rest's FNR minus AA's FNR is positive, so AA's FNR is lower — meaning AA re-offenders are, if anything, caught *more* reliably; it's the FPR gap that's the concerning one). This is the same shape of disparity ProPublica's original investigation reported.

### Step 2: FPDP candidate variables (race_aware, statistical parity)
No feature alone lifts the SP p-value above 5%. `Number_of_Priors` gets closest: fixed at 1, p = 0.018 — still short of 0.05.

### Step 3: mitigating `Number_of_Priors` (the closest, not a true candidate)
| Method | SP flag rate (AA vs Rest) | SP diff | SP p | MH odds ratio | EO: FPR gap (p) | EO: FNR gap (p) |
|---|---|---|---|---|---|---|
| baseline (race_aware) | 0.506 vs 0.239 | +0.267 | < 0.001 | 2.22 | +0.159 (< 0.001) | −0.287 (< 0.001) |
| **re-estimation** (drop the feature, retrain) | 0.540 vs 0.138 | **+0.401** | < 0.001 | **9.89** | +0.373 (< 0.001) | −0.401 (< 0.001) |
| **substitution** (fix at 1 for everyone) | 0.209 vs 0.166 | **+0.043** | 0.018 | 1.76 | +0.052 (0.021) | +0.004 (0.87) |

**This is the headline finding of Step 3, and it's counter-intuitive: the two mitigations pull in opposite directions.**
- **Re-estimation makes the disparity substantially worse** — the SP gap grows from 27 to 40 points, and the MH odds ratio roughly quadruples (2.22 → 9.89). Dropping the model's single strongest, most legitimate predictor doesn't remove its reliance on a proxy for race — it *forces* the model to lean harder on the remaining, weaker features (age, sex, charge degree), which turn out to be worse proxies for race relative to their signal about actual recidivism. Removing a candidate variable is not automatically a fix.
- **Substitution meaningfully shrinks the disparity** without retraining anything: the SP gap falls from 27 to 4 points, and the FNR gap very nearly closes (from −0.287 to +0.004). CSP and CMH still reject at this value (not shown in full above — see `python xgboost/fairness.py` for the complete printout), so it's a large improvement, not a complete fix.

## Caveats
- **One threshold.** All three tests depend on the 0.5 cut-off used for ŷ; a different cut-off changes every flag rate, FPR, and FNR.
- **FPDP is SP-only by design**, a deliberate scope choice (see Step 2). CSP or EO could be swept the same way — `fpdp()` would need generalizing to accept which test to re-run — but aren't here.
- **Impossible combinations in FPDP.** Setting a race dummy for everyone creates profiles that can't exist, such as `African_American = 1` and `Hispanic = 1` simultaneously. That's inherent to FPDP with one-hot features.
- **Strata are fixed** to each defendant's real prior-offence count, so they don't change when FPDP or substitution overrides `Number_of_Priors` for prediction.
- **Re-estimation's substitution-vs-retraining trade-off is specific to this feature and this cohort.** It shouldn't be read as "removing features always backfires" — only that `Number_of_Priors` happens to carry more legitimate signal than proxy risk, relative to what's left once it's gone.
- **Real base-rate differences complicate all of this.** African-American defendants in this cohort do have a higher actual two-year re-offence rate (see the notebook's rate chart in section 4.1). No feature intervention can achieve statistical parity, equalized odds, and calibration to the true outcome simultaneously when the true base rates differ across groups — that's a mathematical impossibility result (Kleinberg/Chouldechova), not a modelling failure. See [`race_proxy.py`](race_proxy.py) and its section below for the mechanism (priors as a race proxy) that produces this tension in the first place.

### Is race recoverable from the race-blind features?
The ablation above shows that dropping race from the model's inputs barely changes the disparity; [`race_proxy.py`](race_proxy.py) shows why: a fresh XGBoost model predicting **race itself** (African-American vs everyone else) from the race-blind features scores well above chance (AUC 0.683 on all 5 race-blind features, 0.644 on `Number_of_Priors` alone — see `python xgboost/race_proxy.py`). Race is recoverable from features that never mention it, mostly through `Number_of_Priors`, which is exactly the feature Step 2/3 above single out.
