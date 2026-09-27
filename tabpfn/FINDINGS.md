# TabPFN: findings log

Results so far for the foundation-model part, with the numbers to quote. Every figure here is
reproducible from the committed `tabpfn/artifacts/` (`make tabpfn-plots`), `make proxy-check`
and `make split-balance`. Methods and file layout are in [README.md](README.md).

**Scope.** Sections 1–4: predictive performance. Sections 5–7: fairness, stability and
interpretability, from the numbered tests in `tabpfn/analysis/` (T01–T18, outputs in
`tabpfn/artifacts/analysis/`). Section 8: what the earlier single-model study
(`artifacts/legacy/`) adds.

**Terms.**
- **COMPAS tool**: Northpointe's own risk score (`score_factor`, 1 = medium/high risk), the
  incumbent TabPFN is compared against. It is not the dataset.
- **t = 0.5**: the default threshold.
- **t = 0.252**: the cost break-even threshold ($13.5k per unneeded detention, $40k per
  missed re-offence). These costs are assumptions.
- **Race leakage**: how well race (African-American vs Caucasian) can be predicted from a
  feature set's non-race features, as a cross-validated AUC. 0.5 = no race information.

---

## 1. Headline performance

Holdout 70/30 test set (n = 1,852), all features:

| | TabPFN, t = 0.5 | TabPFN, t = 0.252 | COMPAS tool |
|---|---|---|---|
| AUC [95% CI] | 0.736 [0.713, 0.758] | same | 0.657 |
| Accuracy | 0.682 | 0.588 | 0.661 |
| Recall (power, 1 − β) | 0.562 | 0.888 | 0.613 |
| Type I error (FPR, α) | 0.218 | 0.663 | 0.298 |
| Cost per defendant | $9,573 | **$6,907** | $9,235 |
| Calibration error (ECE) | 0.018 | same | n/a (binary score) |

- **TabPFN ranks defendants clearly better than the COMPAS tool**, and its probabilities are
  well calibrated (ECE 0.018).
- **The threshold is a trade-off.** At 0.252, TabPFN catches 89% of re-offenders but
  flags 66% of those who would not re-offend.
- **The economic margin is thin.** Detaining everyone costs **$7,355** per defendant.
  - TabPFN at 0.252 beats that by only $448 (6%).
  - At 0.5, TabPFN costs more than detaining everyone.
  - So does the COMPAS tool ($9,235).
  - Under these cost assumptions, a false negative is so expensive that "detain everyone" is
    a hard baseline to beat.

Other designs (TabPFN all features vs COMPAS tool, AUC):

| Design | TabPFN | COMPAS tool |
|---|---|---|
| X1+X2 → X3 | 0.713 | 0.666 |
| X1 → X3 / X2 → X3 | 0.712 / 0.714 | 0.666 |
| Time: first 40% → next 20% | 0.703 | 0.624 |
| Time: first 70% → last 30% | 0.702 | 0.636 |

- **X1 and X2 give the same AUC on X3** (0.712 vs 0.714). Whether they also give the same
  *decisions* is the stability analysis.
- **The time designs use a different cohort.** It is a dated cohort rebuilt from the raw
  export: 5,686 people, 33% base rate, against 45.5% in the modelling table. Compare them
  only with each other.

## 2. Removing protected attributes

Same model, retrained without each attribute. Holdout AUC [95% CI]:

| Features | AUC | Δ vs all | Cost, t = 0.252 |
|---|---|---|---|
| All | 0.736 [0.713, 0.758] | | $6,907 |
| No race | 0.734 [0.711, 0.757] | −0.002 | $6,937 |
| No sex | 0.734 [0.711, 0.756] | −0.002 | $6,974 |
| No age | 0.694 [0.670, 0.717] | −0.042 | $7,090 |
| No race, sex or age (priors + charge degree) | 0.683 [0.659, 0.707] | −0.053 | $7,638 |

- **Race and sex add nothing measurable.** Their CIs overlap almost entirely with all
  features, on every design.
- **Age is the protected attribute that carries predictive signal.**
- **Priors and charge degree alone still beat the COMPAS tool** (0.683 vs 0.657).

## 3. Race proxies: race-blind is not race-blind

| Features | Race leakage (AUC) | Holdout AUC | vs COMPAS tool |
|---|---|---|---|
| All | 0.671 | 0.736 | better |
| No race | **0.671** | 0.734 | better |
| No race or priors | 0.615 | 0.615 [0.590, 0.641] | worse |
| No race or any proxy (under 25 + sex) | 0.569 | 0.576 [0.554, 0.600] | worse |

- **Dropping the race columns removes no race information at all** (0.671 before and
  after). The EDA shows why: priors is strongly associated with race. Mean priors are 4.24
  for African-American defendants vs 2.29 for Caucasian (EDA figure 09).
- **Proxies**, from the EDA association matrix (Cramér's V ≥ 0.1 with a race group):
  priors (0.225), over 45 (0.157) and misdemeanour (0.103). Under 25 (0.09) is just below
  the cutoff, which is why some leakage remains in the last row.
- **Removing the proxies removes the model's usefulness.** Without priors, TabPFN falls
  below the COMPAS tool on every design (0.576 to 0.621 AUC).
- **At t = 0.252 the proxy-free models flag almost everyone:**
  - no race or priors: 98% flagged, specificity 0.03;
  - no race or proxies: 100% flagged, specificity 0.00.
  - With only under 25 and sex there are four possible scores, all above 0.252, so the
    model *is* "detain everyone" (figure `race_proxy_blind/06_score_distribution.png`).

**What this means for the report.**
- Fairness cannot be obtained by removing inputs. Blinding the model to race changes nothing,
  and removing race's proxies removes most of the predictive power.
- The strongest predictor, priors, is also the strongest race proxy. Priors reflect policing
  as well as behaviour, so the historical bias in the data flows through them.
- So disparity has to be **measured and mitigated on the outputs**: conditional statistical
  parity with priors as X_c, then the mitigation step. Dropping columns does not do it.

## 4. Splits: representative or not

Largest gap between a split's group shares and the full dataset's, in percentage points:

| Split | Race | Sex | Age band | Notes |
|---|---|---|---|---|
| Holdout test 30% | 0.75 | 0.40 | 1.77 (p = 0.18) | within noise |
| X1 / X2 / X3 | ≤ 0.11 | ≤ 0.09 | ≤ 0.12 | stratified on outcome × race × sex × age |
| Time windows | up to 2.7 | up to 3.5 (p < 0.001) | up to 2.5 (p = 0.011) | real drift in who was screened |

- **X1/X2/X3 were not balanced at first.** Stratified on the outcome only, X3 was 3.7 points
  off on age (p = 0.012). They are now stratified on protected attributes too, and a test
  keeps them within 0.5 points.
  - Because of this rebalance, X1+X2 → X3 reads **0.713**. The 0.737 quoted earlier came
    from the old partition. The test set changed, not the model.
- **Time windows are cut by date, not sampled.** The sex and age mix shifts over time, so a
  change between time windows can come from the population, not only the model.

## 5. Fairness (T01–T06, shared protocol)

Holdout, all features. Gaps read protected minus reference; p-values Holm-corrected.

| | TabPFN, t = 0.5 | TabPFN, t = 0.252 | COMPAS tool |
|---|---|---|---|
| Flag rate, African-American / Caucasian | 50.0% / 25.3% | 87.7% / 65.3% | 56.7% / 32.9% |
| FPR, African-American / Caucasian | 29.9% / 17.5% (**+12.4 pts**) | 80.0% / 56.5% (**+23.5 pts**) | 41.1% / 22.7% (+18.4 pts) |
| FNR gap | −29.2 pts | −14.3 pts | −20.3 pts |
| Conditional parity (proxies fixed), p | **0.165**, OR 1.09 | **0.001**, OR 1.73 [1.02, 2.94] | 0.025, OR 1.56 |
| FPR gap, Female − Male | −16.6 pts | −26.5 pts | −4.2 pts |
| Tightest certifiable δ, race FPR gap | 0.17 | 0.29 | 0.24 |

- **Statistical parity and equalized odds are rejected for race at both thresholds** (T01,
  T03), for TabPFN as for the COMPAS tool. For Hispanic and Other vs Caucasian they are not
  (though at 0.252 Hispanic defendants are flagged *less* than Caucasian ones with the same
  proxies: conditional parity rejected, OR 0.16).
- **At t = 0.5 the race gap is explained by the proxies; at t = 0.252 it is not** (T02).
  Among defendants with the same priors band, age band and charge degree, TabPFN at 0.5 flags
  African-American defendants at the same odds as Caucasian ones (OR 1.09, p = 0.165). At the
  break-even threshold a residual gap survives (OR 1.73). The threshold chosen for cost is the
  one that brings back a direct race effect.
- **The break-even threshold widens every gap.** The race FPR gap almost doubles (+12 → +24
  pts): lowering the threshold flags many more non-re-offenders, and more of them are Black.
  ProPublica's criticism applies to TabPFN at 0.252 more strongly than to the COMPAS tool.
- **TabPFN has a large sex gap the tool does not.** Women are flagged much less (FPR −27 pts
  at 0.252, conditional parity rejected, OR 0.02 at 0.252). TabPFN uses `Female` heavily.
- **Nothing is certified fair** (T04). No TabPFN model that uses priors is TOST-certified for
  African-American vs Caucasian at δ = 0.05; on holdout the tightest certifiable δ for the FPR
  gap is 0.17 (t = 0.5) to 0.29 (t = 0.252).
- **Race-blind does not remove the gap** (T05). Holdout at 0.252: race FPR gap +23.5 pts with
  race, +22.7 without. Only removing priors closes it, and then the model flags 98–100% of
  defendants (AUC 0.58–0.62). The one row certified fair for race (`race_priors_blind`, 0.252)
  is that degenerate case.
  - Conditional parity is rejected for race_aware at 0.252 (p = 0.001) but not for race_blind
    (p = 0.066): removing the race columns does remove the *direct* effect that survives
    conditioning, while the overall gap, carried by the proxies, stays.
- **FPDP** (T06, holdout): which single variable, set to one value for everybody, removes the
  disparity?
  - **Race, statistical parity: none**, at either threshold. No one variable carries the
    race gap. It comes from several proxies at once.
  - **Race, conditional parity at 0.252** (the residual direct effect): priors (everyone at
    1 prior), misdemeanour, and `African_American` itself (everyone treated as Black) are
    candidates. The residual gap passes through the race column and its interaction with
    priors.
  - **Sex**: `Female` is the candidate (everyone treated as male) for conditional parity at
    both thresholds, and priors for statistical parity at 0.252. The sex gap is the model's
    own use of the `Female` column.
  - Mitigation (step 3) is not done here: fixing `Female` or the race column is removing the
    attribute, which T05 shows costs little AUC for sex (0.736 → 0.734).

## 6. Stability (T07–T09, T17)

TabPFN has no estimated parameters (its weights are frozen), so two fits are compared on their
outputs. Design 1: X1 → X3 vs X2 → X3 (same 1,235 test defendants). Design 2: first 40% →
next 20% vs first 70% → last 30% by date.

| Design 1, all features | t = 0.5 | t = 0.252 |
|---|---|---|
| AUC | 0.712 vs 0.714 | same |
| Mean \|Δ score\| per defendant (max) | 0.030 (0.091) | same |
| Decisions that flip | **3.8%** (47) | **6.6%** (81) |
| Type I error | | 0.638 vs 0.710 (+7 pts) |
| Race FPR gap | +12.4 vs +15.5 pts | **+27.6 vs +12.4 pts** |
| Race conditional parity, OR | 1.37 vs 6.64 | **9.95 vs 0.55** |

- **Same AUC, different decisions** (T07, T08). The two fits agree on ranking (Δ AUC 0.002,
  score correlation 0.985), but 6.6% of defendants get a different decision at 0.252.
- **The fairness verdict depends on the training sample** (T09). With identical test
  defendants, the race FPR gap at 0.252 is +27.6 pts for the X1 fit and +12.4 for the X2 fit;
  the conditional odds ratio even changes direction (9.95 vs 0.55). The sex FPR gap is not
  significant for X1 (p = 0.21) and is for X2 (p < 0.001). A fairness audit of one fit
  says little about the next one.
- **Removing features does not make decisions steadier**: race_blind flips 13.7% of decisions
  at 0.252, twice race_aware. The sets with few features flip 0% only because they flag almost
  everyone.
- **The explanation is stable** (T17). Priors is the top feature for every fit and measure
  except marginal effects; Spearman between the X1 and X2 importance vectors is 0.94 (permutation
  AUC) and 0.96 (SHAP), 0.77 on cost. The story holds over time: 0.72–0.93 between the two time
  windows, with the same top 3 on permutation and SHAP.
- **Over time** (T08, T09): AUC stays at 0.70 (0.703 → 0.702). The cost rises by $398 per
  defendant, the race FPR gap narrows (+31 → +25 pts). Score distributions shift (PSI 0.16 for
  race_aware), partly from the population drift noted in section 4.

## 7. Interpretability (T10–T16)

| Measure | Holdout, all features | Stable across designs? |
|---|---|---|
| Permutation, AUC (T13) | priors 0.178 (72%), under 25 0.037, over 45 0.024, Female 0.005, African-American 0.002 | yes |
| Permutation, cost at 0.252 (T13) | priors $1,917, under 25 $445, over 45 $399, African-American $59 | mostly |
| Mean \|SHAP\| (T14) | priors 0.156, under 25 0.054, over 45 0.051, Female 0.021, misdemeanour 0.019, African-American 0.009 | yes |
| Marginal effect (T10) | under 25 +0.17, over 45 −0.13, Female −0.07, priors +0.067 per prior, African-American +0.015 | not for small groups |
| Surrogate impurity, gini (T11) | priors 46%, under 25 27%, over 45 17%, misdemeanour 11%, race 0% | yes |

- **Every method agrees on the ranking**: priors, then age, then sex, with race small. Race is
  non-zero on AUC (permutation interval [0.0007, 0.0034]) and on cost ($59 [9, 137]), but
  gets no split in the surrogate tree.
- **Marginal effects of the small groups are noise** (T10). Asian: −0.12 on holdout, −0.02 to
  −0.06 elsewhere; Native American: −0.12 to +0.06. TabPFN draws inferences from a handful of
  defendants, as the earlier study's negative XPER contributions for Asian and Other showed.
- **Misdemeanour costs money** (T13): shuffling it lowers cost by $146 [3, 268]. At 0.252,
  TabPFN uses the charge degree in a way that loses more than it saves.
- **Surrogate fidelity is high** (T11): R² 0.89–0.93 on scores, 91–97% agreement on decisions.
  Gini, entropy and misclassification rank the top features the same way, except that
  misclassification error puts under 25 first (54%) and priors second: at the decision
  boundary, age flips more decisions than priors.
- **PDP / ICE** (T12): from the lowest to the highest priors on the grid, risk rises by +0.52
  on average (sd 0.045 across defendants), and never falls for anyone: the PDP describes
  everyone. The effect is smaller in the time-ordered cohort (+0.42), whose base rate is lower.
- **SHAP is exact enough** (T14): efficiency holds to 2e-6 on every run. Cost: 2–12 s per
  explained defendant.
- **LIME is not reproducible** (T15), as in the earlier study: 15 runs on one borderline
  defendant name 4 different top features (priors 6, Native American 5, Other 3, Asian 1), and
  9 of the 15 name a race the defendant does not have.
- **XPER on cost** (T16), 100 defendants, 1,024 coalitions, efficiency error 0:
  - on this sample the model costs **$8,455** per defendant against **$6,885** for the
    no-information benchmark (detain everyone): 10 missed re-offenders at $40k weigh $4,000 a
    head. On the full holdout it is the other way round ($6,907 vs $7,355). 100 rows are too
    few for a cost decomposition; read the shares, not the level;
  - the features that raise cost most are over 45 (+$577), priors (+$482), under 25 (+$274)
    and misdemeanour (+$266): the ones that make the model release people. African-American
    lowers it (−$50).

## 8. What the earlier study adds (`artifacts/legacy/`)

Threshold-free results from the earlier single-model study (a slightly different fit, AUC
0.7355 vs 0.7358, same holdout). Numbers that depend on its 0.21 threshold are not reused.

- **XPER on AUC**: priors +0.119, under 25 +0.052, African-American +0.030; Asian, Other and
  Female contribute negatively: TabPFN learns noise from small groups.
- **100 bootstrap refits**: 24.4% of decisions unstable at 0.252 (19.4% at 0.21). **Row
  order**: 23 of 1,852 decisions change when only the order of the training rows changes
  (at 0.252). **Seeds**: AUC sd 0.0007.
- **Learning curve**: 0.719 at n = 250, 0.730 at 500, flat from 1,000.

## 9. Caveats

- **Native American defendants:** there are about 11 in the whole dataset (1 in X3).
  Per-group numbers for them are noise. Asian and Native American are left out of the
  fairness comparisons.
- **Timings:** four `race_priors_blind` runs were reused from an interrupted run, so their
  fit/predict times in `runs.csv` are blank. Predictions and metrics are complete.
- **Cost assumptions:** every cost result depends on the $40k / $13.5k assumption. A
  different ratio moves the break-even threshold and the "detain everyone" baseline.
- **Explanation budgets:** marginal effects and PDP use 500 test rows per run, SHAP 25, XPER
  100 × 10 background rows. Permutation importance uses every test row.
