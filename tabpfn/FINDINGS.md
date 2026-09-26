# TabPFN: findings log

Results so far for the foundation-model part, with the numbers to quote. Every figure here is
reproducible from the committed `tabpfn/artifacts/` (`make tabpfn-plots`), `make proxy-check`
and `make split-balance`. Methods and file layout are in [README.md](README.md).

**Scope.** Predictive performance only. Interpretability, stability and fairness are still to
do (README, "For the analysis"). Several findings below are inputs to them.

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

## 5. Caveats

- **Native American defendants:** there are about 11 in the whole dataset (1 in X3).
  Per-group numbers for them are noise.
- **Timings:** four `race_priors_blind` runs were reused from an interrupted run, so their
  fit/predict times in `runs.csv` are blank. Predictions and metrics are complete.
- **Cost assumptions:** every cost result depends on the $40k / $13.5k assumption. A
  different ratio moves the break-even threshold and the "detain everyone" baseline.

## 6. Open, for the analysis

- **Fairness:** do TabPFN's decisions still differ by race without the race columns? Check
  selection rate, FPR and FNR by race, race-aware vs race-blind, with and without
  conditioning on priors. The prediction files carry every group label.
- **Stability:** compare X1 vs X2 on X3 decision by decision (flip rate, L2 distance
  between scores), not just AUC.
- **Interpretability:** does TabPFN rely on priors and age the way the ablation suggests?
  SHAP and permutation importance should agree with Section 2.
- **Radar:** add these dimensions as extra spokes through `tabpfn/artifacts/radar_extra.csv`.
