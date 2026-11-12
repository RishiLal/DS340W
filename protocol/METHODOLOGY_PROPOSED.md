# Proposed methodology: stability tuning for conditional theme pricing

**November 6 proposal. Use OUTER_PROTOCOL_20261107.md and METHOD_AMENDMENT_20261107.md for the current dates, universe, and normalization.**

## 1. Plain-English recommendation

Use JKP's interpretable factor themes to ask whether a small pricing model remains useful in predictable high-volatility months. Compare ordinary penalty tuning with the exact local-window selection rule from time-series DML, adapted to pricing-moment validation error. Keep the model linear in 13 theme payoffs plus the market. Use one nonlinear, predetermined volatility indicator to challenge its pricing restrictions. This retains the economic moment idea from Chen–Pelger–Zhu (CPZ) without a neural network, generated cross-sectional covariance regressions, or claims of causal factor survival.

This is a limited empirical methodology contribution: a controlled interaction experiment, not a new general estimator or a reproduction of CPZ or time-series DML. If an operational DML coefficient estimate is essential, this design does not deliver it; adding such an estimate would require a different target and an inference-calibration study. The brief permits a meaningful borrowed component, so the smaller tuning adaptation is preferred.

## 2. Parent and alternatives

See `SELECTION.md` for the three-candidate screening decision and access links. JKP's Bayesian alpha target remains the **native reproduction target only**. The extension changes the research target to prospective pricing-moment loss for a frozen SDF candidate. Never compare its coefficients to JKP posterior alphas or the old project's nonlinear derivatives/linear-adjusted slopes. Retain the factor-zoo question about redundant versus useful pricing information; abandon individual-factor survival and causal interpretations.

## 3. Question and hypotheses

Primary question: Does adding a single lagged-volatility conditional pricing restriction improve later-period pricing errors when a theme-based SDF's penalty is chosen using a stability rule?

Let U/C mean unconditional/conditional training moments and M/G mean minimum-validation-error/Goldilocks tuning. All four models are evaluated on the **same conditional-and-unconditional test-payoff dictionary**. Let L denote squared norm of mean pricing errors in a fixed future regime, using frozen training scales. For independently repeated evaluation samples conditional on the frozen fits, the target contrast is

\[
\Delta=E[L(C,G)-L(U,G)\mid\text{frozen training fits}].
\]

H0: Delta >= 0; H1: Delta < 0. The observed sample contrast estimates this target under a stable evaluation sampling regime. The empirical exercise is exploratory because native replication exposes the historical data. No confirmatory rejection or investment claim follows solely from a negative observed contrast.

One essential secondary question isolates the interaction:
\[
I=\{L(C,G)-L(U,G)\}-\{L(C,M)-L(U,M)\}.
\]
A negative I supports a tuning-by-conditioning interaction in this sample; nonnegative I contradicts that directional hypothesis. A negative Delta with nonnegative I supports conditioning, not a special interaction benefit. Report both component effects rather than labeling any favorable cell the winner.

## 4. Contribution provenance and implementation map

Names below are **planned modules**, not delivered extension code.

| Component | Source/equation | Borrowed or modified element | Project-specific contribution | Planned module | Validating comparison/test |
|---|---|---|---|---|---|
| Native empirical Bayes | JKP Section I and Appendix B, author `emp_bayes` | Unmodified alpha posterior and original bootstrap/default settings | Numerical reproduction audit only | Delivered `tools/run_native.R` | Published Table I; replication rates; complete output inventory |
| Theme representation | JKP `2 - Determine Clusters.R` | Ward.D clustering of market-adjusted return correlations; 13 groups | Relearn clusters using permitted training observations; omit future labels and future-volatility scaling | `prepare.py` | Perturb all later data; cluster map/scales must be unchanged |
| Economic loss | CPZ Section 2, conditional Euler moments | Finite conditional test payoffs, linear SDF, analytic bounded linear adversary | One predeclared lagged-volatility instrument and shared evaluation dictionary | `pricing.py` | U versus C with same SDF dimension and samples |
| Stability selection | Ciganovic et al. v2 equations 3.9–3.12 | Exact min–max window scoring, S=3, within-window minimum | Apply to pricing-moment validation RMSE rather than DML nuisance prediction | `tuning.py` | M versus G on identical stored loss grid |
| Time splitting | Time-series DML Section 2.1 plus project chronology | Primary forward-only tuning; RCF described but not required | Avoid imposing reversibility on historical financial states | `splits.py` | Train/validation chronology and split invariance |
| Interaction | This project; factorial difference above | Standard matched 2x2 comparison | Test whether stability tuning changes the out-of-training cost/benefit of conditional restrictions | `evaluate.py` | Paired interaction I with common months, scales and bootstrap indices |

CPZ retains flexible conditional moments in its original model. Here neural SDF weights, macroeconomic LSTM, firm-level characteristics and neural adversarial training are omitted. A fixed nonlinear instrument is not deep learning. The meaningful link is the pricing-moment restriction and the deliberate search for poorly priced managed payoffs.

## 5. Mathematics and correctness audit

### Observations and representation

Time t is a month; i indexes JKP long-short characteristic portfolios, not independent statistical observations. Let r_t in R^N be decimal monthly zero-cost returns, with market excess return appended as a separate asset. No additional risk-free subtraction is made from long-short returns or `mkt_vw_exc`. Signs follow published factor metadata, which is retrospective research information, not a historical live-trading signal.

Build K=13 clusters on the training sample using the parent's correlation construction. Within cluster c, define f_ct as the equal-weight average of direction-adjusted member returns divided by each member's training standard deviation. Append the market scaled by its training standard deviation. Thus f_t is in R^p, p=14, and is a vector of scaled, zero-cost payoffs. Scale without subtracting means: demeaning payoffs would change the pricing restriction. All scales are frozen, positive and finite. These theme weights differ from the native full-sample construction and must be labeled adapted.

Let v_(t-1) be the standard deviation of the preceding 12 market returns, ending at t-1. Let c be its training-only 2/3 quantile and p_H its training frequency above c. Define
\[
h_{t-1}=\frac{1\{v_{t-1}>c\}-p_H}{\sqrt{p_H(1-p_H)}}.
\]
Stop if p_H is 0 or 1. This is a nonlinear **predetermined** instrument; it does not use month-t volatility or revised macro data. Both the quantile and normalization must be recomputed inside each inner fit, then frozen for its validation block.

Write x_t = diag(s_r)^(-1) r_t using training-only asset standard deviations. Test-payoff vectors are q_U,t=x_t in R^N and q_C,t=(x_t',h_(t-1)x_t')' in R^(2N). Scaling and normalization make all entries dimensionless; report unscaled unconditional pricing errors separately in monthly basis points.

The SDF candidate is M_t(b)=1-b'f_t. The constant 1 is fixed: this is not an assertion that E[M]=1, nor an estimate of the price of a risk-free bond. Restricting to excess/zero-cost payoffs permits this normalization. M can be negative; this experiment tests finite Euler moments and does not establish a positive admissible SDF or global absence of arbitrage.

### Objective and closed form

For a training sample S of n months and chosen dictionary q in R^d,
\[
\mu_q=n^{-1}\sum_{t\in S}q_t,\qquad
A=n^{-1}\sum_{t\in S}q_t f_t'\in\mathbb R^{d\times p},\qquad
e(b)=\mu_q-Ab.
\]

Choose b by
\[
J_\lambda(b)=d^{-1}\|e(b)\|_2^2+\lambda\|b\|_2^2,\qquad
\widehat b_\lambda=(A'A/d+\lambda I_p)^{-1}A'\mu_q/d.
\]

Use a linear solve, never explicit numerical inversion. For lambda>0 the Hessian 2(A'A/d+lambda I_p) is positive definite, so the solution is unique. Do not penalize a free intercept: there is no free intercept in this normalization. Record singular values and coefficient/gross-exposure magnitudes even when the ridge system is invertible.

Independent derivation: solve the augmented least-squares problem with design [A/sqrt(d); sqrt(lambda)I_p] and response [mu_q/sqrt(d);0]. Its normal equations equal the displayed solution. A separate numerical known-answer audit must compare the two, finite-difference gradients and direct optimization. This is linear algebra, not an asymptotic identification theorem.

The analytic adversary identity is
\[
\max_{\|a\|_2\le1}(a'e(b))^2=\|e(b)\|_2^2,
\]
by Cauchy–Schwarz, attained at a=e/||e|| if e is nonzero. Thus this finite adversary is **exactly GMM with an identity weight**, not a novel GAN, and cannot be counted as a separate treatment from the corresponding moment norm. This rules out an attractive but false originality claim. The project change is the conditional test dictionary and its interaction with stability tuning.

Crucial distinction: (mean Mq)^2 is not mean (Mq)^2. The latter penalizes state-by-state variation and estimates a different objective. No Neyman-orthogonal DML score appears here; b is not a causal coefficient or an average derivative. Ridge guarantees a finite solution, not correct economic pricing.

### Goldilocks and time dependence

Use the increasing, fixed dimensionless grid lambda=10^a for 31 equally spaced a from -8 to 1. Compute each candidate's validation pricing RMSE on the same q_C dictionary, even when trained on q_U. Use three forward inner validation blocks after an initial training prefix, with all transformations fitted on the available prefix. Aggregate squared moment errors across validation blocks, then take one square root per lambda. The exact split indices must be frozen after inspecting schema/date availability and before any extension fit.

For adjacent windows W_j of size 3, compute mean RMSE m_j and population variance V_j. Score minmax(m_j)+minmax(V_j); select the lowest-score window, then its lowest-RMSE lambda. Constant normalization terms contribute zero. Resolve ties by smallest grid index. Ordinary tuning chooses the global RMSE minimum on precisely the same grid/losses. Changing the loss from nuisance RMSE to pricing RMSE is a project adaptation, not the source's inference procedure.

RCF verification: source Section 2.1 takes the larger of the past and future auxiliary blocks; with five equal blocks it uses the future for folds 1–2, both sides for fold 3, and past for folds 4–5. The source averages fold slopes (equation 2.5); it does not define the old project's pooled portfolio coefficient. Within a retrospective inference sample, later auxiliary observations are not automatically forecast leakage, but they cannot enter an allegedly available-at-time-t prediction. Joint reversibility, weak dependence, score orthogonality, nuisance rates, nonzero Jacobian and conditional stability are separate requirements. Stationarity alone is insufficient; even vector Gaussian processes need appropriate lag cross-covariance symmetry for joint reversibility. We therefore **do not transfer its theorem or make RCF a necessary primary component**. If added as a sensitivity, it is labeled a retrospective block-tuning adaptation and isolated from outer evaluation.

### Assumptions and support classification

| Claim | Support | Limitation |
|---|---|---|
| Ridge solution, adversary equivalence, augmented least squares | Algebraic derivation; numerical audit required | Establishes optimizer, not economic model truth |
| Predictable h makes h*r a valid conditional test payoff under Euler restrictions | Conditional-expectation identity with integrability | Underlying conditional pricing model may be false |
| Goldilocks selected lambda can improve later pricing | Proposed empirical hypothesis | No theorem or guaranteed advantage |
| Frozen fits cannot change when later data change | Software test required | Does not erase historical research exposure |
| Block-bootstrap uncertainty is useful | Exploratory empirical procedure | Needs stable/mixing evaluation regime, sufficient blocks; short final period limits reliability |
| DML inference, debiasing, causal effects | Not claimed | No compatible score/theorem supplied |

Cross-sectional dependence is retained by treating a full monthly return vector as one row. Stable evaluation-regime moments, finite product moments, lag availability, adequate representation and enough observations in both volatility regimes remain assumptions/diagnostics. No simulation can certify these for actual financial returns. As this is a pricing-loss project rather than coefficient inference, do not add a large coverage simulation merely to borrow DML terminology.

## 6. Strict pipeline and exposure accounting

Acquisition/hash -> unique (country, characteristic, month) keys -> metadata merge validation -> units and signs -> country and minimum-stock screens -> chronological roles -> training-only universe/clusters/scales/state threshold -> inner tuning -> freeze artifact -> development Test -> final Validation -> joint-month uncertainty -> report.

Proposed extension sample: US months January 1975–December 2020, subject to an actual missing-data audit. Prior months supply the 12-month lag and native replication; they are not discarded secretly. A complete 552-month panel implies Training 386, Test 110, Validation 56 (approximately 70/20/10). These are provisional date/count calculations until the data audit establishes usable rows. The split and any missingness exclusions must be finalized in a machine-readable protocol before fitting.

Select the universe using training availability only, not completeness through 2020. No missing-return imputation, backward fill, or future interpolation. If held-out outcomes are missing, stop the primary balanced-universe comparison and report the obstruction; do not silently choose a retrospectively complete universe. A predeclared monthly available-asset metric would be a separate target.

Use Test for diagnostics under the frozen design. Do not refit on Training+Test or choose a favorable method/seed. Validation is the **sole primary reported evaluation endpoint**; Test is development evidence. If Test reveals a necessary scientific redesign, version the protocol and disclose the loss of confirmatory status. Native results expose the entire historical sample; therefore even a sealed validation file provides only computational isolation. Neither native replication nor these data can support a claim of an untouched historical holdout. A truly additional period would require acquisition/versioning and a freeze before inspection; not assumed available.

For uncertainty, jointly resample aligned (r_t,f_t,h_(t-1)) rows in 12-month moving blocks, paired across models, with 6-month sensitivity and 999 scheduled draws. Recompute mean pricing moments, not artificial lags at resampled boundaries. Keep fits/scales/thresholds fixed and describe intervals as conditional and exploratory. Report the small number of effective blocks and failures; do not interpret raw bootstrap sign proportions as null p-values. Keep dimensionless combined pricing loss separate from unconditional pricing basis points, return prediction, portfolio Sharpe, and investment performance.

## 7. Native replication gate and extension plan

See `NATIVE_INVENTORY.md` and generated `evidence/JKP_output_calls.txt`. Native driver first, numerical comparison second, protocol/data freeze third, extension implementation fourth. Missing ancillary stock files remain blocked. If main baseline fails, preserve logs and reconsider JKP; no reduced-iteration or saved-output substitution. A completed default driver does not automatically certify all appendix variants.

## 8. Executable handoff

The delivered acquisition, environment setup, snapshot preparation and native runner are executable now. Extension module names in Section 4 are prospective. `README.md` must report their absence honestly until the gate passes. No author code/data with uncertain redistribution rights, virtual environment, private coursework or credentials should enter a public package. The local snapshot remains preserved.

## 9. Tests and acceptance criteria

Native: seed/settings unchanged, all requested default output calls executed; fresh table/figure values compared against the paper, Table I within rounding interval +/-0.005 percentage points; replication rates within source rounding precision; stochastic discrepancies reported with their Monte Carlo source, never tuned away. Figure trajectories need numerical series plus visual comparison where tables do not suffice. Script completion alone does not pass the scientific gate.

Extension: date/key/unit asserts; disjoint contiguous outer roles; inner transforms recomputed without validation data; later-data perturbation leaves trained model unchanged; return scaling leaves economic payoff invariant under inverse coefficient rescaling; known-answer ridge solve; zero-moment adversary case; constant-window and tie tuning behavior; same four-cell inputs/folds/grid; metric-square identity; joint resample indices; no cache reused across different data/protocol hashes. A clean dependency environment and launch from another directory must reproduce the same numerical outputs within declared tolerances. Same-laptop testing is not cross-machine certification.

## 10. Schedule and risks

Target schedule assumes work begins Friday evening, November 6, 2026. Two to three working days could mean November 9–11 if weekend days are not workdays; the brief does not settle this. The November 15 assignment date is user-supplied, not independently checked against the course system.

Day 1: acquisition, complete workflow inventory, native feasibility and reproduction, mismatch triage. Day 2: only after the gate, finalize split/target and implement the small 2x2 experiment with meaningful tests. Day 3: matched final evaluation, exploratory uncertainty, fresh-input clean-environment rerun, technical report and recording script. The recording, GitHub publication, Kaltura upload and submission remain user actions. If native reproduction or missing-data requirements fail, use the remaining time for a documented candidate decision; do not manufacture an extension completion.

Actual timings belong in the execution ledger. JKP full runtime is not known from the old parent's 73.48-minute run. DJM's reported 81 minutes on 24 cores/128 GB cannot be transferred to this six-core/16 GB laptop. Compilation, numerical-library differences, author-code age, inaccessible appendix, missing stock inputs, peak memory and small Validation blocks are the principal unresolved risks.

## 11. Closest prior work and contribution statement

Targeted searches included `Goldilocks asset pricing cross`, `stochastic discount factor conditional regularized adversarial GMM`, and `Empirical Asset Pricing with Many Test Assets`. They are a focused check, not a systematic proof of originality. Closest work includes CPZ's conditional adversarial Euler moments; the source DML paper's stability selector; JKP's clustered factor analysis; and Lonn–Schotman's *Empirical Asset Pricing with Many Test Assets* (Journal of Financial Econometrics, 2024, DOI 10.1093/jjfinec/nbae002), which already connects regularized tracking portfolios and maximally mispriced portfolios. The journal record and institutional copy confirm this citation.

The contribution of this project is **proposed** as a transparent test of whether local stability in penalty selection changes the benefit of adding a simple, predetermined volatility condition to pricing restrictions on factor themes. The project adapts existing clustering, pricing-moment estimation and stability tuning, and isolates their interaction using four matched models. It does not introduce a new DML estimator or claim worldwide novelty. A null or adverse result remains informative because it shows whether the extra conditional restriction and tuning rule earn their complexity under a common evaluation target. After execution, replace proposed benefits with the actual measured contrasts and limitations; no demonstrated improvement currently exists.

Sources: https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249 ; https://arxiv.org/html/2603.10999v2 ; https://pubsonline.informs.org/doi/10.1287/mnsc.2023.4695 ; https://arxiv.org/abs/1904.00745 ; https://academic.oup.com/jfec/article/22/5/1236/7630144 .

