# Evaluation protocol fixed before downloading post-2020 returns

November 7, 2026. This decision supersedes the provisional 1975–2020 split in the earlier design. The original paper data end in December 2020. The authors have since published a portfolio-return release through December 2025. That permits a test on later calendar months which the native reproduction cannot expose. The newer release uses a revised construction pipeline. Use **one current vintage for every extension partition** and the historical archive for native replication. Validate overlapping definitions and units using Training. Record construction differences.

| Role | Inclusive dates | Calendar months | Authorized use |
|---|---|---:|---|
| Warm-up | Through December 1974 | As needed | Prior 12 monthly market observations for the first state |
| Training | January 1975–December 2009 | 420 | Universe, clustering, transformations, forward tuning, final fit |
| Development Test | January 2010–December 2020 | 132 | Frozen diagnostic evaluation |
| Final Validation | January 2021–December 2025 | 60 | Sole primary endpoint after code, models and checks freeze |

The shares of the 612 target months are 68.6%, 21.6% and 9.8%. Use the first 210 Training months to establish eligibility by consecutive, finite monthly returns and nonzero variation. Freeze the eligible IDs. The three internal validation blocks are the following 70 months apiece. Fit all scales, 13 groups and volatility cutoff within each expanding prefix. For the outer model refit using Training. Require unique month/asset keys and complete necessary outcomes after eligibility. Report a missing-data failure explicitly. Each inner prefix uses only the earlier subset of the same eligible IDs. Analysis of the time-varying asset set is outside this design.

Fix the state as a lagged 12-month market standard deviation (sample denominator 11), above the 2/3 Training quantile (linear interpolation). Freeze the Training frequency used for centering/scaling the indicator. For clustering, market-adjust each factor using its own training covariance/market variance and compute Ward.D clusters using dissimilarity 1-correlation. Cluster membership is an estimated historical representation. Use raw, signed and scaled factor payoffs for constructing economic returns.

Use the four-cell comparison, penalty grid, sliding window, loss and uncertainty in the methodology document. Fix seed 340 for 999 paired moving-block resamples per block length (12 primary, 6 sensitivity). Keep missing-asset, singularity and train-isolation checks in the acceptance report. State both high/low regime counts when evaluating Validation. This finite-sample loss compares the length-60 period's squared mean moments under identical fitted-model exposure. It combines sampling noise with population pricing errors.

Download archives mechanically, hash them, and expose only Training values to design/tuning. Seal later values until their designated stage. If the newer inputs cannot be validated, report that obstruction before changing the endpoint. Native-only results already exposed the pre-2021 periods. Exposure of the new 60 months to the user or other prior research is unknown. The strongest justified claim is a prospective computational evaluation of a post-parent sample within this project.

Author sources: [data download](https://jkpfactors.com/data), [release announcement](https://github.com/bkelly-lab/jkp-data/discussions/89), [release notes](https://jkpfactors.com/release-notes).
