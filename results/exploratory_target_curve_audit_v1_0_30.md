# Exploratory target-session curve audit

This report uses already-generated historical JSON files only. It does not
train a model. For each subject, it compares the recorded final target-session
accuracy with the maximum target-session accuracy observed during that run.
Because this selects on the held-out target labels, the maximum is an
exploratory upper-bound diagnostic and must not be reported as a formal
reproduction result.

| Historical artifact | Final mean +/- sample SD | Per-subject max-test mean +/- sample SD | Mean gain |
|---|---:|---:|---:|
| `all_pool_dropout050_after_pool_1000` | 70.83% +/- 8.95% | 74.27% +/- 9.23% | +3.43 pp |
| `all_pool_dropout050_after_pool_seed1_1000` | 69.10% +/- 7.81% | 71.91% +/- 8.29% | +2.82 pp |
| `all_pool_dropout_1000` | 65.93% +/- 8.40% | 68.56% +/- 8.07% | +2.62 pp |
| `all_parallel_no_fusion_no_eca_1000` | 63.85% +/- 7.79% | 67.36% +/- 7.34% | +3.51 pp |

The paper reports 81.17% for BCI IV 2a within-subject AMEEGNet. The historical
maximum-test audit remains below that value in every listed configuration, so
epoch selection alone is not a sufficient explanation for the reproduction
gap. This does not prove the paper used a different selection rule; it only
rules out a simple explanation for these saved runs.

No new test labels were used to train or modify any model. The definitive
formal result remains the final-only nine-subject batch, not this audit.
