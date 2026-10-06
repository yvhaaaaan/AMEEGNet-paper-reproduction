# v1.0.77 frozen source-checkpoint diagnosis

Four saved checkpoints were evaluated on their 288 source trials, CPU
batch32, without target evaluation, retraining, BN recalibration, or artifact
replacement. Saved input hashes, final model fingerprints, and unchanged
state/checkpoint hashes all passed. Results are in
`results/frozen_collapse_v1_0_77.json`.

| Checkpoint | Source accuracy | Source loss | Hidden pre-ELU range | Nearly constant post-ELU units |
| --- | ---: | ---: | --- | ---: |
| A02 seed1 | 25.00% | 1.386369 | -41.319 to -12.467 | 32/32 |
| A06 seed7 | 25.00% | 1.386495 | -102.345 to -14.351 | 32/32 |
| A02 seed2025 | 100.00% | 0.000049 | -8.177 to 26.124 | 0/32 |
| A06 seed2025 | 100.00% | 0.032960 | -77.659 to 53.528 | 26/32 |

Nearly constant means source-sample SD below the preregistered `1e-5`
threshold. The collapsed checkpoints have **every** hidden preactivation
below -10; post-ELU values are extremely close to -1, and all four output
logits have source-sample SD below `1e-5`. Source prediction counts are
`[0,0,288,0]` for A02 seed1 and `[288,0,0,0]` for A06 seed7, reproducing
single-class collapse on the training Session as well.

The two seed2025 checkpoints predict all source labels correctly. A06 still
has 26 nearly constant hidden units, so partial saturation alone cannot be
used as a failure rule: its remaining units support correct source decoding.

BN running variances in all four checkpoints are finite and positive. This
does not prove BN statistics are appropriate or exclude a role during
training. The read-only result explains the frozen collapsed models' nearly
constant predictions, but does not establish when or why optimization drove
the hidden layer into saturation. It is not evidence of a test-time random
fluctuation or solely overfitting to source data.

No original accuracy, checkpoint, or seed summary was changed. All 23 tests
passed before diagnosis, including immutable state and constructor-setting
checks. The diagnostic reconstruction helper now forwards saved convolution
bias and raw-transfer flags; training/model source files remain unchanged.
