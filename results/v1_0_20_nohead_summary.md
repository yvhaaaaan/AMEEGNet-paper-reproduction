# v1.0.20 no-head-ELU nine-subject audit

The deterministic T-to-E batch used average pooling, Dropout=0.5 after
pooling, standard EEGNet max-norm projection, and no inferred ELU after
Dense(32). The target Session was evaluated once after 1000 epochs.

| Mean accuracy | Sample SD | Difference from paper mean | Runtime median |
| ---: | ---: | ---: | ---: |
| 70.25% | 8.38% | -10.92 pp | 212.00 s |

This literal classification-head variant was lower than the corresponding
head-ELU batch, so the head ELU is not the current limiting discrepancy.
