# v1.0.19 max-norm nine-subject audit

The deterministic T-to-E batch used average pooling, Dropout=0.5 after
pooling, standard EEGNet max-norm projection, and no v1.7 enhancement.

| Mean accuracy | Sample SD | Difference from paper mean | Runtime median |
| ---: | ---: | ---: | ---: |
| 71.60% | 9.06% | -9.57 pp | 212.08 s |

The held-out E session was evaluated once after 1000 epochs. This is still a
controlled reconstruction hypothesis, not a confirmed complete reproduction
of the under-specified original implementation.
