# v1.0.22 no-temporal-BN nine-subject audit

The deterministic T-to-E batch used average pooling, Dropout=0.5 after
pooling, standard EEGNet max-norm projection, head ELU, and no BatchNorm after
the first temporal convolution. The held-out E session was evaluated once
after 1000 epochs.

| Mean accuracy | Sample SD | Difference from paper mean | Runtime median |
| ---: | ---: | ---: | ---: |
| 72.61% | 8.81% | -8.56 pp | 169.76 s |

This is closer than the previous controlled variants but remains below the
published 81.17% mean. Session-direction ambiguity is audited separately.
