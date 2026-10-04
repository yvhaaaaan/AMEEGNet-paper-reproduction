# v1.0.17 standard pooling/dropout nine-subject audit

The complete deterministic batch used the same T-to-E protocol, seed 42,
Adam, 1000 epochs, and final-only target-session evaluation for A01--A09.
It used average pooling and Dropout=0.5 after pooling in each EEGNet branch.

| Mean accuracy | Sample SD | Difference from paper mean | Runtime median |
| ---: | ---: | ---: | ---: |
| 68.29% | 8.55% | -12.88 pp | 209.79 s |

This is a controlled standard EEGNet branch hypothesis, not a confirmed full
original-paper reproduction. No v1.7 enhancement or test-curve selection was
used.
