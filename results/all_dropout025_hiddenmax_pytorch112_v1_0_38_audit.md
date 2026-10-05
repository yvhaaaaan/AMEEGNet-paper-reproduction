# PyTorch 1.12.1 reproduction batch audit

## Scope

This batch re-ran the versioned v1.0.38 source tree in Python 3.10.21 with
PyTorch 1.12.1+cu116 on the RTX 4070 Laptop GPU. The source tree was clean at
commit `926ac7031ce3645d97675ede65c4f19eb3ac846e`; no source file was changed
during execution.

The protocol was the same exploratory reconstruction configuration already
screened in v1.0.37: BCI Competition IV 2a, T-session training and E-session
testing, [1.5, 6.0) s, 22 channels, no explicit input normalization, three
branches with pooling, dropout 0.25 after pooling, no temporal-convolution BN,
ECA, Dense(32) ELU, and the inferred hidden Dense max-norm constraint. Adam
used learning rate 0.001, batch size 64, 1000 epochs, seed 42, and the held-out
session was evaluated once after the final epoch.

## Result

| Subject | Final E accuracy | Runtime (s) |
|---|---:|---:|
| A01 | 85.42% | 204.08 |
| A02 | 60.42% | 204.02 |
| A03 | 92.36% | 204.01 |
| A04 | 71.18% | 203.64 |
| A05 | 72.92% | 202.23 |
| A06 | 59.72% | 203.87 |
| A07 | 86.46% | 206.90 |
| A08 | 80.90% | 203.39 |
| A09 | 77.78% | 203.39 |

**Mean +/- sample SD: 76.35% +/- 11.36%**

Compared with the paper's 81.17% mean, the difference is **-4.82 percentage
points**. All nine result JSON files, checkpoints, and prediction files passed
the batch audit; there were no failed subjects.

## Interpretation boundary

This is a runtime/framework consistency audit, not proof of exact original
implementation. Pooling, dropout, and the hidden Dense max-norm are not fully
specified by the paper and remain reconstruction assumptions. The exact A01
agreement with the Python 3.12 run shows that the remaining gap is not
explained by moving to Python 3.10/PyTorch 1.12 alone.
