# v1.0.63 reverse-session audit

## Results

This batch keeps the primary candidate fixed and exchanges only the two
BCI IV 2a Session roles: Session 2 is used for fitting and Session 1 is used
for final evaluation. The target Session is evaluated once after epoch 1000.
The audit passed with no errors.

| Subject | Final accuracy (%) |
| --- | ---: |
| A01 | 84.72 |
| A02 | 61.81 |
| A03 | 88.89 |
| A04 | 76.39 |
| A05 | 79.86 |
| A06 | 66.32 |
| A07 | 59.72 |
| A08 | 82.99 |
| A09 | 75.00 |
| **Mean ± sample SD** | **75.08 ± 10.37** |

The mean runtime was `200.97 s` per subject, the median was `200.41 s`, and
the total runtime was `1808.74 s` on an NVIDIA GeForce RTX 4070 Laptop GPU.
All nine subjects used the same code, seed, data shape, optimizer, 1000-epoch
schedule, and final-only target evaluation. Prediction files were retained.

## Comparison

The reverse direction is `6.09` percentage points below the paper's reported
`81.17%`, and `3.59` percentage points below the primary Session 1 -> Session
2 result (`78.67%`). Therefore the unspecified Session direction is not a
plausible explanation for the remaining reproduction gap, and the reverse
direction does not replace the primary result.

The final reported primary result remains Session 1 -> Session 2, with this
reverse direction retained as a reproducible protocol audit.
