# v1.0.64 ECA bias audit

## Source-only screening

With ECA `Conv1d` bias enabled, the final source-validation results for the
pre-registered screening subjects were:

| Subject | Final source-val accuracy (%) | Best source-val accuracy (%) |
| --- | ---: | ---: |
| A01 | 82.76 | 87.93 |
| A02 | 67.24 | 68.97 |
| A04 | 77.59 | 81.03 |
| **Mean ± sample SD** | **75.86 ± 7.90** | **79.31 ± 9.60** |

The final source-validation mean exceeded the no-bias reference (`72.99%`),
so the pre-registered rule promoted this setting to the nine-subject final-only
run.

## Nine-subject final-only run

| Subject | Final E accuracy (%) |
| --- | ---: |
| A01 | 84.03 |
| A02 | 61.46 |
| A03 | 94.10 |
| A04 | 81.25 |
| A05 | 55.90 |
| A06 | 67.36 |
| A07 | 84.38 |
| A08 | 80.56 |
| A09 | 81.25 |
| **Mean ± sample SD** | **76.70 ± 12.37** |

The audit passed with no errors. The target Session was evaluated once after
1000 epochs for each subject. Runtime averaged `203.75 s` per subject, with a
median of `203.43 s` and a total of `1833.73 s` on the RTX 4070 Laptop GPU.

## Decision

ECA bias enabled is `4.47` percentage points below the paper's `81.17%` and
`1.97` points below the primary no-bias candidate (`78.67%`). It is therefore
not promoted. A05 showed unstable late training and low final accuracy, but
the recorded result is retained without selective reruns.
