# v1.0.66 convolution bias audit

## Nine-subject final-only result

The audit enabled bias parameters in the temporal, depthwise, and separable
convolutions while keeping ECA bias disabled. All other primary-candidate
settings were unchanged. The target Session was evaluated once after epoch
1000. The batch audit passed with no structural or bookkeeping errors.

| Subject | Final E accuracy (%) |
| --- | ---: |
| A01 | 81.94 |
| A02 | 62.85 |
| A03 | 92.36 |
| A04 | 78.13 |
| A05 | 75.00 |
| A06 | 25.00 |
| A07 | 85.76 |
| A08 | 80.90 |
| A09 | 84.72 |
| **Mean ± sample SD** | **74.07 ± 20.14** |

Runtime averaged `223.44 s` per subject, with a median of `223.20 s` and a
total of `2010.93 s` on the RTX 4070 Laptop GPU.

## Decision

The bias-enabled result is `7.10` percentage points below the paper's
`81.17%` and `4.59` points below the primary candidate (`78.67%`). A06's final
prediction is at chance level after late training instability; it is retained
as recorded and was not selectively rerun. Convolution bias is therefore not
promoted.
