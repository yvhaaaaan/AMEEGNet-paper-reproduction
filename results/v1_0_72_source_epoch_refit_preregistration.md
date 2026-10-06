# v1.0.72 source-selected epoch refit audit

## Question

Test whether the earlier source-validation result was underestimated because
the selected checkpoint was evaluated after fitting only 230 source trials.
The fixed source-only epoch is instead used as a training-length choice, and
the model is retrained from the same seed on all 288 source-session trials.

## Fixed epochs

| Subject | Epoch |
| --- | ---: |
| A01 | 309 |
| A02 | 121 |
| A03 | 963 |
| A04 | 346 |
| A05 | 293 |
| A06 | 571 |
| A07 | 321 |
| A08 | 147 |
| A09 | 273 |

These values were recorded in the v1.0.61 source-only run before this
refit. They are not chosen from target results.

## Fixed protocol

- BCI IV 2a, Session 1 -> Session 2, `[1.5, 6]` s
- all 288 source trials used for fitting; no source validation in refit
- target Session evaluated once after the fixed epoch count
- no pooling, no dropout, no temporal BN
- fusion enabled; ECA at `sep_pre_pool`; post-activation fusion output
- Adam `0.001`, batch size `64`, seed `42`, deterministic CUDA
- max-norm projections `1.0/0.25/0.25`

## Decision

Report the nine-subject mean and sample SD descriptively. Promotion requires
the result to exceed the retained final-only candidate `78.6651%`; the target
scores will not be used to alter the fixed epoch list.
