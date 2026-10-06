# v1.0.75 full-source seed sensitivity batch

The v1.0.74 source-only screen completed for seeds `1`, `7`, and `2025`.
This batch evaluates all three fixed seeds on all nine subjects without
selecting a seed from target accuracy.

## Fixed protocol

- seeds: `1`, `7`, `2025`
- subjects: A01--A09
- all 288 source-session trials used for fitting
- target Session evaluated once after epoch 1000
- no pooling, no dropout, no temporal BN
- fusion enabled; ECA at `sep_pre_pool`; post-activation fusion output
- Adam `0.001`, batch size `64`, deterministic CUDA
- max-norm projections `1.0/0.25/0.25`

## Reporting rule

Report each seed's nine-subject mean and sample SD, plus the pooled
27-cell descriptive distribution. The seed with the highest target mean is
not promoted as the primary reproduction merely because it is highest.
