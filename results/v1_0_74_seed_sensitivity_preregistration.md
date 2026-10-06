# v1.0.74 source-only seed sensitivity screen

## Question

The paper does not report a random seed. Measure the sensitivity of the
retained configuration to deterministic initialization and batch order using
a fixed seed set, without selecting a seed from target-session accuracy.

## Fixed protocol

- source-only subjects: A01, A02, A04
- seeds: `1`, `7`, `2025`
- Session 1 source split: stratified 80/20 validation, random state equal to
  the tested seed
- target Session is not evaluated
- no pooling, no dropout, no temporal BN
- fusion enabled; ECA at `sep_pre_pool`; post-activation fusion output
- Adam `0.001`, batch size `64`, 1000 epochs, deterministic CUDA
- max-norm projections `1.0/0.25/0.25`

## Decision rule

Report all nine source-validation cells and the per-seed means. Do not select
a seed using target labels. A follow-up full target evaluation is allowed only
as a fixed seed-sensitivity analysis if the pre-registered source screen is
completed without changing the configuration.
