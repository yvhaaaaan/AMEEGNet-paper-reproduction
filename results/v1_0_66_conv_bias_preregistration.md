# v1.0.66 convolution bias audit preregistration

## Question

The paper specifies PyTorch but does not state whether the temporal,
depthwise, and separable convolutions use bias parameters. The primary
candidate disables these biases, as in common EEGNet implementations. This
audit tests PyTorch-default convolution bias enabled.

## Fixed protocol

- source-only screening: A01, A02, and A04
- same no-pooling/no-dropout architecture, fusion, ECA `sep_pre_pool`, and
  ECA bias disabled
- same Session 1 -> Session 2 direction, seed `42`, 80/20 stratified source
  validation split, and no target evaluation
- same Adam `0.001`, batch size `64`, 1000 epochs, deterministic CUDA
- same spatial/classifier/hidden max-norm projections

## Decision rule

Promote to a nine-subject final-only run only if the mean final source-
validation accuracy exceeds the registered primary source-only reference of
`72.99%`. No target labels will be used for this decision.
