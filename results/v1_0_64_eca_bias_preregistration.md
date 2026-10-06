# v1.0.64 ECA bias audit preregistration

## Question

The AMEEGNet paper's ECA equation includes a bias term for the one-dimensional
convolution. The current primary candidate uses `bias=False` for ECA. This
audit tests the paper-consistent `bias=True` setting.

## Fixed protocol

- subjects A01, A02, and A04 only for source-domain screening
- same BCI IV 2a data, segmentation, Session 1 -> Session 2 direction, and
  seed `42` as the primary candidate
- same no-pooling/no-dropout graph, fusion, ECA stage `sep_pre_pool`, and
  convolution bias disabled elsewhere
- same Adam `0.001`, batch size `64`, 1000 epochs, deterministic CUDA
- same spatial/classifier/hidden max-norm projections
- target Session is not evaluated; a stratified 80/20 source split is used
  only to measure source validation accuracy

## Decision rule

The three-subject mean source-validation accuracy will be compared with the
same-code primary source-only reference (`72.99%` across A01/A02/A04). The
bias-enabled setting will be promoted to a nine-subject final-only run only if
it improves that mean. The target Session will not be used for this decision.
