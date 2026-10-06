# v1.0.65 pooling/no-dropout audit preregistration

## Question

The paper describes AMEEGNet as three EEGNet branches but does not state the
pooling or dropout details. Standard EEGNet uses temporal average pooling. This
audit keeps the standard two pooling operations while explicitly setting all
dropout rates to zero, matching the paper's omission of dropout.

## Fixed protocol

- source-only screening: A01, A02, and A04
- same BCI IV 2a data, Session 1 -> Session 2 source split, and seed `42`
- same fusion, ECA placement `sep_pre_pool`, ECA bias disabled, and no
  temporal BN
- average pooling `(1,4)` and `(1,8)` in each branch; no dropout in branches
  or classification head
- Adam `0.001`, batch size `64`, 1000 epochs, deterministic CUDA
- same spatial/classifier/hidden max-norm projections as the primary candidate
- target Session is not evaluated; only an 80/20 stratified source split is
  used for screening

## Decision rule

Promote to a nine-subject final-only run only if the mean final source-
validation accuracy exceeds the same-code primary source-only reference of
`72.99%`. No target labels will be used for this decision.
