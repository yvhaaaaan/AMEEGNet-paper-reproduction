# v1.0.68 depth-ECA/pre-fusion audit preregistration

> Historical preregistration. The completed run is invalid for this question:
> the implementation at commit `085d0e7` overwrote the pre-activation fusion
> tensor after applying ECA, so `fusion_pre_activation=true` did not alter the
> `depth_pre_sep` data flow. The registered question is re-opened under
> v1.0.69 after the implementation fix.

## Question

The paper describes ECA after depthwise features and describes fusion in terms
of convolution outputs. Earlier audits tested these choices separately. This
screening tests their joint interpretation: ECA immediately after each
depthwise block, with the depthwise outputs used at the second fusion junction
before BN/ELU.

## Fixed protocol

- source-only screening: A01, A02, and A04
- no pooling, no dropout, no temporal BN
- fusion enabled; `eca_stage=depth_pre_sep`; `fusion_pre_activation=true`
- ECA bias and convolution bias disabled
- Session 1 -> Session 2, seed `42`, 80/20 stratified source validation
- Adam `0.001`, batch size `64`, 1000 epochs, deterministic CUDA
- same spatial/classifier/hidden max-norm projections
- target Session is not evaluated

## Decision rule

Promote to a nine-subject final-only run only if the mean final source-
validation accuracy exceeds the registered primary source-only reference of
`72.99%`. Target labels will not be used for this decision.
