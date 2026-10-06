# v1.0.70 corrected depth-ECA/pre-fusion source screen

## Question

After the v1.0.69 fix, test the registered joint interpretation in which ECA
is applied after each depthwise block and the raw depthwise output is used at
the second fusion junction before BN/ELU. The local branch still continues
with its ECA-gated normalized output.

Post-run wording clarification (v1.0.76; no decision rule changed): the raw
tensor is specifically the transferred **b2 depthwise output**. At the b3
junction it is concatenated with b3's local BN/ELU/ECA output. This is not a
both-sides-raw concatenation or a verified author implementation.

## Fixed protocol

- source-only screening: A01, A02, and A04
- no pooling, no dropout, no temporal BN
- fusion enabled; `eca_stage=depth_pre_sep`; `fusion_pre_activation=true`
- ECA bias and convolution bias disabled
- Session 1 -> Session 2, seed `42`, 80/20 stratified source validation
- Adam `0.001`, batch size `64`, 1000 epochs, deterministic CUDA
- same spatial/classifier/hidden max-norm projections: `1.0/0.25/0.25`
- target Session is not evaluated

## Decision rule

Promote to a nine-subject final-only run only if the mean final source-
validation accuracy exceeds the registered primary source-only reference of
`72.99%`. Target labels will not be used for this decision.
