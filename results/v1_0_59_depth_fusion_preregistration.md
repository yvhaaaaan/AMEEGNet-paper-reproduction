# v1.0.59 depth-fusion source audit

## Question

The paper says that the depthwise outputs of branches 2 and 3 are fused, but
does not specify whether the fused branch-2 feature is taken before or after
BatchNorm and ELU. The implementation audit found that the existing
`fusion_pre_activation` switch was ineffective in the `sep_pre_pool` path.
This version fixes only that switch and adds a shape/forward test.

## Registered comparison

| Variant | Branch-2 depth feature sent to branch-3 fusion |
| --- | --- |
| post | after BatchNorm and ELU |
| pre | before BatchNorm and ELU |

## Fixed source-only protocol

- A01, A02, and A04;
- complete 576-trial BCI IV 2a data, T Session only;
- seed 42, deterministic CUDA, Adam `1e-3`, batch size 64, 1000 epochs;
- no pooling, no dropout, logits into cross-entropy;
- ECA after separable convolution, fusion enabled;
- max-norm values `1.0/0.25/0.25`;
- stratified 20% T-Session validation split;
- target E Session disabled (`test-evaluation=none`);
- final source-validation accuracy is descriptive only; no target-based
  selection or epoch selection.

The two source-only runs are used to determine whether the paper-supported
fusion ambiguity is worth a full nine-subject final-only run. No target result
will be inspected before this source-only comparison is complete.
