# v1.0.61 checkpoint-selection audit

## Question

The paper reports 1000 epochs but does not state whether the reported model
is the epoch-1000 state or a checkpoint selected using a training-only
validation set. This audit separates that protocol ambiguity from the target
test set.

## Registered comparison

| Variant | Checkpoint rule |
| --- | --- |
| final | use epoch 1000 state |
| source-best | restore the highest accuracy on a stratified 20% split of the T Session |

The source-best variant is not the paper's confirmed protocol; it is an
audit of a plausible undocumented checkpoint rule. The target E Session is
never used to select an epoch.

## Fixed source-only screening protocol

- A01, A02, and A04;
- complete 576-trial BCI IV 2a data, T Session only;
- seed 42, deterministic CUDA, PyTorch 1.12.1, Adam `1e-3`, batch size 64,
  1000 epochs;
- no pooling, no dropout, logits into cross-entropy;
- ECA after separable convolution, post-BN/ELU depth fusion;
- max-norm values `1.0/0.25/0.25`;
- stratified 20% T-Session validation split;
- target E Session disabled (`test-evaluation=none`).

If source-best improves the pre-registered source-only comparison, a full
nine-subject run will report its target accuracy once, without using target
labels for checkpoint selection.
