# v1.0.58 Softmax/Cross-Entropy protocol audit

## Question

The AMEEGNet paper describes a final Softmax layer and cross-entropy loss, but
does not state whether the implementation passed probabilities into the loss
or passed logits into PyTorch `CrossEntropyLoss`. This audit isolates that
ambiguity without changing the network or data protocol.

## Fixed protocol

- BCI IV 2a complete-trial data, T Session for fitting and E Session held out;
- subjects A01, A02, and A04;
- seed 42, deterministic CUDA, PyTorch 1.12.1, Adam `1e-3`, batch size 64,
  1000 epochs;
- no pooling, no dropout, ECA after separable convolution and before the
  branch output; fusion enabled;
- spatial/classifier/hidden max-norm values `1.0/0.25/0.25`;
- a stratified 20% split of T Session is used only for source validation;
- final source-validation accuracy at epoch 1000 is the screening metric;
- target E Session is not evaluated (`test-evaluation=none`), and no setting
  is selected using target labels.

## Registered comparison

| Variant | Loss input | Decision rule |
| --- | --- | --- |
| A | logits | compare final source-validation accuracy |
| B | Softmax probabilities | compare final source-validation accuracy |

The higher source-validation result is only a screening result. A full
nine-subject target-session run may be considered only after this source-only
comparison, and must keep its target evaluation at the final epoch.
