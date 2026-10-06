# v1.0.63 reverse-session protocol audit preregistration

## Question

The AMEEGNet paper states that one BCI IV 2a session is used for training and
the other for evaluation, but does not explicitly state whether Session 1 or
Session 2 is the training session. The primary run uses Session 1 -> Session
2. This audit evaluates the reverse direction, Session 2 -> Session 1.

## Fixed protocol

- BCI IV 2a subjects A01-A09
- same data files, labels, segmentation, and class ordering as the primary run
- same AMEEGNet graph and source code commit as the primary candidate
- seed `42`, isolated DataLoader generator, deterministic CUDA execution
- Adam, learning rate `0.001`, batch size `64`, cross-entropy on logits
- 1000 epochs, final epoch checkpoint only
- no internal validation and no target evaluation during training
- no pooling, no dropout, no temporal BN
- same registered max-norm projections: spatial `1.0`, classifier `0.25`,
  hidden `0.25`
- same ECA placement `sep_pre_pool`

Only the train/evaluation Session direction changes. The reverse run is a
protocol audit and will not replace the primary result unless its direction is
explicitly supported by the paper and it is reported as such.

## Planned output

The batch will be stored under `results/all_reverse_sessions_v1_0_63/` with
per-subject JSON/NPZ/PT artifacts, a summary, and an independent audit output.
