# Literal classification-head diagnostic: A01

This is an A01 implementation diagnostic, not a paper-level result. It uses
the same strict configuration as the v1.0.26 BN diagnostics:

- T Session fit, 230 fitting trials and 58 source-Session validation trials;
- target Session evaluation disabled (`test_evaluation=none`);
- seed 42, deterministic CUDA, Adam at 0.001, batch size 64, 1000 epochs;
- pooling/dropout/max-norm branch hypothesis, no temporal-convolution BN,
  fusion and ECA enabled;
- classifier head `Flatten -> Dense(32) -> Dense(4)` with no Dense activation
  and no classifier-head Dropout.

The final internal validation accuracy was **65.52%**. The highest internal
validation accuracy was **81.03% at epoch 743**. The training accuracy at the
final epoch was 100%, and all losses and gradient norms were finite. The target
Session was never evaluated, so this run cannot be compared directly with the
paper's nine-subject mean.

The result does not support using the literal head as the next nine-subject
candidate: its final and best A01 validation summaries were both below the
corresponding head-ELU diagnostic. This remains an exploratory implementation
check, not evidence that the paper used the alternative head.

Source: commit `cb5e87a` before the `v1.0.29` API-only signature correction.
The original paper describes two dense layers but does not specify a hidden
activation or dropout in the text:
[AMEEGNet original article](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1540033/full).
