# Source-Only Convolution Bias Audit

Registered before running either new pilot. The paper does not report
convolutional biases; `bias=False` follows the EEGNet convention whereas
`bias=True` follows the default PyTorch Conv2d constructor.

- Candidate: pooling 4/8, dropout 0.25 after pools, head dropout 0.25,
  no temporal BN, output ECA, concat-width fusion, spatial/classifier/hidden
  max-norm; Adam 0.001, batch size 64, 1000 epochs, deterministic CUDA.
- Subject: A01; seed: 42; stratified 80/20 source-Session split.
- Target-Session evaluations: zero (`--test-evaluation none`).
- Run both `conv_bias=False` and `conv_bias=True` under the same committed
  runner, including a fresh baseline to test default regression.
- Primary comparison: final source-validation accuracy, with validation
  loss as a tie-breaker. Best source-validation accuracy is descriptive only.
- Bias also consumes initialization RNG draws. This is an implementation
  audit, not a controlled claim about the causal effect of bias alone.
- Follow-up only if final source-validation improves: run both settings on
  A02 and A04 source-only splits, with the same seed and epoch budget. Do not
  choose or evaluate a new target-Session candidate based on the pilot alone.
- All prior target-inspected reconstructions remain exploratory evidence,
  even when an individual run evaluates target labels only once.

## Pilot outcome

The no-bias run ended at `93.10%` source validation accuracy, with a best
source-validation accuracy of `94.83%` at epoch 891. The convolution-bias run
ended at `75.86%`, with a best source-validation accuracy of `89.66%` at epoch
337. The bias-enabled variant was rejected before any target-Session
evaluation or multi-subject follow-up.
