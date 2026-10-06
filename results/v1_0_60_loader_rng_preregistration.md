# v1.0.60 DataLoader RNG audit

## Question

The paper reports PyTorch 1.12 and batch size 64 but does not describe the
random-number source used for shuffled batches. The current reconstruction
uses an isolated `torch.Generator` seeded with 42. A plain
`DataLoader(shuffle=True)` uses the process-global PyTorch RNG, after model
initialization has already consumed random numbers. This audit isolates that
implementation choice.

## Registered comparison

| Variant | Batch-order RNG |
| --- | --- |
| isolated | dedicated generator seeded with 42 |
| global | PyTorch global generator, plain shuffled DataLoader behavior |

## Fixed source-only protocol

- A01, A02, and A04;
- complete 576-trial BCI IV 2a data, T Session only;
- seed 42, deterministic CUDA, PyTorch 1.12.1, Adam `1e-3`, batch size 64,
  1000 epochs;
- no pooling, no dropout, logits into cross-entropy;
- ECA after separable convolution, post-BN/ELU depth fusion;
- max-norm values `1.0/0.25/0.25`;
- stratified 20% T-Session validation split;
- target E Session disabled (`test-evaluation=none`);
- final source-validation accuracy is the screening metric, not a target
  selection metric.

If the source-only comparison supports a full run, the target Session will be
evaluated once at epoch 1000 for all nine subjects.
