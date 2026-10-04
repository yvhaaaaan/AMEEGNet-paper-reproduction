# BatchNorm diagnostic: v1.0.26

## Purpose

This was an implementation diagnostic, not a confirmatory test-set result.
The A01 training Session was split once with seed 42 into 230 fitting trials
and 58 internal validation trials. The target Session was never evaluated
(`test_evaluation=none`). The architecture and optimizer were fixed across
all four runs:

- standard EEGNet pooling path;
- branch dropout 0.5 after pooling;
- standard EEGNet max-norm projection;
- no temporal-convolution BN;
- fusion and ECA enabled;
- Adam, learning rate 0.001, batch size 64, 1000 epochs;
- deterministic CUDA, seed 42.

## Results

| Run | BN epsilon | PyTorch BN momentum | Final validation accuracy | Best validation accuracy | Best epoch | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| default | 1e-5 | 0.10 | 67.24% | 87.93% | 792 | 162.04 |
| Keras-like | 1e-3 | 0.01 | 77.59% | 84.48% | 641 | 160.61 |
| epsilon only | 1e-3 | 0.10 | 77.59% | 84.48% | 469 | 162.69 |
| momentum only | 1e-5 | 0.01 | 67.24% | 86.21% | 792 | 164.18 |

All four runs completed with finite loss and gradient norms. Each produced
checkpoints at epochs 250, 500, 750, and 1000. All artifacts use commit
`44a3b970` and the same A01 data SHA-256 recorded in their JSON files.

## Interpretation

The epsilon-only comparison did not materially change the final validation
accuracy relative to the Keras-like pair, while changing the update weight
affected the trajectory and selected best epoch. This is evidence that BN
statistics are a relevant implementation detail, but it does not prove what
the paper's unpublished settings were and does not justify choosing a setting
from the held-out target Session. The paper-level reproduction gap therefore
remains unresolved.
