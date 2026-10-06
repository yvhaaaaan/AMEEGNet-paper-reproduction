# v1.0.61 source-validation checkpoint audit

## Purpose

This batch tests an alternative checkpoint protocol. Each subject's source
Session is split into 80% fitting data and 20% internal validation data. The
checkpoint with the highest source-validation accuracy is restored, and the
target Session is evaluated once after training. The target Session is not
used for checkpoint selection.

This is an audit protocol, not the primary paper-reproduction protocol. The
paper text does not specify an internal validation split or source-validation
checkpoint selection.

## Fixed configuration

- BCI IV 2a, subjects A01-A09, T-to-E protocol
- 22 channels, 250 Hz, `[1.5, 6] s`, 1125 samples per trial
- AMEEGNet fusion and ECA, no additional filtering or artifact removal
- Adam, learning rate `0.001`, batch size `64`, cross-entropy on logits
- 1000 epochs, seed `42`, deterministic CUDA execution
- no pooling, no dropout, no gradient clipping
- spatial/classifier/hidden max-norm projections retained from the current
  registered candidate
- source validation fraction `0.2`, target evaluated once after training

## Results

| Subject | Best source-val epoch | Source-val accuracy (%) | Final E accuracy (%) |
| --- | ---: | ---: | ---: |
| A01 | 309 | 86.21 | 77.08 |
| A02 | 121 | 70.69 | 53.82 |
| A03 | 963 | 96.55 | 88.54 |
| A04 | 346 | 82.76 | 78.13 |
| A05 | 293 | 81.03 | 72.57 |
| A06 | 571 | 86.21 | 60.76 |
| A07 | 321 | 91.38 | 79.86 |
| A08 | 147 | 91.38 | 78.13 |
| A09 | 273 | 93.10 | 74.65 |
| **Mean ± sample SD** |  |  | **73.73 ± 10.45** |

The mean runtime was `174.67 s` per subject, the median was `174.33 s`, and
the total runtime was `1572.06 s` on an NVIDIA GeForce RTX 4070 Laptop GPU.
The audit passed with no errors. Each subject has one final target evaluation;
the target was not evaluated during training.

## Comparison and decision

The result is `7.44` percentage points below the paper's reported
`81.17% ± 10.43%`, and `4.94` percentage points below the current primary
strict final-epoch candidate (`78.67% ± 9.82%`). Source-only checkpoint
selection therefore does not improve the reproduction and is not promoted to
the primary result.

The primary candidate remains the no-internal-validation, final-epoch result.
This batch is retained as a negative control showing that source-domain
validation selection alone does not explain the remaining gap.
