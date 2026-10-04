# AMEEGNet reconstruction status

## Fixed paper protocol

- BCI Competition IV 2a, T session for training and E session for evaluation
- 22 EEG channels, 250 Hz, `[1.5, 6] s`, 1125 samples, four classes
- Adam, learning rate `0.001`, batch size 64, 1000 epochs
- No filtering, standardization, alignment, augmentation, pretraining,
  test-time augmentation, or self-training
- Formal accuracy is the final epoch. Test-set maxima are diagnostic only.

The published paper reports `81.17% +/- 10.43%` within-subject accuracy.
The paper states the T/E session arrangement and segmentation, and states
that only segmentation is used as preprocessing. It does not fully specify
pooling, dropout placement/rate, initialization seed, weight constraints,
or validation/checkpoint selection.

## Runs completed

| Configuration | Scope | Mean final accuracy | Sample SD |
|---|---|---:|---:|
| Literal strict reconstruction: no pooling, no dropout | 9 subjects, historical AdamW run | 47.26% | 4.26% |
| Pooling + Dropout 0.25, dropout before pooling | 9 subjects | 65.93% | 8.40% |
| Parallel EEGNet, no fusion, no ECA, pooling + Dropout 0.25 | 9 subjects | 63.85% | 7.79% |
| Pooling + Dropout 0.5, dropout before pooling | 9 subjects | 66.82% | 9.64% |
| Pooling + Dropout 0.5, dropout after pooling, seed 42 | 9 subjects | **70.83%** | **8.95%** |
| Pooling + Dropout 0.5, dropout after pooling, seed 1 | 9 subjects | 69.10% | 7.81% |

Additional A01 checks:

- Corrected Adam strict, no pooling/dropout: 50.35% final.
- Corrected Adam, pooled/post-pool Dropout 0.5: 77.43% final.
- Four seeds with the pooled/post-pool Dropout 0.5 configuration: 74.13% +/- 2.70%.
- No head ELU: 73.96%; reverse E/T session direction: 74.31%; MaxNorm constraints: 75.35%.
- Pooled/post-pool Dropout 0.1: 64.58%.

## Interpretation

The best fixed paper-like T->E batch is 10.34 percentage points below the
published mean. The relative ablation direction is plausible: removing
fusion or ECA lowers A01 accuracy, but the absolute level does not match the
paper. This is not evidence to add the project's EA/RPA, standardization,
augmentation, cross-subject pretraining, or multi-seed ensemble; those remain
separate self-developed experiments.

The result currently supports a reproducibility boundary rather than a
successful exact reproduction. The remaining plausible causes are
under-specified author implementation details or an evaluation/checkpoint
protocol not recoverable from the article. All raw JSON, checkpoints and
per-trial predictions remain local and are covered by the batch audit files.
