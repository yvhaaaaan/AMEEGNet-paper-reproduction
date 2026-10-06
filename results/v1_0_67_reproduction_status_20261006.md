# AMEEGNet reproduction status at v1.0.67

## Primary result

The best controlled nine-subject final-only candidate currently retained is:

- BCI IV 2a, Session 1 -> Session 2
- `[1.5, 6] s`, 22 channels, 1125 samples
- fusion enabled, ECA at `sep_pre_pool`, no temporal BN
- no pooling or dropout
- Adam `0.001`, batch size `64`, 1000 epochs, seed `42`
- target Session evaluated once after the final epoch
- spatial/classifier/hidden max-norm projections: `1.0/0.25/0.25`

| Subject | Final E accuracy (%) |
| --- | ---: |
| A01 | 82.99 |
| A02 | 62.15 |
| A03 | 94.10 |
| A04 | 80.21 |
| A05 | 75.69 |
| A06 | 65.63 |
| A07 | 85.76 |
| A08 | 81.60 |
| A09 | 79.86 |
| **Mean ± sample SD** | **78.67 ± 9.82** |

Against the paper's `81.17% ± 10.43%`, the mean difference is `-2.50` percentage
points. The primary result is reproducible and target-test leakage free, but
the max-norm assumptions are not fully specified in the paper, so it is a
controlled reconstruction candidate rather than proof of exact source-code
recovery.

## Completed audits

| Audit | Result | Decision |
| --- | ---: | --- |
| Source-validation checkpoint selection | 73.73 ± 10.45 | rejected |
| Reverse Session direction | 75.08 ± 10.37 | rejected |
| ECA bias enabled | 76.70 ± 12.37 | rejected |
| Convolution bias enabled | 74.07 ± 20.14 | rejected |
| Standard pooling, no dropout | 69.54 ± 13.94 source screen | rejected |

The previously completed audits also found no improvement from alternate ECA
placement, fusion activation source, Softmax-before-cross-entropy, global
DataLoader RNG, or the tested normalization/scale hypotheses. Test-set
best-epoch selection reaches a value close to the paper in a diagnostic run,
but it evaluates target labels throughout training and is not a valid formal
reproduction result.

## Conclusion

The public article specifies the data segmentation, branch sizes, fusion,
ECA, optimizer, learning rate, batch size, and epoch count, but leaves several
implementation details unspecified. The current work has therefore reached a
defensible controlled reconstruction, not an exact reproduction at the
reported mean. No v1.7/self-designed enhancement, alignment, augmentation,
EMA, pretraining, or test-time adaptation is included.
