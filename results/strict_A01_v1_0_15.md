# Clean strict A01 run at v1.0.15

This is a paper-text reconstruction run, not a result selected by the target
session. It used T-session training and E-session evaluation, seed 42, Adam
with learning rate 0.001, batch size 64, 1000 epochs, no filtering or
standardization, no pooling, no dropout, and no gradient clipping. CUDA
deterministic algorithms were enabled.

| Metric | Value |
| --- | ---: |
| Final E-session accuracy | 50.35% |
| Final T-session training accuracy | 100.00% |
| Runtime | 210.08 s |
| E-session evaluations | 1 |
| Git commit | `0cb2f3069955428fd1f0026667a9951ce78c367c` |
| Data SHA-256 | `a87452ce68b0b86b2b45431f9200d26f31fa8106e36718d5e003dd285cf5abdf` |

The result is substantially below the paper's reported BCI IV 2a mean of
81.17%. It is therefore evidence that this specific under-specified
reconstruction is inadequate, not evidence that the published method is
incorrect. The paper does not fully specify pooling, dropout, padding,
initialization, or epoch-selection details; the next controlled audit tests
the standard EEGNet pooling/dropout path without adding the later v1.7 method.
