# A01 standard EEGNet pooling/dropout variant

- Version: `v1.0.3`
- Seed: 42; Adam `lr=0.001`; no weight decay; no gradient clipping
- 1000 epochs; batch size 64; CUDA
- Standard EEGNet average pooling `(1,4)` and `(1,8)` retained
- Dropout: 0.25; no standardization or augmentation
- Final test accuracy: **72.57%**
- Highest test accuracy: **73.61%**, epoch 933
- Runtime: 205.07 s

This is the first A01 reconstruction variant close to the paper's reported
scale. It remains below 81.17% on A01 and must be evaluated across all nine
subjects before any conclusion.
