# A01 no temporal-BN diagnostic

- Version: `v1.0.2`
- Same as `s01_adam_no_clip_1000`, except the BN immediately after each
  temporal convolution is removed.
- Final test accuracy: **50.35%**
- Highest test accuracy during training: **51.04%**, epoch 599
- Runtime: 164.21 s

This matched the retained-temporal-BN run exactly on the reported accuracy;
it does not explain the gap to the paper result.
