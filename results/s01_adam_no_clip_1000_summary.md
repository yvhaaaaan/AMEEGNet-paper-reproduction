# A01 corrected strict run

- Version: `v1.0.1`
- Data: BCI Competition IV 2a, A01, official T session for training and E session for testing
- Input: 22 channels, 250 Hz, `[1.5, 6] s`, 1125 samples
- Seed: 42
- Epochs: 1000
- Device: CUDA
- Optimizer: Adam, `lr=0.001`, `weight_decay=0`, no gradient clipping
- Batch size: 64
- Final test accuracy: **50.35%** (`145/288`)
- Highest test accuracy during training: **51.04%**, epoch 599
- Final training accuracy: 100%
- Runtime: 211.11 s

This is a corrected-protocol A01 reconstruction run, not evidence that the
paper implementation has been fully recovered. The test-set maximum is kept
for diagnosis only and is not used as the formal reported result.

Raw files retained locally:

- `results/s01_adam_no_clip_1000.json`
- `results/s01_adam_no_clip_1000.pt`
- `results/s01_adam_no_clip_1000.npz`
