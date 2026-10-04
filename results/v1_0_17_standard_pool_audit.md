# Nine-subject standard EEGNet branch audit

All nine records completed 1000 epochs with the same v1.0.17 configuration. The held-out E session was evaluated once after training and was not used for selection.

| Subject | Final E accuracy (%) | Runtime (s) |
| --- | ---: | ---: |
| A01 | 70.83 | 210.72 |
| A02 | 54.51 | 209.81 |
| A03 | 82.29 | 209.55 |
| A04 | 63.19 | 209.99 |
| A05 | 72.22 | 209.57 |
| A06 | 57.64 | 209.79 |
| A07 | 71.18 | 209.57 |
| A08 | 73.61 | 209.40 |
| A09 | 69.10 | 210.06 |

Mean +/- sample SD: 68.29% +/- 8.55%.
Difference from paper mean 81.17%: -12.88 percentage points.
Difference from paper SD 10.43%: -1.88 percentage points.
Runtime mean: 209.83 s; median: 209.79 s.
Runtime range: 209.40--210.72 s.

This is a controlled standard EEGNet pooling/dropout hypothesis, not a confirmed complete original-paper reproduction. The paper does not fully specify pooling, dropout, padding, initialization, or epoch selection. No v1.7 enhancement was used.

## Configuration

```json
{
  "version": "1.0.17",
  "git_commit": "ab1e40a55096541911fc936b7adcfa5f017f1355",
  "git_dirty": false,
  "strict": true,
  "paper_pooling": true,
  "dropout": 0.5,
  "dropout_after_pool": true,
  "max_norm": false,
  "head_elu": true,
  "fusion": true,
  "eca": true,
  "bn_first": true,
  "norm_then_activation": true,
  "seed": 42,
  "epochs": 1000,
  "device": "cuda",
  "deterministic": true,
  "test_evaluation": "final",
  "training_protocol": {
    "optimizer": "Adam",
    "lr": 0.001,
    "weight_decay": 0.0,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "gradient_clip_max_norm": null,
    "batch_size": 64,
    "fit_samples": 288,
    "validation_samples": null,
    "test_samples": 288,
    "label_smoothing": 0.0
  }
}
```

## Local artifact SHA-256

| File | SHA-256 |
| --- | --- |
| A01.json | 9c407148ae4c7511df10fe5592479ad97316c4c580e54abb4fa65d48dea3eb7e |
| A01.npz | ffc4e8607524d12b623eaefe9072f912066d37b01729fe8367e9b0018ea638f2 |
| A01.pt | 3968d6fd9d71744b1a09a830586a10f7c139f95fbc107e6395f9643e946421ee |
| A02.json | 5f186bbcccf7e135964c35fb1d768cf7e267e79ee5ef87bc63de0a73a96f19b9 |
| A02.npz | 31bf330f6751a3f5a83ecde5e82f70d375192c1c440f39d4f926cbde955f0ed7 |
| A02.pt | 2c7e4dd91a0832c5e41662fb6f9543c61b623bd4473a0107da4c9a695c4f1f8a |
| A03.json | 76535a612162a95573203f6012f6f58ea9a2c30d289cc4835c2ea82f2da6f49e |
| A03.npz | fa50e0f46101cafda725908d6ddd95d0a02b0671a5acda2bca457848eea47142 |
| A03.pt | 876047ce4c0b0802dc9fcdb83551406781eb1e7a146bfb45c22098e291cbd315 |
| A04.json | ed2757d960a3bf0cd3d14371351bc95199f15829cdf2308351041039287d3fdc |
| A04.npz | 85ef7b8bd9d21b614909c9fa5ed1b33a561243bd0b7ec91316e06fa415b70fd5 |
| A04.pt | f41a07de2e64d4cc54d209bd68858141e1deca58a97980ec8d88730f6aae2a60 |
| A05.json | ad1fb5ee9474196972f456a6fc7b5e471aa7fa53d5b09244e1a288c5254d0419 |
| A05.npz | 447fce8b680bd7998d832efa63b7e34e9a2b7a033874d2ca0e57f6864eb02a6f |
| A05.pt | 93eeaa66c0aaddcaffa2928f38e0099fff4b01442d2d552da456554c88281166 |
| A06.json | 9cce7601dc1f2f6ef009885759ebfa1a9875b33910c69a1f13b93b46df3d447f |
| A06.npz | d8511ba350bbc325fc0bdce7e87ff45794f498281da976f1b553e4f6a3a27f22 |
| A06.pt | 1f4f8d3cbc000426a7ca56d27f491528859ed3106253e496796ce6c61cd3fee1 |
| A07.json | 5de890afdd8e4bf2923460adbedeefb9cd796034cf6d45f521c959122068de18 |
| A07.npz | 8c406893d0a8e541ed6635da9b81965316ea93292ca87b604040e3a9632d9073 |
| A07.pt | e68fd2131a777c1b5bb6231ee135617e47327f49099f67f74b711bf9867f0e6f |
| A08.json | 2e4ad198f02dfdc055be9a542304a0fe7d859eff3e58c681a19d2d35f7799d80 |
| A08.npz | 30661ce720d20564a93176cb5f607f8187c5093d2f543445e11b0f5384f4e8e2 |
| A08.pt | b1502ad23154298cfdd3eb7f4bb461d3695e31045059942a49810ae180cf8e2a |
| A09.json | 5c830ef115bbd39ee91d1410005d5a5d185c44de488ca61e8f77d7fb3f73a74e |
| A09.npz | 3ed2d644812b629bbc79e13d8fa828e81ae1b31ffcbd73f9c6a929bc76faf339 |
| A09.pt | 4b172ea0e903f1bbbbf36ca3a41d6d122e98d7b80ddbe28baa10457eec333f27 |
