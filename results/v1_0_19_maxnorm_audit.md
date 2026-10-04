# Nine-subject standard EEGNet branch audit

All nine records completed 1000 epochs with the same v1.0.17 configuration. The held-out E session was evaluated once after training and was not used for selection.

| Subject | Final E accuracy (%) | Runtime (s) |
| --- | ---: | ---: |
| A01 | 75.69 | 212.23 |
| A02 | 57.64 | 211.90 |
| A03 | 85.76 | 212.92 |
| A04 | 65.97 | 212.08 |
| A05 | 72.57 | 211.68 |
| A06 | 60.42 | 212.06 |
| A07 | 78.12 | 212.25 |
| A08 | 78.12 | 213.01 |
| A09 | 70.14 | 212.27 |

Mean +/- sample SD: 71.60% +/- 9.06%.
Difference from paper mean 81.17%: -9.57 percentage points.
Difference from paper SD 10.43%: -1.37 percentage points.
Runtime mean: 212.27 s; median: 212.23 s.
Runtime range: 211.68--213.01 s.

This is a controlled standard EEGNet pooling/dropout hypothesis, not a confirmed complete original-paper reproduction. The paper does not fully specify pooling, dropout, padding, initialization, or epoch selection. No v1.7 enhancement was used.

## Configuration

```json
{
  "version": "1.0.19",
  "git_commit": "027053753fab9e4d410a82639989c251c04069d0",
  "git_dirty": false,
  "strict": true,
  "paper_pooling": true,
  "dropout": 0.5,
  "dropout_after_pool": true,
  "max_norm": true,
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
| A01.json | 629a01fdaa4b7696878d3f919b798e563fe7620fef229d2881dbfbdbb6a682b0 |
| A01.npz | 3bf5da62b9a279442f64d365526f19524cc11b2e2083af38879b40d82af0ec46 |
| A01.pt | 35c56d0943c44526b6d43f5fb2245cf3b026ad1b383c15989036302550f56b3c |
| A02.json | 847dc7656762d412cb454fd42ffd287f2e6db34e87bf8751a00c4a6dec901e98 |
| A02.npz | 6dc5fdde6c4db9d50e0c75021cda3751574e6ab0cd7513e60ac2c846715689e0 |
| A02.pt | a8123ea9d273eebd71c766a20b6415ae969402055e19ea5ca47d1bbb318d0c1b |
| A03.json | 8dab9dc8f01bb319fc87f3de2c71e7a4a4d71d92e6546b239475bbd5aee6bf23 |
| A03.npz | 1b9db78040e1340b944c7bae4cda5e0cd2c63aac9b25b5ac8659d8c84bcf43e1 |
| A03.pt | 270b6f4f1e46a96a6f3dda8462997f55a3f36be5f12a777146a935d4d95c0294 |
| A04.json | bbffaf73d4493c3ba83a4787e9eb104296aa18a8db52d72e27a5dba21fb772fe |
| A04.npz | 98820b061d93f50cc181edbff15ce04d24b1f1ede06b77c7ae227197fc3b748c |
| A04.pt | 636d31d815ccde158e309b98889737a5d9103330b0776a323a02575f6d7ca59c |
| A05.json | cac5a0a9f9a5666735ce47acdddb26bcaa8f063165ceeba0e54f6d3dacbb51a6 |
| A05.npz | 9582f8c4c9a7f7dd01d9722b6825908582cddee8a4d502e7c18d0b883a311727 |
| A05.pt | 6d9e74ef785c8e59443e38bdef66907414ded66db5dcaec067f5a6c3e4b83af8 |
| A06.json | d6250f5e7985f198d29ea1024990c6ea84d29a8c033f131b8c3fc70cff44e6a2 |
| A06.npz | 106f53f1dd659ca4665795e0a51373360a9470053f5f78f92a49f29bcb535613 |
| A06.pt | e02ff9b4276abe4f0b0183d42fe790746de21dc542847f862a07c15b15fb308c |
| A07.json | 23b698990e6492dc060140ccfcc785960bd94550c1ef8e2384032fb4d6a3898f |
| A07.npz | 3e2e7fd86a1043c2759ade10a500154e2dcaebc1200b7e247ff6212ba51503be |
| A07.pt | 0f8359091dfa605c432b795ef99f2de3c7a79096d8517c2c93775116950c79c7 |
| A08.json | f5cd46952a8ac41bbec6509c8d1ccfd1a27c774d8e600ae9a435951a22c7eeed |
| A08.npz | aa53e41752d5974a856118590ed1a405d367f10f18f1031cbe9253a6fb4ed2f3 |
| A08.pt | 70a8748a5cbd9710bacaf7bb2bf4526bc98818c1eadee1fc8fe84c378b27e855 |
| A09.json | 0afa817687b8c9c4b33299acf04d4d99b0a2a3f7e06ccda14018eac5ac148672 |
| A09.npz | b39252256a064340219ffc22dae693d3a94531dcb531353aef257b1a84efa8b2 |
| A09.pt | 96b73b1bd15c8eda55117635a01525bbb8dd8b811049871d0cebfda8ec0279c6 |
