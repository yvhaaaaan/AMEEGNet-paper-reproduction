# Nine-subject standard EEGNet branch audit

All nine records completed 1000 epochs with the same v1.0.17 configuration. The held-out E session was evaluated once after training and was not used for selection.

| Subject | Final E accuracy (%) | Runtime (s) |
| --- | ---: | ---: |
| A01 | 73.61 | 169.11 |
| A02 | 59.72 | 169.76 |
| A03 | 87.15 | 169.57 |
| A04 | 65.97 | 169.90 |
| A05 | 74.31 | 169.57 |
| A06 | 61.81 | 170.08 |
| A07 | 80.56 | 169.47 |
| A08 | 76.39 | 169.95 |
| A09 | 73.96 | 170.16 |

Mean +/- sample SD: 72.61% +/- 8.81%.
Difference from paper mean 81.17%: -8.56 percentage points.
Difference from paper SD 10.43%: -1.62 percentage points.
Runtime mean: 169.73 s; median: 169.76 s.
Runtime range: 169.11--170.16 s.

This is a controlled standard EEGNet pooling/dropout hypothesis, not a confirmed complete original-paper reproduction. The paper does not fully specify pooling, dropout, padding, initialization, or epoch selection. No v1.7 enhancement was used.

## Configuration

```json
{
  "version": "1.0.22",
  "git_commit": "73b3ca505a734a4bbb27222895f00d27a609fb2f",
  "git_dirty": false,
  "strict": true,
  "paper_pooling": true,
  "dropout": 0.5,
  "dropout_after_pool": true,
  "max_norm": true,
  "head_elu": true,
  "fusion": true,
  "eca": true,
  "bn_first": false,
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
| A01.json | 56a56e5463fd0881f25c5e21044e9c553ba0bf0ad403fc8589010e2c2c1dde48 |
| A01.npz | 77bcd065b790c825c253cd14df95ae93666ef7b3471d26c7826da2b1d203fc5a |
| A01.pt | 6aaeef2ee47563b77eee10047662f73de89ccc80ae52abb1e89f9f00c40bc39e |
| A02.json | e1d6647505bd3b2e6480538034261c7a76da07ad10d1f0205b87c504305bbbd9 |
| A02.npz | 2ea518dc8edf0d155892bc9b6ba515abd683f15237c5bda306469b48745e81ba |
| A02.pt | bed28bcdbb48c77fb73190c4cda6967315e8e0758ec5f63374a057eaf3a3ac1b |
| A03.json | 866af7926a57b243f02a0ef61d2a825eef3f2f56cd3f3a59a2aee5679fbc319c |
| A03.npz | e4be6ae10ea572f00d1caf691b2f24051f6fd93ff24ddc1561910bde1b67646d |
| A03.pt | 7d597380e52b6b582d81e4b80d83da25edf8ea1e3003b99ebe686397c9970cf1 |
| A04.json | 31127356f2dc2b44f32bb2bec1ba13b44c942b6820997795b87c4b88130b3803 |
| A04.npz | 3d8505a6b127b15459d04cb06a0e967e47b597371c7c10492786928ebe3b214d |
| A04.pt | d4bbc146ca94249fee107bdfa13977343d368e9525f493fae3ae2bc59c2751dc |
| A05.json | 189e2311f224f2e7f20c18433117ec1b664e14c898a9362c1328045375e4106e |
| A05.npz | 102f0bfc1ca27f9b12dc5a5751b6b6f6733f25ec2aca93c8168cbf7398022445 |
| A05.pt | dc4b749db672cf47b1d1bd62f3036e638750da7f8b810f5627336508375f9636 |
| A06.json | 024595297492c535c5c1cbc6f9831796a54c20ed453344596ac34e8ba1a9667f |
| A06.npz | 974653f9b89480ff2df630dccc096f95f4bcab3c1f24a58b92b6b9fcf39310ba |
| A06.pt | 8a77c247a02df8912900ea39b53e04ab6846edf422ed057f9c7c773c13838ddd |
| A07.json | e4e93434782b7185cc2a183fb588a010e7a1ac3f95667bac25480b0638162a8d |
| A07.npz | 747b7484a4fea8f59af592c919517135c7d7487d70c619cc693657017bfae22f |
| A07.pt | 0a41b8ac5649d81cda000b6c350004d992f4af29e9ba6f76fe03376c7492c898 |
| A08.json | eb254a0cf17b7245226bca3ccb1610be269a001c8865e6b665492cfba6721bba |
| A08.npz | 7a863ecf58bd4ad3b875c78f6cc3852a71faba614a9472161452a506c4f25b4f |
| A08.pt | 944085e09549ba6f44031d5272bd25f0702698af559163abbbaf1192b3fd2874 |
| A09.json | e16fa4d40db2aa313a31e5e7615c6cc78f4c97b172b96494c168bc7d14038f57 |
| A09.npz | 5e663690d0059393d2f2100872c6232428ecaed07e4947fcabc4198248d55c23 |
| A09.pt | d541df76f0eee21f05253b7c5e91405ec391f0ff205fdf36b4a120cca69a7c9a |
