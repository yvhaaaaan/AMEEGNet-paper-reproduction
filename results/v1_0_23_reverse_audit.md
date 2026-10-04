# Nine-subject standard EEGNet branch audit

All nine records completed 1000 epochs with the same v1.0.17 configuration. The held-out E session was evaluated once after training and was not used for selection.

| Subject | Final E accuracy (%) | Runtime (s) |
| --- | ---: | ---: |
| A01 | 76.74 | 169.51 |
| A02 | 56.25 | 169.27 |
| A03 | 81.94 | 169.97 |
| A04 | 66.32 | 169.13 |
| A05 | 74.65 | 169.23 |
| A06 | 63.19 | 166.62 |
| A07 | 80.56 | 160.50 |
| A08 | 77.08 | 165.75 |
| A09 | 75.35 | 169.97 |

Mean +/- sample SD: 72.45% +/- 8.62%.
Difference from paper mean 81.17%: -8.72 percentage points.
Difference from paper SD 10.43%: -1.81 percentage points.
Runtime mean: 167.77 s; median: 169.23 s.
Runtime range: 160.50--169.97 s.

This is a controlled standard EEGNet pooling/dropout hypothesis, not a confirmed complete original-paper reproduction. The paper does not fully specify pooling, dropout, padding, initialization, or epoch selection. No v1.7 enhancement was used.

## Configuration

```json
{
  "version": "1.0.23",
  "git_commit": "2a5e6005744a3006dc681e5e18a5cb2c22faa2ba",
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
| A01.json | 17540c6e9ba5e34ca17e4b3139e3dbc653cbfb106fd8540a8dd2c975b830b6fc |
| A01.npz | 13e5c03d0ba5748a4f89def75f7e9c1c388baa91f1a8bd59950b545553516219 |
| A01.pt | 8dbacc873e80211c7b17310a77a7b6b184b9596c7993cbe21e0a30f95bffe242 |
| A02.json | aae91dfb5d3ca734e5e85a296eee4244de88b02e87c5d3a4007f99991bd5f7f9 |
| A02.npz | 742e90b6254b865a659fbbb8508dad3282bccfe9cc775949014381766bf952dc |
| A02.pt | 9dbaa933c29f0358636d60631c322b81b3a7a408c653b133230e65eca339e7b9 |
| A03.json | 01fb3e8fd92fccafbe864d3b6f40686ccd5d25d7b20b8bae24bfc9a8915d69a3 |
| A03.npz | cf1cdeaf8c3f65973add31d4a3ee8291fa37d64bf8120c13612c8a92af401eb3 |
| A03.pt | 5f61c50fe157cfa812f4543ca12d2cb7c52c241ec00f128445e42ecdb6668135 |
| A04.json | d001b1cfaf92f67a501796616786017cea387f5d58955816b825c59322f9efaa |
| A04.npz | 80a938e21994fd313d05e4d36a056f7c76d3b4a85a9c07d2eb1aeb6e09701256 |
| A04.pt | 5119ae0c5f216694fe96c9e7ed46af3a925b6ee8ba5e7f0455a317114e8eaafe |
| A05.json | 553da40177c31277d7299ab818837d97f4a751da0f5350cb36937261dde9405f |
| A05.npz | b22f627311c49908b6c375827d71b25dcadfd3317a6577aabd4e379b0d8f19b1 |
| A05.pt | 29e917a8da117cf24345b27108faa1155a0a42335943a479e917b84caebc2c5c |
| A06.json | 43af1b12bf430b5cb33fb54706aa0e199e91c6266b8225829d607011e2dc378b |
| A06.npz | 2128004abb562523faff42a2f71af90244eb0339ec2578930b052bf32f4cabe0 |
| A06.pt | f92b8af3412f45ed64e793bcd931f55b2a67e585fdcb2e428df9b7bc679f8222 |
| A07.json | f716f8d3cc132f66b4d6c4b860a69b617b48cdb9c22045abad17559cc8ead9be |
| A07.npz | 0e13ff142f1c1a16283046ee4b2b7ce65f44b680757e7e0dc02cfeca9514be26 |
| A07.pt | 3665ab28bc8d6e4413f03208d9b8a9a2f75bd67921e18a3e76f0e5204be987ac |
| A08.json | ca10ff91764a7516f58f2dca1dea18a452a642d1d7cd59979495830cbd9f868a |
| A08.npz | be027078a62d0af70184c7ede6907387653bc05ac2a599b3fe3557aa9c350c58 |
| A08.pt | 56e2784a19cf1afe47b8b5fe220f6c893cc42707ddc20de6efe10c1e40e9fc87 |
| A09.json | c36e5223c5ae015d4d25f975d7d23ad9f1d0cba42c1b19ba9eb26bd0629be5c7 |
| A09.npz | a9f0ba62375e4e3a4651a4ccfeacd3c2eb36da0e8cbf9a7c7841cc0186841c55 |
| A09.pt | 93c9ae86f697d8c86ac095c53c29a2d93f7adae418a9f0165de4aa672afc2ce9 |
