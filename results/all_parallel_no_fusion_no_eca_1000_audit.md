# Nine-subject pooled variant audit

Training code: v1.0.7 (0ce0b90). Audit release: v1.0.8.

All nine saved prediction files reproduce their final-epoch accuracies.
Configuration matches across subjects; 1000 finite training-loss records each.
Checkpoint presence and hashes verified; checkpoint inference not rerun.

| Subject | Accuracy (%) | Recorded runtime (s) |
| --- | ---: | ---: |
| A01 | 70.14 | 173.37 |
| A02 | 52.08 | 173.80 |
| A03 | 76.39 | 174.25 |
| A04 | 57.99 | 174.12 |
| A05 | 63.54 | 174.59 |
| A06 | 54.51 | 174.41 |
| A07 | 68.40 | 174.62 |
| A08 | 65.97 | 174.78 |
| A09 | 65.62 | 174.93 |

Mean +/- sample SD: 63.85% +/- 7.79%.
Difference from paper mean 81.17%: -17.32 percentage points.
Total recorded runtime: 1568.87 s.

Runtime includes training and per-epoch test evaluation; it is not pure training time.
Pooling and dropout were changed together. Their separate effects are unproven.
This exploratory reconstruction is not a verified original-paper implementation.
A01 test performance informed progression to this batch; results are not a pristine
held-out confirmatory evaluation. Do not tune further on these test results.

## Configuration

```json
{
  "strict": true,
  "paper_pooling": true,
  "fusion": false,
  "eca": false,
  "bn_first": true,
  "norm_then_activation": true,
  "seed": 42,
  "epochs": 1000,
  "device": "cuda",
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
    "train_samples": 288,
    "test_samples": 288
  }
}
```

## Local artifact SHA-256

| File | SHA-256 |
| --- | --- |
| A01.json | 70ed4b949f905d5404d3e9137bde8a45b59cabcc02794fcbb28f4e6e6d76f55a |
| A01.npz | 52307a33209fb2b646974cba7be889f3d67d377146c56e0027e2cc23c0cbe294 |
| A01.pt | fac62a399155545d2ca8e144f9360935224404a56a8621f61b93c982dc02417c |
| A02.json | afa46fe4af14ba528f88329c0d59d383b4d9858dde58f74d8fe8ac4711e1a090 |
| A02.npz | c434dcf2dd196318d78d89691c1dec1116859f23da3edb4b29593132e386f684 |
| A02.pt | 1e65c5c689a54db0e2dad62d3e3ae9e778b24b28cb2f76f9e1b2d90015421f85 |
| A03.json | a5cd520231bf03fa0ddb822bf8c948d268781de9bae17aea1d3f0add825adbbf |
| A03.npz | 9725d67feb897c54dd4cde593a23c0c8183a0bcfd74f12cee1a8082d4253f5cb |
| A03.pt | 2336a91ab741f3c0ffffcc921ce8b4f9964671edf8e525f36d1bcb61cafb4062 |
| A04.json | 219891d31da306d79c8a4d518a80e638e7c46ab0826f3bc4d3b68ede26d2af3e |
| A04.npz | db99deef6a1063ce3d4998b8ad2c7add13d26f089ae8f51287487b0576c01a45 |
| A04.pt | 26658b6213bab89fced220e25e1d77bfd2da11762130ae1cc0bc505c7804ca85 |
| A05.json | 72497684bbcb46912e0fa0c8efb7a9a25f8d2b96d751d67ae8fccc6b1f4f397e |
| A05.npz | d46ea7e9d8ba57bd5a68230156968384596297769f7afebba27ecf384c2bd814 |
| A05.pt | ce5b78c8cb0590b67f27991d69bbb68e473e305ae97e04b64f9db0ccd2bf401b |
| A06.json | 6ab33df746c1b15412772d52f766981fcef51a326a62c084b40c1e7fce53a741 |
| A06.npz | 56d70f5a4bf34d30229ef88a04b1c5408cc1fa67902d7d12391b730b106a97de |
| A06.pt | 7f65e06d4c90c3000b6542d7ce0136f5045f5cab53aabd537bd47b937b0af1d0 |
| A07.json | 697fe721e770b32f12ce93fc8852d44f99ae22d876bd345162fa4a6680abfc2e |
| A07.npz | 73a9cb7f4f6920b8156b674d59d92296f65feb259893a0816e5539a9796d9542 |
| A07.pt | 6abf0a17dfb3485344c17d9488e1eaff80c393d21631f16bcbe3490840a73a06 |
| A08.json | d60bec4963fe24c5be2d80b6e5971e5f5321bc08c1f0270f0bb213a03a9bf513 |
| A08.npz | 0e7592d68ea2a4b33239b03478a25cc499abb69ce375515d1e2c40b253b5c276 |
| A08.pt | 79573dd7457e7565337c4e2b4c508598dbe16539d572143f2c150c79e07429d8 |
| A09.json | 03cd6cb8f80385190bfcce0ec71f3fb4a4971d788606936046edec689cdbb256 |
| A09.npz | b6b21d50b913aad90ca1d3e28d38107aeceb9bee8eb0ed3f4507e4a975da4a8f |
| A09.pt | f9a8ca7b561f59091b4c122e8967630512ceeada55fab5d3ca6496a772952a33 |
