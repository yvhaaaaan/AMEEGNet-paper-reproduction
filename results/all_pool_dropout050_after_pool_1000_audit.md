# Nine-subject pooled variant audit

Training code: v1.0.7 (0ce0b90). Audit release: v1.0.8.

All nine saved prediction files reproduce their final-epoch accuracies.
Configuration matches across subjects; 1000 finite training-loss records each.
Checkpoint presence and hashes verified; checkpoint inference not rerun.

| Subject | Accuracy (%) | Recorded runtime (s) |
| --- | ---: | ---: |
| A01 | 73.61 | 204.48 |
| A02 | 58.33 | 205.14 |
| A03 | 84.03 | 204.99 |
| A04 | 62.85 | 204.85 |
| A05 | 71.18 | 205.32 |
| A06 | 60.76 | 204.64 |
| A07 | 82.29 | 205.38 |
| A08 | 73.26 | 205.14 |
| A09 | 71.18 | 204.87 |

Mean +/- sample SD: 70.83% +/- 8.95%.
Difference from paper mean 81.17%: -10.34 percentage points.
Total recorded runtime: 1844.82 s.

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
  "fusion": true,
  "eca": true,
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
| A01.json | 63f0d89b42a733207112d88afc74ec306e7888512448a49e45b3b8736b69a9ab |
| A01.npz | 016926dc3169db0b6de98c94572c5e9d6c580433713a19041be752db250c3f99 |
| A01.pt | 3ef90b7e430efe51ae0d58db0ad2e5cd789001f2403db0fed404666b2813637f |
| A02.json | aaa120f8c6bfcb5e838af5c91648b6aaced72f6abe6b2018f7898acedeaaf014 |
| A02.npz | af1822282256546dd189ab4703c88ac746c020d24ab4e67a74761ced5bca33ba |
| A02.pt | 06a88e8e33b0b92826d0e26394d6f6bc35ae87e1db2114aeff8a268f402be8d1 |
| A03.json | 4d6658c1276db0ffd44e8369b2c13b0eb4e9ef6bda9a508a67cb2dc45f786598 |
| A03.npz | 92ab68b04ba77cc631001985dda2ec6e8ac99b0b12f2fa3027226340819ac981 |
| A03.pt | 423d6b071fd3214803cabf345ee85fb40c4104602c2473676ed54aa150f47590 |
| A04.json | 960e58a0084ed95ac4702a7abeaab74c922dbde4dc808b838be0f43f871518db |
| A04.npz | c71f4886f82706492ead065fb9b536edc7e094c4d54f1767d0fa2dda0ba9b8a6 |
| A04.pt | 5043ce96637172ca2c69afb591bcdbd0f3719efe66945da27a59efe51de40637 |
| A05.json | 0ec43e11f40e28434b6c2b5060c8a7dd035327a31c7c0f1b0aa8360d9448378c |
| A05.npz | b9293b09c04e50d3b1213f8ea73f03cd5fb914510aa6aa8ba30cf1c5046de80d |
| A05.pt | cf71f4e8065699137ab53e54ca27e0f1e67e3c446754a51e31b1b8015d44069c |
| A06.json | 904c397956b6a9ede294a6b5ae6f762f757f7cebad5875870e92baccd8cb2305 |
| A06.npz | 46ac7f94c37b027b5982ce3e743896ec1fb131dfe4bfa0dcd6378bc3a94c600d |
| A06.pt | d91c86f0da400e5dd419e19eabdbcc671a54e89bade0009ec9fefa18c20e4a17 |
| A07.json | 3d2f46c54d88b99c1da86756e060be5486ebae3d70a1981aa8dbd6bde8e0c915 |
| A07.npz | 031f55901f0e6fd3b3a52c916800c835aa22ad22cda1d48fd2a00e8978611d09 |
| A07.pt | f053a5cebd79093911b019926f77dc2dedd10a7df95a8d8e5ae1855957383e5b |
| A08.json | 5d3fca4ae206808c300f397e56bd057bcc11f8952eac389a4a361d78480997f5 |
| A08.npz | 5be4b48dda9f5c93437199cffeda4306a910d39dc9a2479e7cd1e9b313fd4a6e |
| A08.pt | 5ece7dca3ee3b15a51c3e2b01f4c93f937faec7354ba32342eea9df3a32ab26f |
| A09.json | 5746a632b24466268d8dad1f9cd477e63a741d1be330983d64db4ac8fda04c4c |
| A09.npz | 1a3b5cab76b305dc50afcefc7d2d91dd0f4fae4cc890229aa42cd8f613cae4cc |
| A09.pt | 52f49e89ad18fadb947f19c7df8054f13843642166d5d66df0ef3b595e394627 |
