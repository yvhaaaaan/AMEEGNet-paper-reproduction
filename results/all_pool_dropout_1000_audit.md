# Nine-subject pooled variant audit

Training code: v1.0.3 (bb3d254). Audit release: v1.0.4.

All nine saved prediction files reproduce their final-epoch accuracies.
Configuration matches across subjects; 1000 finite training-loss records each.
Checkpoint presence and hashes verified; checkpoint inference not rerun.

| Subject | Accuracy (%) | Recorded runtime (s) |
| --- | ---: | ---: |
| A01 | 72.57 | 205.27 |
| A02 | 52.08 | 204.44 |
| A03 | 77.78 | 204.66 |
| A04 | 60.76 | 204.91 |
| A05 | 68.06 | 204.82 |
| A06 | 54.86 | 202.32 |
| A07 | 69.79 | 197.05 |
| A08 | 70.14 | 196.64 |
| A09 | 67.36 | 196.20 |

Mean +/- sample SD: 65.93% +/- 8.40%.
Difference from paper mean 81.17%: -15.24 percentage points.
Total recorded runtime: 1816.29 s.

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
| A01.json | 0965ddce5a5666d8eab875bc5d3040a0cb8660a98c5fe0755ac74c244c1e7262 |
| A01.npz | eaa40e47489bc6a7b2c91688f4dbce8c117df9a6c85d60e5bcf61651738a3f6b |
| A01.pt | aebd8ba5ff9278f31d6bd414ead4ff5664b52f736bcb2595b27612d9f94c3c4e |
| A02.json | 4dd4049ed8e982be105de2861e0eea3c49d59a00d8bd9cf64997b1b9291b2e47 |
| A02.npz | 64c7f74f041323a16cce9e027aee9ed0ef7a49ae05071dd1b655a7a1640d2393 |
| A02.pt | e49ed0b8d26f13af765412f07fa0d40d93ae8a5a9a871e79980355f4b98580cd |
| A03.json | d2736e8194120de1bcb174212e44e430d29cc166b99ef8e843e3fc8853c7717f |
| A03.npz | 9ecc52d5cc33a8b4ae68b6fa81b20772a5778e99371da2b2dec6b5d1eb349a3a |
| A03.pt | d0915805df41e9b2f187c67594df1c56e0aadde76addea7322281a564fabbd7a |
| A04.json | e009b1c85f308717db0ce9ee6ffd5f5b9bb6b8f3087864acfb33d09a622f075a |
| A04.npz | 8f6753d8e9145bc2569a53ed6dc873e20385c2593d1af06d983e255e3c62bf40 |
| A04.pt | 80bdb60b6fbd0de6cc7f98cdf82e0c51268279506dd006904898e1b4dd44ff8e |
| A05.json | 24a6eba10e585b0273a523920a42e6c1bd5e31bdb57fb8c0384a606ac1d9217f |
| A05.npz | ec626fc78cdbfea1061be387d3ae72495646921d5c195c7e6e7de0d6cb44b7ec |
| A05.pt | 979b4e070bab07fe658cefc334f6997da3a1ae2c6f2d3783d166540daec10b6c |
| A06.json | 0f1c1c5146af3344ec6a9ab39d3e0f64679e8c72839b6dd252a132cbde4c99f2 |
| A06.npz | a408b5e9eb2c183bed96dc0420d432cebe95bda0a3a419e26c7255c279d9cc2a |
| A06.pt | e67af22a18341dd22514c0c2e10e0875b49c6522e83c0adb888f966e2a7e9537 |
| A07.json | 5fa1befb897eb8794c0e590cc175490366b830fe6f03f8032fc7f8dd466d4f81 |
| A07.npz | 7fcd0d2c091752158538b5b060817dab5c608d1d5f6359a39b03161252a59fb6 |
| A07.pt | 997f2977ea45ad34f753ef05d439c4ae17cd32e70c9bcf447d8f8abae264d753 |
| A08.json | d804e9b8bbe785f1577caaa727d7453102eae4df5684efd430450e8456bba2da |
| A08.npz | 120f6e4c0eb36f25cf830ad05910a67c57267bddef2b1f21af0129a9b81a7b38 |
| A08.pt | 2ff243d4018ff318772cd4c0a582cd2f6fba96b85ecf05399a7684d00538726d |
| A09.json | dc450f50abc8cc0aa85490cd000f29de74e9d00975fa4cdbb302de2e4faed64b |
| A09.npz | 25f7f08c4702d0d91ba4f3c8cc1677d26e881569404cae751166a489156389c6 |
| A09.pt | e73c93aec3678fe45507b8331cbb2cdcf2495bc67d7b18f7a8ed9a19ea98aa47 |
