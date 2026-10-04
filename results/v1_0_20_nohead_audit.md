# Nine-subject standard EEGNet branch audit

All nine records completed 1000 epochs with the same v1.0.17 configuration. The held-out E session was evaluated once after training and was not used for selection.

| Subject | Final E accuracy (%) | Runtime (s) |
| --- | ---: | ---: |
| A01 | 78.47 | 210.32 |
| A02 | 57.64 | 211.38 |
| A03 | 82.29 | 212.22 |
| A04 | 62.85 | 212.00 |
| A05 | 70.83 | 211.85 |
| A06 | 60.76 | 212.41 |
| A07 | 76.04 | 207.30 |
| A08 | 73.61 | 212.00 |
| A09 | 69.79 | 212.52 |

Mean +/- sample SD: 70.25% +/- 8.38%.
Difference from paper mean 81.17%: -10.92 percentage points.
Difference from paper SD 10.43%: -2.05 percentage points.
Runtime mean: 211.33 s; median: 212.00 s.
Runtime range: 207.30--212.52 s.

This is a controlled standard EEGNet pooling/dropout hypothesis, not a confirmed complete original-paper reproduction. The paper does not fully specify pooling, dropout, padding, initialization, or epoch selection. No v1.7 enhancement was used.

## Configuration

```json
{
  "version": "1.0.20",
  "git_commit": "47a1b36205f0b8e48fa25c4df54a5b715f3fbbd4",
  "git_dirty": false,
  "strict": true,
  "paper_pooling": true,
  "dropout": 0.5,
  "dropout_after_pool": true,
  "max_norm": true,
  "head_elu": false,
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
| A01.json | dba2dac4a5bab965817b9ac853ee1579a44c2ab1cdfabcee12d61f06f2b208e6 |
| A01.npz | 97ed17ce1726dc6fcd910b6f2b31a410143ee70b28e80174b46b54b7376f341c |
| A01.pt | 3a9f0582bad69243c44b46896612e686c1ce479940a90fe380a6d06925204461 |
| A02.json | 581933fbab4069c30c1651e0b0eb4b5e201bd5e03d71cdfbcd521e40dd51ea5e |
| A02.npz | 8e7d2293bcdb4622414fbff31108083857f20a85e584dce22e618e3929ee29df |
| A02.pt | 991e6ff027adb963f526e8ce49306ee2facf62b39a9817fe62999818b6cc7c5b |
| A03.json | c6bc859c0d693ccb64f05def3da0ed8dcb5f9d557365c60516c1aa722c23c58b |
| A03.npz | 42047fd8dc8beb4fb7d00dc9b9ced6e065a8e3160bd7bcc6c87971e962e25089 |
| A03.pt | 16d3b77b756173cf88c743524bfb47a91107a6c864221b84cc41f2a43ae61adf |
| A04.json | caeb52d8c31901537c00b8085b3b86b2270b937af4e883cab3a00233e759017c |
| A04.npz | fe3caa3be5ff5200d5b957c3825d8d7d5bc1ae27a68282e27bfc8a61d0564898 |
| A04.pt | 1d74f3a3ea0c45cda9c92ab7e091f82e177392b6b89b42a70801fc2affc3a482 |
| A05.json | c23d1ae7f3cb49458b6c1c8b86a6f722dd94130db5062bf0e82d9b9fac519f4e |
| A05.npz | a6494ef8c0ea05d0ce511b6b58c1ecf87eb83c0c154d850e76844fd91894a08a |
| A05.pt | 7039e1f81a9523de74c70e08586ba11f527779ec47cae2dbaf262fe8f19dee9b |
| A06.json | 935d7a3ced58ea7ae10d4cd6621a3d7e760c1355a1b20c38d54c78594271ab66 |
| A06.npz | 3045ad34d9ef026a4bfec11e0fc2e0cf3b92037a9eefc7c394dbfc77b73238b1 |
| A06.pt | 226d736970de002f9edfaa332bba750dc5f2f7b36892bdb3127ac6206a9127ae |
| A07.json | 218ce444be1390f03667662369a117de39f0afad84e78e136d5ce7e9509eeef2 |
| A07.npz | 84ca21ad04b3eafa5d82565f768e7d81d7554f31bf84a6fa446533a70c75bd74 |
| A07.pt | 52ff02158a6b7939bfc5f55a25f27363e77717ad2e20ebe0b839959b22ba695a |
| A08.json | 65197f38797ce81c668a9280c58045294d97e6be1fc0b978468f0615896c00bd |
| A08.npz | ec615232771178f40309af9144df88957e87e70c1c052efae973590ba7c69b03 |
| A08.pt | 78e3c721b3afd39fb20bf49dc4244fb4bc4fdd52f949f8d5ed7f32aa80acc9e9 |
| A09.json | b9544e2536797126929ad8acb4fc564cc560fa67f31a215b163ed61af3e9a353 |
| A09.npz | 60e849fd40d032affae789054fd3a9147038787073c82fec0ba8b14fd4f7717b |
| A09.pt | 409374c0f80b65e191cd6af6968dba7a7a7034a844d8894f3d4a554b04a352f6 |
