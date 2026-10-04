# Nine-subject pooled variant audit

Training code: v1.0.7 (0ce0b90). Audit release: v1.0.8.

All nine saved prediction files reproduce their final-epoch accuracies.
Configuration matches across subjects; 1000 finite training-loss records each.
Checkpoint presence and hashes verified; checkpoint inference not rerun.

| Subject | Accuracy (%) | Recorded runtime (s) |
| --- | ---: | ---: |
| A01 | 75.69 | 205.56 |
| A02 | 56.60 | 205.76 |
| A03 | 82.64 | 205.28 |
| A04 | 62.50 | 205.11 |
| A05 | 68.40 | 205.58 |
| A06 | 62.50 | 205.84 |
| A07 | 70.49 | 205.00 |
| A08 | 72.92 | 204.95 |
| A09 | 70.14 | 205.55 |

Mean +/- sample SD: 69.10% +/- 7.81%.
Difference from paper mean 81.17%: -12.07 percentage points.
Total recorded runtime: 1848.63 s.

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
  "seed": 1,
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
| A01.json | 688110dc84fa9eadffc8c6e8685851d20e92160dba30ef3796d19a2cfdc888cf |
| A01.npz | 371e86a673ee49600b729f7a02b7911d0c10d228e82dcd7376ebaddf914bbf96 |
| A01.pt | 4a4d55103f1ea79eb1032238227a9da524a3d8cf517fe2416ac56b21652d5cfa |
| A02.json | 53f921e0e967452e58b24e428a23b066e0d7eff7d976ce809b06b4e274553358 |
| A02.npz | 604785ac381260ff9e028ff98f87b37e55bc7bc2707788db14f122edb3375fef |
| A02.pt | 15bb28d7626fc136dfc1e97593e81d84e679d540b7f4d45e223c3a95fdf64dd6 |
| A03.json | f3854a632013d03ec3ebb3504b7880101ec314a434416d65615df03c26ef01c7 |
| A03.npz | f2ec987b61b5873e219316af3c4fc50a9b21495ebe45d3f9e9903039da00e2eb |
| A03.pt | be6d80cbc83856c285814e65a5d4db887136beb8fd7d54cc18b9c7345a78770b |
| A04.json | 9f3a103c6147f621c3ddec81a5a7c97bc166ea112a0c04e694c61a7341951d51 |
| A04.npz | f33d84e10e66b3566887b2208c8677a57832cf57e6e056a76f78a61b10afaba6 |
| A04.pt | 1036e99a1003938756de03d51c5246a2522134d15a56c22d13bdd34e7e470d76 |
| A05.json | 54e049d538702be07c72e6aa53eb9822d5a492d64e746847d3c6935de82dac21 |
| A05.npz | c2af9da6ff5affe54e96831b6091ad0c4439eb2ee97c7991555d7ac09128e790 |
| A05.pt | 85caf6b22359dbf665fb343302c5dda8fb4785b664dc9d55cf014a7c2fcc2780 |
| A06.json | 055d0ebbce84751cfcd545396697a2430490a764fd2c2b786dc55576d1b51eab |
| A06.npz | 565426779b6eabf0a12c4d77a84d8f6d5cbc774331fe4632c02762946aebbc14 |
| A06.pt | 2b2e604e9625456755d810a2221a2d12dd4f56e73f9710ed677ec40858563632 |
| A07.json | f4ff00077460da31e7a228b654fc9ea4ec1c152d5586b200ed0db2db9e72226b |
| A07.npz | 182ef08804724713cf5749d886de701d176cd9c9f7084f5449ed2801de1ed12d |
| A07.pt | 3516396a2cee8613cb5cdc65b6fa3eb08af859b0e27c5fa2110bb0baf9f17b3e |
| A08.json | a2aa1a3390425e0892e7503cc5f1628711801d98d104e9e1af633a4138a9a066 |
| A08.npz | 62ede46198027c8499324f932e243db66d7522a99e6b176233334d46639f1d82 |
| A08.pt | 075dfeb4cc62131670b117e41fd1f8a26231af36623aee7945c2521fc886b309 |
| A09.json | f3ee62e1dc19721400e8e07381b43612b03686e63407efb3bb78df253aab2a5c |
| A09.npz | 7f70402ed9d75b64c107dd351ab292a488af26a24043ab365bbe3d0bd508a930 |
| A09.pt | 909f1935fcf13b6e8a52131234c01623600c69070f89215426408827d84faa27 |
