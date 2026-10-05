# A01 implementation diagnostics audit

PASS: five 1000-epoch runs; identical seed, initialization, data and source-Session split.
230 fitting trials and 58 internal validation trials; zero target-Session evaluations.
All results below are internal validation results, not paper-level test accuracy.

| Variant | BN eps | BN momentum | Head dropout | Final val (%) | Best val (%) | Best epoch | Last 100 val mean (%) | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default_bn | 1e-05 | 0.1 | 0.5 | 67.24 | 87.93 | 792 | 72.50 | 162.04 |
| epsilon_only | 0.001 | 0.1 | 0.5 | 77.59 | 84.48 | 469 | 73.66 | 162.69 |
| momentum_only | 1e-05 | 0.01 | 0.5 | 67.24 | 86.21 | 792 | 72.57 | 164.18 |
| both_bn_changes | 0.001 | 0.01 | 0.5 | 77.59 | 84.48 | 641 | 73.03 | 160.61 |
| head_dropout_off | 1e-05 | 0.1 | 0 | 74.14 | 82.76 | 875 | 70.81 | 161.02 |

## Verified

- Clean commits match tags v1.0.26 and v1.0.27; all four BN runs share source hashes.
- The fit/validation indices are disjoint and jointly cover all 288 source trials.
- Prediction files contain validation labels/predictions and no target labels/predictions.
- Saved final weights reproduce the validation predictions and loss; repeated evaluation is identical and leaves BN state unchanged.
- All losses, gradient norms, weights and BN statistics are finite; BN variances are nonnegative.
- All 20 periodic checkpoints exist and agree with data, commit and split metadata.
- For each fixed epsilon, changing BN momentum leaves all learned parameters and training curves bitwise equal; only BN running means/variances change.

## Interpretation

At epoch 1000, the epsilon-only run differs from the default run by +10.34 percentage points on internal validation. This is not a small endpoint difference. Both momentum settings give the same endpoint accuracy at each epsilon, although their validation trajectories differ.
The head-dropout-off endpoint is 74.14%, compared with 67.24% for the default diagnostic; its best validation accuracy is lower (82.76% versus 87.93%). There is no consistent improvement across these summaries.
These are single-seed, single-subject diagnostic results, not proof of the authors' unpublished settings and not a nine-subject reproduction of 81.17%. No new nine-subject test run is justified solely by selecting the largest value here.
Existing target-Session results were inspected across successive candidates and remain exploratory even when each run evaluated the test Session only once. A future frozen-config run on those same subjects is not an untouched confirmatory test.

## Source and limitations

The paper specifies three branches, fusion positions, ECA, and a Flatten/Dense(32)/Dense(classes) head, but does not explicitly specify head dropout or BN numerical defaults. These variants therefore audit reconstruction assumptions; they are not asserted to be author code.
[AMEEGNet original article](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1540033/full)
The checkpoints save optimizer and RNG state but the runner has no resume command yet. Training history is stored in the final JSON, not continuously appended to JSONL. Neither capability is claimed by this audit.

## Validation confusion matrices

Rows are true classes; columns are predicted classes (left, right, feet, tongue).

### default_bn

```json
[[10, 4, 0, 1], [4, 9, 1, 0], [1, 2, 8, 4], [1, 0, 1, 12]]
```

### epsilon_only

```json
[[9, 3, 0, 3], [3, 11, 0, 0], [0, 0, 12, 3], [0, 0, 1, 13]]
```

### momentum_only

```json
[[10, 4, 0, 1], [4, 9, 1, 0], [1, 1, 8, 5], [1, 0, 1, 12]]
```

### both_bn_changes

```json
[[8, 2, 2, 3], [3, 11, 0, 0], [0, 0, 13, 2], [0, 0, 1, 13]]
```

### head_dropout_off

```json
[[10, 3, 0, 2], [3, 10, 1, 0], [1, 1, 10, 3], [0, 0, 1, 13]]
```

## Artifact SHA-256

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| results/bn_diag_eps1e-5_m01.json | 311592 | 0a5be0cbb80a240e9405576d60554ac6e789a1eaf020f2aea68bd3217c34fcbe |
| results/bn_diag_eps1e-5_m01.pt | 988179 | 3243ad08e21b28fadef9cab65eb4fcd13276b76c874a4aa754b8b84ed68393ab |
| results/bn_diag_eps1e-5_m01.npz | 4796 | 865ac5b8b0542b3d534d4cb84e236fd803d80cb69fa78c5fe999decae4382921 |
| bn_diag_eps1e-5_m01_checkpoints/epoch_0250.pt | 900770 | 32d7fc0a584decd0a1d48af630619036b38f7514fe27ab57ef873408974587c6 |
| bn_diag_eps1e-5_m01_checkpoints/epoch_0500.pt | 900770 | d10ff0eb05944a9a87839339bf3b34454cc393ab7a32ae069922cf87c5e570cb |
| bn_diag_eps1e-5_m01_checkpoints/epoch_0750.pt | 900770 | 183720c459e1c69ca5cd849416bdd925966c07400ae713a98eb05c01a134436b |
| bn_diag_eps1e-5_m01_checkpoints/epoch_1000.pt | 900770 | 4aa278f55243287d498394a8628b052ad90320aec4a27ec91ea389ed91dc1ad0 |
| results/bn_diag_eps1e-3_m01.json | 311783 | 2681809f7fd9d580d2cd4f0a9f0a681c1a9dc38cccf8349c181dc497b000912e |
| results/bn_diag_eps1e-3_m01.pt | 988179 | bd5ae2f98fe1db0e9318fd64953464233c73960022fb383ed6caf852deb904c2 |
| results/bn_diag_eps1e-3_m01.npz | 4796 | 00c89d2067f8d655a70ebafe0d741dde7973a97212a7d012cf8aa7635fd545b3 |
| bn_diag_eps1e-3_m01_checkpoints/epoch_0250.pt | 900770 | f2269ac4ee2a6747d045390f7f8fa5d9fca7f9cb9b61b201dfd77a4eaef2f0e7 |
| bn_diag_eps1e-3_m01_checkpoints/epoch_0500.pt | 900770 | c6146ee0035eb9fc041dd0ba687b4efc7344f833333037d375f4f69897589f05 |
| bn_diag_eps1e-3_m01_checkpoints/epoch_0750.pt | 900770 | 24e8f2a322b629e1c40f00f8b011ddb2ae613f8a23a8e66c5e583791171d52d4 |
| bn_diag_eps1e-3_m01_checkpoints/epoch_1000.pt | 900770 | 66682b4774c38080f5c58e2c904a3311a683c1a80cc579a7fd18484ce2e8c789 |
| results/bn_diag_eps1e-5_m001.json | 311578 | 0f878f20af618dc6f3113f5f8e29207a8bf192f3a63f6772b580b1233f1bc123 |
| results/bn_diag_eps1e-5_m001.pt | 988327 | 1b3ab9d6e50881c53c5643f3fdc0bd35cc9efd5d28a7a5b1298f95d59f8fe7ef |
| results/bn_diag_eps1e-5_m001.npz | 4796 | ac8308b695ca59fa1a4e8708b772f502d76c3a4f05306fcdaca2f7cd396c5903 |
| bn_diag_eps1e-5_m001_checkpoints/epoch_0250.pt | 900770 | 79304c02fd5280049743d089572f68ce0a5c29b5a8ccb6907a74193424fd616b |
| bn_diag_eps1e-5_m001_checkpoints/epoch_0500.pt | 900770 | 1becdd653503bf53cff6c7dfb92c0e98e83626fee315ae8ddd86e7d9c506286b |
| bn_diag_eps1e-5_m001_checkpoints/epoch_0750.pt | 900770 | 1a928d8a4b4192dbad92225a8f849d923127145a3a4695ae03fd74f2eb578e2b |
| bn_diag_eps1e-5_m001_checkpoints/epoch_1000.pt | 900770 | c2d0d55fb1816c90cac7b5545df752c7a64735d2ccd8bed433661ed5a772f6c8 |
| results/bn_diag_eps1e-3_m001.json | 311708 | d6cb7073cd15147ad7e9d30e27d59ca63268b121b808942dcfca85fdb7d90211 |
| results/bn_diag_eps1e-3_m001.pt | 988327 | 3761b294ec1c28ba8c06097457fc09e428767381c349f67dc20eb31bf3698df3 |
| results/bn_diag_eps1e-3_m001.npz | 4796 | c98f4b613013510cf8f899a5d916783601ef6f84783dba0afe46d6dd35943630 |
| bn_diag_eps1e-3_m001_checkpoints/epoch_0250.pt | 900770 | b957d1033eaa613b2b1f97e3e443c86c1275785e2d95802198772abc3381bdac |
| bn_diag_eps1e-3_m001_checkpoints/epoch_0500.pt | 900770 | 1a69ad610fc10762980dfd94a08d0bb82f6f8891c6d090425df20460e247f898 |
| bn_diag_eps1e-3_m001_checkpoints/epoch_0750.pt | 900770 | d10f6479739bd7f52b2d3b0f57495ea54413f5d55973caa78ffba4be6bec2923 |
| bn_diag_eps1e-3_m001_checkpoints/epoch_1000.pt | 900770 | ecf1c64b4741a6eeec03b6a3400af17aba3d3600e2cc3aa155127d9863bb993f |
| results/head_dropout_off_a01.json | 307881 | 9f47284f41fa8f6aae347c48bce7e0eead1642675544254cefe6a2f4f46458d9 |
| results/head_dropout_off_a01.pt | 988391 | 2c0cd0122d0763292f18d99de4ac2d23fb278894b6da9cdc2443186f2b4688dc |
| results/head_dropout_off_a01.npz | 4796 | 48400d4c7a36014d31ded75f84943c4b22b4900c65c9d1faf76b4af264c63498 |
| head_dropout_off_a01_checkpoints/epoch_0250.pt | 900770 | e2670c0ce6e3027434488e4f81404d6d79e96fcf9c39791a1035c21c0b218c12 |
| head_dropout_off_a01_checkpoints/epoch_0500.pt | 900770 | 328273804e1eb987a1b5d3ea8065a8a7963c098d9e0c00ce23de1aadfabfd673 |
| head_dropout_off_a01_checkpoints/epoch_0750.pt | 900770 | 80b1fc0e7bf5f4442a02b4185b1f3952eee2ff72b91e57029ce49809bd280f3d |
| head_dropout_off_a01_checkpoints/epoch_1000.pt | 900770 | 4c5e8090fc864b1054a9ac4dbf88a51363a0c9cad836da5a8ca97b6af5930730 |
