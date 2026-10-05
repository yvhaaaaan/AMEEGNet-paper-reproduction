# AMEEGNet v1.0.45 audit record

Date: 2026-10-06  
Code commit: `a44085b8065d15b19a90d4edacbc32d3e5c141eb`  
Environment: Python 3.10.21, PyTorch 1.12.1+cu116, RTX 4070 Laptop GPU  
Data: BCI IV 2a, 22 EEG channels, 250 Hz, `[1.5, 6] s`, 1125 samples/trial

## Current Best

The best **formal-style final-only nine-subject batch** currently recorded is

```text
seed 2025, dropout 0.25, pooling, post-pooling dropout,
no temporal BN, fusion, ECA on branch outputs, max-norm,
hidden Dense(32) max-norm, Adam 1e-3, batch 64, 1000 epochs:
76.6975% +/- 12.4389% (sample SD), n=9
```

It is 4.4725 percentage points below the paper's reported 81.17% mean. The
seed-42 and seed-1 final-only batches are 76.3503% +/- 11.3599% and 74.0741%
 +/- 12.7530%, respectively. These three batches are retained separately; the
highest seed is not treated as a seed-independent confirmation.

The output directory is `results/all_seed2025_currentbest_v1_0_44/`. Its JSON
records contain one target-session evaluation after the 1000 training epochs,
and its batch audit reports no errors.

## Exploratory Boundary

The seed-2025 batch that evaluated the target session after every epoch and
selected each subject's maximum reached 80.5170% +/- 11.0410%. It used the
target labels 1001 times per subject and is therefore a test-leakage diagnostic,
not a formal result.

## v1.0.45 Source-Only Audits

All of the following used A01 only, a stratified 80/20 split inside the source
Session, 1000 epochs, and zero target-Session evaluations:

| Audit | Final source-val | Best source-val | Decision |
| --- | ---: | ---: | --- |
| Default output ECA, concat-width fusion | 93.10% | 94.83% | retained candidate |
| Fixed fusion channels | 86.21% | 91.38% | do not expand |
| ECA before separable convolution | 91.38% | 93.10% | do not expand |
| Softmax before CrossEntropy | 81.03% | 89.66% | reject; keep logits |

The corresponding machine-readable records are:

- `results/a01_val_fixedfusion_dropout025_hiddenmax_v1_0_44.json`
- `results/a01_val_ecadepth_dropout025_hiddenmax_v1_0_44.json`
- `results/a01_val_softmaxce_dropout025_hiddenmax_v1_0_45.json`

## Interpretation

The paper specifies Adam, learning rate 0.001, batch size 64, 1000 epochs,
the three branches, fusion, ECA, and the 81.17% nine-subject mean, but does
not fully specify every pooling, dropout, initialization, norm-constraint, or
checkpoint detail. The current 76.70% result is consequently the best recorded
controlled candidate, not proof that the unpublished author implementation has
been exactly recovered. No v1.7 enhancement, alignment, augmentation, EMA,
pretraining, or test-time adaptation is included.
