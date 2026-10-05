# Hidden Dense max-norm reconstruction batch

Training source: v1.0.36, commit a561fe45fb7855c302f50970613b1616cdd1e10e.
Audit source: v1.0.37, audit_hidden_maxnorm_batch.py.

All nine subjects finished 1000 epochs. Each run used 288 source-session
training trials and 288 target-session test trials, with 72 trials per class.
Target-session evaluation occurred once after training, and the final epoch
was reported. Saved prediction arrays reproduce all reported accuracies.
Run configurations and source hashes match. The audit returned zero errors.
Checkpoint files were present; this audit did not repeat checkpoint inference.

| Subject | Final accuracy (%) | Recorded elapsed seconds |
| --- | ---: | ---: |
| A01 | 82.99 | 182.33 |
| A02 | 52.78 | 182.19 |
| A03 | 92.71 | 182.63 |
| A04 | 66.32 | 183.61 |
| A05 | 72.92 | 182.71 |
| A06 | 61.46 | 183.60 |
| A07 | 84.38 | 182.10 |
| A08 | 83.68 | 182.78 |
| A09 | 80.21 | 183.32 |

- Mean accuracy: 75.27006173%.
- Between-subject sample SD (ddof=1): 12.86454559%.
- Difference from published mean 81.17%: -5.89993827 percentage points.
- Recorded elapsed-time mean: 182.80921476 s.
- Recorded elapsed-time median: 182.71283920 s.
- Total recorded elapsed time: 1645.28293280 s.

Elapsed time includes training and the final evaluation, not pure training time.
The paper's standard-deviation convention has not been independently established.

## Reconstruction boundary

The explicit paper settings are four classes, 22 channels, 250 Hz, segmentation
[1.5,6) seconds, three branches with F1=(4,8,16), kernels=(16,32,64), D=2,
fusion transmission, ECA k=3, two Dense layers, Adam lr=0.001, batch size 64,
and 1000 epochs. Source: https://doi.org/10.3389/fnbot.2025.1540033.

Average pooling (4,8), dropout=0.5, head ELU/dropout, concatenation details,
and max-norm constraints are inferred, not explicitly specified by that article.
In particular, constraining the added Dense(32) layer is a separate unverified
reconstruction assumption; it is not established as an original-paper setting.
The run used Python 3.12.9 and torch 2.12.0+cu126, unlike the article's
Python 3.10 and PyTorch 1.12. Higher GPU capability does not guarantee higher
accuracy. This result is still below the published mean, not a successful exact
reproduction. The same target sessions have been inspected in prior diagnostic
experiments, so this is not an untouched confirmatory test set.

## Next locked diagnostic

A01 source-session internal validation with dropout=0.25 (otherwise unchanged)
finished 1000 epochs in results/a01_val_dropout025_hiddenmax_v1_0_37.json.
The final internal-validation accuracy is 93.10344828%, with zero target-session
evaluations. The best internal-validation value, 94.82758621%, is not the formal
final-epoch metric. This training-session evidence supports running the same
fixed configuration on all nine subjects. No target labels are used to choose
epochs, subject-specific settings, or seeds. All nine final-epoch results must
be retained, including low accuracies.
