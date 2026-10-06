# v1.0.75 complete seed sensitivity audit

## Fixed Protocol

All 27 registered runs completed: A01--A09 at seeds `1`, `7`, and `2025`.
Each uses 288 source-fit trials and 288 target-test trials, 1000 epochs,
Adam `0.001`, batch size `64`, deterministic CUDA, and exactly one target
evaluation after the final epoch. No source-validation split is used.

The configuration retains fusion, ECA at `sep_pre_pool`, post-activation
b2 transfer, no pooling/dropout/temporal BN, and spatial/classifier/hidden
max-norm `1.0/0.25/0.25`. Hidden max-norm and the hidden ELU are reconstruction
assumptions, not settings explicitly established by the paper.

## Results

Accuracy is independently recomputed from saved true/predicted labels.
All SDs below are sample SD (`ddof=1`); per-subject seed SD is across three
runs, while each seed's group SD is across nine subjects.

| Subject | Seed 1 (%) | Seed 7 (%) | Seed 2025 (%) | Seed mean (%) | Seed SD (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A01 | 84.3750 | 82.9861 | 80.2083 | 82.5231 | 2.1216 |
| A02 | 25.0000 | 50.0000 | 67.0139 | 47.3380 | 21.1331 |
| A03 | 89.9306 | 91.6667 | 88.5417 | 90.0463 | 1.5657 |
| A04 | 79.1667 | 81.2500 | 80.2083 | 80.2083 | 1.0417 |
| A05 | 67.0139 | 71.1806 | 70.1389 | 69.4444 | 2.1684 |
| A06 | 67.0139 | 25.0000 | 59.3750 | 50.4630 | 22.3799 |
| A07 | 75.0000 | 85.0694 | 80.9028 | 80.3241 | 5.0596 |
| A08 | 80.9028 | 81.2500 | 76.7361 | 79.6296 | 2.5119 |
| A09 | 81.2500 | 81.9444 | 82.9861 | 82.0602 | 0.8738 |

| Seed | Nine-subject mean (%) | Subject sample SD (pp) | Recorded seconds |
| --- | ---: | ---: | ---: |
| 1 | 72.1836 | 19.2297 | 1743.4108 |
| 7 | 72.2608 | 21.3347 | 1814.9753 |
| 2025 | 76.2346 | 9.0630 | 1830.6315 |

First averaging seeds within each subject, then summarizing the nine subject
averages gives **73.5597% +/- 14.9549 pp**. Against the paper's mean `81.17%`,
the difference is **-7.6103 pp**. The SD of the three seed group means is
`2.3169 pp`. The pooled 27-cell SD is `16.8173 pp`, descriptive only: there
are nine subjects, not 27 independent subjects. This is repeated training,
not prediction ensembling.

The recorded timer totals `5389.0176 s` (89.8170 minutes). It spans training
through final evaluation, excluding data loading and final result-file
writes; it is not a separately measured training-only GPU time.

## Collapse Diagnostics

- A02/seed1 predicts class 2 for all 288 test trials: counts `[0,0,288,0]`.
  Final training accuracy is 25%, training loss `1.38654485`.
- A06/seed7 predicts class 0 for all 288 test trials: counts `[288,0,0,0]`.
  Final training accuracy is 25%, training loss `1.38662705`.

Both target label distributions are `[72,72,72,72]`. These are single-class
collapses, not evidence of random prediction or solely a generalization gap.
Neither result is omitted or rerun to improve the aggregate. The mechanism
of collapse has not been established.

## Artifact And Independent Audit

- Training commit: `75fac1ab328be6ac5fa8db5e2f8302b5d08de891`, version `1.0.75`.
- Python `3.10.21`; PyTorch `1.12.1+cu116`; CUDA `11.6`; cuDNN `8302`;
  NVIDIA GeForce RTX 4070 Laptop GPU.
- Deterministic algorithms enabled; cuDNN benchmark and TF32 disabled.
- Directory: `results/seed_fullsource_v1_0_75/seed{1,7,2025}`.
- 81/81 nonempty JSON/NPZ/PT artifacts retained (666821538 bytes).
- Histories cover epochs 1--1000 with null target/validation fields; fit
  indices cover the complete source session. JSON and NPZ accuracies match.
- Independent CPU-only QA verified checkpoint final-state fingerprints,
  configurations, source/input hashes, and all statistics without retraining
  or modifying artifacts. The original automated summary has `errors=[]`.
- `summary_audited.json` and `.csv` retain the original summary;
  `summary_audited_v76.json` and `.csv` add strengthened audit checks and an
  SHA-256 manifest for all 81 artifacts. No original run is overwritten.

## Decision

The retained exploratory seed42 candidate remains `78.6651% +/- 9.8180 pp`;
none of these new seeds improves it. The sensitivity results undermine a
claim of robust reproduction at that best observed mean. No seed is promoted
using target results, and no exact-paper reproduction claim is made. The
historical reconstruction includes repeated target-labelled diagnostics;
final-only evaluation within each new run does not erase that exposure.
