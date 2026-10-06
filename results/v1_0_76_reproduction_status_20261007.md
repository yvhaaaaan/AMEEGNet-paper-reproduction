# AMEEGNet reconstruction status at v1.0.76

## Retained Exploratory Result

BCI IV 2a, T session -> E session, four classes, all 288 source trials used
for fitting and all 288 target trials for testing; `[1.5,6] s`, 22 channels,
250 Hz, 1125 samples. Seed42, Adam `0.001`, batch64, final epoch1000, one
target evaluation per run. The current best observed nine-person candidate
remains **78.6651234568% +/- 9.8180285510 pp**, sample SD across nine subjects.

| Subject | Final target accuracy (%) |
| --- | ---: |
| A01 | 82.9861 |
| A02 | 62.1528 |
| A03 | 94.0972 |
| A04 | 80.2083 |
| A05 | 75.6944 |
| A06 | 65.6250 |
| A07 | 85.7639 |
| A08 | 81.5972 |
| A09 | 79.8611 |

The [original article, Table 1](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1540033/full)
reports `81.17% +/- 10.43%`. The current mean is lower by **2.5049 percentage
points**. The article does not explicitly establish its SD denominator.

The candidate contains no alignment, augmentation, EMA, pretraining,
test-time adaptation, or v1.7/self-designed rehabilitation method. It does
use hidden ELU and max-norm reconstruction assumptions not explicitly stated
in the article. Pooling, initialization, fusion widths, and checkpoint
selection remain incompletely specified by the author. Repeated target-
labelled diagnostics during reconstruction mean this is an exploratory
candidate, not an untouched held-out confirmation or exact code recovery.

## Newly Completed Work

| Version | Experiment | Mean +/- subject sample SD (%) | Interpretation |
| --- | --- | ---: | --- |
| 1.0.68 | Depth-ECA + pre-fusion | 78.0864 +/- 10.0940 | Invalid joint audit: pre-fusion switch had no effect |
| 1.0.70 | Corrected joint, 230 source-fit trials | 73.9198 +/- 12.5631 | Retained source split; not comparable to full 288 fit |
| 1.0.71 | Corrected joint, all 288 source-fit trials | 76.3889 +/- 11.2580 | No improvement |
| 1.0.72 | Source-selected fixed epochs, all 288 refit | 77.6235 +/- 10.9858 | Alternative training protocol; no improvement |
| 1.0.75 | Full-source seed1 | 72.1836 +/- 19.2297 | All nine included |
| 1.0.75 | Full-source seed7 | 72.2608 +/- 21.3347 | All nine included |
| 1.0.75 | Full-source seed2025 | 76.2346 +/- 9.0630 | All nine included |

All three newly registered seeds are reported. Averaging seeds per subject
and then summarizing nine subject averages gives **73.5597% +/- 14.9549 pp**;
27 repeated runs are not 27 independent subjects. Two single-class collapses
remain included, with final training and target accuracy both at 25%.

The v1.0.69 data-flow fix preserves raw **b2** depthwise transfer when ECA is
active. b3's local features remain BN/ELU/ECA processed, so the concatenation
is one-sided raw transfer, not both-sides-raw fusion. The default post-
activation model is unchanged. Regression tests now assert the recipient
concatenation explicitly.

The first v1.0.70 launch encountered a CPU allocation error before producing
run artifacts. The failure was retained in the audit; retrying with BLAS
thread limits succeeded. This does not establish the cause of the failure.
No failed or low-accuracy completed run was deleted or overwritten.

## Verification And Files

The strengthened batch auditor checks registered fit counts, source index
coverage, validation-split configuration, complete epoch sequences, one
final target evaluation, finite training loss, and JSON/NPZ accuracy equality.
Independent QA additionally verified all 27 v1.0.75 checkpoints/fingerprints
and source/input hashes. Tests cover data-flow and protocol regression risks.
All 20 tests pass (`unittest discover -s tests -q`, 5.784 s); strengthened
full-source and source-selected-epoch audits both report `errors=[]`.

Primary retained files: `results/all_no_pool_no_dropout_maxnorm_v1_0_52`.
Newly completed result directories: `results/all_depth_eca_pre_fusion_v1_0_70`,
`results/all_depth_eca_pre_fusion_fullsource_v1_0_71`,
`results/all_source_epoch_refit_v1_0_72`, and
`results/seed_fullsource_v1_0_75`. JSON/PT/NPZ remain local; Markdown audits,
the summary utility, and tests are versioned. No training is currently running.

## Open Goal

Matching the reported mean is **not achieved**. A faster GPU reduces runtime,
not the uncertainty in the author's training and model specification. Further
original-paper reconstruction requires resolving those assumptions; selecting
target-best seeds/epochs or omitting collapsed runs cannot satisfy the goal.
