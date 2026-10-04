# Changelog

# v1.0.23 - 2026-10-04

- Archived the deterministic no-temporal-BN batch: 72.61% +/- 8.81%, still
  8.56 percentage points below the paper mean.
- Defined the next protocol audit as the reverse E-to-T Session direction;
  the paper says one Session trains and the other evaluates but does not state
  the direction or whether both directions are averaged.

# v1.0.22 - 2026-10-04

- Forwarded temporal-BN and normalization-order switches through the
  nine-subject runner so the literal architecture audit can run consistently.

# v1.0.21 - 2026-10-04

- Archived the deterministic v1.0.20 no-head-ELU batch: 70.25% +/- 8.38%.
- Defined the next literal architecture audit without temporal-convolution
  BatchNorm, because the paper explicitly places BN/ELU after the second and
  third layers but does not specify BN after the first layer.

# v1.0.20 - 2026-10-04

- Archived the deterministic v1.0.19 nine-subject max-norm batch: 71.60% +/-
  9.06%, 9.57 percentage points below the paper mean.
- Defined the next literal-architecture audit without the inferred Dense(32)
  ELU, while retaining the controlled standard EEGNet pooling/dropout path.

# v1.0.19 - 2026-10-04

- Archived the deterministic v1.0.17 nine-subject standard pooling/dropout
  audit: 68.29% +/- 8.55%, 12.88 percentage points below the paper mean.
- Defined the next controlled candidate as standard EEGNet max-norm projection
  on top of the same pooling/dropout path.

# v1.0.18 - 2026-10-04

- Updated the batch auditor for final-only target-session evaluation and
  included per-subject configuration, runtime, and artifact checks.

# v1.0.17 - 2026-10-04

- Added final-only deterministic batch forwarding for the standard
  EEGNet-pooling hypothesis.
- Recorded the A01 candidate run separately before starting the nine-subject
  batch.

# v1.0.16 - 2026-10-04

- Archived the clean deterministic A01 strict run at v1.0.15: 50.35% final
  target-session accuracy after 1000 epochs, with one final test evaluation.
- Defined the next controlled audit as the standard EEGNet pooling/dropout
  implementation hypothesis; it remains separate from the v1.7 method.
- Updated the nine-subject runner to forward deterministic and test-evaluation
  controls to each subject process.

# v1.0.15 - 2026-10-04

- Added an independent raw-MAT audit for all nine BCI IV 2a subjects. The
  official trial windows, labels, order, and cached NPZ arrays match bitwise.
- Added data finite-value/session/label validation and recorded source SHA-256.
- Made model head-shape inference analytical so construction does not update
  BatchNorm statistics or consume Dropout RNG state with a dummy batch.
- Added deterministic CUDA mode, code/data/runtime provenance, state
  fingerprints, train/validation indices, and explicit final-only test
  evaluation to the training runner.
- Corrected the audit boundary: previous runs that inspected target-session
  curves remain exploratory and are not treated as confirmatory results.

## v1.0.14 - 2026-10-04

- Audited pooled/post-pooling-dropout reproduction batches for seed 42 and
  seed 1.
- Recorded the paper-only reproduction boundary and current gap to the
  published mean.

## v1.0.13 - 2026-10-04

- Added an explicit T/E versus E/T session-direction comparison.

## v1.0.12 - 2026-10-04

- Added an explicit comparison switch for the inferred Dense(32) ELU.

## v1.0.11 - 2026-10-04

- Added optional standard EEGNet max-norm projection for spatial depthwise
  kernels (`1.0`) and the final classifier (`0.25`).

## v1.0.10 - 2026-10-04

- Added controlled standard EEGNet dropout placement after pooling.
- Restored explicit ELU application in both normalization-order paths after
  the dropout refactor.

## v1.0.9 - 2026-10-04

- Added explicit dropout override support for controlled EEGNet protocol
  comparisons and batch forwarding.

## v1.0.8 - 2026-10-04

- Corrected the batch audit release metadata and included fusion/ECA settings
  in the configuration audit.
- Archived the nine-subject no-fusion/no-ECA ablation audit.

## v1.0.7 - 2026-10-04

- Added batch-run forwarding for the no-fusion/no-ECA ablation.

## v1.0.6 - 2026-10-04

- Fixed ablation-only channel construction when fusion is disabled.
- The default fusion path is unchanged; the failed v1.0.5 attempt produced
  no result artifact and is retained as an audit event.

## v1.0.5 - 2026-10-04

- Added explicit fusion and ECA switches to reproduce the paper's ablation
  structures without altering the default AMEEGNet path.

## v1.0.4 - 2026-10-04

- Added reproducible nine-subject artifact auditing and SHA-256 inventory.
- Corrected test-evaluation and pooling-hypothesis documentation.
- Preserved v1.0.3 training results without retraining or selecting epochs.

## v1.0.3 - 2026-10-03

- Added the `--paper-pooling` option to the nine-subject runner.
- Fixed batch-run forwarding so the selected architecture is recorded for
  every subject.

## v1.0.2 - 2026-10-03

- Added an explicit `--paper-pooling` switch for the standard EEGNet
  average-pooling and dropout path, without changing the strict default.
- Recorded the complete 1000-epoch no-temporal-BN A01 diagnostic separately.

## v1.0.1 - 2026-10-03

- Recorded the corrected-protocol A01 1000-epoch run.
- Formal final test accuracy was 50.35%; the in-training test maximum was 51.04%.

## v1.0.0 - 2026-10-03

- Archived the current AMEEGNet paper-reconstruction workspace.
- Added the reproduction matrix and protocol audit.
- Added strict A01 training artifacts and documented the Adam/no-clipping protocol correction.
- Excluded local datasets, checkpoints, and generated JSON artifacts from Git tracking.
