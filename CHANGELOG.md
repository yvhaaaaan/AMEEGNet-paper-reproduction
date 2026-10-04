# Changelog

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
