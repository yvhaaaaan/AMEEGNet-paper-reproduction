# Changelog

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
