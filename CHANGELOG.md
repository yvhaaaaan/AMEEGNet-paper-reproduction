# Changelog

# v1.0.41 - 2026-10-05

- Added an isolated `depth_pre_sep` ECA placement audit for the paper's
  wording that describes attention after the depthwise layer. The default
  remains ECA on the completed branch outputs.
- The alternative is recorded as a structural ambiguity check, not as a new
  method or a promoted result.

# v1.0.40 - 2026-10-05

- Added explicit batch-size and weight-initialization switches for isolated
  audits of paper details that are not reported in the article.
- The default remains batch size 64 with untouched PyTorch initialization, so
  the prior reproduction artifacts are unchanged.

# v1.0.39 - 2026-10-05

- Archived the clean Python 3.10 / PyTorch 1.12.1 + CUDA 11.6 nine-subject
  reproduction batch using the v1.0.38 source tree.
- The final-only T-to-E result was 76.35% +/- 11.36% (n=9), with all nine
  subjects completing successfully and the batch auditor reporting no errors.
- The framework-version audit matched the Python 3.12 A01 result exactly;
  changing to the paper-era interpreter/runtime did not explain the remaining
  gap to the reported 81.17%.

# v1.0.36 - 2026-10-05

- Added an isolated max-norm constraint for the AMEEGNet Dense(32) layer;
  default behavior remains unchanged and only the final classifier is
  constrained when `--max-norm` is enabled.

# v1.0.35 - 2026-10-05

- Corrected the float32 tolerance in the trial-normalization unit test; no
  runtime behavior changed.

# v1.0.34 - 2026-10-05

- Added explicit `none`, per-trial, and per-channel-per-trial input
  normalization diagnostics. The default remains no normalization, matching
  the paper's stated segmentation-only preprocessing.
- Archived the isolated `1e-6` input-scale check as a negative unit hypothesis
  candidate; it is not promoted to a batch.

# v1.0.33 - 2026-10-05

- Added an explicit positive input-scale parameter and provenance field to
  audit whether the paper's data loader used microvolt or SI-unit EEG values.
- Kept the default input scale at `1.0`; no existing result is altered.

# v1.0.32 - 2026-10-05

- Archived the A01 ECA-bias diagnostic: final internal validation was 72.41%
  and the best internal validation was 81.03% at epoch 298; target evaluation
  remained disabled.
- Did not promote the ECA-bias hypothesis to a nine-subject run because it did
  not improve the controlled A01 diagnostic.

# v1.0.31 - 2026-10-05

- Added an isolated ECA convolution-bias switch for a literal audit of the
  paper's `W*z+b` equation; the default remains bias-free ECA.

# v1.0.30 - 2026-10-05

- Archived a post-hoc audit of existing target-session curves. Selecting the
  per-subject maximum test accuracy after training raised historical means by
  only 2.62--3.51 percentage points, to 67.36--74.27%, still below the paper
  mean; these values remain exploratory and are not formal results.

# v1.0.29 - 2026-10-05

- Archived the literal classification-head A01 diagnostic: no Dense(32)
  activation and no head dropout; final internal validation was 65.52% and
  the best internal validation was 81.03% at epoch 743, with zero target
  Session evaluations.
- Moved the new `head_dropout` argument after the existing BatchNorm
  arguments to preserve positional-call compatibility for the model API.

# v1.0.28 - 2026-10-05

- Added an independent A01 diagnostics auditor that re-evaluates saved
  validation predictions, verifies split/provenance/checkpoint integrity, and
  confirms that no target Session was evaluated.
- Added model unit tests for BatchNorm controls, classifier-head dropout,
  finite forward/backward behavior, and deterministic evaluation state.
- Archived the verified v1.0.27 diagnostic report and corrected the BN
  interpretation to preserve its 10.34-point endpoint difference.

# v1.0.27 - 2026-10-05

- Archived the A01 BN epsilon/momentum diagnostic: changing BatchNorm
  statistics changed the validation trajectory but did not establish the
  reproduction gap's cause; no target-session test was evaluated.
- Made classifier-head dropout independent from branch dropout so the paper's
  explicitly stated `Flatten -> Dense(32) -> Dense(classes)` path can be
  audited without adding an undocumented head dropout.
- Forwarded BN and checkpoint controls through the nine-subject runner.

# v1.0.25 - 2026-10-04

- Archived the paired T-to-E/E-to-T Session audit: 72.53% +/- 8.62%, which is
  8.64 percentage points below the paper mean.
- Confirmed that Session direction alone does not explain the reproduction gap.

# v1.0.24 - 2026-10-04

- Added a bidirectional Session audit and archived the E-to-T batch. The
  paired T/E mean is reported descriptively, not substituted for the paper's
  unspecified evaluation protocol.

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

## v1.0.26 - 2026-10-05

- Added configurable BatchNorm epsilon and momentum for controlled
  implementation diagnostics; defaults preserve the existing PyTorch values.
- Added finite gradient-norm logging, periodic checkpoints, validation
  predictions, and source/data provenance to single-subject runs.
- Kept test evaluation final-only by default and did not change the model
  architecture or training protocol.

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
## v1.0.37 - 2026-10-05

- Added an audit script for the nine-subject hidden-Dense max-norm batch.
- Archived the final-only batch provenance and prediction consistency checks.
- Confirmed 9/9 subjects, 1000 epochs, 288/288 train/test trials, and no
  target-session evaluation during training for the audited batch.

## v1.0.38 - 2026-10-05

- Archived the audited nine-subject dropout=0.25 reconstruction batch.
- Added the final-only batch audit and comparison against the preceding
  dropout=0.50 hidden-Dense max-norm candidate.

