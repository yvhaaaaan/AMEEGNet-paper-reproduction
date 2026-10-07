# AMEEGNet reconstruction status at v1.0.83

## Retained Target Result

The existing nine-subject final-epoch candidate was re-audited without
training or target inference: **78.6651% +/- 9.8180%** (subject sample SD).
Compared with the [paper's](https://pmc.ncbi.nlm.nih.gov/articles/PMC11794809/)
81.17% mean, the difference remains **-2.5049 percentage points**.
The dataset is BCI IV 2a four-class T-to-E, 288 training and 288 target
trials per subject, 22 channels, 250 Hz, [1.5,6] seconds, 1125 samples.
All nine existing prediction files match their JSON accuracies and the
registered full-source protocol. No low-accuracy run was removed.

This is an exploratory reconstruction candidate, not confirmed recovery
of the author's original implementation. Max-norm remains an unreported
assumption, and the broader history has repeated target-labelled
diagnostics. A stronger GPU is not evidence of better decoding accuracy.
The paper's reported SD denominator is not established here.

## This Round

| Experiment | Evidence | Decision |
| --- | --- | --- |
| Hidden ELU source-validation comparison | 15 new 1000-epoch GPU runs; 12 pairs over 4 subjects and 3 seeds | No-ELU expansion gate failed |
| ELU versus no-ELU final source validation | 77.4425% versus 77.0115%; paired difference -0.4310 pp | Keep all results; no target batch |
| No-ELU final online training | All twelve at 100% | Fitting stability is not validation improvement |
| Frozen cuDNN TF32 permission comparison | 12 checkpoints, 24 CUDA forwards, 0/696 changed source predictions | No newly trained or tested accuracy |

The fifteen new run timers total 2581.325818 seconds (43.0221 minutes),
median 171.153867 seconds, excluding nine historical controls.
Source validation is not an estimate of cross-session target performance.
The small negative head difference does not prove the network structure
is wrong; it simply fails the registered promotion rule.

## Audit And Records

- [Head screen and independent P2](v1_0_80_source_head_validation_audit_20261007.md):
  24 run records/72 artifacts verified; 1392 CPU source predictions reproduced exactly.
- [Frozen precision and independent P1/P2](v1_0_82_frozen_tf32_audit_20261007.md):
  output reproduced byte-for-byte, final model/BN state unchanged.
- All 39 tests pass; training/model/data code and dependencies are unchanged.
- JSON/NPZ/PT records and earlier failures remain local and are not overwritten.
- Versions 1.0.80 through 1.0.83 are traceable in local Git; no Git remote
  is configured, so these commits/tags have not been uploaded.

No alignment, augmentation, EMA, pretraining, test-time adaptation, or
v1.7/self-designed rehabilitation calibration was introduced. The registered
no-ELU target batch was not launched; this round has zero target evaluations.
Training and diagnosis jobs are no longer running. Matching the paper's
mean remains an open goal, not a completed reproduction claim. A future
training-precision comparison must be separately preregistered and
validated on the source Session; frozen inference cannot establish it.
