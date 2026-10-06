# v1.0.77 source-only frozen checkpoint diagnosis

The v1.0.75 batch exposed final single-class collapse in A02/seed1 and
A06/seed7. Inspect these existing checkpoints alongside the same subjects'
seed2025 checkpoints. Seed2025 is a previously registered comparator, not a
new seed selected to replace the failed runs.

## Fixed Diagnostic

- CPU inference on all 288 **source** trials per checkpoint, batch size32.
- No target evaluation, model/BN update, retraining, or saved replacement model.
- Verify saved data hashes and final model fingerprints before inference,
  and unchanged model state and checkpoint hash afterward.
- Record source accuracy/loss/prediction counts, hidden Dense(32) pre/post
  ELU distributions, output logit variability, and BN buffer ranges.
- Record fractions of hidden preactivations below -10 and -20 and near-
  constant hidden units (source SD below `1e-5`), using fixed thresholds.
- Preserve all original accuracies and seed summaries.

This can identify a checkpoint's state and correlate it with collapsed
predictions, but cannot prove the cause or update order of collapse. It is
not a paper result or an algorithm enhancement.
