# v1.0.78 paired source-only head stability check

## Question

The frozen source diagnostic found complete hidden ELU saturation in two
retained runs. The paper's `Flatten -> Dense(32) -> Dense(4)` description does
not explicitly specify this added activation. Removing it was already tested
at seed42 in v1.0.54 (77.93% nine-subject target mean), so this is a stability
follow-up, not a newly invented method or a target-accuracy search.

## Fixed Runs

- A02 seed1 and A06 seed7: the two previously observed collapse conditions.
- A02 seed2025 and A06 seed2025: previously registered same-subject controls.
- Change only `head_elu=false`; retain fusion, `eca_stage=sep_pre_pool`, no
  pooling/dropout/temporal BN, and max-norm `1.0/0.25/0.25`.
- All 288 source trials, no validation split, deterministic CUDA, isolated
  seeded batch-order RNG, Adam `0.001`, batch64, final epoch1000.
- **Zero target evaluations** (`--test-evaluation none`). Do not use target
  labels to choose whether to report these runs.
- Store separate JSON/NPZ/PT sets under `results/source_head_stability_v1_0_78`.
- Check initial parameter values match across the activation switch at a
  fixed seed. State fingerprint strings differ because the final layer's
  module index changes; do not call them initialization mismatches.
- After training, use the frozen CPU source diagnostic to compare source
  accuracy, loss, output variability, and hidden activations.

## Reporting Rule

Report all four source results, whether or not collapse is eliminated. These
runs do not estimate held-out decoding performance. Do not update the primary
nine-subject result or start new target batches solely because training
accuracy is higher. A paired result can implicate the activation choice in
these seeds' failure mode, but cannot prove the author's hidden activation or
guarantee generalization.
