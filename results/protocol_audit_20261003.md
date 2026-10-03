# Training protocol audit, 2026-10-03

The historical run_s01.py strict flag disabled standardization, pooling,
dropout and label smoothing, but still used AdamW (weight_decay=1e-4)
and gradient clipping (max_norm=5). It did NOT implement the Adam protocol
specified in reproduction_matrix.md. Preserve all historical results;
do not label these runs exact paper reproductions.

Completed 150-epoch A01 diagnostics under that historical optimizer:

| Factor | Final test accuracy |
| --- | --- |
| Temporal BN retained, BN then ELU | 47.57% |
| Temporal BN removed, BN then ELU | 47.57% |
| Temporal BN retained, ELU then BN | 42.71% |

These short runs do not exclude BN-related effects at other durations.
Do not select architecture or epoch based on their test scores. They are
implementation diagnostics, not confirmatory performance experiments.

The corrected strict runner now uses Adam, zero weight decay and no
gradient clipping. Network structure is unchanged for the verification
run s01_adam_no_clip_150.json. This is a protocol correction involving
multiple optimizer settings, not a causal single-factor comparison.
Optimizer defaults are reconstruction choices unless source evidence
specifies them. Architecture fidelity remains unverified.

Final weights and per-trial labels/predictions are saved beside new JSON
results. Existing result artifacts must not be overwritten.

Corrected protocol verification completed on CUDA: 150 epochs, 33.77 s,
final A01 accuracy 48.26% (139/288). The saved predictions independently
reproduce the JSON accuracy; all 150 recorded training losses are finite.
This short run does not establish paper-level reproduction success.
