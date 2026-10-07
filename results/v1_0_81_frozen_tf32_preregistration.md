# v1.0.81 frozen source-validation cuDNN precision diagnosis

## Rationale

The installed pristine torch1.12.1+cu116 process reports
`matmul.allow_tf32=False` and `cudnn.allow_tf32=True`. Our deterministic
runner explicitly disables both. [PyTorch's CUDA documentation](https://docs.pytorch.org/docs/stable/notes/cuda)
describes the separate flags and their defaults. The paper specifies
PyTorch1.12, but does not specify these backend permissions. This is a
possible numerical implementation difference, not an author-confirmed
setting or an added rehabilitation/calibration method.

The no-ELU v1.0.80 source gate failed. This separate read-only backend
diagnosis does not override that failure or launch a target batch.

## Fixed Evidence

- All twelve ELU controls: A01/A02/A04 at seeds1/7/2025 from v1.0.74;
  A06 at the same seeds from v1.0.80.
- Frozen final-epoch weights and recorded58-trial source-validation sets.
- CUDA inference only, `eval()` mode, no optimizer/gradient steps, no BN
  recalibration, no target-session predictions or label evaluation.
- Only the cuDNN TF32 permission changes fromFalse toTrue. Matrix
  multiplication TF32 remainsFalse; deterministic algorithms remainTrue,
  cuDNN deterministicTrue and benchmarkFalse. Data/tensors remainfloat32.
- FP32 mode must reproduce every saved source-validation prediction.
- Record maximum/RMS logit difference, changed predicted trials, source
  validation loss/accuracy, and unchanged weights/checkpoint hashes.
- Allowing TF32 does not prove that a specific chosen kernel uses TF32.

This diagnostic cannot establish the numerical behavior of training or
estimate a newly trained model's target accuracy. Neither an inference
difference nor its absence automatically qualifies another target run.
Any subsequent training experiment requires its own source-only protocol,
fixed subject/seed set and gate. Do not use these validation scores to
choose a per-subject precision mode or reinterpret the paper result.

Tool: `audit_frozen_tf32.py`. Output:
`results/frozen_tf32_v1_0_81/summary.json`, no overwrite.
Independent P1 must pass before the twelve-checkpoint diagnosis; P2 follows
the frozen output. No training/model/data source files change in this version.
