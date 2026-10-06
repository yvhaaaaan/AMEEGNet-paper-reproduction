# v1.0.68 depth-ECA/pre-fusion audit status

## Status

**Invalid for the registered joint-factor question.** The run completed and
the artifacts remain preserved, but the tested code did not implement both
registered factors simultaneously.

## Finding

In `Branch.from_temporal`, `depth_out` was first assigned the raw depthwise
output when `fusion_pre_activation=true`, then unconditionally replaced by
the ECA-gated output. Consequently, under `eca_stage=depth_pre_sep`, the
pre-activation switch had no effect on the tensor returned for the next
fusion junction.

The nine-subject result was `78.0864% +/- 10.0940%`, exactly matching the
earlier depth-ECA result. This identity is consistent with the ineffective
switch and is not evidence that the registered joint interpretation was
tested.

## Action

The historical JSON/PT/NPZ artifacts are retained and must not be used as a
positive or negative result for the v1.0.68 question. v1.0.69 fixes the data
flow, adds a regression test, and requires a new source-only screen before any
new nine-subject run.
