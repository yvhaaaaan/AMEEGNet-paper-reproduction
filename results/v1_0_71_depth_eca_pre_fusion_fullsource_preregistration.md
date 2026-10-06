# v1.0.71 corrected full-source confirmation

The v1.0.70 source-only gate passed with a final source-validation mean of
77.59%. This confirmation uses the same corrected model and factor settings,
but restores the primary final-run training protocol:

- A01--A09, Session 1 -> Session 2
- all 288 source-session trials used for fitting
- no source validation split in the final run
- target Session evaluated once after epoch 1000
- no pooling, no dropout, no temporal BN
- `eca_stage=depth_pre_sep`, `fusion_pre_activation=true`
- Adam `0.001`, batch size 64, seed 42, deterministic CUDA
- max-norm projections `1.0/0.25/0.25`

The result will be compared with both the retained primary candidate and the
paper's `81.17% +/- 10.43%`. The earlier 230-trial split-fit batch is kept as
an exploratory artifact and is not used for the decision.
