# v1.0.54 Dense(32) activation audit

This batch tests the paper-figure interpretation in which the first
Dense(32) layer has no visible activation. The graph, data, and training
protocol otherwise match the current candidate: no pooling, no dropout,
post-separable-convolution ECA, explicit max-norm constraints, Adam at
`1e-3`, batch size 64, 1000 epochs, and one final T-to-E evaluation.

## Result

The nine-subject final-only result is **77.93% +/- 9.49%** (sample SD),
compared with **78.67% +/- 9.82%** for the same configuration with ELU after
Dense(32). The no-ELU result is -3.24 percentage points from the paper's
81.17% mean and is rejected as the current candidate. All nine records passed
the batch audit.

The remaining candidate retains the Dense(32) ELU because the paper's prose
does not specify this activation and the controlled final-only result is
higher with it. This is a reproduction assumption, not a claim about an
unpublished author implementation.
