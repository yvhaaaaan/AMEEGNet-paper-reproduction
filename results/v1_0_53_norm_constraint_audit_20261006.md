# v1.0.53 max-norm constraint audit

## Purpose

This audit isolates the role of the weight-norm projection used by the
current paper-reconstruction candidate. It does not add any method proposed
for the downstream stroke-rehabilitation study. Both batches use the same
v1.0.52 graph, data, seed, T-to-E protocol, and final-epoch target evaluation;
only the constraint switches differ.

## Results

| configuration | final T-to-E accuracy | sample SD | difference from paper |
| --- | ---: | ---: | ---: |
| no max-norm constraints | 47.84% | 4.58% | -33.33 pp |
| spatial/classifier max-norm only; no Dense(32) constraint | 50.50% | 6.03% | -30.67 pp |
| current candidate: spatial 1.0, classifier 0.25, Dense(32) 0.25 | 78.67% | 9.82% | -2.50 pp |

The first two batches completed for all nine subjects and passed the batch
auditor. All histories contain 1000 epochs, target labels were not evaluated
during training, each target session was evaluated once at the end, and the
prediction artifacts are present for every subject.

## Interpretation

The projection is a major implementation factor for this reconstruction:
removing it causes severe cross-session degradation even though the source
training accuracy reaches 100%. Removing only the hidden Dense(32) constraint
has the same qualitative failure. This supports retaining the current
constraint configuration for further paper-protocol investigation, but does
not prove that the original authors used these exact thresholds; the paper
does not specify them.

v1.0.53 therefore exposes the three thresholds as recorded command-line
parameters, with defaults preserving the audited v1.0.52 behavior. The next
screening step uses source-session validation only and does not use target
labels to select thresholds.
