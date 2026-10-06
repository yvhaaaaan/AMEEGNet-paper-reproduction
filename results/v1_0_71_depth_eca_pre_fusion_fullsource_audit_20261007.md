# v1.0.71 corrected full-source depth-ECA/pre-fusion audit

## Result

The source-only gate from v1.0.70 passed, and the protocol-corrected final
run then used all 288 trials from the source session for every subject. The
target session was evaluated once after epoch 1000.

| Subject | Final target accuracy |
| --- | ---: |
| A01 | 84.38% |
| A02 | 62.15% |
| A03 | 91.32% |
| A04 | 83.33% |
| A05 | 69.10% |
| A06 | 58.33% |
| A07 | 73.26% |
| A08 | 84.38% |
| A09 | 81.25% |
| **Mean +/- sample SD** | **76.39% +/- 11.26%** |

Compared with the paper's `81.17% +/- 10.43%`, the mean difference is
`-4.78` percentage points. Runtime was 2467.83 s total, 274.20 s mean, and
269.86 s median per subject.

## Audit

The batch auditor reported no errors. All nine subjects have 288 source-fit
trials, 288 target-test trials, 1000 recorded epochs, exactly one final target
evaluation, balanced four-class target labels, matching configuration fields,
and identical source hashes at commit `e058461`.

## Decision

This is a valid full-source comparison but does not improve the retained
candidate `78.67% +/- 9.82%` from v1.0.52. The corrected depth-ECA plus
pre-activation-fusion interpretation is therefore rejected for promotion.
The earlier v1.0.70 split-fit batch remains archived separately and is not
used for this decision.
