# v1.0.72 source-selected epoch refit audit

## Result

The nine subject-specific epoch counts were fixed from the prior source-only
validation run, then each subject was retrained on all 288 source-session
trials before one target-session evaluation.

| Subject | Fixed epoch | Final target accuracy |
| --- | ---: | ---: |
| A01 | 309 | 82.29% |
| A02 | 121 | 57.64% |
| A03 | 963 | 95.14% |
| A04 | 346 | 79.17% |
| A05 | 293 | 75.69% |
| A06 | 571 | 65.28% |
| A07 | 321 | 86.11% |
| A08 | 147 | 80.21% |
| A09 | 273 | 77.08% |
| **Mean +/- sample SD** |  | **77.62% +/- 10.99%** |

Compared with the paper's `81.17% +/- 10.43%`, the mean difference is
`-3.55` percentage points. Compared with the retained final-epoch candidate
`78.67% +/- 9.82%`, this is `-1.04` percentage points. Runtime was 727.13 s
total, 80.79 s mean, and 67.90 s median per subject.

## Audit

The dedicated epoch-refit auditor passed with no errors. Every subject used
288 source-fit trials, had 288 balanced target trials, recorded exactly one
target evaluation after the fixed epoch count, and had complete JSON/PT/NPZ
artifacts with matching source hashes. The generic 1000-epoch auditor is not
used for this batch because subject-specific epoch counts are intentional.

## Decision

Source-selected refitting improves on the 230-trial checkpoint restoration
audit, but it does not improve the retained final-epoch candidate. It is
therefore not promoted as the primary reproduction result. It remains a
valid exploratory training-protocol audit and does not justify changing the
network architecture.
