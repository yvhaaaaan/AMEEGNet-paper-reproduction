# v1.0.70 corrected depth-ECA/pre-fusion audit

## Protocol gate

The v1.0.69 data-flow fix passed the model regression tests. The source-only
screen was completed before the nine-subject run:

| Subject | Final source validation |
| --- | ---: |
| A01 | 81.03% |
| A02 | 68.97% |
| A04 | 82.76% |
| **Mean** | **77.59%** |

This exceeded the preregistered 72.99% reference, so the nine-subject
final-only run was allowed.

## Protocol correction

The first nine-subject command accidentally retained
`--validation-fraction 0.2` from the source-only screening command. It
therefore fitted only 230 of the 288 source-session trials for each subject.
Because the retained primary candidate fits all 288 source trials, that batch
is not protocol-comparable and must not be used to reject this structural
factor. Its artifacts and audit output remain preserved.

## Exploratory nine-subject split-fit result

Configuration: BCI IV 2a Session 1 -> Session 2, `[1.5, 6]` s, no pooling,
no dropout, no temporal BN, fusion enabled, `eca_stage=depth_pre_sep`,
pre-activation depth output at the fusion junction, Adam `0.001`, batch size
64, 1000 epochs, seed 42, deterministic CUDA, and max-norm projections
`1.0/0.25/0.25`. The target Session was evaluated once after epoch 1000.

| Subject | Final target accuracy |
| --- | ---: |
| A01 | 80.21% |
| A02 | 54.86% |
| A03 | 90.28% |
| A04 | 78.82% |
| A05 | 69.79% |
| A06 | 53.82% |
| A07 | 72.92% |
| A08 | 80.56% |
| A09 | 84.03% |
| **Mean +/- sample SD** | **73.92% +/- 12.56%** |

Compared with the paper's `81.17% +/- 10.43%`, the mean difference is
`-7.25` percentage points. This comparison is descriptive only because the
run used 230 source-fit trials. Runtime was 2475.79 s total, 275.09 s mean,
and 268.48 s median per subject.

## Audit result

The batch auditor reported no errors: all nine subjects had 230 source-fit
trials, 288 target-test trials, matching configuration fields, complete
prediction artifacts, and the same source code commit `35658fc`. The source
screen passed its registered gate. A protocol-correct full-source final run is
required before accepting or rejecting the factor.

This result does not support changing the network again based on the target
scores. The retained best controlled candidate remains `78.67% +/- 9.82%`
from the earlier v1.0.52 configuration, still `-2.50` percentage points below
the paper. The v1.0.68 batch is separately archived as invalid because its
pre-activation switch was ineffective.
