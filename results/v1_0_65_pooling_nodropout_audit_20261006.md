# v1.0.65 pooling/no-dropout audit

The audit retained the two standard EEGNet temporal average-pooling layers but
set all dropout rates to zero. The target Session was never evaluated. The
three-subject source-validation screening produced:

| Subject | Final source-val accuracy (%) | Best source-val accuracy (%) |
| --- | ---: | ---: |
| A01 | 84.48 | 91.38 |
| A02 | 56.90 | 65.52 |
| A04 | 67.24 | 72.41 |
| **Mean ± sample SD** | **69.54 ± 13.94** | **76.44 ± 13.39** |

The final source-validation mean is below the pre-registered no-pooling
reference of `72.99%`. Under the decision rule, this setting was rejected and
was not expanded to the nine-subject target evaluation. The best source
validation values are retained only for diagnosis and were not used for the
decision.
