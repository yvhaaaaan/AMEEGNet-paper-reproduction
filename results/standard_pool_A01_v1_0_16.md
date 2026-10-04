# Standard EEGNet branch hypothesis: A01

Run code version: v1.0.16. The experiment retained the paper's stated data
and optimizer settings and added only the standard EEGNet branch operations:
average pooling, Dropout=0.5 after each pooling operation, and the existing
paper-described fusion and ECA. The target Session was evaluated once after
1000 epochs; it was not used for selection.

| Metric | Value |
| --- | ---: |
| Final E-session accuracy | 70.83% |
| Runtime | 210.21 s |
| E-session evaluations | 1 |
| Git commit | `ad104ad83055c3fafcbb12c217716ff0bdac49e2` |
| Data SHA-256 | `a87452ce68b0b86b2b45431f9200d26f31fa8106e36718d5e003dd285cf5abdf` |

This is an implementation hypothesis, not a confirmed detail of the original
paper. It is eligible for a nine-subject audit because it follows the
standard EEGNet branch definition and does not use any v1.7 enhancement.
