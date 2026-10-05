# Dropout 0.25 hidden Dense max-norm batch

Training source: v1.0.37, commit a4c93ada62f29cd0c3cd9513e1606c02944deb2a.
The batch used the fixed final-only protocol: 9 subjects, 1000 epochs, seed 42,
deterministic CUDA, 288 source-session training trials, and one target-session
evaluation after training. The audit script reported zero errors: all prediction
arrays reproduce the JSON accuracies, all target sessions have 72 trials per
class, histories have 1000 finite records, and no target test accuracy was
recorded during training.

| Subject | Final accuracy (%) | Recorded elapsed seconds |
| --- | ---: | ---: |
| A01 | 85.42 | 183.76 |
| A02 | 60.42 | 181.95 |
| A03 | 92.36 | 182.81 |
| A04 | 71.18 | 182.32 |
| A05 | 72.92 | 182.97 |
| A06 | 59.72 | 181.80 |
| A07 | 86.46 | 182.04 |
| A08 | 80.90 | 182.11 |
| A09 | 77.78 | 182.44 |

- Mean accuracy: 76.35030864%.
- Between-subject sample SD (ddof=1): 11.359851995%.
- Difference from published mean 81.17%: -4.81969136 percentage points.
- Recorded elapsed-time mean: 182.46764832 s.
- Recorded elapsed-time median: 182.32490200 s.
- Total recorded elapsed time: 1642.20883490 s.

The only changed factor from the preceding hidden-max-norm batch was the branch
and hidden-head dropout rate: 0.50 -> 0.25. Pooling, dropout placement,
max-norm projection, fusion, ECA, optimizer, data, seed, and final-epoch
evaluation were held fixed. This is an inferred EEGNet-default reconstruction
assumption, not a parameter explicitly reported by the AMEEGNet article.

The run was exploratory because the same target sessions had been inspected in
earlier diagnostics. It is not evidence that the article's exact implementation
has been recovered, but it is the strongest audited candidate so far.
