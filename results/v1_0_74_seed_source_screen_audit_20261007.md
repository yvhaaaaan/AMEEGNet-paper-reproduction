# v1.0.74 source-only seed screen

The nine registered source-only runs completed at commit `c0c83aa`. Each uses
230 source-fit trials and 58 stratified source-validation trials, 1000 epochs,
and **zero target-session evaluations**. The final epoch is reported below;
validation-best checkpoints are not substituted.

| Seed | A01 final validation (%) | A02 | A04 | Three-subject mean (%) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 84.4828 | 74.1379 | 84.4828 | 81.0345 |
| 7 | 81.0345 | 79.3103 | 81.0345 | 80.4598 |
| 2025 | 89.6552 | 60.3448 | 79.3103 | 76.4368 |

Files are retained under `results/seed_screen_v1_0_74`, one JSON/NPZ/PT set per
subject and seed. Configuration and source hashes match apart from seed and
subject-specific fields. All seeds were carried into the registered v1.0.75
full-source sensitivity batch; none was selected from target accuracy.

## Interpretation Limit

The source split uses `random_state=seed`, so initialization, batch order,
and source-validation partition vary together. This is **not** an isolated
initialization experiment. Independent QA reconstructed all saved splits;
the validation-set symmetric differences between seed pairs 1/7, 1/2025,
and 7/2025 are respectively 106, 100, and 92 trials on each screened subject.

The follow-up full-source batch has no validation split. Its results must be
reported separately rather than assuming a high source-validation score
guarantees high cross-session target accuracy.
