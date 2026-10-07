# v1.0.80 paired source-validation results

## Registered Runs

All 15 new GPU runs completed at clean commit
`707b6d80c9a2a1dbb85feb5f1dba4214a9c20077`, version1.0.80.
Together with the nine reused v1.0.74 controls, the screen contains
24 runs, twelve matched pairs, four subjects and three fixed seeds.
Each run fits230 source trials, validates58, and completes1000 epochs.
No target-session evaluations or predictions were produced.

The [registration](v1_0_80_source_head_validation_preregistration.md)
and `audit_source_head_validation.py` specify the frozen expansion rule.
Only final-epoch validation is compared, not validation-best checkpoints.

## Results

| Subject | ELU seed-mean source validation (%) | No ELU (%) | Paired difference (pp) |
| --- | ---: | ---: | ---: |
| A01 | 85.0575 | 82.7586 | -2.2989 |
| A02 | 71.2644 | 70.1149 | -1.1494 |
| A04 | 81.6092 | 81.6092 | 0.0000 |
| A06 | 71.8391 | 73.5632 | +1.7241 |
| Mean of four subject means | 77.4425 | 77.0115 | -0.4310 |

The twelve pairs contain539 versus536 correct validation predictions,
an integer difference of-3. Repeated seeds/validation partitions are
not independent subjects; the descriptive subject-delta sample SD is
1.7161pp. All twelve no-ELU final online training accuracies are100%.

The audit has no integrity errors, but **the registered performance gate
fails** because the validation difference is negative. CLI exit0 means
the files are valid; `source_gate_pass=false` controls the decision.
No new nine-subject target batch is launched for this head hypothesis.
The small negative difference does not prove that ELU is structurally
correct or that removing it is universally worse.

## Timing And Independent QA

The fifteen new recorded run timers total2581.325818s (43.0221min), with
median171.153867s. The nine historical controls are excluded from this
time total. Timers include training/validation but exclude input loading
and final artifact writes; they are not target prediction latency.

Independent P2 passed: all72 JSON/NPZ/PT artifacts, source/data hashes,
230/58 partitions, source labels, paired numerical initialization and
RNG settings, final checkpoint fingerprints, and PT/JSON results match.
CPU evaluation reproduced all1392 saved source-validation predictions
exactly; maximum CPU/GPU validation-loss difference was2.3842e-7.
Weights and BN buffers were not updated. All86 monitored files remained
hash-identical before/after QA. The hash-ledger digest is:

`ad1de784ebe265786782f1a12e7b23f32bb57acef509345f28281503f8264c75`

All32 regression tests passed. The auditor reproduces the stored summary:

```powershell
python audit_source_head_validation.py --results results/source_head_validation_v1_0_80 --references results/seed_screen_v1_0_74 --out <fresh-summary-path>.json
```

Local full records: `results/source_head_validation_v1_0_80/summary_audited.json`.
Existing files are not overwritten. The prior exploratory nine-subject
best remains78.6651% +/-9.8180%, below the paper's81.17% by2.5049pp.
This source-only negative result does not replace that target result.
