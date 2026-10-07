# v1.0.80 paired source-validation gate for the literal head

## Rationale And Scope

The v1.0.78 full-source diagnostic prevented two fixed-seed collapse cases
by removing hidden ELU, but perfect source fitting is not validation evidence.
The [paper, section 2.3.3](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1540033/full)
specifies Flatten, Dense32, Dense4, and output Softmax, without explicitly
stating a hidden activation. This screen examines that reconstruction
assumption; no alignment, augmentation, or self-designed calibration is added.

## Registered Cells

- Subjects A01/A02/A04/A06; seeds1/7/2025: twelve paired comparisons.
- Controls with ELU for A01/A02/A04 reuse all nine v1.0.74 source-only records.
- New controls: A06 at all three seeds with hidden ELU retained.
- New changes: all four subjects and all three seeds without hidden ELU.
- Total new runs:15, executed sequentially on the GPU, one child process at
  a time. Each completed JSON/NPZ/PT set is preserved; no existing run is overwritten.
- Each pair uses the same seeded stratified 230/58 source split (fraction0.2),
  data hashes, initialization values, and isolated batch-order generator.
- The seed also changes the validation partition across seeds: this is not
  an initialization-only experiment. Within each pair the partition is fixed.
- Adam0.001, batch64, epoch1000, deterministic CUDA, no pooling/dropout/first
  BN, ECA at sep_pre_pool, post-activation b2 transfer; max-norm1/0.25/0.25.
- No target evaluation, no validation-best checkpoint selection; final epoch
  validation accuracy and loss are the registered outcomes.

The independent unit is the subject. This within-subject source validation
does not estimate cross-subject or cross-session accuracy. Repeated seeds
are summarized within each subject before forming the four-subject summary.

## Fixed Gate

Pass only if all audit checks pass, every no-ELU run's final online source
training accuracy is at least95%, and the mean paired final validation
difference (noELU minus ELU), averaged over seeds then subjects, is >=0 pp.
All twelve differences and all failures are reported. Do not change this
criterion, the seed set, or epoch count after observing results.

If the gate passes, a separately preregistered full-source target sensitivity
batch may evaluate all three fixed seeds on all nine subjects. If it fails,
do not launch that target batch or use source-fit100% to bypass the gate.
No seed-specific target maximum or per-subject seed selection is permitted.
The broader reconstruction remains exploratory because earlier target-labelled
diagnostics have been observed.

## Artifacts

New controls: results/source_head_validation_v1_0_80/elu/A06_seed{seed}.*.
Changes: results/source_head_validation_v1_0_80/noelu/Axx_seed{seed}.*.
Auditor: audit_source_head_validation.py. Its JSON summary is append-only
by filename (refuses overwrite), separate from all existing group results.
