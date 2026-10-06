# v1.0.78 source-only paired hidden-head stability results

## Fixed Comparison

All four registered GPU runs completed at commit
`067f77b23ad92f262ffdd068343d784e37f7a947`, version `1.0.78`, clean worktree.
Each fits all 288 source trials for 1000 epochs with **zero target-session
evaluations**. Only the hidden ELU is removed; fusion, ECA position, max-norm,
Adam settings, deterministic CUDA, seed, and isolated batch-order RNG match
the v1.0.75 reference runs. No training/model/data code was changed.

For each pair, the comparable configuration differs only in `head_elu`,
`test_evaluation`, and version/commit metadata. Saved source/input hashes
agree; histories cover epochs1--1000 with finite loss and null target scores.
NPZ target/validation arrays are empty and fit indices cover all 288 trials.

The same-seed initial models were reconstructed from the saved configurations:
each initial fingerprint matches its own record, and the actual parameter
values agree across the activation switch. The fingerprint strings between
models differ because removing ELU changes the final layer's module index.

## Frozen Source Evaluation

The new weights were independently evaluated in CPU `eval()` mode on their
source trials, batch32, with no parameter/BN updates. This is separate from
the online training accuracy accumulated during optimizer updates.

| Subject/seed | Source accuracy with ELU | Without ELU | Frozen source loss without ELU | Recorded GPU seconds |
| --- | ---: | ---: | ---: | ---: |
| A02/1 | 25.00% | 100.00% | 0.00226220 | 208.9170 |
| A06/7 | 25.00% | 100.00% | 0.00004027 | 207.4541 |
| A02/2025 | 100.00% | 100.00% | 0.00004244 | 202.8564 |
| A06/2025 | 100.00% | 100.00% | 0.00022173 | 202.1069 |

All four new source prediction counts are `[72,72,72,72]`, matching the
source labels exactly. No hidden linear unit has source SD below `1e-5`;
output logit SDs remain nonzero. A02/1 hidden linear outputs range from
`-29.963` to `31.863`, and A06/7 from `-22.736` to `22.451`: the complete
post-ELU negative saturation seen in the old failed checkpoints is absent
because the new head does not contain that activation.

The four recorded GPU-run timers total **821.3344 s** (13.6889 minutes).
Data loading and final artifact writes are outside the timer; there is no
target evaluation included in this batch.

## Audit And Provenance

`results/source_head_stability_v1_0_78/frozen_source_audit_v79.json` records
the diagnostic at commit `bf1c8b9bcc151d13085876d0531da58598e38cb2`, clean
worktree, including diagnostic/training code hashes, checkpoint/input hashes,
batch size, and unchanged final model fingerprints. The strengthened tool
enforces training-source hash equality and rejects changed parameters/buffers.
All 26 tests pass, including deliberately mismatched hashes and mutated state.

Artifacts remain under `results/source_head_stability_v1_0_78`, with separate
JSON/NPZ/PT files per run. The old failed and seed2025 reference checkpoints,
accuracies, and group summaries remain unchanged. The prior source diagnostic
and its independent QA are recorded separately in the v1.0.77 audit.

Final independent CPU-only QA passed without findings. It reproduced all
four source evaluations exactly, checked identical initial values for all
31 parameter tensors and 49 state tensors per pair (with classifier keys
matched), and verified provenance and rejection guards. All 33 monitored
files remained hash-identical; no training or target evaluation was performed.

## Interpretation

Under these two fixed-seed failure conditions, removing the unspecified
hidden ELU prevents the observed final source collapse. This supports a
role for that reconstruction choice in the failure mode. It does not prove
the author's hidden activation, establish the full training-time mechanism,
or estimate target-session generalization. The two same-subject seed2025
controls also retain correct source decoding.

No new held-out accuracy is available from this batch. The previous seed42
no-ELU nine-person result remains `77.93% +/- 9.49%`, and the higher observed
exploratory ELU candidate remains `78.6651% +/- 9.8180%`. Neither is replaced
with the 100% source result. Matching the paper's reported mean remains an
open goal; no new target batch is launched or promoted from training fit alone.
