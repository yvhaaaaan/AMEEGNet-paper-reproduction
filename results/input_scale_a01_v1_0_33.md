# Input-scale diagnostic: A01

The official MAT values are approximately microvolt scale (around +/-100 in
the cached arrays). This diagnostic tested the same controlled A01 setup as
the best fixed architecture audit, but multiplied both sessions by `1e-6` to
simulate SI-unit values before training.

| Input scale | Final internal validation | Best internal validation | Best epoch | Target evaluations |
|---:|---:|---:|---:|---:|
| `1e-6` | 51.72% | 53.45% | 877 | 0 |

The run used 230 fitting trials and 58 source-Session validation trials,
seed 42, deterministic CUDA, Adam at 0.001, batch size 64, and 1000 epochs.
The result is substantially below the corresponding scale-1 diagnostic, so a
bare microvolt-to-volt conversion is not a plausible explanation for the
paper-level gap. It also indicates that any SI-unit pipeline would require
another, currently undocumented normalization step.

This result is diagnostic only and does not alter the formal reproduction
configuration.
