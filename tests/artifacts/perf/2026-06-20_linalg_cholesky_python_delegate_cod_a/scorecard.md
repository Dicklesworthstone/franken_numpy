# Python Stacked Cholesky Delegate Proof

Run identity:
- Date: 2026-06-20.
- Agent: `YellowElk` / `cod-a`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate: `fnp-python`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Same-worker performance proof: `vmi1152480`.

## Lever

For exact NumPy ndarray inputs with stacked square shape `(..., n, n)` and
`n >= 4`, default lower-triangle `fnp.linalg.cholesky` delegates before Rust
array extraction. This avoids the previous copy/extraction plus per-lane native
path for the measured stacked-SPD workload. The delegate path caches
`numpy.linalg.cholesky`, reuses the cached ndarray type helper, indexes shape
without a `Vec` allocation, and avoids creating kwargs on the default path.

This is the kept version. Earlier delegate drafts are retained in this
directory as negative/intermediate evidence.

`baseline_cholesky_python_linalg_vmi1227854.txt` is an invalid first Criterion
filter attempt. It compiled but emitted no Cholesky rows, so it is retained only
to explain the artifact trail and is not scored.

## Performance

Command:

```bash
AGENT_NAME=YellowElk RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a \
  rch exec -- cargo bench -p fnp-python --bench criterion_python_surface \
  cholesky_f64 -- --sample-size 10 --measurement-time 2 --warm-up-time 1 \
  --output-format bencher
```

Rows:

| Row | Old FNP | Old NumPy | Old FNP/NumPy | New FNP | New NumPy | New FNP/NumPy | New/Old FNP | Outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `batch10000_4x4` | 2,109,573 ns | 1,810,289 ns | 1.165x | 1,766,423 ns | 2,521,959 ns | 0.700x | 0.837x | WIN |
| `batch4000_8x8` | 5,566,219 ns | 2,149,175 ns | 2.590x | 1,483,647 ns | 1,572,176 ns | 0.944x | 0.267x | WIN |
| `batch2000_16x16` | 6,459,892 ns | 3,207,857 ns | 2.014x | 3,379,216 ns | 3,421,966 ns | 0.988x | 0.523x | WIN |
| `batch1000_32x32` | 11,059,012 ns | 5,741,576 ns | 1.926x | 4,993,838 ns | 4,866,396 ns | 1.026x | 0.452x | neutral/noisy |
| `batch500_64x64` | 21,866,929 ns | 10,619,813 ns | 2.059x | 7,382,796 ns | 7,639,253 ns | 0.966x | 0.338x | WIN |

Summary:
- Baseline FNP vs NumPy: 0 wins / 5 losses / 0 neutral.
- Final FNP vs NumPy: 4 wins / 0 material losses / 1 neutral.
- Final FNP vs old FNP: 5 wins / 0 losses / 0 neutral.

## Validation

Passed:
- `rch exec -- cargo test -p fnp-python --test conformance_linalg_decomp cholesky -- --nocapture`
  - Final rerun: 6 passed, 0 failed, 33 filtered out.
- `rch exec -- cargo check -p fnp-python --lib --bench criterion_python_surface`
  - Passed with inherited warnings.
  - Local rerun after touched-hunk rustfmt alignment also passed.
- `rch exec -- cargo build -p fnp-python --release`
  - Passed with inherited warnings.
- `git diff --check`
  - Passed.

Blocked or known gap:
- `rch exec -- cargo clippy ... -D warnings` selected a worker missing the
  pinned nightly clippy component. Local clippy reached analysis and failed on
  broad pre-existing `fnp-python` lint inventory outside this hunk.
- `cargo fmt -p fnp-python -- --check` fails on broad pre-existing crate drift.
  The touched Cholesky hunk was manually aligned with rustfmt's suggested
  formatting.
- `ubs` over changed files completed nonzero with broad existing findings in
  `fnp-python`, not a Cholesky-specific finding.
- A broad `cargo test -p fnp-python cholesky -- --nocapture` attempt was
  blocked by unrelated lib-test compile errors in
  `spacing/sign/nextafter/hypot/logaddexp` test call sites.

## Retry Predicate

Do not retry Python stacked Cholesky via scalar per-lane Rust micro-tuning for
4x4..64x64 until fresh evidence shows this delegate path has become a material
loss. A future retry should target a lower-overhead Python trampoline or a true
in-extension generated/LAPACK route that beats direct NumPy despite wrapper
overhead, and should keep only if same-worker proof clears a material win.
