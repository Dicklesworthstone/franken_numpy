# BlackThrush Safe-Rust GEMM Row Block Proof

Bead: `franken_numpy-ixs5y.1`
Date: 2026-06-02
Agent: BlackThrush

## Target

Profile target: `fnp-ufunc` dense F64 `UFuncArray::matmul`, Criterion workload
`core_ops/matmul_256x256_by_256x256`.

Vs-upstream gap: existing cross-engine report records `matmul_medium` at FNP
p50 `39.434 us` vs NumPy/OpenBLAS p50 `11.602 us`, ratio `3.3989x` slower.

Alien primitive: communication-avoiding/cache-local dense linear algebra. The
first safe-Rust lever is a row-blocked BLAS-3 inner loop that reuses each RHS row
across a fixed output-row block while preserving the scalar accumulation order
for every output cell.

Fallback: restore the original scalar `i, k, j` loop.

## Benchmark

Remote baseline, original loop:

- Command: `rch exec -- cargo bench -p fnp-conformance --bench criterion_core_ops -- matmul_256x256_by_256x256 --sample-size 12 --measurement-time 5 --warm-up-time 2`
- Worker: `vmi1153651`
- Result: `6.3284 ms` center, CI `[6.0293 ms, 6.5853 ms]`

Remote after, row block `ROW_BLOCK = 8`:

- Command: `rch exec -- cargo bench -p fnp-conformance --bench criterion_core_ops -- matmul_256x256_by_256x256 --sample-size 12 --measurement-time 5 --warm-up-time 2`
- Worker: `vmi1153651`
- Result: `4.6882 ms` center, CI `[4.4164 ms, 5.0392 ms]`

Delta: `25.9%` median reduction. Confidence intervals exclude overlap.

Score: Impact `4` x Confidence `5` / Effort `2` = `10.0`.

## Isomorphism

The original loop applies, for each output cell, additions in `k = 0..K-1`
order:

`out[i, j] += lhs[i, k] * rhs[k, j]`

The row-blocked loop only changes scheduling across rows. For a fixed
`(i, j)`, it still applies the same products in the same `k` order, with the
same initial `+0.0` output value and no fused multiply-add request.

Observable behavior preserved:

- Shape and dtype handling remain in the existing `matmul` wrapper.
- Error behavior for dimension mismatch and broadcast legality is unchanged.
- No RNG, pivoting, ordering tie-breaks, or external BLAS/LAPACK calls are
  introduced.
- No `unsafe` is introduced; `#![forbid(unsafe_code)]` remains intact.

Golden output SHA-256:

- Test: `matmul_row_blocking_preserves_scalar_accumulation_bits`
- Payload: little-endian `f64` output bytes for deterministic `9x7 @ 7x6`
  values including `-0.0` inputs.
- SHA-256: `dd19822c9ba24f9a17d5c6f3e112451265c4ac4659cd5d8e78de97c77f7c2df7`

## Validation

- `rch exec -- cargo test -p fnp-ufunc matmul --lib -- --nocapture`
  - Worker: `vmi1149989`
  - Result: `19 passed`
- `rch exec -- cargo test -p fnp-ufunc matmul_row_blocking_preserves_scalar_accumulation_bits --lib -- --nocapture`
  - Worker: `vmi1293453`
  - Result: `1 passed`
- `rch exec -- cargo check -p fnp-ufunc --all-targets`
  - Worker: `vmi1153651`
  - Result: passed
- `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
  - Worker: `vmi1153651`
  - Result: passed
- `rustfmt --edition 2024 --check crates/fnp-ufunc/src/lib.rs`
  - Result: passed
- `ubs crates/fnp-ufunc/src/lib.rs`
  - Result: exit `1` on pre-existing monolithic `fnp-ufunc` scanner inventory.
  - Clean sections: formatting, clippy, cargo check, tests build, dependency hygiene.
  - Final counts after removing new unwraps from the proof test: critical `376`,
    warnings `12120`, info `1825`.
