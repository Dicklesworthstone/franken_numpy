# franken_numpy-gnhuo proof

## Target

- Bead: `franken_numpy-gnhuo`
- Crate: `fnp-linalg`
- Function: `inv_nxn`
- Lever: reuse the existing row-major multi-RHS LU solver in `inv_nxn` instead of the column-strided `lu_solve_multi` wrapper.

## Profile Evidence

Remote Criterion profile through `rch` selected `inv_nxn/size/256` as the slowest `fnp-linalg` workload:

```text
RCH_FORCE_REMOTE=1 ... rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg -- --warm-up-time 1 --measurement-time 2 --sample-size 10
inv_nxn/size/256: [23.502 ms 27.746 ms 31.655 ms]
```

The prior solver path solved each inverse column through row-major RHS/output buffers with stride `m`. The replacement calls the existing `lu_forward_back_multi`, which processes every RHS column for a row contiguously.

## Benchmark Result

Remote after-benchmark:

```text
RCH_FORCE_REMOTE=1 ... rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg -- inv_nxn/size/256 --warm-up-time 1 --measurement-time 2 --sample-size 10
inv_nxn/size/256: [5.2308 ms 5.5585 ms 5.9170 ms]
```

Median estimate improved from `27.746 ms` to `5.5585 ms`, about `5.0x` faster. The two runs landed on different RCH workers, so the exact ratio is worker-sensitive; the win is large enough to clear the score gate despite that noise.

Score: Impact 5 x Confidence 5 / Effort 1 = 25.0.

## Isomorphism

- LU factorization and pivot selection are unchanged.
- RHS is still an identity matrix in row-major order.
- `lu_forward_back_multi` applies the same permutation, forward substitution, and backward substitution equations as the old per-column solve.
- For each output scalar, the floating-point accumulation order over `j` is unchanged; only independent RHS columns are interleaved by row.
- Ordering, tie-breaking, and RNG behavior are not part of this path.
- Singular, NaN, and infinity behavior is covered by the focused `inv_nxn` tests below.

Golden-output source hashes captured after validation:

```text
7663f23698466fe726f4bb5d40708b7893ef43d1c14c9630c4d6a6fc6f2b0572  crates/fnp-linalg/src/lib.rs
bb2db1a2e56fd2f2c240779a4b18253f5e4b75078b30c113645583616f6a579c  crates/fnp-linalg/tests/golden_linalg.rs
```

## Validation

All cargo commands were run through `rch` with `RCH_FORCE_REMOTE=1`.

```text
cargo test -p fnp-linalg inv_nxn -- --nocapture
PASS: 6 focused tests

cargo test -p fnp-linalg --test golden_linalg -- --nocapture
PASS: 19 golden tests

cargo check -p fnp-linalg --all-targets
PASS

cargo clippy -p fnp-linalg --all-targets -- -D warnings
PASS

rustfmt --edition 2024 --check crates/fnp-linalg/src/lib.rs
PASS
```

`ubs crates/fnp-linalg/src/lib.rs` remained nonzero due to pre-existing broad crate findings in test/oracle code and heuristic false positives; it did not report a new issue at the changed line.
