# Fuzzing FrankenNumPy

The workspace ships **7 fuzz crates** with **27 fuzz targets** and **~200 curated seed corpus files** as of 2026-05-16. Every fuzz crate is excluded from the main workspace (see `Cargo.toml` `[workspace] exclude`) so normal `cargo` commands don't pull in `libfuzzer-sys`.

## Prerequisites

```bash
cargo install cargo-fuzz
```

The fuzz crates require nightly Rust (matching `rust-toolchain.toml` / `env.RUST_TOOLCHAIN` in CI). The host workspace already pins nightly, so no extra setup is needed.

## Fuzz crate inventory

| Crate | Path | Targets |
|---|---|---|
| `fnp-dtype` | `crates/fnp-dtype/fuzz` | `fuzz_dtype_parse`, `fuzz_min_scalar_type`, `fuzz_can_cast`, `fuzz_result_type` |
| `fnp-io` | `crates/fnp-io/fuzz` | `fuzz_npy`, `fuzz_npz`, `fuzz_load_auto`, `fuzz_header`, `fuzz_fromstring`, `fuzz_loadtxt`, `fuzz_fromfile` |
| `fnp-iter` | `crates/fnp-iter/fuzz` | `fuzz_ndindex`, `fuzz_flatiter_indices`, `fuzz_nditer_plan`, `fuzz_transfer_class` |
| `fnp-linalg` | `crates/fnp-linalg/fuzz` | `fuzz_cholesky_nxn`, `fuzz_det_nxn`, `fuzz_qr_mxn` |
| `fnp-ndarray` | `crates/fnp-ndarray/fuzz` | `fuzz_broadcast_shape`, `fuzz_fix_unknown_dim`, `fuzz_as_strided`, `fuzz_sliding_window` |
| `fnp-random` | `crates/fnp-random/fuzz` | `fuzz_from_u64_seed`, `fuzz_seed_sequence` |
| `fnp-ufunc` | `crates/fnp-ufunc/fuzz` | `fuzz_parse_gufunc_signature`, `fuzz_datetime_unit_parse`, `fuzz_parse_fixed_signature` |

## Running a target

```bash
cd crates/fnp-io/fuzz
cargo +nightly fuzz run fuzz_npy
```

Add `-- -max_total_time=300` to bound the run (5 minutes). Crashes land in `artifacts/<target>/crash-*` and can be reproduced via:

```bash
cargo +nightly fuzz run fuzz_npy artifacts/fuzz_npy/crash-<hash>
```

## Seed corpus convention

Curated seeds live under `<crate>/fuzz/corpus/<target>/seed_*`. The repo's `.gitignore` exempts `seed_*` files (auto-generated hash-named files are gitignored, but hand-authored seeds are tracked).

To add a seed:

```bash
# Drop bytes into <target>'s corpus dir with a descriptive name.
echo -n '<binary payload>' > crates/fnp-io/fuzz/corpus/fuzz_npy/seed_my_case
```

## CI integration

CI does not run fuzzing on every PR; fuzz harness compile-checking is implicit in `cargo check --workspace --all-targets` because the corpus-bearing harness still has to build. Schedule a separate workflow if you want recurring coverage runs.

## Bead trail of the 2026-05 fuzz expansion

Search `.beads/issues.jsonl` for the seeding wave: `62oir`, `aaq0g`, `s46p2`, `8fftx`, `cv45i`, `i8ipt`, `y3dhc`, `diqz3`, `m5y8s`. Each bead's close-reason in the JSONL lists the exact seed counts and target families that bead touched.
