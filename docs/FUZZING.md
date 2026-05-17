# Fuzzing FrankenNumPy

The workspace ships **7 fuzz crates** with **27 fuzz targets** and **200 curated seed corpus files** (re-verified 2026-05-17 via `find crates/*/fuzz/corpus -name 'seed_*' -type f | wc -l`, unchanged from the 2026-05-16 baseline). Every fuzz crate is excluded from the main workspace (see `Cargo.toml` `[workspace] exclude`) so normal `cargo` commands don't pull in `libfuzzer-sys`.

## Prerequisites

```bash
cargo install cargo-fuzz
```

The fuzz crates require nightly Rust pinned to `nightly-2026-02-20` (matching `rust-toolchain.toml` / `env.RUST_TOOLCHAIN` in `.github/workflows/ci.yml`). The host workspace already pins it, so no extra setup is needed — but if you see a `libfuzzer-sys` compile error after changing toolchains, re-pin and rebuild.

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
cargo fuzz run fuzz_npy
```

Each fuzz crate ships its own `rust-toolchain.toml` mirroring `/rust-toolchain.toml`, so the workspace nightly pin (`nightly-2026-02-20`) applies automatically — no `+nightly` suffix needed. Add `-- -max_total_time=300` to bound the run (5 minutes). Crashes land in `artifacts/<target>/crash-*` and can be reproduced via:

```bash
cargo fuzz run fuzz_npy artifacts/fuzz_npy/crash-<hash>
```

When updating the workspace nightly, bump all 8 `rust-toolchain.toml` files together (the root + 7 per-fuzz copies) plus the matching `RUST_TOOLCHAIN` env var in `.github/workflows/ci.yml` and the README/AGENTS.md mentions.

## Seed corpus convention

Curated seeds live under `<crate>/fuzz/corpus/<target>/seed_*`. The repo's `.gitignore` exempts `seed_*` files (auto-generated hash-named files are gitignored, but hand-authored seeds are tracked).

To add a seed:

```bash
# Drop bytes into <target>'s corpus dir with a descriptive name.
echo -n '<binary payload>' > crates/fnp-io/fuzz/corpus/fuzz_npy/seed_my_case
```

## CI integration

CI does not run fuzzing on every PR; fuzz harness compile-checking is implicit in `cargo check --workspace --all-targets` because the corpus-bearing harness still has to build. Schedule a separate workflow if you want recurring coverage runs.

## Why each fuzz crate's `Cargo.toml` repeats fields literally

Each fuzz crate has its own `[workspace]` block (empty), making it a separate
1-package workspace that is excluded from the root workspace via the parent
`Cargo.toml`'s `[workspace] exclude` list. That isolation is deliberate — the
fuzz crates pull in `libfuzzer-sys`, which we don't want bleeding into normal
`cargo` invocations. The cost: the fuzz `Cargo.toml` files cannot use
`.workspace = true` inheritance and must declare `edition = "2024"` (and other
package fields) literally. Same reason applies to the per-fuzz
`rust-toolchain.toml` shipped alongside.

## Where to record findings

A fuzz crash that exposes a real bug becomes a bead. The raw crash inputs that `cargo-fuzz` writes land under `crates/<crate>/fuzz/artifacts/<target>/crash-*` — copy the relevant crash bytes into a workspace-root `artifacts/<bead-id>/` directory (or attach them to the bead's `--description`) so the reproducer is permanent rather than living in a gitignored cargo-fuzz directory. A fuzz finding that exposes intentional or parity-debt divergence from NumPy belongs in [`DIVERGENCES.md`](DIVERGENCES.md) — that ledger is the machine-readable handoff point for diagnostic gates and accepts both `intentional` and `parity_debt` rows.

## Bead trail of the 2026-05 fuzz expansion

Search `.beads/issues.jsonl` for the seeding wave: `62oir` (fnp-linalg/random fuzz infra), `aaq0g` (32 seeds for new fuzz crates), `s46p2` (45 seeds for fnp-dtype/io targets), `8fftx` (19 Arbitrary-format seeds), `cv45i` (31 seeds for fnp-ufunc string parsers), `i8ipt` (15 seeds for fnp-ndarray broadcast/reshape), `y3dhc` (46 NPY/NPZ binary seeds), `diqz3` (11 seeds for fnp-iter/fuzz_ndindex), `m5y8s` (15 seeds for fnp-ufunc/fuzz_parse_fixed_signature). All 9 verified present + closed in the JSONL as of 2026-05-17; each bead's close-reason lists the exact seed counts and target families that bead touched.
