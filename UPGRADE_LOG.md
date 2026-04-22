# Dependency Upgrade Log

## 2026-04-22 Session (Clawdstein-libupdater-franken_numpy)

**Date:** 2026-04-22  |  **Project:** franken_numpy  |  **Language:** Rust
**Agent:** Clawdstein-libupdater-franken_numpy

### Summary (this session)

- **Updated (pre-session, separate commit):** asupersync 0.3.0 -> 0.3.1
- **Updated (this session):** _TBD_
- **Skipped (already latest):** half 2.7.1, bytemuck 1.25.0, serde 1.0.228, serde_json 1.0.149, base64 0.22.1, serde_yaml_ng 0.10.0
- **Failed:** _TBD_
- **Needs attention:** _TBD_

### Asupersync bump (separate commit, aadd732)

- `crates/fnp-runtime/Cargo.toml`: asupersync 0.3.0 -> 0.3.1 (feature-gated, optional, default-features = false)
- `crates/fnp-conformance/Cargo.toml`: asupersync 0.3.0 -> 0.3.1 (direct dep used by conformance harness)
- Cargo.lock: companion crates `asupersync-macros`, `franken-decision`, `franken-evidence`, `franken-kernel` all moved 0.3.0 -> 0.3.1.
- asupersync 0.3.1 was published to crates.io on 2026-04-21 (~3h before this session); patch release with no API changes required.
- Verified: `cargo check -p fnp-runtime --features asupersync --all-targets` and `cargo check -p fnp-conformance --all-targets` both pass via `rch exec`.

### Updates (this session)

#### flate2: 1.0.35 -> 1.1.9 (fnp-io)

- **Research:** flate2 1.1.x moved from C-bindings (cloudflare-zlib-sys, libz-rs-sys) to the pure-Rust `zlib-rs` backend; MSRV bumped to 1.67. Public API of `DeflateEncoder`, `DeflateDecoder`, `GzDecoder`, etc. unchanged. Minor additions: `(de)compress_uninit` taking `MaybeUninit<u8>`, `Clone` on error types. No breaking changes for fnp-io's usage (`DeflateDecoder` / `DeflateEncoder` from `flate2::read`/`flate2::write`).
- **Cargo.lock:** already at 1.1.9 (transitive refresh from a prior session pulled it forward; manifest caught up here).
- **Verified:** `cargo check -p fnp-io --all-targets` pass. `cargo test -p fnp-io` 222/222 pass.

#### criterion: 0.5.1 -> 0.8.2 (fnp-conformance, dev-dependency)

- **Research:** Official CHANGELOG only documents up to 0.7.0; no new breaking changes recorded beyond 0.6.0. 0.6.0 removed the `real_blackbox` feature flag (no-op since then) and bumped MSRV to 1.80, with `criterion::black_box` deprecated in favor of `std::hint::black_box()`. All re-exports we rely on (`Criterion`, `BenchmarkId`, `criterion_group!`, `criterion_main!`) are preserved in 0.8.2 per docs.rs.
- **Lockfile churn:** Adds `alloca 0.4.0`, `cc 1.2.60`, `find-msvc-tools 0.1.9`, `itertools 0.13.0`, `page_size 0.6.0`, `shlex 1.3.0`, plus winapi family. Removes `is-terminal` (pulled by older criterion-plot). criterion-plot 0.5.0 -> 0.8.2.
- **Code change:** `crates/fnp-conformance/benches/criterion_core_ops.rs` — switch `use criterion::{..., black_box, ...}` to `use std::hint::black_box;` (11 call sites, all in one file) to eliminate the deprecation warnings that would otherwise become hard errors in a future major.
- **Verified:** `cargo check -p fnp-conformance --all-targets` + `--benches` both clean; 0 warnings from criterion. (Not running `cargo bench` because nothing broke at compile-time and the bench harness is unchanged beyond the import.)

#### sha2: 0.10.9 -> 0.11.0 (fnp-conformance)

- **Research:** sha2 0.11 updates to `digest` 0.11 and converts hash types (`Sha256`, `Sha512`, ...) from type aliases to newtype structs. Module reorg: `compress256`/`compress512` moved to `block_api`. Features `asm`/`asm-aarch64`/`loongarch64_asm`/`compress`/`soft`/`force-soft-compact`/`std` removed; new `alloc` feature. MSRV bumped to 1.85 (we're on edition 2024/nightly — fine).
- **fnp-conformance usage audit:** only `use sha2::{Digest, Sha256};` + `Sha256::digest(bytes)` / `Sha256::new()` / `hasher.update(...)` / `hasher.finalize()`. These APIs are preserved in 0.11 via the `Digest` trait; the newtype conversion does not affect this call style. Transitive removals: `block-buffer`, `cpufeatures`, `crypto-common`, `digest 0.10.x`, `generic-array 0.14.x` (all replaced by digest 0.11 / hybrid-array internals).
- **Verified:** `cargo check -p fnp-conformance --all-targets` passes cleanly. Targeted `cargo test -p fnp-conformance --lib raptorq` (4 tests that exercise the sha2 code paths via `raptorq_artifacts::sha256_hex`) passes 4/4. A broader `cargo test -p fnp-conformance --lib` shows 3 pre-existing failures in `test_contracts::*test_contract_suite_is_green` and `tests::core_suites_are_green` — all complaining about `linalg_differential_cases invalid fixture id linalg_cholesky_solve_identity_L_returns_b`, which is a fixture-registry / data-file issue entirely unrelated to sha2. Confirmed by grep: the failing ID is only defined in `fixtures/linalg_differential_cases.json` and is not registered in the linalg fixture ID enum — pre-existing breakage owned by another agent / the linalg team.

---

## 2026-02-20 Session (legacy)

**Date:** 2026-02-20  |  **Project:** FrankenNumPy  |  **Language:** Rust

### Summary
- **Updated:** 5 direct + 5 transitive  |  **Skipped:** 1 (base64, already latest)  |  **Failed:** 0  |  **Needs attention:** 1 (serde_yaml deprecated)

## Toolchain

### Rust nightly
- **Before:** `channel = "nightly"` → rustc 1.95.0-nightly (7f99507f5 2026-02-19)
- **After:** `channel = "nightly-2026-02-20"` → rustc 1.95.0-nightly (7f99507f5 2026-02-19)
- Pinned to specific date for reproducibility

## Direct Dependency Updates

### serde: 1.0.218 → 1.0.228
- **Breaking:** None (semver-compatible patch)
- **Tests:** Pass (44/44 non-preexisting)

### serde_json: 1.0.139 → 1.0.149
- **Breaking:** None (semver-compatible patch)
- **Tests:** Pass

### sha2: 0.10.8 → 0.10.9
- **Breaking:** None (patch release). Note: 0.11.0-rc.5 exists as pre-release — skipped per version rules.
- **Tests:** Pass

### asupersync: 0.2.0 → 0.2.5
- **Crates:** fnp-conformance, fnp-runtime
- **Breaking:** None (semver-compatible patch)
- **Tests:** Pass

### ftui: 0.2.0 → 0.2.1
- **Crate:** fnp-runtime
- **Breaking:** None (patch release)
- **Tests:** Pass

## Transitive Dependencies (via cargo update)

| Crate | Old | New |
|-------|-----|-----|
| anyhow | 1.0.101 | 1.0.102 |
| bitflags | 2.10.0 | 2.11.0 |
| bumpalo | 3.19.1 | 3.20.2 |
| syn | 2.0.115 | 2.0.117 |
| unicode-ident | 1.0.23 | 1.0.24 |

## Already Latest

### base64: 0.22.1
- No update available.

## Needs Attention

### serde_yaml: 0.9.34 (DEPRECATED)
- **Issue:** Crate is deprecated/unmaintained since March 2024
- **Replacements:** `serde_yml` or `serde_yaml_ng` (both are maintained forks)
- **Action:** Flagged for user decision — migration would be a minor API change

## 2026-03-31 - serde_yaml migration closure

### Summary
- **Completed:** direct migration to `serde_yaml_ng` in `fnp-conformance`
- **Status:** no remaining live `serde_yaml` dependency in workspace manifests or lockfile

### Notes
- `crates/fnp-conformance/Cargo.toml` now uses `serde_yaml_ng = "0.10.0"`.
- YAML parsing and serialization call sites use `serde_yaml_ng::*`.
- The earlier "Needs Attention" entry remains as historical context from the original upgrade sweep.

## 2026-03-21 - Bug Fixes and Parity Improvements

### Summary
- **Fixed:** 4 logic bugs + 3 test failures
- **Coverage:** Improved parity with NumPy for reduction shapes and 0D array rejection

### Fixes

#### fnp-io: Relaxed header validation
- **Issue:** `validate_required_header_keys` required exactly 3 keys, failing on valid NumPy files with extra metadata.
- **Fix:** Changed to require *at least* the 3 mandatory keys.
- **Tests:** Updated `load_structured_accepts_extra_header_keys` and `npy_header_parser_accepts_extra_keys`.

#### fnp-linalg: SVD non-convergence detection
- **Issue:** `svd_bidiag_full` silently ignored non-convergence if the maximum iteration budget was exceeded.
- **Fix:** Added a convergence check that returns `Err(LinAlgError::SvdNonConvergence)`.

#### fnp-ufunc: Reduction shape correction
- **Issue:** `any`, `all`, `mean`, `sum`, etc., incorrectly produced 1D arrays of shape `[1]` when reducing a 1D array along its only axis.
- **Fix:** Removed the logic forcing a `1` into empty output shapes, allowing correct 0D (scalar) results.

#### fnp-ufunc: where_nonzero parity
- **Issue:** `where_nonzero` (and `np.nonzero`) accepted 0D arrays, which NumPy explicitly rejects.
- **Fix:** Added a check to reject 0D arrays with a ValueError-style message matching NumPy.

## 2026-03-21 (Phase 2) - DType, Datetime, and Conformance Log Fixes

### Summary
- **Fixed:** 6 logic bugs + 2 test failures (including a flaky concurrency bug)
- **Coverage:** Improved `can_cast` rules, `busday_offset` weekend behaviors, and thread-safe logging.

### Fixes

#### fnp-dtype: Complex classification and casting rules
- **Issue:** `is_float` incorrectly excluded complex types, and `can_cast_same_kind` was too permissive for string/datetime casting.
- **Fix:** Overhauled `is_float` to include complex, corrected `item_size` for variable-length types (Structured, Str), and enforced strict NumPy hierarchical casting (`bool` < `int` < `float` < `complex`).

#### fnp-ufunc: busday_offset weekend roll
- **Issue:** `busday_offset` incorrectly rejected all weekend inputs under default conditions, violating golden test expectations which assume rolling behavior.
- **Fix:** Implemented direction-dependent rolling (forward for positive offsets, backward for negative offsets) matching NumPy's implicit behaviors in the golden tests.

#### fnp-ufunc: busday_count length validation
- **Issue:** `busday_count` explicitly verified exact length matches, failing on correctly broadcastable shapes.
- **Fix:** Removed strict length parity check to fully support NumPy-style broadcasting in `busday_count`.

#### fnp-conformance: Concurrent log corruption
- **Issue:** `SHAPE_STRIDE_LOG_PATH` and other logs were prone to torn writes and JSON corruption under `cargo test` concurrency.
- **Fix:** Introduced a global `FILE_LOG_MUTEX` to strictly serialize all `maybe_append_` file IO operations.

### Validation
- `cargo check --workspace --all-targets` — Pass
- `cargo test --workspace` — Pass (All 1600+ tests green)
