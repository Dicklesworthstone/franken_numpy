# Dependency Upgrade Log

**Date:** 2026-02-20  |  **Project:** FrankenNumPy  |  **Language:** Rust

## Summary
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
