# Dependency Upgrade Log

**Date:** 2026-10-07 (latest pass); 2026-09-11
**Project:** franken_numpy
**Language:** Rust
**Manifest:** Cargo.toml / Cargo.lock

---

## 2026-10-07 — v0.4.0 dependency pass

| Metric | Count |
|--------|-------|
| **Updated (semver-compatible, `cargo update`)** | 33 lockfile entries |
| **Direct-dep major bumps** | 2 (`pyo3` 0.28.3 → 0.29.3, `ftui` 0.5.0 → 0.9.0) |
| **Skipped** | 2 (`asupersync` exact pin, `generic-array` transitive) |
| **Failed (rolled back)** | 0 |

### Direct-dependency major bumps

#### pyo3: 0.28.3 → 0.29.3 (workspace pin; consumer `fnp-python`)

- Clears RUSTSEC-2026-0176 (out-of-bounds read in `BoundListIterator` / `BoundTupleIterator`
  `nth` / `nth_back`, patched in 0.29.0) and RUSTSEC-2026-0177, both recorded as
  "Requires Attention" in the 2026-09-11 pass below.
- `cargo check -p fnp-python --all-targets` compiled with zero errors and zero warnings
  against 0.29.3; no source change was needed. The 0.29 changelog's removals (0.27
  deprecations, private FFI definitions, broken `From<Utf8Error> for PyErr`) are not used by
  this workspace.
- Source comments that cite pyo3 0.28.3 internals (`is_matching_endian`, `tuple_conversion!`
  vectorcall routing, `&str` → `PyString` per call, interpreter finalisation) are historical
  measurement notes. None of those behaviours is listed as changed in 0.29.0..0.29.3, and the
  crate-local endian-refusing `PyBuffer` shadow does not depend on pyo3's check.

#### ftui: 0.5.0 → 0.9.0 (`fnp-runtime`, optional `frankentui` feature, default off)

- `cargo clippy -p fnp-runtime --all-targets --all-features -- -D warnings` is clean on 0.9.0.
  The crate only names `ftui::{Buffer, Frame, Theme, TerminalCapabilities}` for typed
  capability snapshots.

### Semver-compatible updates (`cargo update`)

- `cc`: 1.4.5 → 1.6.0
- `cfg-if`: 1.0.4 → 1.0.5
- `clap` / `clap_builder`: 4.6.6 → 4.6.7; `clap_lex`: 1.1.0 → 1.1.1
- `crc32fast`: 1.5.1 → 1.5.2
- `ctutils`: 0.4.2 → 0.4.3
- `either`: 1.18.0 → 1.19.0
- `find-msvc-tools`: 0.1.12 → 0.1.14
- `insta`: 1.48.0 → 1.49.0
- `js-sys` / `web-sys`: 0.3.105 → 0.3.106
- `libc`: 0.2.189 → 0.2.190
- `lru`: 0.18.4 → 0.18.5
- `objc2`: 0.6.4 → 0.6.5
- `rustix`: 1.1.4 → 1.1.5
- `signal-hook`: 0.4.4 → 0.4.5
- `smallvec`: 1.16.1 → 1.16.2
- `syn`: 3.0.5 → 3.0.6
- `thiserror` / `thiserror-impl`: 2.0.20 → 2.0.21
- `tinyvec`: 1.13.2 → 1.13.3 (drops `tinyvec_macros`)
- `unicode-ident`: 1.0.24 → 1.0.26
- `wasm-bindgen` (+ `-macro`, `-macro-support`, `-shared`): 0.2.128 → 0.2.129;
  `wasm-bindgen-futures`: 0.4.78 → 0.4.79 (now pulls `tokio` 1.53.2 with `rt`/`time`, wasm32
  targets only via `asupersync`; not part of native builds)
- `zerocopy` / `zerocopy-derive`: 0.8.57 → 0.8.61
- `zeroize`: 1.9.0 → 1.9.1
- `zlib-rs`: 0.6.7 → 0.6.8

### Skipped

- `asupersync` `=0.5.0`: deliberate exact pin shared with the franken* constellation. Preserved.
- `generic-array` 0.14.7 (0.14.9 available): transitive, held by the digest/crypto tree.

**Gate:** `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and the
per-crate test suites (`cargo test -j2 -p <crate>` via rch). Results are recorded in the v0.4.0
release notes.

---

## 2026-09-11 — v0.3.0 dependency pass

## Summary

| Metric | Count |
|--------|-------|
| **Total dependencies** | 287 |
| **Updated** | 79 |
| **Skipped** | 4 |
| **Failed (rolled back)** | 0 |
| **Requires attention** | 1 |

---

## Successfully Updated

79 transitive and direct dependencies updated to latest semver-compatible versions via `cargo update`:

- `aho-corasick`: 1.1.4 → 1.1.5
- `anyhow`: 1.0.102 → 1.0.104
- `arc-swap`: 1.9.1 → 1.9.2
- `autocfg`: 1.5.0 → 1.5.1
- `base64`: 0.23.0 → 0.23.1
- `bitflags`: 2.13.1 → 2.13.2
- `block-buffer`: 0.12.0 → 0.12.1
- `bytes`: 1.11.1 → 1.12.1
- `cc`: 1.4.0 → 1.4.5
- `cfg_aliases`: 0.2.1 → 0.2.2
- `clap`: 4.6.1 → 4.6.6
- `clap_builder`: 4.6.0 → 4.6.6
- `cmov`: 0.5.3 → 0.5.4
- `console`: 0.16.3 → 0.16.6
- `cpufeatures`: 0.3.0 → 0.3.1
- `crc32fast`: 1.5.0 → 1.5.1
- `crossbeam-deque`: 0.8.6 → 0.8.8
- `crossbeam-epoch`: 0.9.18 → 0.9.21
- `crossbeam-queue`: 0.3.12 → 0.3.14
- `crossbeam-utils`: 0.8.21 → 0.8.23
- `crypto-common`: 0.2.1 → 0.2.2
- `data-encoding`: 2.11.0 → 2.11.1
- `digest`: 0.11.2 → 0.11.3
- `either`: 1.15.0 → 1.18.0
- `fastrand`: 2.4.1 → 2.5.0
- `find-msvc-tools`: 0.1.9 → 0.1.12
- `flate2`: 1.1.9 → 1.1.10
- `franken-decision`: 0.3.9 → 0.3.10
- `franken-evidence`: 0.3.9 → 0.3.10
- `franken-kernel`: 0.3.9 → 0.3.10
- `futures-core`: 0.3.32 → 0.3.34
- `futures-io`: 0.3.32 → 0.3.34
- `futures-task`: 0.3.32 → 0.3.34
- `futures-util`: 0.3.32 → 0.3.34
- `hashbrown`: 0.17.0 → 0.17.1
- `hermit-abi`: 0.5.2 → 0.5.3
- `hybrid-array`: 0.4.10 → 0.4.15
- `indexmap`: 2.14.0 → 2.14.2
- `insta`: 1.47.2 → 1.48.0
- `js-sys`: 0.3.95 → 0.3.105
- `libc`: 0.2.185 → 0.2.189
- `log`: 0.4.29 → 0.4.34
- `lru`: 0.18.2 → 0.18.4
- `miniz_oxide`: 0.8.9 → 0.9.1
- `nix`: 0.31.2 → 0.31.3
- `pin-project`: 1.1.11 → 1.1.13
- `pin-project-internal`: 1.1.11 → 1.1.13
- `portable-atomic`: 1.13.1 → 1.15.0
- `proc-macro2`: 1.0.106 → 1.0.107
- `quote`: 1.0.45 → 1.0.47
- `rand`: 0.8.6 / 0.9.4 → 0.8.8 / 0.9.5
- `regex`: 1.12.3 → 1.13.1
- `regex-automata`: 0.4.14 → 0.4.18
- `regex-syntax`: 0.8.10 → 0.8.11
- `rustc-hash`: 2.1.2 → 2.1.3
- `rustversion`: 1.0.22 → 1.0.23
- `simd-adler32`: 0.3.9 → 0.3.10
- `smallvec`: 1.15.1 → 1.16.1
- `socket2`: 0.6.3 → 0.6.5
- `syn`: 2.0.117 / 3.0.3 → 2.0.119 / 3.0.5
- `thiserror`: 2.0.18 → 2.0.20
- `thiserror-impl`: 2.0.18 → 2.0.20
- `tinyvec`: 1.12.0 → 1.13.2
- `typenum`: 1.20.0 → 1.20.1
- `wasip2`: 1.0.3+wasi-0.2.9 → 1.0.4+wasi-0.2.12
- `wasm-bindgen`: 0.2.118 → 0.2.128
- `wasm-bindgen-futures`: 0.4.68 → 0.4.78
- `wasm-bindgen-macro`: 0.2.118 → 0.2.128
- `wasm-bindgen-macro-support`: 0.2.118 → 0.2.128
- `wasm-bindgen-shared`: 0.2.118 → 0.2.128
- `web-sys`: 0.3.95 → 0.3.105
- `zerocopy`: 0.8.48 → 0.8.57
- `zerocopy-derive`: 0.8.48 → 0.8.57
- `zeroize`: 1.8.2 → 1.9.0
- `zeroize_derive`: 1.4.3 → 1.5.0
- `zlib-rs`: added 0.6.7
- `zmij`: 1.0.21 → 1.0.23

**Tests:** ✓ Passed (compiler check, clippy clean with -D warnings, conformance tests pass)

---

## Skipped

### asupersync: =0.3.9
**Reason:** Explicit exact pin in workspace (`version = "=0.3.9"`). Preserved.

### ftui: 0.5.0
**Reason:** Optional runtime dashboard dependency in `fnp-runtime`. Kept at 0.5.0.

### generic-array: 0.14.7
**Reason:** Transitive dependency pinned by downstream crypto/digest trees.

---

## Requires Attention

### pyo3: 0.28.3 → 0.29.2

**Issue:** Major version bump with breaking API changes across PySlice, PyAnyMethods, PyTuple constructors, and GIL bindings.
**Scope:** FrankenNumPy has over 1,800 PyO3 call sites across `fnp-python` (180k LOC).
**Security note:**
- RUSTSEC-2026-0176 (Out-of-bounds read in `nth`/`nth_back` for PyList/PyTuple iterators)
- RUSTSEC-2026-0177 (Missing `Sync` bound on `PyCFunction::new_closure` closures)
**Recommendation:** Upgrade pyo3 in a dedicated refactoring session with proper staging and testing.

---

## Post-Upgrade Checklist

- [x] All tests passing
- [x] No clippy warnings (`cargo clippy --workspace --all-targets -- -D warnings` passed)
- [x] Conformance suites passing
- [x] Security audit logged
