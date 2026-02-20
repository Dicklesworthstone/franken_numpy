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

## Validation

- `cargo check --workspace --all-targets --all-features` — Pass
- `cargo clippy --workspace --all-targets -- -D warnings` — Pass (zero warnings)
- `cargo fmt --check` — Pass
- `cargo test --workspace` — 44 pass, 2 pre-existing failures (RaptorQ artifact staleness, unrelated to upgrades)
