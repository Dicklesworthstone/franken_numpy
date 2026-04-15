# Working Tree Cleanup Triage

Date: 2026-04-15

Scope:
- Root-level untracked regular files only.
- Excludes tracked files, directories (`target/`, `.beads/`, virtualenvs), and non-root paths.

Current state:
- 86 root regular files are not tracked.
- 32 are currently visible as `??` in `git status`.
- 54 are already ignored by the existing root `.gitignore`.
- No files were deleted, moved, or edited during this triage pass.

## Summary

| Category | Count | Default disposition | Notes |
|----------|-------|---------------------|-------|
| tooling-output | 23 | delete (pending approval) | Already ignored; these are transient lint/scan captures. |
| compiled-binary | 26 | delete (pending approval) | Already ignored; these are extensionless executables left in the repo root. |
| fixed-bug-repro | 32 | delete (pending approval) | One-off repro sources/fixtures in the repo root, outside maintained `crates/` or `tests/` layouts. |
| unclear | 5 | review for promotion or delete | Ad hoc helper scripts / one-shot mutation programs; inspect once before deletion. |

Observations:
- The current `.gitignore` already covers the transient output files, root compiled binaries, and the ad hoc helper scripts listed below.
- The remaining visible `test_*.rs`, `test_*.py`, `test_*.c`, and `test_*.csv` files are not ignored by design, which is good: blindly ignoring source-like filenames would hide future legitimate files.
- The cleanup still needs explicit user approval before any deletion.

## Per-File Manifest

| File | Category | Disposition | Rationale |
|------|----------|-------------|-----------|
| clippy_cast.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_casts_all.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_conformance.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_deep.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_indexing.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_indexing_new.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_issues.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_json.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_new.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_out.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_out2.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_out3.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_out4.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_pedantic.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_pedantic_2.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_pedantic_all.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_pedantic_new.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_ptr.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_suspicious.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| clippy_unwrap.txt | tooling-output | delete (pending approval) | Already ignored transient Clippy output; keeping it in the repo root adds noise but no durable value. |
| deep_clippy.txt | tooling-output | delete (pending approval) | Already ignored transient lint output; keeping it in the repo root adds noise but no durable value. |
| rust-bug-scan.txt | tooling-output | delete (pending approval) | Already ignored transient bug-scan output; keeping it in the repo root adds noise but no durable value. |
| ubs_out.txt | tooling-output | delete (pending approval) | Already ignored transient UBS output; keeping it in the repo root adds noise but no durable value. |
| a.out | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| patch_oracle | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_arange | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_contiguous | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_digitize | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_digitize_nan | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_div | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_idx | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_max | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_max_rows | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_min | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_multiple | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_nan_bin | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_nan_cast | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_nan_max | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_nan_min | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_nat | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_neg_zero | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_overlap | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_pad | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_parse_tab | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_rem_panic2 | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_shift | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_shift2 | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_solve | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_split | compiled-binary | delete (pending approval) | Already ignored build residue; extensionless executable in the repo root has no source-of-truth value. |
| test_cast_str.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_cow_bug.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_digitize.py | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_f16.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_fnp_digitize.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_idx.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_max_rows.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_multiple.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_nan_max.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_ncx2.py | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_ncx2_fnp.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_nearest.py | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_nearest_method.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_npz.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_parse_json.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_parse_tab.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_polyder.py | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_polydiv.py | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_polydiv2.py | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_polydiv3.py | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_ptrs.c | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_put_along.py | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_put_along.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_rem_panic.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_rem_panic2.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_reshape.py | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_roots.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_skip.csv | fixed-bug-repro | delete (pending approval) | Supporting fixture for a root repro script; keep only if intentionally promoted into permanent tests. |
| test_skip.py | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_skip_fnp.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_slice_reverse.rs | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| test_trace.py | fixed-bug-repro | delete (pending approval) | One-off root repro source outside maintained `crates/` or `tests/` layout; keep only if intentionally promoted into permanent tests. |
| apply_fixes.py | unclear | review for promotion or delete | Ad hoc rewrite script that edits a specific file with brittle string replacements; not obviously reusable as tracked tooling. |
| fix_clippy.py | unclear | review for promotion or delete | Ad hoc local diagnostic script for one crate and two lint codes; useful only if intentionally promoted into `scripts/dev/`. |
| fix_product.py | unclear | review for promotion or delete | Ad hoc regex rewrite script spanning specific files and patterns; not obviously safe or reusable as tracked tooling. |
| get_context.py | unclear | review for promotion or delete | Local line-dump helper tied to hard-coded file paths and line numbers; not obviously reusable as tracked tooling. |
| patch_oracle.rs | unclear | review for promotion or delete | Single-target mutation program that injects a debug `println!` into one source file; not reusable as-is. |

## Recommended Next Step

1. User approves category-level dispositions:
   - `tooling-output`: delete
   - `compiled-binary`: delete
   - `fixed-bug-repro`: delete unless a specific file should be promoted into maintained tests
   - `unclear`: decide case-by-case whether to promote into `scripts/dev/` or delete
2. After approval, delete only the approved files, then re-run:
   - `git status`
   - `git status --ignored`
   - `CARGO_INCREMENTAL=0 rch exec -- cargo check --workspace --all-targets`
   - `CARGO_INCREMENTAL=0 rch exec -- cargo test --workspace`
