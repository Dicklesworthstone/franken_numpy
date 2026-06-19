# 2026-06-19 ufunc extract verify slice

Scope: `franken_numpy-ixs5y.259` verification of the recent
`franken_numpy-ixs5y.244` code-first `fnp-ufunc` `extract` optimization.

Decision: reject and revert the parallel F64 extract masked-gather fast path.

Artifacts:
- `extract_guard.txt`: pre-revert remote golden guard for the landed candidate; passed.
- `criterion_extract_local.txt`: same-host local Criterion timing for the candidate.
- `numpy_extract_local.txt`: first local NumPy timing run for the same values and mask.
- `numpy_extract_local_rerun.txt`: longer local NumPy timing with GC disabled; used for the decision ratios.
- `extract_boolean_index_guards_post_revert.txt`: rejected Cargo invocation with two test filters; retained as a command correction artifact.
- `extract_guard_post_revert.txt`: post-revert remote extract golden guard; passed.
- `boolean_index_guard_post_revert.txt`: post-revert remote boolean-index golden guard; passed.
- `criterion_extract_local_post_revert.txt`: same-host local Criterion timing for the final serial path.
- `cargo_check_fnp_ufunc_post_revert.txt`: `rch exec -- cargo check -p fnp-ufunc --all-targets`; passed.
- `cargo_clippy_fnp_ufunc_post_revert.txt`: `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`; passed.
- `cargo_fmt_check.txt`: workspace `cargo fmt --check`; failed on broad pre-existing format drift outside this slice.
- `cargo_fmt_fnp_ufunc_check.txt`: package-scoped fmt check; failed on pre-existing untouched regions.
- `rustfmt_fnp_ufunc_lib_check.txt`: invalid direct rustfmt attempt without Cargo's Rust 2024 edition context.
- `git_diff_check.txt`: `git diff --check`; passed.
- `ubs_changed_files.txt`: UBS changed-file scan; nonzero on pre-existing broad `fnp-ufunc` inventory, not on the extract revert itself.

Same-host ratio vs NumPy, using `numpy_extract_local_rerun.txt`:

| Code state | Size | FNP median | NumPy median | FNP / NumPy | Verdict |
|---|---:|---:|---:|---:|---|
| Parallel candidate | 100,000 | 275.46 us | 126.540 us | 2.18x slower | Reject |
| Parallel candidate | 1,000,000 | 668.54 us | 547.298 us | 1.22x slower | Reject |
| Post-revert serial path | 100,000 | 79.896 us | 126.540 us | 1.58x faster | Final code |
| Post-revert serial path | 1,000,000 | 951.42 us | 547.298 us | 1.74x slower | Open gap |

The candidate was also 3.45x slower than the final serial path at 100k.
Although it was faster than serial at 1M, it still lost to NumPy and failed the
realistic mixed-size gate, so the production fast path was removed.
