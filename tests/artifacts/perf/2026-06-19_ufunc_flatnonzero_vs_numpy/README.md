# 2026-06-19 ufunc flatnonzero verify slice

Scope: `franken_numpy-ixs5y.260` verification of the recent
`franken_numpy-ixs5y.245` code-first `fnp-ufunc` `flatnonzero` optimization.

Decision: reject and revert the parallel F64 flatnonzero per-chunk index-gather
fast path. Keep the existing serial exact int64 sidecar export path.

Artifacts:
- `flatnonzero_guard.txt`: pre-revert remote golden guard for the landed candidate; passed.
- `criterion_flatnonzero_candidate.txt`: remote `vmi1227854` Criterion timing; retained as routing evidence only because NumPy could not be run on the same worker through `rch`.
- `criterion_flatnonzero_local_candidate.txt`: same-host local Criterion timing for the candidate; used for decision ratios.
- `numpy_flatnonzero_local.txt`: local NumPy timing for the same sparse F64 fixture; used for decision ratios.
- `flatnonzero_guard_post_revert.txt`: post-revert remote golden guard; passed.
- `criterion_flatnonzero_local_post_revert.txt`: same-host local Criterion timing for the final serial path.
- `cargo_check_fnp_ufunc_post_revert.txt`: `rch exec -- cargo check -p fnp-ufunc --all-targets`; passed.
- `cargo_clippy_fnp_ufunc_post_revert.txt`: `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`; passed.
- `cargo_fmt_check.txt`: workspace `cargo fmt --check`; failed on broad pre-existing format drift outside this slice.
- `cargo_fmt_fnp_ufunc_check.txt`: package-scoped fmt check; failed on pre-existing untouched regions.
- `git_diff_check.txt`: `git diff --check`; passed.
- `ubs_changed_files.txt`: UBS changed-file scan; nonzero on pre-existing broad `fnp-ufunc` inventory, not on the flatnonzero revert itself.

Same-host ratio vs NumPy, using `numpy_flatnonzero_local.txt`:

| Code state | Size | FNP median | NumPy median | FNP / NumPy | Verdict |
|---|---:|---:|---:|---:|---|
| Parallel candidate | 100,000 | 255.53 us | 239.794 us | 1.07x slower | Reject |
| Parallel candidate | 1,000,000 | 703.16 us | 2512.132 us | 0.280x, 3.57x faster | Reject despite win |
| Post-revert serial path | 100,000 | 74.662 us | 239.794 us | 0.311x, 3.21x faster | Final code |
| Post-revert serial path | 1,000,000 | 789.35 us | 2512.132 us | 0.314x, 3.18x faster | Final code |

The candidate won at 1M but regressed the 100k realistic workload and used an
allocation-heavy per-chunk `Vec<Vec<i64>>` shape. The production parallel branch
was removed; final code stays faster than NumPy on both measured rows.
