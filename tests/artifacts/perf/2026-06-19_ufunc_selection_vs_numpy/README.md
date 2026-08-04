# 2026-06-19 ufunc selection verify slice

Scope: `franken_numpy-ixs5y.249`, `fnp-ufunc` `compress(condition, axis=None)`.

Decision: reject and revert the parallel F64 compress bool-mask gather fast path.

Artifacts:
- `compress_guard.txt`: first remote guard attempt on `ovh-a`; failed because the guard compared `NaN` with `assert_eq!`.
- `compress_guard_after_fix.txt`: remote guard rerun on `hz2`; passed after the guard used bitwise comparison for the short-mask edge.
- `criterion_compress.txt`: remote Criterion routing evidence on `vmi1149989`; candidate medians were 103.09 us at 100k and 618.49 us at 1M.
- `numpy_compress_local.txt`: local NumPy 2.4.3 reference timings on `thinkstation1`.
- `criterion_compress_local.txt`: local candidate Criterion timings on `thinkstation1`; candidate medians were 472.85 us at 100k and 1.0645 ms at 1M.
- `compress_tests_post_revert.txt`: local post-revert compress tests; 7 passed.
- `criterion_compress_local_post_revert.txt`: local post-revert Criterion timings; final-code medians were 90.800 us at 100k and 1.1369 ms at 1M.
- `cargo_check_fnp_ufunc_post_revert.txt`: local `cargo check -p fnp-ufunc`; passed.
- `cargo_check_fnp_ufunc_final.txt`: final local `cargo check -p fnp-ufunc`; passed.
- `cargo_clippy_fnp_ufunc_post_revert.txt`: first local `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`; failed on a pre-existing approximate constant literal surfaced by clippy.
- `cargo_clippy_fnp_ufunc_post_revert_after_fix.txt`: final local clippy rerun; passed after replacing the literal with `std::f64::consts::FRAC_2_PI`.
- `window_special_test_after_clippy_fix.txt`: focused regression for the constant cleanup; passed.
- `cargo_fmt_check.txt`: workspace `cargo fmt --check`; reports broad pre-existing format drift outside this slice, so no formatting rewrite was applied.

Same-host ratio vs NumPy:

| Code state | Size | FNP median | NumPy median | FNP / NumPy | Verdict |
|---|---:|---:|---:|---:|---|
| Parallel candidate | 100,000 | 472.85 us | 66.147 us | 7.15x slower | Reject |
| Parallel candidate | 1,000,000 | 1.0645 ms | 518.349 us | 2.05x slower | Reject |
| Post-revert serial path | 100,000 | 90.800 us | 66.147 us | 1.37x slower | Open gap |
| Post-revert serial path | 1,000,000 | 1.1369 ms | 518.349 us | 2.19x slower | Open gap |

The candidate also regressed the local Criterion history by +339.69% at 100k and +66.84% at 1M before revert.
