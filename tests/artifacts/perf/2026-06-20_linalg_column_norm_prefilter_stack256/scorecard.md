# 2026-06-20 fnp-linalg column-norm residual no-ship

Run identity:
- Parent bead: `franken_numpy-ixs5y`.
- Agent: `BlackThrush` / `cod-a`.
- Worker: `hz2`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Crate scope: `fnp-linalg` only.
- Workload: direct Rust `matrix_norm_nxn_orders` for `ord="1"` and `ord="-1"`.
- NumPy comparator: same-worker `tests/artifacts/perf/2026-06-20_linalg_batch_vs_numpy/numpy_linalg_hz2.txt`.

Lever attempts:
- `branchless-prefilter+stack256`: scan the matrix once for NaN before the cache-linear column sum, remove the per-element `is_nan` branch from the accumulation loop, and extend stack scratch to 256 columns.
- `stack256-only`: keep the original per-element NaN branch and only extend the stack scratch window from 512 columns down to 256.

Commands:
- `AGENT_NAME=BlackThrush RCH_WORKER=hz2 RCH_WORKERS=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'matrix_norm_nxn_orders' -- --sample-size 10 --warm-up-time 1 --measurement-time 2 --output-format bencher`
- `AGENT_NAME=BlackThrush RCH_WORKER=hz2 RCH_WORKERS=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-linalg matrix_norm_column_reduction_matches_strided_reference_bits -- --nocapture`

Focused correctness:
- `cargo_test_column_reduction_candidate.txt`: pass, 1 focused test passed.
- `cargo_test_column_reduction_stack256.txt`: pass, 1 focused test passed.
- Correctness did not justify keeping either lever because the performance gate failed.

Same-worker evidence:

| Workload | NumPy ns | Baseline FNP ns | Prefilter+stack256 FNP ns | Stack256-only FNP ns | Baseline/NumPy | Prefilter/NumPy | Stack256/NumPy | Stack256/Baseline | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `one/256` | 27712 | 29785 | 45590 | 29766 | 1.075x | 1.645x | 1.074x | 0.999x | no-ship |
| `neg_one/256` | 28312 | 30303 | 44570 | 29630 | 1.070x | 1.574x | 1.047x | 0.978x | no-ship |
| `one/512` | 103667 | 115964 | 182591 | 114610 | 1.119x | 1.761x | 1.106x | 0.988x | no-ship |
| `neg_one/512` | 102987 | 113597 | 183733 | 119919 | 1.103x | 1.784x | 1.164x | 1.056x | no-ship |
| `one/1024` | 397192 | 457106 | 723751 | 465194 | 1.151x | 1.822x | 1.171x | 1.018x | no-ship |
| `neg_one/1024` | 393621 | 458114 | 727149 | 456385 | 1.164x | 1.848x | 1.159x | 0.996x | no-ship |

Decision:
- Reverted both production source changes.
- The NaN prefilter doubled memory traffic and lost badly despite removing the hot-loop branch.
- The stack-threshold-only variant produced only one modest improvement (`neg_one/256`, 0.978x baseline) while regressing `neg_one/512` and `one/1024`; this is not a keepable lever.

Negative retry predicate:
- Do not retry a whole-matrix NaN prefilter for matrix column norms unless it is fused with another required scan.
- Do not retry the 256-column stack threshold as a standalone lever.
- The remaining gap needs a real SIMD or strip-mined multi-column accumulation primitive that preserves column-addition order and NaN behavior, followed by same-worker NumPy proof on all 256/512/1024 `ord=1/-1` rows.
