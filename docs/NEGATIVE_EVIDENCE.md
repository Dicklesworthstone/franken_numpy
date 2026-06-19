# FrankenNumPy Negative-Evidence Ledger

This ledger is append-only evidence for performance hypotheses. It records wins,
losses, neutral results, noisy discarded measurements, and retry predicates so
dead ends are not rediscovered as fresh ideas.

## 2026-06-19 - fnp-random PCG raw fill and bytes cluster

Artifact directory: `tests/artifacts/perf/2026-06-19_random_vs_numpy_pcg/`

Run identity:
- Random subject commit before measured commit: `e32d58ea`.
- Integration base before this commit: `70bae5da`; intervening changes were ufunc evidence/docs and did not touch `fnp-random`.
- Subject API: direct Rust `fnp-random` Criterion rows.
- Oracle/reference: NumPy `np.random.Generator(np.random.PCG64(42))`; local preflight observed NumPy 2.4.3 on `/usr/bin/python3`.
- Workers: `ovh-a` for pre-revert candidate run, `hz1` for final-code run.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-random --benches`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random parallel_pcg_fill_u64_matches_serial_stream_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random bytes_large_calls_match_serial_uint32_stream_state -- --nocapture`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | Ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.255` | Parallel PCG64 `fill_u64` jump-ahead | 100k u64 final code, `hz1` | `criterion_random_vs_numpy_post_revert.txt` | 144,560 ns | 538,164 ns | 0.269x | Keep |
| `franken_numpy-ixs5y.255` | Parallel PCG64 `fill_u64` jump-ahead | 1M u64 final code, `hz1` | `criterion_random_vs_numpy_post_revert.txt` | 2,194,406 ns | 4,447,414 ns | 0.493x | Keep |
| `franken_numpy-ixs5y.257` | PCG bytes via u64 word-fill transcode | 100k bytes pre-revert, `ovh-a` | `criterion_random_vs_numpy_prerevert.txt` | 80,103 ns | 48,911 ns | 1.638x | Reverted |
| `franken_numpy-ixs5y.257` | PCG bytes via u64 word-fill transcode | 1M bytes pre-revert, `ovh-a` | `criterion_random_vs_numpy_prerevert.txt` | 850,688 ns | 428,300 ns | 1.986x | Reverted |
| `franken_numpy-ixs5y.257` | Current serial `Generator::bytes` after revert | 100k bytes final code, `hz1` | `criterion_random_vs_numpy_post_revert.txt` | 119,901 ns | 74,988 ns | 1.599x | Open gap |
| `franken_numpy-ixs5y.257` | Current serial `Generator::bytes` after revert | 1M bytes final code, `hz1` | `criterion_random_vs_numpy_post_revert.txt` | 1,214,954 ns | 982,093 ns | 1.237x | Open gap |

Notes:
- `.255` is kept because the final code remains faster than NumPy on both large raw-buffer workloads while preserving stream state.
- `.257` is rejected and the production word-fill path was removed. The u64-word transcode approach allocated/interpreted an intermediate word buffer and lost to NumPy bytes on both measured rows.
- Retry condition for `.257`: only revisit `Generator::bytes` if the candidate fills the final `Vec<u8>` directly from PCG state without an intermediate `Vec<u64>`, preserves the exact `next_uint32` half-buffer contract, and is remeasured head-to-head against NumPy on the same worker. Do not retry the removed `fill_u64(...).to_le_bytes()` transcode family.

## 2026-06-19 - fnp-random PCG gumbel/laplace distribution cluster

Artifact directory: `tests/artifacts/perf/2026-06-19_random_vs_numpy_pcg_distributions/`

Run identity:
- Subject commit before measured commit: `0442da80`.
- Subject API: direct Rust `fnp-random` Criterion rows.
- Oracle/reference: NumPy `np.random.Generator(np.random.PCG64(42))`; local preflight observed NumPy 2.4.3 on `/usr/bin/python3`.
- Worker: `ovh-a` for both benchmark filters and all targeted correctness tests.
- Target dir requested: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- RCH worker-scoped target observed: `/data/projects/franken_numpy/.rch-target-ovh-a-pool-f4ecbc5a8032ed7eb8c61438ab6b2cc8`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-random --benches`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- gumbel --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- laplace --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random parallel_pcg_gumbel_matches_serial_stream_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random parallel_pcg_laplace_matches_serial_stream_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random gumbel_matches_live_numpy_oracle -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random laplace_matches_live_numpy_oracle -- --nocapture`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.250` | Parallel PCG64 gumbel inverse-CDF fill | 100k f64, `ovh-a` | `criterion_gumbel_vs_numpy.txt` | 248,006 ns | 1,489,338 ns | 0.167x | Keep |
| `franken_numpy-ixs5y.250` | Parallel PCG64 gumbel inverse-CDF fill | 1M f64, `ovh-a` | `criterion_gumbel_vs_numpy.txt` | 2,105,737 ns | 15,047,299 ns | 0.140x | Keep |
| `franken_numpy-ixs5y.253` | Parallel PCG64 laplace inverse-CDF fill | 100k f64, `ovh-a` | `criterion_laplace_vs_numpy.txt` | 204,760 ns | 1,384,891 ns | 0.148x | Keep |
| `franken_numpy-ixs5y.253` | Parallel PCG64 laplace inverse-CDF fill | 1M f64, `ovh-a` | `criterion_laplace_vs_numpy.txt` | 1,599,666 ns | 13,871,270 ns | 0.115x | Keep |

Notes:
- `.250` is kept because both gumbel rows beat NumPy by 6.01x and 7.15x while `parallel_pcg_gumbel_matches_serial_stream_state` and `gumbel_matches_live_numpy_oracle` passed.
- `.253` is kept because both laplace rows beat NumPy by 6.76x and 8.67x while `parallel_pcg_laplace_matches_serial_stream_state` and `laplace_matches_live_numpy_oracle` passed.
- No optimization was reverted in this distribution slice.
- Retry condition for `.250`: revisit only if a same-worker rerun shows the PCG64 gumbel median at or above NumPy's median, if a broader distribution gate exposes a stream-state mismatch, or if NumPy changes PCG64 gumbel semantics in a way that invalidates fixed one-uniform jump-ahead.
- Retry condition for `.253`: revisit only if a same-worker rerun shows the PCG64 laplace median at or above NumPy's median, if a broader distribution gate exposes a stream-state mismatch, or if NumPy changes PCG64 laplace semantics in a way that invalidates fixed one-uniform jump-ahead.

## Carried No-Retry Families

These remain excluded unless a new profile identifies a different primitive and the retry condition is explicit:

| Family | Status | Retry condition |
|---|---|---|
| SVD row/panel/finalization micro-levers | Rejected in prior gauntlet runs | Only retry through a deeper bidiagonal/full-to-band primitive with a fresh `svd_mxn_full/512` proof. |
| Inverse/TRSM broad `batch_solve` routing | Rejected in prior gauntlet runs | Only retry with a different algorithmic route and same-worker evidence beating NumPy. |
| Packed-GEMM tile-width retunes | Rejected in prior gauntlet runs | Only retry as shared packed-panel/RHS redesign, not `PACKED_NR` width tuning. |
| Variable-consumption random distributions | Rejected for jump-ahead parallelization | Only retry if a live NumPy oracle proves fixed-consumption semantics for that distribution. |

## 2026-06-19 - Gauntlet Verify: FNP ufunc data movement vs NumPy

Run identity:
- Subject commit: `e32d58ea` (`main`, mirrored to `master` before this verify pass).
- Subject API: direct Rust `fnp-ufunc` `UFuncArray` Criterion rows.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Worker: `thinkstation1` via `rch exec`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Commands:
  - `cargo bench -p fnp-ufunc --bench elementwise delete_flat_f64_sparse_indices -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
  - `cargo bench -p fnp-ufunc --bench elementwise insert_flat_f64_midpoint_many -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
  - Python batched NumPy timing script with 41 samples, warmups, and per-sample inner loops.

Measurement caveat: these rows compare the optimized Rust `fnp-ufunc` API to
NumPy's Python API on equivalent flat array workloads. They are valid for the
`fnp-ufunc` data-movement cluster, not a full `fnp-python` boundary claim.

| Bead | Workload | Size | FNP Criterion median | NumPy batched median | FNP speed vs NumPy | NumPy CV | Verdict | Retry predicate |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `franken_numpy-ixs5y.256` | `delete_flat_f64_sparse_indices` | 100,000 | 48.657 us | 74.719 us | 1.54x faster | 4.33% | KEEP | Reopen only if same-worker Criterion median regresses above 72 us or a batched NumPy rerun beats the FNP upper CI bound. |
| `franken_numpy-ixs5y.256` | `delete_flat_f64_sparse_indices` | 1,000,000 | 659.58 us | 787.97 us | 1.19x faster | 5.08% | KEEP, borderline CV | Reopen if a low-CV same-worker rerun shows NumPy median <= FNP median, or if broader delete workloads show the sort/dedup span path below 1.05x. |
| `franken_numpy-ixs5y.258` | `insert_flat_f64_midpoint_many` | 100,000 | 17.695 us | 24.896 us | 1.41x faster | 3.60% | KEEP | Reopen only if same-worker Criterion median regresses above 24 us or NumPy batched median drops below FNP upper CI bound. |
| `franken_numpy-ixs5y.258` | `insert_flat_f64_midpoint_many` | 1,000,000 | 256.97 us | 445.04 us | 1.73x faster | 17.92% | KEEP, noisy NumPy allocation row | Reopen with allocator-isolated batching if future evidence puts NumPy p50 below 272 us; current NumPy minimum was still 333.31 us. |

Discarded / non-decision evidence:
- Raw unbatched Python timings were discarded for keep-gate decisions because CV
  was 15-71% on microsecond-scale operations. Those numbers may be useful only
  as a smoke check that the workload shape was valid, not as a pass/fail gate.
- An exact-filter test invocation ran zero inline tests because Cargo's exact
  test name did not match the module-qualified path. The subsequent substring
  filter ran and passed both targeted golden guards.

Conformance / correctness guard:
- `cargo test -p fnp-ufunc delete_flat_f64_span_copy_matches_hashset_reference_and_golden_sha256 -- --nocapture` passed: 1 test run.
- `cargo test -p fnp-ufunc insert_flat_f64_splice_matches_repeated_insert_and_golden_sha256 -- --nocapture` passed: 1 test run.
- `cargo check -p fnp-ufunc` passed after the test-only type inference fix.

Action:
- No optimization reverted. Both measured rows clear the head-to-head NumPy median
  gate and have targeted golden guards green.
- Do not retry the prior per-element `HashSet` scan for large flat F64 delete
  or repeated `Vec::insert` shifting for large flat F64 insert unless the retry
  predicates above fire.

## 2026-06-19 - Gauntlet Verify: FNP compress bool-mask vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_selection_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.249`.
- Subject commit before revert: `0442da80` plus the local candidate guard fix.
- Final code: `.249` parallel compress fast path removed; serial `compress` path retained.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::compress` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-host decision worker: `thinkstation1` for both local FNP Criterion and local NumPy timing.
- Remote routing evidence: `vmi1149989` Criterion candidate run, not used as the keep/reject gate because the NumPy command could not be pinned to that worker.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `cargo test -p fnp-ufunc compress_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `cargo bench -p fnp-ufunc --bench elementwise compress_f64_bool_flat_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
- Python NumPy timing script in `numpy_compress_local.txt` using the same value and bool-mask formulas.
- `cargo test -p fnp-ufunc compress -- --nocapture`
- `cargo check -p fnp-ufunc`
- `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | Ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.249` | Parallel F64 `compress` bool-mask gather candidate | 100k local candidate | `criterion_compress_local.txt`, `numpy_compress_local.txt` | 472.85 us | 66.147 us | 7.15x slower | Reverted |
| `franken_numpy-ixs5y.249` | Parallel F64 `compress` bool-mask gather candidate | 1M local candidate | `criterion_compress_local.txt`, `numpy_compress_local.txt` | 1.0645 ms | 518.349 us | 2.05x slower | Reverted |
| `franken_numpy-ixs5y.249` | Final serial `compress` after revert | 100k post-revert | `criterion_compress_local_post_revert.txt`, `numpy_compress_local.txt` | 90.800 us | 66.147 us | 1.37x slower | Open gap |
| `franken_numpy-ixs5y.249` | Final serial `compress` after revert | 1M post-revert | `criterion_compress_local_post_revert.txt`, `numpy_compress_local.txt` | 1.1369 ms | 518.349 us | 2.19x slower | Open gap |

Notes:
- The first guard attempt failed because the test used `assert_eq!` on a selected slice containing `NaN`; after switching that edge assertion to bitwise comparison, the candidate guard passed.
- Passing correctness was not enough: same-host local Criterion showed the parallel candidate regressed the local Criterion history by +339.69% at 100k and +66.84% at 1M, and it lost badly to NumPy on both measured sizes.
- The production parallel fast path was removed. The remaining serial path is still slower than NumPy, but it is less bad at 100k and avoids keeping a regressing optimization.
- Final focused validation passed for `cargo test -p fnp-ufunc compress`, `cargo check -p fnp-ufunc`, and `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`; `cargo fmt --check` still reports broad workspace format drift outside this slice.
- Retry condition: retry `compress(condition, axis=None)` only if a new design avoids per-chunk `Vec<Vec<f64>>` allocation and proves same-host speed over NumPy on both 100k and 1M bool-mask rows with CV below 10%; do not retry this per-chunk parallel gather shape as a standalone patch.
