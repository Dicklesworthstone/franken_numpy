# FrankenNumPy Negative-Evidence Ledger

This ledger is append-only evidence for performance hypotheses. It records wins,
losses, neutral results, noisy discarded measurements, and retry predicates so
dead ends are not rediscovered as fresh ideas.

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
