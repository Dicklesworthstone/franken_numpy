# Hypothesis Ledger (INTERPRET phase)

Run `2026-06-01_perf_baseline`. Three orthogonal angles
(baseline timing · stage isolation · source read) must agree before a verdict.

| # | Hypothesis | Verdict | Evidence |
|---|-----------|---------|----------|
| H1 | Per-element float-error checking (`note_binary_float_errors`, default errstate = warn) blocks vectorization of the add loop | **REJECTS** | `levelsplit_errstate.txt`: default-warn vs all-ignore ratio = 0.97–1.02× (no difference). Both ~14.5 ms. |
| H2 | The broadcast path misses its fast lane and falls to a per-element index loop | **REJECTS** | `tail_contiguous_broadcast_span([1024],[1024,1024])` → `Some(1024)` and lhs==out, so the tail-contiguous fast path *is* taken; broadcast (14.29 ms) ≈ equal-shape (14.51 ms). The slowness is not broadcast-specific. |
| H3 | The f64 add loop itself is the cost (poor codegen / no SIMD) | **REJECTS** | `raw_handwritten_add_1m` = 0.526 ms; an identical hand loop is 27× faster than `add()`. The loop is not the bottleneck. |
| H4 | Output allocation/zeroing of the 8 MB result dominates | **REJECTS** | `alloc_zero_8mb` = 0.092 ms — negligible. |
| H5 | **Result-array construction dominates**: `from_storage_with_dtype` routes F64 storage through `cast_to(F64)` → `to_f64_vec()`, and `to_f64_vec` rebuilds the `Vec<f64>` via per-element `get_f64(i)` (enum match + bounds + `Result`), ~19 ns/elem | **SUPPORTS** | `construct_from_vec_1m` = 18.90 ms in isolation; `add` ≈ 14.5 ms ≈ construction; `reduce_sum` = 0.70 ms because its output is 1024 elems not 1M. Source: `fnp-dtype/src/lib.rs:1204` (`to_f64_vec`), `:974` (`get_f64`), `fnp-ufunc/src/lib.rs:4764` (`from_storage_with_dtype`, F64 has no move-fast-path). |

## Scaling law

The cost is a **per-element slope (~14–19 ns/elem)**, not a fixed intercept: `add`
on 1M elems ≈ 27 × the 0.5 ms arithmetic, and `reduce_sum` (same 1M *inputs*, 1024
*outputs*) is 21× cheaper — cost tracks output element count through the per-element
`get_f64` accessor. So it scales linearly with output size and dominates at the 1M
fixture; at tiny sizes the dispatch fixed cost would matter more (not measured this pass).

## Hypothesized fix (for the optimizer agents — do NOT apply here)

In `UFuncArray::from_storage_with_dtype` (`fnp-ufunc/src/lib.rs:4764`), add a fast path
for storage whose dtype already equals the target (at least F64): **move the inner
`Vec` straight into `Self::new` / `from_storage`**, skipping `cast_to` + `to_f64_vec`.
Equivalently/additionally, specialize `ArrayStorage::to_f64_vec` (`fnp-dtype:1204`) to
return the inner `Vec<f64>` directly for the `F64` variant (`match self { F64(v) =>
v.clone(), .. }`) instead of the per-element `get_f64` loop. Expected: elementwise
F64 ops drop from ~14.5 ms toward the ~0.5–1 ms memory-bound floor (order-of-magnitude).
Guard with the existing differential/conformance suites + ufunc invariants; one lever,
re-baseline against this artifact.

## Filed beads (perf-labeled)

- **franken_numpy-71n7p** (P1) — H5 root cause: `to_f64_vec`/`from_storage_with_dtype` F64 per-element construction. THE unblock for the optimizer umbrella bead `franken_numpy-perf-current-baseline-hotspot-fbv3z`.
- **franken_numpy-evqs4** (P2) — `matmul` naive 256³ triple loop (2.76 ms); cache-blocking / B-transpose.
- **franken_numpy-g9jvo** (P2) — `fft` Cooley–Tukey 65 536 (2.51 ms); twiddle reuse / iterative in-place.

`sort_quicksort` (4.67 ms) left unfiled — algorithmically expected O(n log n); revisit only after #1.

## Code-first batch attempt ledger

| Date | Bead | Lever | Status | Benchmark guard | Conformance guard | Do not retry unless |
|------|------|-------|--------|-----------------|-------------------|---------------------|
| 2026-06-18 | franken_numpy-ixs5y | Reuse per-worker LU/permutation scratch in `batch_det`/`batch_slogdet` for small stacked matrices (`n < 16`) and write scalar outputs directly instead of allocating a fresh LU bundle and pair vector per lane. | **PENDING BATCH TEST** - compile-only batch by instruction; no performance claim yet. | New Criterion group `batch_det_slogdet/{det,slogdet}` at `8192x4x4` and `2048x8x8`, plus the existing Python-vs-NumPy `fnp_slogdet_f64_batch8192_4x4` gate. | New bitwise guard compares the scratch path against scalar `det_nxn`/`slogdet_nxn` lane references, including singular, NaN, Inf, and negative-zero lanes. | Same-worker Criterion and Python-vs-NumPy evidence show a real win with unchanged guards. Reject and record as negative evidence if the new rows regress or the nonfinite/singular guard fails. |

Avoided negative-evidence families for this batch: broad SVD row/panel/finalization retreads, packed-GEMM tile retunes, and batch-solve blocking experiments. Prior runs already routed those deeper; this attempt targets a different realistic stacked-small-matrix cost center.
