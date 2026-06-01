# Scaling law — elementwise add (SECONDARY evidence for hotspot #1)

Driver: `crates/fnp-ufunc/benches/elementwise.rs`, group `elementwise_add`
(equal-shape `add`, size sweep). criterion n=30, taskset-pinned. Raw:
`elementwise_add_sweep.log`.

| Size (elems) | Working set | Median | Per-element | Throughput |
|--------------|-------------|--------|-------------|-----------|
| 100 | 0.8 KB | 455.5 ns | 4.55 ns | 219 M/s |
| 1 000 | 8 KB | 3.517 µs | 3.52 ns | 284 M/s |
| 10 000 | 80 KB | 34.08 µs | 3.41 ns | 293 M/s |
| 100 000 | 800 KB | 1.217 ms | **12.17 ns** | 82 M/s |
| 1 000 000 | 8 MB | 14.896 ms | **14.90 ns** | 67 M/s |

## Reading

- **Two regimes.** While the operands + output fit in cache (≤10k elems ≈ ≤80 KB),
  per-element cost is a flat ~3.4 ns — the per-element `get_f64` enum-match overhead
  is real but cheap when memory is hot. Past the L2/L3 boundary (100k = 800 KB, 1M =
  8 MB) it jumps 3.6× to ~12–15 ns/elem and throughput collapses from ~290 M/s to
  ~67 M/s.
- **Why.** The result-construction path (`from_storage_with_dtype` → `cast_to(F64)`
  clone → `to_f64_vec()` per-element rebuild → `Self::new`) makes **~3 full passes**
  over the data (clone 8 MB, read 8 MB inside `get_f64`, write 8 MB out) instead of
  the 2 passes a memory-bound add needs (read + write). Once the array exceeds cache,
  that tripled traffic is bandwidth-bound and dominates — exactly the cliff above.
- **Consistency with stage isolation.** `raw_handwritten_add_1m` = 0.526 ms (1M elems,
  2 passes, ~0.5 ns/elem) vs `add()` ≈ 14.9 ms confirms the ~27× gap is the extra
  construction traffic + per-element accessor, not the arithmetic.

## Implication for the fix (bead franken_numpy-71n7p)

Eliminating the redundant `cast_to`/`to_f64_vec` round-trip for already-matching F64
storage removes ~1–2 of the 3 passes AND the per-element `get_f64`, which should both
flatten the cache cliff and drop the small-size constant. Re-running this sweep is the
acceptance test for the optimization.
