# Ranked Hotspot Table (INTERPRET phase → hand-off)

Run `2026-06-01_perf_baseline`. Each row cites a reproducible evidence artifact.
Profiling-only level-split harness: `crates/fnp-ufunc/examples/perf_levelsplit_errstate.rs`
→ raw output `levelsplit_errstate.txt`.

## Stage-isolation evidence (the key measurement)

From `levelsplit_errstate.txt` (taskset-pinned, median of 60 iters, 1024×1024 = 1M elems):

| Stage | Median | Note |
|-------|--------|------|
| `raw_handwritten_add_1m` | **0.526 ms** | what the f64 add *should* cost (memory-bound) |
| `alloc_zero_8mb` | 0.092 ms | output allocation is negligible |
| `add_equal_1024x1024` (default warn) | 14.51 ms | full `add()`, equal shapes |
| `add_equal_1024x1024` (all-ignore) | 15.03 ms | **errstate makes no difference (0.97×)** |
| `add_broadcast` (default warn) | 14.29 ms | full `add()`, broadcast |
| `construct_from_vec_1m` | **18.90 ms** | building ONE `UFuncArray` from a 1M `Vec` |
| `reduce_sum_axis1` | 0.699 ms | fast: builds a *1024-elem* output, not 1M |

**Conclusion:** the arithmetic is 0.5 ms; the full op is ~14.5 ms. The ~14 ms delta
is **result-array construction**, paid once per op on the output. `reduce_sum` is
fast only because its output is tiny. Float-error state is irrelevant (refutes the
first hypothesis — see `30_hypothesis_ledger.md`).

## Ranked hotspots

| Rank | Location | Metric | Value | Category | Evidence |
|------|----------|--------|-------|----------|----------|
| 1 | `fnp-dtype/src/lib.rs:1204` `ArrayStorage::to_f64_vec` (per-element `get_f64`, called from `fnp-ufunc/src/lib.rs:4764` `from_storage_with_dtype` for F64 results) | construction / 1M elem | **~14–19 ms** (~19 ns/elem; 27× the 0.5 ms arithmetic) | CPU | `levelsplit_errstate.txt`; src `fnp-dtype/src/lib.rs:1204-1209`, `974-1004`, `fnp-ufunc/src/lib.rs:4764-4791` |
| 2 | `fnp-ufunc` elementwise binary/unary result path (every op that returns an F64 array routes through #1) | systemic tax | dominates all elementwise ops | CPU | derived from #1; `10_baseline.md` add vs reduce 21.3× |
| 3 | `fnp-ufunc` `matmul` (256³, naive triple loop, no blocking/SIMD) | 256³ matmul | 2.759 ms (~12.2 GFLOP/s) | CPU | `10_baseline.md`; criterion `core_ops/matmul_256x256_by_256x256` |
| 4 | `fnp-ufunc` `fft` (Cooley–Tukey, 65 536) | transform | 2.509 ms (26.1 M elem/s) | CPU | `10_baseline.md`; criterion `core_ops/fft_65536` |
| 5 | `fnp-ufunc` `sort` quicksort (1M) | sort | 4.669 ms (214 M elem/s) | CPU | `10_baseline.md`; algorithmically expected — lowest priority |

### Why rank 1 is the target

`from_storage_with_dtype` (fnp-ufunc:4764) has a fast path that *moves* the inner
`Vec` only for `I64`/`U64` storage (lines ~4743-4760 in `from_storage`). For **F64**
storage it instead runs `cast_to(F64)` → `to_f64_vec()` → `Self::new()`. `cast_to`
returns early with a clone for matching dtype (cheap), but `to_f64_vec()` rebuilds the
`Vec<f64>` via `(0..n).map(|i| self.get_f64(i))` — a **per-element 15-arm enum match +
bounds check + `Result`** that cannot vectorize. That is the ~19 ns/elem. Every
elementwise op pays it on its output, so fixing #1 lifts the entire ufunc surface, not
just `add`.

Hand-off: filed as `perf`-tagged beads (see `30_hypothesis_ledger.md` tail for IDs).
Measurement only — no code changed in fnp-dtype/fnp-ufunc this pass.
