# np.sum last-axis parallel pairwise keep

Agent: BlackThrush / cod-a
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.sum(a, axis=-1)` for exact C-contiguous f64 ndarrays with no `out`, `dtype`,
`initial`, or extra kwargs.

## Change

`sum` was a pure NumPy passthrough. The new `try_zerocopy_f64_sum_lastaxis` path reads
the contiguous f64 buffer directly, reduces each row with the existing `pairwise_simd_f64`
tree, and parallelizes independent rows with rayon above a size gate. Empty axes and
unsupported options defer to NumPy.

## Benchmark

`vmi1149989`, sample-size 15 + 2 s warmup + 4 s measurement:

| Row | FNP median | NumPy median | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `sum(axis=-1)`, 8192x1024 f64 | 1.1513 ms | 4.5198 ms | 0.255x | 3.93x |
| `sum(axis=-1)`, 65536x256 f64 | 2.4695 ms | 8.8303 ms | 0.280x | 3.58x |

Verdict: keep.

## Correctness

`conformance_sum` passed 27/27. The new row checks 2-D and 3-D last-axis reductions,
negative axis, keepdims, NaN/Inf propagation, and non-last-axis fallback against NumPy
with exact shape, dtype, and raw `tobytes()` equality.

`cargo check -p fnp-python --all-targets` passed on `hz2`. `cargo clippy -p fnp-python
--all-targets -- -D warnings` still fails before this change in the existing workspace
dependency warning `fnp-ufunc::UFuncArray::nan_filtered` dead code.
