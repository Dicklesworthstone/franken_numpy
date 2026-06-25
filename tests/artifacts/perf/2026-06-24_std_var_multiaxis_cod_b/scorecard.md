# np.var / np.std multi-axis trailing native two-pass fold keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.var(x, axis=(-2,-1))` / `np.std(x, axis=(...))` for C-contiguous f64
arrays where the axis tuple resolves to the contiguous trailing axes.

## Change

Generalizes the committed `try_zerocopy_f64_var_axis` (single last axis) to accept
an axis TUPLE resolving to the contiguous trailing axes (the per-block "lane" is the
product of those trailing dims). numpy.var allocates whole-array temporaries
(a - mean broadcast, then squared) before its multi-axis pairwise reduce; the native
per-block two-pass pairwise fold reads each contiguous block twice with no
allocation, parallel across blocks.

Bit-exact: numpy's multi-axis reduce over a contiguous trailing block is identical to
a flat per-block reduce (verified `var(x, axis=(-2,-1)) == var(x.reshape(B,M*N), -1)`
bit-for-bit, ddof 0/1, var and std). Variance is symmetric in the reduced axes, so the
axis tuple is sorted and order is irrelevant. Gate: f64 C-contiguous, axis a tuple
resolving exactly to `[ndim-k .. ndim)`, native ddof, no out/dtype, empty kwargs;
non-trailing/duplicate axes, axis_len <= ddof, and non-finite block means defer to numpy.

## Benchmark

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface var_multiaxis \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `var(axis=(-2,-1)) 4096x16x16` | 843,892 | 2,470,544 | 0.342x | 2.93x |
| `std(axis=(-2,-1)) 4096x16x16` | 380,081 | 1,056,932 | 0.360x | 2.78x |
| `var(axis=(-2,-1)) 2048x32x32` | 300,667 | 2,496,925 | 0.120x | 8.30x |
| `std(axis=(-2,-1)) 2048x32x32` | 252,473 | 2,514,051 | 0.100x | 9.96x |

Verdict: keep. Measured 2.8-10.0x faster than NumPy on the targeted multi-axis rows.

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_var --test conformance_std
```

New test `var_std_multiaxis_trailing_matches_numpy` (in conformance_var) compares
var AND std against numpy under `np.allclose(..., rtol=0, atol=0, equal_nan=True)`
(bit-exact incl. dtype/shape): ddof 0/1, keepdims (-> trailing 1s), reversed axis
order (variance symmetric), 3-axis trailing reduce, plain 2-D axis=(0,1), a
non-trailing axis fallthrough, and a NaN block (defers + matches numpy's NaN).
