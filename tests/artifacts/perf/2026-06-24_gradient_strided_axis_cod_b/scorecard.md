# np.gradient non-last (strided) single-axis native row-combine keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.gradient(f, axis=<non-last int>)` for C-contiguous f64 arrays, uniform
scalar spacing, edge_order=1.

## Change

`gradient()` already had a native last-axis path (`try_zerocopy_f64_gradient_1d`); a
non-last (strided) axis fell to the numpy passthrough. numpy implements gradient in
Python via whole-array slice temporaries. New helper
`try_zerocopy_f64_gradient_strided_axis`: the array is laid out as outer x n x inner,
and the edge_order=1 central-difference stencil along the `n` axis is a per-output-ROW
vectorized combination of two input rows of length `inner`
(out[i] = (f[i+1] - f[i-1]) / (2*dx); edges forward/backward / dx) — cache-friendly
(sequential row reads) and parallel across all outer*n output rows, writing the output
buffer directly with no temporaries.

Bit-exact: same subtraction/division operands and order as numpy (verified
`np.gradient(f, axis=ax) == row-combine`, maxulp 0). NaN/Inf propagate. Gate: f64
C-contiguous, single int axis resolving to a non-last axis, uniform scalar spacing,
edge_order=1, n>=2. edge_order=2, coordinate-array spacing, axis=None on N-D (list
return), and non-f64 all defer to numpy.

## Benchmark

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface gradient_axis \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `gradient(axis=0) 4096x1024` | 5,866,803 | 8,125,327 | 0.722x | 1.39x |
| `gradient(axis=0) 1024x4096` | 6,081,379 | 8,077,433 | 0.753x | 1.33x |

Verdict: keep — a modest but real, honest 1.33-1.39x. This op is memory-bandwidth-bound
(each output row reads two input rows + a write ~= 96 MB of traffic for 4M f64), so
temp-avoidance is the only lever and ~1.4x is near the achievable ceiling; numpy's
extra slice temporaries are exactly the difference. Not a ~0-gain result.

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_gradient --test conformance_diff_gradient
```

- `conformance_gradient`: 23 passed / 0 failed (new `gradient_strided_nonlast_axis_matches_numpy`)
- `conformance_diff_gradient`: 12 passed / 0 failed

The new test compares against numpy under `np.allclose(..., rtol=0, atol=0,
equal_nan=True)` (bit-exact incl. dtype/shape): 2-D axis=0, 3-D middle/first axes,
negative axis index, a scalar non-unit spacing, an edge_order=2 fallthrough, a NaN/Inf
array, and the last-axis case (routes through the existing contiguous path).
