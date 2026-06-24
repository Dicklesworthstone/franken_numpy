# np.linalg.norm vector +-inf (ord=±inf) last-axis native fold keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.linalg.norm(x, ord=np.inf | -np.inf, axis=<int last axis>)` for
C-contiguous f64 arrays (max|x| / min|x| along the contiguous last axis).

## Change

Extends the last-axis vector-norm fold (commits `6355309e` L2, `657a1137` L1) to
`ord=+inf` (max|x|) and `ord=-inf` (min|x|). NumPy's axis-int inf-norm is
`abs(x).max(axis)` / `abs(x).min(axis)`: it materializes a whole-array `abs(x)`
temporary, then a SEPARATE per-axis max/min reduce (two passes over the temp) -
which is why numpy's inf-norm is even slower than its L1/L2 here. The native
per-lane `lane_extreme_abs_f64` fold computes max/min of |x| in one pass, no temp.

- `lane_extreme_abs_f64(lane, want_max)`: NaN-PROPAGATING (numpy's
  maximum/minimum.reduce propagate NaN, unlike f64::max/min) - any NaN lane -> NaN;
  abs() and max/min are exact -> bit-identical; +-Inf flow through.
- `VectorNormKind::{MaxAbs, MinAbs}`; `norm()` maps `ord=+inf` -> MaxAbs,
  `ord=-inf` -> MinAbs. Empty axis still defers to numpy (raises zero-size reduce).

## Benchmark

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface norm_axis \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `norm(ord=inf, axis=-1) 4096x512` | 338,984 | 3,175,403 | 0.107x | 9.37x |
| `norm(ord=inf, axis=-1) 8192x1024` | 1,670,441 | 11,513,939 | 0.145x | 6.89x |

Verdict: keep. Measured 6.9-9.4x faster than NumPy on the targeted inf rows.

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_linalg_basic
```

New test `norm_axis_vector_inf_matches_numpy` compares the +inf (max|x|) and -inf
(min|x|) paths against numpy under `np.allclose(..., rtol=0, atol=0,
equal_nan=True)` (bit-exact incl. dtype/shape): axis=-1 keepdims both, a non-last
axis fallthrough, a 1-D scalar axis, and NaN/Inf lanes for both max and min.
