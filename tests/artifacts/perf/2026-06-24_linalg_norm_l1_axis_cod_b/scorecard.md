# np.linalg.norm vector L1 (ord=1) last-axis native fold keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.linalg.norm(x, ord=1, axis=<int last axis>)` for C-contiguous f64
arrays (vector L1 norm along the contiguous last axis).

## Change

Extends the just-landed last-axis vector-norm fold (commit `6355309e`, which
covered `ord in {None, 2}`) to `ord=1`. NumPy's axis-int L1 norm is
`np.add.reduce(abs(x), axis)`, materializing a whole-array `abs(x)` temporary
before the per-axis pairwise reduce; the native per-lane fold reduces each
contiguous last-axis row with NO temporary, parallel across lanes.

- New kernel `pairwise_abs_f64`: same pairwise tree as `pairwise_sq_f64` but sums
  `|v|` (abs only clears the sign bit -> exact), so it matches numpy's reduce over
  the materialized abs temp bit-for-bit; NaN/Inf propagate.
- New `VectorNormKind { L2, L1 }` threaded into `try_zerocopy_f64_vector_norm_axis`;
  `norm()` dispatch now maps `ord in {1, 1.0}` -> L1 and `ord in {None, 2, 2.0}` -> L2.
  All other cases (other orders, non-last/tuple axes, non-f64/non-contiguous,
  complex, matrix norms) still delegate to numpy.

## Benchmark

Command:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface norm_axis \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `norm(ord=1, axis=-1) 4096x512` | 391,714 | 1,610,950 | 0.243x | 4.11x |
| `norm(ord=1, axis=-1) 8192x1024` | 899,202 | 11,135,812 | 0.081x | 12.38x |

(Same-run L2 control re-confirmed the prior win: 4096x512 5.30x, 8192x1024 9.63x.)

Verdict: keep. Measured 4.1-12.4x faster than NumPy on the targeted L1 rows.

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_linalg_basic
```

New test `norm_axis_vector_l1_matches_numpy` compares the exact L1 path against
numpy under `np.allclose(..., rtol=0, atol=0, equal_nan=True)` (bit-exact incl.
dtype/shape): ord=1 int and 1.0 float, axis=-1 keepdims both, a non-last axis
fallthrough, a 1-D scalar-axis case, and a NaN/Inf lane.
