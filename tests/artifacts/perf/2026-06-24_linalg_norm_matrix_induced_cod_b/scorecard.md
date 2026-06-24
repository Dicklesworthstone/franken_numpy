# np.linalg.norm induced matrix p-norm (ord ±1/±inf, trailing 2-axis) native fold keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.linalg.norm(x, ord in {1,-1,inf,-inf}, axis=(-2,-1))` for C-contiguous
f64 stacks (induced matrix p-norms over the trailing two contiguous axes).

## Change

Fifth member of the trailing-axis norm fold family (vector L1/L2/±inf, Frobenius).
The committed `norm()` fast-pathed only SVD orders (2/-2/'nuc') and Frobenius for a
2-tuple axis; ord 1/-1/±inf with a 2-tuple axis fell to the numpy passthrough.
NumPy computes a whole-array `abs(x)` temporary, then for ord=±inf a per-ROW
`add.reduce(|x|, axis=-1)` + max/min over rows, and for ord=±1 a per-COLUMN
`add.reduce(|x|, axis=-2)` + max/min over columns — three single-threaded passes.

New helper `try_zerocopy_f64_matrix_norm_lastaxes` (`MatrixNormKind` =
MaxRowSum/MinRowSum=±inf, MaxColSum/MinColSum=±1): per (M,N) contiguous block, reuse
`pairwise_abs_f64` per contiguous row (±inf) or per gathered column (±1, M-buffer
gather), then a NaN-propagating max/min, parallel across blocks. Bit-exact: each
row/column abs-sum matches numpy's pairwise reduce; verified
`add.reduce(|x|,axis=-1).max(-1) == inf-norm`, `add.reduce(|x|,axis=-2).max(-1) ==
1-norm`. **Axis order matters** (1 and inf are transposes): requires row_axis==ndim-2
AND col_axis==ndim-1 exactly; a reversed (-1,-2) pair defers to numpy. NaN/Inf
propagate (no defer).

## Benchmark

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface norm_frobenius \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `norm(ord=inf, axis=(-2,-1)) 4096x16x16` | 334,369 | 1,559,593 | 0.214x | 4.66x |
| `norm(ord=inf, axis=(-2,-1)) 2048x32x32` | 336,430 | 2,461,451 | 0.137x | 7.32x |
| `norm(ord=1, axis=(-2,-1)) 4096x16x16` | 501,909 | 1,849,520 | 0.271x | 3.69x |
| `norm(ord=1, axis=(-2,-1)) 2048x32x32` | 1,045,776 | 2,757,652 | 0.379x | 2.64x |

Verdict: keep. inf-norm 4.7-7.3x, 1-norm 2.6-3.7x faster than NumPy (the 1-norm's
strided per-column gather is the lower-margin path but still a solid win).

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_linalg_basic
```

New test `norm_matrix_induced_lastaxes_matches_numpy` compares ord 1/-1/inf/-inf
against numpy under `np.allclose(..., rtol=0, atol=0, equal_nan=True)` (bit-exact
incl. dtype/shape): 3-D and 4-D stacks, plain 2-D axis=(0,1), keepdims (-> (...,1,1)),
a reversed-axis (-1,-2) fallthrough, a non-trailing axis fallthrough, and NaN/Inf blocks.
