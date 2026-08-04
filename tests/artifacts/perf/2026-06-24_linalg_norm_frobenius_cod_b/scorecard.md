# np.linalg.norm batched Frobenius (trailing 2-axis) native fold keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.linalg.norm(x, ord in {None,'fro','f'}, axis=(-2,-1))` for C-contiguous
f64 stacks (per-matrix Frobenius norm over the trailing two contiguous axes).

## Change

The committed `norm()` fast-pathed only the SVD matrix orders (`ord in {2,-2,'nuc'}`)
and the single-int-axis vector norms; a 2-tuple-axis Frobenius norm fell to the
numpy passthrough. NumPy runs `sqrt(np.add.reduce((x.conj()*x).real, axis=(row,col)))`,
materializing a whole `(..., M, N)` squared temporary then a single-threaded 2-axis
reduce.

New helper `try_zerocopy_f64_frobenius_lastaxes`: for a C-contiguous f64 array each
`(M, N)` matrix is one contiguous `M*N` block, and numpy's 2-axis reduce is
bit-identical to a flat pairwise sum-of-squares over that block (verified:
`add.reduce(x, axis=(-2,-1)) == add.reduce(x.reshape(B, M*N), axis=-1)`, maxulp 0.0).
So `pairwise_sq_f64` over each block + sqrt, parallel across blocks, is bit-exact
and allocates only the reduced output. Gated to ord None/'fro'/'f', an explicit
2-tuple axis resolving to the trailing two axes, f64 C-contiguous; everything else
(non-trailing axes, axis=None BLAS-dot ravel, other dtypes) defers to numpy.
NaN/Inf propagate (no defer).

## Benchmark

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface norm_frobenius \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `norm(axis=(-2,-1)) 4096x16x16` | 136,826 | 480,724 | 0.285x | 3.51x |
| `norm(axis=(-2,-1)) 2048x32x32` | 155,441 | 1,206,268 | 0.129x | 7.76x |

Verdict: keep. Measured 3.5-7.8x faster than NumPy on the targeted stacks.

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_linalg_basic
```

New test `norm_frobenius_lastaxes_matches_numpy` compares the exact path against
numpy under `np.allclose(..., rtol=0, atol=0, equal_nan=True)` (bit-exact incl.
dtype/shape): 3-D and 4-D stacks, plain 2-D with axis=(0,1), reversed axis order
(-1,-2), keepdims (-> (...,1,1)), a non-trailing-axis fallthrough, and a NaN/Inf block.
