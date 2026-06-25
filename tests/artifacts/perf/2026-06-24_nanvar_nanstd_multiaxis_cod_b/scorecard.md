# np.nanvar / np.nanstd multi-axis trailing native fold keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.nanvar(x, axis=(-2,-1))` / `np.nanstd(x, axis=(...))` for C-contiguous
f64 arrays where the axis tuple resolves to the contiguous trailing axes.

## Change

Generalizes the committed `try_zerocopy_f64_nanvar_axis` (single contiguous last
axis) to accept an axis TUPLE resolving to the contiguous trailing axes (per-block
"lane" = product of those trailing dims) — the same generalization just applied to
the plain var/std path. numpy.nanvar materializes an isnan mask, a where/zeroed
temporary, and squared-deviation temporaries before its multi-axis reduce (very slow:
~7-14 ms at these sizes); the native per-block pairwise nansum/count + sum-of-squared-
deviations fold reads each contiguous block with no allocation, parallel across blocks.

Bit-exact: numpy's multi-axis reduce over a contiguous trailing block is identical to a
flat per-block reduce (verified `nanvar(x, axis=(-2,-1)) == nanvar(x.reshape(B,M*N), -1)`
bit-for-bit, ddof 0/1, nanvar and nanstd). nanvar is symmetric in the reduced axes, so
the tuple is sorted. Gate: f64 C-contiguous, axis tuple == exactly `[ndim-k .. ndim)`,
ddof >= 0; non-trailing/duplicate axes defer; any block with count <= ddof (e.g. an
all-NaN block) defers the WHOLE call to numpy so its "Degrees of freedom <= 0" warning
+ NaN parity stays exact.

## Benchmark

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface var_multiaxis \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `nanvar(axis=(-2,-1)) 4096x16x16` | 275,998 | 3,137,466 | 0.088x | 11.37x |
| `nanstd(axis=(-2,-1)) 4096x16x16` | 597,599 | 3,088,155 | 0.194x | 5.17x |
| `nanvar(axis=(-2,-1)) 2048x32x32` | 768,824 | 6,921,506 | 0.111x | 9.00x |
| `nanstd(axis=(-2,-1)) 2048x32x32` | 796,808 | 6,654,613 | 0.120x | 8.35x |

Verdict: keep. Measured 5.2-11.4x faster than NumPy on the targeted multi-axis rows.

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_nan_funcs
```

New test `nanvar_nanstd_multiaxis_trailing_matches_numpy` compares nanvar AND nanstd
against numpy under `np.allclose(..., rtol=0, atol=0, equal_nan=True)` (bit-exact incl.
dtype/shape): ddof 0/1, keepdims, reversed axis order, 3-axis trailing reduce, plain
2-D axis=(0,1), a non-trailing axis fallthrough, NaN-containing blocks, and an all-NaN
block (which defers + matches numpy's NaN and warning).
