# np.nanvar / np.nanstd along axis=0 (first axis) native streaming keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.nanvar(x, axis=0)` / `np.nanstd(x, axis=0)` for C-contiguous f64 arrays —
the missing-data analog of the ML feature-standardization reduction.

## Change

`try_zerocopy_f64_nanvar_axis0`: the NaN-aware sibling of the axis=0 var streaming fold
(47942af4). numpy.nanvar materializes a NaN->0 copy + isnan mask + per-column count +
a-mean broadcast + squared temp before two SEQUENTIAL axis-0 reduces. This streaming
two-pass accumulates per column skipping NaN (= numpy's NaN->0 then sum): pass 1 ->
per-column sum + non-NaN count -> mean; pass 2 -> sum of (slab-mean)^2 over non-NaN ->
var; std = sqrt. Serial cache-friendly streaming (column-block parallelism is
cache-hostile on row-major; see var_axis0). Bit-exact (numpy reduces axis=0 sequentially;
masked positions contribute 0 in both — verified). If ANY column has count <= ddof (e.g.
an all-NaN column) the WHOLE call defers to numpy so its "Degrees of freedom <= 0"
warning + per-column NaN parity stay exact.

## Benchmark

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface var_axis0 \
  -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `nanvar(axis=0) 4096x512` | 3,203,279 | 6,933,889 | 0.462x | 2.16x |
| `nanstd(axis=0) 4096x512` | 3,169,327 | 6,956,400 | 0.456x | 2.19x |
| `nanvar(axis=0) 50000x64` | 4,979,087 | 13,031,312 | 0.382x | 2.62x |
| `nanstd(axis=0) 50000x64` | 4,908,205 | 13,174,345 | 0.382x | 2.68x |

Verdict: keep, clean low-variance 2.16-2.68x. Lower than plain var axis=0 (~4x) because
the per-element NaN check + count adds work; and the benched input is all-finite, so this
UNDERSTATES the gap — numpy.nanvar(axis=0) on actual 10%-NaN data measured ~14 ms (vs ~7
ms here), where the win is larger.

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_nan_funcs
```

New test `nanvar_nanstd_axis0_first_axis_matches_numpy` compares nanvar AND nanstd under
`np.allclose(..., rtol=0, atol=0, equal_nan=True)` (bit-exact incl. dtype/shape): ddof
0/1, keepdims, negative axis, 3-D axis=0, NaN columns, an Inf column, and an all-NaN
column (defers + matches numpy NaN + warning).
