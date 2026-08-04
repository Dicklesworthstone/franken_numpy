# np.nanmean along axis=0 (first axis) native streaming fused-pass keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.nanmean(x, axis=0)` for C-contiguous f64 arrays — the missing-data
imputation reduction `np.nanmean(X, axis=0)`.

## Change

The committed nanmean fast path covered the contiguous last axis; a first-axis (axis=0)
reduction fell to numpy. numpy.nanmean materializes a NaN->0 copy + an isnan mask, then
TWO sequential axis-0 reduces (the NaN->0 sum AND the ~mask count) -> ~19x slower than
plain mean(axis=0). New helper `try_zerocopy_f64_nanmean_axis0`: a SINGLE streaming pass
accumulates per column both the NaN->0 sum and the non-NaN count, then mean = sum/count.
Serial cache-friendly streaming (column-block parallelism is cache-hostile on row-major;
see try_zerocopy_f64_var_axis0).

Bit-exact (numpy reduces axis=0 sequentially; masked positions contribute 0 in both). An
all-NaN column (count==0) yields 0.0/0.0 == numpy's NaN bit pattern directly, plus the
single "Mean of empty slice" RuntimeWarning, so NO defer is needed.

## Benchmark

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface var_axis0 \
  -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `nanmean(axis=0) 4096x512` | 1,415,014 | 3,337,235 | 0.424x | 2.36x |
| `nanmean(axis=0) 50000x64` | 2,129,479 | 6,218,453 | 0.342x | 2.92x |

Verdict: keep, clean 2.36-2.92x. The benched input is all-finite, so this UNDERSTATES
the gap — numpy.nanmean(axis=0) on actual 10%-NaN data measured ~6.8 ms (vs ~3.3 ms
here), where the win is larger.

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_nan_funcs
```

New test `nanmean_axis0_first_axis_matches_numpy` compares against numpy under
`np.allclose(..., rtol=0, atol=0, equal_nan=True)` (bit-exact incl. dtype/shape):
keepdims, negative axis, 3-D axis=0, NaN columns, an Inf column, and an all-NaN column
(NaN + "Mean of empty slice" warning, computed directly). 37/37 passed.
