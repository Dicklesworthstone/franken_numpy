# linalg.norm last-axis native no-temporary keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.linalg.norm(a, axis=-1)` for C-contiguous f64 arrays and vector 2-norm

## Change

Add `try_zerocopy_f64_vector_norm_axis` and a matching Criterion bench group.
The helper covers `ord=None` and `ord=2` with a single integer last-axis
reduction. It reads each contiguous f64 lane, squares and pairwise-sums without a
whole-array squared temporary, then returns the square root. Unsupported orders,
tuple axes, non-last axes, non-f64 dtypes, and non-contiguous inputs fall through
to NumPy.

## Benchmark

Command:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface norm_axis \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

RCH worker: `hz2`.

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `linalg.norm(axis=-1) 4096x512` | 170,756 | 1,255,182 | 0.1360x | 7.35x |
| `linalg.norm(axis=-1) 8192x1024` | 531,883 | 9,089,148 | 0.0585x | 17.09x |

Verdict: keep. The native last-axis vector norm path is a measured 7.35-17.09x
faster than NumPy on the targeted rows.

## Correctness

Command:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b \
  rch exec -- cargo test -p fnp-python --test conformance_linalg_basic norm -- --nocapture
```

Result: the existing norm-filtered conformance run passed 12 tests / 0 failed.
The new dedicated `norm_axis_vector_l2_matches_numpy` test passed separately.

The behavior-preservation boundary is explicit: only f64 C-contiguous ndarray
vector 2-norm over the last axis is native. Matrix norms, other orders, tuple
axes, and non-contiguous inputs continue to use the NumPy passthrough.

## Tool Notes

- `cargo test -p fnp-python --test conformance_linalg_basic norm -- --nocapture`
  passed through RCH on `hz2`.
- `cargo test -p fnp-python --test conformance_linalg_basic
  norm_axis_vector_l2_matches_numpy -- --nocapture` passed through RCH on
  `vmi1152480`.
- `cargo check -p fnp-python --all-targets` passed through RCH on `vmi1152480`.
- `cargo bench -p fnp-python --profile release --bench criterion_python_surface
  norm_axis` passed through RCH on `hz2`.
- Existing default warnings remain in `fnp-ufunc` and `fnp-python`; no new
  source error was introduced by this lever.
