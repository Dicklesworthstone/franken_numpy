# np.gradient full (axis=None, N-D) native per-axis tuple keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.gradient(f)` (no axis) on an N-D f64 C-contiguous array, uniform spacing,
edge_order=1 — the common image/field gradient that returns a TUPLE of per-axis arrays.

## Change

The committed gradient fast paths covered a single last axis and a single non-last
(strided) axis; the no-axis full-gradient case (axis=None on an N-D array) fell to numpy,
which returns a tuple of per-axis gradients, each via its slow pure-Python slice path
(~17.5 ms for a 4096x1024 2-D call). New dispatch branch computes each axis with the
existing native helpers (`try_zerocopy_f64_gradient_1d` for the contiguous last axis,
`try_zerocopy_f64_gradient_strided_axis` for the others) and returns the tuple.

Bit-exact: each axis is the same central-difference stencil numpy uses (verified per-axis
maxulp 0). NaN/Inf propagate. If any axis can't take the native path (edge_order=2,
non-f64, non-contiguous, multi/per-axis spacing) the whole call aborts to numpy so the
result matches exactly. Returns a tuple (numpy's modern return type).

## Benchmark

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface gradient_axis \
  -- --sample-size 20 --warm-up-time 2 --measurement-time 4 --output-format bencher
```

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `gradient(f) 4096x1024` | 2,884,069 | 17,502,646 | 0.165x | 6.07x |
| `gradient(f) 1024x4096` | 5,391,569 | 15,893,558 | 0.295x | 2.95x |

Verdict: keep, 2.95-6.07x. Much larger than the single strided-axis gradient (1.4x,
bandwidth-bound) because numpy is slow on BOTH axes and fnp does both fast (the contiguous
last axis via gradient_1d crushes numpy there).

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_gradient
```

New test `gradient_full_no_axis_tuple_matches_numpy` compares against numpy under
`np.allclose(..., rtol=0, atol=0, equal_nan=True)` (bit-exact incl. dtype/shape, per tuple
element): 2-D and 3-D arrays, default and scalar spacing, 1-D (single-array return), and an
edge_order=2 fallthrough. 13/13 passed.
