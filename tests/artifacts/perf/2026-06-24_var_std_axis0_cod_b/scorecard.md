# np.var / np.std along axis=0 (first axis) native streaming two-pass keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.var(x, axis=0)` / `np.std(x, axis=0)` for C-contiguous f64 arrays — the
ubiquitous ML feature-standardization reduction (`(X - X.mean(0)) / X.std(0)`).

## Change

The committed var/std fast paths covered the contiguous LAST axis and trailing-axis
tuples; a first-axis (axis=0) reduction fell to numpy. Key discovery: numpy reduces
the OUTER axis SEQUENTIALLY, not pairwise (`add.reduce(a, axis=0)` == a straight
row-by-row accumulation, verified bit-exact; the pairwise tree only applies to the
contiguous last axis). So a streaming two-pass is bit-exact.

New helper `try_zerocopy_f64_var_axis0`: M = shape[0], inner = product(shape[1..]).
Pass 1 streams the M contiguous slabs accumulating the row sum -> mean; pass 2 streams
again accumulating (slab - mean)^2 -> var; std takes sqrt. No temporaries (numpy
materializes the a-mean broadcast and the squared array). NaN/Inf propagate through the
straight (a-mean)^2 exactly as numpy does (no NaN->0 leaf), so no non-finite defer.

SERIAL by design: the two passes read the array sequentially (full cache lines) =
bandwidth-optimal. Column-block parallelism was tried and REJECTED — each thread
strides through all M rows reading only its column slice, so a row-major array is read
with poor spatial locality; the parallel version measured ~2.6x SLOWER (1.63 ms vs 0.63
ms) and far noisier (+/- 1.1 ms vs +/- 39 us). The win is numpy temp-avoidance.

## Benchmark

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface var_axis0 \
  -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `var(axis=0) 4096x512` | 628,648 | 2,575,503 | 0.244x | 4.10x |
| `std(axis=0) 4096x512` | 637,617 | 2,774,411 | 0.230x | 4.35x |
| `var(axis=0) 50000x64` | 1,595,322 | 5,948,924 | 0.268x | 3.73x |
| `std(axis=0) 50000x64` | 1,441,285 | 5,936,180 | 0.243x | 4.12x |

Verdict: keep. A clean, low-variance 3.7-4.35x on a very high-traffic op.

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_var --test conformance_std
```

New test `var_std_axis0_first_axis_matches_numpy` (in conformance_var) compares var AND
std under `np.allclose(..., rtol=0, atol=0, equal_nan=True)` (bit-exact incl.
dtype/shape): ddof 0/1, keepdims (-> (1, N...)), negative axis index, 3-D axis=0,
NaN/Inf columns (propagate, no defer), an M<=ddof defer case (numpy NaN + warning), and
tall/wide shapes.
