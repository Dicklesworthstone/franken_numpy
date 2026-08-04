# np.vander native fused cumulative-product keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.vander(x, N, increasing)` for an f64 C-contiguous 1-D `x`.

## Change

`vander` was a pure numpy passthrough. numpy builds the Vandermonde matrix via
`tmp[:,1:] = x[:,None]; multiply.accumulate(tmp, axis=1)` — a broadcast temp plus a
strided in-place cumulative product — which is ~11 ms for a tiny 200k x 8 output and
~43 ms for 500k x 12. New helper `try_zerocopy_f64_vander` computes each row's powers as
a per-row left-to-right cumulative product in registers, writing the (len(x), N) output
buffer directly with no temporary, parallel across rows.

Bit-exact: the powers are the SAME left-to-right cumulative product numpy's accumulate
performs (1, x, x*x, x^2*x, ...). `increasing=False` (default) only stores that sequence
into reversed column positions — the multiplications are identical. f64 C-contiguous 1-D
x only (numpy keeps int64 for int input -> defer); N==0 / empty x defer.

## Benchmark

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface vander \
  -- --sample-size 20 --warm-up-time 2 --measurement-time 4 --output-format bencher
```

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `vander 200k x8` | 3,253,763 | 11,236,864 | 0.290x | 3.45x |
| `vander 500k x12` | 5,738,239 | 42,925,213 | 0.134x | 7.48x |

Verdict: keep. 3.45-7.48x. fnp had ~30-50% run-to-run variance (loaded box / rayon
scheduling) but the point estimates beat NumPy unambiguously (NumPy itself was stable
and 3-7x slower).

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_array_creation_base
```

New test `vander_native_cumprod_bitexact_matches_numpy` compares against numpy under
`np.allclose(..., rtol=0, atol=0, equal_nan=True)` (bit-exact incl. dtype/shape): default
N, explicit N (wider and narrower than len(x)), increasing True/False, a NaN/Inf x, and
an int-x fallthrough (numpy keeps int64). 12/12 passed.
