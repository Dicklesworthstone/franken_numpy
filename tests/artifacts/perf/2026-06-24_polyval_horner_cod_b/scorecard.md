# np.polyval native fused Horner keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.polyval(p, x)` for real coefficients `p` and an f64 C-contiguous `x` array.

## Change

`polyval` was a pure numpy passthrough. numpy.polyval runs a PYTHON loop
`y = zeros_like(x); for pv in p: y = y*x + pv`, materializing two whole-array
temporaries (`y*x`, then `+pv`) per coefficient — O(deg) passes + temps, tens of ms for
large x. New helper `try_zerocopy_f64_polyval` evaluates the polynomial per element in
registers (one pass over x, deg fused mul-then-add steps), parallel across elements,
writing the output buffer directly with no temporaries.

Bit-exact: reproduces numpy's exact recurrence including the first `0.0*x + p[0]` step
(so x = +-inf -> 0*inf = NaN matches), and uses SEPARATE multiply then add — Rust does
NOT contract `a*b+c` to FMA, so it is the same two roundings numpy's `y*x` then `y+pv`
produce. Real (int/uint/float) coefficients (cast to f64, matching numpy's promotion with
an f64 x); complex p, non-f64 / scalar / non-contiguous / 0-d x defer to numpy.

## Benchmark

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface polyval \
  -- --sample-size 30 --warm-up-time 2 --measurement-time 5 --output-format bencher
```

(sample-size 10 was discarded: cold/loaded-machine warmup gave ~100% variance; 30 samples
+ 2 s warmup is clean.)

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `polyval 1M deg5` | 412,178 | 2,858,107 | 0.144x | 6.94x |
| `polyval 4M deg8` | 5,389,886 | 33,273,713 | 0.162x | 6.17x |

Verdict: keep, clean 6.17-6.94x.

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_poly_ops
```

New test `polyval_native_horner_bitexact_matches_numpy` compares against numpy under
`np.allclose(..., rtol=0, atol=0, equal_nan=True)` (bit-exact incl. dtype/shape): degrees
5/13/4, 1-D and 2-D x, integer coefficients (promote to f64), a degree-0 polynomial, and
a +-inf/NaN x array (numpy's first 0*x+p[0] step makes 0*inf=NaN). 24/24 passed — this
also confirms the build does NOT FMA-contract the Horner step.
