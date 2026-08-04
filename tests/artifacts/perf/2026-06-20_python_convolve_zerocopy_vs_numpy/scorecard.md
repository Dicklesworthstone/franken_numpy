# 2026-06-20 fnp-python convolve/correlate zero-copy short-kernel vs NumPy

Bead: under directive `franken_numpy-ixs5y` ([perf][no-gaps])
Agent: `BlackThrush` / `cod-b`
Measurement: `thinkstation1` (64 cores), NumPy 2.4.3, OMP/OPENBLAS=1, load ~13.

## Problem (diagnosed before fixing)

`np.convolve`/`np.correlate` short-kernel (k<=~48) 1-D f64 was 9–38x slower than
NumPy. Per-step instrumentation showed the SIMD gather KERNEL was already at NumPy
parity (1.38ms @ 1M×8); the loss was two full-array copies the UFuncArray wrapper
paid and NumPy avoids: `extract_numeric_array` (input → owned Vec, ~4.75ms/8MB)
and `build_numpy_array_from_ufunc` (result Vec → numpy, ~5.5ms).

## Fix

Extracted `convolve_mode`'s SIMD-across-outputs gather into a shared
`fnp_ufunc::convolve_gather_fill(a, kr, n, m, out, lo)` (bit-identical refactor;
19 fnp-ufunc convolve tests still green). Added a zero-copy fnp-python fast path
(`try_zerocopy_conv_corr_f64`): reads both f64 1-D buffers as `&[f64]` (no copy),
allocates the numpy output once, and runs `convolve_gather_fill` writing the mode
region DIRECTLY into the output buffer — one input read + one output write, with
the output band split across cores for large outputs. Convolve is commutative
(signal = longer, kernel = shorter reversed); correlate requires len(a)>=len(v)
(else defers). Gated to kernel<=48 (pure-gather regime, never shadows the FFT
path). Non-f64 / non-contiguous / long-kernel / list inputs fall through to the
existing path unchanged.

## Head-to-head (FNP/NumPy, 'same'; >1 = slower)

| N | k=3 | k=5 | k=16 | k=32 |
|---|---|---|---|---|
| 10,000 | 1.27 par | 1.07 par | **0.28** | **0.49** |
| 100,000 | 0.97 par | 0.91 par | **0.25** | **0.47** |
| 1,000,000 | 1.10 par | **0.48** | **0.09** | **0.17** |
| 2,000,000 | 1.02 par | **0.85** | **0.06** | **0.09** |

Was **9–38x LOSS** across this grid; now **WIN or parity everywhere** — up to
**16x faster** than NumPy at k=16/32 (fnp parallelizes + SIMDs the gather; NumPy's
convolve is serial O(N·k)). The only non-wins are tiny-N k=3 (~1.0–1.3x), the
~6µs PyBuffer/alloc dispatch floor — negligible absolute.

## Validation

| Gate | Result |
|---|---|
| Exhaustive convolve+correlate vs numpy (243 size×mode cases incl swap/boundaries) | **0 fails** |
| Defer cases (lists, f32 dtype-preserved, correlate La<Lv, kernel>48, 2-D) | 9/9 match numpy |
| `cargo test -p fnp-python --test conformance_convolution` | PASS |
| `cargo test -p fnp-ufunc convolve` (refactor bit-identical) | 19 PASS |
| `cargo build` both crates | clean |
| clippy (my regions) | clean (pre-existing `eq_op` elsewhere unchanged) |

## Notes

The refactor shares one gather impl between `convolve_mode` and the wrapper (no
SIMD duplication). The win generalizes the established zero-copy-PyBuffer pattern
(cov/corrcoef/where/select) to convolve: the kernel was fine; eliminating the
wrapper's extract+build copies is what unlocked it.
