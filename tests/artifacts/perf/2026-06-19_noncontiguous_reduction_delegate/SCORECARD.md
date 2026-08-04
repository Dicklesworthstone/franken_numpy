# Non-contiguous (transposed/strided) reductions — delegate to numpy (BlackThrush, 2026-06-19)

## Gap (we LOST) — a CLASS
ptp/nanmax/nanmin/nansum/nanmean zero-copy fast paths gate on c_contiguous, so a
transposed (F-contiguous) or strided ndarray bailed into the cold extract -> scalar
scan, which is far slower than numpy's cache-blocked strided reduction:
ptp(d.T,axis=1) 21.94x, nanmax(strided1d) 8.63x, nansum(strided1d) 3.37x,
nanmean(d.T,axis=1) 3.19x SLOWER.

## Lever (defer-where-we-lose)
numpy reduces strided/transposed arrays efficiently (cache-blocked); fnp can't beat
that without an expensive contiguity copy. Added a non-c-contiguous ndarray check
before the extract path in each wrapper that delegates straight to numpy (byte-
identical / same parity). Contiguous inputs keep the parallel zero-copy fast path.

## MEASURED (64 cores)
| case (4096^2 / 4M strided) | before | after |
|----------------------------|--------|-------|
| ptp(d.T, axis=1)           | 21.94x | **1.04x** |
| nanmax(strided 1d)         | 8.63x  | **1.01x** |
| nansum(strided 1d)         | 3.37x  | **0.97x** |
| nanmean(d.T, axis=1)       | 3.19x  | **1.05x** |
| (contiguous, unchanged)    | —      | nanmax 0.42x, ptp 0.27x |

Before: 3-22x SLOWER. After: parity. LOSS -> NEUTRAL.

## Parity
16 transposed/strided/sliced cases x {nanmax,nanmin,nansum,nanmean} x axis{None,0,1}
+ 1-D strided + ptp transposed/strided/sliced = 0 fails (allclose equal_nan).
conformance_diagnostics + conformance_reductions green; nan_funcs 33 real tests green.

## FOLLOW-UP: plain max/min + argmax/argmin non-contiguous (same class)
np.max(d.T,axis=1) was 32x, np.max(strided1d) 19x, argmax 2.78x SLOWER. The plain
max/min zero-copy fold and argextreme paths also gate c_contiguous. For max/min the
delegate must go BEFORE try_zerocopy_f64_minmax's any-NaN scan (itself O(n) on a
non-contiguous array) — placed right after the out/where fallback. argmax/argmin
delegate before extract. Contiguous int/f64 fast paths preserved.
Measured: max(d.T,axis=1) 32x->1.01x, max(strided1d) 19x->~1.0x, argmax(strided) 2.78x->1.02x.
12 transposed/sliced/int/with-NaN cases x {max,min} x axis{None,0,1} = 0 fails.
