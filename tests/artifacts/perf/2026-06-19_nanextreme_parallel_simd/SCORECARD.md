# np.nanmax / np.nanmin flat reduction — direct-SIMD + parallel (BlackThrush, 2026-06-19)

## Gap targeted (we LOST)
np.nanmax(a) / np.nanmin(a) over large 1-D f64 arrays (axis=None) was
1.46-1.83x SLOWER than NumPy. The zero-copy fast path (try_zerocopy_f64_nanextreme
-> simd_nanextreme_f64) staged each 64 KiB block through a reused `buf` before the
SIMD fold (extra L2 traffic) and ran single-threaded, while NumPy's nanmax is also
single-threaded — leaving the extra memory channels idle.

## Lever (extreme-software-optimization / privatized parallel reduce)
1. Read the read-only contiguous PyBuffer cells as a plain &[f64] (ReadOnlyCell<f64>
   is repr(transparent) over f64; sound under the GIL) and SIMD-fold directly — no
   per-block staging copy (new simd_nanextreme_slice).
2. For n >= 1<<18, par_chunks(256 KiB) across the rayon pool, each chunk runs the
   direct-SIMD fold, reduced by IEEE max/min + OR of saw-nonnan. NaN-skipping
   min/max is order-independent, and the ±0-sign tie still defers to numpy, so the
   merge is bit-exact.

## MEASURED (4M f64, 64 cores)
| case          | NumPy us | fnp us | fnp/np |
|---------------|----------|--------|--------|
| nanmax_nonan  | 1255     | 196    | **0.16** (6.4x) |
| nanmin_nonan  | 1476     | 136    | **0.09** (10.9x) |
| nanmax_nan    | 1368     | 204    | **0.15** (6.7x) |
| nanmin_nan    | 1335     | 144    | **0.11** (9.3x) |

Before: 1.46-1.83x SLOWER. After: 6.4-10.9x FASTER.

## Parity
72/72 differential cases bit-exact (np.float64 tobytes equality, NaN==NaN) across
n {7,8,100,1<<15,(1<<18)+3,1<<22} x nan-frac {0,0.001,0.5,0.999} x scale
{1,1e300,1e-300} x {nanmax,nanmin}. All-NaN/empty/±0-tie still defer to numpy
(warning/error/sign parity preserved).

## Still open (next)
- nanmax_ax0 1.45x / nanmax_ax1 1.27x: the axis reduction path is still serial.
- amax/amin 1.13-1.19x: routes to py_max (same as np.max which is at parity) —
  likely measurement noise; max_plain measured 0.94x same run.

## FOLLOW-UP: axis reduction parallelized (same commit family)
nanmax/nanmin along an axis was also serial: nanmax_ax0 1.45x, nanmax_ax1 1.27x slow.
- Last axis (inner==1): independent contiguous lanes — par_chunks_exact across the pool,
  each lane via simd_nanextreme_slice.
- Non-last axis: privatized inner-wide (extreme, saw) plane fold. ≥2 outer groups fan
  across groups; a single group (2-D axis=0) privatizes across row-blocks and merges
  planes elementwise (order-independent min/max; ±0-sign tie still defers).
- Replaced the now-dead staged simd_nanextreme_raw + NANEXTREME_BLK.

MEASURED (4096x4096 f64): nanmax_ax0 1.45x->0.66x, ax1 1.27x->0.39x,
nanmin_ax0 0.55x, ax1 0.44x. 160/160 axis differential cases bit-exact across
9 shapes x all axes x 4 nan-fractions x {max,min}.
