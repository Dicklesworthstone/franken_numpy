# np.argmax / np.argmin along last axis — parallel per-lane (BlackThrush, 2026-06-19)

## Gap targeted (we LOST)
np.argmax(d, axis=1) / argmin on 4096x4096: f64 1.70x SLOW, int64 1.52x SLOW. The
last-axis fast paths (try_zerocopy_lastaxis_argextreme f64 branch + lastaxis_argextreme_int)
ran a SERIAL loop over the `outer` lanes (and the f64 branch copied each lane into a
scratch `buf` before simd_argextreme_f64).

## Lever
Each lane's argmax is independent, so read the read-only contiguous buffer as &[T]
(sound under the GIL) and par_chunks_exact over lanes:
- f64: pass each lane sub-slice straight to simd_argextreme_f64 (drops the per-lane
  buf copy too). NaN lane -> defer whole call.
- int (i8..i64,u8..u64): the same branchless extreme + first-index two-pass per lane.
simd_argextreme_f64 / the int two-pass keep the FIRST max on ties, so per-lane
results are bit-identical to numpy's first-occurrence semantics.

## MEASURED (4096x4096, 64 cores)
| case           | NumPy us | fnp us | fnp/np |
|----------------|----------|--------|--------|
| argmax_ax1 f64 | 7516     | 3433   | **0.46** (2.2x) |
| argmin_ax1 f64 | 10833    | 3726   | **0.34** (2.9x) |
| argmax_ax1 i64 | 9507     | 3170   | **0.33** (3.0x) |

Before: f64 1.70x SLOW, i64 1.52x SLOW. After: 2.2-3.0x FASTER.

## Parity
36/36 differential cases bit-exact (array_equal, incl. heavy-ties and all-zero-dup
lanes that exercise first-index tie-break) across 6 shapes x last-axis x
{plain,ties,dup} x {argmax,argmin}. conformance_argmax + conformance_argmin green
(10+10, no failures).
