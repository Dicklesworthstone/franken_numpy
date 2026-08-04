# nan-family reductions along AXIS with keepdims=True — expand_dims (BlackThrush, 2026-06-19)

## Gap (we LOST) — a CLASS
The nan-family single-axis fast paths (nanmax/nanmin/nansum/nanmean) gated on
!keepdims, so axis + keepdims=True fell to the cold extract path:
nanmax(d,axis=1,keepdims=True) was 31.5x SLOWER than numpy, nanmin similar,
nansum 2.06x. (nanstd/nanvar axis keepdims was already fixed in bb09f60d.)

## Lever
numpy keepdims=True for a single-int-axis reduction == the reduced result with a
length-1 axis re-inserted at the (normalized) reduced position. Added shared
keepdims_expand_axis (np.expand_dims at the normalized axis); let the axis fast
paths run for keepdims and restore the kept axis. Bit-identical to the
already-shipped non-keepdims result reshaped (verified), so parity is unchanged.

## MEASURED (4096x4096 f64, axis=1, keepdims=True, 64 cores)
| op      | NumPy us | fnp us | before | after |
|---------|----------|--------|--------|-------|
| nanmax  | ~5900    | ~2800  | 31.5x  | **0.48x** |
| nanmin  | ~5900    | ~1800  | ~31x   | **0.30x** |
| nansum  | ~91000   | ~16000 | 2.06x  | **0.18x** |
| nanmean | ~114000  | ~7400  | ~16x   | **0.065x** |

Before: 2-31x SLOWER. After: 2-15x FASTER.

## Parity
22 shapes/axes x {plain,partial-NaN} x {nanmax,nanmin,nansum,nanmean} = 0 fails
(shape + allclose equal_nan; nansum is allclose-not-bitexact vs numpy on BOTH the
keepdims and non-keepdims paths — naive axis sum, pre-existing). keepdims result ==
non-keepdims result with axis re-inserted (exact). conformance_nan_funcs 33 real
tests green (the 1 failure is the pre-existing outcome_body harness codegen bug).

## Still-open (same sweep): nanprod axis has NO fast path (1.65x), strided/transposed
nan reductions gate on c_contiguous (nanmax_strided1d 8.6x, ptp_transposed_ax1 22x).
