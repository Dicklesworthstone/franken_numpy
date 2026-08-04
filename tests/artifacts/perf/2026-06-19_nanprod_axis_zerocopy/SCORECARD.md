# np.nanprod along last axis — per-lane sequential product (BlackThrush, 2026-06-19)

## Gap (we LOST)
np.nanprod(d, axis=1) had NO axis fast path (only axis=None), so it fell to the cold
extract -> native path = 1.65x SLOWER than numpy. (Completes the nan-reduction axis
family.)

## Lever
numpy folds each lane's product left-to-right with NaN replaced by 1.0; float multiply
is not associative so WITHIN-lane order must match, but lanes are independent. Added
try_zerocopy_f64_nanprod_axis: per-lane sequential `acc *= (NaN?1.0:v)` fold (the same
identity-multiply as the axis=None flat path -> bit-exact) fanned across the rayon pool
over a zero-copy &[f64] view; keepdims via keepdims_expand_axis. numpy's nanprod(axis)
builds a full NaN->1 replacement temp; the per-lane fold has none and parallelizes.

## MEASURED (4096x4096 f64, axis=1, 64 cores)
| case             | NumPy us | fnp us | fnp/np |
|------------------|----------|--------|--------|
| keepdims=False   | 98512    | 2880   | **0.029** (34x) |
| keepdims=True    | 100641   | 3774   | **0.038** (27x) |

Before: 1.65x SLOWER. After: 27-34x FASTER.

## Parity
30/30 differential cases BIT-EXACT (tobytes / array_equal equal_nan) across 5 shapes x
keepdims {T,F} x {plain, partial-NaN, all-NaN lane}. conformance_nan_funcs nanprod
tests green.
