# nansum/nanmean/nanprod axis=None keepdims=True — flat fast path via scalar reshape (BlackThrush, 2026-06-19)

## Gap targeted (we LOST) — a CLASS
All flat (axis=None) nan-reduction fast paths gated on !keepdims, so keepdims=True
fell to the cold extract -> native path: nansum_kd 1.99x, nanmean_kd 1.60x,
nanprod_kd 1.57x SLOWER than numpy (also nanmax/nanmin_kd 20x, fixed in ff9105aa).

## Lever
A full reduction yields ONE element; numpy's keepdims=True result is just that scalar
in an all-ones-shaped ndarray of the input ndim. Added shared keepdims_reshape_scalar
(asarray(scalar).reshape([1; ndim]) — value + dtype preserved) and took the flat fast
path for keepdims too. all-NaN/empty still defer to numpy inside the fast path.

## MEASURED (4M f64, 64 cores)
| case       | NumPy us | fnp us | fnp/np |
|------------|----------|--------|--------|
| nansum_kd  | 21287    | 1648   | **0.077** (13x) |
| nanmean_kd | 27046    | 1716   | **0.063** (16x) |
| nanprod_kd | 23986    | 3082   | **0.128** (7.8x) |

Before: 1.6-2.0x SLOWER. After: 7.8-16x FASTER.

## Parity
15/15 differential cases match (shape + dtype + allclose/equal_nan) across 5 shapes x
{plain, partial-NaN, all-NaN} x {nansum, nanmean, nanprod}; conformance_nan_funcs green.
Still-open keepdims: count_nonzero_kd 6.87x (small absolute), nanvar/nanstd_kd flat
~1.2x (borderline).
