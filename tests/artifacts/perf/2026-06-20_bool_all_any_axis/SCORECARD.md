# bool all/any along axis — delegate to numpy SIMD (BlackThrush, 2026-06-20)

## Gap (we LOST)
np.all/np.any of a bool array ALONG AN AXIS was 2.2x SLOWER than numpy: the
try_zerocopy_any_all per-lane scalar fold can't match numpy's SIMD byte-scan
(with early-exit) axis reduction. all/any ax1 ~2.16-2.21x.

## Lever
Delegate bool-dtype all/any WITH an explicit axis to numpy's SIMD reduction. axis=None
bool (early-exit, sub-us / fnp wins the all-true full scan 0.78x) and f64 stay native.

## MEASURED (4096x4096 bool)
all ax1 2.16x->1.02x; any ax1 2.21x->1.01x; ax0 1.00-1.03x; flat unchanged (sub-us
dispatch on early-exit; all-true full scan fnp 0.78x faster). LOSS -> parity.

## Parity
3 shapes x all axes x {all,any} + all-true/all-false edges 0 fails (array_equal);
conformance_reductions green.
