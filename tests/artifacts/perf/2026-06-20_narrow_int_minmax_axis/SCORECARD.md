# narrow-int min/max along axis — delegate to numpy SIMD (BlackThrush, 2026-06-20)

## Gap (we LOST)
np.max/np.min of 1/2-byte integer arrays ALONG AN AXIS was 1.3-2.1x SLOWER than numpy:
the native per-lane scalar fold (minmax_int_typed) can't match numpy's SIMD axis
reduction (16-64 narrow-int lanes per instruction). int8/uint8 ax1 ~1.8x, ax0 ~1.3x.

## Lever
Delegate narrow ints (itemsize<=2) WITH an explicit axis to numpy's SIMD reduction
(forwarding axis + keepdims). Flat (axis=None) narrow-int min/max stays native
(memory-bound, already parity); 4/8-byte ints stay native (parity/win).

## MEASURED (4096x4096)
int8 max ax1 1.79x->1.06x, ax0 1.30x->1.01x; uint8 ax1 1.77x->1.04x; flat unchanged
(1.00-1.03x); int32 unchanged. LOSS -> parity.

## Parity
4 narrow-int dtypes x 3 shapes x all axes x {max,min} x keepdims 0 fails (array_equal
+ dtype preserved); conformance_reductions green.
