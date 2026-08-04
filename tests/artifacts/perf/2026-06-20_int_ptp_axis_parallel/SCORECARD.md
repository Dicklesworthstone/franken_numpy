# integer ptp along axis — parallel native + narrow/wide split (BlackThrush, 2026-06-20)

## Gap (we LOST)
np.ptp of integer arrays along an axis was up to 3.41x SLOWER than numpy (int64 ax0
~36ms gap): ptp_axis_typed ran a serial per-inner (max,min) accumulator while numpy
uses SIMD max/min.

## Lever
Narrow ints (1/2-byte): delegate to numpy's SIMD ptp. Wide ints (4/8-byte):
parallelize the (max,min)-then-subtract fold over a zero-copy &[T] view -- inner==1
fans lanes (last axis) or chunk-reduces (1-D); non-last privatizes outer groups or
row-blocks tracking (max,min) planes. Order-independent -> bit-identical.

## MEASURED (4096x4096)
int8 (narrow): ax0/ax1 ~1.0x (parity). int32: ax0 0.25x, ax1 0.11x (4-9x WIN). int64:
0.23/0.15x. uint64: 0.22/0.13x. Before: up to 3.41x SLOW. After: parity / 4-9x WIN.

## Parity
7 int dtypes x 5 shapes x all axes 0 fails (array_equal + dtype); conformance green.
