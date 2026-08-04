# integer min/max (all axes) — parallel native + narrow/wide split (BlackThrush, 2026-06-20)

## Gap (we LOST)
np.max/np.min of integer arrays was 1.3-2.6x SLOWER than numpy across flat AND axis:
minmax_int_typed ran a SERIAL scalar fold (inner==1 per-lane; non-last inner-wide)
while numpy uses SIMD. Wide ints (int32 flat 2.58x, ax1 2.01x; int64 1.4-1.6x) and
narrow ints along axis (int8/uint8 1.8x) all lost.

## Lever
(1) Narrow ints (1/2-byte): delegate to numpy's SIMD (16-64 lanes/instruction, flat +
axis) — a scalar Rust fold can't beat it. (2) Wide ints (4/8-byte): parallelize the
native fold across the rayon pool over a zero-copy &[T] view — inner==1 fans lanes
(last axis) or chunk-reduces the single run (flat); non-last privatizes outer groups
or row-blocks. min/max is order-independent -> bit-identical.

## MEASURED (4096x4096)
int8 (narrow): flat/ax1/ax0 ~1.0x (numpy SIMD, parity). int32: flat 0.21x, ax1 0.24x,
ax0 0.33x (3-5x WIN). int64: 0.25/0.30/0.37x. uint64: 0.23-0.37x. Before: 1.3-2.6x
SLOW. After: parity (narrow) / 3-5x WIN (wide).

## Parity
8 int dtypes x 5 shapes x all axes x {max,min} x keepdims 0 fails (array_equal +
dtype); conformance_reductions green.
