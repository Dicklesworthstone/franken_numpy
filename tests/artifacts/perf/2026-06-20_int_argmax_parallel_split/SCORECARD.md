# np.argmax/argmin integer flat — parallel single-pass + narrow/wide split (BlackThrush, 2026-06-20)

## Gap (we LOST)
Integer np.argmax/argmin (axis=None) was 1.7-3.9x SLOWER than numpy: argextreme_typed
did TWO full scans (.max() then .position()) serially, while numpy's int argmax is one
SIMD pass. int16 3.92x, int32 1.92x, int64 1.67x.

## Lever
(1) Single-pass (value, first-index) fold — halves the scans; parallel reduce across
the rayon pool with first-occurrence tie-break (higher value wins; equal -> lower
index) = bit-identical to numpy. (2) Narrow/wide SPLIT: numpy's SIMD does 16-64
narrow-int lanes/instruction (a scalar parallel fold can't beat that), so 1/2-byte
ints delegate to numpy; 4/8-byte ints use the parallel native fold (memory-bound, 64
cores win over numpy's narrower wide-int SIMD).

## MEASURED (4M, flat)
int8 1.77x->1.11x; int16 3.92x->1.18x (numpy SIMD); int32 1.92x->0.73x; int64 1.67x->
0.116x (8.6x!); uint32 0.45x. f64 unchanged. LOSS -> parity (narrow) / WIN (wide).

## Parity
8 int dtypes x {1,2,1000,>1<<16,1<<20} x {argmax,argmin} + heavy-ties (first
occurrence) 0 fails (exact index); conformance_argmax + conformance_argmin green.
