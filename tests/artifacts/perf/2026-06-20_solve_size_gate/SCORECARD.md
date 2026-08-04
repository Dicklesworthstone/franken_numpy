# linalg.solve — size-gate at OpenBLAS gesv cliff (BlackThrush, 2026-06-20)

## Gap (we LOST)
np.linalg.solve of small 2-D systems was 1.4-2.7x SLOWER than numpy: the native LU
solve loses to OpenBLAS gesv for small n, AND solve extracted BOTH operands into
UFuncArrays + ran an is_finite scan before any size check, so even delegating
post-extract stayed 1.4-2.7x slow (the copies dominate a ~3us solve).

## Lever
Shape-PEEK before extraction (det/slogdet getrf-cliff pattern): for a 2-D square
ndarray with n<104, delegate to numpy WITHOUT extracting. This box's gesv cliffs
sharply above n~100 (n<=96 beats native ~2.2x; n>=104 native LU wins 1.2-50x as
numpy's solve degrades). Post-extract size-gate (n<104) backstops list inputs.

## MEASURED
n=16 2.74x->1.18x, n=64 1.54x->1.07x, n=96 1.44x->1.04x (parity, delegated). n=104
0.75x, n=128 0.05x, n=256 0.20x, n=512 0.09x (WIN, native, numpy cliffs). Only n=8
1.32x residual = inherent sub-us dispatch on a ~3us solve. losses 3->0 (n>=16).

## Parity
n in {3,16,64,96,103,104,150,256} x {vector,matrix} RHS + list + batched 0 fails
(allclose); conformance_linalg green. Solution is unique -> bit-stable.
