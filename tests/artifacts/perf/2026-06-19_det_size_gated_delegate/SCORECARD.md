# np.linalg.det — size-gated native/numpy routing (BlackThrush, 2026-06-19)

## Gap targeted (we LOST)
np.linalg.det of a real 2-D square matrix was 1.5-2.4x SLOWER than NumPy for n in
[64, 768]: det() routed ALL single matrices through fnp_linalg::det_nxn, which loses
to OpenBLAS getrf at those sizes. (solve/inv/matmul already win.)

## Key measured fact (this box, scipy-openblas 0.3.31 DYNAMIC_ARCH)
OpenBLAS getrf hits a sharp degradation CLIFF above n~800: n=768 ~12ms but
n=832 ~756ms, n=1024 ~976ms (consistent across trials). fnp's parallel blocked LU is
flat (~23-40ms), so fnp wins 25-33x above the cliff.

## Lever (defer-where-we-lose, same pattern as multi_dot/matrix_rank)
Gate the native single-matrix path on n >= DET_NATIVE_MIN_DIM (832); delegate smaller
real-float 2-D square matrices to numpy. A cheap shape/dtype PEEK (getattr, no
extract) delegates the common medium case straight to numpy, skipping the wasted
extract→UFuncArray copy that left a ~1.1x residual. Batched / complex / large paths
unchanged.

## MEASURED (f64, 64 cores)
| n    | NumPy us | fnp us | before | after |
|------|----------|--------|--------|-------|
| 128  | 96.7     | 99.8   | 1.47x  | **1.03x** |
| 256  | 576.9    | 562.2  | 2.42x  | **0.98x** |
| 512  | 3069.7   | 3036.7 | 2.02x  | **0.99x** |
| 768  | 11715.3  | 11507.8| 1.58x  | **0.98x** |
| 832  | 756054.5 | 22755.4| (win)  | **0.03x** (33x) |
| 1024 | 975696.2 | 40375.2| (win)  | **0.04x** (24x) |

Before: 1.5-2.4x SLOWER for n<=768. After: parity (0.97-1.07x) below the cliff,
24-33x faster above it. LOSS -> NEUTRAL/WIN.

## Parity
17/17 differential cases match (isclose) across n {1..1024} incl boundary 831/832,
float32 (dtype preserved via numpy delegate), int, complex, singular, Python-list
input, batched. conformance_linalg_basic 56/56 + metamorphic_array_ops 84/84
(det(A@B)==det(A)*det(B)) green.

## NOTE
The n>=832 win rides numpy's OpenBLAS getrf cliff on this box; native is kept there
(unchanged existing behavior), so no regression risk even if another machine's BLAS
lacks the cliff.
