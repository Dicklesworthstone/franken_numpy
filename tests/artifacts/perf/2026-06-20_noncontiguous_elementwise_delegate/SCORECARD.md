# Non-contiguous elementwise unary/where/clip/round — delegate to numpy (BlackThrush, 2026-06-20)

## Gap (we LOST) — a CLASS
Elementwise unary ops + where/clip/round on transposed (F-contiguous) or strided
ndarrays bailed out of the contiguous-only zero-copy fast paths into a cold extract ->
rebuild that TRANSPOSE-COPIES the input, far slower than numpy's strided ufunc:
isnan(d.T) 108x, where(d.T) 6.87x, clip(d.T) 4.95x, negative/abs(d.T) ~4.7x,
round(d.T) 4.50x, sqrt(d.T) 2.89x SLOWER. (Binary ops + exp already fine.)

## Lever (defer-where-we-lose)
numpy applies ufuncs to strided/transposed arrays efficiently; fnp can't beat that
without a contiguity copy. Added a non-c-contiguous ndarray check that delegates to
numpy at the shared chokepoints: native_unary_elementwise + native_unary_promoting
(covers negative/abs/sqrt/sin/floor/rint/square/...), isnan_native, around/round,
clip, and where (any non-contiguous operand). Contiguous fast paths untouched.

## MEASURED (4096x4096 f64, 64 cores)
| op          | before | after |
|-------------|--------|-------|
| isnan(d.T)  | 108x   | **0.98x** |
| where(d.T)  | 6.87x  | **1.01x** |
| clip(d.T)   | 4.95x  | **1.00x** |
| negative(d.T)| 4.79x | **1.00x** |
| abs(d.T)    | 4.74x  | **1.00x** |
| round(d.T)  | 4.50x  | **1.00x** |
| sqrt(d.T)   | 2.89x  | **1.00x** |

Before: 2.9-108x SLOWER. After: parity. Contiguous unchanged (round/where C still
0.64-0.84x faster).

## Parity
transposed/sliced/with-NaN cases x {negative,abs,isnan,sqrt,round,sin,floor,rint,
clip,where,where-scalar} 0 fails (array_equal equal_nan). conformance_special_math +
conformance_nan_to_num_clip (clip, 17) green; 22/23 conformance_where pass (the 1
failure `where_python_container_surfaces_match_numpy` is a PRE-EXISTING where x=/y=
keyword-signature gap from a peer's audit test d998138b — where_py signature is
`(condition, /, *args)`; rejected at the signature level, unrelated to this body-only
perf change).
