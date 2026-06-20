# Non-contiguous average/isclose/trapezoid/nanvar/nanstd/tile/take/cumprod — delegate (BlackThrush, 2026-06-20)

## Gap (we LOST) — final non-contiguous batch
average(d.T) 44.69x, isclose(d.T) 3.65x, trapezoid(d.T) 2.77x, nanvar/nanstd(d.T) 2.5x,
tile(d.T) 1.55x, take(d.T) 1.48x, cumprod(d.T) 1.26x SLOWER than numpy — all bailed the
contiguous-only fast paths into the cold transpose-copy extract.

## Lever
Delegated non-c-contiguous operands to numpy (shared noncontiguous_ndarray helper) at:
average (a or weights non-contig), nanvar, nanstd, cumprod, isclose, tile, take, and
trapezoid_impl (a or x non-contig, forwarding x/dx/axis). Contiguous fast paths untouched.

## MEASURED (4096x4096 f64)
average(d.T) 44.69x->1.00x; isclose 3.65x->1.01x; trapezoid 2.77x->0.99x;
nanvar 2.46x->1.00x; tile 1.55x->1.16x; take 1.48x->0.99x; cumprod 1.26x->1.01x.
Before: 1.3-45x SLOWER. After: parity. Contiguous unchanged (average_C 0.87x faster).

## Parity
T/slice/contig x {average,nanvar,nanstd,cumprod,trapezoid,isclose,tile,take(+clip)}
x axis{None,0,1} 0 fails (allclose equal_nan). conformance: isclose 16, take 20, tile
23, cumprod 18 green; average's own test green. (2 unrelated PRE-EXISTING failures in
the shared test files: cov/corrcoef 1-ULP gram diff exact-tolist test, and the
nan_funcs outcome_body harness codegen bug — neither touches these ops.)
