# bool ptp/around parity bugs (load-independent differential, BlackThrush, 2026-06-20)

## Bugs (found under load 146 via container/array_like differential, not perf)
1. np.ptp(bool) raises TypeError ("numpy boolean subtract ... not supported"); fnp
   computed the native max-min and returned np.True_ (too permissive). Bool list,
   bool ndarray, and bool 2-D all diverged.
2. np.round/around(bool LIST) promotes bool->float16 [1.,0.,1.]; fnp returned a bool
   array. The bool->float16 delegate was gated on is_exact_instance(ndarray), so a
   Python bool LIST (array_like) skipped it and the native Rint path kept bool.
   (bool ndarray was already correct.)

## Fix
1. ptp: after extract, `matches!(array.dtype(), DType::Bool) -> fallback()` (numpy
   raises the same TypeError) -- catches list + ndarray uniformly.
2. around: post-extract `DType::Bool -> fallback()` so bool list gets numpy's float16
   promotion (the ndarray-gated check above missed non-ndarray inputs).

## Verification (load-independent: pass/fail, not timing)
330 container/array_like surface cases x {ptp,round,around,...} 0 fails (was 2);
ptp/round on f64/int/f32/lists/big-arrays unchanged; conformance green.
REUSABLE: array_like (list/tuple/scalar) inputs hit dtype-promotion + error-parity
paths that ndarray-gated fast-path checks skip -- post-extract DType guards catch both.
