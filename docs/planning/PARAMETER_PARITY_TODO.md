# Parameter Parity TODO — franken_numpy-ksr

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [D] Parity debt/deferred — observable NumPy behavior not yet closed
- [A] Architecture note — no current behavior gap unless future evidence proves one

---

## GROUP 1: unique(axis=) — Multi-dimensional uniqueness
- [x] Implement `unique_axis(&self, axis: isize)` on UFuncArray
- [x] Handle axis parameter: extract slices along axis, compare them, deduplicate
- [x] Return unique slices preserving sorted order
- [x] Add tests: 2D array unique rows, unique columns

## GROUP 2: concatenate/stack dtype+casting parameters
- [x] Add `concatenate_with_dtype(arrays, axis, dtype, casting)` method
- [x] Validate casting via `fnp_dtype::can_cast` for each input → output dtype
- [x] Cast each input array to target dtype before concatenation
- [x] Add `stack_with_dtype(arrays, axis, dtype, casting)` variant
- [x] Add tests: concatenate with dtype promotion, casting rejection

## GROUP 3: sum/prod `initial` parameter
- [x] Add `reduce_sum_initial(&self, axis, keepdims, initial: f64)` method
- [x] Add `reduce_prod_initial(&self, axis, keepdims, initial: f64)` method
- [x] Initial value added to sum / multiplied into product
- [x] Add tests: sum with initial, prod with initial

## GROUP 4: sum/prod/mean/var/std `dtype` parameter
- [x] Rust-core `reduce_sum_with_dtype` and `reduce_prod_with_dtype` select both accumulator and result dtype for bool, all signed/unsigned integer widths, and float16/32/64; inputs are cast before accumulation, integer overflow wraps at the requested width, and exact wide-integer sidecars survive the result
- [x] Rust-core `reduce_mean_with_dtype`, `reduce_var_with_dtype`, and `reduce_std_with_dtype` preserve accumulator/result precision, source-precision deviations, axis/keepdims, empty inputs, and ddof; interleaved complex sum/prod/mean/var/std support logical axes and real or complex accumulator dtypes
- [x] Complex nansum/nanprod/nanmean/nanvar/nanstd exclude an entire logical element when either component is NaN; complex cumsum/cumprod and their NaN-aware variants preserve component precision, logical axes, and first-element bits
- [x] Default float16 mean preserves source precision and float32 conversion-buffer ordering; variance/std retain integer count precision through division, including counts above float16's finite range
- [x] Default float32/64 and complex sum/prod/mean/var/std over axis tuples use joint C-order reductions, preserving logical axes, source precision, NumPy reduction ordering, and real-valued complex variance/std
- [D] Linux x86_64 floating-to-integer casts now model NumPy's signed/narrow integer conversion intermediates and uint64 invalid-value results; uint32 vector-body/scalar-tail differences, other platforms, and warning/error-policy integration remain open (`franken_numpy-5cf9h`)
- [x] Python FFI explicit `dtype=` keyword conformance is covered for `sum`, `prod`, `mean`, `std`, and `var`

## GROUP 5: percentile/quantile `keepdims` parameter
- [x] Add `percentile_keepdims` method
- [x] Add `quantile_keepdims` method
- [x] When keepdims=true, insert size-1 dimension at reduced axis
- [x] Handle axis=None case (all dims become size 1)
- [x] Add tests: axis keepdims, None axis keepdims

## GROUP 6: partition/argpartition `kind` parameter
- [A] NumPy currently exposes one accepted algorithm token (`introselect`); parity work should still verify accepted/invalid token behavior and error text

## GROUP 7: unique `return_index/inverse/counts` with axis
- [x] Add `unique_axis_with_info(axis, return_index, return_inverse, return_counts)` on UFuncArray
- [x] Return NumPy-shaped `return_index`, `return_inverse`, and `return_counts` metadata for unique axis slices
- [x] Cover large-integer sidecars, signed-zero representatives, and NaN slice distinctness

---

## ALREADY DONE (earlier this session)
- [x] clip_optional (one-sided clipping with None min/max)
- [x] reshape_order (F-order reshape)
- [x] histogram_full (density, range, weights parameters)

## KNOWN PARITY DEBT / ARCHITECTURE NOTES
- [D] `out` parameter on array functions — Rust internals are immutable, but Python/NumPy-facing APIs still expose in-place output semantics that need explicit parity handling or documented wrapper delegation
- [D] `order` on sort/partition — structured/object-array ordering remains observable NumPy behavior and should stay tracked as parity debt
- [D] `overwrite_input` on percentile/quantile — immutable internals do not remove the need to match accepted parameter behavior and error surfaces
- [D] Explicit accumulator dtype variants on tuple-axis and NaN-aware reduction methods remain open; default float32/64 and complex tuple-axis reductions now have dedicated joint-ordering coverage
