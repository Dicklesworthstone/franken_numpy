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
- [x] Cast original inputs directly to the target dtype, retaining exact wide-integer sidecars and source precision; Linux x86_64 uint32 conversion follows each operand's contiguous destination runs and aggregates invalid events once per operand
- [x] Add `stack_with_dtype(arrays, axis, dtype, casting)` variant
- [x] Add tests: concatenate with dtype promotion, casting rejection
- [x] Complex concatenate/stack validate logical shapes and axes before conversion, retain paired components, and support mixed real/complex operands and explicit dtype/casting controls
- [x] Complex append, flattened concatenation, hstack/vstack/dstack/column_stack, ravel, and atleast_1d/2d/3d preserve logical elements and dimension promotion; empty joins return immediately and checked axis lengths reject overflow
- [x] Live NumPy oracle coverage includes logical negative axes, scalar/empty inputs, source precision, exact integer overrides, casting rejection, signed zero, destination-dependent uint32 values, and invalid-policy callback counts (`complex_joins.rs`)

## GROUP 3: sum/prod `initial` parameter
- [x] Add `reduce_sum_initial(&self, axis, keepdims, initial: f64)` method
- [x] Add `reduce_prod_initial(&self, axis, keepdims, initial: f64)` method
- [x] Initial value added to sum / multiplied into product
- [x] Add tests: sum with initial, prod with initial

## GROUP 4: sum/prod/mean/var/std `dtype` parameter
- [x] Rust-core `reduce_sum_with_dtype` and `reduce_prod_with_dtype` select both accumulator and result dtype for bool, all signed/unsigned integer widths, and float16/32/64; inputs are cast before accumulation, integer overflow wraps at the requested width, and exact wide-integer sidecars survive the result
- [x] Rust-core `reduce_mean_with_dtype`, `reduce_var_with_dtype`, and `reduce_std_with_dtype` preserve accumulator/result precision, source-precision deviations, axis/keepdims, empty inputs, and ddof; interleaved complex sum/prod/mean/var/std support logical axes and real or complex accumulator dtypes
- [x] Complex nansum/nanprod/nanmean/nanvar/nanstd exclude an entire logical element when either component is NaN; complex cumsum/cumprod and their NaN-aware variants preserve component precision, logical axes, and first-element bits
- [x] Rust-core `nansum_with_dtype`, `nanprod_with_dtype`, `nanmean_with_dtype`, `nanvar_with_dtype`, and `nanstd_with_dtype` support real/complex accumulators, exact integer inputs, per-lane valid counts, and source-precision in-place deviations; default flattened float64 NaN statistics preserve masked positions and NumPy reduction order
- [x] Default float16 mean preserves source precision and float32 conversion-buffer ordering; variance/std retain integer count precision through division, including counts above float16's finite range
- [x] Default float32/64 and complex sum/prod/mean/var/std over axis tuples use joint C-order reductions, preserving logical axes, source precision, NumPy reduction ordering, and real-valued complex variance/std
- [x] Rust-core `nansum_axes`, `nanprod_axes`, `nanmean_axes`, `nanvar_axes`, `nanstd_axes`, and their explicit accumulator dtype variants support joint tuple reductions with per-output valid counts, whole-element complex NaN masks, source-precision deviations, empty/all-NaN lanes, and checked complex output capacities
- [D] Linux x86_64 floating-to-integer scalar/bulk casts and invalid-status queries model NumPy's signed/narrow intermediates, uint64 split, and source-specific uint32 vector-body/scalar-tail results. Real and complex `astype_with_casting` dispatch invalid errors through the existing runtime policy using the same typed source as conversion. Native verification on other platforms and NumPy's separate `ComplexWarning` channel remain open (`franken_numpy-5cf9h`).
- [x] Rust-core numeric `astype` conversions preserve logical complex elements, source precision, exact wide-integer sidecars, and the trailing physical component dimension. Complex-to-real conversion discards only imaginary components; complex-to-bool tests both components. The fallible API validates complex layouts and casting rules before invalid-status dispatch.
- [x] Complex `any`, `all`, tuple-axis truth reductions, `count_nonzero`, and nonzero-index APIs operate on logical complex elements rather than individual components, including scalar/empty inputs, complex64 underflow, NaN/Inf, and logical coordinates.
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
- [x] Rust-core `reduce_sum_axes_with_dtype`, `reduce_prod_axes_with_dtype`, `reduce_mean_axes_with_dtype`, `reduce_var_axes_with_dtype`, and `reduce_std_axes_with_dtype` support joint tuple reductions across all numeric accumulator dtypes; dedicated coverage includes exact integer wrapping, complex-to-real deviations, conversion buffers, float16 source precision, and impossible output capacities
