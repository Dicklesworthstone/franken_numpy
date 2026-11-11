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

## GROUP 8: where(condition, x, y) with complex operands
- [x] Complex conditions use one Boolean per logical element; real and imaginary components both participate in truth at the declared source precision
- [x] Complex branches broadcast logical shapes, promote numeric dtypes, and select complete component pairs; exact integer branches retain the existing sidecar-aware real selection path
- [x] Live NumPy coverage includes scalar/empty/broadcast inputs, mixed dtypes, source rounding, signed zero, NaN/Inf, exact wide integers, and malformed complex layouts (`complex_selection.rs`)

## GROUP 9: logical complex shape transformations
- [x] `reshape`, C/F/A `reshape_order`, transpose/permutation, matrix transpose, squeeze/expand tuple axes, swap/move/roll axes, ravel/flatten, and `broadcast_to` operate on logical complex elements while retaining the trailing physical component dimension and declared component precision. Reshape accepts one negative inferred extent and case-insensitive order tokens.
- [x] Flip, tuple-axis flip, left/right and up/down flips, and quarter-turn rotations validate logical dimensions and keep real/imaginary pairs intact. Duplicate flip axes are rejected; real transforms retain exact wide-integer sidecars.
- [x] `MaskedArray` reshape, flatten, expansion, squeeze, and transpose keep existing physical masks aligned with logical complex data, preserving mask dtype, raw payload, exact integer sidecars, and fill/hard-mask settings.
- [x] `complex_shape_transforms.rs` adds NumPy differential cases for logical shapes, component bits and signed zero, all numeric dtypes, C/F order, tuple and negative axes, scalar/empty inputs, malformed layouts, exact integer sidecars, and parallel-sized relocation kernels. Executable validation status is recorded with the implementing commit.

## GROUP 10: logical complex matrix and vector contractions
- [x] `matmul` handles logical complex dimensions, vector promotion, broadcasted batches, mixed numeric input dtypes, declared source precision, exact wide-integer conversion, empty contractions, and checked physical output capacity. Complex128 uses the existing complex matrix kernel; Complex64 accumulates in its component type.
- [x] `matvec` and `vecmat` preserve logical output dimensions and singleton batch axes; `vecmat` conjugates the vector. Complex `vdot` flattens complete elements and conjugates the first operand; `vecdot` contracts independently normalized logical axes and broadcasts the remaining dimensions.
- [x] The original matrix/vector coverage in `complex_matmul.rs` includes 11 tests with 1,866 NumPy differential cases plus direct arithmetic, malformed-layout, and allocation assertions. Finite comparisons use componentwise forward-error bounds derived from promoted input values; executable Rust validation remains recorded with the implementing commit.
- [D] Portable complex accumulation can differ from vendor BLAS in reduction order, rounding, cancellation, and exceptional-value behavior. These APIs do not yet establish backend-identical numerical or floating-error-policy parity.
- [D] `multi_dot` chains with three or more complex operands still plan using physical component shapes. Logical matrix-chain dimensions and the corresponding NumPy coverage remain open.

## GROUP 11: logical complex comparisons and Boolean operations
- [x] Rust-core `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`, `logical_and`, `logical_or`, and `logical_xor` broadcast logical complex shapes and return one Boolean per element. Mixed numeric inputs retain declared source precision and exact integer conversion; malformed layouts and unsupported dtypes are rejected.
- [x] Ordered comparisons use NumPy's complex lexicographic and quiet-NaN rules; corresponding invalid events are aggregated once through the existing floating-error policy. Dispatch planning uses logical output shapes and Boolean output dtype while retaining registry and signature validation.
- [D] Signaling-NaN invalid-event parity remains open for equality, logical ufuncs, and some ordered short-circuit cases; coverage requires bit-preserving fixtures because the JSON oracle canonicalizes NaNs.
- [~] `complex_comparisons.rs` adds live NumPy cases for broadcasting, numeric promotion, source precision, wide integers, NaN ordering, registry selection, and all floating-error modes. Runtime verification is pending: library rebuilds were killed before these tests executed, and the executor subsequently became unavailable. The all-targets metadata check identified two nonexistent test dtype variants; they are corrected to the actual `Structured` variant, but a fresh check is still required.

## GROUP 12: logical complex tensor products and explicit contraction axes
- [x] Rust-core complex `dot`, `inner`, `outer`, `kron`, and count-based `tensordot` preserve complete elements, logical scalar/array shapes, mixed numeric promotion, source precision, and exact wide-integer conversion. These products do not conjugate inputs; empty outputs/contractions and physical output capacities are checked.
- [x] New Rust-core `tensordot_axes` accepts paired explicit axis lists, including negative axes, for real and complex operands. It validates duplicates and dimension matches, preserves the original order of free axes, and retains typed source/result precision and exact integer sidecars.
- [~] Expanded `complex_matmul.rs` has 18 tests, adding NumPy cases for these products, explicit axes, real dtypes, scalar/empty inputs, exact integers, and allocation/error cases. Final executable validation is pending executor recovery; the numerical and floating-error limitations in Group 10 remain applicable.

## GROUP 13: logical complex Einstein summation
- [x] Rust-core `einsum`, `einsum_optimized` (`greedy`, `optimal`, and `auto`), and `einsum_path` interpret logical dimensions without exposing the physical component axis. Execution supports implicit/explicit outputs, repeated labels and diagonals, permutations, ellipsis broadcasting, scalar/empty inputs, and multiple operands.
- [x] Direct execution and optimized expressions without index reduction promote the full expression together; optimized reduction paths apply pairwise promotion. The no-reduction path matches NumPy's all-operand shortcut and avoids premature narrow-integer wrapping or half/float32 overflow. Paired arithmetic preserves source precision and exact real-integer conversion, pure relocations preserve component bits, and shape/layout validation and physical output capacity checks run before evaluation.
- [D] Full optimizer-path parity remains open: the existing `optimal` mode chooses pairs by retained intermediate size rather than performing NumPy's complete optimal-path search. Different contraction orders can also change mixed-dtype intermediates.
- [~] `complex_einsum.rs` adds live NumPy coverage for execution and planning, including signed zero, non-finite values, and invalid expressions. Final executable validation is pending executor recovery. Numerical comparisons use forward-error bounds; backend-identical accumulation and floating-error-policy parity remain open as described in Group 10.


---

## GROUP 14: logical complex extrema and index reductions
- [x] Rust-core `reduce_min`, `reduce_max`, their tuple-axis variants, `argmin`, `argmax`, `nanmin`, `nanmax`, `nanargmin`, `nanargmax`, and `ptp` compare complete complex elements with logical axes and output shapes. Complex64 inputs and peak-to-peak subtraction use component precision; ties preserve the first complete pair and logical index.
- [x] Ordinary extrema propagate the first whole-element NaN; NaN-aware extrema retain valid pairs and preserve NumPy's infinity replacement ties for argument reductions. Joint tuple reductions preserve C-order traversal, scalar axis rules, signed zeros, empty-axis errors, and checked output capacity.
- [x] All 24 tests in `complex_tuple_reductions.rs` pass against NumPy 2.4.3, including six new extrema tests covering source precision, tuple order, NaN/Inf ties, malformed layouts, and huge empty shapes.

## GROUP 15: logical diagonal, trace, and triangular operations
- [x] Rust-core `diag`, `diagflat`, `diagonal`, `trace`, and `trace_axis` preserve complete complex elements and logical dimensions. Diagonal extraction handles arbitrary axes and offsets; trace uses the declared component precision and NumPy's narrow-integer accumulator promotion while preserving exact wide integers.
- [x] `tril` and `triu` mask the final two logical dimensions, including batched arrays and NumPy's one-dimensional-to-square broadcasting. The existing contiguous-span and parallel kernels remain available for real matrices. Complex index helpers use logical ranks, and diagonal/triangular construction checks output capacity and extreme offsets.
- [x] `complex_shape_transforms.rs` covers all 19 tests against NumPy 2.4.3, including six new diagonal/trace/triangle tests and direct malformed-layout, huge-empty-shape, exact-integer, and extreme-offset checks.

## ALREADY DONE (earlier this session)
- [x] clip_optional (one-sided clipping with None min/max)
- [x] reshape_order (F-order reshape)
- [x] histogram_full (density, range, weights parameters)

## KNOWN PARITY DEBT / ARCHITECTURE NOTES
- [D] Rust-core generic complex unary dispatch remains incomplete for `square`, `sqrt`, `exp`, `log`, `sign`, and `logical_not`. Beyond the dedicated absolute-value and NaN/Inf/finiteness predicates, paired-element kernels, source precision, logical Boolean output shapes, and floating-error behavior still need implementation and NumPy coverage.
- [D] Standard Rust-core `sort`, `argsort`, `partition`/`argpartition`, and `unique` variants still interpret complex component storage as ordinary axes or values. `searchsorted` rejects a valid logical complex vector because its physical storage has rank two. Logical complex ordering, NaN placement, and index/inverse/count metadata remain to be implemented consistently; the separate `sort_complex` helper does not close these API gaps.
- [x] Public Rust-core `from_storage`, `from_storage_with_dtype`, and `to_storage` bridge `ArrayStorage::Complex64`/`Complex128` using logical shapes and precision-preserving component pairs. Typed numeric casts preserve exact wide-integer sources, while real sidecars and temporal guards retain their existing behavior. All 45 tests in `complex_reductions.rs` pass against NumPy 2.4.3, including six new bridge tests with 632 constructor-cast comparisons and the previously added narrowing-cast status/byte checks.
- [D] The Python typed-storage exporter still explicitly rejects complex storage; the completed public bridge above is the Rust-core API.
- [D] `out` parameter on array functions — Rust internals are immutable, but Python/NumPy-facing APIs still expose in-place output semantics that need explicit parity handling or documented wrapper delegation
- [D] `order` on sort/partition — structured/object-array ordering remains observable NumPy behavior and should stay tracked as parity debt
- [D] `overwrite_input` on percentile/quantile — immutable internals do not remove the need to match accepted parameter behavior and error surfaces
- [x] Multi-branch `select` broadcasts logical complex shapes, preserves complete component pairs and first-match precedence, validates Boolean conditions, and applies common choice promotion with a weak Python-float default. Source precision, exact integer conversions, default-cast overflow/underflow policy, and allocation/error precedence have live NumPy coverage (`complex_selection.rs`).
- [x] `take`, `fancy_index`, `take_along_axis`, and `compress` preserve complete complex elements and logical axes, including flattening, negative indices, broadcasting of non-axis dimensions, scalar inputs, and empty-axis bounds rules. Output capacities and malformed complex layouts are checked; real wide-integer sidecars survive gathers (`complex_gather.rs`).
- [x] `extract` uses logical condition truth and accepts short masks and excess false tails; `boolean_index` supports scalar and prefix Boolean masks, retains trailing logical dimensions, and rejects non-Boolean or mismatched masks. Complex64 condition precision, signed zero, non-finite values, exact integers, and empty shapes have live NumPy coverage (`complex_mask_indexing.rs`).
- [x] `choose` and `choose_with_mode` broadcast indices and all choices to a common logical shape, promote original choice dtypes together, preserve complex pairs and exact wide integers, and implement raise/wrap/clip index modes. Live NumPy coverage checks numeric promotion, source precision, broadcasting, index validation, empty outputs, and allocation limits (`choose_broadcasting.rs`). Homogeneous temporal/opaque choices retain their existing payload and dtype behavior.
- [x] Numeric `copyto` broadcasts logical complex elements and Boolean masks, applies source and destination precision, preserves exact wide integers, and respects casting policies. Live NumPy coverage includes selected-run conversion boundaries, scalar pre-casting, floating-point warning/error policy, and destination contents after errors (`complex_copyto.rs`).
- [x] `place` and `putmask` validate logical mask sizes and safe array casts, evaluate numeric masks at their declared precision, and assign complete complex elements or exact wide integers. Their distinct selected-position and flat-position value cycles, empty-value behavior, and validation precedence have live NumPy coverage (`complex_mask_mutation.rs`).
- [x] `put` and `put_along_axis` scatter complete numeric logical elements with source/destination precision, exact wide integers, repeated and negative indices, and logical axis/value broadcasting. Live NumPy coverage includes scalar versus array cast timing, destination state after errors, and uint32 conversion-loop boundaries (`complex_scatter.rs`).
- [x] Scalar `boolean_set` accepts full, prefix, scalar, and empty Boolean masks over logical dimensions, writes complete complex pairs, and preserves unselected exact wide integers. Python-float scalar packing truncates valid integer values, rejects invalid/out-of-range integers before writes, and reports floating overflow even for empty selections; live NumPy coverage checks all numeric dtypes and floating-error policies (`complex_boolean_assignment.rs`).
- [x] Scalar-sequence `fancy_set` and `put_mask` preserve complete logical complex elements and exact integer sidecars. Python-float values are packed before writes; short sequences cycle, flat assignment preserves earlier writes before a later bounds error, and floating overflow reporting remains per scalar. Differential cases are added to `complex_mask_mutation.rs`.
- [x] New Rust-core `boolean_set_array` assigns typed array values through full, prefix, and scalar Boolean masks, broadcasts into the indexed logical shape, preserves complex pairs and exact integers, and applies ndarray casting. Validation precedes mutation, cast status follows selected writes, and empty selections follow NumPy's scalar/pre-cast distinctions. All 14 tests in `complex_boolean_assignment.rs` pass against NumPy 2.4.3, including seven new array-assignment tests and uint32 conversion-run boundaries.
- [D] NumPy's separate `ComplexWarning` channel for discarding imaginary components is not yet modeled by the Rust floating-error runtime.
- [x] Rust-core `reduce_sum_axes_with_dtype`, `reduce_prod_axes_with_dtype`, `reduce_mean_axes_with_dtype`, `reduce_var_axes_with_dtype`, and `reduce_std_axes_with_dtype` support joint tuple reductions across all numeric accumulator dtypes; dedicated coverage includes exact integer wrapping, complex-to-real deviations, conversion buffers, float16 source precision, and impossible output capacities
