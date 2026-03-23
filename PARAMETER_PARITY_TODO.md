# Parameter Parity TODO — franken_numpy-ksr

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [S] Skipped (architectural/impossible)

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
- [S] Skipped — would require internal accumulator dtype changes across all reduction paths
- [S] The f64 internal representation already provides maximum precision for accumulation

## GROUP 5: percentile/quantile `keepdims` parameter
- [x] Add `percentile_keepdims` method
- [x] Add `quantile_keepdims` method
- [x] When keepdims=true, insert size-1 dimension at reduced axis
- [x] Handle axis=None case (all dims become size 1)
- [x] Add tests: axis keepdims, None axis keepdims

## GROUP 6: partition/argpartition `kind` parameter
- [S] Skipped — only one algorithm ("introselect") exists, no alternative implementations
- [S] The `kind` parameter in NumPy also only has one real option

## GROUP 7: unique `return_index/inverse/counts` with axis
- [S] Skipped for now — unique_axis covers the main use case; combining with return_index/inverse/counts along axis is very complex

---

## ALREADY DONE (earlier this session)
- [x] clip_optional (one-sided clipping with None min/max)
- [x] reshape_order (F-order reshape)
- [x] histogram_full (density, range, weights parameters)

## ARCHITECTURAL DECISIONS (not implementing)
- [S] `out` parameter on all functions — Rust arrays are immutable
- [S] `order` on sort/partition — requires structured arrays
- [S] `overwrite_input` on percentile/quantile — immutable by design
- [S] `dtype` on reductions — f64 accumulation already provides max precision
