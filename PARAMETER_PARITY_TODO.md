# Parameter Parity TODO — franken_numpy-ksr

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [S] Skipped (architectural/impossible)

---

## GROUP 1: unique(axis=) — Multi-dimensional uniqueness
- [ ] Implement `unique_axis(&self, axis: isize)` on UFuncArray
- [ ] Handle axis parameter: extract slices along axis, compare them, deduplicate
- [ ] Return unique slices preserving order of first occurrence
- [ ] Add tests: 2D array unique rows, unique columns, empty input

## GROUP 2: concatenate/stack dtype+casting parameters
- [ ] Add `concatenate_with_dtype(arrays, axis, dtype, casting)` method
- [ ] Validate casting via `fnp_dtype::can_cast` for each input → output dtype
- [ ] Cast each input array to target dtype before concatenation
- [ ] Add `stack_with_dtype(arrays, axis, dtype, casting)` variant
- [ ] Add tests: concatenate with dtype promotion, casting rejection

## GROUP 3: sum/prod `initial` parameter
- [ ] Add `reduce_sum_initial(&self, axis, keepdims, initial: f64)` method
- [ ] Add `reduce_prod_initial(&self, axis, keepdims, initial: f64)` method
- [ ] Initial value is used as the starting accumulator (before first element)
- [ ] Add tests: sum with initial, prod with initial, empty array with initial

## GROUP 4: sum/prod/mean/var/std `dtype` parameter
- [ ] Add `reduce_sum_dtype(&self, axis, keepdims, dtype: DType)` method
- [ ] Perform accumulation in the requested dtype precision
- [ ] Apply to all 5 reduction functions (sum, prod, mean, var, std)
- [ ] Add tests: sum in f32, mean in f64 from i32 input

## GROUP 5: percentile/quantile `keepdims` parameter
- [ ] Add `keepdims` parameter to `percentile` and `quantile`
- [ ] When keepdims=true, the reduced axis has size 1 in the output
- [ ] Add `keepdims` to `percentile_method` and `quantile_method` too
- [ ] Add tests: percentile keepdims, quantile keepdims

## GROUP 6: partition/argpartition `kind` parameter
- [ ] Add `kind` parameter to `partition()` and `argpartition()`
- [ ] Support "introselect" (default, already implemented)
- [ ] Accept "introselect" string without error
- [ ] Add tests: partition with kind="introselect"

## GROUP 7: unique `return_index/inverse/counts` with axis
- [ ] Extend `unique_with_info` to support `axis` parameter
- [ ] When axis is specified, return indices/inverse/counts relative to axis slices
- [ ] Add tests: unique_with_info with axis, return_inverse with axis

---

## ALREADY DONE (this session)
- [x] clip_optional (one-sided clipping with None min/max)
- [x] reshape_order (F-order reshape)
- [x] histogram_full (density, range, weights parameters)

## ARCHITECTURAL DECISIONS (not implementing)
- [S] `out` parameter on all functions — Rust arrays are immutable
- [S] `order` on sort/partition — requires structured arrays
- [S] `overwrite_input` on percentile/quantile — immutable by design
