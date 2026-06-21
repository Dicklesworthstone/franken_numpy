# Ready-to-apply: delegate medium-N native losses to numpy (BlackThrush)

Status: QUEUED — `crates/fnp-python/src/lib.rs` is exclusively reserved by YellowElk
(fresh lock ~2h, no commits yet). Land when free. All three are the SAME pattern: a native
path that WINS large but LOSES at medium N → add a size gate that delegates the losing range
to numpy (cf. the datetime-diff small-N gate fix 84acc931).

MEASURED vs numpy (fnp/np ratio; <1 = win). Load was 18–20 during measurement → RE-VERIFY
crossovers under low load before finalizing each gate.

## KERNEL PATH RULED OUT (2026-06-21, fnp-ufunc inspected)

Confirmed the medium-N losses are NOT the kernel: UFuncArray::unique f64 already
`par_sort_unstable_by` (fnp-ufunc:~24481) and median already `par_select_median`
(fnp-ufunc:~17348). The loss is purely the fnp-python BINDING — extract_numeric_array (numpy
-> UFuncArray copy) + build_numpy_array_from_ufunc (UFuncArray -> numpy copy). So the ONLY
fix is in fnp-python (delegate medium-N to numpy, or a zero-copy binding) — there is no
fnp-ufunc lever. Don't re-chase the kernel.

## 1. unique f64 medium-N  [SOLID — native loses the WHOLE medium range]

Native f64 `np.unique` (extract UFuncArray + serial sort+dedup) loses across all medium N:
50K 1.6-2.1x, 131K 2.0-2.4x, 262K 1.3-1.4x, 524K 1.1-1.3x. My parallel path (742fa7ac)
wins only ≥ 1<<20. So float64 below the parallel gate should DELEGATE to numpy, not run the
native serial path.

FIX (in `fn unique`, right AFTER the `try_zerocopy_f64_unique_flat` dispatch call):
```rust
// f64 below the parallel gate / non-contiguous / NaN: numpy's sort+dedup beats our native
// extract+serial across the whole medium range (measured 1.1-2.4x) -> delegate.
if item.is_exact_instance(&py.import("numpy")?.getattr("ndarray")?)?
    && numpy_dtype_is_f64(py, &item)
{
    return core_numpy_passthrough(py, "unique", args, kwargs);
}
```
Leaves int64 (large-range, non-counting-sort) on the native path UNTOUCHED (not measured —
do NOT delegate it without measuring). Verify: conformance_setops, np.unique f64 bit-exact,
and that ≥1<<20 still hits the parallel path (0.28-0.86x).

## 2. median small-medium  [re-verify under low load]

`np.median` f64: ~256K shows 2-3x loss (noisy, load 18), but 512K+ clearly WINS (524K 0.61x,
1M 0.38x, 16M 0.48x). So the native path is good from ~512K; only the small-medium band
loses. Gate: in `fn median`, BEFORE `extract_numeric_array`, if axis is None and the flat
size < MEDIAN_NATIVE_MIN (start ~1<<19=524288, re-measure) → `return fallback();`.

## 3. nanmedian small-medium  [re-verify under low load]

`np.nanmedian` f64: 256K 3.2x, 524K 1.3x, 786K ~0.98x, 1M 0.63x, 16M 0.70x. Native wins from
~786K. Gate: in `fn nanmedian`, before extract, flat size < NANMEDIAN_NATIVE_MIN (~1<<20=
1048576, re-measure) → `return fallback();`.

## Verify before commit (all)
- conformance_setops (unique) + the nan/stat suites for median/nanmedian.
- bit-exact / allclose vs numpy across the gated boundary (just-below = numpy, just-above =
  native), incl NaN (nanmedian) and even/odd length (median averages the two middle values).
- GOTCHA: insert helper fns ABOVE the `#[pyfunction]`/`#[pyo3]` attrs (else E0433).
- If fnp-python carries peer STALE WIP, stash-PUSH/POP (never DROP); recover a bad pop via
  `git show HEAD:path > path` + git add (dcg blocks reset --hard/restore/checkout--).
