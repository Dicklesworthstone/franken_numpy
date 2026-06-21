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

## QUEUED: nanmedian flat DOUBLE-ALLOC fix (fnp-ufunc, blocked by YellowElk lock)

DIAGNOSED the nanmedian flat medium loss (1.1-1.3x): `UFuncArray::nanmedian(None)` does
`self.nan_filtered().median(None)` — nan_filtered() builds a new Vec (alloc #1), then
median(None) does `self.values.clone()` (alloc #2) + select. median(None) alone is ONE clone
(wins 0.78x at 131K after the gate fix); nanmedian pays TWO allocs -> the ~1.3x. FIX (clean,
in fnp-ufunc ~25803): give the filtered Vec straight to a select helper instead of re-cloning.
Either (a) extract median's select+interpolate body into `fn median_of_owned_values(mut v:
Vec<f64>, parallel_gate) -> f64` and call it from BOTH median (clone->call) and nanmedian
(filter->call), or (b) inline the select on the filtered Vec in nanmedian(None). Preserves the
1<<19 gate + even/odd interpolation. Expect nanmedian medium 1.3x->~0.8x (like median) AND
large 0.64x->~0.4x. BLOCKED: crates/fnp-ufunc/src/lib.rs reserved by YellowElk (til ~16:48).
Apply when free; bit-identical (same order statistic), verify conformance_percentile_median.

## np.char swept (2026-06-21): DOMINATED — upper/lower 0.03x (native), strip 0.8x, add/
## multiply/find/replace/str_len/startswith win/parity. No lever.

## GATE SWEEP COMPLETE (2026-06-21) — frontier clean after the par-select family.

Finished the medium-N (16K-262K) gate sweep. NO more mistuned-gate losses:
- axis median/percentile/nanmedian (1<<14 per-lane gates): all WIN at medium (0.1-0.85x) —
  many lanes give good parallelism, gates well-tuned.
- aggregates at medium: histogram 0.4-0.8x, cumsum 0.26x, cumprod 0.2-0.28x, digitize 0.9x,
  searchsorted 0.83-0.94x — all win/parity. sort-axis passthrough, vander parity.
- bincount medium 1.1-1.24x: NOT a gate (BINCOUNT_PARALLEL_MIN already 1<<19, correct; below
  it serial bincount just trails numpy's C impl ~1.2x; lowering the gate would fan-out WORSE).
- nanmedian flat medium 1.1-1.3x: serial NaN-filter+select+binding (not a gate), mild, U-shaped.
CONCLUSION: the mistuned-gate lever is EXHAUSTED — its wins were the 3 global par-select gates
(median a127d3d2 + single/multi-q percentile ab5e0c68), 2-9.6x catastrophes at 131K -> wins.
Residual medium-N losses (bincount, nanmedian) are serial-vs-numpy floors, not gates; mild;
nanmedian would need a fiddly fnp-python middle-band delegate. Low priority.

## SYSTEMATIC MISTUNED-GATE SWEEP (2026-06-21) — median was not alone

After the median gate fix, swept fnp-ufunc parallel gates at MEDIUM N (16K-131K, where
fan-out losses hide — the large-N gauntlet misses them). Found the SAME catastrophe in
percentile/quantile: PERCENTILE_GLOBAL_PARALLEL_MIN (single-q) + PERCENTILE_MULTI_Q_GLOBAL
_PARALLEL_MIN both 1<<17 -> 131K single-q 6.8-9.1x / multi-q 2.1x SLOWER. Raised both to
1<<19 (ab5e0c68): single-q 131K 0.79x, multi-q 0.82x, large unchanged. The par radix-select
(median/percentile/quantile share it) only wins from ~512K -> ALL its gates belong at 1<<19.
FALSE POSITIVES checked: cross 'loss' was a non-contiguous .T test artifact (contiguous cross
WINS 0.05-0.34x); count_nonzero 32K 3.6x is serial small-array overhead (gate already 1<<19).

GIT HAZARD (cost me a bad commit bd84e754, force-fixed to ab5e0c68): in a shared tree with
peer WIP, `git add myfile` then commit can sweep in PRE-STAGED peer changes. ALWAYS run
`git diff --cached --stat` and confirm it's ONLY your file BEFORE every commit. Recover a
contaminated pushed commit via reset --soft HEAD~1 + `git reset HEAD -- <peer files>` +
recommit + `git push --force-with-lease` (back up peer files to /tmp first).

## STATUS 2026-06-21: #1 unique SHIPPED (c6b87f00), #2 median SHIPPED (a127d3d2). #3 nanmedian deferred.

#2 median DONE — it was a MISTUNED KERNEL GATE, not binding: MEDIAN_GLOBAL_PARALLEL_MIN was
1<<17=131072 but par_select_median only wins from ~400K; at 131K-256K it ran 1.4-9.6x slower
(worst 5.95-9.6x right at 131072). Raised to 1<<19 -> 131K 5.95x->0.78x WIN, 262K ->1.18x,
large unchanged, bit-exact, conformance_percentile_median 24/24. (Residual: serial select
still mildly loses ~1.2x at some medium sizes e.g. 65K/262K — minor, optional binding delegate.)

#3 nanmedian DEFERRED — its medium loss (50K-512K 1.1-1.3x) is NOT a kernel gate: flat
nanmedian (outer_count=1) is serial (the NANMEDIAN_PARALLEL_MIN_ELEMS gate is the AXIS/lane
path). The flat loss is the NaN-filter Vec-alloc + serial select_percentile + extract/build
binding. It WINS small (10K 0.66x) and large (1M 0.64x), loses the medium band. Fix options:
(a) a flat-parallel nanmedian kernel path (par NaN-filter + par_select, HIGH gate ~1<<19) for
the large end — but large already wins; (b) a middle-band binding delegate in fnp-python
(lo<=N<hi -> numpy) — fiddly + mild. Low priority; revisit if a clean approach appears.

## STATUS (earlier): #1 unique SHIPPED (c6b87f00). #2/#3 median/nanmedian NEED MORE WORK.

#1 DONE: delegate exact-float64 unique that misses the parallel path -> numpy. Medium-N
0.98-1.01x parity (was 1.1-2.4x), large still 0.82x, int unchanged, conformance_setops pass.

#2/#3 REVISED — median/nanmedian are U-SHAPED, not "delegate below X":
  median:    10K 0.70x WIN, 50K 0.85x win, 131K 4.24x LOSS, 262K 1.1-1.3x loss, 524K 0.50x WIN, 1M 0.35x WIN
  nanmedian: 10K 0.66x WIN, 50K 1.12x loss, 131K 1.31x loss, 262K 1.15x loss, 524K 0.87x WIN, 1M 0.64x WIN
They WIN small (native beats numpy's Python median wrapper dispatch) AND large (par_select_
median kernel), but LOSE a MIDDLE band (~50K-512K). The median 131K 4.24x is extreme ->
SUSPECT the par_select_median kernel gate (MEDIAN_GLOBAL_PARALLEL_MIN in fnp-ufunc) is too
LOW, so 131K pays parallel fan-out on too-little work (cf the cheap-unary high-crossover
lesson). TWO candidate fixes to investigate (low load):
  (a) KERNEL (fnp-ufunc, cleaner if true): RAISE MEDIAN_GLOBAL_PARALLEL_MIN so medium N uses
      serial select — would fix BOTH median+nanmedian + benefit all callers. Verify 131K.
  (b) BINDING (fnp-python): delegate the MIDDLE band only (lo<=N<hi) to numpy — fiddlier,
      needs precise lo/hi under low load.
Prefer (a) if the 4.24x is fan-out. Re-measure under low load first (131K 4.24x may be partly
load noise at load~15). DON'T ship a simple below-threshold gate (would regress small-N wins).

## 1. unique f64 medium-N  [SHIPPED c6b87f00]

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

## QUEUED CANDIDATE: compress/extract medium-N gate (fnp-ufunc, blocked)
compress/extract WIN at 8M (0.27x, parallelized) but LOSE medium (131K-2M 1.2-1.8x). Likely
the parallel gate is too HIGH -> medium runs serial scalar compaction (the wall). When
fnp-ufunc frees: check if 131K-2M is serial; if so try lowering the compress parallel gate to
parallelize medium (may win like 8M, OR the privatized-compaction merge overhead negates it
-> MEASURE the crossover). NO AVX-512 here so unsafe vpcompress is unavailable; parallelism is
the only lever. SPECULATIVE (not a sure win).
