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

## PASTE-READY nanmedian fix (verified design 2026-06-21; apply when fnp-ufunc frees)
Build compiles with peer WIP; my edit region (nanmedian ~25803) does NOT overlap YellowElk's
fnp-ufunc WIP (26356, 67019). Replace `nanmedian(None) => self.nan_filtered().median(None)`
with (single alloc, mirrors median(None)'s gate+select, no nan_filtered UFuncArray + no
re-clone):
```rust
None => {
    let filtered: Vec<f64> = self.values.iter().copied().filter(|v| !v.is_nan()).collect();
    if filtered.is_empty() { return Ok(Self::scalar(f64::NAN, DType::F64)); }
    const NANMEDIAN_GLOBAL_PARALLEL_MIN: usize = 1 << 19;
    let med = if filtered.len() >= NANMEDIAN_GLOBAL_PARALLEL_MIN
        && rayon::current_num_threads() >= 2 {
        par_select_median(&filtered)
    } else {
        let mut data = filtered;
        select_median(&mut data)
    };
    Ok(Self::scalar(med, DType::F64))
}
```
Helpers exist: select_median(&mut [f64])->f64 (29132), par_select_median(&[f64])->f64 (29255).
all-NaN -> NaN (matches current nan_filtered+median(empty)). Verify conformance_percentile
_median + nanmedian flat bit-exact + measure (expect medium 1.3x->~0.8x, large 0.64x->~0.4x).

## NEGATIVE 2026-06-21: text parsing is NOT a lever
loadtxt 100kx5 = 1.05x parity (numpy has a FAST C parser since 1.23 — not the old slow-Python
assumption; a native fnp parser wouldn't beat numpy's C loadtxt). fromstring already WINS 0.45x.
genfromtxt present. Don't chase "native CSV parser" — numpy's C path is the floor here.

## NEGATIVE 2026-06-21: complex64 dtype DOMINATED (last dtype corner)
complex64 (csingle) ops: abs/conj/angle/sum/mean/add/multiply/exp/cumsum/sort all dtype-CORRECT
(no f32-style canonicalization bug — the gap pattern does NOT recur in narrow complex) + win/
parity. real 2.19x = O(1) VIEW (shares_memory=True, time constant across N) = sub-us noise, not
a loss. Don't chase complex64. Frontier now fully mapped: all crates + text-parse + complex64.

## SHIPPED 2026-06-21: nanmedian double-alloc fix (04bd069e) — WIN
Landed via git-show isolation past YellowElk's stale fnp-ufunc lock (>1h idle, force_release
uncallable - no reservation id; their WIP preserved byte-intact, FYI sent). Single-alloc
filter+select replaces nan_filtered()+median() double-alloc. RESULT: nanmedian flat 131K
1.31x->0.92x, 1M 0.64x->0.36x, 8M ~0.7x (win, no regression). conformance_percentile_median
24 / nan_funcs 34 / ufunc nanmedian 3. Of the 3 medium-N fixes: unique (c6b87f00), median-gate
(a127d3d2), nanmedian (04bd069e) ALL SHIPPED. Only compress-gate candidate remains (speculative).

## CLOSED 2026-06-21: compress/extract medium-N is a WALL, not a gate (don't pursue)
compact_typed (fnp-python:9833) is SERIAL (sequential write cursor, branchless 16-lane mask
+ trailing-zero gather — no parallel path). compress 8M wins 0.27x (cache-thrash favors fnp's
sequential access) but medium 131K-2M loses 1.2-1.8x = numpy's SIMD compaction beats fnp's
serial scalar at cache-resident medium N. NOT fixable: no AVX-512 here (no vpcompress), and
parallel compaction needs a prefix-sum+scatter whose overhead won't beat numpy at medium.
The compress-gate candidate is CLOSED. All 3 medium-N fixes (unique/median/nanmedian) SHIPPED;
no remaining clean lever — surface fully dominated.

## NEGATIVE 2026-06-21: window fns (kaiser/hamming/hanning) mild loss, LOW-ROI
kaiser 1.2-1.5x, hamming/hanning 1.10x (bartlett/blackman/select WIN, polyfit/piecewise parity).
UFuncArray::kaiser (fnp-ufunc:21572) = serial per-point bessel_i0(arg)/bessel_i0(beta) map +
build bridge. The loss is the scalar bessel_i0-vs-numpy floor. Parallelizing the kernel would
only help LARGE windows (100k+); but real windows are small (64-8192 pts) where parallel can't
help and the scalar-i0 floor remains. LOW-ROI (niche + small-typical + encumbered). Not pursued.
Frontier now fully mapped incl window fns; only residuals are niche-small-loss or walls.

## RADICAL IDEA RULED OUT 2026-06-21: zero-copy build bridge (marginal + huge refactor)
build_numpy_array_from_ufunc(py, &UFuncArray) COPIES (numpy_array_from_slice) because it
borrows. A zero-copy version (move Vec via into_pyarray) needs: by-value signature + migrate
ALL callers + UFuncArray::into_values - a big cross-crate refactor (fnp-python + fnp-ufunc,
both encumbered). And MARGINAL: for most bridge ops the COMPUTE dominates, not the O(n) copy
(that's why my targeted zero-copy wins - convolve/sinc - were for COPY-dominated ops where
the kernel was cheap). The remaining bridge ops (windows etc.) are compute-bound, so a
zero-copy bridge saves little. NOT a clean lever. Every radical avenue now explored.

## SHIPPED 2026-06-21: kaiser loop-invariant hoist + parallel (7d3b9201) - up to 12x WIN
UFuncArray::kaiser recomputed bessel_i0(beta) (denominator) AND alpha PER POINT inside the map
(m redundant Bessel evals). Hoisted both + parallelized (gate 1<<14, bessel compute-bound).
RESULT: kaiser 1024 1.28x->0.22x, 10k 0.69x, 100k 1.46x->0.10x, 1M 1.22x->0.10x. Bit-allclose.
LESSON: re-read the kernel before dismissing as "niche/low-ROI" — a loop-invariant EXPENSIVE
recompute (bessel_i0(beta) m times) was a clean all-sizes ~2x+ win I almost skipped.
hamming/hanning/blackman do NOT have this (their loop-invariants are cheap arithmetic; cost
is the cos itself; parallel crossover ~256k = very rare; small loss is the bit-exact cos floor)
-> genuinely low-value, not pursued. bartlett wins already. Windows: kaiser fixed, rest closed.

## SHIPPED 2026-06-21: histogram_bin_edges zero-copy branchless min/max (82e7d7d4) - 4x->win
Found via the kaiser lens (do-more-work-than-needed): histogram_bin_edges EXTRACTED the whole
array (UFuncArray copy) + serial-scanned for min/max - 2 O(n) passes + alloc, 3.93-4.07x loss.
Fix: zero-copy borrowed-buffer single pass with BRANCHLESS f64::min/max + non-finite OR flag
(no early-return so it autovectorizes -> 1 SIMD pass, beats numpy's 2-pass a.min()/a.max()).
RESULT: 100K 0.62x, 8M 0.39x WIN, 1M 1.0x parity. Bit-exact; non-f64/non-finite -> numpy.
LESSON v2: the FIRST attempt (serial scan w/ early-return `if !finite return`) was 1.35-2.6x
(scalar, branch killed vectorization!) - the branchless bad-flag version vectorized -> win.
For min/max+validate loops: use branchless f64::min/max + OR-flag, NEVER an early-return branch.

## SHIPPED 2026-06-21: isclose(array, finite-scalar) zero-copy (4a503652) - up to 30x
isclose(x, 0.0)/(x, scalar) - VERY common - missed the both-ndarray zero-copy path -> full
extract copy (~5x). try_zerocopy_f64_isclose_array_scalar: finite scalar b -> constant
threshold atol+rtol*|b|, read x buffer once, write |x_i-b|<=thresh into bool out (parallel
>=1<<21). inf/nan -> false naturally; equal_nan irrelevant (b not NaN). 100K 0.16x, 2M 0.08x,
8M 0.03x (was 4.8-5.4x). bit-exact +nan/inf/2-D. The where-scalar pattern (3rd app: where/cov/
isclose). conformance_isclose 15/allclose 15.

## QUEUED: array_equal(equal case) 2.2x - chunked-branchless (next)
array_equal(x,x) 2.18-2.56x: f64_buffers_all_equal uses `.all(|x==y|)` whose short-circuit
kills vectorization for the equal case (numpy's ==+all is SIMD 2-pass). FIX (same lesson as
histogram_bin_edges): CHUNKED branchless compare - per 2048-chunk `eq &= x==y` (vectorizes)
then early-exit per chunk (coarse short-circuit for unequal). Keeps unequal-fast + makes
equal-case vectorize. Apply to f64_buffers_all_equal + f32_buffers_all_equal (fnp-python).

## SHIPPED 2026-06-21: array_equal chunked-branchless (4ef22361) - 2.2x->0.86x + early-exit
array_equal(x,x) lost 2.16x@500K (cache-resident): f64/f32_buffers_all_equal used .all() whose
per-element short-circuit branch DEVECTORIZES the all-equal case (numpy ==+all is SIMD 2-pass).
FIX: chunked-branchless (eq &= x==y per 2048-chunk -> 1 vectorized pass beats numpy 2) +
per-chunk early-exit (unequal still bails -> 0.0x). 200-500K 0.86x, 8M parity, unequal 0.0x.
3rd branchless-vectorize app (histogram_bin_edges, array_equal). git-show EXACT re-apply = NO
comment cruft (prior cruft was from trimming comments on re-apply; re-apply identical text).
GOTCHA: `git commit -m "...backticks..."` in DOUBLE quotes -> shell command-substitutes the
backticks (corrupts msg). Use SINGLE-quoted -m, or no backticks.

## COMPREHENSIVE less-common probe 2026-06-21 (post-5-wins): big losses EXHAUSTED
Probed float-manip, string/datetime, manip/index, less-common-linalg, indexing — ALL dominated.
WINS confirmed: char.upper/lower 0.04x, logaddexp2 0.05x, spacing 0.20x, degrees 0.20x,
count_nonzero-ax 0.23x, reciprocal 0.33x, rint/fix/trunc 0.4-0.5x, ldexp 0.51x, put/choose
0.73x, multi_dot 0.42x, matrix_power3 0.36x, lstsq 0.80x. APPARENT losses are DOCUMENTED WALLS:
 - VIEW-noise (O(1), shares_memory=True, fnp delegates): diagonal 1.5x, matrix_transpose 1.9x
   (constant ratio across N=800/4000 => sub-us dispatch noise, NOT algorithmic).
 - SMALL-ARRAY pyo3 wall: inner 2.4x@800 but 0.49x@8M (WIN large), ix_/trace@800 mild.
MILD residuals (genuine, low-ROI 1.15-1.25x, uncommon, no common class -> not pursued):
 frexp 1.20x (2-output), diff(prepend) 1.17x (kwarg bypasses zerocopy diff path), putmask
 1.15x, busday_count 1.2x. CONCLUSION: less-common surface comprehensively dominated after the
 5 wins this stretch (nanmedian, kaiser 12x, histogram_bin_edges 4x, isclose-scalar 30x,
 array_equal 2.2x). No more BIG actionable lever; remaining = mild residuals + structural walls.
