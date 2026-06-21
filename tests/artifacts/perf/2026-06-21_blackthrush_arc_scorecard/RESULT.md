# BlackThrush perf arc — scorecard + conformance verification (2026-06-21)

## UPDATE (later 2026-06-21): 24 wins; 3 fixes queued behind fnp-python peer-lock

The parallel-vs-single-threaded-numpy thread (from_raw_parts + rayon) added 8 wins beyond
the original 16: sqrt 8x, cheap-unary-class ~7x, binary no-copy + hypot 20x, argmax/argmin
2.5-3x, nanargmax/nanargmin 30x, sort 1.6-2.3x, argsort 2.2-4.3x, unique up to 3.5x. = 24
total.

GAUNTLET (later, fnp-python locked by YellowElk so read-only only): swept less-covered
families — fft/rfft ~parity (pocketfft delegate), polyval/roots/vander/gradient2/unwrap/
ediff1d/real_if_close/tri ~parity, cumprod 0.40x / nancumsum 0.25x / i0 0.30x / sinc 0.04x
WIN, corrcoef 1.26x = known BLAS-Gram floor (no-ship, no-C-BLAS directive). NO new actionable
losses. The vs-numpy surface is comprehensively dominated; remaining actionable work = the 3
QUEUED medium-N delegate fixes (unique/median/nanmedian, see medium_n_delegate_recipe.md),
all blocked on the fnp-python exclusive lock. Walls unchanged (SIMD-compaction, small-array
crossing, BLAS, dense LAPACK, sequential).

---

## FINAL TALLY (16 measured wins this arc, all bit-exact/allclose, conformance-green)

bincount 9x · trapezoid (1-D 50x / N-D 33x / axis0-kernel 1.8x / f32 250x + dtype-bug-fix) ·
gradient (1-D 20x / N-D 9x / scalar-dx) · sinc 50x · angle(complex128) 25x · pad (1-D-f64
4.5x / byte-level all-dtype 4.3x) · delete-single-int · insert-scalar 3.4x · datetime/timedelta
diff 2.4x (gated >=1<<14) · where(cond,arr,scalar) f32/int 20x. Plus earlier-arc: interp
~12-30x, roll→parity, module-cache, cov/argmax gate fixes, flatnonzero delegate, wide-int
argmax delegate, histogramdd list-sample bug, the 8-suite oracle-harness IndentationError
class fix, trapezoid axis=0 cache-kernel.

ARC-WIDE CONFORMANCE (re-verified together, 2026-06-21): 151 tests / 9 changed families /
0 fail — diff_gradient 23, interp_trapz 16, moveaxis_pad 19, concat_append 29, histogram
_bincount 32, angle 8, sinc 5, argmax 10, flatnonzero 9. No cross-regressions, no peer
breakage.

SURFACE STATUS: COMPREHENSIVELY DOMINATED across elementwise / reductions / transforms /
manipulation / construction / char-datetime-struct / f32-int-complex dtype-gaps / indexing /
set ops / broadcasting-binary. Remaining losses are all STRUCTURAL WALLS (see ledger):
SIMD-compaction (compress/extract), small-array pyo3 crossing (clip/passthrough small-N),
BLAS (matmul/dot/cov-gram = cod-a directive), pure-Rust dense LAPACK (batched inv/solve/
cholesky), sequential (cumprod/unwrap), forbid(unsafe) zero-init (sqrt). The forbid(unsafe)
+ no-C-BLAS walls need a project-level decision; not unilaterally changeable.

---


Agent: BlackThrush / cod-b. Recorded here (own artifact) because the shared ledger
(`docs/NEGATIVE_EVIDENCE.md`) and `docs/PERF_RELEASE_READINESS_SCORECARD.md` were
peer-dirty (YellowElk uncommitted) and `crates/fnp-python/src/lib.rs` was held
exclusively by YellowElk — committing those would capture peer WIP (ONLY-your-files).

## Lever vein this arc: serial/passthrough-fnp vs SINGLE-THREADED numpy (+temps/+python)

numpy runs most reductions/transforms single-threaded, and many of its functions are
Python wrappers that allocate temporaries; an fnp path that (a) reads the buffer
zero-copy and (b) parallelizes the work wins large. Wins shipped (all bit-exact or
allclose, conformance-green, only-my-file, on main+master):

| op | before | after | commit |
|----|--------|-------|--------|
| bincount (plain, large) | ~parity serial | 0.11x @4M (~9x) | 8bd0aaa9 |
| trapezoid 1-D | 1.2-1.78x loss | 0.02x @8M (50x) | f091be6b |
| gradient 1-D (unit dx) | passthrough ~parity | 0.05x @8M (20x) | a938669b |
| gradient 1-D (scalar dx) | passthrough | 0.09x @4M (11x) | 22528cde |
| sinc | 1.18x loss | 0.02x @4M (50x) | be6621ce |
| angle (complex128) | passthrough ~parity | 0.04x @4M (25x) | d84296c4 |

Earlier in the session (same/adjacent veins): interp ~12-30x, roll→parity, module-cache
−20% dispatch, cov/corrcoef gate fix, argmax last-axis + flat gate fixes, wide-int
argmax delegate, flatnonzero delegate, histogramdd list-sample bug fix, and the
8-suite oracle-harness IndentationError class fix.

## Conformance: ALL GREEN (132 tests, 9 families, 0 failures)

conformance_diff_gradient 23 · conformance_sinc 5 · conformance_angle 8 ·
conformance_interp_trapz 16 · conformance_histogram_bincount 32 · conformance_argmax 10 ·
conformance_argmin 10 · conformance_flatnonzero 9 · conformance_histogram2d_dd 19.

## Surface state: DOMINATED

Three broad full-threads sweeps this session (transforms, set/sort/index, linalg, fft,
random, complex elementwise, misc) show NO clear remaining loss — every probed op is
win or parity. Parity residuals are kernel floors (numpy SIMD sort, pocketfft) or the
sub-µs pyo3 dispatch wall (small-array passthrough, real/imag O(1) views). Complex
elementwise vein exhausted (abs/exp/conjugate are fast numpy ufuncs).

## Next lever (deferred — fnp-python locked)

gradient along the LAST axis for N-D (gradient(M, axis=-1) ~1.08x parity → parallel
per-row stencil, est. ~2-3x): a clean generalization of the proven 1-D path
(parallel over outer rows via par_chunks(L); 1-D stays interior-parallel). Blocked this
turn by YellowElk's exclusive reservation on crates/fnp-python/src/lib.rs.

## CONSOLIDATED LOW-LOAD SCORECARD (2026-06-21 ~15:42 UTC, load 7.1, 8M) — all wins confirmed

Representative vs-numpy ratios (fnp/np, <1 = win), measured at LOW load (reliable):
  sqrt 0.13 | negative 0.14 | hypot 0.05 | arctan2 0.04 | argmax 0.17 | nanargmax 0.03
  sort 0.56 | argsort 0.26 | unique 0.66 | median 0.56 | percentile 0.56 | cumsum 0.66
  nansum 0.07 | where 0.61
All 27 arc wins intact incl the median/percentile gate fixes (a127d3d2/ab5e0c68, 0.56x at 8M).
Full-crate sweep complete (elementwise/reductions/sort-select/char/indexing/set/utility/gates/
linalg) — surface comprehensively dominated. Remaining: nanmedian double-alloc + compress-gate
(paste-ready, handed off, behind YellowElk's live fnp-ufunc lock); structural walls (small-array
crossing, BLAS, SIMD-compaction[no-AVX512]). No new actionable lever available.
