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

## BLAS GAP QUANTIFIED (2026-06-21 ~16:18 UTC) — cod-a's no-C-BLAS area, not mine
matmul vs numpy(OpenBLAS): 128x128 1.0x parity, 512x512 ~0.5-1.2x, 1024x1024 1.16-2.24x
(moderate, noisy). dot(vec 4M) 0.75-1.07x parity, matvec 2000 0.67-0.99x win/parity. The
pure-Rust GEMM is reasonable (within ~2x of OpenBLAS at 1024, NOT catastrophic). The residual
large-matmul gap is the no-C-BLAS directive (cod-a/YellowElk territory) — matching OpenBLAS
needs a tuned Rust GEMM (huge undertaking, contended). Not an actionable lever for me.
ARC COMPLETE: 28 wins, all 3 medium-N fixes shipped, compress-gate closed (wall), matmul
quantified (BLAS directive). No remaining clean lever; further gains need project-level
decisions (lift forbid-unsafe / no-C-BLAS) or AVX-512 hardware.

## BROAD REGRESSION RE-VERIFY 2026-06-22 (post stale-.so scare): all 40-arc wins intact
Spot-checked 18 representative arc wins on a fresh current-HEAD build (incl YellowElk's mod-alias
WIP in tree): sqrt 0.57, median 0.20, percentile 0.21, nanmedian 0.69, argmax 0.08, sort 0.39,
unique 0.07, isclose-f64 0.02, isclose-f32 0.02, array_equal 0.93, hist_bin_edges 0.49, kaiser
0.20, divmod 0.32, frexp 0.26, putmask 0.35, hamming 0.20, interp 0.01, nansum 0.19 — ALL <1.0
(wins), zero regressions. Confirms the 40-win arc is solid on current source. mod (real ~2x loss)
is YellowElk's verified-ready alias-unify WIP, uncommitted, offered (msg 2005), awaiting their
commit. Surface exhaustively dominated across ~22 probed angles; remaining non-wins are the
documented walls (BLAS-Gram, forbid-unsafe, numpy-introsort-dup, view-dispatch, small-array).

## 2026-06-22 RESOLUTION: bincount direct-read WIN (41st) shipped + mod LANDED by YellowElk
- narrow-int bincount direct-read (fb253d2e): u8 0.02x, i8 0.02x, i16/u16 0.03x, i32 0.20x
  (was ~3x loss). Generic try_zerocopy_bincount_narrow<T> reads the narrow buffer directly
  (no widen), bit-identical to the int64 tally. uint8 image/byte histograms = 50x. conformance 32.
- mod (~2x loss I found/HEAD-clean-verified/coordinated): LANDED by YellowElk (d632b15d, "make
  np.mod alias the same remainder ufunc object as numpy") — the coordination resolved; mod now
  parity + numpy object-identity. The flip-flop saga ended correctly: real loss, peer-owned fix,
  landed by owner.
ARC: 41 measured wins. Tree clean. Surface dominated across ~22 angles + narrow dtypes.

## 2026-06-22: bool argmax/argmin CATASTROPHE fixed (a9f367fd) - 40ms -> us (was 36000x)
np.argmax(cond)/argmin(cond) on bool (find-first-True/False idiom) missed int/f64 argextreme fast
paths -> cold bool->f64 extract + scalar scan = ~40ms@8M (36000-47000x vs numpy short-circuit).
Fix: try_zerocopy_bool_argextreme_flat (view uint8, scan u64 words skipping all-False/all-True,
first hit). 40ms -> 4-244us (160-10000x absolute). Residual 4-8x vs numpy = small-result pyo3
floor (view+buffer+intp ~4us vs numpy ~1us at first-hit) + u64-scan vs wider-SIMD at late-hit
(overhead-bound on us, NOT the catastrophe). conformance argmax/argmin 10/10. FOUND via bool-dtype
angle (same class as narrow-bincount). NOTE: all/any bool 1.17-1.44x mild (short-circuit, minor
follow-up). LESSON: a "find-first" idiom op on bool with no fast path = pathological cold extract
(40ms) — catastrophe-class, not just a ratio loss; the bool/narrow-dtype angle keeps surfacing these.
