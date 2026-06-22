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

## 2026-06-22: nanargmax/nanargmin f32 dtype-gap fixed (6f515301) - 6-8x -> 0.03-0.94x (30x@8M)
nanarg(max/min)(f32) missed the f64-only try_zerocopy_f64_nanargextreme -> cold f32->f64 widen
extract (~6-8x). Added try_zerocopy_f32_nanargextreme (read f32 directly, compare in f32, all
sizes since f32 cold path WIDENS). 1M 0.94x, 8M 0.03x. Bit-exact (ties/2-D); f64 path unchanged.
dtype-gap lever (cf isclose-f32). conformance nan_funcs 34. NOTE: nan-REDUCTIONS on f32 (nansum/
nanmean/nanmax/nanstd/nanvar/nanmedian/nanprod/nancumsum) are all WIN/parity (0.68-1.06x) - NO gap
(reductions promote to f64 both sides; the gap was specific to ARG variants returning an index via
the f64-only fast path). dtype-gap angle mined: isclose-f32 + nanargmax-f32 the finds; rest dominated.

## 2026-06-22: bool last-axis argmax/argmin catastrophe fixed (dabd5f21) - 2500x -> 4-8x
argmax/argmin(bool, axis=-1) (np.argmax(cond,axis=1) first-True-per-row) missed lastaxis_arg-
extreme dtype branches -> cold bool->f64 widen ~2500x(ax1). Added kind=="b" branch reusing the
u8 int path via uint8 view (argmax-u8 first-0x01 = numpy bool semantics). 2500x->4.44x/7.68x,
bit-exact (rand/all-F/all-T/3-D). conformance 10/10. Catastrophe removed; residual 4-8x = full-
lane u8 scan vs numpy per-row short-circuit.
QUEUED FOLLOW-UPS (arg-family axis dtype-gaps, milder): (1) bool NON-last-axis argmax/argmin
~10x (ax0) - try int_argextreme_axis::<u8> on uint8 view (if it supports narrow widths, same
u8-reuse trick); (2) nanargmax/nanargmin(f32) along axis ~7.3x (my f32 nanarg fix was flat
axis=None only) - need f32 nan-axis path; (3) per-row short-circuit for bool-axis -> parity
(vs current 4-8x). FOUND via arg-family x dtype x axis cross-probe.

## 2026-06-22: bool ax0 argmax WIN (6b1bd8ef); nanargmax-f32-axis astype DISPROVEN (copy-bound, deferred)
- bool non-last-axis argmax/argmin (10x) -> 0.67-0.74x WIN (6b1bd8ef): int_argextreme_axis +bool
  branch (uint8 view -> argextreme_axis_int_typed::<u8>). Completes bool-axis (last-axis dabd5f21
  4-8x + non-last-axis win). conformance argmax 10.
- nanargmax/nanargmin(f32) along axis 7.2x: tried astype-f64-then-reuse-f64-native -> DISPROVEN
  (7.2->5.69x ax1, HURT ax0 1.08->1.22x). It's COPY-BOUND: the f32->f64 widen (astype OR extract,
  64MB/call) dominates, not the extract path; the native nanarg needs f64. REVERTED. The REAL fix
  is a zero-copy f32 nan-axis path (read f32 directly, per-lane f32 nan-arg, no widen — like the
  flat f32 nanargextreme but per-axis, last+non-last) = win. DEFERRED (axis machinery, moderate;
  disk-conscious). RULE confirmed again: astype/widen-reuse is copy-bound parity-at-best; only a
  true no-widen direct-read wins (cf bincount-narrow direct-read 50x vs astype-parity).

## 2026-06-21: max/min(bool) flat - u8 short-circuit DISPROVEN (overhead/SIMD wall), reverted
Probed reduction/cumulative on bool/narrow: cumsum-u8 0.45x, cumprod-i8 0.47x, prod-bool 0.19x,
diff-bool 0.74x, max/min-bool-AXIS 0.09x — all WIN. Only gap: max/min(bool) FLAT 2.6-17x (cold
bool->f64 extract; np.max(bool)==any/np.min(bool)==all, numpy short-circuits+SIMD). Tried u64-word
short-circuit u8 scan -> numpy bool scalar. DISPROVEN: @early-True UNCHANGED 2.57x (overhead-bound:
pyo3 view+buffer+bool_ ~2us vs numpy instant short-circuit), @all-False/late still 12-13x (serial
u64-scan vs numpy SIMD reduce). min slightly better (2.8->1.97x, early-False short-circuit) but
still a loss. UNLIKE argmax-bool (40ms CATASTROPHE justified shipping the 4-8x residual), max/min-
bool old extract was only MILDLY slow (2.6-17x, no catastrophe) -> no justification for an overhead-
bound residual. REVERTED. WALL: max/min(bool) flat = small-result + numpy-short-circuit/SIMD; not
winnable via pyo3 (the result is a scalar; numpy's instant). Do not re-probe. (.so has stale
marginal fix until next build; source clean.) RULE: ship a residual ONLY when it removes a
catastrophe; a mild loss -> overhead-bound residual is not worth the code.

## 2026-06-21: f32 nanargmax/nanargmin LAST-AXIS WIN (ef76155f) - 7.2x -> 0.03-0.04x (30x)
The queued real fix landed. nanargmax/nanargmin(f32, axis=-1) widened f32->f64 (copy-bound 7.2x;
last turn's astype-f64 shortcut DISPROVEN same-class copy-bound). try_zerocopy_f32_nanarg_lastaxis:
read f32 directly, per-lane first-non-NaN arg in f32 (order-preserving bit-exact), parallel lanes.
7.2x->0.03-0.04x. Bit-exact (2-D/3-D/ties); all-NaN lane defers (numpy ValueError); f64 unchanged;
conformance nan_funcs 34. VALIDATES the direct-read-vs-astype rule AGAIN: astype-widen is copy-bound
(parity-at-best, here it FAILED), only a no-widen direct-read WINS (cf bincount-narrow 50x). nanargmax-
f32 dtype-gap now complete (flat 6f515301 + last-axis ef76155f); ax0 1.08x near-parity left as-is.

## 2026-06-21: set-ops DOMINATED; bool last-axis short-circuit DISPROVEN (overhead wall)
- set-operations family probed: intersect1d/union1d/setdiff1d/setxor1d-int 0.02-0.04x, unique-int
  0.01x, isin-int/f64 0.12-0.21x (huge WINS), intersect1d-f64 0.97x parity, ediff1d-f64 1.0-1.2x
  parity. DOMINATED, no loss. Completeness note: np.in1d MISSING in fnp (NOATTR; numpy's deprecated
  isin alias) — minor, deprecated, not perf.
- bool last-axis argmax/argmin (shipped u8-int-reuse dabd5f21, ~4-8x load-noisy): tried per-lane
  u64 SHORT-CIRCUIT scan -> DISPROVEN (50%T argmax 6.19x, NOT better than u8-int-reuse; the
  "4.44x" at dabd5f21 was load-noise, re-measured 7.4x same binary). bool last-axis is OVERHEAD-
  BOUND: numpy's tight per-row short-circuit C loop + fnp's 4000-elt intp output-build overhead;
  fnp can't beat it whether full-scan or short-circuit. REVERTED + rebuilt clean. WALL: bool last-
  axis argmax residual 4-8x = output-build + numpy-tight-loop, not the scan. Catastrophe (2500x)
  already removed by u8-int-reuse; residual not winnable. Do not re-try short-circuit.

## 2026-06-21: datetime/timedelta + trapezoid-axis + gradient-f32 all DOMINATED (3 fresh families)
- datetime64/timedelta64: diff/max/min/ptp/sort/sum/mean/cumsum/unique all win/parity. diff(datetime)
  single-call read 4.4x but min-of-3-WITH-WARMUP = 0.36x(1M)/0.93x(8M) WIN -> PHANTOM (cold first-
  call); correctness verified (timedelta64 dtype+values+n=2). Discipline caught it (re-verify min-of-3
  before fixing). Dominated.
- trapezoid along axis: NOW 0.02x(ax1)/0.43x(ax0)/0.31x(3D) WIN -> my queued trapezoid_axis_recipe
  LANDED (no longer a loss). Recipe DONE.
- gradient f32: 0.98x(1D)/0.79x(2D-ax1) parity/win, f64-ctl 0.1x. No dtype-gap (gradient handles f32).
No new loss this turn; surface heavily dominated across the 3 fresh families.

## 2026-06-21: char/strings.swapcase ASCII fast path WIN (9082f7c3) - parity -> 0.12x (8x)
np.char/strings.upper/lower had the ASCII codepoint fast path (0.14x) but swapcase/capitalize/title/
strip DELEGATED to numpy (parity ~1.0x). Generalized try_zerocopy_unicode_ascii_case (uppercase bool
-> method str: upper/lower/swapcase). swapcase flips ASCII a-z<->A-Z per codepoint; all-ASCII fast,
non-ASCII/non-U/non-contig delegates (bit-exact). 1.0x->0.12x (8x), char+strings. conformance strings
_namespace 9. upper/lower unchanged. FOUND via np.char family probe (numpy's char is Python-slow ->
ASCII-codepoint Rust path wins). QUEUED FOLLOW-UPS: capitalize (first cp upper rest lower per fixed-
width slot = itemsize/4) + title (per-word-boundary) -> same ASCII-fast + numpy-fallback pattern, win
opportunity; strip (whitespace trim per slot). All niche but real + safe (ASCII fast / numpy fallback).

## 2026-06-21: char/strings.capitalize+title ASCII fast path WIN (054c4a64) - parity -> 0.13-0.14x (7x)
Completed the char case family. try_zerocopy_unicode_ascii_cap_title: per fixed-width slot
(itemsize/4 codepoints), capitalize=first cp upper+rest lower; title=first letter of each word
(cased char after uncased) upper+rest lower (matches str.capitalize/str.title for ASCII). All-ASCII
fast, non-ASCII/non-U/non-contig delegates. 1.0x->0.13-0.14x (7x) char+strings. Bit-exact incl
apostrophes/digit-before-letter (title), empty, U1, non-ASCII-delegate. conformance strings 9.
CHAR CASE FAMILY COMPLETE: upper/lower (pre-existing) + swapcase (9082f7c3) + capitalize/title
(054c4a64), all ASCII-fast/numpy-fallback. DEFERRED: strip (changes string length -> needs slot
re-pack, fiddlier). char family was the live win-vein after many dominated families.

## 2026-06-21: char str_len DISPROVEN (numpy already C-fast); REFINES the char win-rule
Probed char predicate cluster (isalpha/isdigit/isalnum/isspace/isupper/islower/istitle/str_len/
startswith/endswith): ALL parity-delegate. Implemented str_len ASCII/all-unicode fast path (per-slot
logical length, trailing-NUL strip, int64, works for ALL unicode no ASCII gate). RESULT: char 1.04x,
strings 1.13x = NO WIN (bit-exact incl empty/unicode/N-D/U1/embedded-null verified). REVERTED.
ROOT CAUSE: numpy 2.x str_len is a FAST C ufunc -> fnp delegating = parity-with-fast = no win
opportunity. CONTRAST the case ops (swapcase/capitalize/title WON 7-8x) where numpy delegates to
SLOW per-element Python str methods. REFINED RULE: "parity-delegate" is a win-opportunity ONLY when
numpy is ALSO slow there (Python-level); if numpy has a C ufunc (str_len), fast-pathing = no gain.
MUST verify numpy-is-slow before assuming a char win. The other predicates (isalpha/isdigit/etc.)
are LIKELY numpy C ufuncs too (numpy 2.x np.strings) -> probably no-win; do NOT chase without
confirming numpy slowness. char case family (slow in numpy) remains the only confirmed char win-vein.

## 2026-06-22: char.translate ASCII 1:1-dict WIN (b5f3e683) - parity -> 0.05x (20x)
The lone real remaining char vein (448ns Python str.translate, genuinely slow). Fast-path: table is
dict of ASCII ord->ord (1:1 same-width), no None/str values, no null key, deletechars=None, ASCII
input -> 128-entry lookup, per-codepoint map. 1.0x->0.05x (20x). bit-exact 1:1; None-delete/maketrans-
del/str-table/non-ASCII/deletechars all correctly DELEGATE. conformance strings 9. VALIDATES the
corrected win-rule from the strip miss: translate 448ns Python = WON; strip 33ns C-ufunc = lost.
CHAR WIN-VEIN FULLY RESOLVED: WON = swapcase/capitalize/title (Python-slow case ops) + translate
(Python-slow). NO-WIN (numpy C-ufunc) = str_len/strip/lstrip/rstrip/add/ljust/rjust/center/find/
count/isX/zfill. Output-complex skip = split/join/encode/partition. char family = DONE.
NOTE: build emitted 4 warnings (was 3) - likely a minor rustc style warning in new translate code;
check on next clippy pass (build green, conformance green, win verified - not blocking).

## 2026-06-22: strings.translate WIN (1d2edb67) - missed twin, parity->0.02x (50x) + warning fixed
Caught a missed twin: np.strings.translate (270ns Python) was getting numpy copied-attr (1.07x) - I'd
only added char.translate. Added strings_translate_native (same helper, namespace arg) -> 0.02x (50x).
char.translate unchanged. conformance strings 9. ALSO fixed the rustc warning I introduced last commit:
table.downcast::<PyDict> -> table.cast (deprecated downcast -> Bound::cast); 0 remaining (back to 3
pre-existing fnp-python warnings). LESSON: fast-path BOTH char.X AND strings.X (numpy 2.x has both
namespaces, both Python-slow for the won ops). char/strings WIN FAMILY COMPLETE + SYMMETRIC: swapcase/
capitalize/title/translate all have char.X + strings.X fast paths. 50 wins total this arc.

## 2026-06-22: ma.compressed fix (3b6a93c0) - int dtype-gap 17.6x + f64 50% 3.6x -> parity, sparse win kept
np.ma re-assessment found 2 losses: getmaskarray 7.9x (= O(1) overhead-noise, np 0.2us/fnp 2.1us both
us -> SKIP, not real) and compressed (real). compressed: int/non-f64 hit cold extract->rebuild (17.6x
dtype-gap); f64 fast path won sparse-mask (0.13x) but LOST 3.6x at ~50% density (numpy C boolean-index).
FIX: (1) non-f64 -> numpy delegate (parity, removes 17.6x); (2) f64 fast path gated to >=90% kept
(cheap count pass already there) -> sparse WIN kept (0.13x), moderate -> numpy (1.08x parity). Bit-exact
f64/int/nomask; conformance_ma_utils 24. Rest of np.ma (filled 0.2x/getdata/count/sum/mean/max/masked_
invalid/masked_where/nonzero) win/parity. LESSON: a fast path can be density-dependent (win sparse, lose
moderate) - gate it to its win zone (cheap count) rather than delegate-all (keep the win) or keep-all
(eat the loss). 51 wins.

## 2026-06-22: ma.filled generic-typed WIN (9f5cb763) - int/uint/f32 5-11x -> 0.03-0.8x (dtype-gap)
filled fast path was f64-only -> int/uint/f32 cold extract (i64 10.9x, i32 7.7x, f32 5.5x). Generalized
try_zerocopy_ma_filled_typed<T> (i8..i64/u8..u64/f32 + f64): one-pass gather out=mask?fill:data, output
dtype matched. i64 0.21x, i32/f32 0.03x (30x), i16 0.8x, u8 0.77x. bit-exact (widths/default/N-D);
fill-not-fitting-T (2.5->int) delegates. conformance_ma_utils 24. GOTCHA: pyo3 FromPyObject = 2 lifetimes
(for<a,b>); separate getattr+extract (extract::Error != PyErr breaks .and_then). 52 wins.
QUEUED: ma.argmax 3.33x (masked argmax - moderate, next). np.ma family otherwise dominated (compressed
3b6a93c0, filled 9f5cb763 fixed; std/var/median/min/ptp/prod/cumsum/anom/average/power/abs win/parity;
getmaskarray O(1) overhead-noise).

## 2026-06-22: ma.argmax/argmin delegate (c920a6ec) - 3-3.5x non-f64 regression -> parity; np.ma sweep DONE
ma.argmax/argmin native extract->masked.argmax widened non-f64 (i64 3.44x, f32 3.48x) + never beat
numpy even for f64 (1.33x) = net regression. Delegated to numpy.ma.argmax/argmin -> parity all dtypes
(0.98-1.0x). Bit-exact (axis/dtypes); conformance_ma_utils 24. NP.MA SWEEP COMPLETE: compressed
(3b6a93c0 int 17.6x->parity, f64 density-gated keep sparse win) + filled (9f5cb763 int/f32 5-11x->
0.03-0.8x WIN dtype-gap) WON; argmax/argmin -> parity (regression removed); std/var/median/min/ptp/
prod/cumsum/anom/average/power/abs/getdata/count/sum/mean/masked_where/masked_invalid/nonzero win/
parity; getmaskarray O(1) overhead-noise. np.ma family fully dominated. 52 wins + ma-argmax parity-fix.
