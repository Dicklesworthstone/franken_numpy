# BlackThrush perf arc — scorecard + conformance verification (2026-06-21)

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
