# 2026-06-20 fnp-python `eigvals` correctness fix + perf-loss removal

Bead: ad-hoc BOLD-VERIFY gap hunt (follow-on to pinv size-gate)
Agent: `BlackThrush` / `cod-b`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`
Measurement host: `thinkstation1`, NumPy 2.4.3, load ~4,
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. (`hz2`, the NumPy 2.3.5 comparator,
was saturated at load ~33/16-core; the fix is a delegation so post-fix parity is
reference-independent.)

## Problem (CORRECTNESS, not just perf)

`fnp.eigvals` of a real 2-D square matrix ran the native Francis double-shift QR
(`eig_nxn` / `hessenberg_qr_iter`). That solver does **not reliably converge**:
its iteration budget (`EIGEN_QR_ITERATION_COEFF * n*n`) can be exhausted, after
which it silently returns the unconverged quasi-triangular diagonal as
"eigenvalues".

Order-independent power-sum invariants (sum(λ^k) must equal trace(A^k)) over 120
random real matrices (d = 16/32/64/128) exposed it:

| Check | Result (before) |
|---|---|
| power-sum k=1,2,3 relerr | **bad = 11/120**, worst relerr **15.2** |
| worst case | d=32 seed=13: k=1 (trace) OK 4.6e-15, but k=3 relerr **15.2** |

The trace (first power sum) is preserved — which is why a naive
`sum(eig)==trace` smoke test (and the existing conformance suite, which only
tests symmetric / diagonal / small / complex matrices) **missed the bug**.
Failures are matrix-dependent (even a symmetric 64×64 missed), so there is no
safe size gate.

It was also a perf loss at scale: d=200 ran **39.78x** slower (and wrong),
d=450 1.63x, d=600 1.73x, d=800 2.36x. The native "win" at small sizes
(0.6–0.8x) was on this unreliable path — a fast wrong answer.

## Fix

Delegate all real 2-D `eigvals` to NumPy's LAPACK `geev` (robust, and faster on
large n). `eig` already delegated; `eigvalsh` keeps its separate, reliable
symmetric QR path. One contained change to the `eigvals` pyfunction in
`crates/fnp-python/src/lib.rs`; the now-unused complex-output helper is retained
behind `#[allow(dead_code)]` for a future robust native solver.

## After

| Check | Result (after) |
|---|---|
| power-sum k=1,2,3 relerr | **bad = 0/120**, worst relerr **5.6e-13** |
| perf d=200 | 39.78x → **1.03x** (par) |
| perf d=600 | 1.73x → **0.93x** (win) |
| perf d=800 | 2.36x → **1.00x** (par) |
| perf d=16..128 | parity (was unreliable wins) |

Perf win/loss/neutral across d=16..800: 3 win / 1 minor-loss (d=96 1.16x) / 6
neutral — but the headline is **correctness restored** (0 wrong eigenvalues) on
a path that was silently wrong ~9% of the time.

## Validation

| Gate | Result | Artifact |
|---|---|---|
| `cargo test -p fnp-python conformance_linalg_advanced` | PASS 29/29 (4 eigvals) | `cargo_test_conformance.txt` |
| `cargo test -p fnp-python conformance_linalg_decomp` | PASS 38/38 (eigvalsh) | `cargo_test_conformance.txt` |
| power-sum invariant probe (120 matrices) | 0/120 bad | `summary.txt` |
| `cargo build -p fnp-python --release` | PASS (clean) | built 59.5s |
| non-square raises LinAlgError | PASS | numpy behavior preserved |

## Retry predicate

Do not re-enable the native `eig_nxn` path for `eigvals` without first proving
the Francis QR converges on a large random + adversarial corpus (defective,
clustered-spectrum, near-symmetric) to LAPACK tolerance with 0 failures. A robust
native unsymmetric eigensolver (exceptional-shift schedule, Wilkinson 2×2
deflation, aggressive early deflation) is a separate, large numerical effort —
not a `eigvals` wrapper tweak.
