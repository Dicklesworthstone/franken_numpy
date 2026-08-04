# 2026-06-20 fnp-python `cov(m, y)` two-operand zero-copy Gram vs NumPy

Bead: under directive `franken_numpy-ixs5y` ([perf][no-gaps])
Agent: `BlackThrush` / `cod-b`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`
Measurement host: `thinkstation1`, NumPy 2.4.3, load ~6, OMP/OPENBLAS=1.
(`hz2` / NumPy 2.3.5 was load-saturated; the kernel is the same regardless.)

## Problem (LOSE-gap)

`np.cov(a, b)` — the extremely common "covariance/correlation of two series"
idiom — was 4-17x SLOWER than NumPy:

| nobs | NumPy | FNP (before) | FNP/NumPy |
|---|---:|---:|---:|
| 10,000 | 51 us | 832 us | **17.4x** |
| 100,000 | 270 us | 2,797 us | **10.4x** |
| 1,000,000 | 5,761 us | 25,619 us | **4.45x** |

Root cause: `cov` had two zero-copy fast paths (a SIMD path for 16<=n_vars<128
and the general `cov_gram_rowvar_f64`), but **both are gated on `y is None`**.
The two-operand form (`cov(m, y)`) fell through to `native_cov_unweighted`, which
`extract`s both operands to owned arrays, `concatenate`s them (another copy), and
runs a generic Gram — the slow path. `np.cov(a,b)` is exactly
`np.cov(concatenate([rows(a), rows(b)]))`.

## Fix (radical lever: zero-copy two-buffer Gram)

Extracted the autovectorized 8-accumulator Gram into a shared
`cov_gram_from_centered(centered, n_vars, n_obs, ddof)` helper (single-operand
path now calls it — verified byte-identical). Added
`try_zerocopy_cov_two_rowvar_f64`: for rowvar=True it reads `m` and `y` f64
buffers directly (zero-copy `PyBuffer`), centers each variable row from its own
buffer into one `centered` array (NO raw-input stacking copy), and reuses the
shared Gram. Wired into the `cov` dispatch before the slow native fallback.

The arithmetic is byte-for-byte the single-operand fast path's, so it inherits
that path's allclose-verified conformance.

## After (head-to-head)

| nobs | NumPy_us | FNP_us | FNP/NumPy | verdict |
|---|---:|---:|---:|---|
| 10,000 | 50.7 | 30.3 | **0.596** | WIN |
| 100,000 | 293.0 | 263.0 | **0.898** | WIN |
| 1,000,000 | 4,836 | 4,438 | **0.918** | WIN |
| 4,000,000 | 62,738 | 50,713 | **0.808** | WIN |

Win/loss/neutral: **4/0/0** (was 0/4/0, all losses). All values match NumPy.

## Validation

| Gate | Result |
|---|---|
| two-operand cov correctness (160 random cases incl offset means, ddof 0/1/2) | 0 fails (allclose rtol 1e-10) |
| single-operand cov/corrcoef byte-identity after refactor | clean (allclose rtol 1e-12) |
| `cargo test -p fnp-python conformance_statistics` | 28 pass / 1 fail |
| edit regions clippy | clean |
| `cargo build -p fnp-python --release` | clean |

### Pre-existing failure (NOT introduced here, proven on HEAD)

`cov_corrcoef_python_container_keyword_outcomes_match_numpy` fails on its
"cov y ddof" case: `cov([1.,2.,4.], y=[2.,1.,0.], ddof=0)[0][0]` is
`1.5555555555555554` (fnp) vs `1.5555555555555556` (numpy) — a **1-ULP**
reduction-order difference in the n=3 list case, in the untouched
`native_cov_unweighted` path. Verified RED on HEAD with this change stashed
(identical output). The test uses exact float comparison; matching numpy's
BLAS summation bit-for-bit (likely FMA) would touch the pinned cov goldens and is
a separate precision concern, not part of this perf fix. My new array path
returns the same value, so it neither causes nor worsens this case.

## Retry predicate

The two-operand rowvar=True f64 form is now on the fast path. NOT yet fast:
`rowvar=False` with `y` (separate pre-existing scalar-shape bug in
`native_cov_unweighted` — `cov(a,b,rowvar=False)` wrongly returns a scalar
instead of 2x2; file separately), and weighted (`fweights`/`aweights`) forms
(deferred to numpy). The 1-ULP exact-match test needs a numpy-BLAS-bit-exact
reduction decision, separate from perf.
