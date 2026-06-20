# 2026-06-20 fnp-python `pinv` 2-D size-gate vs NumPy

Bead: ad-hoc BOLD-VERIFY gap hunt
Agent: `BlackThrush` / `cod-b`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`
Measurement host: `thinkstation1`, NumPy 2.4.3, load avg ~5.5,
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

NOTE on reference host: the repo convention measures against NumPy 2.3.5 on
`hz2`, but `hz2` was saturated at load ~33 on 16 cores at the time of this slice
(unusable for clean A/B). The local box (load ~5.5) was the clean comparator and
runs an even newer NumPy (2.4.3). The fix is structurally a delegation, so the
~1.0x post-fix ratio is reference-version-independent (see below).

## Problem

`fnp.pinv` of a 2-D matrix routed *all* non-hermitian shapes (and hermitian
squares) through the pure-Rust dense path (`pinv_mxn` → `svd_mxn_full`, or
`pinv_hermitian_nxn`). That pure-Rust SVD/eigensolve beats NumPy's LAPACK only
for tiny matrices; above max-dim ~40 it scales far worse, and for larger
rectangular matrices it is **catastrophic**:

| Workload | NumPy | FNP (before) | FNP/NumPy (before) |
|---|---:|---:|---:|
| `pinv (600,400)` | ~41 ms | ~8.80 s | **~215x slower** |
| `pinv (400,600)` | ~38 ms | ~8.74 s | **~233x slower** |
| `pinv (128,128)` | ~2.3 ms | ~6.2 ms | 2.75x slower |
| `pinv (64,64)`   | ~0.46 ms | ~0.66 ms | 1.45x slower |
| `pinv herm(600)` | ~46 ms | ~322 ms | 6.98x slower |

The standalone `fnp.svd` (LAPACK-backed) is at parity; the loss was entirely in
the native 2-D `pinv` dense-SVD path, not in SVD itself.

## Fix

Size-gate the native 2-D `pinv` block to `max(m, n) <= 32` (the regime where the
pure-Rust path measurably wins by dodging numpy/LAPACK dispatch overhead, both
hermitian and non-hermitian). Larger 2-D matrices fall through to the existing
NumPy `linalg.pinv` delegation (LAPACK gesdd). Batched (>=3-D) `pinv` is
untouched and stays native (it wins decisively, 0.27-0.62x).

One-line change in `crates/fnp-python/src/lib.rs` (the 2-D branch guard).

## Head-to-head (after)

| Workload | NumPy_us | FNP_us | FNP/NumPy | Verdict | match |
|---|---:|---:|---:|---|---|
| `rect_600x400` | 45,773 | 47,074 | 1.028 | par (was 215x) | True |
| `rect_400x600` | 47,044 | 44,475 | 0.945 | win (was 233x) | True |
| `sq_128` | 1,978 | 2,024 | 1.023 | par (was 2.75x) | True |
| `sq_64` | 466 | 490 | 1.051 | par (was 1.45x) | True |
| `sq_32_native` | 129 | 123 | 0.957 | win (kept) | True |
| `sq_8_native` | 33 | 13 | 0.384 | win (kept) | True |
| `tall_5x3_native` | 26 | 8 | 0.290 | win (kept) | True |
| `batched_256x8x8` | 2,214 | 681 | 0.307 | win (kept) | True |

Win/loss/neutral: 5 win / 0 loss / 3 neutral (parity). The three formerly
catastrophic/loss large-2-D cases are now parity; the eight small/batched wins
are preserved. No regressions.

## Validation

| Gate | Result | Artifact |
|---|---|---|
| `cargo test -p fnp-python --test conformance_linalg_advanced` | PASS (29/29) | `cargo_test_conformance_linalg_advanced.txt` |
| pinv conformance + gate-boundary probe (22 cases) | PASS (0 fails) | `pinv_conformance_probe.txt` |
| `cargo build -p fnp-python --release` | PASS | built clean (1m31s) |
| clippy (edit region) | clean | only pre-existing `eq_op`/dead-code elsewhere |

Gate-boundary coverage: dim 32 (native) vs 33 (delegate), 32x33 / 33x32 / 40x8 /
8x40, 600x400, 400x600, rcond/rtol passthrough, hermitian 16/32/64/200, complex,
singular, batched — all bit/allclose match NumPy.

## Retry predicate

Do not re-tune the native 2-D `pinv` threshold or re-test the dense-SVD path as a
standalone perf lever. The native `svd_mxn_full` is intrinsically slower than
LAPACK gesdd for max-dim > ~40; closing the small residual on mid-size matrices
(33-63, currently ~1.0-1.1x via numpy delegation wrapper overhead) would require
replacing the pure-Rust dense SVD with a blocked/LAPACK-class SVD — a large,
separate effort, not a `pinv` gating tweak.
