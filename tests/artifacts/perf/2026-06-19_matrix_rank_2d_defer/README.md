# 2026-06-19 matrix_rank 2-D: size-gate native SVD vs defer to numpy — SHIP (win)

Agent: CobaltForge (claude-opus-4-8). Host: thinkstation1 (Threadripper PRO 5975WX).
NumPy 2.4.3 scipy-openblas64. `.probe` cdylib differential + rch conformance.

## Gap
`np.linalg.matrix_rank` on a 2-D f64 matrix measured **1.4–2.6x slower** than numpy
(n=512: 2.07x). Anomaly: `svd`, `svdvals`, `qr`, `cholesky` all PASS THROUGH to numpy
(fnp's bidiagonal SVD is ~2 ULPs off LAPACK and the strict tolist-repr parity oracle
can't tolerate it), but the 2-D matrix_rank path routed to the native bidiagonal
values-only SVD (`array.matrix_rank` → `svd_bidiag_values`), which is ~2x slower than
LAPACK gesdd for all but tiny matrices. rank is an *integer count* (no ULP/parity
concern), so this was a pure, avoidable loss.

## Measured crossover (native vs numpy, 2-D single matrix)
| n | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|
| native/numpy | 0.46 | 0.48 | 0.89 | 1.38 | 1.84 | (1.4) | 1.6 | 2.1 |

Native wins ONLY for max(M,N) <= 16 (it skips numpy's per-call SVD overhead);
loses from n=32 up. No reliable native regime above 16 (n=96 already 1.49x).

## Fix
Gate the 2-D native path at `MATRIX_RANK_NATIVE_MAX_DIM = 16`. Peek the ndarray shape
BEFORE the full extraction + finiteness scan (which alone is ~20% of numpy's runtime
at n~32-128) so a large 2-D matrix defers straight to numpy's LAPACK at true parity.
A second crossover check after extraction covers list/promoted inputs. The batched
(>=3-D) native path is untouched — it beats numpy's serial per-lane SVD 5-6x.

## Result (after fix)
| n | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | batch 64³ |
|---|---|---|---|---|---|---|---|---|---|
| fnp/numpy | 0.47 | 0.56 | 0.86 | 1.03 | 1.02 | 0.98 | 1.00 | **1.01** | 0.20 |

Large 2-D loss (up to 2.07x) → **parity**; tiny-matrix native win preserved
(n<=16: 0.47–0.86x); batched native win preserved (0.20x). Results match numpy
exactly for all sizes incl. rank-deficient (numpy is now the oracle for deferred
sizes). Conformance: matrix_rank_full_rank / _deficient / _zero_matrix all pass via
rch; 28/29 conformance_linalg_advanced pass (the 1 failure is `solve_triangular_complex`
— scipy not installed on the worker, pre-existing and unrelated). My hunk is
rustfmt-clean (the file has 148 pre-existing fmt diffs on main, none in my region).

## Ledger row
- WIN: matrix_rank 2-D defer-above-16 — up to 2.07x loss → parity, tiny+batched native
  wins preserved, exact numpy parity. Shipped.
