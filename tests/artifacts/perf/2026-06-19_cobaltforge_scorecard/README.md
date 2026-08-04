# 2026-06-19 CobaltForge — fnp-vs-NumPy perf scorecard (BOLD-VERIFY)

Agent: CobaltForge (claude-opus-4-8). Host: thinkstation1, AMD Threadripper PRO
5975WX (32c/64t, L1d 32KB/core, L2 512KB/core, L3 128MB). NumPy 2.4.3,
scipy-openblas64 (DYNAMIC_ARCH Haswell, MAX_THREADS=64). Python 3.13.7. Load ~2.8
(UNLOADED regime → numpy's threaded BLAS/LAPACK gets all cores).

Method: `.probe/fnp_python.so` (end-to-end Python surface) min-of-N microbench vs
numpy, plus native `rch exec -- cargo bench/test` for kernel-level A/B. ~150 ops
swept across `.probe/gap_sweep_cf{1,2,3}.py`.

## Headline: the codebase is at the pure-Rust Pareto frontier
Of ~150 ops swept, the overwhelming majority are at parity (0.9–1.1x) or FASTER than
numpy (often dramatically: unique 0.06x, nansum 0.09x, interp 0.13x, histogram2d
0.11x, char.upper 0.02x, intersect1d/union1d/setdiff1d 0.03–0.04x, choose 0.23x,
argmin/cumsum-axis 0.29–0.31x, quantile 0.26x, fill_diagonal 0.23x, eigh 0.15–0.32x).
Every remaining LOSS falls into exactly one of two fundamental categories below.

## LOSS category A — native GEMM vs OpenBLAS (matmul, dot, cov, corrcoef)
| op | n | fnp/np | note |
|---|---|---|---|
| matmul/dot (2-D f64) | 320–1024 | 2.3–6.5x | native GEMM ~76–90 GFLOP/s (bandwidth-bound, plateaus past 16 threads) vs OpenBLAS 260–500 GFLOP/s |
| cov / corrcoef | 1000 | 1.18–1.32x | gram via same native GEMM |

- Native GEMM only BEATS numpy when numpy is thread-starved (OPENBLAS_NUM_THREADS=1
  → 0.59–1.03x). With ≥2 BLAS threads numpy wins. The `PY_NATIVE_GEMM_MIN_FLOPS`/
  `MAX_DIM=1024` gate (fnp-python ~L40087) is a deliberate LOAD-AWARE design: a net
  loss unloaded, a win only thread-starved. **Left to maintainers; not flipped.**
- **Attempted & REVERTED** (see `../2026-06-19_gemm_pack_once_NOSHIP/`): pack-B-once +
  band-loop reorder. Rigorous interleaved (same-binary) A/B shows it REGRESSES the
  gated sizes (n768 0.68x, n1024 0.82x); only n>1024 (internal-use, beyond gate)
  gains ~1.18x. The OLD per-band kernel is already well-amortized; bit-exactness
  (kc=K, no K-blocking) caps cache efficiency. **Do not retry pack-once.**

## LOSS category B — pure-Rust dense linalg vs LAPACK (no external BLAS/LAPACK linked)
Stable across sizes (256/512/768), clean re-measure:
| op | fnp/np | competes with |
|---|---|---|
| eigvalsh | 2.9–3.85x | LAPACK dsterf (root-free QL). fnp ALREADY values-only (`tridiag_reduce_values`+`tridiag_eig_qr q=None`); gap is tridiagonal-solver tuning, not redundant work |
| cholesky | 1.88–2.16x | dpotrf. fnp already blocked + parallel TRSM + lower-tri SYRK + packed GEMM |
| solve | 1.67x | dgesv (LU + trsm) |
| slogdet | 1.46x | dgetrf LU |

These are the inherent residual of a dependency-free, bit-reproducible pure-Rust
implementation vs decades-tuned LAPACK Fortran. All heavily optimized already and
actively touched by peers (fnp-linalg commits through 2026-06-18). NOT low-hanging
fruit; a further win is research-grade (e.g. a dsterf-class root-free tridiagonal
eigenvalue kernel for eigvalsh — the most promising single target).

## LOSS category C — sub-microsecond dispatch overhead (low value)
`add(100)` 2.8x, `dot(100)` 3.1x, `add(1000)` 2.4x — all <1µs absolute (PyO3
marshaling fixed cost). Not worth chasing.

## Ledger summary
- WINS shipped this session: 0 (no tractable, non-contended, non-regressing win found).
- REGRESSIONS prevented: 1 (GEMM pack-once, rigorously disproven & reverted).
- Methodology fix banked: rch cross-invocation perf A/B is unreliable (152–293
  GFLOP/s spread for one build); always time OLD vs NEW interleaved in ONE binary.
