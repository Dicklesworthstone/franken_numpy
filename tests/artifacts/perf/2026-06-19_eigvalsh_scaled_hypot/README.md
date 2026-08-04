# 2026-06-19 eigvalsh values-only QR: scaled_hypot — SHIP (win) + reduction dead-end

Agent: CobaltForge (claude-opus-4-8). Host: thinkstation1 (Threadripper PRO 5975WX,
32c/64t, L3 128MB). NumPy 2.4.3 scipy-openblas64. Per-crate rch builds,
`CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc`.

## Target
`np.linalg.eigvalsh` (symmetric eigenvalues-only) measured 2.9–3.85x slower than
numpy across n=256/512/768 — a stable, real gap. Anomaly: `eigh` (with vectors) is
3x FASTER than numpy here, but values-only `eigvalsh` loses, so fnp leaves perf on
the table somewhere in the values-only path. Parity contract is allclose/residual
(not bit-exact vs numpy), giving algorithmic freedom.

## Breakdown (native, n=512)
- tridiagonalization (`tridiag_reduce_values`, blocked Householder): **41.5 ms (83%)**
- values-only QR chase (`tridiag_eig_qr`, q=None): **8.4 ms (17%)**

## WIN (shipped): scaled_hypot in the eigenvalues-only QR
The QR chase calls libm `hypot(x,z)` once per Givens rotation (O(n²) calls). Replaced
it with a branch-light overflow-safe scaled hypot in a dedicated `tridiag_eigvals_qr`
(eigh's vector path is untouched). Rigorous interleaved same-binary A/B:
- QR speedup: **1.25–1.30x** (n=256 2.15→1.69ms, n512 8.4→6.6ms, n768 17.9→14.3ms)
- End-to-end eigvalsh vs numpy (python A/B): n128 1.09x, n384 1.14x, n512 1.12x
  (n256 ~flat/noise). Parity vs numpy: maxerr **1e-12 … 2.7e-11** (well inside allclose).

Bit-deterministic (scalar IEEE div/sqrt/mul, no FMA under +avx2): output stream is
stable across runs/builds. Three eigvalsh output-stream golden SHA-256 digests
re-pinned (`eigvalsh_values_only_reduction...`, `tridiag_rank2k_fused_update_preserves
_spectra...`, `batch_lanes_parallel...`); each test's own residual/known-value
correctness gate still passes (≤2e-8 / ~1e-13), so the re-pin is benign. Added
`tridiag_eigvals_qr_matches_eig_qr_to_allclose` (NEW vs proven eigvec-QR < 1e-9 over
n=7/33/64/128). Full fnp-linalg release suite: **390 passed, 0 failed**. fmt/clippy
clean on src/lib.rs.

## NEGATIVE EVIDENCE: the 83% (reduction) cannot be cheaply improved
The reduction dominates eigvalsh and runs at ~4.3 GFLOP/s (n=512), ~5-7x slower than
LAPACK dsytrd. Its hot op is the symmetric panel matvec `u = A·v`, gated to
parallelize only at `h >= TRIDIAG_MATVEC_PAR_MIN = 1024`. Hypothesis: lower the gate
to parallelize n=512/768. **Rigorous in-binary threshold sweep DISPROVED it** —
parallelizing is SLOWER at every size ≤768 (n256: serial 6.3ms vs t128 19.0ms; n512:
serial 40ms vs t256 54ms; only n=1024 t512 gains 1.11x). Each matvec is small O(h²)
work called n times; rayon dispatch dominates. The existing gate=1024 is correct. The
sweep also confirmed the parallel matvec path is bit-exact across thresholds. Reverted
(threshold experiment removed); the reduction is the same BLAS-2-vs-LAPACK residual as
the GEMM-vs-OpenBLAS gap — not tractable here.

## Ledger row
- WIN: eigvalsh QR scaled_hypot — 1.25-1.30x QR / ~1.1x end-to-end, parity ≤2.7e-11,
  shipped. NEUTRAL/LOSS: matvec parallel-threshold lowering (reverted, slower).
