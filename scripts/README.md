# franken_numpy scripts — vs-NumPy guards & recovery

Build-independent tooling produced during the 2026-06 BOLD-VERIFY work. The three
"guard" scripts run against a **built** `fnp_python` (point `PYTHONPATH` at the
`.so`, e.g. `.probe/`) and need **no cargo** — use them for fast post-build checks
and for the on-recovery verification after a build freeze.

## Guards

| script | purpose | how to run | last validated |
|---|---|---|---|
| `correctness_sweep_vs_numpy.py` | correctness regression guard — encodes the *subtle comparators* the Rust conformance suite lacked (see below). exit = #fails. | `PYTHONPATH=.probe python3 scripts/correctness_sweep_vs_numpy.py` | 0 fails / 27 |
| `perf_gap_sweep_vs_numpy.py` | vs-NumPy perf-regression sweep over the characterized op families + view-op `shares_memory` check. exit = #losses. | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.probe python3 scripts/perf_gap_sweep_vs_numpy.py [--full]` | API-validated (25/25 ops) |
| `on_recovery_check_linalg_delegates.sh` | one-command build + conformance + re-measure of the 4 code-only 2-D linalg delegates (eigvalsh/eigh/cholesky/matrix_power). | `scripts/on_recovery_check_linalg_delegates.sh` | bash-syntax-checked |

(Other scripts here — `*_compliance_matrix.py`, `check_compliance_matrices.sh`,
`regen_raptorq.sh`, `e2e/` — predate this work.)

## On-recovery procedure (run when a build freeze lifts)

1. **Reclaim disk** so cargo can build (the freeze was disk-gated): the big
   regenerable caches are `.rch-targets/franken_numpy-cod-b` (~14G) and `-cod-a`
   (~7.7G), plus `.probe/` (~2.7G stale `.so`). `cargo clean` or remove the cache.
2. **Verify the 4 delegates:** `scripts/on_recovery_check_linalg_delegates.sh`
   (builds fnp-python, runs `conformance_linalg*` + fnp-linalg tests, re-measures
   the four ops vs numpy — expect ~parity for single 2-D, WIN for batched).
3. **Sweep the whole surface:** `correctness_sweep_vs_numpy.py` (expect 0 fails)
   then `perf_gap_sweep_vs_numpy.py` (expect no LOSS; eigvalsh/cholesky read as
   LOSS only until step 2's build lands the delegates).
4. If agent-mail shows reservation drift, the live owner reconciles it (do not
   force `am doctor` on a live-owned mailbox).

## Why the correctness comparators are what they are (do not weaken)

- **eig/eigvals → power-sum invariants** `sum(λ^k)==trace(A^k)`, k=1..3, on
  **random non-symmetric** matrices. A native iterative QR once returned the
  unconverged diagonal on timeout → 11/120 silently-wrong eigenvalues; the
  symmetric-only conformance suite missed it, and `sort_complex`/greedy-match
  comparators give false results. Power sums are order-independent and exact.
- **View ops** (transpose/rollaxis/ravel/diagonal) → `np.shares_memory(out,in)`
  must be True; materializing a copy is both ~10⁴x slower and a semantics bug.
- **Selection ops** (take/choice) → must preserve input dtype (compute indices,
  then gather), not coerce to f64.
- **Special values** → singular factorization raises `LinAlgError`; `det(nan)`→nan;
  `cond(singular finite)`→`+inf`.

## Recurring failure mode this guards against

Perf **size-gates tuned against a dependency's perf cliff go stale** when the dep
is upgraded and **silently flip from win to loss** (det/inv/solve/eigvalsh lost
2–6x when NumPy 2.4.3 removed an OpenBLAS cliff). Run the perf sweep after **any**
numpy/BLAS bump.
