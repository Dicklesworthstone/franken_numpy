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
| `on_recovery_check_linalg_delegates.sh` | one-command build + conformance + re-measure of the 4 code-only 2-D linalg delegates (eigvalsh/eigh/cholesky/matrix_power). Exits non-zero on a failed shard, a shard that never ran, a failed deploy, or an unloadable module. | `scripts/on_recovery_check_linalg_delegates.sh` (set `CARGO` to drive it under a stub) | negative-tested: 4 fault scenarios exit 1, all-pass exits 0 (deadlock-audit-8hzmg) |

(Other scripts here — `*_compliance_matrix.py`, `check_compliance_matrices.sh`,
`regen_raptorq.sh`, `e2e/` — predate this work.)

## Drop-in check: numpy's own test suite against fnp

| script | purpose | how to run | last validated |
|---|---|---|---|
| `run_numpy_dropin_suite.sh` + `numpy_dropin_plugin.py` | runs numpy's OWN test modules twice — A/A lane (real numpy) and swap lane (fnp_python in numpy's place: public names resolve on fnp, private ones on numpy) — and lists every test that passes on numpy but fails with fnp. Each is a drop-in defect candidate. | `PYTHON=<python with numpy+pytest+hypothesis> scripts/run_numpy_dropin_suite.sh <fnp_python.so> <out_dir> [numpy.test.module ...]` | 2026-09-26, local .so at cc229e30, numpy 2.4.3, host thinkstation1: 121 modules, 47,238 A/A-passing tests (A/A failures 0), 53 divergences, **0 unowned** - every one is an identity row owned by deadlock-audit-rc0923-epic-71qy3.11 (the accelerator-vs-replacement decision): ufunc/function identity handed to override hooks, `__module__` identity, numpy-written pickles, ctypes, version. The run writes `<out_dir>/report.json` (per-module counts; each divergence's id, message, class, owner); this one is checked in as `artifacts/dropin-numpy-suite-2026-09-26.json`. History: 2026-09-24 at f11c7752, 24 modules, 421 divergences; 2026-09-25 at ccee9d2d, 57 modules, 31; the fixes these runs drove are the deadlock-audit-rc0923-epic-71qy3.8 commits |

Read a divergence before fixing it: a few are harness artifacts (a test that mixes a
private-name list with a public-name lookup, or reads private attributes of fnp objects),
listed in the plugin's docstring.

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
