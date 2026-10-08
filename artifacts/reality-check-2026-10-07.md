# Reality check 2026-10-07 — FrankenNumPy v0.4.0

Subject: origin/main **526d814c4** (v0.4.0 tag at e84c5267c, released 2026-10-07). Third whole-project check
(previous: `artifacts/reality-check-2026-09-23.md`, epic `deadlock-audit-rc0923-epic-71qy3`). Every number
below was produced by a command on 2026-10-07/08 against that commit; nothing is carried over unverified.

Method: read AGENTS.md (suite + project) and README.md in full; exported origin/main with `git archive`
(the shared checkout at /data/projects/franken_numpy is 185 commits behind and dirty — see §8); built the
`fnp_python` cdylib from that export (`cargo build --release -p fnp-python --lib`, 2m42s, sha256
61b230d3…07fb); ran it in a clean venv (CPython 3.13.12, numpy 2.4.3, pytest 9.1.1) on host ts1 (64 cores,
load 8–15). Four read-only audits (architecture, beads/process, docs, README Rust examples) ran in parallel.

## 1. Verdict in one paragraph

The **Python product is a high-fidelity NumPy accelerator, and a good one**: all 499 `numpy.__all__` names,
numpy's own test suite passes with fnp swapped in except for 47 deliberate "identity" cases, 244 of 246
measured cells are byte-identical to numpy (the other 2 are a ledgered FMA divergence), and in this triage
map fnp is faster than numpy in 97 of 246 cells, at parity in 118, slower in 31. It is **not** what the
README describes: it is not a "clean-room reimplementation" running on the Stride Calculus Engine and a
risk-aware runtime — about 40% of measured cells compute entirely inside numpy, the SCE is called at 12 sites
in 202k lines, and the runtime's posterior/loss model never chooses an action. The project's identity
question (accelerator vs replacement, bead `.11`) has been open for 13 days and is now the binding
constraint: it owns all 47 remaining divergences and blocks the README truth pass (`.15`) and the
FFT/linalg work (`.12`). Meanwhile CI cannot report on the tip of main (27 queued runs, ~25 h push→G8) and
v0.4.0 was published to crates.io from a commit that has no CI run at all. The swarm is currently one active
agent doing careful small-call/parallel-floor perf work (1uf80, vc4p4) whose 234 new ledger rows are all
`maintenance` class; no incumbent-grade result has been banked since 2026-09-12.

## 2. Vision checklist

| # | Goal (source) | Status | Evidence | Beads |
|---|---|---|---|---|
| 1 | 100% `numpy.__all__` reachable (README L7, AGENTS) | WORKING | live: 499/499; 161 are numpy's own objects, 338 fnp objects; 594 pyfunctions, all registered | lock test (hasattr) |
| 2 | Full behavioural compatibility (README "The Solution") | WORKING except identity class | numpy's suite, 121 modules: A/A 47,236 pass, swap 47,189 pass, 47 divergences, all `identity`, 0 unowned (= 09-27); matrix 244/246 byte-identical | `.11` owns all 47 |
| 3 | "Hot operations land on the Rust engine" (README Recipe 4, tier table) | PARTIAL | perf attribution of 164 cells: ~54% compute in fnp (some via glibc libm), ~40% entirely in numpy (all fft, all 2-D linalg, f64/f32 matmul, large f64 add/sub/mul, 1-D min/max/argmax, partition, interp, dot, tanh, exp f32) | `.12` (blocked by `.11`), new polynomial bead |
| 4 | SCE is the compatibility kernel (AGENTS "CRITICAL NON-REGRESSION RULE") | WRONG_APPROACH at the Python surface; WORKING for the Rust API | fnp-python calls `fnp_ndarray` at 12 sites; numpy's ndarray is the shape authority for Python users; README now admits "two engines" | NO_BEAD (decided by `.11`) |
| 5 | Strict/hardened runtime with risk-aware decision engine + evidence ledger | PARTIAL | hardened enforces at 3 sites (verified live: linalg non-finite, 4 GiB cap, spawn>4096; bad env fails import); 4 sites record only; posterior never selects; posterior not exported | new runtime-engine bead |
| 6 | Fail-closed on unknown semantics | WORKING (Rust API) / by-design fallback (Python) | Python surface falls back to numpy for anything it does not own | — |
| 7 | Differential conformance vs real NumPy on every CI run | PARTIAL | G3 = 381 fixtures through Rust `UFuncArray`, not fnp_python; Python parity = G2 shards + manual drop-in harness (not in CI) | new drop-in-scheduled bead |
| 8 | CI G1–G9 green | UNPROVEN at tip | last full green 37531639701 @ eb61f0922 (10-06); 64 commits since; 27 runs queued; G1 red 10-07 (rustfmt, fixed); tag commit e84c5267c has no run | new CI-latency bead; `.4` |
| 9 | RaptorQ sidecars for every artifact | PARTIAL | 10 sidecars among 280 artifact files | README-truth bead |
| 10 | Bit-exact RNG vs numpy | WORKING | README example: 1000 PCG64DXSM normals bit-identical; spawn(8) streams equal; rng cells byte-identical | — |
| 11 | Bounded, fuzzed NPY/NPZ parsers for untrusted input | WORKING (Rust API) / NOT for Python | `fnp.load` (BytesIO, .npy path, .npz path) calls numpy.load + numpy's header parser; hostile header gives numpy's message in both modes | new hardened-load bead |
| 12 | Safe-Rust numeric core | WORKING (10 crates) / PARTIAL (fnp-python) | 1,144 `unsafe {` (+182 since 09-01), 421 without SAFETY comment, ~35 glibc libm FFI functions | `.21` (open, unassigned) |
| 13 | Faster than numpy where it matters (README §Performance) | PARTIAL at triage grade, UNPROVEN at contract grade | §5 map; ledger: 0 incumbent-win rows since 09-12; README table 1/28 contract-grade; whole-job rows from July | new scorecard bead; 1uf80, vc4p4 |
| 14 | Python distribution | PARTIAL | `pip install .` (Linux, cp313) + G9 wheel job; no PyPI (404), no macOS/Windows wheels | NO_BEAD → new wheels bead |
| 15 | Rust API usable as documented | WORKING (13/14) | README examples compiled on rch (vmi1153651/hz2): 13 compile and match stated values; Recipe 2 E0277 | README-truth bead |
| 16 | Docs truthful | PARTIAL | 23 contradiction/stale items (§6) | README-truth bead, `.15` |
| 17 | Release process | NO_BEAD | crates 0.4.0 published 2026-10-08 01:36–01:38Z; tag commit has no CI run | new release-gate bead |
| 18 | Owner decisions recorded | BLOCKED | `.11` open 13 days; `.13` (ADR-001 BLAS / native i64) open, 0 comments; ADR-001 says ACCEPTED and "never decided" at once | owner |

"If every open bead closed, would the vision be delivered?" **No.** Goals 4, 5, 7, 8, 11, 14, 17 had no
covering open bead before this check, and `.12`/`.15` cannot move until `.11` is decided.

## 3. What is working (evidence)

- **Surface and parity.** See checklist rows 1–2. The drop-in harness swaps module-level functions only;
  ndarray methods (`a.sum()`) still run numpy, so method-heavy tests do not exercise fnp. Within that scope,
  ~185 commits since 2026-09-27 introduced no swap-lane regression.
- **Byte-exactness.** 82 ops × 3 sizes (f64/f32/i64/bool, n ∈ {16, 4096, 2^20}; matrices 8–1024): 244/246
  byte-identical; `cov` (8,8) within 1e-12, (256,256) within 1e-6 per entry (DIV-COV-GRAM-NO-FMA; see §5).
- **Hardened mode** does what README §Architecture now says (row 5).
- **Rust crates**: 10 published at 0.4.0 on crates.io (verified via API); README Rust examples run (row 15);
  the v0.4.0 release notes record 9,003 per-crate tests passing on rch workers, fmt + clippy clean.
- **Tracker discipline**: every bead closed since 09-23 cites evidence (31/33 name a test or probe + CI run
  or worker + commit); 87% of perf commits since 09-23 have a ledger row, 100% since 09-29; `br dep cycles`
  empty.

## 4. What is not working, and why

### 4.1 The identity decision is the binding constraint
`deadlock-audit-rc0923-epic-71qy3.11` (accelerator vs standalone replacement) is open and unassigned since
2026-09-24. It owns all 47 drop-in divergences (pickles written by numpy, `ctypes` on bit generators,
`__module__ == 'fnp_python'`, protocol hooks receiving numpy's function objects, `np.version`), blocks
`.15` (README positioning) and `.12` (native FFT / 2-D linalg). `.13` (ADR-001: BLAS backend, native i64
streams) is open with 0 comments; ADR-001's header says ACCEPTED while its body says "never decided" and
leaves `[USER FILLS IN]`.

The code has already answered `.11` in practice: 427/594 pyfunctions fetch a numpy compute function in
their own body (141 are pure trampolines), every result is a `numpy.ndarray`, 161 public names are numpy's
own objects, and the two engines are drifting apart (raw `UFuncArray` mentions in fnp-python 267 → 217
since 09-01; `unsafe {` 962 → 1,144). See §7 for a recommendation.

### 4.2 CI cannot vouch for the tip
`ci.yml` gives every push to main its own never-cancelled concurrency group (deliberate, ci.yml:18-29). With
~30–60 pushes a day and an account runner pool that starts one gate every few hours (run 37590362189:
created 07:55Z, G1 13:44Z, G2 17:34Z, G3 20:43Z, G4 01:04Z, G5 still queued), the queue only grows: 27 runs
queued now; the last full green (eb61f0922) took ~25 h from creation to G8. Non-CI workflows that fire on
every push compete for the same runners ("Canonical Branch Policy": 162 cancelled / 36 success runs; 9
code-writing migration workflows from 2026-09-03 are still on main — `.4`, awaiting owner approval; the
"Make Random Core no_std" step never ran and fnp-random-core is still std). The v0.4.0 tag commit
(e84c5267c) and release commit (430746371) have **no** CI run; crates were published anyway.

### 4.3 The README describes a different product
23 verified items (full list in the README-truth bead). The consequential ones: fft/linalg/polynomial named
as native when they are numpy's (live `fnp.polynomial.chebyshev is numpy.polynomial.chebyshev` → True);
`strings/char/ma/testing` called identity re-exports when they are fnp overlays; `np.empty` said to zero-fill
(it is numpy.empty); `np.linalg.solve` said to be pure-Rust LU (it is LAPACK via numpy); untrusted `.npy`
said to go through fnp-io's bounded parser (it goes through numpy's); "differential conformance on every CI
run" (G3 tests the Rust engine, not the Python product); RaptorQ "every artifact" (10/280); unsafe "about
1,000" (1,144); stale counts everywhere (tests 9,081, pub fn 1,799, fnp-python 201,890 lines, closed beads
2,871, divergence rows 7).

### 4.4 Crown jewels that are decorative at the Python surface
- **Runtime engine**: 7 sites; 3 enforce inside `if hardened` with a hard-coded risk 1.0 vs threshold 0.5;
  4 record and discard the action; `decide_and_record_with_context` picks the action by threshold and only
  logs the posterior; `get_runtime_decisions()` exports no posterior/loss fields.
- **SCE**: 12 call sites in fnp-python (element_count ×9, broadcast_shapes ×2, broadcast_shape ×1); numpy's
  ndarray decides shapes/strides for Python users.

### 4.5 Performance proof class
0 `incumbent-win` ledger rows since 2026-09-12 (64 total); 234 rows since 09-23 = SHIP 213 / FIX 13 /
REJECT 4 / LOSS 2 / MEASURED 2, classes maintenance-self-speedup 217, maintenance-diagnostic 13.
`ratio_convention=` appears 0 times although `new_incumbent_win_rows_declare_a_winning_ratio_direction`
requires it from 09-26. README's headline table: 1 of 28 contract-grade. Scorecard whole-job rows (1.90–3.75x)
are from 2026-07-29/30.

### 4.6 Smaller defects found
- `cov` square 256×256: native route **1.40x slower** than numpy [p25 1.24, A/A 0.985] and not
  byte-identical (§5); the DIVERGENCES row's "1e-12 relative" is exceeded per entry (2.0e-11 at 256, 3.5e-10
  at 512).
- Never-compiled sources: `crates/fnp-python/src/dot_f64_passthrough.rs`, `crates/fnp-dtype/src/bin_test.rs`;
  25 unwired `allow(dead_code)` staging kernels in fnp-linalg; stale "Native fftshift" comment.
- Stale in_progress claims: `franken_numpy-ixs5y` (cod-a, idle 31 d), `tztko` (64 d), `9g7u0` (42 d),
  `ixs5y.409` (title says 3.72x slower; its last comment measured 1.14x).
- `41n96` (batched linalg bytes differ from numpy, no divergence row) open with 0 comments.

## 5. Triage map: fnp vs numpy, and who computes (NOT campaign grade)

Same process, `fnp_python` v0.4.0 vs numpy 2.4.3; per cell 15 rounds, alternating arm order, each arm the
min of 3 batches of ≥3 ms; ratio = median of per-round fnp/numpy times (<1 = fnp faster); an A/A null
(numpy vs numpy) ran in every round and sat in [0.97, 1.03] for 243/246 cells (outliers: linalg.inv 256
0.918, linalg.eigh 256 1.056, linalg.qr 256 0.917 — treat those three cells as undecided). Host ts1, 64
cores, load 8–15, unpinned, default thread pools in both libraries (fnp fans out over rayon; numpy is
single-threaded except BLAS), so large-n parallel wins here are optimistic for a loaded or small host (see
vc4p4's realistic-regime maps). "Who computes" = `perf record -e cpu-clock` over a 2 s loop of the fnp call
at that size, samples bucketed by shared object (OpenBLAS `blas_thread_server` startup spin and the
input-building numpy.random objects excluded): `fnp` ≥70% of compute samples in fnp_python.so; `fnp+libm`
fnp plus glibc libm; `numpy` ≥70% in numpy/OpenBLAS objects. Sizes 16 were not attributed.

Totals: 97 cells faster (<0.8), 118 parity (0.8–1.1), 26 slower (1.1–1.3), 5 slower (>1.3: arange@16 1.47,
cov@256 1.43, searchsorted@16 1.40, linalg.inv@256 1.39 [null-noisy], maximum@16 1.37).

| op | size: fnp/numpy time (who computes) |
|---|---|
| add f64 | 16: 0.90 · 4096: 0.96 (fnp) · 1048576: 1.00 (numpy) |
| subtract f64 | 16: 0.91 · 4096: 0.95 (fnp) · 1048576: 0.99 (numpy) |
| multiply f64 | 16: 0.92 · 4096: 0.98 (fnp) · 1048576: 1.01 (numpy) |
| divide f64 | 16: 0.92 · 4096: 0.97 (fnp) · 1048576: 1.00 (numpy) |
| power f64 | 16: 1.28 · 4096: 1.00 (fnp+libm) · 1048576: 0.33 (fnp+libm) |
| maximum f64 | 16: 1.37 · 4096: 1.14 (numpy) · 1048576: 1.03 (fnp) |
| arctan2 f64 | 16: 1.18 · 4096: 0.90 (fnp+libm) · 1048576: 0.06 (fnp+libm) |
| hypot f64 | 16: 1.22 · 4096: 0.80 (fnp+libm) · 1048576: 0.06 (fnp+libm) |
| less f64 | 16: 0.90 · 4096: 0.93 (fnp) · 1048576: 1.00 (numpy) |
| add f32 | 16: 0.88 · 4096: 0.90 (fnp) · 1048576: 1.01 (numpy) |
| add i64 | 16: 0.90 · 4096: 0.93 (fnp) · 1048576: 1.00 (numpy) |
| multiply i64 | 16: 0.90 · 4096: 0.95 (fnp) · 1048576: 1.00 (numpy) |
| floor_divide i64 | 16: 1.17 · 4096: 1.02 (numpy) · 1048576: 0.32 (fnp) |
| add scalar f64 | 16: 0.58 · 4096: 0.74 (fnp) · 1048576: 1.01 (numpy) |
| sqrt f64 | 16: 0.85 · 4096: 0.68 (fnp) · 1048576: 0.27 (fnp) |
| exp f64 | 16: 1.20 · 4096: 1.01 (fnp+libm) · 1048576: 0.08 (fnp+libm) |
| log f64 | 16: 1.28 · 4096: 1.01 (fnp+libm) · 1048576: 0.07 (fnp+libm) |
| sin f64 | 16: 1.22 · 4096: 1.08 (fnp+libm) · 1048576: 0.07 (fnp+libm) |
| tanh f64 | 16: 1.19 · 4096: 1.01 (numpy) · 1048576: 1.00 (numpy) |
| absolute f64 | 16: 0.85 · 4096: 0.93 (fnp) · 1048576: 1.06 (fnp) |
| floor f64 | 16: 0.85 · 4096: 0.91 (fnp) · 1048576: 1.07 (fnp) |
| isnan f64 | 16: 0.85 · 4096: 1.00 (fnp) · 1048576: 0.97 (fnp) |
| exp f32 | 16: 1.27 · 4096: 1.03 (numpy) · 1048576: 1.00 (numpy) |
| sum f64 | 16: 0.14 · 4096: 0.23 (fnp) · 1048576: 0.63 (fnp) |
| sum f32 | 16: 0.14 · 4096: 0.21 (fnp) · 1048576: 0.48 (fnp) |
| sum i64 | 16: 0.76 · 4096: 0.82 (numpy) · 1048576: 1.00 (numpy) |
| mean f64 | 16: 0.09 · 4096: 0.16 (fnp) · 1048576: 0.62 (fnp) |
| std f64 | 16: 0.06 · 4096: 0.15 (fnp) · 1048576: 0.61 (fnp) |
| var f64 | 16: 0.06 · 4096: 0.16 (fnp) · 1048576: 0.59 (fnp) |
| min f64 | 16: 0.64 · 4096: 1.01 (numpy) · 1048576: 1.00 (numpy) |
| max f64 | 16: 0.65 · 4096: 1.01 (numpy) · 1048576: 1.00 (numpy) |
| argmax f64 | 16: 0.72 · 4096: 0.83 (numpy) · 1048576: 0.99 (numpy) |
| prod f64 | 16: 0.81 · 4096: 0.90 (mixed) · 1048576: 0.99 (fnp) |
| cumsum f64 | 16: 0.75 · 4096: 0.31 (fnp) · 1048576: 0.26 (fnp) |
| nansum f64 | 16: 0.24 · 4096: 0.29 (fnp) · 1048576: 0.45 (fnp) |
| median f64 | 16: 0.19 · 4096: 0.51 (fnp) · 1048576: 0.37 (fnp) |
| percentile f64 | 16: 0.08 · 4096: 0.26 (fnp) · 1048576: 0.27 (fnp) |
| any bool | 16: 0.53 · 4096: 0.52 (mixed) · 1048576: 0.56 (mixed) |
| count_nonzero bool | 16: 0.97 · 4096: 0.99 (numpy) · 1048576: 0.68 (fnp) |
| sum axis0 f64 2d | 16: 0.45 · 4096: 0.56 (fnp) · 1048576: 0.21 (fnp) |
| sum axis1 f64 2d | 16: 0.42 · 4096: 0.27 (fnp) · 1048576: 0.20 (fnp) |
| sort f64 | 16: 1.16 · 4096: 1.02 (numpy) · 1048576: 0.38 (fnp) |
| sort i64 | 16: 0.94 · 4096: 1.01 (numpy) · 1048576: 0.37 (fnp) |
| argsort f64 | 16: 1.11 · 4096: 1.01 (numpy) · 1048576: 0.72 (fnp) |
| unique i64 | 16: 0.94 · 4096: 0.05 (fnp) · 1048576: 0.06 (fnp) |
| unique f64 | 16: 1.04 · 4096: 1.01 (numpy) · 1048576: 0.48 (fnp) |
| searchsorted f64 | 16: 1.40 · 4096: 0.76 (fnp) · 1048576: 0.09 (fnp) |
| partition f64 | 16: 1.01 · 4096: 1.00 (numpy) · 1048576: 1.00 (numpy) |
| isin i64 | 16: 0.08 · 4096: 0.26 (fnp) · 1048576: 0.12 (fnp) |
| where f64 | 16: 1.12 · 4096: 1.00 (fnp) · 1048576: 1.03 (fnp) |
| clip f64 | 16: 0.66 · 4096: 0.61 (fnp) · 1048576: 0.61 (fnp) |
| concatenate f64 | 16: 1.27 · 4096: 1.06 (fnp) · 1048576: 1.02 (fnp) |
| diff f64 | 16: 0.56 · 4096: 0.64 (fnp) · 1048576: 0.99 (fnp) |
| roll f64 | 16: 0.17 · 4096: 0.26 (mixed) · 1048576: 0.96 (numpy) |
| tile f64 | 16: 0.29 · 4096: 0.31 (mixed) · 1048576: 0.82 (mixed) |
| zeros | 16: 1.23 · 4096: 1.13 (mixed) · 1048576: 1.00 (mixed) |
| arange | 16: 1.47 · 4096: 1.11 (numpy) · 1048576: 1.00 (numpy) |
| linspace | 16: 0.43 · 4096: 0.45 (fnp) · 1048576: 1.00 (numpy) |
| histogram f64 | 16: 0.16 · 4096: 0.19 (fnp) · 1048576: 0.27 (fnp) |
| bincount i64 | 16: 1.02 · 4096: 0.70 (mixed) · 1048576: 0.57 (mixed) |
| interp f64 | 16: 1.18 · 4096: 1.00 (numpy) · 1048576: 1.00 (numpy) |
| dot 1d f64 | 16: 1.22 · 4096: 1.12 (numpy) · 1048576: 1.00 (numpy) |
| nan_to_num f64 | 16: 0.12 · 4096: 0.14 (fnp) · 1048576: 0.16 (fnp) |
| fft.fft c128 | 16: 1.04 · 4096: 1.01 (numpy) · 1048576: 1.00 (numpy) |
| fft.rfft f64 | 16: 1.06 · 4096: 1.01 (numpy) · 1048576: 1.00 (numpy) |
| rng.standard_normal | 16: 0.48 · 4096: 0.52 (fnp) · 1048576: 0.29 (fnp) |
| rng.random | 16: 0.48 · 4096: 0.60 (fnp) · 1048576: 0.09 (fnp) |
| rng.integers | 16: 0.29 · 4096: 0.45 (fnp) · 1048576: 0.76 (fnp) |
| matmul f64 | 64: 1.02 · 256: 1.00 (numpy) · 1024: 1.04 (numpy) |
| matmul f32 | 64: 1.05 · 256: 1.03 (numpy) · 1024: 1.00 (numpy) |
| matmul i64 | 64: 0.93 · 256: 0.02 (fnp) · 1024: 0.00 (fnp) |
| linalg.solve | 8: 1.11 · 64: 1.02 (numpy) · 256: 0.97 (numpy) |
| linalg.inv | 8: 1.10 · 64: 1.02 (numpy) · 256: 1.39 (numpy) |
| linalg.det | 8: 1.18 · 64: 1.02 (numpy) · 256: 1.00 (numpy) |
| linalg.eigh | 8: 1.09 · 64: 1.00 (numpy) · 256: 0.93 (numpy) |
| linalg.svd | 8: 1.08 · 64: 1.01 (numpy) · 256: 1.00 (numpy) |
| linalg.qr | 8: 1.05 · 64: 1.01 (numpy) · 256: 0.99 (numpy) |
| linalg.cholesky | 8: 1.17 · 64: 1.06 (numpy) · 256: 1.00 (numpy) |
| linalg.norm | 8: 1.17 · 64: 1.15 (numpy) · 256: 0.98 (numpy) |
| transpose copy f64 | 64: 1.00 · 256: 1.00 (numpy) · 1024: 1.00 (numpy) |
| einsum ij,jk | 8: 1.11 · 64: 0.38 (fnp) · 256: 0.26 (fnp) |
| cov f64 | 8: 0.13 [CLOSE(1e-12)] · 64: 1.07 (numpy) · 256: 1.43 (fnp) [CLOSE(1e-6)] |

Reading the map: small-n wins come from numpy's Python-level dispatch (np.sum → _wrapreduction) that fnp's
native entry skips; small-n losses (1.16–1.47x at n=16 for transcendentals, maximum, power, searchsorted,
concatenate, arange, zeros, dot, interp, sort) are the wrapper floor that bead `1uf80` is working on; the
large-n transcendental/hypot/arctan2 wins are multi-core fan-out against single-threaded numpy. Cells that
delegate cost 0–3% at n ≥ 4096 in most cases (up to 15%: maximum@4096 1.14, dot@4096 1.12, arange@4096
1.11, linalg.norm@64 1.15) and 1.0–1.27x at n = 16 (tanh 1.19, exp f32 1.27, interp 1.18, dot 1.22).

## 6. Documentation

All 23 items, with README line numbers and the code/live evidence, are in the README-truth bead created by
this check. Items that depend on `.11` (positioning: "clean-room reimplementation", "drop-in", the two-engine
story) stay with `.15`.

## 7. Recommendations to the owner (decisions only you can make)

1. **Decide `.11`.** Recommendation: declare `fnp_python` a **NumPy-compatible accelerator** (numpy is a
   required runtime dependency, `numpy.ndarray` is the array type, the 47 identity divergences become
   documented, permanent behaviour) and position the Rust crates as the **standalone NumPy-semantics library**
   (where the SCE, the strict/hardened runtime and fnp-io's bounded parsers genuinely are the core). Why:
   that is what the code is (§4.1); a "replacement" would need an fnp-owned ndarray, dtype objects and a
   C-API that pandas/SciPy/scikit-learn can link against — a different, multi-month project that would put
   the 47,189-test parity at risk; and every remaining divergence resolves by declaration. If you choose
   "replacement" instead, the first bead is an fnp-owned array type design, and most of §4.3 stays false
   until it lands.
2. **Decide `.13`.** A C BLAS/LAPACK backend contradicts the suite rule against linking C to win benchmarks;
   recommend rejecting it in ADR-001 explicitly. Native i64/u64 storage in `UFuncArray` matters only to
   Rust-API users with integers above 2^53 — schedule it only if Rust-API adoption is a goal.
3. **Approve or reject `.4`** (delete the 9 migration workflows) — they consume the runner pool CI needs.
4. **Pick a CI-latency option** (new bead): self-hosted runners on the worker fleet keep per-commit coverage.
5. **Approve deletions** listed in the dead-sources bead (two never-compiled files).

## 8. Environment hazards found during the check (not beads; need your action)

- **Shared checkout `/data/projects/franken_numpy` is unsafe to commit from.** Its `main` is 185 commits
  behind origin; its working-tree `crates/fnp-python/src/lib.rs` is a 2026-09-23 copy (11,270 insertions /
  19,917 deletions against its own HEAD); its `.beads/beads.db` is behind origin's JSONL (missing the newest
  comments on 1uf80, 6y5wp, 71qy3.23). A `git commit -a`, or a `br sync --flush-only` + commit, from there
  would silently revert work. The active agent works in `/data/projects/.scratch/franken_numpy-TealKnoll-rc0923`.
  Refreshing the checkout is destructive to that stale working tree, so it needs your explicit go-ahead.
- **Stuck rch job** `#30054041371280031` on `vmi1227854` (wrapper `rchw-7f639065-8669-43e9-8770-818ff2ae9804`),
  started by this check's README-examples audit from a scratch tree outside /data/projects. `rch jobs
  cancel` and `rch jobs recover` both fail ("cancellation was not acknowledged" / "source authority must be a
  canonical absolute path …/tree/" — a trailing-slash bug). It holds vmi1227854's only slot until an operator
  clears it.

## 9. Bridge plan

Ordered by what unblocks trust in everything else. Beads were created under the epic named below; each is
self-contained (background, evidence, acceptance probe, a negative case a naive fix would fail) and ships
its own tests — no separate test beads (AGENTS.md forbids scope-splitting).

Epic: **`deadlock-audit-3ltbd`** (label `reality-check-2026-10-07`). Closes only after this check's
measurement set is re-run at the then-current tip (epic description, "Closure condition").

| Order | Bead | P | What closes it |
|---|---|---|---|
| 0 | `deadlock-audit-rc0923-epic-71qy3.11`, `.13` (existing, owner) | P1/P2 | Your decisions (§7) |
| 1 | `deadlock-audit-3ltbd.1` CI tip verdict latency | P1 | 7 days of tip commits with a completed G1–G9 run within 6 h; a breaking commit still named; no gate threshold changes |
| 1 | `deadlock-audit-3ltbd.2` release gate | P1 | script refuses e84c5267c (no CI), passes eb61f0922; G7 `skipped` ≠ pass |
| 1 | `deadlock-audit-3ltbd.4` drop-in suite on a schedule / per RC | P1 | job fails on a re-introduced known defect; report refreshed by the job |
| 2 | `deadlock-audit-3ltbd.3` README/docs truth (23 items, decision-independent) | P1 | every item corrected to a reproducible number; re-run of the bead's commands finds no mismatch |
| 3 | `deadlock-audit-3ltbd.5` runtime engine selects by expected loss (or docs say "flag") | P2 | evidence-flip test fails on v0.4.0 and passes after; posterior exported |
| 3 | `deadlock-audit-3ltbd.6` hardened-mode `load()` enforces fnp-io bounds | P2 | fuzz-seed corpus: hardened refuses per bound with ledger event, strict == numpy; 5,000-member npz refused without opening a member |
| 4 | `deadlock-audit-3ltbd.7` incumbent-grade v0.4.0 scorecard | P2 | loss map + 3 whole-job rows banked as incumbent-win/LOSS with all markers; README perf table rebuilt from them |
| 4 | `deadlock-audit-3ltbd.8` native routes for numpy.polynomial family functions (measure first) | P2 | byte parity shard + incumbent-win or REJECT rows per function |
| 4 | `deadlock-audit-3ltbd.9` cov square 256² loses 1.40x and diverges | P2 | square shapes delegate (bytes equal), long-observation win kept, ledger bound restated |
| 5 | `deadlock-audit-3ltbd.10` wheels (linux x86_64/aarch64, macOS arm64, Windows x64) → TestPyPI → PyPI on approval | P2 | per-platform drop-in + libm-parity reports; no SIGILL on non-AVX2; blocked by .2, .3, `.11` |
| 5 | `deadlock-audit-3ltbd.11` never-compiled sources + unwired linalg staging kernels | P3 | owner-approved deletions or measured wiring; hygiene ceiling lowered |

Existing beads this plan relies on (unchanged by this check): `.11`, `.13` (owner decisions), `.4`
(workflows), `.12` (native fft/2-D linalg, measure-first), `.15` (positioning README items), `.21` (unsafe
audit — 421 blocks without SAFETY comments; currently unassigned), `1uf80` (small-call floor — the n=16
losses in §5), `vc4p4` (parallel floors), `41n96` (batched linalg bytes).

## 10. Reproducing this check

- Build: `git archive origin/main | tar -x -C <dir>`; in `<dir>`: `RCH_CARGO_WRAPPER_BYPASS=1
  PYO3_PYTHON=<python3.13> cargo build --release -p fnp-python --lib`; copy `libfnp_python.so` to
  `<so_dir>/fnp_python.so`.
- Drop-in: `PYTHON=<venv python with numpy 2.4.3, pytest, hypothesis> scripts/run_numpy_dropin_suite.sh
  <so_dir>/fnp_python.so <out>` (~8 min on 64 cores).
- Route spy: wrap numpy module functions BEFORE `import fnp_python` (cached lookups happen after import);
  a spy cannot see ufunc-level delegation, so pair it with perf attribution (`perf record -e cpu-clock`,
  `perf report --sort dso,sym`, drop `blas_thread_server` samples — OpenBLAS workers spin at startup and
  otherwise dominate short profiles).
