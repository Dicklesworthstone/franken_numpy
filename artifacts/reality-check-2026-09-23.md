# Reality check 2026-09-23: README/AGENTS vision vs. HEAD `52ca8020`

**Date:** 2026-09-23 (UTC 2026-09-24). **Tree:** `52ca8020` (main, 189 commits past `v0.3.0`).
**Method:** documents read in full (README 2,469 lines, AGENTS.md, spec, port plan, ADR-001, the 09-03 and
09-20 reality checks); CI history via `gh`; remote test runs pinned to HEAD
(`rch exec --base 52ca8020 --clean-overlay --no-overlay`); a HEAD cdylib built locally from a
`git archive` export (sha256 `b0954eea…`, dep-info verified) and probed live from python3.13 /
numpy 2.4.3 (the CI oracle line); four parallel read-only code audits. Every figure below was
produced by a command in this session unless marked *prior*.

---

## 1. The short answer

The **numerical core is real and well-tested**, and the **Python surface is broad and mostly
bit-faithful**. The project's **headline claims, though, describe a product that does not exist**:

* `fnp_python` is an **accelerator layered over NumPy**, not a replacement. It requires numpy
  (`pyproject: numpy>=2.3`), returns `numpy.ndarray`, and re-exports numpy's own `ndarray`,
  `dtype`, `ufunc` and submodules. It sends calls back to numpy from 620 `cached_numpy` sites. Among
  the delegated calls are **17 of 18 `fnp.fft` functions** and **2-D `linalg` solve/inv/det/svd/qr/eig/
  cholesky/eigh/pinv/norm/matrix_rank/slogdet** (verified live by monkeypatching numpy). The
  README's "Native" rows for `numpy.fft` and `numpy.linalg` are false.
* The **Stride Calculus Engine and the strict/hardened runtime are not on the path Python users hit**.
  SCE: 2 references in fnp-python. Runtime: a Python mode switch has existed since 09-15, but every
  one of the 4 recording sites discards the decision. Hardened mode therefore **only writes to a log**, and its
  ledger grows without bound. Fail-closed is unreachable from Python.
* **CI has been red on every run since 2026-09-08**, when it was fully green (G1-G9) three times.
  G2 has hit three failures in a row (`rng_adversarial` → `raptorq_artifact_suite` →
  `conformance_max`). Because G2 runs without `--no-fail-fast`, each failure hid the tests behind it, so
  for 15 days CI never ran 83 of the 198 fnp-python test files. It never ran
  `fnp-random`/`-random-core`/`-runtime`/`-ufunc` or G3-G8 either. The README badge and its "Update 3" say G2 is green.
* **About three weeks of work (332 commits) produced little measured value.** None of the 94 `perf`
  commits has a ledger row. 93 of them carry no measurement and 5 are duplicate patches. 235 of 293
  non-merge commits name no bead. The owner's 09-05 decision to revive the BLAS-capability and
  native-i64 streams has **no live bead**: both implementation beads were closed on 09-09 as
  "design banked in ADR".

## 2. What IS working (verified today)

| Area | Evidence |
|---|---|
| Rust engine differential (G3) | fresh numpy 2.4.3 oracle capture + `run_ufunc_differential`: **381/381** |
| fnp-ufunc tests | **2,439 passed, 0 failed**, 46 ignored (vmi1264463, HEAD) |
| fnp-random + random-core + runtime | **613 passed, 0 failed**, 2 ignored (vmi1153651, HEAD) |
| Python behavioural smoke (09-02 harness) | **373/381 bit-exact**, 7 same-exception, 1 within 1.2e-15 (batched det) |
| Python delegation/parity spy (526 cells) | **518 bit-exact**, 6 close, 2 MemoryError; 107 cells delegate to numpy |
| 09-02 defect list | **FIXED:** clip subclass, f32 batched det dtype, `negative(np.uint64)`, `bit_generator` type, `allclose` type, interp / f32 cumsum bit-exactness, cov (4, 2^18) bit-exact, big-endian input on 8 ops. Still open: ufunc objects; cov/corrcoef on other shapes (deliberately not bit-exact) |
| RNG | seeded Generator/RandomState/legacy streams bit-exact in smoke |
| Packaging | CI G9 builds and imports a cp313 wheel; `v0.3.0` tagged 2026-09-11 |
| Numeric-core safety | 10 of 11 crates `#![forbid(unsafe_code)]` |
| Large-n native wins (triage, same-invocation, loaded host) | exp 0.32x, sqrt 0.32x, cumsum 0.27x, clip 0.61x, sort 0.64x, take 0.75x at 2^20 (fnp time / numpy time) |

## 3. What is NOT working / is false

### Parity (live, HEAD cdylib, numpy 2.4.3)

* **CI G2 blocker, root-caused:** `fnp.max`/`fnp.min` canonicalize NaN payload and sign (numpy keeps the
  first NaN's exact bits: `0x7FF8…01` stays `…01`, and `-nan` keeps its sign bit). The cause is `ecb5bed8`
  (09-13), which made `try_zerocopy_f64_minmax` allocate a **0-d** preshaped output. PyO3 rejects 0-d buffers,
  so the call falls through to `UFuncArray::reduce_max`, whose `nan_max` returns `f64::NAN`. This is the same
  0-d-PyBuffer class as a prior incident. Collateral: the native small path never runs, so **max n=16 is 2.33x
  slower (+2.3 µs/call)**. `conformance_min` fails the same way and is hidden behind it.
* **ufunc objects:** 105/106 numpy ufunc names fail `isinstance(fnp.x, np.ufunc)`, and
  `isinstance(fnp.add, fnp.ufunc)` is False (`fnp.ufunc` is numpy's class, `fnp.add` is fnp's own `ufunc`
  pyclass). About 83 names are plain functions with no `.reduce/.accumulate/.outer/.at`. Bead `7evbk` was
  closed "all 5 resolved" with this still open.
* **Generator array parameters:** `default_rng(1).normal(loc=np.zeros(3))` raises TypeError (the binding takes `loc: f64`).
* **cov/corrcoef:** MemoryError on (32768, 8), an 8.6 GB output, under a 24 GB cap where numpy succeeds.
  Bit parity was deliberately given up (rel 1e-12) and is not registered anywhere.
* **Divergence bookkeeping:** `docs/DIVERGENCES.md` has 0 rows, while `crates/fnp-conformance/DISCREPANCIES.md`
  has **8 ACCEPTED** divergences plus DISC-011 (WILL-FIX, 5 `#[ignore]`d signed-zero tests since 05-23) and
  there is an ignored ISA-dependent nansum/var/std byte-parity test. `run_divergence_ledger` is vacuous:
  its expectation table is empty, it cannot register `DISC-` ids, and it is not run in CI.
* **Test tolerance:** `CompareMode::Close` is documented as "1 ULP" but uses `np.allclose` defaults
  (rtol 1e-5) in 105 cases, plus ~714 further default-`allclose` checks. 19 SHOULD/MAY cases can never fail.
* **Architecture (see §1):** 17/18 fft and 2-D linalg decompositions delegate. The Python route references
  the SCE on 2 lines. Hardened mode was checked live: after 200,000 hardened `clip` calls the ledger holds
  200,000 events (+93 MB, unbounded), each `action=allow`, and the output is identical to strict.
  `FNP_RUNTIME_MODE=bogus` imports silently as strict.

### What was hidden behind the CI wall (run today, HEAD, `--no-fail-fast`)

| Suite (not run in CI since 09-08) | Result |
|---|---|
| fnp-python: the 84 test files from `conformance_max` onward (rch vmi1152480) | 1,361 passed, **2 failed** (`conformance_max`, `conformance_min`: the same NaN-payload root cause), 4 ignored. The `numpy.__all__` lock, `e2e_workflow` and `metamorphic_array_ops` pass. Live check: 499/499 names present on numpy 2.4.3 |
| fnp-ufunc (vmi1264463) | 2,439 passed, 0 failed, 46 ignored |
| fnp-random + fnp-random-core + fnp-runtime (vmi1153651) | 613 passed, 0 failed, 2 ignored |
| G3 differential (local, fresh numpy 2.4.3 oracle) | 381/381 |

So the wall is narrow. A single fix (bead `.1`) is all the evidence says G2 needs; CI on its numpy 2.4.x
host must confirm (bead `.3`).

### Performance, measured today (same process, interleaved, A/A null per round; host thinkstation1, load 21-26, triage grade)

| cell | fnp/numpy | note |
|---|---|---|
| max f64 n=16 | **2.33x** (IQR 2.30-2.37) | +2.3 µs per call; stable |
| exp / sqrt f64 n=16 | 1.81x / 2.02x | |
| add / multiply / divide n=16-256 | 1.31-1.47x | +150-200 ns/call. The README's "700-1,450 ns wrapper floor" now overstates the cost for these ops. The floor shrank, but no commit measured by how much |
| argsort f64 normal 2^20 | **2.0-2.15x** (two runs) | contradicts README headline "radix argsort 12-15x" as a general statement |
| argsort i64 uniform [0,2^40) 2^20 | **3.87x** | |
| sum / mean f64 2^20 | 1.37-3.6x (magnitude unstable) | ledger already rules this cell undecidable (qpylx); report the sign only |

Ledger-side (perf audit): the FAQ's "2.7-4.8x whole-job" misstates its own scorecard (honest
conservative range **1.90-3.75x**). "f64 divide delegates at every size" is wrong: divide is
native from 2^21. KEEP-claim incumbent coverage is now **54/834 (6.5%)**. Proof-class
inflation: L67195 is classed `incumbent-win` although fnp is the *slower* arm (1.150x); the 09-20
report quotes "5.01x clip" from non-interleaved best-of-5 prints; `3aa61557` claims 19x and touches only `.beads`.
The 08-31 survey criterion (`9a71376a`) made "every remaining LOSS disappear" from the
triage board. The small-n losses above are stable enough to be decided by any criterion.

### False or stale public claims (docs audit)

* "zero production `.unwrap()`" / "no hidden panics": **42 `.unwrap()`, 138 `.expect(`, 23 `unreachable!`** in non-test library code (fnp-python, -ufunc, -linalg).
* fnp-python `unsafe` "only for zero-copy buffer views": **1,037 `unsafe {` lines**, including `transmute` and `get_unchecked`.
* SCE "every other crate flows through it": 5 crates never touch it; fnp-python reads numpy `.shape/.strides` 449 times.
* IO "byte-identical to numpy.save for every supported dtype": proven for **7 of 27** dtype variants.
* "every change has a bead ID": 235/293 commits since 09-03 have none.
* Stale counts: 27-vs-30 fuzz targets, 8-vs-13 hygiene tests, 8,688-vs-8,716-vs-8,718 tests, 191-vs-192 shards, README L393 says the runtime decision is "open" (bead closed) and "no Python switch" (exists since 09-15).
* Spec §17 SLOs (e.g. broadcast add on 100M elements p95 ≤ 180 ms) were **never measured**. The G7 gate reuses the budget numbers on 256×256 inputs, which are 1,500x smaller, and has done so since 02-17.

### Process / work-graph
* Before this check: 0 open beads and 7 in progress, **all untouched for 17-58 days** (latest update 09-06).
* ADR-001 says "ACCEPTED (09-05)" and "[USER FILLS IN AFTER REVIEW]" at the same time. Its revived streams
  (`ceb7k` BLAS capability, `reqc1` native i64) were closed on 09-09 with no code. `u2z2b` was closed as
  "routed to" those beads. `uzb4u` ("…policy **enforcement**") was closed with the reason "done"; no enforcement exists.
* The 09-20 reality check proposed three beads. None were created.
* Three workflow files are malformed YAML and have failed on each of ~455 pushes. They are code-rewriting
  migrations whose target changes never landed. `fnp-random-core` is depended on by no crate; fnp-random keeps its own PCG64DXSM.
* A commit in this repo carries another project's bead ID (`9b7d1044`, `hfdt-…`).

## 4. Would finishing all open + in-progress beads close the gap?

**No.** The 7 in-progress beads are all performance items (wrapper floor, sort n=256, divide, take,
allocation lifecycle, the ixs5y umbrella), and none has moved in 17+ days. None of the following has a bead:
the red CI, the `max` NaN parity bug, the ufunc-object protocol, Generator array parameters, the
logs-only hardened mode, the unbounded ledger, the fft/linalg "Native" gap, the false README
claims, the parity-debt ledger, the unmeasured spec SLOs, or the revived ADR streams.

## 5. Vision checklist

| # | Goal (source) | Status | Evidence |
|---|---|---|---|
| 1 | 100% `numpy.__all__` reachable (README L7) | WORKING (surface only) | hasattr lock; its test file had not run in CI since 09-08 (sorts after the wall) |
| 2 | Drop-in behavioural parity (spec §2, README L1147) | PARTIAL | 373/381 smoke bit-exact; breaks: ufunc protocol, Generator array params, max/min NaN bits |
| 3 | Clean-room replacement / "true drop-in" (spec §0, plan §3) | WRONG_APPROACH (undecided) | Python layer is an accelerator over numpy.ndarray; needs numpy; 620 delegate sites |
| 4 | numpy.fft / numpy.linalg "Native" (README L1366-67) | NOT AS CLAIMED | 1/18 fft native; 2-D linalg decompositions delegate |
| 5 | SCE owns all shape transforms (README L259, L487) | PARTIAL | Rust engine yes; Python path 2 refs |
| 6 | Strict/hardened runtime + evidence ledger (spec §4) | STUB (Python) | switch exists; decisions discarded; FailClosed unreachable; unbounded ledger |
| 7 | Differential conformance every CI run (README L54) | REGRESSED | G3 skipped in CI since 09-08; local 381/381 at HEAD |
| 8 | CI G1-G8 blocking gates (spec §18) | REGRESSED | red every run since 09-09; 3 serial G2 walls |
| 9 | Parity debt owned with closure gates (spec §2) | NOT WORKING | 0-row ledger vs 8 ACCEPTED + untracked ignores; vacuous gate |
| 10 | Bit-exact RNG (README L7) | WORKING (scalar params) | smoke RNG cells bit-exact; 613 RNG/runtime tests pass |
| 11 | Safe-Rust numeric core (README L78) | WORKING | 10/11 forbid(unsafe) |
| 12 | fnp-python unsafe "only buffer views" | OVERSTATED / UNPROVEN | 1,037 unsafe blocks incl. transmute/get_unchecked; no audit |
| 13 | No panics on user paths (README L1084) | FALSE | 42 unwrap / 138 expect / 23 unreachable |
| 14 | IO byte-identical to numpy.save, every dtype (README L924) | PARTIAL | 7/27 dtype variants proven |
| 15 | Fail-closed on unknown modes (AGENTS doctrine) | VIOLATED (Python) | invalid FNP_RUNTIME_MODE silently ignored |
| 16 | Performance-competitive (spec §0) | MIXED | large-n wins (exp/sqrt/cumsum/clip/sort/take); losses argsort 2^20, small-n 1.1-2.3x; f64 add/sub/mul delegate |
| 17 | Spec §17 SLO budgets (Gate C) | NOT_STARTED | G7 applies budgets to 1,500x smaller workloads |
| 18 | Every perf claim ledgered with provenance (AGENTS) | REGRESSED | 94/94 perf commits since 09-03 without a ledger row |
| 19 | Packaging (ADR-001 stream 1) | PARTIAL | Linux cp313 wheel in CI G9; no PyPI/macOS/Windows |
| 20 | ADR-001 revived streams: BLAS capability, native i64 | NO_BEAD (until today) | ceb7k/reqc1 closed as "design banked", no code |

## 6. Bridge plan → beads

All beads carry label `reality-check-2026-09-23` under epic `deadlock-audit-rc0923-epic-71qy3`
(children `.1`–`.24`). Each has evidence, scope, a named acceptance probe with a negative case, and test/logging requirements.

| Track | Bead | P | What |
|---|---|---|---|
| A CI | .1 | P0 | max/min 0-d decline → canonical NaN (CI wall) + small-n 2.33x |
| A CI | .2 | P0 | G2 `--no-fail-fast` so walls cannot hide tests |
| A CI | .3 | P0 | Acceptance: G1-G9 green on main (depends on .1, .2) |
| A CI | .4 | P1 | Silence 3 malformed workflows; retire dead ones (deletion needs owner OK) |
| B drop-in | .5 | P1 | ufunc object protocol for all 106 names |
| B drop-in | .6 | P1 | Generator/RandomState array-valued parameters |
| B drop-in | .7 | P2 | cov/corrcoef peak memory |
| B drop-in | .8 | P1 | numpy's own test suite run against fnp callables (ambition item) |
| C runtime/arch | .9 | P1 | bounded ledger; fail-closed env mode |
| C runtime/arch | .10 | P1 | hardened mode must act (depends on .9) |
| C runtime/arch | .11 | P1 | **owner decision:** accelerator vs replacement |
| C runtime/arch | .12 | P2 | fft/linalg route-or-document (depends on .11) |
| C runtime/arch | .13 | P2 | **owner decision:** ADR-001 streams re-confirm or withdraw |
| C runtime/arch | .14 | P3 | fnp-random-core integrate or remove |
| D honesty | .15 | P1 | README/AGENTS truth pass (depends on .11) |
| D honesty | .16 | P1 | one parity ledger + non-vacuous gate in CI |
| D honesty | .17 | P1 | Close-mode tolerance honesty, soft cases |
| D honesty | .18 | P2 | perf-record proof-class corrections; one row for the 94 commits |
| D honesty | .19 | P2 | loss-map criterion re-derived on median-CI |
| E safety | .20 | P2 | panic audit |
| E safety | .21 | P2 | fnp-python unsafe audit |
| E safety | .22 | P2 | IO byte identity for all dtype variants |
| F perf | .23 | P1 | argsort 2^20 losses: contract-grade grid, then fix |
| F perf | .24 | P3 | spec-scale SLO workloads |

Incident comments added to beads closed without their evidence: `uzb4u`, `7evbk`, `ixs5y.410`, `u2z2b`,
`ceb7k`, `reqc1`. Data comments added to in-progress `1uf80` (re-measured floor) and `ixs5y.409` (stale
3.72x; unmeasured threshold extension to 8 dtypes).

### Steering notes (not beads)
1. **Point the fleet at Track A first.** One ~20-line guard (bead .1) plus a one-flag CI change (.2) should restore
   CI signal after 15 days. Nothing else can be verified in CI until then.
2. **Stop unmeasured perf micro-commits.** 94 commits in three weeks produced no ledger evidence. Require a
   same-invocation number per lever (AGENTS "one lever, proof-backed").
3. **Re-activate or release the 7 stale in-progress beads** (untouched 17-58 days; three are unassigned).
4. **Enforce bead ids in commit subjects** or drop the README claim.
5. **Two decisions only the owner can make:** .11 (what fnp_python is) and .13 (ADR-001 streams).
