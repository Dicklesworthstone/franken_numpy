# Reality-Check Audit: README.md + AGENTS.md Claims vs Live Tree

**Date:** 2026-09-20  
**Auditor:** Antigravity  
**Tree state:** HEAD `6405e6ca` (2026-09-20), shared multi-agent checkout on branch `main` (synced to `master` and `origin/main`).  
**Tooling & Methodology:** Static verification (`git ls-files`, `rg`, `wc`, manifests), bead work-graph audit (`bv --robot-triage`, `br list`), GitHub Actions CI run logs (run `35535175504`), and RCH remote execution.  

---

## 1. Executive Summary & Ground Truth

FrankenNumPy is in an exceptionally mature, high-integrity state of feature delivery. Surface parity with legacy NumPy stands at **100% of `numpy.__all__`** (all 499 names on the `numpy<2.5` CI oracle / 501 names on live host numpy 2.3.5) and is structurally locked by CI test `fnp_python_covers_full_numpy_all`.

The pure Rust numeric core now spans **11 crates** (with the addition of `fnp-random-core`), **1,643 `pub fn` declarations**, and **8,715 tests**.

### Critical Breakthrough: The Real Status of CI Gate G2

Previous audits (e.g. `artifacts/reality-check-2026-09-03.md`) and current documentation state:
> `CI gates: G1 green · G2 red (ledger hygiene) · G3–G8 blocked`

**This claim is now false in its stated cause:**
1. **Ledger hygiene is 100% RESOLVED.** All 25 tests in `crates/fnp-conformance/tests/ledger_hygiene.rs` pass cleanly! The prior 216 unworkered rows in `docs/NEGATIVE_EVIDENCE.md` were completely cured by provenance re-banking commits (`43515a61`, `e25ac6d8`, etc.).
2. **G2 is red solely due to a single newly-exposed contract divergence in `fnp-conformance`:**
   - Test failure: `tests::rng_adversarial_suite_is_green` and `tests::core_suites_are_green` in `crates/fnp-conformance/src/lib.rs`.
   - Exact panic: `expected error containing 'seed sequence state generation contract violated' but operation 'seedsequence_empty_entropy' succeeded`.
   - Root Cause: Bead `franken_numpy-iqo31` legalized `SeedSequence::new(&[])` by falling back to OS entropy to achieve behavioral parity with Python NumPy's unseeded `np.random.SeedSequence()`. However, the adversarial test fixture `rng_seedsequence_empty_entropy` in `crates/fnp-conformance/fixtures/rng_adversarial_cases.json` and its dispatch at `crates/fnp-conformance/src/lib.rs:20617` were not synchronized to reflect this contract shift, causing CI to expect an error when the operation now legitimately succeeds.

Once this single fixture expectation is aligned with the legalized contract, **G2 will pass**, unblocking the downstream verification pipeline (G3–G8).

---

## 2. Phase 1: Core Reality Check Questions

### Where are we REALLY on this project?
The project has essentially completed the monumental task of reimplementing NumPy's surface and core numerical semantics in safe Rust. The array API, ufuncs, reductions, einsum, linalg, random number generation, and I/O are real, tested, and working. Wheel packaging and module import in Python 3.12 and 3.13 are fully functional (verified by passing CI gate G9).

The project is currently in the late-stage convergence and optimization phase:
- **Functional/Surface Completion:** ~100% complete.
- **CI Test Suite:** 1 single contract test failure blocking G2.
- **Performance Parity:** Actively being optimized by the remaining 8 in-progress beads.

### 1. What specifically IS working right now?
1. **100% Surface Coverage:** All 499 symbols in `numpy.__all__` are exposed via PyO3 in `fnp-python`, structurally locked by `conformance_remaining_top_level_attrs.rs`.
2. **Memory Safety & Unsafe Invariant:** 10 of 11 crates strictly declare and enforce `#![forbid(unsafe_code)]`. Only `fnp-python` contains narrow, audited unsafe blocks for zero-copy PyBuffer borrowing. Enforced by `no_unsafe_code_blocks_or_items` in codebase hygiene.
3. **Stride Calculus Engine (SCE):** Zero-copy contiguous/strided views, broadcast arithmetic, reshape `-1` inference, transpose, and slice operations in `fnp-ndarray`.
4. **Numeric Core (fnp-ufunc):** 850+ array operations, binary/unary ufuncs, multi-axis reductions, einsum, compensated summation (Neumaier / KBK), FFT (Cooley-Tukey + Bluestein), and polynomial arithmetic. 2,474 unit tests passing.
5. **Linear Algebra (fnp-linalg):** Complete pure-Rust decomposition routines (QR, SVD, eig, eigh, Cholesky, LU, solve, lstsq, matrix_power). 459 unit tests passing.
6. **Random Generators (fnp-random & fnp-random-core):** PCG64, PCG64DXSM, MT19937, Philox, SFC64, with bit-exact PCG64DXSM parity with NumPy and OS entropy fallback for unseeded constructors.
7. **Array & Text I/O (fnp-io):** NPY versions 1.0, 2.0, 3.0, NPZ archives, text I/O (`loadtxt`, `savetxt`, `genfromtxt`) with hardened boundaries (MAX_HEADER_BYTES, MAX_ARCHIVE_MEMBERS).
8. **Dual-Mode Runtime (fnp-runtime):** Strict mode vs Hardened mode execution, wire format validation, and decision provenance tracking.
9. **Hygiene Enforcement:** Zero stubs/mocks/TODOs in production code, verified by 13 test functions in `crates/fnp-conformance/tests/codebase_hygiene.rs`.
10. **Wheel Packaging & CI Import:** Maturin wheel build and dynamic import tested on Python 3.12 and 3.13 in CI G9.
11. **Ledger Hygiene:** All 25 tests in `crates/fnp-conformance/tests/ledger_hygiene.rs` pass cleanly.

### 2. What is NOT working or not yet implemented?
1. **CI Gate G2 Failure:** Test `rng_adversarial_suite_is_green` panics because `seedsequence_empty_entropy` fixture in `fnp-conformance` expects an error that `fnp-random` no longer produces after bead `franken_numpy-iqo31` legalized empty entropy.
2. **Performance Gaps vs Upstream NumPy:** 8 active performance bottlenecks are currently being optimized:
   - Python-to-Rust wrapper dispatch overhead (~700-1450 ns per call floor).
   - Small-$n$ integer and float sort/argsort kernel latency (e.g. int64 $n=256$ sort is ~3.72x slower).
   - Float64 division kernel codegen gap (~94 µs overhead).
   - Native `take` route on strided/non-contiguous layouts (1.25–1.49x slower).
3. **Documentation Drift:** Metrics in `README.md` and `AGENTS.md` lag behind the live codebase (crate count, version, commit count, closed beads, `pub fn` count, test count).

### 3. What is blocking us from getting there?
1. **CI Unblocking:** Modifying `crates/fnp-conformance/fixtures/rng_adversarial_cases.json` and `crates/fnp-conformance/src/lib.rs:20617` to assert success or updating the fixture to test a genuinely invalid state (e.g. invalid pool size or exceeded words).
2. **Performance Floor:** The PyO3 argument extraction and wrapping floor (~700-1450 ns) in `fnp-python` requires shared helper deduplication (e.g. bead `deadlock-audit-omyno` for 62 reshape sites) and result buffer lifecycle optimization (`deadlock-audit-tztko`).

### 4. If we were to implement all open and in-progress beads, would we close the gap completely? Why or why not?
- **On Performance:** YES. All 8 in-progress beads (`deadlock-audit-1uf80`, `deadlock-audit-6y5wp`, `deadlock-audit-9g7u0`, `deadlock-audit-ddoeq`, `deadlock-audit-omyno`, `deadlock-audit-tztko`, `franken_numpy-ixs5y`, `franken_numpy-ixs5y.409`) are targeted directly at closing every measured performance deficit against upstream NumPy.
- **On CI Gate G2:** NO. There is currently **no open bead** tracking the `rng_seedsequence_empty_entropy` fixture discrepancy. It is a newly discovered bug from this audit. A dedicated bead must be created and resolved to close the CI gap.

### 5. What goals from the vision are NOT covered by ANY existing bead?
1. **NO_BEAD: Conformance fixture sync for empty entropy:** Synchronize `rng_adversarial_cases.json` and `fnp-conformance/src/lib.rs` with `franken_numpy-iqo31` empty-entropy legality.
2. **NO_BEAD: Documentation metric synchronization:** Update `README.md` and `AGENTS.md` with accurate live metrics (11 crates, version 0.3.0, 10/11 `#![forbid(unsafe_code)]`, 1,643 `pub fn`, 8,715 tests, 7,631 commits, 2,833 closed beads, CI status update).

---

## 3. Vision Checklist

| # | Goal | Source | Status | Evidence | Category |
|---|------|--------|--------|----------|----------|
| 1 | 100% surface parity with `numpy.__all__` | README L7 | **WORKING** | 499/499 names exposed; `conformance_remaining_top_level_attrs.rs` passes | Feature Parity |
| 2 | Memory safety: `#![forbid(unsafe_code)]` across computational crates | README L324, AGENTS L76 | **WORKING** | 10/11 crates carry `#![forbid(unsafe_code)]`; `codebase_hygiene.rs` enforces | Safety |
| 3 | Stride Calculus Engine (SCE) view & broadcast calculus | README L336-368 | **WORKING** | `fnp-ndarray` unit tests (231 tests) and `fnp-iter` (205 tests) verify zero-copy and broadcast rules | Architecture |
| 4 | Comprehensive Ufunc and Reduction dispatcher | README L458-540 | **WORKING** | 2,474 unit tests in `fnp-ufunc`; binary, unary, einsum, FFT, polynomial families | Numeric Engine |
| 5 | Pure Rust Linear Algebra (no BLAS/LAPACK dependency) | README L544-570 | **WORKING** | 459 unit tests in `fnp-linalg`; QR, SVD, eig, eigh, Cholesky, LU, solve | Linalg |
| 6 | Bit-exact Random Generators & NumPy SeedSequence parity | README L572-605 | **WORKING** | 5 bit generators; PCG64DXSM bit-exact vectors pass; `fnp-random-core` dependency-free | Random |
| 7 | Secure Array & Text I/O with hard bounds | README L607-640 | **WORKING** | 409 tests in `fnp-io`; NPY v1/2/3, NPZ, loadtxt, savetxt with MAX limits | I/O |
| 8 | Dual-mode runtime (Strict vs Hardened) | README L642-670 | **WORKING** | `fnp-runtime` (134 tests); wire format and policy checks active | Runtime |
| 9 | Zero stubs, mocks, or TODOs in production | README L1620, AGENTS L78 | **WORKING** | 13 test functions in `codebase_hygiene.rs` pass; no mock markers | Hygiene |
| 10 | Ledger hygiene in negative evidence docs | README L1780-1820 | **WORKING** | 25/25 tests in `crates/fnp-conformance/tests/ledger_hygiene.rs` pass | Provenance |
| 11 | Differential Conformance Suite | README L1665-1750 | **WORKING** | 192 python conformance shards, 48 conformance binaries | Testing |
| 12 | Wheel Packaging and import parity | README L672-710 | **WORKING** | CI G9 passes on Python 3.12 and 3.13; wheel builds and imports cleanly | Packaging |
| 13 | CI Gate Pipeline G1–G9 Green | README L1760-1850 | **REGRESSED** | G1 (pass), G9 (pass), G2 (fails on `seedsequence_empty_entropy`), G3-G8 (skipped) | CI Gates |
| 14 | Performance Parity without C BLAS | README L1860-1940 | **PARTIAL** | Core kernels match/win; wrapper floor and small-n sort/f64 divide being closed | Performance |

---

## 4. Deep-Dive: Root Cause Analysis of CI Gate G2 Failure

In CI run `35535175504` (commit `6405e6ca` on `main`), G2 failed during `cargo test --workspace` with the following trace:

```text
running 2 tests
test tests::rng_adversarial_suite_is_green ... FAILED
test tests::core_suites_are_green ... FAILED

failures:
---- tests::rng_adversarial_suite_is_green stdout ----
thread 'tests::rng_adversarial_suite_is_green' panicked at crates/fnp-conformance/src/lib.rs:20538:13:
RNG adversarial suite failed conformance verification: FixtureMismatch("expected error containing 'seed sequence state generation contract violated' but operation 'seedsequence_empty_entropy' succeeded")
```

### The Mechanism of Failure
1. **Upstream NumPy Parity Fix (`franken_numpy-iqo31`):**  
   NumPy allows `SeedSequence()` or `SeedSequence([])` without explicit entropy, falling back to OS entropy. FrankenNumPy adopted this in `crates/fnp-random/src/lib.rs`:
   ```rust
   pub fn new(entropy: &[u32]) -> Result<Self, SeedSequenceError> {
       let pool = if entropy.is_empty() {
           // Bead franken_numpy-iqo31: unseeded / empty entropy sources OS entropy for NumPy parity
           let mut words = [0u32; DEFAULT_POOL_SIZE];
           getrandom::fill(bytemuck::cast_slice_mut(&mut words))
               .map_err(|_| SeedSequenceError::EntropySourceFailed)?;
           mix_entropy(&words, DEFAULT_POOL_SIZE)?
       } else {
           mix_entropy(entropy, DEFAULT_POOL_SIZE)?
       };
       Ok(Self { ... })
   }
   ```
2. **Outdated Conformance Fixture:**  
   `crates/fnp-conformance/fixtures/rng_adversarial_cases.json` retained an obsolete case:
   ```json
   {
     "id": "rng_seedsequence_empty_entropy",
     "operation": "seedsequence_empty_entropy",
     "expected_error_contains": "seed sequence state generation contract violated",
     "expected_reason_code": "rng_seedsequence_generate_state_failed",
     ...
   }
   ```
3. **Dispatch Assertion:**  
   In `crates/fnp-conformance/src/lib.rs:20617`:
   ```rust
   "seedsequence_empty_entropy" => SeedSequence::new(&[])
       .map(|_| ())
       .map_err(map_seedsequence_error_to_rng_suite),
   ```
   Because `SeedSequence::new(&[])` returns `Ok`, the test framework detects that an expected failure did not occur and panics.

**Remedy:** Update the fixture or replace the test with an operation that is genuinely invalid under the new contract (or assert `Ok(())` for empty entropy). This immediately unblocks G2.

---

## 5. Machine-Checkable Claims Audit

| Claim Item | Prior Documented Claim | Codebase Ground Truth (Live) | Delta | Severity |
|---|---|---|---|---|
| **Workspace Crates** | 10 crates | **11 crates** (`crates/fnp-random-core` added) | +1 crate | Medium |
| **Workspace Version** | `0.2.0` | **`0.3.0`** (root `Cargo.toml`) | Bumped | Medium |
| **Unsafe Forbid Invariant** | "9 of 10 crates" | **10 of 11 crates** carry `#![forbid(unsafe_code)]` | +1 crate | Medium |
| **`pub fn` Count** | 1,626 declarations | **1,643 declarations** | +17 | Medium |
| **Workspace `#[test]` Count** | 8,691 tests | **8,715 tests** | +24 | Medium |
| **Closed Beads** | 2,786 closed | **2,833 closed** | +47 | Medium |
| **Open Beads** | 0 open | **0 open** | 0 | None |
| **In-Progress Beads** | ~38 | **8 in-progress** | -30 | Medium |
| **Total Git Commits** | 7,336 commits ("7,300+") | **7,631 commits** | +295 | Low |
| **Bench Files** | 61 bench `.rs` files | **66 bench `.rs` files** | +5 | Low |
| **Integration Test Files** | 240 integration test files | **240 integration test files** | 0 | Exact |
| **fnp-python Conformance Shards** | 192 dedicated shards | **192 dedicated shards** | 0 | Exact |
| **Hygiene Tests** | 13 `#[test]` functions | **13 `#[test]` functions** | 0 | Exact |
| **Parity Divergence Ledger** | 0 active rows | **0 active rows** | 0 | Exact |
| **G2 Failure Cause** | Ledger hygiene missing workers | **`seedsequence_empty_entropy` fixture mismatch** (Ledger hygiene 100% pass) | Ground Truth Inverted | High |

---

## 6. Live Test Counts by Crate

| Crate | Documented Count (Sept 3) | Live Count (Sept 20) | Δ | Status |
|---|---:|---:|---:|---|
| `crates/fnp-ufunc` | 2,472 | 2,474 | +2 | Green |
| `crates/fnp-python` | 3,637 | 3,648 | +11 | Green |
| `crates/fnp-random` | 477 | 479 | +2 | Green |
| `crates/fnp-random-core` | — | 5 | +5 | Green (new) |
| `crates/fnp-linalg` | 459 | 459 | 0 | Green |
| `crates/fnp-io` | 407 | 409 | +2 | Green |
| `crates/fnp-conformance` | 395 | 395 | 0 | Red on fixture |
| `crates/fnp-dtype` | 276 | 276 | 0 | Green |
| `crates/fnp-ndarray` | 231 | 231 | 0 | Green |
| `crates/fnp-iter` | 203 | 205 | +2 | Green |
| `crates/fnp-runtime` | 134 | 134 | 0 | Green |
| **Total Workspace Tests** | **8,691** | **8,715** | **+24** | **8,713 passing** |

---

## 7. Active Bead Inventory Analysis

The bead tracker contains **2,841 active issues** (plus 1 tombstone), of which **2,833 are closed**, **0 are open**, and exactly **8 are in-progress**:

| Bead ID | Pri | Title | Focus Area |
|---|---|---|---|
| `deadlock-audit-1uf80` | P1 | `perf: REDUCE THE WRAPPER FLOOR ITSELF - ~700-1450 ns of per-call cost...` | fnp-python wrapper dispatch latency |
| `deadlock-audit-6y5wp` | P1 | `[perf][fnp-python] our f64 divide KERNEL is 1.2652x slower than numpy's...` | Codegen for f64 binary division |
| `deadlock-audit-9g7u0` | P1 | `perf(sort): locate and widen live-small-n losses for float64/int32 sort...` | Small-$n$ sort micro-benchmarks |
| `deadlock-audit-ddoeq` | P2 | `perf(take): fnp.take is 1.25-1.49x SLOWER than NumPy at m=2^10..2^20...` | Native strided take paths |
| `deadlock-audit-omyno` | P2 | `[perf][fnp-python] the no-op reshape-to-current-shape block is repeated...` | Helper deduplication across 62 sites |
| `deadlock-audit-tztko` | P1 | `[perf][structural] Measure fnp-python result-buffer allocation lifecycle...` | Buffer allocation profiling |
| `franken_numpy-ixs5y` | P1 | `[perf][no-gaps] Close ALL vs-upstream perf gaps in pure safe Rust...` | Overarching performance parity epic |
| `franken_numpy-ixs5y.409` | P1 | `perf(sort): int64 n=256 sort is 0.2687x vs numpy (3.72x slower)...` | Int64 $n=256$ sort kernel tuning |

Work-graph health (`bv --robot-triage`):
- Density: 0.000012
- Dependency cycles: 0
- Blocked beads: 0

---

## 8. Phase 2: Bridge Plan

### Action Item 1: Resolve Conformance Fixture Divergence (Unblock CI Gate G2)
1. **Target:** `crates/fnp-conformance/fixtures/rng_adversarial_cases.json` and `crates/fnp-conformance/src/lib.rs`.
2. **Action:**
   - Either update case `rng_seedsequence_empty_entropy` to assert `Ok(())` (documenting the shift from error to OS entropy sourcing per bead `franken_numpy-iqo31`),
   - Or replace the case with a genuinely illegal input (e.g. attempting to generate state with zero/overflowing parameters) that produces an intentional contract violation.
3. **Verification:** Run `rch exec -- cargo test -p fnp-conformance --test rng_adversarial_cases` and verify exit code 0.

### Action Item 2: Synchronize Machine-Checkable Documentation Claims
1. Update `README.md` and `AGENTS.md`:
   - Crates: 10 -> 11 (adding `crates/fnp-random-core`).
   - Version: 0.2.0 -> 0.3.0.
   - Unsafe forbid: 9/10 -> 10/11 crates.
   - `pub fn`: 1,626 -> 1,643.
   - Tests: 8,691 -> 8,715.
   - Commits: 7,300+ -> 7,631.
   - Closed beads: 2,786 -> 2,833.
   - CI Status Badge & Block: Record that Ledger Hygiene is 100% green; update G2 failure reason to the adversarial RNG fixture sync.

### Action Item 3: Execute the Remaining 8 Performance Beads
1. **Wrapper Floor (`deadlock-audit-1uf80`, `deadlock-audit-omyno`):** Extract shared helpers for the 62 repeated reshape sites to reduce binary footprint and inline-cache miss rate.
2. **Sort Kernels (`deadlock-audit-9g7u0`, `franken_numpy-ixs5y.409`):** Implement small-$n$ specialized insertion/network sorts for $n \le 256$ to close the 3.72x gap against NumPy.
3. **Division Kernel (`deadlock-audit-6y5wp`):** Eliminate redundant bounds and alignment checks in the unrolled f64 division loop.
4. **Buffer Lifecycle (`deadlock-audit-tztko`):** Measure arena vs direct PyBuffer reuse potential.

---

## 9. Conclusion

FrankenNumPy is in an outstanding, world-class condition. The codebase is clean, completely stub-free, and delivers 100% of NumPy's public surface in safe Rust. The historical ledger hygiene issue that blocked CI for weeks is completely cured. Fixing the single RNG fixture synchronization mismatch will turn CI gate G2 green and unlock full automated pipeline verification across all 9 gates.
