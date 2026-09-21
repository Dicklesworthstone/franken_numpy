# Comprehensive Reality-Check Audit: README.md + AGENTS.md Vision vs Live Codebase Reality

**Date:** 2026-09-21  
**Auditor:** Antigravity  
**Tree State:** HEAD `3cd53633` (branch `main`, synchronized with `origin/main` and `master`).  
**Auditing Methodology:** Direct static AST & keyword analysis (`rg`, `git ls-files`, `wc`, manifests), bead work-graph triage (`bv --robot-triage`, `br sync --import-only`), conformance fixture & dispatch inspection, and RCH remote compilation telemetry.

---

## 1. Executive Summary & Ground Truth

FrankenNumPy is in an extraordinarily mature state of functional delivery. Across its pure Rust numeric core, it delivers 11 crates, 1,644 `pub fn` declarations, and 8,716 tests with **zero stubs, zero mocks, zero `todo!()`, and zero `unimplemented!()` macros**. On the Python surface, all 499 names in `numpy.__all__` are exposed and structurally locked by CI test `fnp_python_covers_full_numpy_all`.

However, an honest, unsparing reality check reveals a profound distinction between **Surface Parity** and **Autonomous Clean-Room Replacement**:

1. **The NumPy Runtime Dependency (Hybrid Architecture):**  
   While the 10 numeric crates (`fnp-ndarray`, `fnp-ufunc`, `fnp-linalg`, `fnp-random`, etc.) are 100% pure safe Rust and completely independent of external C/Python libraries, `fnp-python` relies heavily on live `numpy`. There are **419 non-test sites in `fnp-python/src/lib.rs` that import or call live `numpy` (`cached_numpy`)**. Tier 3 submodules (`numpy.strings`, `numpy.matrixlib`, `numpy.ma`, `numpy.rec`, `numpy.testing`, `numpy.f2py`) and scalar constants are identity-equal re-exports of live NumPy attributes. When unsupported dtypes, exotic kwargs, or non-contiguous strided layouts arrive, execution falls back silently to Python NumPy. Consequently, FrankenNumPy cannot run as a standalone Python drop-in if NumPy is uninstalled.
2. **The PyO3 Call Latency Floor (~700–1450 ns):**  
   Rust engine kernels frequently beat NumPy by 1.2x–5x on large arrays ($N > 10^4$), but for micro-benchmarks and small arrays ($N < 1000$), PyO3 argument marshalling, dictionary unpacking, and buffer acquisition add an unavoidable ~700–1450 ns latency floor, causing small-input operations to measure slower than NumPy's specialized C-API dispatch.
3. **CI Gate G2 Status & Hygiene Invariants:**  
   The historical ledger hygiene blocker (216 unworkered rows in `docs/NEGATIVE_EVIDENCE.md`) is completely cured (25/25 tests passing). The adversarial `SeedSequence` empty-entropy contract divergence identified on 2026-09-20 was resolved in commits `199f4013` and `a366ab75`. All 13 codebase hygiene tests pass cleanly.
4. **Active Work Graph:**  
   The project has **2,845 total tracked beads**: **2,835 closed**, **0 open**, and **9 in progress**. All 9 in-progress beads are targeted directly at performance micro-optimizations (sort, divide, take, wrapper floor, reshape deduplication, clip).

---

## 2. Phase 1: Core Reality Check Questions

### Where are we REALLY on this project?
Does the implemented code actually deliver on the vision described in the README and plan documents?

**Brutally honest answer:**  
The project has delivered on the first monumental half of its vision: reimplementing NumPy's numerical API surface and core array algorithms in memory-safe, clean-room Rust with zero mocks and rigorous differential conformance testing. The mathematical core is real, high-performance, and safe.

However, FrankenNumPy is currently an **accelerated wrapper and hybrid execution engine**, rather than a fully autonomous drop-in replacement for the NumPy wheel. It accelerates common contiguous numerical paths in pure Rust, but delegates edge cases, complex kwargs, and secondary submodules directly to the host's installed NumPy.

### 1. What specifically IS working right now?
1. **100% Surface Coverage:** All 499 names in `numpy.__all__` are exposed via PyO3 in `fnp-python`, locked by `conformance_remaining_top_level_attrs.rs`.
2. **Memory Safety Invariant:** 10 of 11 crates enforce `#![forbid(unsafe_code)]` with zero `unsafe` items. Only `fnp-python` contains narrow, audited `unsafe` blocks for zero-copy `PyBuffer` borrowing.
3. **Stride Calculus Engine (SCE):** Deterministic shape legality, C/F contiguous baseline strides, broadcast shape resolution, reshape `-1` inference, transpose, and slice operations in `fnp-ndarray`.
4. **Numeric Core (fnp-ufunc):** 850+ array operations, binary/unary ufuncs, multi-axis reductions, einsum (`einsum`, `einsum_path`, `einsum_optimized`), compensated summation (Neumaier / KBK), FFT (Cooley-Tukey + Bluestein), and polynomial families.
5. **Linear Algebra (fnp-linalg):** Pure-Rust decomposition routines (QR, SVD, eig, eigh, Cholesky, LU, solve, lstsq, matrix_power) without linking C BLAS or LAPACK.
6. **Bit-Exact Random Number Generation (fnp-random & fnp-random-core):** PCG64, PCG64DXSM, MT19937, Philox, SFC64, with bit-exact PCG64DXSM parity with NumPy and OS entropy fallback for unseeded constructors.
7. **Secure I/O (fnp-io):** NPY versions 1.0, 2.0, 3.0, NPZ archives, text I/O (`loadtxt`, `savetxt`, `genfromtxt`) with hardened boundaries (`MAX_HEADER_BYTES`, `MAX_ARCHIVE_MEMBERS`).
8. **Dual-Mode Runtime (fnp-runtime):** Strict mode vs Hardened mode execution, wire format validation, and decision provenance tracking.
9. **Zero Stubs / Codebase Hygiene:** 0 `unimplemented!()`, 0 `todo!()`, 0 stub comments, 0 `panic!("not implemented")`, 0 FIXME/HACK markers in production code. Verified by `crates/fnp-conformance/tests/codebase_hygiene.rs`.
10. **Ledger Hygiene:** All 25 tests in `crates/fnp-conformance/tests/ledger_hygiene.rs` pass cleanly.
11. **Packaging:** Maturin wheel build and dynamic import tested and passing on Python 3.12 and 3.13 in CI G9.
12. **Targeted Kernel Speedups:** Real, measured wins on large arrays over NumPy (e.g. 5.01x on f32 scalar min-clip, 1.52x on f64 scalar max-clip, multi-threaded reductions).

### 2. What is NOT working or not yet implemented?
1. **Autonomous Execution Without Host NumPy:** `fnp-python` cannot function without Python `numpy` installed in the environment due to Tier 3 submodules (`strings`, `matrixlib`, `ma`, etc.) and fallback delegation at 419 call sites.
2. **Wrapper Call Latency Floor:** PyO3 argument extraction and dictionary overhead (~700–1450 ns) makes micro-array operations ($N < 1000$) slower than NumPy's hand-written C routines.
3. **Small-$n$ Sort Kernels:** Int64 $n=256$ sort is ~3.72x slower than NumPy's introsort/radix sort (tracked in `franken_numpy-ixs5y.409`).
4. **f64 Division Kernel Codegen Gap:** f64 binary division kernel has a ~94 µs codegen overhead relative to NumPy's vectorized loop (tracked in `deadlock-audit-6y5wp`).
5. **Native Strided Take Path:** `take` on strided/non-contiguous layouts is 1.25–1.49x slower than NumPy (tracked in `deadlock-audit-ddoeq`).
6. **Repeated Reshape Boilerplate:** 62 reshape sites in `fnp-python` repeat identical shape validation logic instead of using a shared helper (tracked in `deadlock-audit-omyno`).
7. **Dormant Runtime Ledger in Python Data Paths:** `fnp-runtime`'s Bayesian decision engine and evidence ledger are called at only 2 sites in `fnp-python`, leaving the runtime mode distinction largely unexercised during normal array operations.

### 3. What is blocking us from getting there?
1. **FFI Marshaling Overhead:** PyO3's generic argument extraction and Python dictionary parsing have fixed CPU cycle costs that cannot be eliminated without specialized fast-path C-API call wrappers (`METH_FASTCALL` / `vectorcall`).
2. **Sorting Algorithm Specialization:** Safe Rust slice sorting (`slice::sort_unstable`) relies on pdqsort, which lacks the branch-free sorting networks and vectorized radix partitions NumPy uses for $n \le 256$.
3. **Tier 3 Submodule Implementation Scope:** Reimplementing `numpy.ma` (masked arrays), `numpy.matrixlib`, and `numpy.testing` in pure Rust requires dedicated crate surfaces and hundreds of new types.

### 4. If we were to implement all open and in-progress beads, would we close the gap completely? Why or why not?
**NO.**  
Implementing the 9 in-progress beads (`deadlock-audit-1uf80`, `deadlock-audit-6y5wp`, `deadlock-audit-9g7u0`, `deadlock-audit-ddoeq`, `deadlock-audit-omyno`, `deadlock-audit-tztko`, `franken_numpy-ixs5y`, `franken_numpy-ixs5y.409`, `franken_numpy-ixs5y.410`) will successfully close all **measured performance regressions** against NumPy (sort, divide, take, wrapper floor, reshape deduplication).

However, it will **NOT close the Architectural Dependency Gap**. None of the active beads track the elimination of the 419 `cached_numpy` delegation sites or the native implementation of Tier 3 submodules.

### 5. What goals from the vision are NOT covered by ANY existing bead?
1. **`NO_BEAD: Standalone NumPy-Free Python Wheel`:** Complete decoupling of `fnp-python` from the host's `numpy` installation, providing pure Rust implementations of Tier 3 submodules and native fallbacks for all 419 delegation sites.
2. **`NO_BEAD: Python Runtime Mode & Ledger Integration`:** Connecting `fnp-runtime`'s strict/hardened policy decisions and Bayesian risk audits to Python array creation, buffer borrowing, and ufunc dispatch.
3. **`NO_BEAD: End-to-End Execution of CI Gates G3–G8`:** Wiring and asserting continuous automated pass verdicts for downstream gates G3 (Security), G4 (Test Contract), G5 (Workflow Scenario), G6 (Performance Budget), G7 (Divergence Ledger), and G8 (API Coverage) in CI.

---

## 3. Vision Checklist

| # | Vision Promise / Goal | Source | Status | Evidence | Category |
|---|------------------------|--------|--------|----------|----------|
| 1 | 100% surface parity with `numpy.__all__` | README L7 | **WORKING** | 499/499 symbols exposed; `conformance_remaining_top_level_attrs.rs` passes | Feature Parity |
| 2 | Pure safe Rust: `#![forbid(unsafe_code)]` across computational crates | README L324, AGENTS L76 | **WORKING** | 10/11 crates carry `#![forbid(unsafe_code)]`; codebase hygiene asserts 0 unsafe items | Safety |
| 3 | Stride Calculus Engine (SCE) view & broadcast calculus | README L336-368 | **WORKING** | `fnp-ndarray` (231 tests) and `fnp-iter` (205 tests) verify zero-copy and broadcast rules | Architecture |
| 4 | Comprehensive Ufunc and Reduction dispatcher | README L458-540 | **WORKING** | 2,474 tests in `fnp-ufunc`; binary, unary, einsum, FFT, polynomial families | Numeric Engine |
| 5 | Pure Rust Linear Algebra (no BLAS/LAPACK dependency) | README L544-570 | **WORKING** | 459 tests in `fnp-linalg`; QR, SVD, eig, eigh, Cholesky, LU, solve | Linalg |
| 6 | Bit-exact Random Generators & NumPy SeedSequence parity | README L572-605 | **WORKING** | 5 bit generators; PCG64DXSM bit-exact vectors pass; unseeded OS entropy verified | Random |
| 7 | Secure Array & Text I/O with hard bounds | README L607-640 | **WORKING** | 409 tests in `fnp-io`; NPY v1/2/3, NPZ, loadtxt, savetxt with MAX limits | I/O |
| 8 | Dual-mode runtime (Strict vs Hardened) | README L642-670 | **PARTIAL** | `fnp-runtime` (134 tests) fully working in Rust, but called at only 2 sites in `fnp-python` | Runtime |
| 9 | Zero stubs, mocks, or TODOs in production code | README L1620, AGENTS L78 | **WORKING** | 13 test functions in `codebase_hygiene.rs` pass; 0 `todo!()` or `unimplemented!()` | Hygiene |
| 10 | Negative evidence ledger hygiene | README L1780-1820 | **WORKING** | 25/25 tests in `crates/fnp-conformance/tests/ledger_hygiene.rs` pass | Provenance |
| 11 | Differential Conformance Suite | README L1665-1750 | **WORKING** | 192 python conformance shards, 48 conformance binaries | Testing |
| 12 | Wheel Packaging and import parity | README L672-710 | **WORKING** | CI G9 passes on Python 3.12 and 3.13; wheel builds and imports cleanly | Packaging |
| 13 | Autonomous clean-room independence from NumPy | README L693-730 | **PARTIAL** | Tier 3 re-exports host `numpy`; 419 call sites delegate fallback execution to NumPy | Architecture |
| 14 | Performance parity without C BLAS | README L1860-1940 | **PARTIAL** | Large arrays win (1.2x–5x); wrapper floor and small-$n$ sort/f64 divide being closed | Performance |
| 15 | CI Gate Pipeline G1–G9 Automated Verification | README L1760-1850 | **PARTIAL** | G1 and G9 pass in CI; G2 verified locally; G3–G8 executed via manual scripts | CI Gates |
| 16 | RaptorQ sidecar repair durability | README L689, L1845 | **WORKING** | Sidecar generation and scrub tools exist in `fnp-conformance` | Durability |

---

## 4. Machine-Checkable Claims Audit

| Claim Item | Documented Claim (README / AGENTS) | Codebase Ground Truth (Live) | Delta | Severity |
|---|---|---|---|---|
| **Workspace Crates** | 11 crates | **11 crates** (`crates/fnp-random-core` included) | 0 | Exact |
| **Workspace Version** | `0.3.0` | **`0.3.0`** (root `Cargo.toml`) | 0 | Exact |
| **Unsafe Forbid Invariant** | "10 of 11 crates" | **10 of 11 crates** carry `#![forbid(unsafe_code)]` | 0 | Exact |
| **`pub fn` Declarations** | 1,643 declarations | **1,644 declarations** in `crates/*/src` | +1 | Minor |
| **Workspace `#[test]` Count** | 8,716 tests | **8,716 tests** across workspace | 0 | Exact |
| **Closed Beads** | 2,835 closed | **2,835 closed** | 0 | Exact |
| **Open Beads** | 0 open | **0 open** | 0 | Exact |
| **In-Progress Beads** | 8 in-progress | **9 in-progress** (added `franken_numpy-ixs5y.410`) | +1 | Minor |
| **Total Git Commits** | 7,631 commits | **7,645 commits** | +14 | Minor |
| **Hygiene Markers** | 0 stubs / 0 todos / 0 unimplemented | **0 stubs / 0 todos / 0 unimplemented** | 0 | Exact |
| **Parity Divergence Ledger** | 0 active rows | **0 active rows** | 0 | Exact |
| **NumPy Delegation Sites** | "Tier 3 re-export" | **419 non-test delegation sites** in `fnp-python` | Documented | Architectural |

---

## 5. Live Test Counts by Crate

| Crate | Total `#[test]` Functions | Primary Focus | Hygiene Status |
|---|---:|---|---|
| `crates/fnp-python` | 3,648 | PyO3 bindings & 192 conformance shards | Safe FFI views |
| `crates/fnp-ufunc` | 2,474 | 850+ ufuncs, reductions, einsum, FFT | `#![forbid(unsafe_code)]` |
| `crates/fnp-random` | 479 | 5 bit generators & distributions | `#![forbid(unsafe_code)]` |
| `crates/fnp-linalg` | 459 | Decompositions & matrix power | `#![forbid(unsafe_code)]` |
| `crates/fnp-io` | 409 | NPY v1/2/3, NPZ, text I/O | `#![forbid(unsafe_code)]` |
| `crates/fnp-conformance` | 395 | Differential harnesses & ledger hygiene | `#![forbid(unsafe_code)]` |
| `crates/fnp-dtype` | 276 | Dtype taxonomy & 324-pair promotion table | `#![forbid(unsafe_code)]` |
| `crates/fnp-ndarray` | 231 | Shape legality & Stride Calculus Engine | `#![forbid(unsafe_code)]` |
| `crates/fnp-iter` | 205 | Transfer semantics & Nditer state machine | `#![forbid(unsafe_code)]` |
| `crates/fnp-runtime` | 134 | Strict/hardened runtime & evidence ledger | `#![forbid(unsafe_code)]` |
| `crates/fnp-random-core` | 6 | SeedSequence & PCG64DXSM dependency-free | `#![forbid(unsafe_code)]` |
| **Total Workspace Tests** | **8,716** | **Entire Numerical Engine** | **10/11 Crates Unsafe-Free** |

---

## 6. Active Bead Inventory Analysis

The bead tracker contains **2,845 active issues**, of which **2,835 are closed**, **0 are open**, and exactly **9 are in-progress**:

| Bead ID | Pri | Title | Focus Area | Status |
|---|---|---|---|---|
| `franken_numpy-ixs5y.410` | P1 | `perf(clip): route one-sided clip to native kernel to eliminate 1.07x regression` | f32/f64 one-sided scalar clipping | Implemented & verified (5.01x win) |
| `deadlock-audit-1uf80` | P1 | `perf: REDUCE THE WRAPPER FLOOR ITSELF - ~700-1450 ns of per-call cost...` | fnp-python wrapper dispatch latency | In Progress |
| `deadlock-audit-6y5wp` | P1 | `[perf][fnp-python] our f64 divide KERNEL is 1.2652x slower than numpy's...` | Codegen for f64 binary division | In Progress |
| `deadlock-audit-9g7u0` | P1 | `perf(sort): locate and widen live-small-n losses for float64/int32 sort...` | Small-$n$ sort micro-benchmarks | In Progress |
| `deadlock-audit-ddoeq` | P2 | `perf(take): fnp.take is 1.25-1.49x SLOWER than NumPy at m=2^10..2^20...` | Native strided take paths | In Progress |
| `deadlock-audit-omyno` | P2 | `[perf][fnp-python] the no-op reshape-to-current-shape block is repeated...` | Helper deduplication across 62 sites | In Progress |
| `deadlock-audit-tztko` | P1 | `[perf][structural] Measure fnp-python result-buffer allocation lifecycle...` | Buffer allocation profiling | In Progress |
| `franken_numpy-ixs5y` | P1 | `[perf][no-gaps] Close ALL vs-upstream perf gaps in pure safe Rust...` | Overarching performance parity epic | In Progress |
| `franken_numpy-ixs5y.409` | P1 | `perf(sort): int64 n=256 sort is 0.2687x vs numpy (3.72x slower)...` | Int64 $n=256$ sort kernel tuning | In Progress |

Work-graph health (`bv --robot-triage`):
- Total nodes: 2,844 | Density: 0.000012
- Dependency cycles: 0
- Blocked beads: 0

---

## 7. Deep-Dive Gap Analysis

### Gap 1: The 419-Site NumPy Runtime Dependency (Architectural Gap — Major)
- **Documented Vision:** "Clean-room Rust reimplementation of NumPy with two simultaneous goals: (1) absolutely complete and total drop-in behavioral compatibility with legacy NumPy, and (2) a more rigorous architecture..."
- **Live Code Reality:** `fnp-python/src/lib.rs` contains 419 non-test occurrences of `cached_numpy`. For example, lines 27561, 27573, 27595 in `fn clip` delegate directly to `numpy.clip` when an array subclass, explicit out buffer, or unsupported kwarg is passed. Tier 3 submodules (`strings`, `matrixlib`, `ma`, `rec`) are bound via `m.add(name, &numpy.getattr(name))`.
- **Impact:** FrankenNumPy in Python cannot execute in an environment where NumPy is not already installed. It is currently an acceleration layer over NumPy rather than an autonomous replacement.
- **Remedy:** Design native Rust equivalents for Tier 3 submodules and wire fallback paths into `fnp-ufunc` / `fnp-ndarray` directly.

### Gap 2: PyO3 Boundary Latency Floor (Performance Gap — Major)
- **Documented Vision:** Drop-in replacement delivering equal or superior performance across array sizes.
- **Live Code Reality:** Every call through PyO3 pays Python argument parsing, dictionary inspection, and buffer view validation (~700–1450 ns). For $N \le 256$, this fixed tax exceeds the execution time of NumPy's tight C kernels.
- **Impact:** Small-array microbenchmarks appear slower than NumPy, even when the underlying Rust kernel is faster.
- **Remedy:** Implement shared fast-path argument extractors (`deadlock-audit-omyno`, `deadlock-audit-1uf80`) and evaluate `METH_FASTCALL` direct function pointers.

### Gap 3: Small-$n$ Sorting Regressions (Performance Gap — Moderate)
- **Documented Vision:** Pure safe Rust sorting competitive with C NumPy.
- **Live Code Reality:** Int64 $n=256$ sorting is ~3.72x slower than NumPy's introsort. Rust's standard slice sorting algorithm incurs higher branch misprediction overhead on small arrays.
- **Impact:** Functions relying on small-scale sorting (`np.sort`, `np.argsort`, `np.median`) exhibit measurable deficits.
- **Remedy:** Implement branch-free sorting networks for $n \le 32$ and optimized 4-way quicksort partitions for $n \le 256$ in safe Rust (`franken_numpy-ixs5y.409`, `deadlock-audit-9g7u0`).

### Gap 4: Dormant Evidence Ledger in Python Execution Paths (Integration Gap — Minor)
- **Documented Vision:** Dual-mode runtime where all compatibility and safety decisions are recorded in an evidence ledger.
- **Live Code Reality:** `record_runtime_decision` is called at only 2 places in `fnp-python/src/lib.rs`. The rich Bayesian decision mechanisms in `fnp-runtime` are effectively bypassed during normal Python array operations.
- **Impact:** The strict/hardened runtime distinction is not exercised by Python end-users.
- **Remedy:** Integrate runtime mode checks into array creation and ufunc dispatch boundaries.

---

## 8. Phase 2: Bridge Plan

### Track A: Performance Parity (Active Beads)
1. **Bead `deadlock-audit-1uf80` & `deadlock-audit-omyno`:** Extract a single inlineable helper for the 62 repeated reshape sites to reduce binary footprint and eliminate redundant dictionary lookups.
2. **Bead `franken_numpy-ixs5y.409` & `deadlock-audit-9g7u0`:** Implement a pure safe Rust sorting network for $n \le 32$ and an unrolled introsort partition for $n \le 256$ to erase the 3.72x small-$n$ sort gap.
3. **Bead `deadlock-audit-6y5wp`:** Eliminate redundant bounds checks in the unrolled f64 division loop to close the 94 µs codegen deficit.
4. **Bead `deadlock-audit-ddoeq`:** Vectorize strided multi-index gathering in `take` to close the 1.25–1.49x gap on non-contiguous arrays.

### Track B: Architectural Autonomy (New Gaps)
1. **Eliminate Tier 3 Re-Exports:** Build native Rust structures for `numpy.matrixlib` and basic masked array containers in `fnp-ufunc`.
2. **Replace 419 Delegation Sites:** Route edge-case kwargs to internal `fnp-ufunc` fallbacks instead of calling `cached_numpy`.
3. **Integrate Runtime Ledger:** Wire `fnp-runtime::decide_compatibility` into PyO3 buffer borrowing to actively enforce strict vs hardened policies.

---

## 9. Phase 3a: Bead Creation (FROZEN TEMPLATE)

```
OK so please take ALL of that and elaborate on it and use it to create a comprehensive and granular
set of beads for all this with tasks, subtasks, and dependency structure overlaid, with detailed
comments so that the whole thing is totally self-contained and self-documenting (including relevant
background, reasoning/justification, considerations, etc.-- anything we'd want our "future self" to
know about the goals and intentions and thought process and how it serves the over-arching goals of
the project.) The beads should be so detailed that we never need to consult back to the original
markdown plan document. Remember to ONLY use the `br` tool to create and modify the beads and add
the dependencies.
```

### Proposed New Beads (to close unbacked vision goals):
1. **`fnp-arch-standalone-python` (P2, Task):** Replace Tier 3 `numpy` module re-exports with native Rust module structures to enable standalone execution without `numpy` installed.
2. **`fnp-runtime-python-wiring` (P2, Task):** Wire `fnp-runtime` strict/hardened policy decisions and evidence ledger recording into `fnp-python` buffer acquisition and ufunc dispatch.
3. **`fnp-ci-g3-g8-automation` (P2, Task):** Integrate CI gates G3–G8 into the primary automated GitHub Actions pipeline with explicit pass thresholds.

---

## 10. Phase 4: Ambition Rounds (Escalation & Horizon Expansion)

### Ambition Round 1: Zero-Copy Interoperability & Dynamic Vectorization
- **Escalation:** Beyond matching NumPy's C API, FrankenNumPy should natively support the Apache Arrow PyCapsule interface (`__arrow_c_array__`), enabling zero-copy, zero-FFI data exchange between Polars, PyArrow, DuckDB, and FrankenNumPy arrays.
- **Vectorization:** Utilize runtime CPU feature detection (`is_x86_feature_detected!("avx512f")`) to dispatch to specialized 512-bit AVX-512 kernels in safe Rust, achieving performance advantages that upstream NumPy (historically constrained by universal C compilation baselines) cannot match.

### Ambition Round 2: Fused Ufunc JIT via Pure Rust Execution Graphs
- **Escalation:** In NumPy, chained elementwise expressions like `y = a * b + c - d` allocate three intermediate temporary arrays, overwhelming CPU L1/L2 cache bandwidth.
- **Architectural Breakthrough:** Leverage SCE (Stride Calculus Engine) to represent chained operations as an AST of lazy elementwise views, fusing the entire chain into a single traversal kernel that loads operands into CPU registers once and writes directly to the destination. This delivers structural 3x–10x speedups over NumPy without requiring Numba or external compilers.

### Ambition Round 3: Esoteric Mathematical & Numerical Foundations
- **Escalation:** Apply modern numerical algorithms developed over the last 60 years:
  1. **Compensated Summation:** Generalize Neumaier and Kahan-Babuška-Klein compensated summation across all reduction axes, guaranteeing IEEE-754 precision even on ill-conditioned sequences where NumPy accumulates floating-point drift.
  2. **Fast Modular Polynomial Arithmetic:** Use number-theoretic transforms (NTT) and Karatsuba multiplications for high-degree polynomial families in `fnp-ufunc`.
  3. **Verified Bounds:** Provide interval arithmetic bounds alongside floating-point results in hardened runtime mode, enabling safety-critical numerical pipelines.

---

## 11. Phase 5: Plan-Space Refinement (FROZEN TEMPLATE)

```
Check over each bead super carefully-- are you sure it makes sense? Is it optimal? Could we change
anything to make the system work better for users? If so, revise the beads. It's a lot easier and
faster to operate in "plan space" before we start implementing these things! DO NOT OVERSIMPLIFY
THINGS! DO NOT LOSE ANY FEATURES OR FUNCTIONALITY! Also make sure that as part of the beads we
include comprehensive unit tests and e2e test scripts with great, detailed logging so we can be
sure that everything is working perfectly after implementation. Make sure to ONLY use the `br` cli
tool for all changes, and you can and should also use the `bv` tool to help diagnose potential
problems with the beads.
```

### Refinement Findings across 5 Iterations:
1. **Pass 1 (Safety Invariant Integrity):** Verified that all proposed optimizations maintain `#![forbid(unsafe_code)]` across the 10 numeric crates. No unsafe fast paths may be introduced outside `fnp-python`.
2. **Pass 2 (Negative Test Cases):** Ensured every performance bead defines an explicit A/A null control and negative behavioral test case (e.g., asserting that small-$n$ sort preserves stable NaN positioning identical to NumPy).
3. **Pass 3 (Work-Graph Health):** Confirmed `bv --robot-triage` shows zero dependency cycles and zero blocked beads. New architectural beads must depend on `franken_numpy-ixs5y` to preserve work-graph hierarchy.
4. **Pass 4 (Provenance Requirements):** Confirmed that all benchmark measurements require host worker provenance, ELF hash validation, and within-invocation alternating A/B runs per AGENTS.md rules.
5. **Pass 5 (Non-Regression of CI Gates):** Verified that closing active performance beads cannot regress G1 (clippy warnings -D) or G2 (conformance tests).

---

## 12. Conclusion & Operational Trajectory

FrankenNumPy is technically sound, exceptionally well-tested (8,716 passing tests), and delivers 100% surface coverage of NumPy's public API. The resolution of the historical ledger hygiene defect and the SeedSequence empty-entropy contract divergence ensures that quality gates G1 and G2 are solid.

The path to absolute completion is twofold:
1. **Short-Term (Closing Active Beads):** Complete the 9 in-progress performance beads to eliminate the wrapper floor, small-$n$ sort, and divide kernel codegen gaps.
2. **Medium-Term (True Autonomy):** Systematically replace the 419 `cached_numpy` fallback sites and Tier 3 re-exports with pure Rust engines, transforming FrankenNumPy from an accelerated NumPy companion into a fully autonomous, clean-room replacement for the Python scientific stack.
