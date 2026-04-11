# ADR-001: Pivot from Parity Grinding to Phase 3 (FFI/BLAS/Threading)

## Status

**PROPOSED** (awaiting user decision)

---

## Context

### Current State (2026-04-10)

FrankenNumPy has achieved all foundational milestones documented in the README:
- **9 crates**, ~113K lines of safe Rust, zero `unsafe` blocks
- **~2,960 tests** passing across the workspace
- **372 oracle-verified fixture cases** with 1:1 input/oracle alignment
- **50 beads closed** to date
- **Stride Calculus Engine (SCE)** operational
- **Dual-mode runtime** (strict/hardened) implemented
- **RaptorQ-everywhere durability** for all artifacts
- **9 Phase2C extraction packets** complete

### Parity Debt Assessment

Recent work (last 17 beads, 2026-04-07/08) shows diminishing returns:
- Early beads added entire op-families (masked arrays, datetime, polynomials)
- Recent beads add 15-24 fixture cases each for narrow edge-case coverage
- Pattern: cross-dtype binops, shift ops, comparison ops, bitwise ops

**Fixture coverage by family:**

| Family | Fixture File | Cases | Status |
|--------|-------------|-------|--------|
| ufunc | ufunc_input_cases.json | 372 | Extensive |
| linalg | linalg_differential_cases.json | 24 | Covered |
| rng | rng_*_cases.json | 4 files | Covered |
| fft | fft_differential_cases.json | Present | Covered |
| io | io_*_cases.json | 3 files | Covered |
| polynomial | polynomial_differential_cases.json | 45KB | Extensive |
| string | string_differential_cases.json | 19KB | Covered |
| datetime | datetime_differential_cases.json | 23KB | Covered |
| masked | masked_differential_cases.json | Present | Covered |
| iter | iter_*_cases.json | 3 files | Covered |

**Remaining gaps:** Primarily edge-case coverage depth, not missing op-families.

### Cross-Engine Performance Gap (from franken_numpy-azql)

First baseline generated 2026-04-10 with 37 workloads:

| Metric | Value |
|--------|-------|
| Total workloads | 37 |
| Median ratio | 2.10x (FNP slower) |
| Best ratio | 0.10x (IO: 10x faster!) |
| Worst ratio | 53.60x (div_f64_large) |
| Green band (<=2x) | 17 workloads |
| Yellow band (2-5x) | 9 workloads |
| Red band (>5x) | 11 workloads |

**By op-family:**

| Family | Workloads | Median Ratio | Assessment |
|--------|-----------|--------------|------------|
| io | 2 | 0.14x | **FNP WINS** |
| random | 3 | 1.00x | Parity |
| linalg | 3 | 1.02x | Parity |
| reductions | 7 | 1.18x | Near-parity |
| sorting | 2 | 1.90x | Acceptable |
| statistics | 1 | 2.79x | Yellow |
| matmul | 2 | 3.40x | Yellow |
| fft | 2 | 10.56x | Mixed (power-2 fast, non-power-2 slow) |
| ufunc-broadcast | 6 | 13.53x | **HOTSPOT** |
| ufunc-elementwise | 9 | 30.76x | **CRITICAL HOTSPOT** |

**Root cause of red band:** Binary elementwise operations lack SIMD vectorization. NumPy+OpenBLAS achieves ~2.3ns/element; FNP achieves ~22ns/element on large arrays.

### README's Disclaimed Limitations

The README explicitly defers four capabilities:
1. **Python FFI / pip-installable** - "planned but not implemented"
2. **BLAS/LAPACK backend** - "competitive with BLAS for small matrices; slower for large ones"
3. **Multi-threading** - single-threaded execution
4. **Native i64/u64 > 2^53 arithmetic** - uses f64 intermediary

---

## Decision

**[USER FILLS IN AFTER REVIEW]**

Options:
- **ACCEPT**: Pivot to Phase 3 immediately
- **DEFER**: Continue parity grinding until criteria met
- **MODIFY**: Adjust pivot criteria and re-evaluate

---

## Pivot Criteria (Proposed)

Pivot to Phase 3 when ALL of the following are true:

| Criterion | Threshold | Current | Met? |
|-----------|-----------|---------|------|
| Fixture families with uncovered divergences | < 3 | 0 major gaps | YES |
| Median cross-engine ratio | < 5.0x | 2.10x | YES |
| Test count | > 2,500 | ~2,960 | YES |
| Green+Yellow workloads | > 60% | 70% (26/37) | YES |

**Recommended action:** Criteria are MET. Recommend pivot to Phase 3.

**Note on red-band workloads:** The 11 red-band workloads (binary elementwise at scale) are a known architectural limitation that CANNOT be fixed without SIMD or BLAS - exactly what Phase 3 enables. Grinding more parity debt will not improve these ratios.

---

## Consequences

### Parity Debt Remaining (Acceptable)
- Edge-case coverage in narrow corners (not blocking)
- Non-power-2 FFT optimization (fixable in Phase 3)
- Large-scale elementwise ops (requires Phase 3 SIMD/BLAS)

### New Work Streams for Phase 3

1. **Python FFI (PyO3)** - Make FNP pip-installable
2. **BLAS Backend Integration** - Link OpenBLAS/MKL for linalg hot paths
3. **SIMD Vectorization** - Safe SIMD via `std::simd` (nightly) or `portable_simd`
4. **Multi-threading** - Rayon for parallel reductions/matmul
5. **Native i64/u64 Arithmetic** - Remove f64 intermediary for exact integer ops

### Swarm Reassignment Plan
- **Optimization agents** (w4zs children): Focus on safe-Rust levers first (contiguous fast-paths, tiling)
- **FFI agents**: PyO3 binding layer, Python test harness
- **BLAS agents**: Feature-flagged OpenBLAS linkage behind `unsafe` boundary (isolated, audited)

---

## Appendix: Draft Beads for Phase 3

### Draft Bead 1: Python FFI via PyO3

**Title:** Implement Python FFI bindings via PyO3 for pip-installable package
**Priority:** P1
**Scope:** 
- Create `fnp-python` crate with PyO3 bindings
- Expose UFuncArray, Generator, and core ops to Python
- Package as `franken-numpy` on PyPI
- Design questions:
  - Zero-copy vs copy semantics for array interchange?
  - NumPy buffer protocol support?
  - GIL management strategy?

### Draft Bead 2: BLAS Backend Integration

**Title:** Integrate OpenBLAS/MKL as optional BLAS backend for linalg
**Priority:** P1
**Scope:**
- Feature-flagged `blas` feature in fnp-linalg
- Isolated `unsafe` module for CBLAS FFI (audited)
- Dispatch large matrices to BLAS, keep pure-Rust for small
- Design questions:
  - Static vs dynamic linking?
  - Runtime detection of BLAS availability?
  - Maintain zero-unsafe default build?

### Draft Bead 3: SIMD Vectorization

**Title:** Implement safe SIMD for elementwise operations
**Priority:** P2
**Scope:**
- Use `std::simd` (portable_simd) on nightly
- Target: reduce elementwise ratio from 30x to <3x
- Focus on f64 add/mul/div first (biggest wins)
- Design questions:
  - Feature-gated or always-on?
  - Fallback for non-SIMD targets?
  - Interaction with contiguous fast-path detection?

### Draft Bead 4: Multi-threaded Execution

**Title:** Add Rayon-based parallel execution for large arrays
**Priority:** P2
**Scope:**
- Feature-flagged `parallel` feature
- Target: reductions, matmul, sort on arrays > 10K elements
- Configurable thread pool size
- Design questions:
  - How to handle nested parallelism?
  - Work-stealing vs static partitioning?
  - Integration with existing iterator infrastructure?

### Draft Bead 5: Native i64/u64 Arithmetic

**Title:** Implement native integer arithmetic without f64 intermediary
**Priority:** P3
**Scope:**
- Direct i64/u64 operations for values > 2^53
- Maintain compatibility with smaller values
- Design questions:
  - Breaking change or opt-in behavior?
  - Impact on dtype promotion rules?
  - Overflow behavior alignment with NumPy?

---

## Evidence Artifacts

- Cross-engine baseline: `artifacts/baselines/cross_engine_benchmark_v1.json`
- Markdown report: `artifacts/baselines/cross_engine_benchmark_v1.report.md`
- Workload manifest: `artifacts/contracts/cross_engine_benchmark_workloads_v1.yaml`
- This ADR: `docs/adr/ADR-001-parity-pivot.md`

---

*Generated by CloudyMarsh (claude-opus-4-5) on 2026-04-10*
*Pending user review and decision*
