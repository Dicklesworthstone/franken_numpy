# ADR-001: Pivot from Parity Grinding to Phase 3 (FFI/BLAS/Threading)

## Status

**ACCEPTED** (owner decision 2026-09-05, via bead `deadlock-audit-u2z2b`): revive the three dormant streams rather than close the ADR as partially adopted.

Decision record 2026-09-05: the owner reviewed the state above and ruled to **revive the dormant streams**:
1. **Python packaging** — effectively decided by events: shipped 2026-09-03 (`pyproject.toml` + maturin). Remaining work (PyPI publish) tracked separately; no new stream bead opened.
2. **BLAS/LAPACK linkage** — revived as an opt-in *capability* exploration. Scope note: the suite-wide dependency-smuggling rule still forbids linking C BLAS *to win a benchmark*; the revived stream explores optional linkage for capability/compat surfaces (e.g. LAPACK-only fallbacks for decompositions), never as a benchmark arm, and must not gate the default no-external-C build.
3. **Native i64 GEMM** — revived against the measured 4–7x per-thread wall (docs/NEGATIVE_EVIDENCE.md, 2026-06-27), which stands as the recorded baseline any attempt must beat or explicitly accept; the integer-BLAS-missing frontier (`int` matmul has no BLAS) is the primary target, per the ledger's own domination analysis.
Implementation beads for streams 2 and 3 are filed in the tracker; SIMD/multi-threading streams remain shipped and closed.

Update 2026-09-03 (reality check): this ADR was never decided, but events decided three of its five streams. **SIMD** and **multi-threading** shipped during the June-July v0.2.0 campaign (`core::simd` kernels, `+avx2` codegen, `rayon` as an unconditional dependency with per-op size gates rather than the feature-gated `parallel` flag proposed below). **Python packaging** landed on 2026-09-03 as `pyproject.toml` + maturin (`pip install .`, Linux / CPython 3.13 verified; no PyPI publish yet). **BLAS/LAPACK linkage** was superseded by the suite-wide rule that forbids linking C BLAS to win a benchmark (`/data/projects/AGENTS.md`, "Dependency smuggling"), and the pure-Rust GEMM was quantified as a 4-7x per-thread wall (docs/NEGATIVE_EVIDENCE.md, 2026-06-27), with f64 2-D GEMM now delegating to NumPy in the default regime; the "Phase 3 candidate" wording in the README is therefore stale for this stream. **Native i64/u64 arithmetic** has not started: `UFuncArray` still holds `values: Vec<f64>` plus `IntegerSidecar` (ten integer binary ops recompute exactly on the sidecar), while the `fnp_python` fast paths compute integers in native width without touching `UFuncArray`. The decision block below remains for the owner; the practical question left is whether to record the BLAS stream as withdrawn and the native-i64 stream as the one open Phase 3 item.

Update 2026-05-16: The parity track this ADR considered pivoting *away from* has since reached 100% numpy.__all__ coverage (see audit_numpy_reality.md). The proposal's core question — should the project move on to Phase 3 (FFI / BLAS / Threading) — remains open for decision, but the empirical case has changed: the parity surface that looked stalled in April is now complete and structurally locked.

---

## Context

### Historical State Snapshot (2026-04-10)

This snapshot captured the repo state when the pivot ADR was drafted. It is historical context, not the live tracker or test count; use `bv --robot-triage` and current validation output for present-day counts.
- **9 crates**, ~113K lines of safe Rust, zero `unsafe` blocks
- **~2,960 tests** passing across the workspace
- **372 oracle-verified fixture cases** with 1:1 input/oracle alignment
- **50 beads closed** in the tracker snapshot at the time
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
1. **Python packaging / drop-in installability** - an early `fnp-python` PyO3 bridge exists, but there is still no `pip install frankennumpy` packaging story or packaged drop-in NumPy replacement.
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

1. **Python packaging and FFI hardening** - package the existing `fnp-python` PyO3 bridge and widen coverage toward drop-in installability
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

### Draft Bead 1: Python Packaging and FFI Hardening

**Title:** Package and harden the existing PyO3 bridge for a pip-installable preview
**Priority:** P1
**Scope:** 
- Add repo-root Python packaging metadata for the existing `crates/fnp-python` PyO3 extension module.
- Continue exposing UFuncArray, Generator, and core ops through `fnp-python` until the documented wrapper surface is packaged and tested.
- Package as a preview `franken-numpy` distribution only after import, wheel, and parity smoke tests are reproducible.
- Design questions:
  - Zero-copy vs copy semantics for array interchange?
  - NumPy buffer protocol support?
  - GIL management strategy?

### Draft Bead 2: BLAS Backend Integration (Decided: bead deadlock-audit-ceb7k)

**Title:** Integrate OpenBLAS/MKL as optional BLAS backend for linalg
**Priority:** P1 / Phase 3 Capability Exploration
**Status:** DESIGN DECIDED (bead `deadlock-audit-ceb7k`, 2026-09-08)
**Scope & Decisions:**
- Feature name: `cblas-lapack` in `crates/fnp-linalg/Cargo.toml` (default = off), exposed as `blas` / `lapack` in workspace root.
- Invariant: The default 9-crate build remains strictly `#![forbid(unsafe_code)]` with zero external C dependencies.
- Linkage: Dynamic system linkage via pkg-config (`cblas` / `lapack`) to avoid compiling external Fortran/C sources.
- FFI quarantine: Isolated, audited submodule under `crates/fnp-linalg/src/ffi/`.
- Suite-wide governance: Strict adherence to AGENTS.md "Dependency smuggling" — this feature is an opt-in capability provider, never enabled in benchmark comparisons against NumPy or used to claim speedups.
- Route mapping: Level 3 BLAS (`gemm`, `syrk`, `trsm` for $N \ge 64$), LAPACK solvers (`dgesv`, `dpotrf`/`dpotrs`, `dgeqrf`/`dorgqr`, `dgesdd`, `dsyev`/`dgeev`).

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

### Draft Bead 5: Native i64/u64 Arithmetic (Decided: bead deadlock-audit-reqc1)

**Title:** Implement native integer arithmetic without f64 intermediary
**Priority:** P3 / Phase 3 Stream
**Status:** DESIGN DECIDED (bead `deadlock-audit-reqc1`, 2026-09-08)
**Scope & Decisions:**
- Python surface: High-throughput integer paths in `fnp-python` (matmul, vecmat, batched matmul, floordiv, rem, divmod, gcd, lcm, pow, where, clip, diff, ediff1d, putmask) already operate on native `PyBuffer<i64>/<u64>/<i32>` without f64 conversion, preserving exact integers $> 2^{53}$.
- Integer GEMM: MR=4 register-blocked parallel wrapping GEMM delivers bit-exact execution with large speedups where NumPy lacks BLAS.
- Storage evolution: Target architecture evolves `UFuncArray` storage from `values: Vec<f64> + IntegerSidecar` to unified `ArrayStorage` enum (`F64(Vec<f64>)`, `I64(Vec<i64>)`, `U64(Vec<u64>)`, etc.) to eliminate sidecar sync tax while preserving SCE/fnp-dtype determinism.
- Overflow & promotion semantics: Wrapping two's complement (`wrapping_add`, `wrapping_sub`, `wrapping_mul`) strictly matching NumPy C integer behavior. Deterministic promotion lattice via SCE.
- Competitive rule discipline: Any competitive claim must run same-invocation head-to-head against NumPy under A/A dual nulls without violating bit-exactness (no k-blocking).

---

## Evidence Artifacts

- Cross-engine baseline: `artifacts/baselines/cross_engine_benchmark_v1.json`
- Markdown report: `artifacts/baselines/cross_engine_benchmark_v1.report.md`
- Workload manifest: `artifacts/contracts/cross_engine_benchmark_workloads_v1.yaml`
- This ADR: `docs/adr/ADR-001-parity-pivot.md`

---

*Generated by CloudyMarsh (claude-opus-4-5) on 2026-04-10*
*Decision recorded 2026-09-05 (owner, via deadlock-audit-u2z2b): ACCEPTED — dormant streams revived; see the Status section.*
