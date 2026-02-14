# EXISTING_NUMPY_STRUCTURE

## DOC-PASS-00 Baseline Gap Matrix + Quantitative Expansion Targets

Snapshot baseline (2026-02-14):

| Document | Baseline lines | Target lines | Expansion multiplier | Status |
|---|---:|---:|---:|---|
| `EXISTING_NUMPY_STRUCTURE.md` | 62 | 992 | 16.0x | baseline pass complete (`bd-23m.24.1` closed) |
| `EXHAUSTIVE_LEGACY_ANALYSIS.md` | 275 | 3300 | 12.0x | baseline pass complete (`bd-23m.24.1` closed) |

Gap matrix for this document (pass-1 planning):

| Area | Current state | Missing for parity-grade docs | Coverage implications to record |
|---|---|---|---|
| Subsystem map detail | coarse bullets | per-package ownership + boundary invariants + packet IDs | unit/property + differential + e2e + structured logging per subsystem |
| Semantic hotspots | short list | explicit legality formulas and tie-break rules | metamorphic/property families and adversarial classes |
| Compatibility-critical behavior | high-level statements | strict/hardened split with drift budget and fail-closed triggers | runtime-policy reason-code matrix and replay fields |
| Risk areas | broad categories | threat-to-control-to-test mappings | parser fuzz corpus, crash triage IDs, forensic log fields |
| Conformance families | list only | fixture IDs, oracle anchors, and closure gates | coverage ledger with covered/missing/deferred status |

Pass-1 explicit coverage/traceability matrix (covered, missing, deferred):

| Subsystem | Legacy anchors | Executable evidence anchors | Unit/Property | Differential | E2E | Structured logging |
|---|---|---|---|---|---|---|
| Shape/stride | `numpy/_core/src/multiarray/shape.c`, `shape.h` | `crates/fnp-conformance/fixtures/shape_stride_cases.json` | covered (partial) | covered (partial via ufunc diff) | missing | covered (partial via runtime-policy logs) |
| Dtype promotion/cast | `numpy/_core/src/multiarray/dtypemeta.c`, `descriptor.c`, `can_cast_table.h` | `crates/fnp-conformance/fixtures/dtype_promotion_cases.json` | covered (partial) | missing (cast-matrix diff missing) | missing | missing |
| Ufunc dispatch | `numpy/_core/src/umath/ufunc_object.c` | `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-conformance/fixtures/ufunc_*` | covered (partial metamorphic/adversarial) | covered (partial) | missing | covered (fixture-level fields) |
| Transfer/alias | `numpy/_core/src/multiarray/dtype_transfer.c`, `lowlevel_strided_loops.c.src` | none yet (legacy only) | missing | missing | missing | missing |
| NDIter | `numpy/_core/src/multiarray/nditer*` | none yet (legacy only) | missing | missing | missing | missing |
| Random | `numpy/random/*.pyx`, `numpy/random/src/*` | `crates/fnp-random/src/lib.rs` (stub) | missing | missing | missing | missing |
| Linalg | `numpy/linalg/lapack_lite/*` | `crates/fnp-linalg/src/lib.rs` (stub) | missing | missing | missing | missing |
| IO | `numpy/lib/format.py`, npy/npz handling paths | `crates/fnp-io/src/lib.rs` (stub) | missing | missing | missing | missing |

Traceability anchors:

- Legacy source roots: `/data/projects/franken_numpy/legacy_numpy_code/numpy`
- Machine contracts: `artifacts/contracts/test_logging_contract_v1.json`, `artifacts/contracts/phase2c_contract_schema_v1.json`
- Gate artifacts: `crates/fnp-conformance/src/test_contracts.rs`, `crates/fnp-conformance/src/ufunc_differential.rs`, `scripts/e2e/run_test_contract_gate.sh`

Contradictions/unknowns register (for closure in doc passes 01-10):

| ID | Item | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `DOC-C001` | Section 6 carries historical reduced-scope language incompatible with full drop-in parity doctrine. | critical | `bd-23m.24.2` | Replace with parity-debt sequencing model and explicit owner/blocker table. |
| `DOC-C002` | No per-subsystem mapping of unit/property, differential, e2e, and structured log implications. | high | `bd-23m.24.4` | Add verification matrix with artifact links and missing/deferred rationale. |
| `DOC-C003` | Unknown ownership for unresolved NDIter/RNG/linalg/IO extraction evidence. | high | `bd-23m.24.3` | Add packet ownership + closure gates for each unresolved behavior family. |

## 1. Legacy Oracle

- Root: /dp/franken_numpy/legacy_numpy_code/numpy
- Upstream: numpy/numpy

## 2. Subsystem Map

- numpy/_core/src/multiarray: ndarray construction, shape/stride logic, assignment, nditer, text parsing.
- numpy/_core/src/umath: ufunc machinery and reduction kernels.
- numpy/_core/include/numpy: public ndarray/dtype APIs and ABI contracts.
- numpy/_core/src/_simd: CPU feature detection and SIMD dispatch checks.
- numpy/random and random/src: BitGenerator implementations and distribution paths.
- numpy/lib, numpy/linalg, numpy/fft, numpy/matrixlib, numpy/ma: higher-level Python semantics.
- numpy/tests and package-specific test folders: regression and parity baseline.

## 3. Semantic Hotspots (Must Preserve)

1. shape.c/shape.h dimensional and broadcast legality.
2. descriptor/dtypemeta and casting tables.
3. lowlevel_strided_loops and dtype_transfer assignment semantics.
4. nditer behavior across memory order/stride/layout combinations.
5. stride_tricks behavior for views and broadcasted representations.
6. ufunc override and dispatch selection semantics.
7. dtype promotion matrix determinism.

## 4. Compatibility-Critical Behaviors

- Array API and array-function overrides.
- Public PyArrayObject layout expectations for downstream interop.
- Runtime SIMD capability detection affecting chosen kernels.
- ufunc reduction vs elementwise override precedence.

## 5. Security and Stability Risk Areas

- textreading parser paths and malformed input handling.
- stringdtype conversion routines and buffer bounds.
- stride arithmetic UB risk in low-level loops.
- random generator state determinism and serialization paths.
- external data and dlpack interoperability boundaries.

## 6. Historical V1 Extraction Boundary (Deprecated; convert to parity-debt ledger)

Include now:
- ndarray/dtype core, ufunc/reduction core, promotion/casting contracts, basic npy/npz and RNG parity.

Exclude for V1:
- f2py, packaging/docs/tooling stacks, non-critical module breadth.

## 7. High-Value Conformance Fixture Families

- _core/tests for dtype, array interface, overlap, simd, string dtype.
- random/tests for deterministic generator streams.
- lib/linalg/fft/polynomial/matrixlib/ma tests for scoped behavioral parity.
- tests/test_public_api and testing/tests for API surface stability.

## 8. Extraction Notes for Rust Spec

- Start with shape/stride/dtype model before any heavy optimization.
- Keep promotion matrix and casting behavior explicit and versioned.
- Use parity-by-op-family reports as release gates.
