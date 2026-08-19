# COMPREHENSIVE_SPEC_FOR_FRANKENNUMPY_V1

Historical filename note: the `_V1` suffix is retained for lineage only; this spec defines a full-parity, true drop-in replacement target.

## 0. Prime Directive

Build a system that is simultaneously:

1. Behaviorally trustworthy at full drop-in NumPy parity.
2. Mathematically explicit in decision and risk handling.
3. Operationally resilient via RaptorQ-backed durability.
4. Performance-competitive via profile-and-proof discipline.

Crown-jewel innovation:

Stride Calculus Engine (SCE): deterministic shape/stride/broadcast legality and zero-copy view guarantees.

Legacy oracle:

- /dp/franken_numpy/legacy_numpy_code/numpy
- upstream: https://github.com/numpy/numpy

## 1. Product Thesis

Most reimplementations fail by being partially compatible and operationally brittle. FrankenNumPy will instead combine compatibility realism with first-principles architecture and strict quality gates.

## 2. Absolute Parity Contract (Non-Negotiable)

FrankenNumPy targets ABSOLUTELY COMPLETE and TOTAL feature/functionality overlap with legacy NumPy as a true drop-in replacement.

Delivery may be phased, but phase boundaries are sequencing only:

- uncovered behavior is parity debt, not accepted scope reduction
- each parity-debt item must carry owner, risk, blocker, and closure gate
- reduced-scope "V1 is good enough" acceptance is explicitly forbidden

## 3. Architecture Blueprint

array API -> shape/stride engine -> ufunc dispatcher -> numeric kernels -> IO

Planned crate families:
- fnp-dtype
- fnp-ndarray
- fnp-ufunc
- fnp-linalg
- fnp-random
- fnp-fft
- fnp-io
- fnp-conformance
- frankennumpy

## 4. Compatibility Model (frankenlibc/frankenfs-inspired)

Two explicit operating modes:

1. strict mode:
   - maximize observable compatibility for the full legacy NumPy behavior surface
   - no behavior-altering repair heuristics
2. hardened mode:
   - maintain outward contract while enabling defensive runtime checks and bounded repairs

Compatibility focus for this project:

Preserve NumPy-observable shape, broadcast, promotion, and view semantics across the full legacy behavior matrix.

Fail-closed policy:

- unknown incompatible features or protocol fields must fail closed by default
- compatibility exceptions require explicit allowlist entries and audit traces

## 5. Security Model

Security focus for this project:

Harden parser/IO and shape-validation boundaries; prevent malformed shape and unsafe cast pathways.

Threat model baseline:

1. malformed input and parser abuse
2. state-machine desynchronization
3. downgrade and compatibility confusion paths
4. persistence corruption and replay tampering

Mandatory controls:

- adversarial fixtures and fuzz/property suites for high-risk entry points
- deterministic audit trail for recoveries and mode/policy overrides
- explicit subsystem ownership and trust-boundary notes

## 6. Alien-Artifact Decision Layer

Runtime controllers (scheduling, adaptation, fallback, admission) must document:

1. state space
2. evidence signals
3. loss matrix with asymmetric costs
4. posterior or confidence update model
5. action rule minimizing expected loss
6. calibration fallback trigger

Output requirements:

- evidence ledger entries for consequential decisions
- calibrated confidence metrics and drift alarms

## 7. Extreme Optimization Contract

Measure op-family throughput, tail latency, and memory bandwidth efficiency; gate regressions for broadcast and reduction hotspots.

Optimization loop is mandatory:

1. baseline metrics
2. hotspot profile
3. single-lever optimization
4. behavior-isomorphism proof
5. re-profile and compare

No optimization is accepted without associated correctness evidence.

## 8. Correctness and Conformance Contract

Maintain deterministic shape calculus, alias correctness, and dtype promotion table invariants.

Conformance process:

1. generate canonical fixture corpus
2. run legacy oracle and capture normalized outputs
3. run FrankenNumPy and compare under explicit equality/tolerance policy
4. produce machine-readable parity report artifact

Assurance ladder:

- Tier A: unit/integration/golden fixtures
- Tier B: differential conformance
- Tier C: property/fuzz/adversarial tests
- Tier D: regression corpus for historical failures

Test contract lock (release-blocking for scoped surfaces):

1. Machine schema: `artifacts/contracts/test_logging_contract_v1.json`
2. Human conventions: `artifacts/contracts/TESTING_AND_LOGGING_CONVENTIONS_V1.md`
3. Gate runner: `cargo run -p fnp-conformance --bin run_test_contract_gate`
4. E2E wrapper: `scripts/e2e/run_test_contract_gate.sh`

Mandatory fields for fixture-level structured logs:

1. `fixture_id`
2. `seed`
3. `mode`
4. `env_fingerprint`
5. `artifact_refs`
6. `reason_code`

## 9. RaptorQ-Everywhere Durability Contract

RaptorQ repair-symbol sidecars are required for long-lived project evidence:

1. conformance snapshots
2. benchmark baselines
3. migration manifests
4. reproducibility ledgers
5. release-grade state artifacts

Required artifacts:

- symbol generation manifest
- scrub verification report
- decode proof for each recovery event

## 10. Milestones and Exit Criteria

### M0 — Bootstrap

- workspace skeleton
- CI and quality gate wiring

Exit:
- fmt/check/clippy/test baseline green

### M1 — Core Model

- core data/runtime structures
- first invariant suite

Exit:
- invariant suite green
- first conformance fixtures passing

### M2 — First Vertical Slice

- end-to-end workflow implemented for first-wave capabilities with explicit parity-debt ledger updates

Exit:
- differential parity for first major API family
- baseline benchmark report published

### M3 — Parity Closure Expansion

- additional API families closed toward full legacy matrix parity

Exit:
- expanded parity reports green
- no unresolved critical compatibility defects

### M4 — Hardening

- adversarial coverage and perf hardening

Exit:
- regression gates stable
- conformance drift zero for implemented surface plus parity-debt burn-down evidence toward full closure

## 11. Acceptance Gates

Gate A: compatibility parity report passes against the full legacy matrix for implemented surface, with explicit parity-debt ledger for remaining gaps.

Gate B: security/fuzz/adversarial suite passes for high-risk paths.

Gate C: performance budgets pass with no semantic regressions.

Gate D: RaptorQ durability artifacts validated and scrub-clean.

All four gates must pass for release readiness; no release claim is valid without explicit full-parity trajectory and owned parity-debt closure plan.

## 12. Risk Register

Primary risk focus:

Subtle aliasing or promotion bugs hidden behind passing micro-tests.

Mitigations:

1. compatibility-first development for risky API families
2. explicit invariants and adversarial tests
3. profile-driven optimization with proof artifacts
4. strict mode/hardened mode separation with audited policy transitions
5. RaptorQ-backed resilience for critical persistent artifacts

## 13. Immediate Execution Checklist

1. Create workspace and crate skeleton.
2. Implement highest-impact end-to-end path while preserving full-parity target and updating parity-debt governance artifacts.
3. Stand up differential conformance harness against legacy oracle.
4. Add benchmark baseline generation and regression gating.
5. Add RaptorQ sidecar pipeline for conformance and benchmark artifacts.

## 14. Detailed Crate Contracts

| Crate | Primary Responsibility | Explicit Non-Goal | Invariants | Mandatory Tests |
|---|---|---|---|---|
| fnp-dtype | dtype taxonomy + promotion tables | kernel execution | promotion table deterministic and versioned | pairwise promotion matrix tests |
| fnp-ndarray | shape/stride/storage model | ufunc dispatch | no invalid stride state, view metadata consistency | shape/stride property tests |
| fnp-ufunc | elementwise dispatch + broadcast executor | IO | broadcast legality and result-shape determinism | broadcast edge corpus, scalar-vector-matrix parity |
| fnp-linalg | dense algebra bridge and kernels | random/distribution APIs | solver input/output shape and tolerance contracts | decomposition fixtures, residual checks |
| fnp-random | RNG streams and distributions | fft/linear algebra | deterministic seed replay for all shipped generators; unshipped generator parity tracked as explicit debt | seed replay matrix, distribution sanity checks |
| fnp-fft | fft execution and plan policy | parsing/storage | transform shape and inverse consistency | fft round-trip fixtures |
| fnp-io | npy/npz parsing and encoding | compute scheduling | metadata parsing deterministic and auditable | malformed header adversarial tests |
| fnp-conformance | NumPy differential harness | production API | oracle compare policy explicit by dtype/op | differential comparator tests |
| frankennumpy | integration + policy/runtime modes | kernel internals | strict/hardened mode and evidence output always wired | startup policy tests |

## 15. Conformance Matrix

| Family | Oracle Workload | Pass Criterion | Drift Severity |
|---|---|---|---|
| Shape and reshape | dimensionality fixture corpus | exact shape parity | critical |
| Strides and views | slicing/view transformation suite | alias + stride parity | critical |
| Broadcasting | mixed rank elementwise suite | result-shape + value parity | critical |
| Ufunc arithmetic | core ufunc corpus | value parity under dtype policy | high |
| Reductions | axis and keepdims fixtures | shape and value parity | high |
| Dtype promotion | pairwise mixed-dtype suite | exact promotion parity against legacy matrix for shipped ops; explicit debt entries for remaining ops | critical |
| NPY/NPZ IO | round-trip + malformed cases | data + metadata parity | high |
| E2E pipeline | io -> broadcast -> reduction chain | reproducible parity report | critical |

## 16. Security and Compatibility Threat Matrix

| Threat | Strict Mode Response | Hardened Mode Response | Required Artifact |
|---|---|---|---|
| Malformed npy/npz metadata | fail-closed | fail-closed with bounded diagnostics | parser incident ledger |
| Shape bomb / extreme dimensions | execute if in limit | enforce stricter admission caps | admission guard report |
| Unsafe cast/coercion attempt | reject unsupported cast | allow only allowlisted policy casts | cast decision ledger |
| Stride alias confusion | fail invalid layout transitions | fail invalid layout transitions | stride invariant report |
| Unknown incompatible metadata version | fail-closed | fail-closed | compatibility drift report |
| Differential mismatch | release gate failure | release gate failure | parity failure bundle |
| Corrupt benchmark/conformance artifact | reject | RaptorQ recover then verify | decode proof + scrub report |
| Policy override abuse | explicit audited override only | explicit audited override only | override audit log |

## 17. Performance Budgets and SLO Targets

| Path | Workload | Budget |
|---|---|---|
| broadcast add/mul | 100M elements mixed rank | p95 <= 180 ms |
| reduction sum/mean | 100M elements axis reductions | p95 <= 210 ms |
| reshape/view operations | 10k complex transforms | p95 <= 40 ms |
| dtype conversion | 100M element conversion | throughput >= 1.2 GB/s |
| npy parse + load | 1 GB npy | throughput >= 400 MB/s |
| memory footprint | representative pipeline | peak RSS regression <= +8% |
| allocation churn | ufunc-heavy trace | alloc count regression <= +10% |
| tail stability | all benchmark families | p99 regression <= +7% |

## 18. CI Gate Topology (Release-Critical)

| Gate | Name | Blocking | Output Artifact |
|---|---|---|---|
| G1 | format + lint | yes | lint report |
| G2 | unit + integration | yes | junit report |
| G3 | differential conformance | yes | parity report JSON |
| G4 | adversarial + property tests | yes | counterexample corpus |
| G5 | benchmark regression | yes | baseline delta report |
| G6 | RaptorQ scrub + recovery drill | yes | scrub report + decode proof |

## 19. RaptorQ Artifact Envelope (Project-Wide)

All long-lived artifacts use sidecar durability manifests and decode proofs.

~~~json
{
  "artifact_id": "string",
  "artifact_type": "conformance|benchmark|ledger|manifest",
  "source_hash": "blake3:...",
  "raptorq": {
    "k": 0,
    "repair_symbols": 0,
    "overhead_ratio": 0.0
  },
  "scrub": {
    "status": "ok|recovered|failed",
    "last_ok_unix_ms": 0
  },
  "decode_proofs": [
    {
      "ts_unix_ms": 0,
      "reason": "...",
      "proof_hash": "blake3:..."
    }
  ]
}
~~~

## 20. 90-Day Execution Plan

Weeks 1-2:
- workspace skeleton and dtype/promotion table lock

Weeks 3-5:
- ndarray shape/stride core + strict conformance harness

Weeks 6-8:
- ufunc + broadcast + reductions with first performance baselines

Weeks 9-10:
- parser hardening and adversarial corpora
- strict/hardened runtime policy split with audits

Weeks 11-12:
- full gate topology G1-G6 wired and release-candidate drill

## 21. Porting Artifact Index

This spec is paired with the following methodology artifacts:

1. PLAN_TO_PORT_NUMPY_TO_RUST.md
2. EXISTING_NUMPY_STRUCTURE.md
3. PROPOSED_ARCHITECTURE.md
4. FEATURE_PARITY.md

Rule of use:

- Extraction and behavior understanding happens in EXISTING_NUMPY_STRUCTURE.md.
- Phase sequencing and parity-debt governance live in PLAN_TO_PORT_NUMPY_TO_RUST.md.
- Rust crate boundaries live in PROPOSED_ARCHITECTURE.md.
- Delivery readiness is tracked in FEATURE_PARITY.md.

## 22. FrankenSQLite Exemplar Alignment (Normative)

The copied exemplar `COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md` is normative for methodology quality and artifact rigor.

FrankenNumPy adapts the same style in these mappings:

1. Mode-split discipline:
   - FrankenSQLite `compatibility/native` pattern becomes FrankenNumPy `strict/hardened`.
2. Evidence-first operation:
   - decision events are ledgered for policy-sensitive operations.
3. Gate topology:
   - release readiness is artifact-driven (G1-G6), not narrative-driven.
4. Durability posture:
   - conformance/benchmark/ledger artifacts require sidecar durability contracts.
5. Differential validation:
   - legacy oracle comparison is a mandatory release gate.

## 23. asupersync and frankentui Leverage Plan

FrankenNumPy must leverage both crates as first-class integrations.

### 23.1 asupersync integration responsibilities

- cancellation-safe orchestration for long-running conformance and benchmark jobs
- structured telemetry channel for evidence ledger streaming
- durable pipeline supervision for artifact generation and scrub runs

### 23.2 frankentui integration responsibilities

- live terminal dashboards for parity drift and perf regressions
- interactive strict/hardened divergence inspection views
- operator-facing recovery/audit drill interfaces

### 23.3 enforcement

A release candidate is incomplete if either integration surface is absent or stub-only for all critical workflows.
