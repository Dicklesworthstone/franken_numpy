# ALIEN_GRAVEYARD_RECOMMENDATION_CARDS_ROUND2

## Card 1

Change:
- Add Bayesian posterior + expected-loss terms to runtime decision evidence events.

Hotspot evidence:
- Runtime policy decisions are correctness-critical and require explainability; current code previously logged only action + scalar risk.

Mapped graveyard sections:
- `alien_cs_graveyard.md` §0.4 (decision layer), §0.19 (evidence ledger)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.1, §0.12, §0.19

EV score (Impact * Confidence * Reuse / Effort * Friction):
- (4 * 4 * 5) / (2 * 2) = 20.0

Priority tier (S/A/B/C):
- A

Adoption wedge (boundary/compatibility/rollout):
- Extend `DecisionEvent` schema only; keep `decide_compatibility` action behavior unchanged.

Budgeted mode (default budget + on-exhaustion behavior):
- Constant-time math per decision; no unbounded loops; fallback to fail-closed policy behavior.

Expected-loss model (states/actions/loss):
- States: `{compatible, incompatible}`
- Actions: `{allow, full_validate, fail_closed}`
- Loss matrix encoded in `DecisionLossModel`.

Calibration + fallback trigger:
- Posterior derived from class prior + risk-vs-threshold log-odds margin.
- If numeric inputs are invalid/NaN, clamp to conservative finite bounds.

Isomorphism proof plan:
- Preserve selected action outputs under existing fixture corpus; verify with runtime-policy suite + conformance tests.

p50/p95/p99 before/after target:
- Neutral for throughput path; measure no regression in `generate_benchmark_baseline` command.

Primary failure risk + countermeasure:
- Risk: schema drift in evidence readers.
- Countermeasure: keep action semantics unchanged and include explicit contract doc updates.

Repro artifact pack (env/manifest/repro.lock/legal/provenance):
- `artifacts/optimization/hyperfine_generate_benchmark_baseline_before.json`
- `artifacts/optimization/hyperfine_generate_benchmark_baseline_after.json`
- `artifacts/proofs/golden_checksums_round2.txt`

Primary paper status (hypothesis/read/reproduced + checklist state):
- Read/hypothesis level for decision-theoretic runtime control (no external paper claim promoted to reproduced here).

Interference test status (required when composing controllers):
- Not composed with another adaptive controller in this round.

Demo linkage (`demo_id` + `claim_id`, if production-facing):
- `demo_id: runtime-policy-evidence-round2`, `claim_id: fnp-runtime-expected-loss-ledger-v1`

Rollback:
- Revert `crates/fnp-runtime/src/lib.rs` changes.

Baseline comparator (what are we beating?):
- Prior minimal ledger schema (action-only event records).

## Card 2

Change:
- Optimize broadcast kernel index mapping with incremental cursor (single-lever optimization).

Hotspot evidence:
- Highest-score item in opportunity matrix; benchmark command contains repeated broadcast kernels.

Mapped graveyard sections:
- `alien_cs_graveyard.md` §0.1, §0.2, §0.3

EV score (Impact * Confidence * Reuse / Effort * Friction):
- (5 * 4 * 4) / (2 * 2) = 20.0

Priority tier (S/A/B/C):
- A

Adoption wedge (boundary/compatibility/rollout):
- Internal kernel implementation only; no public API changes.

Budgeted mode (default budget + on-exhaustion behavior):
- Bounded memory O(ndim), linear iteration O(nelems), deterministic traversal.

Expected-loss model (states/actions/loss):
- Not adaptive; direct deterministic kernel optimization.

Calibration + fallback trigger:
- If mismatch is detected in differential suites, revert immediately.

Isomorphism proof plan:
- Fixture checksum validation + full unit/conformance runs.

p50/p95/p99 before/after target:
- Target: reduce mean and tail of benchmark-generator command.

Primary failure risk + countermeasure:
- Risk: index rollover bug.
- Countermeasure: retain existing broadcast semantics and validate through differential suite.

Repro artifact pack (env/manifest/repro.lock/legal/provenance):
- Same artifact set as Card 1 plus `artifacts/optimization/ROUND2_OPPORTUNITY_MATRIX.md`.

Primary paper status (hypothesis/read/reproduced + checklist state):
- Reproduced internally at optimization-pattern level.

Interference test status (required when composing controllers):
- N/A.

Demo linkage (`demo_id` + `claim_id`, if production-facing):
- `demo_id: ufunc-broadcast-hotpath-round2`, `claim_id: fnp-ufunc-broadcast-cursor-v1`

Rollback:
- Revert `crates/fnp-ufunc/src/lib.rs`.

Baseline comparator (what are we beating?):
- Prior per-element unravel + broadcast-source recomputation path.

## Card 3

Change:
- Plan and bead-level uplift: enforce recommendation-contract fields and fallback-safe optimization workflow on open backlog.

Hotspot evidence:
- Backlog quality is the execution bottleneck for high-assurance shipping.

Mapped graveyard sections:
- `alien_cs_graveyard.md` §0.16, §0.19, §16
- `high_level_summary...` Project-Level Decision Contracts + Artifact Graph sections

EV score (Impact * Confidence * Reuse / Effort * Friction):
- (4 * 4 * 5) / (3 * 2) = 13.3

Priority tier (S/A/B/C):
- A

Adoption wedge (boundary/compatibility/rollout):
- Use `br` updates to enrich bead contracts without removing functionality.

Budgeted mode (default budget + on-exhaustion behavior):
- If contract details unknown, mark as explicit `TBD` with owner and blocking dependency.

Expected-loss model (states/actions/loss):
- States: `{under-specified, executable}`
- Action: `{refine now, defer}`
- Loss emphasizes avoiding under-specified implementation starts.

Calibration + fallback trigger:
- `br lint == 0`, `br dep cycles == []`, and `bv --robot-alerts` non-critical before implementation.

Isomorphism proof plan:
- Ensure no feature/functionality removals in bead descriptions while adding contracts.

p50/p95/p99 before/after target:
- N/A for runtime; target planning throughput and reduced blocked churn.

Primary failure risk + countermeasure:
- Risk: over-constraining graph.
- Countermeasure: accept only dependencies that preserve parallelism and measurable user value.

Repro artifact pack (env/manifest/repro.lock/legal/provenance):
- `.beads/issues.jsonl`, `bv --robot-*` snapshots.

Primary paper status (hypothesis/read/reproduced + checklist state):
- Methodological adoption; no paper-specific claim in this round.

Interference test status (required when composing controllers):
- N/A.

Demo linkage (`demo_id` + `claim_id`, if production-facing):
- `demo_id: bead-contract-uplift-round2`, `claim_id: franken-numpy-plan-space-alien-uplift-v1`

Rollback:
- Revert bead updates via `br` history restore or git revert of `.beads/issues.jsonl`/DB snapshot.

Baseline comparator (what are we beating?):
- Previous beads lacking explicit recommendation contracts and fallback-safe decision fields.
