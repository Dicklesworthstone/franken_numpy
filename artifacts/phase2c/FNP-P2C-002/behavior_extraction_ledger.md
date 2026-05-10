# FNP-P2C-002 Behavior Extraction Ledger

Packet: `FNP-P2C-002-A`  
Subsystem: `Dtype descriptors and promotion`

## 1. Observable Contract Ledger

| Contract ID | Observable behavior | Strict mode expectation | Hardened mode expectation | Legacy anchors |
|---|---|---|---|---|
| `P2C002-C01` | Python type/descriptor inputs normalize to canonical dtype descriptors | deterministic descriptor normalization for scoped inputs | same outward normalization with malformed metadata fail-closed | `descriptor.c:1550`, `descriptor.c:1626` |
| `P2C002-C02` | Promotion outcomes are deterministic for identical dtype pairs | deterministic promotion result for each scoped pair | same result; unknown pairs fail-closed | `can_cast_table.h:66`, `test_dtype.py:1462` |
| `P2C002-C03` | Lossless cast policy is class-stable and matrix-driven | safe cast decisions follow deterministic matrix rules | same decisions with bounded guardrails for malformed metadata | `can_cast_table.h:35`, `can_cast_table.h:66`, `test_dtype.py:214` |
| `P2C002-C04` | Incompatible promotion pairings produce stable failure class | reject pairs with no common dtype | same failure class, plus deterministic audit context | `test_dtype.py:1420` |
| `P2C002-C05` | Python integer/scalar promotion respects default-int and scoped precedence semantics | stable default-int behavior for scoped scalar cases | same behavior with fail-closed handling for unknown scalar metadata | `test_dtype.py:1430` |
| `P2C002-C06` | Cast failure semantics under overflow/invalid conversions remain class-stable | cast failures retain deterministic class family | same failure class + bounded diagnostics | `test_casting_floatingpoint_errors.py:143` |

## 2. Compatibility Invariants

1. Promotion determinism invariant: same ordered/scoped dtype inputs always yield the same promoted dtype.
2. Promotion symmetry invariant (for currently scoped matrix): `promote(lhs, rhs) == promote(rhs, lhs)`.
3. Lossless cast monotonicity invariant: if cast is marked lossless in matrix, runtime cast gate cannot reject it in strict mode.
4. Error-class stability invariant: incompatible promotion and cast-failure classes remain stable across strict/hardened modes.
5. Descriptor normalization invariant: canonical parse output for supported dtype identifiers is unique and deterministic.

## 3. Resolved Edges and Residual Breadth (Tagged)

| Edge ID | Current status | Risk | Evidence owner | Residual criteria |
|---|---|---|---|---|
| `P2C002-U01` | Current packet matrix has unit/property and differential coverage for scoped promotion/cast lanes; full legacy cast-table parity remains residual. | high | packet-E/F evidence | expand descriptor/cast matrix coverage toward the full legacy table while preserving fail-closed unknown pairs |
| `P2C002-U02` | Structured dtype cast semantics have current packet coverage but not full legacy alias-sensitive field breadth. | high | unit/property + risk note | extend structured dtype oracle fixtures and reason-coded logs for alias-sensitive field behavior |
| `P2C002-U03` | Differential/metamorphic/adversarial coverage exists for current incompatible-pair and scalar-promotion edge classes. | medium | `differential_metamorphic_adversarial_evidence.json` | deepen exotic dtype, weak-scalar, and scalar-metadata interaction matrices |
| `P2C002-U04` | Descriptor metadata abuse is covered by packet-specific fail-closed threat controls and executable security/test gates. | medium | risk note + final evidence pack | recalibrate hardened metadata normalization and cast-policy budgets against broader adversarial corpora |

## 4. Verification Hooks

| Verification lane | Hook | Artifact target |
|---|---|---|
| Unit/property | `fnp-dtype` promotion/cast properties across scoped matrix and failure taxonomy | `artifacts/phase2c/FNP-P2C-002/unit_property_evidence.json`, `crates/fnp-dtype/src/lib.rs`, `crates/fnp-conformance/fixtures/dtype_promotion_cases.json` |
| Differential/metamorphic/adversarial | dtype oracle cases for incompatible/common-type resolution, scalar-promotion edges, and adversarial cast-failure categories | `artifacts/phase2c/FNP-P2C-002/differential_metamorphic_adversarial_evidence.json`, `crates/fnp-conformance/fixtures/packet002_dtype/*` |
| E2E/workflow | mixed-dtype pipeline replay scenarios with strict/hardened audit trail | `artifacts/phase2c/FNP-P2C-002/e2e_replay_forensics_evidence.json`, `scripts/e2e/run_dtype_contract_journey.sh`, workflow scenario artifacts |
| Structured logging | `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code` coverage in dtype tests and workflow gates | `artifacts/contracts/test_logging_contract_v1.json`, packet-E/F/G evidence, test-contract gate outputs |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: any policy mediation at descriptor/cast boundaries must record state/action/loss rationale.
- Optimization gate: promotion/cast optimizations require baseline/profile + one-lever + isomorphism proof.
- EV gate: optimization levers advance only when `EV >= 2.0`; otherwise remain deferred debt.
- RaptorQ scope: packet-I parity bundles include sidecar/scrub/decode-proof linkage.

## 6. Rollback Handle

If packet-local dtype contract changes introduce compatibility drift, revert `artifacts/phase2c/FNP-P2C-002/*` and restore the last green promotion/cast differential baseline.
