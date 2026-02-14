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

## 3. Undefined or Under-Specified Edges (Tagged)

| Unknown ID | Description | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `P2C002-U01` | Full legacy cast-table parity is not yet represented (current Rust matrix is intentionally scoped subset). | high | `bd-23m.13.4` | Expand descriptor/cast matrix implementation plan with explicit legacy-matrix closure path. |
| `P2C002-U02` | Structured dtype cast semantics and alias-sensitive field behavior are only partially represented in scoped fixtures. | high | `bd-23m.13.5` | Unit/property suites include structured dtype cast invariants with reason-coded logs. |
| `P2C002-U03` | Differential oracle coverage for edge promotion failures and scalar-metadata interactions is incomplete. | medium | `bd-23m.13.6` | Differential/metamorphic fixtures cover incompatible-pair and scalar-promotion edge classes. |
| `P2C002-U04` | Hardened policy boundaries for descriptor metadata abuse need packet-specific threat controls. | high | `bd-23m.13.3` | Threat model locks fail-closed and bounded-recovery controls with executable hooks. |

## 4. Planned Verification Hooks

| Verification lane | Planned hook | Artifact target |
|---|---|---|
| Unit/property | Extend `fnp-dtype` promotion/cast properties across expanded matrix and failure taxonomy | `crates/fnp-dtype/src/lib.rs` + packet-E additions |
| Differential/metamorphic/adversarial | Add dtype oracle cases for incompatible/common-type resolution and cast-failure categories | `crates/fnp-conformance/fixtures/dtype_promotion_cases.json` + packet-F expansions |
| E2E | Add mixed-dtype pipeline replay scenario with strict/hardened audit trail | packet-G scenario scripts + `artifacts/logs/` |
| Structured logging | enforce `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code` | `artifacts/contracts/test_logging_contract_v1.json`, packet-E/G hooks |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: any policy mediation at descriptor/cast boundaries must record state/action/loss rationale.
- Optimization gate: promotion/cast optimizations require baseline/profile + one-lever + isomorphism proof.
- EV gate: optimization levers advance only when `EV >= 2.0`; otherwise remain deferred debt.
- RaptorQ scope: packet-I parity bundles require sidecar/scrub/decode-proof linkage.

## 6. Rollback Handle

If packet-local dtype contract changes introduce compatibility drift, revert `artifacts/phase2c/FNP-P2C-002/*` and restore the last green promotion/cast differential baseline.
