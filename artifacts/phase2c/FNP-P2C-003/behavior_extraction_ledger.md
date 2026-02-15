# FNP-P2C-003 Behavior Extraction Ledger

Packet: `FNP-P2C-003-A`  
Subsystem: `strided transfer semantics`

## 1. Observable Contract Ledger

| Contract ID | Observable behavior | Strict mode expectation | Hardened mode expectation | Legacy anchors |
|---|---|---|---|---|
| `P2C003-C01` | Transfer-loop selection is deterministic for identical `(src_dtype, dst_dtype, aligned, src_stride, dst_stride, move_refs)` inputs. | same loop class/function selected for identical transfer context | same loop class; unknown metadata/unsupported combinations fail-closed | `dtype_transfer.c:3089`, `dtype_transfer.c:2746` |
| `P2C003-C02` | Overlap-sensitive array assignment preserves result equivalence by direction adjustment or copy mediation when required. | read/write overlap never silently corrupts output | same outward result with explicit overlap-risk audit tagging | `array_assign_array.c:124`, `test_mem_overlap.py:67`, `test_mem_overlap.py:600` |
| `P2C003-C03` | Where-masked transfer writes only where mask is true and preserves non-selected output cells. | deterministic mask-gated writes with stable error classes | same observable mask semantics; malformed mask metadata fails-closed | `array_assign_array.c:190`, `test_mem_overlap.py:869` |
| `P2C003-C04` | Scalar and array assignment cast pathways honor same-value casting contracts for lossy vs non-lossy conversions. | same-value mode rejects lossy conversions with stable failure class | same class with bounded diagnostics and deterministic reason code | `array_assign_scalar.c:70`, `test_casting_unittests.py:836`, `test_casting_unittests.py:889` |
| `P2C003-C05` | String/unicode transfer specialization follows zero-pad/truncate/copyswap rules without shape/stride ambiguity. | deterministic transfer variant selection and output bytes | same semantics; malformed transfer descriptors fail-closed | `dtype_transfer.c:460`, `dtype_transfer.c:372`, `dtype_transfer.c:400`, `dtype_transfer.c:424` |
| `P2C003-C06` | Subarray and grouped transfer paths (`1->1`, `n->n`, subarray broadcast) preserve deterministic stride-based data movement classes. | grouped transfer families keep stable class behavior | same behavior with bounded hardened validation | `dtype_transfer.c:1597`, `dtype_transfer.c:1608`, `dtype_transfer.c:1798` |
| `P2C003-C07` | Flatiter slice/fancy/bool indexing and assignment copy semantics route through transfer loops with stable errors. | flatiter transfer behavior is class-stable for supported index kinds | same outcomes; malformed indices fail-closed with deterministic reason codes | `iterators.c:574`, `iterators.c:854`, `test_regression.py:2401` |
| `P2C003-C08` | NDIter overlap handling (`copy_if_overlap`, `overlap_assume_elementwise`) preserves documented copy/no-copy outcomes. | overlap-copy policy matches legacy classes | same outcomes with overlap-policy decision logging | `test_nditer.py:1325`, `test_nditer.py:2603` |
| `P2C003-C09` | Transfer/cast paths preserve floating-point error reporting semantics for cast operations. | FPE status behavior remains class-stable | same error class with deterministic logging context | `array_assign_array.c:146`, `array_assign_scalar.c:92`, `test_regression.py:1673` |

## 2. Compatibility Invariants

1. Transfer determinism invariant: identical transfer context resolves to the same transfer-loop behavior class.
2. Overlap safety invariant: read/write overlap never yields silent data corruption; copy/ordering controls remain deterministic.
3. Mask isolation invariant: where-mask false positions remain unchanged after masked transfer operations.
4. Same-value cast invariant: explicit same-value mode rejects lossy transforms with stable failure classes.
5. Flatiter transfer invariant: indexing/assignment transfer pathways preserve stable error taxonomy and value movement classes.
6. Strict/hardened parity invariant: hardened mode may add bounded validation and audit details but cannot change public success/failure class for covered contracts.

## 3. Undefined or Under-Specified Edges (Tagged)

| Unknown ID | Description | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `P2C003-U01` | `fnp-iter` has no transfer-loop selector/state machine; transfer semantics currently live indirectly in ad-hoc ufunc traversal code. | high | `bd-23m.14.4` | land module boundary skeleton with explicit transfer selector interfaces and state transition model |
| `P2C003-U02` | Overlap and where-mask transfer invariants lack packet-scoped unit/property coverage in Rust. | high | `bd-23m.14.5` | packet-E suite covers overlap/where/same-value invariant families with structured logs |
| `P2C003-U03` | Differential/adversarial oracle corpus for transfer families (grouped/subarray/flatiter transfer) is not yet packet-scoped. | high | `bd-23m.14.6` | packet-F fixtures and runner additions cover transfer class matrix and adversarial cases |
| `P2C003-U04` | Workflow replay does not yet include packet transfer journeys with reason-code-linked artifacts. | medium | `bd-23m.14.7` | packet-G scenarios emit replay-complete structured fields and transfer fixture links |
| `P2C003-U05` | String/unicode specialized transfer parity (zero-pad/truncate/copyswap) is not yet represented in Rust behavior suites. | medium | `bd-23m.14.5` + `bd-23m.14.6` | add packet-E/F test matrix for fixed-width string transfer classes |
| `P2C003-U06` | Exact mapping from legacy same-value cast context to Rust runtime policy reason-code taxonomy is unpinned. | medium | `bd-23m.14.2` + `bd-23m.14.3` | contract rows + threat model explicitly define reason-code mapping and fail-closed policy |

## 4. Planned Verification Hooks

| Verification lane | Planned hook | Artifact target |
|---|---|---|
| Unit/property | transfer invariants for overlap direction, mask isolation, same-value rejection, grouped/subarray movement classes | packet-E test modules in `fnp-iter`/`fnp-ufunc` with structured logs |
| Differential/metamorphic/adversarial | transfer fixture corpus for overlap/where/cast/flatiter paths vs legacy oracle | packet-F fixtures + runner extensions in `crates/fnp-conformance` |
| E2E | strict/hardened transfer journey scenarios with replay/forensics links | packet-G workflow scenarios + `artifacts/logs/` outputs |
| Structured logging | enforce required fields on packet transfer test outputs (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`) | `artifacts/contracts/TESTING_AND_LOGGING_CONVENTIONS_V1.md` + packet-E/F/G logs |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: transfer-policy mediation must record state/action/loss rationale and deterministic fallback path.
- Optimization gate: transfer performance changes require baseline/profile + one-lever + behavior-isomorphism proof.
- EV gate: optimization/policy levers advance only when `EV >= 2.0`; otherwise remain explicit deferred parity debt.
- RaptorQ scope: packet-I durability bundle must include sidecar/scrub/decode-proof linkage for packet transfer artifacts.

## 6. Rollback Handle

If packet-local transfer behavior extraction drifts or is contradicted, revert `artifacts/phase2c/FNP-P2C-003/*` to the prior green baseline and re-run packet validation before reapplying changes.
