# FNP-P2C-008 Contract Table

packet_id: `FNP-P2C-008`  
subsystem: `linalg bridge first wave`

## strict_mode_policy

Strict mode preserves legacy-observable linalg API behavior classes for shape legality, solver/factorization/spectral outputs, tolerance handling, and `LinAlgError` failure taxonomy across the scoped packet surface.

## hardened_mode_policy

Hardened mode preserves the same public linalg outcomes while adding bounded validation and deterministic audit semantics; unknown or incompatible semantics remain fail-closed.

## excluded_scope

- Full parity for every backend-specific floating-point warning text is deferred to packet-E/F closure.
- Exact warning/error message text parity is deferred; class/family parity is required now.
- APIs outside packet `FNP-P2C-008` are excluded from this table.

## performance_sentinels

- solve/inv throughput and failure-path overhead for singular and near-singular systems.
- decomposition (`qr`, `svd`, `cholesky`) mode-branch overhead and shape-sensitive tail latency.
- spectral (`eig`, `eigh`, `eigvals`, `eigvalsh`) non-convergence handling overhead.
- batched linalg shape-normalization and backend bridge call overhead.

## Machine-Checkable Contract Rows

| contract_id | preconditions | shape_stride_contract | dtype_cast_contract | memory_alias_contract | strict_mode_policy | hardened_mode_policy | postconditions | error_contract | failure_reason_code | unit_property_ids | differential_ids | e2e_ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `P2C008-R01` | linalg entrypoint receives array-like input with requested operation/mode | operation-specific dimensionality and square-ness legality checks are deterministic | dtype admission/rejection follows deterministic linalg surface rules | validation path cannot mutate caller buffers | invalid ndim/shape class rejects with legacy-compatible error family | same class; hostile metadata fails closed | accepted inputs proceed with deterministic operation route | reject incompatible dimensionality/shape classes | `linalg_shape_contract_violation` | `UP-008-shape-preconditions` | `DF-008-shape-preconditions-oracle` | `E2E-008-linalg-golden-journey` |
| `P2C008-R02` | `solve`/`inv` called with candidate matrix/system pair | solve/inv shape checks are deterministic for fixed inputs | promotion/cast path is deterministic for supported numeric dtypes | solve/inv path cannot silently bypass alias/writeability safeguards | singular/incompatible systems fail with legacy class behavior | same class with bounded diagnostics only | successful solve/inv outputs remain deterministic class for fixed inputs | reject singular/incompatible systems with stable failure class | `linalg_solver_singularity` | `UP-008-solve-inv-contract` | `DF-008-solve-inv-oracle` | `E2E-008-solver-failure-replay` |
| `P2C008-R03` | `cholesky` requested with declared triangle mode and numeric matrix | cholesky requires valid square positive-definite class contract | dtype lane preserves deterministic admissibility and output class | factorization path cannot partially mutate visible outputs on failure | non-PD and shape-invalid failures map to legacy class family | same class; no permissive fallback for incompatible inputs | successful factorization respects deterministic triangle/output class | reject invalid or non-positive-definite inputs | `linalg_cholesky_contract_violation` | `UP-008-cholesky-modes` | `DF-008-cholesky-oracle` | `E2E-008-factorization-replay` |
| `P2C008-R04` | `qr` requested with supported mode (`reduced`, `complete`, `r`, `raw`) | mode-specific output shape family is deterministic | mode/output dtype lane remains deterministic and class-stable | mode switch must not bypass alias/writeability constraints | mode accept/reject and output family behavior matches legacy class | same behavior with bounded mode validation | deterministic output tuple class for fixed mode/input shape | reject unsupported mode or shape-class combinations | `linalg_qr_mode_invalid` | `UP-008-qr-mode-shapes` | `DF-008-qr-mode-oracle` | `E2E-008-factorization-replay` |
| `P2C008-R05` | `svd`/`svdvals` requested with mode flags and finite inputs | mode-specific output shape contract is deterministic | dtype/promotion lane remains deterministic for supported numeric classes | decomposition failure path cannot commit partial invalid writes | convergence and invalid-shape failure classes match legacy family | same classes with deterministic reason-code emission | successful decomposition yields deterministic tuple/value class | reject non-convergence and invalid mode/shape classes | `linalg_svd_nonconvergence` | `UP-008-svd-modes` | `DF-008-svd-oracle` | `E2E-008-spectral-factorization-replay` |
| `P2C008-R06` | spectral op (`eig`, `eigvals`, `eigh`, `eigvalsh`) requested with optional `UPLO` | output value/vector shape families are deterministic for fixed input class | dtype lane and hermitian branch handling are deterministic | spectral failure path cannot bypass alias and validation rules | convergence and hermitian-branch behavior classes match legacy | same classes with bounded branch validation | deterministic spectral output/error class for fixed inputs | reject non-convergence or invalid hermitian branch class | `linalg_spectral_convergence_failed` | `UP-008-spectral-branches` | `DF-008-spectral-oracle` | `E2E-008-spectral-factorization-replay` |
| `P2C008-R07` | `lstsq` requested with matrix/rhs pair and rcond/tolerance parameters | tuple output shape contract (`x`, `residuals`, `rank`, `s`) is deterministic | tolerance/rcond lane yields deterministic class outcome | least-squares path cannot mutate source operands | acceptance/failure classes for rank-deficient and edge shapes match legacy | same behavior with bounded tolerance guards | deterministic tuple field classes for fixed input/tolerance | reject invalid tolerance payloads and unsupported shape classes | `linalg_lstsq_tuple_contract_violation` | `UP-008-lstsq-tuple-contract` | `DF-008-lstsq-oracle` | `E2E-008-lstsq-replay` |
| `P2C008-R08` | norm/determinant/rank/pinv family invoked on valid numeric arrays | shape and axis legality checks are deterministic | cast/promotion/tolerance path remains deterministic per operation family | read-only and alias constraints are preserved throughout evaluation | success/failure classes for tolerance-edge cases match legacy family | same classes with bounded tolerance validation | deterministic result class for fixed inputs/flags | reject unsupported axis/order/tolerance metadata classes | `linalg_norm_det_rank_policy_violation` | `UP-008-norm-det-rank-pinv` | `DF-008-norm-det-rank-oracle` | `E2E-008-norm-det-rank-replay` |
| `P2C008-R09` | backend bridge or lapack-lite adapter boundary is exercised | backend-call shape contract remains deterministic and prevalidated | backend dtype expectations must match validated operation dtype lane | adapter bridge cannot leak partial state or alias-unsafe transitions | backend error-hook/failure-class mapping matches legacy-observable family | same mapping; unsupported backend states fail closed | backend outcomes remain deterministic and replay-auditable | reject malformed backend parameter states and unsupported backend class | `linalg_backend_bridge_invalid` | `UP-008-backend-error-hook` | `DF-008-backend-bridge-oracle` | `E2E-008-backend-hook-replay` |
| `P2C008-R10` | runtime policy mediation receives mode/class metadata at linalg boundary | unknown shape/operation metadata classes are non-admissible | unknown dtype/tolerance policy classes are non-admissible | overrides cannot bypass incompatible-class policy gates | unknown/incompatible metadata fails closed | allowlisted compatible overrides may be audited in hardened mode only | decision action/reason-code remains deterministic and replay-complete | reject unknown metadata and incompatible policy classes | `linalg_policy_unknown_metadata` | `UP-008-policy-fail-closed` | `DF-008-policy-adversarial-oracle` | `E2E-008-policy-replay` |

## Logging and Failure Semantics

All packet linalg validations must emit:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

Reason-code vocabulary for this packet:

- `linalg_shape_contract_violation`
- `linalg_solver_singularity`
- `linalg_cholesky_contract_violation`
- `linalg_qr_mode_invalid`
- `linalg_svd_nonconvergence`
- `linalg_spectral_convergence_failed`
- `linalg_lstsq_tuple_contract_violation`
- `linalg_norm_det_rank_policy_violation`
- `linalg_backend_bridge_invalid`
- `linalg_policy_unknown_metadata`

## Budgeted Mode and Expected-Loss Notes

- Budgeted mode: hardened linalg execution enforces explicit caps on decomposition retries, tolerance-search depth, backend revalidation attempts, and adversarial input normalization with deterministic exhaustion behavior.
- Expected-loss model: solver/decomposition/policy mediation records state/action/loss rationale for replayable audits.
- Calibration trigger: if strict/hardened linalg failure-class drift exceeds packet threshold, fallback to conservative deterministic behavior (`full_validate` or `fail_closed`) and emit audited reason codes.

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Rollback Handle

If linalg contract drift is detected, revert `artifacts/phase2c/FNP-P2C-008/contract_table.md` and restore the last green packet boundary contract baseline.
