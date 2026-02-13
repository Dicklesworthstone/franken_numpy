# EVIDENCE_LEDGER_CONTRACT

The runtime evidence ledger records policy-sensitive decisions with these fields.

## Core Fields

1. `timestamp` (unix ms)
2. `runtime_mode` (`strict` or `hardened`)
3. `compatibility_class` (`known_compatible`, `known_incompatible`, `unknown`)
4. `risk_score`
5. `action` (`allow`, `full_validate`, `fail_closed`)
6. `note` (context tag)

## Alien-Artifact Decision Fields (Round 2)

7. `posterior_incompatible` (Bayesian posterior of incompatible state)
8. `expected_loss_allow`
9. `expected_loss_full_validate`
10. `expected_loss_fail_closed`
11. `selected_expected_loss`
12. `evidence_terms[]` where each term contains:
   - `name`
   - `log_likelihood_ratio`

## Decision Core

Posterior model (operational form):

- `posterior_log_odds = prior_log_odds(class) + (logit(risk_score) - logit(validation_threshold))`
- `posterior_incompatible = sigmoid(posterior_log_odds)`

Expected-loss model:

- `E[L(action)] = P(compatible) * L(action, compatible) + P(incompatible) * L(action, incompatible)`

Loss matrix is encoded in `DecisionLossModel`.

## Contract Guarantees

- Policy action selection remains compatibility-safe and fail-closed.
- All recorded decision metrics are finite and auditable.
- Ledger events are deterministic for fixed inputs.
- Unknown/incompatible compatibility classes remain non-bypassable in both modes.
