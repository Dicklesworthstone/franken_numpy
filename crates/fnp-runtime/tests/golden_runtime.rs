//! Golden tests for fnp-runtime decision functions.
//!
//! These tests lock in expected behavior of the runtime decision system.
//! Any change to decision logic will cause these tests to fail, ensuring
//! deliberate review of behavior changes.

use fnp_runtime::{
    CompatibilityClass, DecisionAction, DecisionLossModel, RuntimeMode,
    decide_compatibility, decide_compatibility_from_wire, evaluate_policy_override,
    expected_loss_for_action, posterior_incompatibility,
};

// ─────────────────────────────────────────────────────────────────────────────
// RuntimeMode parsing golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_runtime_mode_from_wire_strict() {
    assert_eq!(RuntimeMode::from_wire("strict"), Some(RuntimeMode::Strict));
}

#[test]
fn golden_runtime_mode_from_wire_hardened() {
    assert_eq!(RuntimeMode::from_wire("hardened"), Some(RuntimeMode::Hardened));
}

#[test]
fn golden_runtime_mode_from_wire_invalid() {
    assert_eq!(RuntimeMode::from_wire("unknown"), None);
    assert_eq!(RuntimeMode::from_wire(""), None);
    assert_eq!(RuntimeMode::from_wire("STRICT"), None);
}

#[test]
fn golden_runtime_mode_as_str() {
    assert_eq!(RuntimeMode::Strict.as_str(), "strict");
    assert_eq!(RuntimeMode::Hardened.as_str(), "hardened");
}

// ─────────────────────────────────────────────────────────────────────────────
// CompatibilityClass parsing golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_compatibility_class_parse_known_compatible() {
    assert_eq!(
        CompatibilityClass::parse_wire("known_compatible"),
        Some(CompatibilityClass::KnownCompatible)
    );
    assert_eq!(
        CompatibilityClass::parse_wire("known_compatible_low_risk"),
        Some(CompatibilityClass::KnownCompatible)
    );
    assert_eq!(
        CompatibilityClass::parse_wire("known_compatible_high_risk"),
        Some(CompatibilityClass::KnownCompatible)
    );
}

#[test]
fn golden_compatibility_class_parse_known_incompatible() {
    assert_eq!(
        CompatibilityClass::parse_wire("known_incompatible"),
        Some(CompatibilityClass::KnownIncompatible)
    );
    assert_eq!(
        CompatibilityClass::parse_wire("known_incompatible_semantics"),
        Some(CompatibilityClass::KnownIncompatible)
    );
}

#[test]
fn golden_compatibility_class_parse_unknown() {
    assert_eq!(
        CompatibilityClass::parse_wire("unknown"),
        Some(CompatibilityClass::Unknown)
    );
    assert_eq!(
        CompatibilityClass::parse_wire("unknown_semantics"),
        Some(CompatibilityClass::Unknown)
    );
}

#[test]
fn golden_compatibility_class_parse_invalid() {
    assert_eq!(CompatibilityClass::parse_wire("invalid"), None);
    assert_eq!(CompatibilityClass::parse_wire(""), None);
}

#[test]
fn golden_compatibility_class_from_wire_fallback() {
    assert_eq!(
        CompatibilityClass::from_wire("invalid"),
        CompatibilityClass::Unknown
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// decide_compatibility golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_decide_strict_known_compatible_allows() {
    assert_eq!(
        decide_compatibility(
            RuntimeMode::Strict,
            CompatibilityClass::KnownCompatible,
            0.1,
            0.5
        ),
        DecisionAction::Allow
    );
}

#[test]
fn golden_decide_strict_known_compatible_high_risk_still_allows() {
    assert_eq!(
        decide_compatibility(
            RuntimeMode::Strict,
            CompatibilityClass::KnownCompatible,
            0.9,
            0.5
        ),
        DecisionAction::Allow
    );
}

#[test]
fn golden_decide_strict_known_incompatible_fails() {
    assert_eq!(
        decide_compatibility(
            RuntimeMode::Strict,
            CompatibilityClass::KnownIncompatible,
            0.1,
            0.5
        ),
        DecisionAction::FailClosed
    );
}

#[test]
fn golden_decide_strict_unknown_fails() {
    assert_eq!(
        decide_compatibility(RuntimeMode::Strict, CompatibilityClass::Unknown, 0.1, 0.5),
        DecisionAction::FailClosed
    );
}

#[test]
fn golden_decide_hardened_low_risk_allows() {
    assert_eq!(
        decide_compatibility(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            0.1,
            0.5
        ),
        DecisionAction::Allow
    );
}

#[test]
fn golden_decide_hardened_high_risk_validates() {
    assert_eq!(
        decide_compatibility(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            0.6,
            0.5
        ),
        DecisionAction::FullValidate
    );
}

#[test]
fn golden_decide_hardened_at_threshold_validates() {
    assert_eq!(
        decide_compatibility(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            0.5,
            0.5
        ),
        DecisionAction::FullValidate
    );
}

#[test]
fn golden_decide_hardened_below_threshold_allows() {
    assert_eq!(
        decide_compatibility(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            0.49,
            0.5
        ),
        DecisionAction::Allow
    );
}

#[test]
fn golden_decide_hardened_incompatible_fails() {
    assert_eq!(
        decide_compatibility(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownIncompatible,
            0.1,
            0.5
        ),
        DecisionAction::FailClosed
    );
}

#[test]
fn golden_decide_hardened_nan_risk_validates() {
    assert_eq!(
        decide_compatibility(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            f64::NAN,
            0.5
        ),
        DecisionAction::FullValidate
    );
}

#[test]
fn golden_decide_hardened_infinite_risk_validates() {
    assert_eq!(
        decide_compatibility(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            f64::INFINITY,
            0.5
        ),
        DecisionAction::FullValidate
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// decide_compatibility_from_wire golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_decide_from_wire_valid() {
    assert_eq!(
        decide_compatibility_from_wire("strict", "known_compatible", 0.1, 0.5),
        DecisionAction::Allow
    );
}

#[test]
fn golden_decide_from_wire_invalid_mode() {
    assert_eq!(
        decide_compatibility_from_wire("invalid", "known_compatible", 0.1, 0.5),
        DecisionAction::FailClosed
    );
}

#[test]
fn golden_decide_from_wire_invalid_class_treated_as_unknown() {
    assert_eq!(
        decide_compatibility_from_wire("strict", "invalid_class", 0.1, 0.5),
        DecisionAction::FailClosed
    );
}

#[test]
fn golden_decide_from_wire_whitespace_tolerant() {
    assert_eq!(
        decide_compatibility_from_wire("  strict  ", "  known_compatible  ", 0.1, 0.5),
        DecisionAction::Allow
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// posterior_incompatibility golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_posterior_known_compatible_low_risk() {
    let (posterior, terms) =
        posterior_incompatibility(CompatibilityClass::KnownCompatible, 0.1, 0.5);
    assert!(posterior < 0.01, "posterior should be very low: {posterior}");
    assert_eq!(terms.len(), 2);
    assert_eq!(terms[0].name, "prior_class_log_odds");
    assert_eq!(terms[1].name, "risk_vs_threshold_llr");
}

#[test]
fn golden_posterior_known_compatible_high_risk() {
    let (posterior, _) =
        posterior_incompatibility(CompatibilityClass::KnownCompatible, 0.9, 0.5);
    assert!(posterior > 0.01 && posterior < 0.5, "posterior should be moderate: {posterior}");
}

#[test]
fn golden_posterior_known_incompatible() {
    let (posterior, _) =
        posterior_incompatibility(CompatibilityClass::KnownIncompatible, 0.1, 0.5);
    assert!(posterior > 0.9, "posterior should be very high: {posterior}");
}

#[test]
fn golden_posterior_unknown_class() {
    let (posterior, _) = posterior_incompatibility(CompatibilityClass::Unknown, 0.5, 0.5);
    assert!(
        (posterior - 0.5).abs() < 0.01,
        "posterior should be ~0.5 for unknown: {posterior}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// expected_loss_for_action golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_expected_loss_allow_low_posterior() {
    let model = DecisionLossModel::default();
    let loss = expected_loss_for_action(DecisionAction::Allow, 0.01, model);
    assert!(loss < 2.0, "allow with low posterior should have low loss: {loss}");
}

#[test]
fn golden_expected_loss_allow_high_posterior() {
    let model = DecisionLossModel::default();
    let loss = expected_loss_for_action(DecisionAction::Allow, 0.99, model);
    assert!(loss > 90.0, "allow with high posterior should have high loss: {loss}");
}

#[test]
fn golden_expected_loss_fail_closed_low_posterior() {
    let model = DecisionLossModel::default();
    let loss = expected_loss_for_action(DecisionAction::FailClosed, 0.01, model);
    assert!(loss > 100.0, "fail_closed with low posterior should have high loss: {loss}");
}

#[test]
fn golden_expected_loss_fail_closed_high_posterior() {
    let model = DecisionLossModel::default();
    let loss = expected_loss_for_action(DecisionAction::FailClosed, 0.99, model);
    assert!(loss < 5.0, "fail_closed with high posterior should have low loss: {loss}");
}

#[test]
fn golden_expected_loss_full_validate_moderate() {
    let model = DecisionLossModel::default();
    let loss = expected_loss_for_action(DecisionAction::FullValidate, 0.5, model);
    assert!(loss > 2.0 && loss < 4.0, "full_validate at 0.5 should be moderate: {loss}");
}

// ─────────────────────────────────────────────────────────────────────────────
// evaluate_policy_override golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_override_strict_mode_never_approves() {
    let event = evaluate_policy_override(
        RuntimeMode::Strict,
        CompatibilityClass::KnownCompatible,
        "dtype_overflow",
        &["dtype_overflow"],
        "pkt-123",
        "user@test",
        "testing",
    );
    assert!(!event.approved);
    assert_eq!(event.action, DecisionAction::FailClosed);
}

#[test]
fn golden_override_hardened_with_allowlist_approves() {
    let event = evaluate_policy_override(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        "dtype_overflow",
        &["dtype_overflow", "precision_loss"],
        "pkt-123",
        "user@test",
        "testing",
    );
    assert!(event.approved);
    assert_eq!(event.action, DecisionAction::FullValidate);
}

#[test]
fn golden_override_hardened_not_in_allowlist() {
    let event = evaluate_policy_override(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        "dtype_overflow",
        &["precision_loss"],
        "pkt-123",
        "user@test",
        "testing",
    );
    assert!(!event.approved);
    assert_eq!(event.action, DecisionAction::FailClosed);
}

#[test]
fn golden_override_incompatible_class_never_approves() {
    let event = evaluate_policy_override(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownIncompatible,
        "dtype_overflow",
        &["dtype_overflow"],
        "pkt-123",
        "user@test",
        "testing",
    );
    assert!(!event.approved);
    assert_eq!(event.action, DecisionAction::FailClosed);
}

#[test]
fn golden_override_empty_deviation_never_approves() {
    let event = evaluate_policy_override(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        "",
        &["dtype_overflow"],
        "pkt-123",
        "user@test",
        "testing",
    );
    assert!(!event.approved);
}

#[test]
fn golden_override_audit_ref_format() {
    let event = evaluate_policy_override(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        "dtype_overflow",
        &["dtype_overflow"],
        "pkt-123",
        "user@test",
        "testing",
    );
    assert_eq!(
        event.audit_ref,
        "override:pkt-123:dtype_overflow:hardened:testing"
    );
}

#[test]
fn golden_override_normalizes_empty_packet() {
    let event = evaluate_policy_override(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        "dtype_overflow",
        &["dtype_overflow"],
        "",
        "user@test",
        "testing",
    );
    assert!(event.audit_ref.contains("unknown_packet"));
}

#[test]
fn golden_override_normalizes_empty_reason() {
    let event = evaluate_policy_override(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        "dtype_overflow",
        &["dtype_overflow"],
        "pkt-123",
        "user@test",
        "",
    );
    assert_eq!(event.reason_code, "unspecified");
}

// ─────────────────────────────────────────────────────────────────────────────
// DecisionAction golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_decision_action_as_str() {
    assert_eq!(DecisionAction::Allow.as_str(), "allow");
    assert_eq!(DecisionAction::FullValidate.as_str(), "full_validate");
    assert_eq!(DecisionAction::FailClosed.as_str(), "fail_closed");
}

// ─────────────────────────────────────────────────────────────────────────────
// DecisionLossModel golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_loss_model_default_values() {
    let model = DecisionLossModel::default();
    assert_eq!(model.allow_if_compatible, 0.0);
    assert_eq!(model.allow_if_incompatible, 100.0);
    assert_eq!(model.full_validate_if_compatible, 4.0);
    assert_eq!(model.full_validate_if_incompatible, 2.0);
    assert_eq!(model.fail_closed_if_compatible, 125.0);
    assert_eq!(model.fail_closed_if_incompatible, 1.0);
}
