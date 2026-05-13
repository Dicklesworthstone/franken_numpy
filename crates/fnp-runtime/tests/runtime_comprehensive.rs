//! Comprehensive tests for fnp-runtime.
//!
//! This test suite covers:
//! - RuntimeMode parsing and serialization
//! - CompatibilityClass parsing and serialization
//! - DecisionAction selection logic
//! - decide_compatibility() behavior in Strict vs Hardened modes
//! - EvidenceLedger recording
//! - Policy override evaluation
//! - Probability clamping and malformed input handling
//!
//! Finding: fnp-runtime had 0 test files despite 1527 LOC. This completes
//! the coverage gap identified in tick-26 project analysis.

use fnp_runtime::{
    CompatibilityClass, DecisionAction, DecisionAuditContext, DecisionLossModel, EvidenceLedger,
    RuntimeMode, decide_and_record, decide_compatibility, decide_compatibility_from_wire,
    evaluate_policy_override,
};

// ─────────────────────────────────────────────────────────────────────────────
// RuntimeMode tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn runtime_mode_from_wire_strict() {
    assert_eq!(RuntimeMode::from_wire("strict"), Some(RuntimeMode::Strict));
}

#[test]
fn runtime_mode_from_wire_hardened() {
    assert_eq!(
        RuntimeMode::from_wire("hardened"),
        Some(RuntimeMode::Hardened)
    );
}

#[test]
fn runtime_mode_from_wire_invalid() {
    assert_eq!(RuntimeMode::from_wire("unknown"), None);
    assert_eq!(RuntimeMode::from_wire(""), None);
    assert_eq!(RuntimeMode::from_wire("STRICT"), None);
}

#[test]
fn runtime_mode_as_str_roundtrip() {
    assert_eq!(RuntimeMode::Strict.as_str(), "strict");
    assert_eq!(RuntimeMode::Hardened.as_str(), "hardened");

    assert_eq!(
        RuntimeMode::from_wire(RuntimeMode::Strict.as_str()),
        Some(RuntimeMode::Strict)
    );
    assert_eq!(
        RuntimeMode::from_wire(RuntimeMode::Hardened.as_str()),
        Some(RuntimeMode::Hardened)
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// CompatibilityClass tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compatibility_class_parse_known_compatible() {
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
fn compatibility_class_parse_known_incompatible() {
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
fn compatibility_class_parse_unknown() {
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
fn compatibility_class_from_wire_defaults_to_unknown() {
    assert_eq!(
        CompatibilityClass::from_wire("invalid_value"),
        CompatibilityClass::Unknown
    );
}

#[test]
fn compatibility_class_as_str_roundtrip() {
    let classes = [
        CompatibilityClass::KnownCompatible,
        CompatibilityClass::KnownIncompatible,
        CompatibilityClass::Unknown,
    ];
    for class in classes {
        let wire = class.as_str();
        assert_eq!(CompatibilityClass::parse_wire(wire), Some(class));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DecisionAction tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn decision_action_as_str() {
    assert_eq!(DecisionAction::Allow.as_str(), "allow");
    assert_eq!(DecisionAction::FullValidate.as_str(), "full_validate");
    assert_eq!(DecisionAction::FailClosed.as_str(), "fail_closed");
}

// ─────────────────────────────────────────────────────────────────────────────
// decide_compatibility tests — Strict mode
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn strict_mode_allows_known_compatible() {
    let action = decide_compatibility(
        RuntimeMode::Strict,
        CompatibilityClass::KnownCompatible,
        0.5,
        0.5,
    );
    assert_eq!(action, DecisionAction::Allow);
}

#[test]
fn strict_mode_fails_known_incompatible() {
    let action = decide_compatibility(
        RuntimeMode::Strict,
        CompatibilityClass::KnownIncompatible,
        0.0,
        0.5,
    );
    assert_eq!(action, DecisionAction::FailClosed);
}

#[test]
fn strict_mode_fails_unknown() {
    let action = decide_compatibility(RuntimeMode::Strict, CompatibilityClass::Unknown, 0.0, 0.5);
    assert_eq!(action, DecisionAction::FailClosed);
}

// ─────────────────────────────────────────────────────────────────────────────
// decide_compatibility tests — Hardened mode
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hardened_mode_allows_low_risk_compatible() {
    let action = decide_compatibility(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        0.1,
        0.5,
    );
    assert_eq!(action, DecisionAction::Allow);
}

#[test]
fn hardened_mode_validates_high_risk_compatible() {
    let action = decide_compatibility(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        0.8,
        0.5,
    );
    assert_eq!(action, DecisionAction::FullValidate);
}

#[test]
fn hardened_mode_validates_at_threshold() {
    let action = decide_compatibility(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        0.5,
        0.5,
    );
    assert_eq!(action, DecisionAction::FullValidate);
}

#[test]
fn hardened_mode_fails_incompatible() {
    let action = decide_compatibility(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownIncompatible,
        0.0,
        0.5,
    );
    assert_eq!(action, DecisionAction::FailClosed);
}

#[test]
fn hardened_mode_validates_on_nan_risk() {
    let action = decide_compatibility(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        f64::NAN,
        0.5,
    );
    assert_eq!(action, DecisionAction::FullValidate);
}

#[test]
fn hardened_mode_validates_on_infinite_risk() {
    let action = decide_compatibility(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        f64::INFINITY,
        0.5,
    );
    assert_eq!(action, DecisionAction::FullValidate);
}

// ─────────────────────────────────────────────────────────────────────────────
// decide_compatibility_from_wire tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn from_wire_parses_valid_inputs() {
    let action = decide_compatibility_from_wire("strict", "known_compatible", 0.1, 0.5);
    assert_eq!(action, DecisionAction::Allow);
}

#[test]
fn from_wire_trims_whitespace() {
    let action = decide_compatibility_from_wire("  hardened  ", "  known_compatible  ", 0.1, 0.5);
    assert_eq!(action, DecisionAction::Allow);
}

#[test]
fn from_wire_fails_on_invalid_mode() {
    let action = decide_compatibility_from_wire("invalid", "known_compatible", 0.1, 0.5);
    assert_eq!(action, DecisionAction::FailClosed);
}

// ─────────────────────────────────────────────────────────────────────────────
// EvidenceLedger tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ledger_starts_empty() {
    let ledger = EvidenceLedger::new();
    assert!(ledger.events().is_empty());
    assert!(ledger.last().is_none());
}

#[test]
fn decide_and_record_adds_event() {
    let mut ledger = EvidenceLedger::new();
    let action = decide_and_record(
        &mut ledger,
        RuntimeMode::Strict,
        CompatibilityClass::KnownCompatible,
        0.1,
        0.5,
        "test note",
    );
    assert_eq!(action, DecisionAction::Allow);
    assert_eq!(ledger.events().len(), 1);

    let event = ledger.last().unwrap();
    assert_eq!(event.mode, RuntimeMode::Strict);
    assert_eq!(event.class, CompatibilityClass::KnownCompatible);
    assert_eq!(event.action, DecisionAction::Allow);
    assert_eq!(event.note, "test note");
}

#[test]
fn ledger_records_multiple_events() {
    let mut ledger = EvidenceLedger::new();

    decide_and_record(
        &mut ledger,
        RuntimeMode::Strict,
        CompatibilityClass::KnownCompatible,
        0.1,
        0.5,
        "first",
    );
    decide_and_record(
        &mut ledger,
        RuntimeMode::Hardened,
        CompatibilityClass::Unknown,
        0.9,
        0.5,
        "second",
    );

    assert_eq!(ledger.events().len(), 2);
    assert_eq!(ledger.last().unwrap().note, "second");
}

// ─────────────────────────────────────────────────────────────────────────────
// Policy override tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn override_approved_in_hardened_compatible_allowlisted() {
    let event = evaluate_policy_override(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        "special_case",
        &["special_case", "another_case"],
        "PKT-001",
        "user@example.com",
        "testing",
    );
    assert!(event.approved);
    assert_eq!(event.action, DecisionAction::FullValidate);
}

#[test]
fn override_rejected_in_strict_mode() {
    let event = evaluate_policy_override(
        RuntimeMode::Strict,
        CompatibilityClass::KnownCompatible,
        "special_case",
        &["special_case"],
        "PKT-001",
        "user@example.com",
        "testing",
    );
    assert!(!event.approved);
    assert_eq!(event.action, DecisionAction::FailClosed);
}

#[test]
fn override_rejected_for_incompatible_class() {
    let event = evaluate_policy_override(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownIncompatible,
        "special_case",
        &["special_case"],
        "PKT-001",
        "user@example.com",
        "testing",
    );
    assert!(!event.approved);
    assert_eq!(event.action, DecisionAction::FailClosed);
}

#[test]
fn override_rejected_when_not_allowlisted() {
    let event = evaluate_policy_override(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        "not_in_list",
        &["special_case"],
        "PKT-001",
        "user@example.com",
        "testing",
    );
    assert!(!event.approved);
    assert_eq!(event.action, DecisionAction::FailClosed);
}

#[test]
fn override_normalizes_empty_fields() {
    let event = evaluate_policy_override(
        RuntimeMode::Hardened,
        CompatibilityClass::KnownCompatible,
        "",
        &[],
        "",
        "",
        "",
    );
    assert_eq!(event.packet_id, "unknown_packet");
    assert_eq!(event.requested_by, "unknown_requester");
    assert_eq!(event.reason_code, "unspecified");
    assert!(!event.approved);
}

// ─────────────────────────────────────────────────────────────────────────────
// DecisionLossModel tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn loss_model_default_values() {
    let model = DecisionLossModel::default();
    assert_eq!(model.allow_if_compatible, 0.0);
    assert_eq!(model.allow_if_incompatible, 100.0);
    assert!(model.fail_closed_if_compatible > model.allow_if_compatible);
}

// ─────────────────────────────────────────────────────────────────────────────
// DecisionAuditContext tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn audit_context_default_values() {
    let ctx = DecisionAuditContext::default();
    assert_eq!(ctx.fixture_id, "unknown_fixture");
    assert_eq!(ctx.seed, 0);
    assert_eq!(ctx.env_fingerprint, "unknown_env");
    assert!(ctx.artifact_refs.is_empty());
    assert_eq!(ctx.reason_code, "unspecified");
}
