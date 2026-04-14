#![forbid(unsafe_code)]

use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    Strict,
    Hardened,
}

impl RuntimeMode {
    #[must_use]
    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "strict" => Some(Self::Strict),
            "hardened" => Some(Self::Hardened),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Hardened => "hardened",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityClass {
    KnownCompatible,
    KnownIncompatible,
    Unknown,
}

impl CompatibilityClass {
    #[must_use]
    pub fn parse_wire(value: &str) -> Option<Self> {
        match value {
            "known_compatible" | "known_compatible_low_risk" | "known_compatible_high_risk" => {
                Some(Self::KnownCompatible)
            }
            "known_incompatible" | "known_incompatible_semantics" => Some(Self::KnownIncompatible),
            "unknown" | "unknown_semantics" => Some(Self::Unknown),
            _ => None,
        }
    }

    #[must_use]
    pub fn from_wire(value: &str) -> Self {
        Self::parse_wire(value).unwrap_or(Self::Unknown)
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::KnownCompatible => "known_compatible",
            Self::KnownIncompatible => "known_incompatible",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionAction {
    Allow,
    FullValidate,
    FailClosed,
}

impl DecisionAction {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::FullValidate => "full_validate",
            Self::FailClosed => "fail_closed",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceTerm {
    pub name: &'static str,
    pub log_likelihood_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionLossModel {
    pub allow_if_compatible: f64,
    pub allow_if_incompatible: f64,
    pub full_validate_if_compatible: f64,
    pub full_validate_if_incompatible: f64,
    pub fail_closed_if_compatible: f64,
    pub fail_closed_if_incompatible: f64,
}

impl Default for DecisionLossModel {
    fn default() -> Self {
        Self {
            allow_if_compatible: 0.0,
            allow_if_incompatible: 100.0,
            full_validate_if_compatible: 4.0,
            full_validate_if_incompatible: 2.0,
            fail_closed_if_compatible: 125.0,
            fail_closed_if_incompatible: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionAuditContext {
    pub fixture_id: String,
    pub seed: u64,
    pub env_fingerprint: String,
    pub artifact_refs: Vec<String>,
    pub reason_code: String,
}

impl Default for DecisionAuditContext {
    fn default() -> Self {
        Self {
            fixture_id: "unknown_fixture".to_string(),
            seed: 0,
            env_fingerprint: "unknown_env".to_string(),
            artifact_refs: Vec::new(),
            reason_code: "unspecified".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionEvent {
    pub ts_millis: u128,
    pub mode: RuntimeMode,
    pub class: CompatibilityClass,
    pub risk_score: f64,
    pub action: DecisionAction,
    pub posterior_incompatible: f64,
    pub expected_loss_allow: f64,
    pub expected_loss_full_validate: f64,
    pub expected_loss_fail_closed: f64,
    pub selected_expected_loss: f64,
    pub evidence_terms: Vec<EvidenceTerm>,
    pub fixture_id: String,
    pub seed: u64,
    pub env_fingerprint: String,
    pub artifact_refs: Vec<String>,
    pub reason_code: String,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverrideAuditEvent {
    pub ts_millis: u128,
    pub mode: RuntimeMode,
    pub class: CompatibilityClass,
    pub requested_deviation_class: String,
    pub packet_id: String,
    pub requested_by: String,
    pub reason_code: String,
    pub approved: bool,
    pub action: DecisionAction,
    pub audit_ref: String,
}

#[derive(Debug, Default, Clone)]
pub struct EvidenceLedger {
    events: Vec<DecisionEvent>,
}

impl EvidenceLedger {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, event: DecisionEvent) {
        self.events.push(event);
    }

    #[must_use]
    pub fn events(&self) -> &[DecisionEvent] {
        &self.events
    }

    #[must_use]
    pub fn last(&self) -> Option<&DecisionEvent> {
        self.events.last()
    }
}

#[must_use]
pub fn decide_compatibility(
    mode: RuntimeMode,
    class: CompatibilityClass,
    risk_score: f64,
    hardened_validation_threshold: f64,
) -> DecisionAction {
    match class {
        CompatibilityClass::KnownIncompatible | CompatibilityClass::Unknown => {
            DecisionAction::FailClosed
        }
        CompatibilityClass::KnownCompatible => match mode {
            RuntimeMode::Strict => DecisionAction::Allow,
            RuntimeMode::Hardened => {
                if is_malformed_probability_input(risk_score)
                    || is_malformed_probability_input(hardened_validation_threshold)
                {
                    return DecisionAction::FullValidate;
                }
                let risk = clamp_probability(risk_score);
                let threshold = clamp_probability(hardened_validation_threshold);
                if risk >= threshold {
                    DecisionAction::FullValidate
                } else {
                    DecisionAction::Allow
                }
            }
        },
    }
}

#[must_use]
pub fn decide_compatibility_from_wire(
    mode_raw: &str,
    class_raw: &str,
    risk_score: f64,
    hardened_validation_threshold: f64,
) -> DecisionAction {
    let Some(mode) = RuntimeMode::from_wire(mode_raw.trim()) else {
        return DecisionAction::FailClosed;
    };
    let class = CompatibilityClass::from_wire(class_raw.trim());
    decide_compatibility(mode, class, risk_score, hardened_validation_threshold)
}

pub fn decide_and_record(
    ledger: &mut EvidenceLedger,
    mode: RuntimeMode,
    class: CompatibilityClass,
    risk_score: f64,
    hardened_validation_threshold: f64,
    note: impl Into<String>,
) -> DecisionAction {
    decide_and_record_with_context(
        ledger,
        mode,
        class,
        risk_score,
        hardened_validation_threshold,
        DecisionAuditContext::default(),
        note,
    )
}

pub fn decide_and_record_with_context(
    ledger: &mut EvidenceLedger,
    mode: RuntimeMode,
    class: CompatibilityClass,
    risk_score: f64,
    hardened_validation_threshold: f64,
    context: DecisionAuditContext,
    note: impl Into<String>,
) -> DecisionAction {
    let action = decide_compatibility(mode, class, risk_score, hardened_validation_threshold);
    let (posterior_incompatible, evidence_terms) =
        posterior_incompatibility(class, risk_score, hardened_validation_threshold);
    let loss_model = DecisionLossModel::default();
    let expected_loss_allow =
        expected_loss_for_action(DecisionAction::Allow, posterior_incompatible, loss_model);
    let expected_loss_full_validate = expected_loss_for_action(
        DecisionAction::FullValidate,
        posterior_incompatible,
        loss_model,
    );
    let expected_loss_fail_closed = expected_loss_for_action(
        DecisionAction::FailClosed,
        posterior_incompatible,
        loss_model,
    );
    let selected_expected_loss = match action {
        DecisionAction::Allow => expected_loss_allow,
        DecisionAction::FullValidate => expected_loss_full_validate,
        DecisionAction::FailClosed => expected_loss_fail_closed,
    };

    let ts_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let normalized = normalize_audit_context(context);
    ledger.record(DecisionEvent {
        ts_millis,
        mode,
        class,
        risk_score,
        action,
        posterior_incompatible,
        expected_loss_allow,
        expected_loss_full_validate,
        expected_loss_fail_closed,
        selected_expected_loss,
        evidence_terms,
        fixture_id: normalized.fixture_id,
        seed: normalized.seed,
        env_fingerprint: normalized.env_fingerprint,
        artifact_refs: normalized.artifact_refs,
        reason_code: normalized.reason_code,
        note: note.into(),
    });
    action
}

#[must_use]
pub fn evaluate_policy_override(
    mode: RuntimeMode,
    class: CompatibilityClass,
    requested_deviation_class: &str,
    allowlisted_classes: &[&str],
    packet_id: &str,
    requested_by: &str,
    reason_code: &str,
) -> OverrideAuditEvent {
    let ts_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());

    let requested = requested_deviation_class.trim();
    let allowlisted = !requested.is_empty() && allowlisted_classes.contains(&requested);
    let approved = matches!(mode, RuntimeMode::Hardened)
        && matches!(class, CompatibilityClass::KnownCompatible)
        && allowlisted;
    let action = if approved {
        // Any approved override still pays the safety tax by forcing full validation.
        DecisionAction::FullValidate
    } else {
        DecisionAction::FailClosed
    };

    let packet = if packet_id.trim().is_empty() {
        "unknown_packet"
    } else {
        packet_id.trim()
    };
    let requester = if requested_by.trim().is_empty() {
        "unknown_requester"
    } else {
        requested_by.trim()
    };
    let normalized_reason = if reason_code.trim().is_empty() {
        "unspecified"
    } else {
        reason_code.trim()
    };

    let audit_ref = format!(
        "override:{}:{}:{}:{}",
        packet,
        requested,
        mode.as_str(),
        normalized_reason
    );

    OverrideAuditEvent {
        ts_millis,
        mode,
        class,
        requested_deviation_class: requested.to_string(),
        packet_id: packet.to_string(),
        requested_by: requester.to_string(),
        reason_code: normalized_reason.to_string(),
        approved,
        action,
        audit_ref,
    }
}

fn normalize_audit_context(mut context: DecisionAuditContext) -> DecisionAuditContext {
    if context.fixture_id.trim().is_empty() {
        context.fixture_id = "unknown_fixture".to_string();
    } else {
        context.fixture_id = context.fixture_id.trim().to_string();
    }
    if context.env_fingerprint.trim().is_empty() {
        context.env_fingerprint = "unknown_env".to_string();
    } else {
        context.env_fingerprint = context.env_fingerprint.trim().to_string();
    }
    if context.reason_code.trim().is_empty() {
        context.reason_code = "unspecified".to_string();
    } else {
        context.reason_code = context.reason_code.trim().to_string();
    }
    context.artifact_refs = context
        .artifact_refs
        .into_iter()
        .map(|artifact| artifact.trim().to_string())
        .filter(|artifact| !artifact.is_empty())
        .collect();
    if context.artifact_refs.is_empty() {
        context.artifact_refs = vec!["unknown_artifact".to_string()];
    }
    context
}

fn clamp_probability(p: f64) -> f64 {
    if p.is_nan() {
        return 0.5;
    }
    p.clamp(1e-9, 1.0 - 1e-9)
}

fn is_malformed_probability_input(p: f64) -> bool {
    !p.is_finite() || !(0.0..=1.0).contains(&p)
}

fn class_prior_incompatible(class: CompatibilityClass) -> f64 {
    match class {
        CompatibilityClass::KnownCompatible => 0.01,
        CompatibilityClass::KnownIncompatible => 0.99,
        CompatibilityClass::Unknown => 0.5,
    }
}

#[must_use]
pub fn posterior_incompatibility(
    class: CompatibilityClass,
    risk_score: f64,
    hardened_validation_threshold: f64,
) -> (f64, Vec<EvidenceTerm>) {
    let prior = clamp_probability(class_prior_incompatible(class));
    let risk = clamp_probability(risk_score);
    let threshold = clamp_probability(hardened_validation_threshold);

    let prior_log_odds = (prior / (1.0 - prior)).ln();
    let risk_log_odds = (risk / (1.0 - risk)).ln();
    let threshold_log_odds = (threshold / (1.0 - threshold)).ln();
    let risk_margin_llr = risk_log_odds - threshold_log_odds;
    let posterior_log_odds = prior_log_odds + risk_margin_llr;
    let posterior = 1.0 / (1.0 + (-posterior_log_odds).exp());

    (
        posterior,
        vec![
            EvidenceTerm {
                name: "prior_class_log_odds",
                log_likelihood_ratio: prior_log_odds,
            },
            EvidenceTerm {
                name: "risk_vs_threshold_llr",
                log_likelihood_ratio: risk_margin_llr,
            },
        ],
    )
}

#[must_use]
pub fn expected_loss_for_action(
    action: DecisionAction,
    posterior_incompatible: f64,
    model: DecisionLossModel,
) -> f64 {
    let p_incompatible = clamp_probability(posterior_incompatible);
    let p_compatible = 1.0 - p_incompatible;

    match action {
        DecisionAction::Allow => {
            p_compatible * model.allow_if_compatible + p_incompatible * model.allow_if_incompatible
        }
        DecisionAction::FullValidate => {
            p_compatible * model.full_validate_if_compatible
                + p_incompatible * model.full_validate_if_incompatible
        }
        DecisionAction::FailClosed => {
            p_compatible * model.fail_closed_if_compatible
                + p_incompatible * model.fail_closed_if_incompatible
        }
    }
}

#[cfg(feature = "asupersync")]
pub mod asupersync_integration {
    /// Marker function proving asupersync linkage is available for runtime plumbing.
    #[must_use]
    pub fn runtime_tag() -> &'static str {
        "asupersync"
    }
}

#[cfg(feature = "frankentui")]
pub mod frankentui_integration {
    /// Marker function proving FrankenTUI linkage is available for observability UIs.
    #[must_use]
    pub fn ui_tag() -> &'static str {
        "frankentui"
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompatibilityClass, DecisionAction, DecisionAuditContext, DecisionLossModel,
        EvidenceLedger, RuntimeMode, decide_and_record, decide_and_record_with_context,
        decide_compatibility, decide_compatibility_from_wire, evaluate_policy_override,
        expected_loss_for_action, posterior_incompatibility,
    };

    #[test]
    fn strict_mode_allows_only_known_compatible() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Strict,
                CompatibilityClass::KnownCompatible,
                0.2,
                0.5,
            ),
            DecisionAction::Allow
        );
        assert_eq!(
            decide_compatibility(RuntimeMode::Strict, CompatibilityClass::Unknown, 0.2, 0.5),
            DecisionAction::FailClosed
        );
    }

    #[test]
    fn hardened_mode_escalates_risky_inputs() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                0.9,
                0.7,
            ),
            DecisionAction::FullValidate
        );
    }

    #[test]
    fn hardened_mode_full_validates_nan_risk() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                f64::NAN,
                0.7,
            ),
            DecisionAction::FullValidate
        );
    }

    #[test]
    fn hardened_mode_full_validates_nan_threshold() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                0.1,
                f64::NAN,
            ),
            DecisionAction::FullValidate
        );
    }

    #[test]
    fn hardened_mode_full_validates_out_of_range_probability_inputs() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                1.5,
                0.7,
            ),
            DecisionAction::FullValidate
        );
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                0.1,
                -0.2,
            ),
            DecisionAction::FullValidate
        );
    }

    #[test]
    fn records_evidence() {
        let mut ledger = EvidenceLedger::new();
        let action = decide_and_record(
            &mut ledger,
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            0.1,
            0.7,
            "broadcast_request",
        );
        assert_eq!(action, DecisionAction::Allow);
        assert_eq!(ledger.events().len(), 1);
        let event = ledger.last().expect("event should be present");
        assert_eq!(event.note, "broadcast_request");
        assert!((0.0..=1.0).contains(&event.posterior_incompatible));
        assert!(!event.evidence_terms.is_empty());
        assert!(event.selected_expected_loss.is_finite());
        assert_eq!(event.fixture_id, "unknown_fixture");
        assert_eq!(event.reason_code, "unspecified");
    }

    #[test]
    fn recorded_decision_full_validates_malformed_probability_inputs() {
        let mut ledger = EvidenceLedger::new();
        let action = decide_and_record(
            &mut ledger,
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            f64::NAN,
            0.7,
            "nan-risk",
        );
        assert_eq!(action, DecisionAction::FullValidate);
        let event = ledger.last().expect("event should be present");
        assert_eq!(event.action, DecisionAction::FullValidate);
        assert!(event.posterior_incompatible.is_finite());
    }

    #[test]
    fn decision_context_is_recorded_for_forensics() {
        let mut ledger = EvidenceLedger::new();
        let context = DecisionAuditContext {
            fixture_id: "strict_unknown_fail_closed".to_string(),
            seed: 1337,
            env_fingerprint: "linux-x86_64-rust-2024".to_string(),
            artifact_refs: vec![
                "crates/fnp-conformance/fixtures/runtime_policy_cases.json".to_string(),
                "artifacts/contracts/SECURITY_COMPATIBILITY_THREAT_MATRIX_V1.md".to_string(),
            ],
            reason_code: "unknown_metadata_version".to_string(),
        };

        let action = decide_and_record_with_context(
            &mut ledger,
            RuntimeMode::Strict,
            CompatibilityClass::Unknown,
            0.1,
            0.7,
            context,
            "wire-class-decode",
        );
        assert_eq!(action, DecisionAction::FailClosed);

        let event = ledger.last().expect("event should be present");
        assert_eq!(event.fixture_id, "strict_unknown_fail_closed");
        assert_eq!(event.seed, 1337);
        assert_eq!(event.env_fingerprint, "linux-x86_64-rust-2024");
        assert_eq!(event.reason_code, "unknown_metadata_version");
        assert_eq!(event.artifact_refs.len(), 2);
    }

    #[test]
    fn wire_decoding_fails_closed_for_unknown_inputs() {
        assert_eq!(
            decide_compatibility_from_wire("strict", "completely_unknown", 0.2, 0.7),
            DecisionAction::FailClosed
        );
        assert_eq!(
            decide_compatibility_from_wire("mystery_mode", "known_compatible", 0.2, 0.7),
            DecisionAction::FailClosed
        );
    }

    #[test]
    fn policy_override_requires_hardened_allowlist_and_known_compatible() {
        let allowlisted = [
            "parser_diagnostic_enrichment",
            "admission_guard_caps",
            "recovery_with_integrity_proof",
        ];

        let approved = evaluate_policy_override(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            "admission_guard_caps",
            &allowlisted,
            "FNP-P2C-006",
            "ci-bot",
            "defensive_cap",
        );
        assert!(approved.approved);
        assert_eq!(approved.action, DecisionAction::FullValidate);

        let denied = evaluate_policy_override(
            RuntimeMode::Strict,
            CompatibilityClass::KnownCompatible,
            "admission_guard_caps",
            &allowlisted,
            "FNP-P2C-006",
            "ci-bot",
            "strict_override_attempt",
        );
        assert!(!denied.approved);
        assert_eq!(denied.action, DecisionAction::FailClosed);

        let denied_unknown = evaluate_policy_override(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            "unknown_override",
            &allowlisted,
            "FNP-P2C-006",
            "ci-bot",
            "unknown_override_attempt",
        );
        assert!(!denied_unknown.approved);
        assert_eq!(denied_unknown.action, DecisionAction::FailClosed);
    }

    #[test]
    fn posterior_rises_with_risk_score() {
        let (p_low, _) = posterior_incompatibility(CompatibilityClass::KnownCompatible, 0.1, 0.5);
        let (p_high, _) = posterior_incompatibility(CompatibilityClass::KnownCompatible, 0.9, 0.5);
        assert!(p_high > p_low);
    }

    #[test]
    fn expected_loss_matches_loss_matrix_intuition() {
        let model = DecisionLossModel::default();
        let low_risk = 0.05;
        let high_risk = 0.95;

        let allow_low = expected_loss_for_action(DecisionAction::Allow, low_risk, model);
        let fail_closed_low = expected_loss_for_action(DecisionAction::FailClosed, low_risk, model);
        assert!(allow_low < fail_closed_low);

        let allow_high = expected_loss_for_action(DecisionAction::Allow, high_risk, model);
        let validate_high =
            expected_loss_for_action(DecisionAction::FullValidate, high_risk, model);
        assert!(validate_high < allow_high);
    }

    // -----------------------------------------------------------------------
    // Decision matrix exhaustive coverage
    // -----------------------------------------------------------------------

    #[test]
    fn hardened_mode_allows_low_risk_known_compatible() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                0.1,
                0.7,
            ),
            DecisionAction::Allow
        );
    }

    #[test]
    fn known_incompatible_always_fails_closed_strict() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Strict,
                CompatibilityClass::KnownIncompatible,
                0.0,
                0.0,
            ),
            DecisionAction::FailClosed
        );
    }

    #[test]
    fn known_incompatible_always_fails_closed_hardened() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Hardened,
                CompatibilityClass::KnownIncompatible,
                0.0,
                0.0,
            ),
            DecisionAction::FailClosed
        );
    }

    #[test]
    fn unknown_class_always_fails_closed_strict() {
        assert_eq!(
            decide_compatibility(RuntimeMode::Strict, CompatibilityClass::Unknown, 0.0, 0.0),
            DecisionAction::FailClosed
        );
    }

    #[test]
    fn unknown_class_always_fails_closed_hardened() {
        assert_eq!(
            decide_compatibility(RuntimeMode::Hardened, CompatibilityClass::Unknown, 0.0, 0.0),
            DecisionAction::FailClosed
        );
    }

    #[test]
    fn hardened_exact_threshold_triggers_full_validate() {
        // risk == threshold should trigger full_validate (>= comparison)
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                0.5,
                0.5,
            ),
            DecisionAction::FullValidate
        );
    }

    #[test]
    fn hardened_just_below_threshold_allows() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                0.49,
                0.5,
            ),
            DecisionAction::Allow
        );
    }

    #[test]
    fn hardened_infinity_risk_full_validates() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                f64::INFINITY,
                0.5,
            ),
            DecisionAction::FullValidate
        );
    }

    #[test]
    fn hardened_neg_infinity_threshold_full_validates() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                0.5,
                f64::NEG_INFINITY,
            ),
            DecisionAction::FullValidate
        );
    }

    #[test]
    fn hardened_neg_infinity_risk_full_validates() {
        assert_eq!(
            decide_compatibility(
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                f64::NEG_INFINITY,
                0.5,
            ),
            DecisionAction::FullValidate
        );
    }

    // -----------------------------------------------------------------------
    // Wire decoding edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn runtime_mode_from_wire_roundtrip() {
        assert_eq!(RuntimeMode::from_wire("strict"), Some(RuntimeMode::Strict));
        assert_eq!(
            RuntimeMode::from_wire("hardened"),
            Some(RuntimeMode::Hardened)
        );
        assert_eq!(RuntimeMode::from_wire("unknown"), None);
        assert_eq!(RuntimeMode::from_wire(""), None);
        assert_eq!(RuntimeMode::from_wire("STRICT"), None); // case-sensitive
    }

    #[test]
    fn compatibility_class_from_wire_defaults_to_unknown() {
        assert_eq!(
            CompatibilityClass::from_wire("known_compatible"),
            CompatibilityClass::KnownCompatible
        );
        assert_eq!(
            CompatibilityClass::from_wire("known_incompatible"),
            CompatibilityClass::KnownIncompatible
        );
        assert_eq!(
            CompatibilityClass::from_wire("anything_else"),
            CompatibilityClass::Unknown
        );
        assert_eq!(
            CompatibilityClass::from_wire(""),
            CompatibilityClass::Unknown
        );
    }

    #[test]
    fn wire_decoding_valid_combinations() {
        assert_eq!(
            decide_compatibility_from_wire("strict", "known_compatible", 0.1, 0.5),
            DecisionAction::Allow
        );
        assert_eq!(
            decide_compatibility_from_wire("hardened", "known_compatible", 0.1, 0.5),
            DecisionAction::Allow
        );
        assert_eq!(
            decide_compatibility_from_wire("hardened", "known_compatible", 0.9, 0.5),
            DecisionAction::FullValidate
        );
        assert_eq!(
            decide_compatibility_from_wire("strict", "known_compatible_low_risk", 0.1, 0.5),
            DecisionAction::Allow
        );
        assert_eq!(
            decide_compatibility_from_wire("hardened", "known_compatible_high_risk", 0.9, 0.5),
            DecisionAction::FullValidate
        );
        assert_eq!(
            decide_compatibility_from_wire("strict", "known_incompatible_semantics", 0.1, 0.5),
            DecisionAction::FailClosed
        );
        assert_eq!(
            decide_compatibility_from_wire("strict", "unknown_semantics", 0.1, 0.5),
            DecisionAction::FailClosed
        );
    }

    #[test]
    fn wire_decoding_trims_metadata_before_parsing() {
        assert_eq!(
            decide_compatibility_from_wire(" strict ", " known_compatible ", 0.1, 0.5),
            DecisionAction::Allow
        );
        assert_eq!(
            decide_compatibility_from_wire(" hardened ", " unknown ", 0.1, 0.5),
            DecisionAction::FailClosed
        );
    }

    // -----------------------------------------------------------------------
    // Posterior incompatibility edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn posterior_is_finite_for_nan_inputs() {
        let (p, terms) =
            posterior_incompatibility(CompatibilityClass::KnownCompatible, f64::NAN, 0.5);
        assert!(p.is_finite(), "posterior must be finite even with NaN risk");
        assert_eq!(terms.len(), 2);
    }

    #[test]
    fn posterior_is_finite_for_extreme_inputs() {
        let (p1, _) = posterior_incompatibility(CompatibilityClass::KnownCompatible, 0.0, 0.0);
        assert!(p1.is_finite());
        let (p2, _) = posterior_incompatibility(CompatibilityClass::KnownCompatible, 1.0, 1.0);
        assert!(p2.is_finite());
    }

    #[test]
    fn posterior_known_incompatible_is_high() {
        let (p, _) = posterior_incompatibility(CompatibilityClass::KnownIncompatible, 0.5, 0.5);
        assert!(
            p > 0.9,
            "known_incompatible prior should yield high posterior, got {p}"
        );
    }

    #[test]
    fn posterior_unknown_class_is_moderate() {
        let (p, _) = posterior_incompatibility(CompatibilityClass::Unknown, 0.5, 0.5);
        assert!(
            (0.3..=0.7).contains(&p),
            "unknown class with 50/50 risk/threshold should yield moderate posterior, got {p}"
        );
    }

    #[test]
    fn posterior_determinism() {
        for class in [
            CompatibilityClass::KnownCompatible,
            CompatibilityClass::KnownIncompatible,
            CompatibilityClass::Unknown,
        ] {
            let (p1, t1) = posterior_incompatibility(class, 0.3, 0.7);
            let (p2, t2) = posterior_incompatibility(class, 0.3, 0.7);
            assert_eq!(p1, p2, "posterior must be deterministic for {class:?}");
            assert_eq!(t1.len(), t2.len());
        }
    }

    // -----------------------------------------------------------------------
    // Expected loss with custom model
    // -----------------------------------------------------------------------

    #[test]
    fn expected_loss_with_custom_model() {
        let model = DecisionLossModel {
            allow_if_compatible: 0.0,
            allow_if_incompatible: 10.0,
            full_validate_if_compatible: 1.0,
            full_validate_if_incompatible: 1.0,
            fail_closed_if_compatible: 5.0,
            fail_closed_if_incompatible: 0.0,
        };

        // At 50% posterior: allow = 0*0.5 + 10*0.5 = 5.0
        let allow = expected_loss_for_action(DecisionAction::Allow, 0.5, model);
        assert!((allow - 5.0).abs() < 0.1, "allow loss at 50%: {allow}");

        // At 50% posterior: full_validate = 1*0.5 + 1*0.5 = 1.0
        let fv = expected_loss_for_action(DecisionAction::FullValidate, 0.5, model);
        assert!((fv - 1.0).abs() < 0.1, "full_validate loss at 50%: {fv}");

        // At 50% posterior: fail_closed = 5*0.5 + 0*0.5 = 2.5
        let fc = expected_loss_for_action(DecisionAction::FailClosed, 0.5, model);
        assert!((fc - 2.5).abs() < 0.1, "fail_closed loss at 50%: {fc}");
    }

    #[test]
    fn expected_loss_is_finite_for_extreme_posteriors() {
        let model = DecisionLossModel::default();
        for p in [0.0, 1.0, f64::NAN, f64::INFINITY] {
            let loss = expected_loss_for_action(DecisionAction::Allow, p, model);
            assert!(loss.is_finite(), "loss must be finite for posterior={p}");
        }
    }

    // -----------------------------------------------------------------------
    // Override policy edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn override_denied_for_incompatible_even_if_allowlisted() {
        let allowlisted = ["special_override"];
        let event = evaluate_policy_override(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownIncompatible,
            "special_override",
            &allowlisted,
            "FNP-P2C-001",
            "admin",
            "emergency",
        );
        assert!(!event.approved);
        assert_eq!(event.action, DecisionAction::FailClosed);
    }

    #[test]
    fn override_denied_for_unknown_class_even_if_allowlisted() {
        let allowlisted = ["special_override"];
        let event = evaluate_policy_override(
            RuntimeMode::Hardened,
            CompatibilityClass::Unknown,
            "special_override",
            &allowlisted,
            "FNP-P2C-001",
            "admin",
            "emergency",
        );
        assert!(!event.approved);
    }

    #[test]
    fn override_empty_deviation_class_is_denied() {
        let allowlisted = [""];
        let event = evaluate_policy_override(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            "",
            &allowlisted,
            "FNP-P2C-001",
            "admin",
            "test",
        );
        assert!(!event.approved);
    }

    #[test]
    fn override_whitespace_deviation_class_is_denied() {
        let allowlisted = ["   "];
        let event = evaluate_policy_override(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            "   ",
            &allowlisted,
            "FNP-P2C-001",
            "admin",
            "test",
        );
        assert!(!event.approved);
    }

    #[test]
    fn override_normalizes_empty_metadata_fields() {
        let event = evaluate_policy_override(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            "some_override",
            &[],
            "",
            "",
            "",
        );
        assert_eq!(event.packet_id, "unknown_packet");
        assert_eq!(event.requested_by, "unknown_requester");
        assert_eq!(event.reason_code, "unspecified");
    }

    #[test]
    fn override_audit_ref_format_is_stable() {
        let allowlisted = ["cap_override"];
        let event = evaluate_policy_override(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            "cap_override",
            &allowlisted,
            "FNP-P2C-003",
            "ci-bot",
            "transfer_cap",
        );
        assert!(event.approved);
        assert_eq!(
            event.audit_ref,
            "override:FNP-P2C-003:cap_override:hardened:transfer_cap"
        );
    }

    // -----------------------------------------------------------------------
    // Evidence ledger multi-event
    // -----------------------------------------------------------------------

    #[test]
    fn evidence_ledger_records_multiple_events_in_order() {
        let mut ledger = EvidenceLedger::new();
        assert!(ledger.events().is_empty());

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
        decide_and_record(
            &mut ledger,
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            0.3,
            0.5,
            "third",
        );

        assert_eq!(ledger.events().len(), 3);
        assert_eq!(ledger.events()[0].note, "first");
        assert_eq!(ledger.events()[0].action, DecisionAction::Allow);
        assert_eq!(ledger.events()[1].note, "second");
        assert_eq!(ledger.events()[1].action, DecisionAction::FailClosed);
        assert_eq!(ledger.events()[2].note, "third");
        assert_eq!(ledger.events()[2].action, DecisionAction::Allow);
        assert_eq!(ledger.last().unwrap().note, "third");
    }

    // -----------------------------------------------------------------------
    // Context normalization
    // -----------------------------------------------------------------------

    #[test]
    fn context_normalization_fills_empty_fields() {
        let mut ledger = EvidenceLedger::new();
        let context = DecisionAuditContext {
            fixture_id: "  ".to_string(),
            seed: 0,
            env_fingerprint: String::new(),
            artifact_refs: Vec::new(),
            reason_code: "   ".to_string(),
        };
        decide_and_record_with_context(
            &mut ledger,
            RuntimeMode::Strict,
            CompatibilityClass::KnownCompatible,
            0.1,
            0.5,
            context,
            "normalization test",
        );
        let event = ledger.last().unwrap();
        assert_eq!(event.fixture_id, "unknown_fixture");
        assert_eq!(event.env_fingerprint, "unknown_env");
        assert_eq!(event.reason_code, "unspecified");
        assert_eq!(event.artifact_refs, vec!["unknown_artifact"]);
    }

    #[test]
    fn context_preserves_non_empty_fields() {
        let mut ledger = EvidenceLedger::new();
        let context = DecisionAuditContext {
            fixture_id: " test-42 ".to_string(),
            seed: 42,
            env_fingerprint: " linux-test ".to_string(),
            artifact_refs: vec![" ref1 ".to_string()],
            reason_code: " test_reason ".to_string(),
        };
        decide_and_record_with_context(
            &mut ledger,
            RuntimeMode::Strict,
            CompatibilityClass::KnownCompatible,
            0.1,
            0.5,
            context,
            "preserve test",
        );
        let event = ledger.last().unwrap();
        assert_eq!(event.fixture_id, "test-42");
        assert_eq!(event.seed, 42);
        assert_eq!(event.env_fingerprint, "linux-test");
        assert_eq!(event.artifact_refs, vec!["ref1"]);
        assert_eq!(event.reason_code, "test_reason");
    }

    #[test]
    fn context_normalization_drops_blank_artifact_refs() {
        let mut ledger = EvidenceLedger::new();
        let context = DecisionAuditContext {
            fixture_id: "fixture".to_string(),
            seed: 7,
            env_fingerprint: "env".to_string(),
            artifact_refs: vec![" artifact-a ".to_string(), "   ".to_string()],
            reason_code: "reason".to_string(),
        };
        decide_and_record_with_context(
            &mut ledger,
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            0.2,
            0.5,
            context,
            "artifact normalization",
        );
        let event = ledger.last().unwrap();
        assert_eq!(event.artifact_refs, vec!["artifact-a"]);
    }

    // -----------------------------------------------------------------------
    // Decision action string representations
    // -----------------------------------------------------------------------

    #[test]
    fn decision_action_as_str_is_stable() {
        assert_eq!(DecisionAction::Allow.as_str(), "allow");
        assert_eq!(DecisionAction::FullValidate.as_str(), "full_validate");
        assert_eq!(DecisionAction::FailClosed.as_str(), "fail_closed");
    }

    #[test]
    fn runtime_mode_as_str_is_stable() {
        assert_eq!(RuntimeMode::Strict.as_str(), "strict");
        assert_eq!(RuntimeMode::Hardened.as_str(), "hardened");
    }

    #[test]
    fn compatibility_class_as_str_is_stable() {
        assert_eq!(
            CompatibilityClass::KnownCompatible.as_str(),
            "known_compatible"
        );
        assert_eq!(
            CompatibilityClass::KnownIncompatible.as_str(),
            "known_incompatible"
        );
        assert_eq!(CompatibilityClass::Unknown.as_str(), "unknown");
    }

    // -----------------------------------------------------------------------
    // Bayesian decision engine edge cases (br-x2y)
    // -----------------------------------------------------------------------

    #[test]
    fn expected_loss_at_zero_posterior_favors_allow() {
        // With posterior_incompatible ≈ 0.0 (clamped to 1e-9), the cost of
        // allowing is near-zero while fail_closed costs ~125.
        let model = DecisionLossModel::default();
        let allow = expected_loss_for_action(DecisionAction::Allow, 0.0, model);
        let fv = expected_loss_for_action(DecisionAction::FullValidate, 0.0, model);
        let fc = expected_loss_for_action(DecisionAction::FailClosed, 0.0, model);
        assert!(
            allow < fv,
            "allow should beat full_validate at ~0% posterior"
        );
        assert!(allow < fc, "allow should beat fail_closed at ~0% posterior");
    }

    #[test]
    fn expected_loss_at_one_posterior_favors_fail_closed() {
        // With posterior_incompatible ≈ 1.0 (clamped to 1-1e-9), fail_closed
        // costs ~1.0 while allow costs ~100.
        let model = DecisionLossModel::default();
        let allow = expected_loss_for_action(DecisionAction::Allow, 1.0, model);
        let fv = expected_loss_for_action(DecisionAction::FullValidate, 1.0, model);
        let fc = expected_loss_for_action(DecisionAction::FailClosed, 1.0, model);
        assert!(
            fc < allow,
            "fail_closed should beat allow at ~100% posterior"
        );
        assert!(
            fc < fv,
            "fail_closed should beat full_validate at ~100% posterior"
        );
    }

    #[test]
    fn expected_loss_all_equal_model_is_constant() {
        // If all loss cells are identical, all actions produce the same expected loss.
        let model = DecisionLossModel {
            allow_if_compatible: 5.0,
            allow_if_incompatible: 5.0,
            full_validate_if_compatible: 5.0,
            full_validate_if_incompatible: 5.0,
            fail_closed_if_compatible: 5.0,
            fail_closed_if_incompatible: 5.0,
        };
        for p in [0.0, 0.5, 1.0] {
            let allow = expected_loss_for_action(DecisionAction::Allow, p, model);
            let fv = expected_loss_for_action(DecisionAction::FullValidate, p, model);
            let fc = expected_loss_for_action(DecisionAction::FailClosed, p, model);
            assert!((allow - 5.0).abs() < 0.01, "all-5 model: allow={allow}");
            assert!((fv - 5.0).abs() < 0.01, "all-5 model: fv={fv}");
            assert!((fc - 5.0).abs() < 0.01, "all-5 model: fc={fc}");
        }
    }

    #[test]
    fn posterior_monotonicity_across_risk_values() {
        // For each compatibility class, posterior should increase monotonically
        // with risk_score when threshold is fixed.
        let threshold = 0.5;
        for class in [
            CompatibilityClass::KnownCompatible,
            CompatibilityClass::KnownIncompatible,
            CompatibilityClass::Unknown,
        ] {
            let mut prev = 0.0;
            for risk_pct in 1..=9 {
                let risk = risk_pct as f64 / 10.0;
                let (p, _) = posterior_incompatibility(class, risk, threshold);
                assert!(
                    p >= prev,
                    "posterior should be monotonically non-decreasing with risk for {class:?}: \
                     at risk={risk}, posterior {p} < prev {prev}"
                );
                prev = p;
            }
        }
    }

    #[test]
    fn posterior_prior_dominance_at_equal_risk_threshold() {
        // When risk == threshold, the risk_margin_llr is 0, so posterior is
        // driven entirely by the class prior.
        let (p_compat, _) =
            posterior_incompatibility(CompatibilityClass::KnownCompatible, 0.5, 0.5);
        let (p_unknown, _) = posterior_incompatibility(CompatibilityClass::Unknown, 0.5, 0.5);
        let (p_incompat, _) =
            posterior_incompatibility(CompatibilityClass::KnownIncompatible, 0.5, 0.5);
        assert!(
            p_compat < p_unknown,
            "KnownCompatible prior (0.01) should yield lower posterior than Unknown (0.5)"
        );
        assert!(
            p_unknown < p_incompat,
            "Unknown prior (0.5) should yield lower posterior than KnownIncompatible (0.99)"
        );
    }

    #[test]
    fn decide_compatibility_full_grid() {
        // Exhaustive decision matrix: every (mode, class) combination.
        let cases = [
            (
                RuntimeMode::Strict,
                CompatibilityClass::KnownCompatible,
                0.1,
                0.5,
                DecisionAction::Allow,
            ),
            (
                RuntimeMode::Strict,
                CompatibilityClass::KnownCompatible,
                0.9,
                0.5,
                DecisionAction::Allow,
            ), // strict always allows KC
            (
                RuntimeMode::Strict,
                CompatibilityClass::KnownIncompatible,
                0.1,
                0.5,
                DecisionAction::FailClosed,
            ),
            (
                RuntimeMode::Strict,
                CompatibilityClass::Unknown,
                0.1,
                0.5,
                DecisionAction::FailClosed,
            ),
            (
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                0.1,
                0.5,
                DecisionAction::Allow,
            ),
            (
                RuntimeMode::Hardened,
                CompatibilityClass::KnownCompatible,
                0.9,
                0.5,
                DecisionAction::FullValidate,
            ),
            (
                RuntimeMode::Hardened,
                CompatibilityClass::KnownIncompatible,
                0.1,
                0.5,
                DecisionAction::FailClosed,
            ),
            (
                RuntimeMode::Hardened,
                CompatibilityClass::Unknown,
                0.1,
                0.5,
                DecisionAction::FailClosed,
            ),
        ];
        for (mode, class, risk, threshold, expected) in cases {
            let got = decide_compatibility(mode, class, risk, threshold);
            assert_eq!(
                got, expected,
                "mode={:?} class={:?} risk={risk} threshold={threshold}: expected {expected:?}, got {got:?}",
                mode, class
            );
        }
    }

    #[test]
    fn ledger_empty_last_returns_none() {
        let ledger = EvidenceLedger::new();
        assert!(ledger.last().is_none());
        assert!(ledger.events().is_empty());
    }

    #[test]
    fn decide_and_record_captures_all_loss_components() {
        let mut ledger = EvidenceLedger::new();
        decide_and_record(
            &mut ledger,
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            0.8,
            0.5,
            "high risk scenario",
        );
        let event = ledger.last().unwrap();
        // All loss components should be finite and non-negative.
        assert!(event.expected_loss_allow >= 0.0);
        assert!(event.expected_loss_full_validate >= 0.0);
        assert!(event.expected_loss_fail_closed >= 0.0);
        assert!(event.expected_loss_allow.is_finite());
        assert!(event.expected_loss_full_validate.is_finite());
        assert!(event.expected_loss_fail_closed.is_finite());
        // The selected loss should match the action chosen.
        assert_eq!(event.action, DecisionAction::FullValidate);
        assert!(
            (event.selected_expected_loss - event.expected_loss_full_validate).abs() < 1e-12,
            "selected loss must match the action's loss"
        );
    }

    #[test]
    fn override_with_empty_allowlist_always_denies() {
        let event = evaluate_policy_override(
            RuntimeMode::Hardened,
            CompatibilityClass::KnownCompatible,
            "any_class",
            &[],
            "FNP-P2C-001",
            "admin",
            "test",
        );
        assert!(!event.approved);
        assert_eq!(event.action, DecisionAction::FailClosed);
    }
}
