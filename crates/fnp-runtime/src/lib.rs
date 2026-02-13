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
    pub fn from_wire(value: &str) -> Self {
        match value {
            "known_compatible" => Self::KnownCompatible,
            "known_incompatible" => Self::KnownIncompatible,
            _ => Self::Unknown,
        }
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
                if risk_score >= hardened_validation_threshold {
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
    let Some(mode) = RuntimeMode::from_wire(mode_raw) else {
        return DecisionAction::FailClosed;
    };
    let class = CompatibilityClass::from_wire(class_raw);
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
    }
    if context.env_fingerprint.trim().is_empty() {
        context.env_fingerprint = "unknown_env".to_string();
    }
    if context.reason_code.trim().is_empty() {
        context.reason_code = "unspecified".to_string();
    }
    context
}

fn clamp_probability(p: f64) -> f64 {
    if p.is_nan() {
        return 0.5;
    }
    p.clamp(1e-9, 1.0 - 1e-9)
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
}
