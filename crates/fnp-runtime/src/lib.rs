#![forbid(unsafe_code)]

use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    Strict,
    Hardened,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityClass {
    KnownCompatible,
    KnownIncompatible,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionAction {
    Allow,
    FullValidate,
    FailClosed,
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
    pub note: String,
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

pub fn decide_and_record(
    ledger: &mut EvidenceLedger,
    mode: RuntimeMode,
    class: CompatibilityClass,
    risk_score: f64,
    hardened_validation_threshold: f64,
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
        note: note.into(),
    });
    action
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
        CompatibilityClass, DecisionAction, DecisionLossModel, EvidenceLedger, RuntimeMode,
        decide_and_record, decide_compatibility, expected_loss_for_action,
        posterior_incompatibility,
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
