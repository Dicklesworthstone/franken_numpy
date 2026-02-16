#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    Strict,
    Hardened,
}

impl RuntimeMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Hardened => "hardened",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferClass {
    Contiguous,
    Strided,
    StridedCast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapAction {
    NoCopy,
    ForwardCopy,
    BackwardCopy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferSelectorInput {
    pub src_stride: isize,
    pub dst_stride: isize,
    pub item_size: usize,
    pub element_count: usize,
    pub aligned: bool,
    pub cast_is_lossless: bool,
    pub same_value_cast: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NditerTransferFlags {
    pub copy_if_overlap: bool,
    pub no_broadcast: bool,
    pub observed_overlap: bool,
    pub observed_broadcast: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlatIterIndex {
    Single(usize),
    Slice {
        start: usize,
        stop: usize,
        step: usize,
    },
    Fancy(Vec<usize>),
    BoolMask(Vec<bool>),
}

pub const TRANSFER_PACKET_REASON_CODES: [&str; 10] = [
    "transfer_selector_invalid_context",
    "transfer_overlap_policy_triggered",
    "transfer_where_mask_contract_violation",
    "transfer_same_value_cast_rejected",
    "transfer_string_width_mismatch",
    "transfer_subarray_broadcast_contract_violation",
    "flatiter_transfer_read_violation",
    "flatiter_transfer_write_violation",
    "transfer_nditer_overlap_policy",
    "transfer_fpe_cast_error",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferError {
    SelectorInvalidContext(&'static str),
    OverlapPolicyTriggered(&'static str),
    SameValueCastRejected,
    FlatiterReadViolation(&'static str),
    FlatiterWriteViolation(&'static str),
    NditerOverlapPolicy(&'static str),
}

impl TransferError {
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::SelectorInvalidContext(_) => "transfer_selector_invalid_context",
            Self::OverlapPolicyTriggered(_) => "transfer_overlap_policy_triggered",
            Self::SameValueCastRejected => "transfer_same_value_cast_rejected",
            Self::FlatiterReadViolation(_) => "flatiter_transfer_read_violation",
            Self::FlatiterWriteViolation(_) => "flatiter_transfer_write_violation",
            Self::NditerOverlapPolicy(_) => "transfer_nditer_overlap_policy",
        }
    }
}

impl std::fmt::Display for TransferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SelectorInvalidContext(msg) => write!(f, "{msg}"),
            Self::OverlapPolicyTriggered(msg) => write!(f, "{msg}"),
            Self::SameValueCastRejected => write!(f, "lossy same-value cast is rejected"),
            Self::FlatiterReadViolation(msg) => write!(f, "{msg}"),
            Self::FlatiterWriteViolation(msg) => write!(f, "{msg}"),
            Self::NditerOverlapPolicy(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for TransferError {}

pub fn select_transfer_class(input: TransferSelectorInput) -> Result<TransferClass, TransferError> {
    if input.item_size == 0 {
        return Err(TransferError::SelectorInvalidContext(
            "item_size must be > 0",
        ));
    }
    if input.element_count == 0 {
        return Err(TransferError::SelectorInvalidContext(
            "element_count must be > 0",
        ));
    }
    if input.same_value_cast && !input.cast_is_lossless {
        return Err(TransferError::SameValueCastRejected);
    }

    let item_size = isize::try_from(input.item_size).map_err(|_| {
        TransferError::SelectorInvalidContext("item_size exceeds isize range for stride checks")
    })?;
    let src_multiple = input.src_stride.rem_euclid(item_size) == 0;
    let dst_multiple = input.dst_stride.rem_euclid(item_size) == 0;
    if !src_multiple || !dst_multiple {
        return Err(TransferError::SelectorInvalidContext(
            "src/dst stride must be multiples of item_size",
        ));
    }

    let src_unit = input.src_stride.unsigned_abs() == input.item_size;
    let dst_unit = input.dst_stride.unsigned_abs() == input.item_size;
    if input.aligned && src_unit && dst_unit {
        if input.cast_is_lossless {
            Ok(TransferClass::Contiguous)
        } else {
            Ok(TransferClass::StridedCast)
        }
    } else if input.cast_is_lossless {
        Ok(TransferClass::Strided)
    } else {
        Ok(TransferClass::StridedCast)
    }
}

pub fn overlap_copy_policy(
    src_offset: usize,
    dst_offset: usize,
    byte_len: usize,
) -> Result<OverlapAction, TransferError> {
    if byte_len == 0 {
        return Err(TransferError::OverlapPolicyTriggered(
            "byte_len must be > 0 for overlap policy",
        ));
    }

    let src_end = src_offset
        .checked_add(byte_len)
        .ok_or(TransferError::OverlapPolicyTriggered(
            "source range overflow in overlap policy",
        ))?;
    let dst_end = dst_offset
        .checked_add(byte_len)
        .ok_or(TransferError::OverlapPolicyTriggered(
            "destination range overflow in overlap policy",
        ))?;

    if src_end <= dst_offset || dst_end <= src_offset {
        return Ok(OverlapAction::NoCopy);
    }

    if dst_offset > src_offset {
        Ok(OverlapAction::BackwardCopy)
    } else {
        Ok(OverlapAction::ForwardCopy)
    }
}

pub fn validate_nditer_flags(flags: NditerTransferFlags) -> Result<(), TransferError> {
    if flags.no_broadcast && flags.observed_broadcast {
        return Err(TransferError::NditerOverlapPolicy(
            "no_broadcast=true with observed broadcast",
        ));
    }
    if !flags.copy_if_overlap && flags.observed_overlap {
        return Err(TransferError::NditerOverlapPolicy(
            "copy_if_overlap=false with observed overlap",
        ));
    }
    Ok(())
}

pub fn validate_flatiter_read(len: usize, index: &FlatIterIndex) -> Result<usize, TransferError> {
    count_selected_indices(len, index)
}

pub fn validate_flatiter_write(
    len: usize,
    index: &FlatIterIndex,
    values_len: usize,
) -> Result<usize, TransferError> {
    let selected = count_selected_indices(len, index).map_err(|err| match err {
        TransferError::FlatiterReadViolation(msg) => TransferError::FlatiterWriteViolation(msg),
        _ => TransferError::FlatiterWriteViolation("invalid flatiter index for write"),
    })?;
    if selected != values_len {
        return Err(TransferError::FlatiterWriteViolation(
            "values_len must match selected write lanes",
        ));
    }
    Ok(selected)
}

fn count_selected_indices(len: usize, index: &FlatIterIndex) -> Result<usize, TransferError> {
    match index {
        FlatIterIndex::Single(i) => {
            if *i >= len {
                Err(TransferError::FlatiterReadViolation(
                    "single index out of bounds",
                ))
            } else {
                Ok(1)
            }
        }
        FlatIterIndex::Slice { start, stop, step } => {
            if *step == 0 {
                return Err(TransferError::FlatiterReadViolation(
                    "slice step must be > 0",
                ));
            }
            if *start > *stop || *stop > len {
                return Err(TransferError::FlatiterReadViolation(
                    "slice bounds are invalid for flatiter",
                ));
            }
            let span = stop - start;
            Ok(span.div_ceil(*step))
        }
        FlatIterIndex::Fancy(indices) => {
            if indices.iter().any(|idx| *idx >= len) {
                Err(TransferError::FlatiterReadViolation(
                    "fancy index out of bounds",
                ))
            } else {
                Ok(indices.len())
            }
        }
        FlatIterIndex::BoolMask(mask) => {
            if mask.len() != len {
                return Err(TransferError::FlatiterReadViolation(
                    "bool mask length must equal flatiter length",
                ));
            }
            Ok(mask.iter().filter(|flag| **flag).count())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferLogRecord {
    pub fixture_id: String,
    pub seed: u64,
    pub mode: RuntimeMode,
    pub env_fingerprint: String,
    pub artifact_refs: Vec<String>,
    pub reason_code: String,
    pub passed: bool,
}

impl TransferLogRecord {
    #[must_use]
    pub fn is_replay_complete(&self) -> bool {
        !self.fixture_id.trim().is_empty()
            && !self.mode.as_str().is_empty()
            && !self.env_fingerprint.trim().is_empty()
            && !self.reason_code.trim().is_empty()
            && self
                .artifact_refs
                .iter()
                .all(|artifact| !artifact.trim().is_empty())
            && !self.artifact_refs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet003_artifacts() -> Vec<String> {
        vec![
            "artifacts/phase2c/FNP-P2C-003/fixture_manifest.json".to_string(),
            "artifacts/phase2c/FNP-P2C-003/parity_gate.yaml".to_string(),
        ]
    }

    #[test]
    fn reason_code_registry_matches_packet_contract() {
        assert_eq!(
            TRANSFER_PACKET_REASON_CODES,
            [
                "transfer_selector_invalid_context",
                "transfer_overlap_policy_triggered",
                "transfer_where_mask_contract_violation",
                "transfer_same_value_cast_rejected",
                "transfer_string_width_mismatch",
                "transfer_subarray_broadcast_contract_violation",
                "flatiter_transfer_read_violation",
                "flatiter_transfer_write_violation",
                "transfer_nditer_overlap_policy",
                "transfer_fpe_cast_error",
            ]
        );
    }

    #[test]
    fn selector_is_deterministic_for_fixed_context() {
        let input = TransferSelectorInput {
            src_stride: 8,
            dst_stride: 8,
            item_size: 8,
            element_count: 16,
            aligned: true,
            cast_is_lossless: true,
            same_value_cast: false,
        };
        let first = select_transfer_class(input).expect("selector should resolve");
        let second =
            select_transfer_class(input).expect("selector should resolve deterministically");
        assert_eq!(first, second);
        assert_eq!(first, TransferClass::Contiguous);
    }

    #[test]
    fn selector_rejects_invalid_context() {
        let err = select_transfer_class(TransferSelectorInput {
            src_stride: 8,
            dst_stride: 8,
            item_size: 0,
            element_count: 1,
            aligned: true,
            cast_is_lossless: true,
            same_value_cast: false,
        })
        .expect_err("invalid context should be rejected");
        assert_eq!(err.reason_code(), "transfer_selector_invalid_context");
    }

    #[test]
    fn selector_rejects_lossy_same_value_cast() {
        let err = select_transfer_class(TransferSelectorInput {
            src_stride: 8,
            dst_stride: 8,
            item_size: 8,
            element_count: 4,
            aligned: true,
            cast_is_lossless: false,
            same_value_cast: true,
        })
        .expect_err("lossy same-value cast should be rejected");
        assert_eq!(err.reason_code(), "transfer_same_value_cast_rejected");
    }

    #[test]
    fn overlap_policy_resolves_expected_copy_direction() {
        assert_eq!(
            overlap_copy_policy(0, 32, 8).expect("disjoint ranges"),
            OverlapAction::NoCopy
        );
        assert_eq!(
            overlap_copy_policy(0, 4, 8).expect("overlap with forward dst"),
            OverlapAction::BackwardCopy
        );
        assert_eq!(
            overlap_copy_policy(8, 4, 8).expect("overlap with backward dst"),
            OverlapAction::ForwardCopy
        );
    }

    #[test]
    fn flatiter_read_and_write_contracts_cover_nominal_and_adversarial_paths() {
        let fancy = FlatIterIndex::Fancy(vec![0, 2, 4, 6]);
        assert_eq!(
            validate_flatiter_read(8, &fancy).expect("fancy read should succeed"),
            4
        );
        assert_eq!(
            validate_flatiter_write(8, &fancy, 4).expect("fancy write should match values"),
            4
        );

        let bad_mask = FlatIterIndex::BoolMask(vec![true, false]);
        let err = validate_flatiter_read(8, &bad_mask).expect_err("mask mismatch should fail");
        assert_eq!(err.reason_code(), "flatiter_transfer_read_violation");

        let err =
            validate_flatiter_write(8, &FlatIterIndex::Single(7), 2).expect_err("arity mismatch");
        assert_eq!(err.reason_code(), "flatiter_transfer_write_violation");
    }

    #[test]
    fn nditer_flags_enforce_overlap_and_broadcast_policy() {
        validate_nditer_flags(NditerTransferFlags {
            copy_if_overlap: true,
            no_broadcast: false,
            observed_overlap: true,
            observed_broadcast: true,
        })
        .expect("valid policy combination");

        let err = validate_nditer_flags(NditerTransferFlags {
            copy_if_overlap: false,
            no_broadcast: false,
            observed_overlap: true,
            observed_broadcast: false,
        })
        .expect_err("overlap without copy_if_overlap should fail");
        assert_eq!(err.reason_code(), "transfer_nditer_overlap_policy");

        let err = validate_nditer_flags(NditerTransferFlags {
            copy_if_overlap: true,
            no_broadcast: true,
            observed_overlap: false,
            observed_broadcast: true,
        })
        .expect_err("observed broadcast should fail when no_broadcast=true");
        assert_eq!(err.reason_code(), "transfer_nditer_overlap_policy");
    }

    #[test]
    fn selector_property_grid_is_deterministic_and_reason_code_stable() {
        let item_sizes = [1usize, 2, 4, 8];
        let stride_factors = [-2isize, -1, 1, 2, 4];
        for item_size in item_sizes {
            for src_factor in stride_factors {
                for dst_factor in stride_factors {
                    for aligned in [true, false] {
                        for cast_is_lossless in [true, false] {
                            for same_value_cast in [true, false] {
                                let item_size_isize =
                                    isize::try_from(item_size).expect("small item_size");
                                let input = TransferSelectorInput {
                                    src_stride: src_factor * item_size_isize,
                                    dst_stride: dst_factor * item_size_isize,
                                    item_size,
                                    element_count: 3,
                                    aligned,
                                    cast_is_lossless,
                                    same_value_cast,
                                };
                                let first = select_transfer_class(input);
                                let second = select_transfer_class(input);
                                assert_eq!(
                                    first, second,
                                    "selector must be deterministic input={input:?}"
                                );
                                if same_value_cast && !cast_is_lossless {
                                    let err = first.expect_err("lossy same-value cast must reject");
                                    assert_eq!(
                                        err.reason_code(),
                                        "transfer_same_value_cast_rejected"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn overlap_policy_property_grid_is_stable() {
        let offsets = [0usize, 4, 8, 16, 24];
        let lengths = [1usize, 4, 8, 16];
        for src in offsets {
            for dst in offsets {
                for len in lengths {
                    let first = overlap_copy_policy(src, dst, len).expect("policy should resolve");
                    let second = overlap_copy_policy(src, dst, len)
                        .expect("policy should resolve deterministically");
                    assert_eq!(first, second);

                    let src_end = src + len;
                    let dst_end = dst + len;
                    let overlap = src_end > dst && dst_end > src;
                    if !overlap {
                        assert_eq!(first, OverlapAction::NoCopy);
                    } else if dst > src {
                        assert_eq!(first, OverlapAction::BackwardCopy);
                    } else {
                        assert_eq!(first, OverlapAction::ForwardCopy);
                    }
                }
            }
        }
    }

    #[test]
    fn packet003_log_record_is_replay_complete() {
        let record = TransferLogRecord {
            fixture_id: "UP-003-transfer-selector-determinism".to_string(),
            seed: 3001,
            mode: RuntimeMode::Strict,
            env_fingerprint: "fnp-iter-tests".to_string(),
            artifact_refs: packet003_artifacts(),
            reason_code: "transfer_selector_invalid_context".to_string(),
            passed: true,
        };
        assert!(record.is_replay_complete());
    }

    #[test]
    fn packet003_log_record_rejects_missing_fields() {
        let missing = TransferLogRecord {
            fixture_id: String::new(),
            seed: 3002,
            mode: RuntimeMode::Hardened,
            env_fingerprint: String::new(),
            artifact_refs: Vec::new(),
            reason_code: String::new(),
            passed: false,
        };
        assert!(!missing.is_replay_complete());
    }

    #[test]
    fn packet003_reason_codes_round_trip_into_replay_logs() {
        for (idx, reason_code) in TRANSFER_PACKET_REASON_CODES.iter().enumerate() {
            let record = TransferLogRecord {
                fixture_id: format!("UP-003-{idx}"),
                seed: 4000 + u64::try_from(idx).expect("small index"),
                mode: RuntimeMode::Strict,
                env_fingerprint: "fnp-iter-tests".to_string(),
                artifact_refs: packet003_artifacts(),
                reason_code: (*reason_code).to_string(),
                passed: true,
            };
            assert!(record.is_replay_complete());
            assert_eq!(record.reason_code, *reason_code);
        }
    }
}
