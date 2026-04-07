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
    WhereMaskContractViolation(&'static str),
    SameValueCastRejected,
    StringWidthMismatch(&'static str),
    SubarrayBroadcastContractViolation(&'static str),
    FlatiterReadViolation(&'static str),
    FlatiterWriteViolation(&'static str),
    NditerOverlapPolicy(&'static str),
    FpeCastError(&'static str),
}

impl TransferError {
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::SelectorInvalidContext(_) => "transfer_selector_invalid_context",
            Self::OverlapPolicyTriggered(_) => "transfer_overlap_policy_triggered",
            Self::WhereMaskContractViolation(_) => "transfer_where_mask_contract_violation",
            Self::SameValueCastRejected => "transfer_same_value_cast_rejected",
            Self::StringWidthMismatch(_) => "transfer_string_width_mismatch",
            Self::SubarrayBroadcastContractViolation(_) => {
                "transfer_subarray_broadcast_contract_violation"
            }
            Self::FlatiterReadViolation(_) => "flatiter_transfer_read_violation",
            Self::FlatiterWriteViolation(_) => "flatiter_transfer_write_violation",
            Self::NditerOverlapPolicy(_) => "transfer_nditer_overlap_policy",
            Self::FpeCastError(_) => "transfer_fpe_cast_error",
        }
    }
}

impl std::fmt::Display for TransferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SelectorInvalidContext(msg)
            | Self::OverlapPolicyTriggered(msg)
            | Self::WhereMaskContractViolation(msg)
            | Self::StringWidthMismatch(msg)
            | Self::SubarrayBroadcastContractViolation(msg)
            | Self::FlatiterReadViolation(msg)
            | Self::FlatiterWriteViolation(msg)
            | Self::NditerOverlapPolicy(msg)
            | Self::FpeCastError(msg) => write!(f, "{msg}"),
            Self::SameValueCastRejected => write!(f, "lossy same-value cast is rejected"),
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

    let src_unit = input.src_stride == item_size;
    let dst_unit = input.dst_stride == item_size;
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

    // Disjoint ranges
    if src_end <= dst_offset || dst_end <= src_offset {
        return Ok(OverlapAction::NoCopy);
    }

    // Overlapping ranges: check for simple unit-stride cases
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
    if values_len != 1 && selected != values_len {
        return Err(TransferError::FlatiterWriteViolation(
            "values_len must be scalar or match selected write lanes",
        ));
    }
    Ok(selected)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlatIterContractError {
    IndexingViolation(&'static str),
}

impl FlatIterContractError {
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        "flatiter_indexing_contract_violation"
    }
}

impl std::fmt::Display for FlatIterContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexingViolation(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for FlatIterContractError {}

pub fn resolve_flatiter_indices(
    len: usize,
    index: &FlatIterIndex,
) -> Result<Vec<usize>, FlatIterContractError> {
    match index {
        FlatIterIndex::Single(i) => {
            if *i >= len {
                Err(FlatIterContractError::IndexingViolation(
                    "single index out of bounds",
                ))
            } else {
                Ok(vec![*i])
            }
        }
        FlatIterIndex::Slice { start, stop, step } => {
            if *step == 0 {
                return Err(FlatIterContractError::IndexingViolation(
                    "slice step must be > 0",
                ));
            }
            if *start > *stop || *stop > len {
                return Err(FlatIterContractError::IndexingViolation(
                    "slice bounds are invalid for flatiter",
                ));
            }
            Ok((*start..*stop).step_by(*step).collect())
        }
        FlatIterIndex::Fancy(indices) => {
            if indices.iter().any(|idx| *idx >= len) {
                Err(FlatIterContractError::IndexingViolation(
                    "fancy index out of bounds",
                ))
            } else {
                Ok(indices.clone())
            }
        }
        FlatIterIndex::BoolMask(mask) => {
            if mask.len() != len {
                return Err(FlatIterContractError::IndexingViolation(
                    "bool mask length must equal flatiter length",
                ));
            }
            Ok(mask
                .iter()
                .enumerate()
                .filter_map(|(idx, &selected)| selected.then_some(idx))
                .collect())
        }
    }
}

pub fn read_flatiter<T: Copy>(
    values: &[T],
    index: &FlatIterIndex,
) -> Result<Vec<T>, FlatIterContractError> {
    let indices = resolve_flatiter_indices(values.len(), index)?;
    Ok(indices.into_iter().map(|idx| values[idx]).collect())
}

pub fn write_flatiter<T: Copy>(
    values: &mut [T],
    writeable: bool,
    index: &FlatIterIndex,
    assignment: &[T],
) -> Result<usize, FlatIterContractError> {
    if !writeable {
        return Err(FlatIterContractError::IndexingViolation(
            "flatiter write requires a writeable underlying array",
        ));
    }

    let indices = resolve_flatiter_indices(values.len(), index)?;
    let selected = indices.len();
    if assignment.len() != 1 && assignment.len() != selected {
        return Err(FlatIterContractError::IndexingViolation(
            "assignment must be scalar or match selected lane count",
        ));
    }

    if assignment.len() == 1 {
        for idx in indices {
            values[idx] = assignment[0];
        }
        return Ok(selected);
    }

    for (idx, value) in indices.into_iter().zip(assignment.iter().copied()) {
        values[idx] = value;
    }
    Ok(selected)
}

pub fn ndindex(shape: &[usize]) -> Result<Vec<Vec<usize>>, NditerError> {
    let total = ndindex_element_count(shape)?;
    if shape.is_empty() {
        return Ok(vec![Vec::new()]);
    }
    if total == 0 {
        return Ok(Vec::new());
    }

    let mut indices = Vec::with_capacity(total);
    let mut current = vec![0; shape.len()];
    loop {
        indices.push(current.clone());

        let mut advanced = false;
        for axis in (0..shape.len()).rev() {
            current[axis] += 1;
            if current[axis] < shape[axis] {
                advanced = true;
                break;
            }
            current[axis] = 0;
        }

        if !advanced {
            break;
        }
    }

    Ok(indices)
}

pub fn ndenumerate<T: Copy>(
    shape: &[usize],
    values: &[T],
) -> Result<Vec<(Vec<usize>, T)>, NditerError> {
    let total = ndindex_element_count(shape)?;
    if values.len() != total {
        return Err(NditerError::NdindexShapeValidation(
            "ndenumerate values length must match shape element count",
        ));
    }

    Ok(ndindex(shape)?
        .into_iter()
        .zip(values.iter().copied())
        .collect())
}

fn count_selected_indices(len: usize, index: &FlatIterIndex) -> Result<usize, TransferError> {
    match index {
        FlatIterIndex::BoolMask(mask) => {
            if mask.len() != len {
                return Err(TransferError::FlatiterReadViolation(
                    "bool mask length must equal flatiter length",
                ));
            }
            Ok(count_true_mask(mask))
        }
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
            if *start == *stop {
                Ok(0)
            } else {
                Ok((*stop - *start - 1) / *step + 1)
            }
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
    }
}

#[must_use]
fn count_true_mask(mask: &[bool]) -> usize {
    let mut count = 0usize;
    let mut chunks = mask.chunks_exact(128);
    for chunk in &mut chunks {
        for &b in chunk {
            if b {
                count += 1;
            }
        }
    }
    for &b in chunks.remainder() {
        if b {
            count += 1;
        }
    }
    count
}

fn ndindex_element_count(shape: &[usize]) -> Result<usize, NditerError> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or(NditerError::NdindexShapeValidation(
                "ndindex shape element count overflow",
            ))
    })
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NditerOrder {
    C,
    F,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NditerOptions {
    pub order: NditerOrder,
    pub external_loop: bool,
}

impl Default for NditerOptions {
    fn default() -> Self {
        Self {
            order: NditerOrder::C,
            external_loop: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NditerPlan {
    shape: Vec<usize>,
    compatible_strides: Vec<isize>,
    order: NditerOrder,
    external_loop: bool,
    iteration_shape: Vec<usize>,
    inner_loop_axis: Option<usize>,
    inner_loop_len: usize,
    element_count: usize,
}

impl NditerPlan {
    pub fn new(
        shape: Vec<usize>,
        item_size: usize,
        options: NditerOptions,
    ) -> Result<Self, NditerError> {
        if item_size == 0 {
            return Err(NditerError::InvalidConfiguration(
                "item_size must be > 0 for nditer planning",
            ));
        }

        let element_count = checked_element_count(&shape)?;
        let compatible_strides = compatible_nditer_strides(&shape, item_size, options.order)?;
        let (iteration_shape, inner_loop_axis, inner_loop_len) =
            plan_external_loop(&shape, options);

        Ok(Self {
            shape,
            compatible_strides,
            order: options.order,
            external_loop: options.external_loop,
            iteration_shape,
            inner_loop_axis,
            inner_loop_len,
            element_count,
        })
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn compatible_strides(&self) -> &[isize] {
        &self.compatible_strides
    }

    #[must_use]
    pub fn order(&self) -> NditerOrder {
        self.order
    }

    #[must_use]
    pub fn external_loop(&self) -> bool {
        self.external_loop
    }

    #[must_use]
    pub fn iteration_shape(&self) -> &[usize] {
        &self.iteration_shape
    }

    #[must_use]
    pub fn inner_loop_axis(&self) -> Option<usize> {
        self.inner_loop_axis
    }

    #[must_use]
    pub fn inner_loop_len(&self) -> usize {
        self.inner_loop_len
    }

    #[must_use]
    pub fn element_count(&self) -> usize {
        self.element_count
    }

    pub fn linear_index_to_multi_index(
        &self,
        linear_index: usize,
    ) -> Result<Vec<usize>, NditerError> {
        if self.shape.is_empty() {
            return if linear_index == 0 {
                Ok(Vec::new())
            } else {
                Err(NditerError::MultiIndexViolation(
                    "scalar nditer only accepts linear index 0",
                ))
            };
        }
        if linear_index >= self.element_count {
            return Err(NditerError::MultiIndexViolation(
                "linear index out of bounds for nditer shape",
            ));
        }

        let mut remainder = linear_index;
        let mut multi = vec![0usize; self.shape.len()];
        match self.order {
            NditerOrder::C => {
                for axis in (0..self.shape.len()).rev() {
                    let dim = self.shape[axis];
                    if dim == 0 {
                        multi[axis] = 0;
                    } else {
                        multi[axis] = remainder % dim;
                        remainder /= dim;
                    }
                }
            }
            NditerOrder::F => {
                for (axis, dim) in self.shape.iter().copied().enumerate() {
                    if dim == 0 {
                        multi[axis] = 0;
                    } else {
                        multi[axis] = remainder % dim;
                        remainder /= dim;
                    }
                }
            }
        }
        Ok(multi)
    }

    pub fn multi_index_to_linear_index(&self, multi_index: &[usize]) -> Result<usize, NditerError> {
        if multi_index.len() != self.shape.len() {
            return Err(NditerError::MultiIndexViolation(
                "multi-index rank must match nditer rank",
            ));
        }

        if self.shape.is_empty() {
            return Ok(0);
        }

        let mut linear = 0usize;
        match self.order {
            NditerOrder::C => {
                for (axis, &idx) in multi_index.iter().enumerate() {
                    let dim = self.shape[axis];
                    if idx >= dim {
                        return Err(NditerError::MultiIndexViolation(
                            "multi-index component out of bounds",
                        ));
                    }
                    linear = linear
                        .checked_mul(dim)
                        .and_then(|value| value.checked_add(idx))
                        .ok_or(NditerError::InvalidConfiguration(
                            "linear index overflow while seeking multi-index",
                        ))?;
                }
            }
            NditerOrder::F => {
                let mut stride = 1usize;
                for (axis, &idx) in multi_index.iter().enumerate() {
                    let dim = self.shape[axis];
                    if idx >= dim {
                        return Err(NditerError::MultiIndexViolation(
                            "multi-index component out of bounds",
                        ));
                    }
                    linear = linear
                        .checked_add(idx.checked_mul(stride).ok_or(
                            NditerError::InvalidConfiguration(
                                "linear index overflow while seeking multi-index",
                            ),
                        )?)
                        .ok_or(NditerError::InvalidConfiguration(
                            "linear index overflow while seeking multi-index",
                        ))?;
                    stride = stride
                        .checked_mul(dim)
                        .ok_or(NditerError::InvalidConfiguration(
                            "linear index overflow while seeking multi-index",
                        ))?;
                }
            }
        }

        Ok(linear)
    }

    pub fn seek_multi_index(&self, multi_index: &[usize]) -> Result<usize, NditerError> {
        self.multi_index_to_linear_index(multi_index)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NditerOperandSpec {
    pub shape: Vec<usize>,
    pub no_broadcast: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NditerOperandBroadcastPlan {
    pub shape: Vec<usize>,
    pub broadcast_strides: Vec<isize>,
    pub expanded_axes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NditerBroadcastPlan {
    pub broadcast_shape: Vec<usize>,
    pub operands: Vec<NditerOperandBroadcastPlan>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NditerOverlapPolicyInput {
    pub copy_if_overlap: bool,
    pub overlap_assume_elementwise: bool,
    pub observed_overlap: bool,
    pub src_offset: usize,
    pub dst_offset: usize,
    pub byte_len: usize,
    pub mode: RuntimeMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NditerOverlapDecision {
    pub overlap_action: OverlapAction,
    pub used_elementwise_assumption: bool,
    pub audit_required: bool,
}

pub fn plan_nditer_broadcast(
    operands: &[NditerOperandSpec],
    item_size: usize,
    order: NditerOrder,
) -> Result<NditerBroadcastPlan, NditerError> {
    if operands.is_empty() {
        return Err(NditerError::InvalidConfiguration(
            "nditer broadcast planner requires at least one operand",
        ));
    }

    let mut broadcast_shape = Vec::<usize>::new();
    for operand in operands {
        broadcast_shape = merge_broadcast_shape(&broadcast_shape, &operand.shape)?;
    }

    let mut operand_plans = Vec::with_capacity(operands.len());
    for operand in operands {
        let base_strides = compatible_nditer_strides(&operand.shape, item_size, order)?;
        let rank_delta = broadcast_shape.len().saturating_sub(operand.shape.len());
        let mut broadcast_strides = Vec::with_capacity(broadcast_shape.len());
        let mut expanded_axes = Vec::new();

        for (axis, &broadcast_dim) in broadcast_shape.iter().enumerate() {
            let aligned_axis = axis.checked_sub(rank_delta);
            let (operand_dim, operand_stride) =
                aligned_axis.map_or((1, 0), |idx| (operand.shape[idx], base_strides[idx]));

            if operand_dim == broadcast_dim {
                broadcast_strides.push(operand_stride);
                continue;
            }

            if operand_dim == 1 && broadcast_dim > 1 {
                if operand.no_broadcast {
                    return Err(NditerError::NoBroadcastViolation(
                        "no_broadcast operand would need axis expansion",
                    ));
                }
                broadcast_strides.push(0);
                expanded_axes.push(axis);
                continue;
            }

            return Err(NditerError::InvalidConfiguration(
                "operand shape is not broadcast-compatible with iterator shape",
            ));
        }

        operand_plans.push(NditerOperandBroadcastPlan {
            shape: operand.shape.clone(),
            broadcast_strides,
            expanded_axes,
        });
    }

    Ok(NditerBroadcastPlan {
        broadcast_shape,
        operands: operand_plans,
    })
}

pub fn resolve_nditer_overlap_policy(
    input: NditerOverlapPolicyInput,
) -> Result<NditerOverlapDecision, NditerError> {
    if !input.observed_overlap {
        return Ok(NditerOverlapDecision {
            overlap_action: OverlapAction::NoCopy,
            used_elementwise_assumption: false,
            audit_required: false,
        });
    }

    if input.overlap_assume_elementwise {
        return Ok(NditerOverlapDecision {
            overlap_action: OverlapAction::NoCopy,
            used_elementwise_assumption: true,
            audit_required: input.mode == RuntimeMode::Hardened,
        });
    }

    if !input.copy_if_overlap {
        return Err(NditerError::OverlapPolicyTriggered(
            "observed overlap requires copy_if_overlap or overlap_assume_elementwise",
        ));
    }

    let overlap_action = overlap_copy_policy(input.src_offset, input.dst_offset, input.byte_len)
        .map_err(|err| match err {
            TransferError::OverlapPolicyTriggered(msg) => NditerError::OverlapPolicyTriggered(msg),
            _ => NditerError::OverlapPolicyTriggered(
                "unexpected transfer overlap policy error in nditer",
            ),
        })?;
    Ok(NditerOverlapDecision {
        overlap_action,
        used_elementwise_assumption: false,
        audit_required: input.mode == RuntimeMode::Hardened,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NditerError {
    InvalidConfiguration(&'static str),
    MultiIndexViolation(&'static str),
    NoBroadcastViolation(&'static str),
    OverlapPolicyTriggered(&'static str),
    NdindexShapeValidation(&'static str),
}

impl NditerError {
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::InvalidConfiguration(_) => "nditer_constructor_invalid_configuration",
            Self::MultiIndexViolation(_) => "nditer_multi_index_contract_violation",
            Self::NoBroadcastViolation(_) => "nditer_no_broadcast_violation",
            Self::OverlapPolicyTriggered(_) => "nditer_overlap_policy_triggered",
            Self::NdindexShapeValidation(_) => "ndindex_shape_validation_failed",
        }
    }
}

impl std::fmt::Display for NditerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(msg)
            | Self::MultiIndexViolation(msg)
            | Self::NoBroadcastViolation(msg)
            | Self::OverlapPolicyTriggered(msg)
            | Self::NdindexShapeValidation(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for NditerError {}

fn checked_element_count(shape: &[usize]) -> Result<usize, NditerError> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or(NditerError::InvalidConfiguration(
                "shape element count overflow in nditer planning",
            ))
    })
}

fn compatible_nditer_strides(
    shape: &[usize],
    item_size: usize,
    order: NditerOrder,
) -> Result<Vec<isize>, NditerError> {
    let item_size_isize = isize::try_from(item_size).map_err(|_| {
        NditerError::InvalidConfiguration("item_size exceeds isize range for stride planning")
    })?;

    if shape.is_empty() {
        return Ok(Vec::new());
    }

    let mut strides = vec![0isize; shape.len()];
    let mut stride = item_size_isize;
    match order {
        NditerOrder::C => {
            for axis in (0..shape.len()).rev() {
                strides[axis] = stride;
                let dim = isize::try_from(shape[axis]).map_err(|_| {
                    NditerError::InvalidConfiguration(
                        "shape dimension exceeds isize range for stride planning",
                    )
                })?;
                stride =
                    stride
                        .checked_mul(dim.max(1))
                        .ok_or(NditerError::InvalidConfiguration(
                            "compatible stride overflow in C-order nditer planning",
                        ))?;
            }
        }
        NditerOrder::F => {
            for (axis, &extent) in shape.iter().enumerate() {
                strides[axis] = stride;
                let dim = isize::try_from(extent).map_err(|_| {
                    NditerError::InvalidConfiguration(
                        "shape dimension exceeds isize range for stride planning",
                    )
                })?;
                stride =
                    stride
                        .checked_mul(dim.max(1))
                        .ok_or(NditerError::InvalidConfiguration(
                            "compatible stride overflow in F-order nditer planning",
                        ))?;
            }
        }
    }
    Ok(strides)
}

fn merge_broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, NditerError> {
    let rank = lhs.len().max(rhs.len());
    let mut merged = Vec::with_capacity(rank);
    for axis in 0..rank {
        let lhs_dim = if axis < rank - lhs.len() {
            1
        } else {
            lhs[axis - (rank - lhs.len())]
        };
        let rhs_dim = if axis < rank - rhs.len() {
            1
        } else {
            rhs[axis - (rank - rhs.len())]
        };

        let merged_dim = if lhs_dim == rhs_dim {
            lhs_dim
        } else if lhs_dim == 1 {
            rhs_dim
        } else if rhs_dim == 1 {
            lhs_dim
        } else {
            return Err(NditerError::InvalidConfiguration(
                "operand shapes are not broadcast-compatible",
            ));
        };
        merged.push(merged_dim);
    }
    Ok(merged)
}

fn plan_external_loop(
    shape: &[usize],
    options: NditerOptions,
) -> (Vec<usize>, Option<usize>, usize) {
    if !options.external_loop || shape.is_empty() {
        return (shape.to_vec(), None, 1);
    }

    let axis = match options.order {
        NditerOrder::C => shape.len() - 1,
        NditerOrder::F => 0,
    };
    let inner_loop_len = shape[axis];
    let iteration_shape = shape
        .iter()
        .enumerate()
        .filter_map(|(idx, &dim)| (idx != axis).then_some(dim))
        .collect();
    (iteration_shape, Some(axis), inner_loop_len)
}

// ---------------------------------------------------------------------------
// Transfer-loop selector state machine (P2C003-R01 through R10)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferLoopClass {
    /// Memcpy-eligible: contiguous, aligned, no cast.
    SimpleContiguous,
    /// Unit-stride with lossless cast (e.g. widening int to float).
    ContiguousCast,
    /// Non-unit stride, no cast needed.
    StridedNoCast,
    /// Non-unit stride with lossless cast.
    StridedWithCast,
    /// Where-mask gated transfer (writes only on true lanes).
    MaskedTransfer,
    /// Subarray or grouped element transfer (`1->1`, `n->n`).
    SubarrayGrouped,
    /// Fixed-width string/unicode pad/truncate/copyswap path.
    StringFixedWidth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDtypeRelation {
    /// Source and destination dtypes are identical (no cast).
    Same,
    /// Lossless widening cast (e.g. i32 -> i64, f32 -> f64).
    LosslessCast,
    /// Lossy narrowing cast (e.g. f64 -> f32, i64 -> i32).
    LossyCast,
    /// Fixed-width string/unicode requiring pad/truncate.
    StringWidthChange { src_width: usize, dst_width: usize },
    /// Subarray broadcast cast (grouped element transfer).
    SubarrayBroadcast,
}

/// Floating-point exception status flags observed during a cast transfer.
///
/// These mirror the NumPy `NPY_FPE_*` flag family and are used by P2C003-R10
/// to enforce deterministic FPE reporting semantics after transfer loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FpeStatus {
    pub overflow: bool,
    pub underflow: bool,
    pub invalid: bool,
    pub divide_by_zero: bool,
}

impl FpeStatus {
    /// Returns `true` if any FPE flag is set.
    #[must_use]
    pub fn any_set(self) -> bool {
        self.overflow || self.underflow || self.invalid || self.divide_by_zero
    }

    /// Returns a stable reason description for the first set flag.
    #[must_use]
    pub fn first_flag_name(self) -> Option<&'static str> {
        if self.invalid {
            Some("invalid")
        } else if self.divide_by_zero {
            Some("divide_by_zero")
        } else if self.overflow {
            Some("overflow")
        } else if self.underflow {
            Some("underflow")
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferContext {
    pub src_stride: isize,
    pub dst_stride: isize,
    pub item_size: usize,
    pub element_count: usize,
    pub aligned: bool,
    pub dtype_relation: TransferDtypeRelation,
    pub same_value_cast: bool,
    pub has_where_mask: bool,
    pub has_overlap: bool,
    pub mode: RuntimeMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferLoopDecision {
    pub loop_class: TransferLoopClass,
    pub overlap_action: OverlapAction,
    pub reason_code: &'static str,
}

/// Deterministic transfer-loop selector: maps a full transfer context into
/// a `TransferLoopDecision`. This is the P2C003-R01 contract entry point.
///
/// The selector is a pure function — identical contexts always produce
/// identical decisions.
pub fn select_transfer_loop(ctx: TransferContext) -> Result<TransferLoopDecision, TransferError> {
    // Gate: item_size and element_count must be positive.
    if ctx.item_size == 0 {
        return Err(TransferError::SelectorInvalidContext(
            "item_size must be > 0",
        ));
    }
    if ctx.element_count == 0 {
        return Err(TransferError::SelectorInvalidContext(
            "element_count must be > 0",
        ));
    }

    // Gate: same-value cast rejects lossy transforms (P2C003-R04).
    if ctx.same_value_cast {
        match ctx.dtype_relation {
            TransferDtypeRelation::LossyCast => {
                return Err(TransferError::SameValueCastRejected);
            }
            TransferDtypeRelation::StringWidthChange {
                src_width,
                dst_width,
            } if dst_width < src_width => {
                return Err(TransferError::SameValueCastRejected);
            }
            _ => {}
        }
    }

    // Gate: stride alignment validation.
    let item_size_isize = isize::try_from(ctx.item_size)
        .map_err(|_| TransferError::SelectorInvalidContext("item_size exceeds isize range"))?;
    if item_size_isize != 0 {
        let src_ok = ctx.src_stride.rem_euclid(item_size_isize) == 0;
        let dst_ok = ctx.dst_stride.rem_euclid(item_size_isize) == 0;
        if !src_ok || !dst_ok {
            return Err(TransferError::SelectorInvalidContext(
                "src/dst stride must be multiples of item_size",
            ));
        }
    }

    // Resolve overlap action.
    let overlap_action = if ctx.has_overlap {
        let byte_len = ctx.element_count.checked_mul(ctx.item_size).ok_or(
            TransferError::OverlapPolicyTriggered("byte length overflow in overlap check"),
        )?;
        if byte_len == 0 {
            OverlapAction::NoCopy
        } else {
            let same_sign = (ctx.dst_stride >= 0 && ctx.src_stride >= 0)
                || (ctx.dst_stride <= 0 && ctx.src_stride <= 0);
            if same_sign {
                if ctx.dst_stride.unsigned_abs() > ctx.src_stride.unsigned_abs() {
                    OverlapAction::BackwardCopy
                } else {
                    OverlapAction::ForwardCopy
                }
            } else {
                // Mixed signs - crossing overlap is unsafe for simple loops.
                return Err(TransferError::OverlapPolicyTriggered(
                    "crossing overlap (mixed-sign strides) is unsupported for simple loops",
                ));
            }
        }
    } else {
        OverlapAction::NoCopy
    };

    // Select loop class based on dtype relation and stride geometry.
    let loop_class = match ctx.dtype_relation {
        // Where-mask takes priority when present (P2C003-R03).
        _ if ctx.has_where_mask => TransferLoopClass::MaskedTransfer,

        // String fixed-width path (P2C003-R05).
        TransferDtypeRelation::StringWidthChange {
            src_width,
            dst_width,
        } => {
            if src_width == 0 || dst_width == 0 {
                return Err(TransferError::StringWidthMismatch(
                    "string width must be > 0",
                ));
            }
            TransferLoopClass::StringFixedWidth
        }

        // Subarray grouped path (P2C003-R06).
        TransferDtypeRelation::SubarrayBroadcast => TransferLoopClass::SubarrayGrouped,

        // Same dtype, no cast needed.
        TransferDtypeRelation::Same => {
            let src_unit = ctx.src_stride == item_size_isize;
            let dst_unit = ctx.dst_stride == item_size_isize;
            if ctx.aligned && src_unit && dst_unit {
                TransferLoopClass::SimpleContiguous
            } else {
                TransferLoopClass::StridedNoCast
            }
        }

        // Lossless cast.
        TransferDtypeRelation::LosslessCast => {
            let src_unit = ctx.src_stride == item_size_isize;
            let dst_unit = ctx.dst_stride == item_size_isize;
            if ctx.aligned && src_unit && dst_unit {
                TransferLoopClass::ContiguousCast
            } else {
                TransferLoopClass::StridedWithCast
            }
        }

        // Lossy cast (hardened mode may full-validate).
        TransferDtypeRelation::LossyCast => TransferLoopClass::StridedWithCast,
    };

    // Hardened mode: fail-closed for unknown combinations that might be risky.
    if ctx.mode == RuntimeMode::Hardened && ctx.has_overlap {
        match loop_class {
            TransferLoopClass::SubarrayGrouped => {
                return Err(TransferError::SubarrayBroadcastContractViolation(
                    "hardened mode rejects overlapping subarray transfers",
                ));
            }
            TransferLoopClass::StringFixedWidth => {
                return Err(TransferError::StringWidthMismatch(
                    "hardened mode rejects overlapping string transfers",
                ));
            }
            _ => {}
        }
    }

    Ok(TransferLoopDecision {
        loop_class,
        overlap_action,
        reason_code: transfer_loop_class_reason(loop_class),
    })
}

/// Maps a resolved loop class to the contract family reason code for audit trail.
///
/// For successful resolutions, this identifies which P2C003 contract row governs
/// the selected transfer path. This is NOT an error code — it's the contract
/// family tag used by the evidence ledger.
fn transfer_loop_class_reason(class: TransferLoopClass) -> &'static str {
    match class {
        TransferLoopClass::SimpleContiguous
        | TransferLoopClass::ContiguousCast
        | TransferLoopClass::StridedNoCast
        | TransferLoopClass::StridedWithCast => "transfer_selector_resolved",
        TransferLoopClass::MaskedTransfer => "transfer_where_mask_resolved",
        TransferLoopClass::SubarrayGrouped => "transfer_subarray_resolved",
        TransferLoopClass::StringFixedWidth => "transfer_string_width_resolved",
    }
}

/// Validate a where-mask against the transfer shape (P2C003-R03).
///
/// Ensures the mask length matches the element count and that masked
/// transfer semantics are enforceable.
pub fn validate_where_mask(mask: &[bool], element_count: usize) -> Result<usize, TransferError> {
    if mask.len() != element_count {
        return Err(TransferError::WhereMaskContractViolation(
            "where-mask length must equal element count",
        ));
    }
    Ok(count_true_mask(mask))
}

/// Validate subarray broadcast transfer dimensions (P2C003-R06).
///
/// For `n->n` grouped transfers, source and destination subarray
/// element counts must match. For broadcast transfers, the destination
/// count must be a multiple of the source count.
pub fn validate_subarray_transfer(
    src_subarray_len: usize,
    dst_subarray_len: usize,
) -> Result<(), TransferError> {
    if src_subarray_len == 0 || dst_subarray_len == 0 {
        return Err(TransferError::SubarrayBroadcastContractViolation(
            "subarray length must be > 0",
        ));
    }
    if src_subarray_len != dst_subarray_len && !dst_subarray_len.is_multiple_of(src_subarray_len) {
        return Err(TransferError::SubarrayBroadcastContractViolation(
            "dst subarray length must equal or be a multiple of src subarray length",
        ));
    }
    Ok(())
}

/// Validate post-transfer FPE status (P2C003-R10).
///
/// In strict mode, FPE flags are reported but do not block the transfer.
/// In hardened mode, any FPE flag triggers a fail-closed error with a
/// deterministic reason code.
pub fn validate_post_transfer_fpe(
    fpe: FpeStatus,
    mode: RuntimeMode,
) -> Result<FpeStatus, TransferError> {
    if mode == RuntimeMode::Hardened && fpe.any_set() {
        let flag_name = fpe.first_flag_name().unwrap_or("unknown");
        return Err(match flag_name {
            "invalid" => TransferError::FpeCastError("cast produced invalid (NaN) result"),
            "divide_by_zero" => TransferError::FpeCastError("cast produced division by zero"),
            "overflow" => TransferError::FpeCastError("cast produced overflow"),
            "underflow" => TransferError::FpeCastError("cast produced underflow"),
            _ => TransferError::FpeCastError("cast produced unknown FPE"),
        });
    }
    Ok(fpe)
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
        let err = overlap_copy_policy(0, 0, 0).expect_err("zero length should fail");
        assert_eq!(err.reason_code(), "transfer_overlap_policy_triggered");
        assert_eq!(
            overlap_copy_policy(0, 0, 8).expect("identity overlap"),
            OverlapAction::ForwardCopy
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
    fn bool_mask_count_matches_reference() {
        let masks = vec![
            vec![false; 64],
            vec![true; 64],
            (0..257).map(|idx| idx % 3 == 0).collect::<Vec<_>>(),
            (0..511).map(|idx| (idx * 17) % 11 < 5).collect::<Vec<_>>(),
        ];
        for mask in masks {
            let fast = count_true_mask(&mask);
            let reference = mask.iter().filter(|&&flag| flag).count();
            assert_eq!(fast, reference);
        }
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

    #[test]
    fn nditer_plan_c_order_reports_contiguous_strides_and_seek() {
        let plan = NditerPlan::new(
            vec![2, 3, 4],
            8,
            NditerOptions {
                order: NditerOrder::C,
                external_loop: false,
            },
        )
        .expect("plan");
        assert_eq!(plan.compatible_strides(), &[96, 32, 8]);
        assert_eq!(
            plan.linear_index_to_multi_index(17).expect("seek"),
            vec![1, 1, 1]
        );
        assert_eq!(plan.seek_multi_index(&[1, 1, 1]).expect("reverse seek"), 17);
    }

    #[test]
    fn nditer_plan_f_order_reports_contiguous_strides_and_seek() {
        let plan = NditerPlan::new(
            vec![2, 3, 4],
            8,
            NditerOptions {
                order: NditerOrder::F,
                external_loop: false,
            },
        )
        .expect("plan");
        assert_eq!(plan.compatible_strides(), &[8, 16, 48]);
        assert_eq!(
            plan.linear_index_to_multi_index(17).expect("seek"),
            vec![1, 2, 2]
        );
        assert_eq!(plan.seek_multi_index(&[1, 2, 2]).expect("reverse seek"), 17);
    }

    #[test]
    fn nditer_plan_external_loop_c_order_tracks_last_axis() {
        let plan = NditerPlan::new(
            vec![2, 3, 4],
            8,
            NditerOptions {
                order: NditerOrder::C,
                external_loop: true,
            },
        )
        .expect("plan");
        assert_eq!(plan.iteration_shape(), &[2, 3]);
        assert_eq!(plan.inner_loop_axis(), Some(2));
        assert_eq!(plan.inner_loop_len(), 4);
    }

    #[test]
    fn nditer_plan_external_loop_f_order_tracks_first_axis() {
        let plan = NditerPlan::new(
            vec![2, 3, 4],
            8,
            NditerOptions {
                order: NditerOrder::F,
                external_loop: true,
            },
        )
        .expect("plan");
        assert_eq!(plan.iteration_shape(), &[3, 4]);
        assert_eq!(plan.inner_loop_axis(), Some(0));
        assert_eq!(plan.inner_loop_len(), 2);
    }

    #[test]
    fn nditer_plan_rejects_zero_item_size() {
        let err = NditerPlan::new(vec![2, 3], 0, NditerOptions::default())
            .expect_err("zero item size should fail");
        assert_eq!(
            err.reason_code(),
            "nditer_constructor_invalid_configuration"
        );
    }

    #[test]
    fn nditer_plan_rejects_out_of_bounds_multi_index() {
        let plan = NditerPlan::new(vec![2, 3], 8, NditerOptions::default()).expect("plan");
        let err = plan
            .seek_multi_index(&[2, 0])
            .expect_err("out-of-bounds multi-index should fail");
        assert_eq!(err.reason_code(), "nditer_multi_index_contract_violation");
    }

    #[test]
    fn nditer_broadcast_plan_zero_stride_propagates_for_expanded_axes() {
        let plan = plan_nditer_broadcast(
            &[
                NditerOperandSpec {
                    shape: vec![2, 3],
                    no_broadcast: false,
                },
                NditerOperandSpec {
                    shape: vec![1, 3],
                    no_broadcast: false,
                },
            ],
            8,
            NditerOrder::C,
        )
        .expect("broadcast plan");
        assert_eq!(plan.broadcast_shape, vec![2, 3]);
        assert_eq!(plan.operands[0].broadcast_strides, vec![24, 8]);
        assert_eq!(plan.operands[0].expanded_axes, Vec::<usize>::new());
        assert_eq!(plan.operands[1].broadcast_strides, vec![0, 8]);
        assert_eq!(plan.operands[1].expanded_axes, vec![0]);
    }

    #[test]
    fn nditer_broadcast_plan_rejects_protected_operand_expansion() {
        let err = plan_nditer_broadcast(
            &[
                NditerOperandSpec {
                    shape: vec![2, 3],
                    no_broadcast: false,
                },
                NditerOperandSpec {
                    shape: vec![1, 3],
                    no_broadcast: true,
                },
            ],
            8,
            NditerOrder::C,
        )
        .expect_err("protected operand should reject broadcast expansion");
        assert_eq!(err.reason_code(), "nditer_no_broadcast_violation");
    }

    #[test]
    fn nditer_broadcast_plan_rejects_shape_mismatch() {
        let err = plan_nditer_broadcast(
            &[
                NditerOperandSpec {
                    shape: vec![2, 3],
                    no_broadcast: false,
                },
                NditerOperandSpec {
                    shape: vec![2, 4],
                    no_broadcast: false,
                },
            ],
            8,
            NditerOrder::C,
        )
        .expect_err("incompatible shapes should fail");
        assert_eq!(
            err.reason_code(),
            "nditer_constructor_invalid_configuration"
        );
    }

    #[test]
    fn nditer_overlap_policy_uses_copy_direction_when_copy_if_overlap_enabled() {
        let decision = resolve_nditer_overlap_policy(NditerOverlapPolicyInput {
            copy_if_overlap: true,
            overlap_assume_elementwise: false,
            observed_overlap: true,
            src_offset: 0,
            dst_offset: 4,
            byte_len: 8,
            mode: RuntimeMode::Strict,
        })
        .expect("copy_if_overlap should resolve");
        assert_eq!(decision.overlap_action, OverlapAction::BackwardCopy);
        assert!(!decision.used_elementwise_assumption);
        assert!(!decision.audit_required);
    }

    #[test]
    fn nditer_overlap_policy_allows_explicit_elementwise_assumption() {
        let decision = resolve_nditer_overlap_policy(NditerOverlapPolicyInput {
            copy_if_overlap: false,
            overlap_assume_elementwise: true,
            observed_overlap: true,
            src_offset: 0,
            dst_offset: 4,
            byte_len: 8,
            mode: RuntimeMode::Hardened,
        })
        .expect("elementwise assumption should bypass copy path");
        assert_eq!(decision.overlap_action, OverlapAction::NoCopy);
        assert!(decision.used_elementwise_assumption);
        assert!(decision.audit_required);
    }

    #[test]
    fn nditer_overlap_policy_rejects_unmediated_overlap() {
        let err = resolve_nditer_overlap_policy(NditerOverlapPolicyInput {
            copy_if_overlap: false,
            overlap_assume_elementwise: false,
            observed_overlap: true,
            src_offset: 0,
            dst_offset: 4,
            byte_len: 8,
            mode: RuntimeMode::Strict,
        })
        .expect_err("unmediated overlap should reject");
        assert_eq!(err.reason_code(), "nditer_overlap_policy_triggered");
    }

    // -----------------------------------------------------------------------
    // Transfer-loop selector state machine tests (P2C003-R01..R10)
    // -----------------------------------------------------------------------

    fn base_transfer_ctx() -> TransferContext {
        TransferContext {
            src_stride: 8,
            dst_stride: 8,
            item_size: 8,
            element_count: 16,
            aligned: true,
            dtype_relation: TransferDtypeRelation::Same,
            same_value_cast: false,
            has_where_mask: false,
            has_overlap: false,
            mode: RuntimeMode::Strict,
        }
    }

    #[test]
    fn transfer_loop_simple_contiguous_for_same_dtype_unit_stride() {
        let ctx = base_transfer_ctx();
        let decision = select_transfer_loop(ctx).expect("should resolve");
        assert_eq!(decision.loop_class, TransferLoopClass::SimpleContiguous);
        assert_eq!(decision.overlap_action, OverlapAction::NoCopy);
    }

    #[test]
    fn transfer_loop_strided_no_cast_for_non_unit_stride() {
        let ctx = TransferContext {
            src_stride: 16,
            dst_stride: 16,
            ..base_transfer_ctx()
        };
        let decision = select_transfer_loop(ctx).expect("should resolve");
        assert_eq!(decision.loop_class, TransferLoopClass::StridedNoCast);
    }

    #[test]
    fn transfer_loop_contiguous_cast_for_lossless_unit_stride() {
        let ctx = TransferContext {
            dtype_relation: TransferDtypeRelation::LosslessCast,
            ..base_transfer_ctx()
        };
        let decision = select_transfer_loop(ctx).expect("should resolve");
        assert_eq!(decision.loop_class, TransferLoopClass::ContiguousCast);
    }

    #[test]
    fn transfer_loop_strided_with_cast_for_lossy() {
        let ctx = TransferContext {
            dtype_relation: TransferDtypeRelation::LossyCast,
            ..base_transfer_ctx()
        };
        let decision = select_transfer_loop(ctx).expect("should resolve");
        assert_eq!(decision.loop_class, TransferLoopClass::StridedWithCast);
    }

    #[test]
    fn transfer_loop_masked_when_where_mask_present() {
        let ctx = TransferContext {
            has_where_mask: true,
            ..base_transfer_ctx()
        };
        let decision = select_transfer_loop(ctx).expect("should resolve");
        assert_eq!(decision.loop_class, TransferLoopClass::MaskedTransfer);
    }

    #[test]
    fn transfer_loop_string_fixed_width() {
        let ctx = TransferContext {
            dtype_relation: TransferDtypeRelation::StringWidthChange {
                src_width: 4,
                dst_width: 8,
            },
            ..base_transfer_ctx()
        };
        let decision = select_transfer_loop(ctx).expect("should resolve");
        assert_eq!(decision.loop_class, TransferLoopClass::StringFixedWidth);
    }

    #[test]
    fn transfer_loop_string_zero_width_rejected() {
        let ctx = TransferContext {
            dtype_relation: TransferDtypeRelation::StringWidthChange {
                src_width: 0,
                dst_width: 8,
            },
            ..base_transfer_ctx()
        };
        let err = select_transfer_loop(ctx).expect_err("zero width should fail");
        assert_eq!(err.reason_code(), "transfer_string_width_mismatch");
    }

    #[test]
    fn transfer_loop_subarray_grouped() {
        let ctx = TransferContext {
            dtype_relation: TransferDtypeRelation::SubarrayBroadcast,
            ..base_transfer_ctx()
        };
        let decision = select_transfer_loop(ctx).expect("should resolve");
        assert_eq!(decision.loop_class, TransferLoopClass::SubarrayGrouped);
    }

    #[test]
    fn transfer_loop_rejects_lossy_same_value_cast() {
        let ctx = TransferContext {
            dtype_relation: TransferDtypeRelation::LossyCast,
            same_value_cast: true,
            ..base_transfer_ctx()
        };
        let err = select_transfer_loop(ctx).expect_err("lossy same-value should reject");
        assert_eq!(err.reason_code(), "transfer_same_value_cast_rejected");
    }

    #[test]
    fn transfer_loop_overlap_resolves_copy_direction() {
        let ctx = TransferContext {
            has_overlap: true,
            dst_stride: 16,
            ..base_transfer_ctx()
        };
        let decision = select_transfer_loop(ctx).expect("should resolve with overlap");
        assert_eq!(decision.overlap_action, OverlapAction::BackwardCopy);
    }

    #[test]
    fn transfer_loop_hardened_rejects_overlapping_subarray() {
        let ctx = TransferContext {
            dtype_relation: TransferDtypeRelation::SubarrayBroadcast,
            has_overlap: true,
            mode: RuntimeMode::Hardened,
            ..base_transfer_ctx()
        };
        let err =
            select_transfer_loop(ctx).expect_err("hardened should reject overlapping subarray");
        assert_eq!(
            err.reason_code(),
            "transfer_subarray_broadcast_contract_violation"
        );
    }

    #[test]
    fn transfer_loop_hardened_rejects_overlapping_string() {
        let ctx = TransferContext {
            dtype_relation: TransferDtypeRelation::StringWidthChange {
                src_width: 4,
                dst_width: 8,
            },
            has_overlap: true,
            mode: RuntimeMode::Hardened,
            ..base_transfer_ctx()
        };
        let err = select_transfer_loop(ctx).expect_err("hardened should reject overlapping string");
        assert_eq!(err.reason_code(), "transfer_string_width_mismatch");
    }

    #[test]
    fn transfer_loop_selector_is_deterministic() {
        let contexts = [
            base_transfer_ctx(),
            TransferContext {
                dtype_relation: TransferDtypeRelation::LosslessCast,
                ..base_transfer_ctx()
            },
            TransferContext {
                has_where_mask: true,
                ..base_transfer_ctx()
            },
            TransferContext {
                has_overlap: true,
                ..base_transfer_ctx()
            },
            TransferContext {
                mode: RuntimeMode::Hardened,
                ..base_transfer_ctx()
            },
        ];
        for ctx in contexts {
            let first = select_transfer_loop(ctx);
            let second = select_transfer_loop(ctx);
            assert_eq!(first, second, "selector must be deterministic for {ctx:?}");
        }
    }

    #[test]
    fn transfer_loop_rejects_zero_item_size() {
        let ctx = TransferContext {
            item_size: 0,
            ..base_transfer_ctx()
        };
        let err = select_transfer_loop(ctx).expect_err("zero item_size should fail");
        assert_eq!(err.reason_code(), "transfer_selector_invalid_context");
    }

    #[test]
    fn transfer_loop_rejects_zero_element_count() {
        let ctx = TransferContext {
            element_count: 0,
            ..base_transfer_ctx()
        };
        let err = select_transfer_loop(ctx).expect_err("zero element_count should fail");
        assert_eq!(err.reason_code(), "transfer_selector_invalid_context");
    }

    #[test]
    fn where_mask_validation_accepts_matching_length() {
        let mask = vec![true, false, true, false];
        let count = validate_where_mask(&mask, 4).expect("should accept");
        assert_eq!(count, 2);
    }

    #[test]
    fn where_mask_validation_rejects_length_mismatch() {
        let mask = vec![true, false];
        let err = validate_where_mask(&mask, 4).expect_err("length mismatch should fail");
        assert_eq!(err.reason_code(), "transfer_where_mask_contract_violation");
    }

    #[test]
    fn subarray_transfer_validation_accepts_matching() {
        validate_subarray_transfer(4, 4).expect("same length should pass");
        validate_subarray_transfer(4, 8).expect("multiple should pass");
        validate_subarray_transfer(4, 12).expect("multiple should pass");
    }

    #[test]
    fn subarray_transfer_validation_rejects_invalid() {
        let err = validate_subarray_transfer(0, 4).expect_err("zero src should fail");
        assert_eq!(
            err.reason_code(),
            "transfer_subarray_broadcast_contract_violation"
        );

        let err = validate_subarray_transfer(3, 5).expect_err("non-multiple should fail");
        assert_eq!(
            err.reason_code(),
            "transfer_subarray_broadcast_contract_violation"
        );
    }

    #[test]
    fn all_error_variants_have_stable_reason_codes() {
        let errors: Vec<TransferError> = vec![
            TransferError::SelectorInvalidContext("test"),
            TransferError::OverlapPolicyTriggered("test"),
            TransferError::WhereMaskContractViolation("test"),
            TransferError::SameValueCastRejected,
            TransferError::StringWidthMismatch("test"),
            TransferError::SubarrayBroadcastContractViolation("test"),
            TransferError::FlatiterReadViolation("test"),
            TransferError::FlatiterWriteViolation("test"),
            TransferError::NditerOverlapPolicy("test"),
            TransferError::FpeCastError("test"),
        ];
        let expected_codes = &TRANSFER_PACKET_REASON_CODES;
        assert_eq!(errors.len(), expected_codes.len());
        for (err, expected) in errors.iter().zip(expected_codes.iter()) {
            assert_eq!(
                err.reason_code(),
                *expected,
                "reason code mismatch for {err:?}"
            );
        }
    }

    #[test]
    fn transfer_loop_masked_takes_priority_over_all_dtype_relations() {
        let relations = [
            TransferDtypeRelation::Same,
            TransferDtypeRelation::LosslessCast,
            TransferDtypeRelation::LossyCast,
            TransferDtypeRelation::SubarrayBroadcast,
            TransferDtypeRelation::StringWidthChange {
                src_width: 4,
                dst_width: 8,
            },
        ];
        for rel in relations {
            let ctx = TransferContext {
                has_where_mask: true,
                dtype_relation: rel,
                ..base_transfer_ctx()
            };
            let decision = select_transfer_loop(ctx).expect("masked should always resolve");
            assert_eq!(
                decision.loop_class,
                TransferLoopClass::MaskedTransfer,
                "mask must take priority over {rel:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Strided vs contiguous transfer class property tests
    // -----------------------------------------------------------------------

    #[test]
    fn strided_no_cast_for_same_dtype_non_unit_stride_grid() {
        let non_unit_strides: [isize; 4] = [16, 24, -16, 32];
        for src in non_unit_strides {
            for dst in non_unit_strides {
                let ctx = TransferContext {
                    src_stride: src,
                    dst_stride: dst,
                    dtype_relation: TransferDtypeRelation::Same,
                    ..base_transfer_ctx()
                };
                let decision = select_transfer_loop(ctx).expect("should resolve");
                assert_eq!(
                    decision.loop_class,
                    TransferLoopClass::StridedNoCast,
                    "non-unit stride ({src},{dst}) with Same dtype should be StridedNoCast"
                );
            }
        }
    }

    #[test]
    fn contiguous_cast_only_for_unit_stride_lossless() {
        // Unit stride + lossless = ContiguousCast
        let ctx = TransferContext {
            src_stride: 8,
            dst_stride: 8,
            aligned: true,
            dtype_relation: TransferDtypeRelation::LosslessCast,
            ..base_transfer_ctx()
        };
        assert_eq!(
            select_transfer_loop(ctx).expect("unit lossless").loop_class,
            TransferLoopClass::ContiguousCast
        );

        // Non-unit stride + lossless = StridedWithCast
        let ctx = TransferContext {
            src_stride: 16,
            dst_stride: 16,
            dtype_relation: TransferDtypeRelation::LosslessCast,
            ..base_transfer_ctx()
        };
        assert_eq!(
            select_transfer_loop(ctx)
                .expect("non-unit lossless")
                .loop_class,
            TransferLoopClass::StridedWithCast
        );
    }

    #[test]
    fn lossy_cast_always_strided_with_cast_regardless_of_stride() {
        for stride in [8isize, 16, 24, -8] {
            let ctx = TransferContext {
                src_stride: stride,
                dst_stride: stride,
                dtype_relation: TransferDtypeRelation::LossyCast,
                ..base_transfer_ctx()
            };
            let decision = select_transfer_loop(ctx).expect("lossy should resolve");
            assert_eq!(
                decision.loop_class,
                TransferLoopClass::StridedWithCast,
                "lossy cast with stride={stride} should be StridedWithCast"
            );
        }
    }

    #[test]
    fn unaligned_same_dtype_unit_stride_is_strided_no_cast() {
        let ctx = TransferContext {
            aligned: false,
            dtype_relation: TransferDtypeRelation::Same,
            ..base_transfer_ctx()
        };
        assert_eq!(
            select_transfer_loop(ctx).expect("unaligned").loop_class,
            TransferLoopClass::StridedNoCast
        );
    }

    #[test]
    fn negative_stride_preserves_loop_class_selection() {
        // Negative unit stride with Same dtype -> StridedNoCast (needs iteration)
        let ctx = TransferContext {
            src_stride: -8,
            dst_stride: -8,
            dtype_relation: TransferDtypeRelation::Same,
            ..base_transfer_ctx()
        };
        assert_eq!(
            select_transfer_loop(ctx).expect("neg stride").loop_class,
            TransferLoopClass::StridedNoCast
        );

        // Negative unit stride with LosslessCast -> StridedWithCast
        let ctx = TransferContext {
            src_stride: -8,
            dst_stride: -8,
            aligned: true,
            dtype_relation: TransferDtypeRelation::LosslessCast,
            ..base_transfer_ctx()
        };
        assert_eq!(
            select_transfer_loop(ctx).expect("neg lossless").loop_class,
            TransferLoopClass::StridedWithCast
        );
    }

    // -----------------------------------------------------------------------
    // FPE status and post-transfer validation tests (P2C003-R10)
    // -----------------------------------------------------------------------

    #[test]
    fn fpe_status_default_is_clean() {
        let fpe = FpeStatus::default();
        assert!(!fpe.any_set());
        assert_eq!(fpe.first_flag_name(), None);
    }

    #[test]
    fn fpe_status_reports_first_flag_by_severity() {
        // invalid takes priority
        let fpe = FpeStatus {
            overflow: true,
            invalid: true,
            ..FpeStatus::default()
        };
        assert!(fpe.any_set());
        assert_eq!(fpe.first_flag_name(), Some("invalid"));

        // divide_by_zero before overflow
        let fpe = FpeStatus {
            overflow: true,
            divide_by_zero: true,
            ..FpeStatus::default()
        };
        assert_eq!(fpe.first_flag_name(), Some("divide_by_zero"));

        // overflow before underflow
        let fpe = FpeStatus {
            overflow: true,
            underflow: true,
            ..FpeStatus::default()
        };
        assert_eq!(fpe.first_flag_name(), Some("overflow"));

        // underflow alone
        let fpe = FpeStatus {
            underflow: true,
            ..FpeStatus::default()
        };
        assert_eq!(fpe.first_flag_name(), Some("underflow"));
    }

    #[test]
    fn fpe_strict_mode_allows_all_flags() {
        let fpe = FpeStatus {
            overflow: true,
            invalid: true,
            divide_by_zero: true,
            underflow: true,
        };
        let result =
            validate_post_transfer_fpe(fpe, RuntimeMode::Strict).expect("strict allows FPE");
        assert!(result.any_set());
    }

    #[test]
    fn fpe_hardened_mode_rejects_overflow() {
        let fpe = FpeStatus {
            overflow: true,
            ..FpeStatus::default()
        };
        let err = validate_post_transfer_fpe(fpe, RuntimeMode::Hardened)
            .expect_err("hardened rejects overflow");
        assert_eq!(err.reason_code(), "transfer_fpe_cast_error");
    }

    #[test]
    fn fpe_hardened_mode_rejects_invalid() {
        let fpe = FpeStatus {
            invalid: true,
            ..FpeStatus::default()
        };
        let err = validate_post_transfer_fpe(fpe, RuntimeMode::Hardened)
            .expect_err("hardened rejects invalid");
        assert_eq!(err.reason_code(), "transfer_fpe_cast_error");
    }

    #[test]
    fn fpe_hardened_mode_allows_clean() {
        let fpe = FpeStatus::default();
        validate_post_transfer_fpe(fpe, RuntimeMode::Hardened).expect("clean FPE is allowed");
    }

    #[test]
    fn fpe_hardened_rejects_divide_by_zero() {
        let fpe = FpeStatus {
            divide_by_zero: true,
            ..FpeStatus::default()
        };
        let err = validate_post_transfer_fpe(fpe, RuntimeMode::Hardened)
            .expect_err("hardened rejects div0");
        assert_eq!(err.reason_code(), "transfer_fpe_cast_error");
        assert!(err.to_string().contains("division by zero"));
    }

    // -----------------------------------------------------------------------
    // Flatiter indexing edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn flatiter_single_index_boundary() {
        // Last valid index
        assert_eq!(
            validate_flatiter_read(10, &FlatIterIndex::Single(9)).unwrap(),
            1
        );
        // First out-of-bounds
        let err = validate_flatiter_read(10, &FlatIterIndex::Single(10))
            .expect_err("boundary index should fail");
        assert_eq!(err.reason_code(), "flatiter_transfer_read_violation");
    }

    #[test]
    fn flatiter_single_index_zero_length_array() {
        let err = validate_flatiter_read(0, &FlatIterIndex::Single(0))
            .expect_err("index 0 in empty array should fail");
        assert_eq!(err.reason_code(), "flatiter_transfer_read_violation");
    }

    #[test]
    fn flatiter_slice_full_range() {
        let idx = FlatIterIndex::Slice {
            start: 0,
            stop: 8,
            step: 1,
        };
        assert_eq!(validate_flatiter_read(8, &idx).unwrap(), 8);
    }

    #[test]
    fn flatiter_slice_with_step() {
        let idx = FlatIterIndex::Slice {
            start: 0,
            stop: 10,
            step: 3,
        };
        // Elements at 0, 3, 6, 9 = 4 elements
        assert_eq!(validate_flatiter_read(10, &idx).unwrap(), 4);
    }

    #[test]
    fn flatiter_slice_empty_range() {
        let idx = FlatIterIndex::Slice {
            start: 5,
            stop: 5,
            step: 1,
        };
        assert_eq!(validate_flatiter_read(10, &idx).unwrap(), 0);
    }

    #[test]
    fn flatiter_slice_zero_step_rejected() {
        let idx = FlatIterIndex::Slice {
            start: 0,
            stop: 5,
            step: 0,
        };
        let err = validate_flatiter_read(10, &idx).expect_err("zero step should fail");
        assert_eq!(err.reason_code(), "flatiter_transfer_read_violation");
    }

    #[test]
    fn flatiter_slice_start_exceeds_stop_rejected() {
        let idx = FlatIterIndex::Slice {
            start: 5,
            stop: 3,
            step: 1,
        };
        let err = validate_flatiter_read(10, &idx).expect_err("start > stop should fail");
        assert_eq!(err.reason_code(), "flatiter_transfer_read_violation");
    }

    #[test]
    fn flatiter_slice_stop_exceeds_length_rejected() {
        let idx = FlatIterIndex::Slice {
            start: 0,
            stop: 11,
            step: 1,
        };
        let err = validate_flatiter_read(10, &idx).expect_err("stop > length should fail");
        assert_eq!(err.reason_code(), "flatiter_transfer_read_violation");
    }

    #[test]
    fn flatiter_fancy_empty_indices() {
        let idx = FlatIterIndex::Fancy(vec![]);
        assert_eq!(validate_flatiter_read(10, &idx).unwrap(), 0);
    }

    #[test]
    fn flatiter_fancy_duplicate_indices_allowed() {
        let idx = FlatIterIndex::Fancy(vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(validate_flatiter_read(10, &idx).unwrap(), 6);
    }

    #[test]
    fn flatiter_bool_mask_all_true() {
        let mask = FlatIterIndex::BoolMask(vec![true; 8]);
        assert_eq!(validate_flatiter_read(8, &mask).unwrap(), 8);
    }

    #[test]
    fn flatiter_bool_mask_all_false() {
        let mask = FlatIterIndex::BoolMask(vec![false; 8]);
        assert_eq!(validate_flatiter_read(8, &mask).unwrap(), 0);
    }

    #[test]
    fn flatiter_bool_mask_empty() {
        let mask = FlatIterIndex::BoolMask(vec![]);
        assert_eq!(validate_flatiter_read(0, &mask).unwrap(), 0);
    }

    #[test]
    fn flatiter_write_values_mismatch_rejected() {
        let idx = FlatIterIndex::Slice {
            start: 0,
            stop: 4,
            step: 1,
        };
        let err =
            validate_flatiter_write(10, &idx, 3).expect_err("3 values for 4 lanes should fail");
        assert_eq!(err.reason_code(), "flatiter_transfer_write_violation");
    }

    #[test]
    fn flatiter_write_scalar_broadcast_validation_is_allowed() {
        let idx = FlatIterIndex::Slice {
            start: 0,
            stop: 4,
            step: 1,
        };
        assert_eq!(validate_flatiter_write(10, &idx, 1).unwrap(), 4);
    }

    #[test]
    fn flatiter_write_correct_values_count() {
        let idx = FlatIterIndex::Fancy(vec![0, 2, 4]);
        assert_eq!(validate_flatiter_write(10, &idx, 3).unwrap(), 3);
    }

    #[test]
    fn flatiter_write_propagates_read_violation_as_write() {
        let idx = FlatIterIndex::Single(100);
        let err = validate_flatiter_write(10, &idx, 1).expect_err("out of bounds should fail");
        assert_eq!(err.reason_code(), "flatiter_transfer_write_violation");
    }

    #[test]
    fn flatiter_index_resolution_matches_slice_and_mask_semantics() {
        let slice = FlatIterIndex::Slice {
            start: 1,
            stop: 8,
            step: 3,
        };
        assert_eq!(
            resolve_flatiter_indices(10, &slice).expect("slice should resolve"),
            vec![1, 4, 7]
        );

        let mask = FlatIterIndex::BoolMask(vec![false, true, false, true, false]);
        assert_eq!(
            resolve_flatiter_indices(5, &mask).expect("mask should resolve"),
            vec![1, 3]
        );
    }

    #[test]
    fn flatiter_read_returns_selected_values_in_order() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let idx = FlatIterIndex::Fancy(vec![4, 1, 1]);
        assert_eq!(
            read_flatiter(&values, &idx).expect("read should succeed"),
            vec![50.0, 20.0, 20.0]
        );
    }

    #[test]
    fn flatiter_write_requires_writeable_storage() {
        let mut values = vec![1.0, 2.0, 3.0];
        let err = write_flatiter(&mut values, false, &FlatIterIndex::Single(0), &[9.0])
            .expect_err("non-writeable flatiter write should fail");
        assert_eq!(err.reason_code(), "flatiter_indexing_contract_violation");
    }

    #[test]
    fn flatiter_write_supports_scalar_broadcast_assignment() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let idx = FlatIterIndex::Slice {
            start: 1,
            stop: 4,
            step: 2,
        };
        let written = write_flatiter(&mut values, true, &idx, &[9.0])
            .expect("scalar assignment should broadcast");
        assert_eq!(written, 2);
        assert_eq!(values, vec![1.0, 9.0, 3.0, 9.0]);
    }

    #[test]
    fn flatiter_write_rejects_assignment_arity_mismatch() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let idx = FlatIterIndex::Fancy(vec![0, 2, 3]);
        let err = write_flatiter(&mut values, true, &idx, &[7.0, 8.0])
            .expect_err("non-scalar mismatched assignment should fail");
        assert_eq!(err.reason_code(), "flatiter_indexing_contract_violation");
    }

    #[test]
    fn ndindex_emits_cartesian_coordinates_in_c_order() {
        assert_eq!(
            ndindex(&[2, 3]).expect("ndindex should succeed"),
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![0, 2],
                vec![1, 0],
                vec![1, 1],
                vec![1, 2]
            ]
        );
    }

    #[test]
    fn ndindex_handles_zero_dim_and_zero_extent_shapes() {
        assert_eq!(
            ndindex(&[]).expect("scalar ndindex"),
            vec![Vec::<usize>::new()]
        );
        assert!(
            ndindex(&[2, 0, 3])
                .expect("zero extent should be empty")
                .is_empty()
        );
    }

    #[test]
    fn ndenumerate_pairs_coordinates_with_values() {
        let pairs = ndenumerate(&[2, 2], &[10, 20, 30, 40]).expect("ndenumerate should succeed");
        assert_eq!(
            pairs,
            vec![
                (vec![0, 0], 10),
                (vec![0, 1], 20),
                (vec![1, 0], 30),
                (vec![1, 1], 40)
            ]
        );
    }

    #[test]
    fn ndenumerate_rejects_value_length_mismatch() {
        let err = ndenumerate(&[2, 2], &[1, 2, 3]).expect_err("length mismatch should fail");
        assert_eq!(err.reason_code(), "ndindex_shape_validation_failed");
    }

    // -----------------------------------------------------------------------
    // Nditer flags validation edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn nditer_flags_all_false_is_valid() {
        validate_nditer_flags(NditerTransferFlags {
            copy_if_overlap: false,
            no_broadcast: false,
            observed_overlap: false,
            observed_broadcast: false,
        })
        .expect("all false flags should pass");
    }

    #[test]
    fn nditer_flags_all_true_is_valid() {
        // copy_if_overlap=true handles observed_overlap, no_broadcast=false allows broadcast
        validate_nditer_flags(NditerTransferFlags {
            copy_if_overlap: true,
            no_broadcast: false,
            observed_overlap: true,
            observed_broadcast: true,
        })
        .expect("valid combination should pass");
    }

    // -----------------------------------------------------------------------
    // Transfer selector (old API) additional edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn selector_non_aligned_unit_stride_is_strided() {
        let input = TransferSelectorInput {
            src_stride: 8,
            dst_stride: 8,
            item_size: 8,
            element_count: 4,
            aligned: false,
            cast_is_lossless: true,
            same_value_cast: false,
        };
        let result = select_transfer_class(input).expect("should resolve");
        assert_eq!(result, TransferClass::Strided);
    }

    #[test]
    fn selector_non_unit_stride_with_cast_is_strided_cast() {
        let input = TransferSelectorInput {
            src_stride: 16,
            dst_stride: 16,
            item_size: 8,
            element_count: 4,
            aligned: true,
            cast_is_lossless: false,
            same_value_cast: false,
        };
        let result = select_transfer_class(input).expect("should resolve");
        assert_eq!(result, TransferClass::StridedCast);
    }

    #[test]
    fn selector_misaligned_stride_rejected() {
        let input = TransferSelectorInput {
            src_stride: 7, // not multiple of item_size=8
            dst_stride: 8,
            item_size: 8,
            element_count: 4,
            aligned: true,
            cast_is_lossless: true,
            same_value_cast: false,
        };
        let err = select_transfer_class(input).expect_err("misaligned stride should fail");
        assert_eq!(err.reason_code(), "transfer_selector_invalid_context");
    }

    // -----------------------------------------------------------------------
    // Transfer error Display coverage
    // -----------------------------------------------------------------------

    #[test]
    fn transfer_error_display_and_reason_code_consistency() {
        let errors = vec![
            TransferError::SelectorInvalidContext("test context"),
            TransferError::OverlapPolicyTriggered("test overlap"),
            TransferError::WhereMaskContractViolation("test mask"),
            TransferError::SameValueCastRejected,
            TransferError::StringWidthMismatch("test width"),
            TransferError::SubarrayBroadcastContractViolation("test subarray"),
            TransferError::FlatiterReadViolation("test read"),
            TransferError::FlatiterWriteViolation("test write"),
            TransferError::NditerOverlapPolicy("test nditer"),
            TransferError::FpeCastError("test fpe"),
        ];
        for err in &errors {
            // Display should produce non-empty string
            let display = err.to_string();
            assert!(!display.is_empty(), "Display must be non-empty for {err:?}");
            // reason_code should be non-empty
            let code = err.reason_code();
            assert!(
                !code.is_empty(),
                "reason_code must be non-empty for {err:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Transfer log record completeness
    // -----------------------------------------------------------------------

    #[test]
    fn transfer_log_record_rejects_empty_artifact_refs() {
        let record = TransferLogRecord {
            fixture_id: "test".to_string(),
            seed: 1,
            mode: RuntimeMode::Strict,
            env_fingerprint: "test-env".to_string(),
            artifact_refs: Vec::new(),
            reason_code: "test_code".to_string(),
            passed: true,
        };
        assert!(!record.is_replay_complete());
    }

    #[test]
    fn transfer_log_record_rejects_whitespace_artifact_ref() {
        let record = TransferLogRecord {
            fixture_id: "test".to_string(),
            seed: 1,
            mode: RuntimeMode::Strict,
            env_fingerprint: "test-env".to_string(),
            artifact_refs: vec!["valid_ref".to_string(), "   ".to_string()],
            reason_code: "test_code".to_string(),
            passed: true,
        };
        assert!(!record.is_replay_complete());
    }

    #[test]
    fn transfer_log_record_accepts_complete_record() {
        let record = TransferLogRecord {
            fixture_id: "UP-003-test".to_string(),
            seed: 42,
            mode: RuntimeMode::Hardened,
            env_fingerprint: "fnp-iter-tests".to_string(),
            artifact_refs: vec!["artifact1.json".to_string(), "artifact2.yaml".to_string()],
            reason_code: "transfer_selector_invalid_context".to_string(),
            passed: true,
        };
        assert!(record.is_replay_complete());
    }

    // -----------------------------------------------------------------------
    // Zero-element and boundary transfer edge cases (br-k36)
    // -----------------------------------------------------------------------

    #[test]
    fn select_transfer_class_rejects_zero_element_count() {
        let input = TransferSelectorInput {
            src_stride: 8,
            dst_stride: 8,
            item_size: 8,
            element_count: 0,
            aligned: true,
            cast_is_lossless: true,
            same_value_cast: false,
        };
        let err = select_transfer_class(input).unwrap_err();
        assert!(matches!(err, TransferError::SelectorInvalidContext(_)));
    }

    #[test]
    fn select_transfer_class_rejects_zero_item_size() {
        let input = TransferSelectorInput {
            src_stride: 0,
            dst_stride: 0,
            item_size: 0,
            element_count: 10,
            aligned: true,
            cast_is_lossless: true,
            same_value_cast: false,
        };
        let err = select_transfer_class(input).unwrap_err();
        assert!(matches!(err, TransferError::SelectorInvalidContext(_)));
    }

    #[test]
    fn overlap_policy_rejects_zero_byte_len() {
        let err = overlap_copy_policy(0, 0, 0).unwrap_err();
        assert!(matches!(err, TransferError::OverlapPolicyTriggered(_)));
    }

    #[test]
    fn select_transfer_class_negative_strides_not_contiguous() {
        // Negative strides are not C-contiguous.
        let input = TransferSelectorInput {
            src_stride: -8,
            dst_stride: -8,
            item_size: 8,
            element_count: 5,
            aligned: true,
            cast_is_lossless: true,
            same_value_cast: false,
        };
        let class = select_transfer_class(input).unwrap();
        assert_eq!(class, TransferClass::Strided);
    }

    #[test]
    fn select_transfer_class_mixed_sign_strides() {
        // One positive, one negative stride. Definitely not contiguous.
        let input = TransferSelectorInput {
            src_stride: 8,
            dst_stride: -8,
            item_size: 8,
            element_count: 3,
            aligned: true,
            cast_is_lossless: true,
            same_value_cast: false,
        };
        let class = select_transfer_class(input).unwrap();
        assert_eq!(class, TransferClass::Strided);
    }

    #[test]
    fn transfer_loop_overlap_negative_strides_clobber_check() {
        // Case: dst_stride = -16, src_stride = -8, both negative.
        // |dst_stride| > |src_stride|, so forward copy should clobber.
        // It now correctly returns BackwardCopy.
        let ctx = TransferContext {
            src_stride: -8,
            dst_stride: -16,
            item_size: 8,
            element_count: 4,
            aligned: true,
            dtype_relation: TransferDtypeRelation::Same,
            same_value_cast: false,
            has_where_mask: false,
            has_overlap: true,
            mode: RuntimeMode::Strict,
        };
        let decision = select_transfer_loop(ctx).expect("should resolve");
        assert_eq!(decision.overlap_action, OverlapAction::BackwardCopy);
    }

    #[test]
    fn transfer_loop_overlap_mixed_sign_rejected() {
        // Mixed signs - crossing overlap is unsafe.
        let ctx = TransferContext {
            src_stride: 8,
            dst_stride: -8,
            item_size: 8,
            element_count: 4,
            aligned: true,
            dtype_relation: TransferDtypeRelation::Same,
            same_value_cast: false,
            has_where_mask: false,
            has_overlap: true,
            mode: RuntimeMode::Strict,
        };
        let err = select_transfer_loop(ctx).expect_err("crossing overlap should be rejected");
        assert_eq!(err.reason_code(), "transfer_overlap_policy_triggered");
    }

    #[test]
    fn flatiter_count_true_mask_empty() {
        assert_eq!(count_true_mask(&[]), 0);
    }

    #[test]
    fn flatiter_count_true_mask_all_false() {
        assert_eq!(count_true_mask(&[false; 16]), 0);
    }

    #[test]
    fn flatiter_count_true_mask_all_true() {
        assert_eq!(count_true_mask(&[true; 17]), 17);
    }
}
