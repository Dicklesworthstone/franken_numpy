#![forbid(unsafe_code)]

use core::fmt;
use std::collections::HashSet;

pub const IO_PACKET_ID: &str = "FNP-P2C-009";
pub const NPY_MAGIC_PREFIX: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];
pub const NPZ_MAGIC_PREFIX: [u8; 4] = [b'P', b'K', 0x03, 0x04];

pub const MAX_HEADER_BYTES: usize = 65_536;
pub const MAX_ARCHIVE_MEMBERS: usize = 4_096;
pub const MAX_ARCHIVE_UNCOMPRESSED_BYTES: usize = 2 * 1024 * 1024 * 1024;
pub const MAX_DISPATCH_RETRIES: usize = 8;
pub const MAX_MEMMAP_VALIDATION_RETRIES: usize = 64;

pub const IO_PACKET_REASON_CODES: [&str; 10] = [
    "io_magic_invalid",
    "io_header_schema_invalid",
    "io_dtype_descriptor_invalid",
    "io_write_contract_violation",
    "io_read_payload_incomplete",
    "io_pickle_policy_violation",
    "io_memmap_contract_violation",
    "io_load_dispatch_invalid",
    "io_npz_archive_contract_violation",
    "io_policy_unknown_metadata",
];
const NPY_HEADER_REQUIRED_KEYS: [&str; 3] = ["descr", "fortran_order", "shape"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IORuntimeMode {
    Strict,
    Hardened,
}

impl IORuntimeMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Hardened => "hardened",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IOSupportedDType {
    Bool,
    I32,
    I64,
    F32,
    F64,
    Object,
}

impl IOSupportedDType {
    #[must_use]
    pub const fn descr(self) -> &'static str {
        match self {
            Self::Bool => "|b1",
            Self::I32 => "<i4",
            Self::I64 => "<i8",
            Self::F32 => "<f4",
            Self::F64 => "<f8",
            Self::Object => "|O",
        }
    }

    pub fn decode(descr: &str) -> Result<Self, IOError> {
        match descr {
            "|b1" => Ok(Self::Bool),
            "<i4" => Ok(Self::I32),
            "<i8" => Ok(Self::I64),
            "<f4" => Ok(Self::F32),
            "<f8" => Ok(Self::F64),
            "|O" => Ok(Self::Object),
            _ => Err(IOError::DTypeDescriptorInvalid),
        }
    }

    #[must_use]
    pub const fn item_size(self) -> Option<usize> {
        match self {
            Self::Bool => Some(1),
            Self::I32 | Self::F32 => Some(4),
            Self::I64 | Self::F64 => Some(8),
            Self::Object => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemmapMode {
    ReadOnly,
    ReadWrite,
    Write,
    CopyOnWrite,
}

impl MemmapMode {
    pub fn parse(token: &str) -> Result<Self, IOError> {
        match token {
            "r" => Ok(Self::ReadOnly),
            "r+" => Ok(Self::ReadWrite),
            "w+" => Ok(Self::Write),
            "c" => Ok(Self::CopyOnWrite),
            _ => Err(IOError::MemmapContractViolation(
                "invalid memmap mode token",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadDispatch {
    Npy,
    Npz,
    Pickle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IOError {
    MagicInvalid,
    HeaderSchemaInvalid(&'static str),
    DTypeDescriptorInvalid,
    WriteContractViolation(&'static str),
    ReadPayloadIncomplete(&'static str),
    PicklePolicyViolation,
    MemmapContractViolation(&'static str),
    LoadDispatchInvalid(&'static str),
    NpzArchiveContractViolation(&'static str),
    PolicyUnknownMetadata(&'static str),
}

impl IOError {
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::MagicInvalid => "io_magic_invalid",
            Self::HeaderSchemaInvalid(_) => "io_header_schema_invalid",
            Self::DTypeDescriptorInvalid => "io_dtype_descriptor_invalid",
            Self::WriteContractViolation(_) => "io_write_contract_violation",
            Self::ReadPayloadIncomplete(_) => "io_read_payload_incomplete",
            Self::PicklePolicyViolation => "io_pickle_policy_violation",
            Self::MemmapContractViolation(_) => "io_memmap_contract_violation",
            Self::LoadDispatchInvalid(_) => "io_load_dispatch_invalid",
            Self::NpzArchiveContractViolation(_) => "io_npz_archive_contract_violation",
            Self::PolicyUnknownMetadata(_) => "io_policy_unknown_metadata",
        }
    }
}

impl fmt::Display for IOError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MagicInvalid => write!(f, "invalid or unsupported npy/npz magic/version"),
            Self::HeaderSchemaInvalid(msg) => write!(f, "{msg}"),
            Self::DTypeDescriptorInvalid => write!(f, "dtype descriptor is invalid or unsupported"),
            Self::WriteContractViolation(msg) => write!(f, "{msg}"),
            Self::ReadPayloadIncomplete(msg) => write!(f, "{msg}"),
            Self::PicklePolicyViolation => write!(f, "pickle/object payload rejected by policy"),
            Self::MemmapContractViolation(msg) => write!(f, "{msg}"),
            Self::LoadDispatchInvalid(msg) => write!(f, "{msg}"),
            Self::NpzArchiveContractViolation(msg) => write!(f, "{msg}"),
            Self::PolicyUnknownMetadata(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for IOError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpyHeader {
    pub shape: Vec<usize>,
    pub fortran_order: bool,
    pub descr: IOSupportedDType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpyArrayBytes {
    pub version: (u8, u8),
    pub header: NpyHeader,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IOLogRecord {
    pub ts_utc: String,
    pub suite_id: String,
    pub test_id: String,
    pub packet_id: String,
    pub fixture_id: String,
    pub mode: IORuntimeMode,
    pub seed: u64,
    pub input_digest: String,
    pub output_digest: String,
    pub env_fingerprint: String,
    pub artifact_refs: Vec<String>,
    pub duration_ms: u64,
    pub outcome: String,
    pub reason_code: String,
}

impl IOLogRecord {
    #[must_use]
    pub fn is_replay_complete(&self) -> bool {
        if self.ts_utc.trim().is_empty()
            || self.suite_id.trim().is_empty()
            || self.test_id.trim().is_empty()
            || self.packet_id.trim().is_empty()
            || self.fixture_id.trim().is_empty()
            || self.input_digest.trim().is_empty()
            || self.output_digest.trim().is_empty()
            || self.env_fingerprint.trim().is_empty()
            || self.reason_code.trim().is_empty()
        {
            return false;
        }

        if self.packet_id != IO_PACKET_ID {
            return false;
        }

        if self.outcome != "pass" && self.outcome != "fail" {
            return false;
        }

        if self.artifact_refs.is_empty()
            || self
                .artifact_refs
                .iter()
                .any(|artifact| artifact.trim().is_empty())
        {
            return false;
        }

        IO_PACKET_REASON_CODES
            .iter()
            .any(|code| *code == self.reason_code)
    }
}

fn element_count(shape: &[usize]) -> Result<usize, IOError> {
    shape
        .iter()
        .copied()
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
        .ok_or(IOError::HeaderSchemaInvalid(
            "shape element-count overflowed",
        ))
}

fn validate_npy_version(version: (u8, u8)) -> Result<(), IOError> {
    if version == (1, 0) || version == (2, 0) || version == (3, 0) {
        Ok(())
    } else {
        Err(IOError::MagicInvalid)
    }
}

fn npy_length_field_size(version: (u8, u8)) -> Result<usize, IOError> {
    match version {
        (1, 0) => Ok(2),
        (2, 0) | (3, 0) => Ok(4),
        _ => Err(IOError::MagicInvalid),
    }
}

fn format_shape_tuple(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_string(),
        [single] => format!("({single},)"),
        _ => {
            let joined = shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("({joined},)")
        }
    }
}

fn encode_header_dict(header: &NpyHeader) -> String {
    let fortran_order = if header.fortran_order {
        "True"
    } else {
        "False"
    };
    let shape = format_shape_tuple(&header.shape);
    format!(
        "{{'descr': '{}', 'fortran_order': {fortran_order}, 'shape': {shape}, }}",
        header.descr.descr()
    )
}

fn encode_npy_header_bytes(header: &NpyHeader, version: (u8, u8)) -> Result<Vec<u8>, IOError> {
    let length_field_size = npy_length_field_size(version)?;
    let dictionary = encode_header_dict(header);
    let dictionary_bytes = dictionary.as_bytes();
    let prefix_len = NPY_MAGIC_PREFIX.len() + 2 + length_field_size;
    let base_header_len =
        dictionary_bytes
            .len()
            .checked_add(1)
            .ok_or(IOError::HeaderSchemaInvalid(
                "header bytes must be within bounded budget",
            ))?;
    let padding = (16 - ((prefix_len + base_header_len) % 16)) % 16;
    let header_len = base_header_len
        .checked_add(padding)
        .ok_or(IOError::HeaderSchemaInvalid(
            "header bytes must be within bounded budget",
        ))?;
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return Err(IOError::HeaderSchemaInvalid(
            "header bytes must be within bounded budget",
        ));
    }

    let mut header_bytes = Vec::with_capacity(header_len);
    header_bytes.extend_from_slice(dictionary_bytes);
    header_bytes.extend(std::iter::repeat_n(b' ', padding));
    header_bytes.push(b'\n');
    Ok(header_bytes)
}

fn write_npy_preamble(
    buffer: &mut Vec<u8>,
    version: (u8, u8),
    header_len: usize,
) -> Result<(), IOError> {
    buffer.extend_from_slice(&NPY_MAGIC_PREFIX);
    buffer.push(version.0);
    buffer.push(version.1);
    match version {
        (1, 0) => {
            let header_len = u16::try_from(header_len).map_err(|_| {
                IOError::HeaderSchemaInvalid("version 1.0 header length exceeds u16 boundary")
            })?;
            buffer.extend_from_slice(&header_len.to_le_bytes());
        }
        (2, 0) | (3, 0) => {
            let header_len = u32::try_from(header_len)
                .map_err(|_| IOError::HeaderSchemaInvalid("header length exceeds u32 boundary"))?;
            buffer.extend_from_slice(&header_len.to_le_bytes());
        }
        _ => return Err(IOError::MagicInvalid),
    }
    Ok(())
}

fn read_header_span(payload: &[u8], version: (u8, u8)) -> Result<(usize, usize), IOError> {
    let length_field_size = npy_length_field_size(version)?;
    let header_offset = NPY_MAGIC_PREFIX.len() + 2 + length_field_size;
    let header_len = match version {
        (1, 0) => {
            if payload.len() < 10 {
                return Err(IOError::HeaderSchemaInvalid(
                    "payload truncated before v1 header length field",
                ));
            }
            usize::from(u16::from_le_bytes([payload[8], payload[9]]))
        }
        (2, 0) | (3, 0) => {
            if payload.len() < 12 {
                return Err(IOError::HeaderSchemaInvalid(
                    "payload truncated before v2/v3 header length field",
                ));
            }
            let raw = u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]);
            usize::try_from(raw).map_err(|_| {
                IOError::HeaderSchemaInvalid("header length exceeds platform usize boundary")
            })?
        }
        _ => return Err(IOError::MagicInvalid),
    };

    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return Err(IOError::HeaderSchemaInvalid(
            "header bytes must be within bounded budget",
        ));
    }
    let end = header_offset
        .checked_add(header_len)
        .ok_or(IOError::HeaderSchemaInvalid(
            "header offset/length overflowed",
        ))?;
    if payload.len() < end {
        return Err(IOError::HeaderSchemaInvalid(
            "payload truncated before declared header bytes",
        ));
    }

    Ok((header_offset, header_len))
}

fn extract_after_key<'a>(dictionary: &'a str, key: &str) -> Result<&'a str, IOError> {
    let single = format!("'{key}'");
    let double = format!("\"{key}\"");
    let key_start = dictionary
        .find(&single)
        .or_else(|| dictionary.find(&double))
        .ok_or(IOError::HeaderSchemaInvalid(
            "required header field is missing",
        ))?;
    let tail = &dictionary[key_start + single.len()..];
    let tail = tail.trim_start();
    let tail = tail.strip_prefix(':').ok_or(IOError::HeaderSchemaInvalid(
        "header field is missing ':' separator",
    ))?;
    Ok(tail.trim_start())
}

fn parse_quoted_value(value: &str) -> Result<&str, IOError> {
    let quote = value
        .as_bytes()
        .first()
        .copied()
        .ok_or(IOError::HeaderSchemaInvalid("header quoted value is empty"))?;
    if quote != b'\'' && quote != b'"' {
        return Err(IOError::HeaderSchemaInvalid(
            "header quoted value must start with quote",
        ));
    }

    let tail = &value[1..];
    let end = tail
        .find(char::from(quote))
        .ok_or(IOError::HeaderSchemaInvalid(
            "header quoted value missing closing quote",
        ))?;
    Ok(&tail[..end])
}

fn parse_shape_tuple(tuple_literal: &str) -> Result<Vec<usize>, IOError> {
    let inner = tuple_literal.trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    let has_comma = inner.contains(',');

    let mut shape = Vec::new();
    let mut saw_non_empty = false;
    for token in inner.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        saw_non_empty = true;
        let dim = token
            .parse::<usize>()
            .map_err(|_| IOError::HeaderSchemaInvalid("shape tuple entries must be usize"))?;
        shape.push(dim);
    }

    if !saw_non_empty {
        return Err(IOError::HeaderSchemaInvalid(
            "shape tuple contains no dimensions",
        ));
    }
    if shape.len() == 1 && !has_comma {
        return Err(IOError::HeaderSchemaInvalid(
            "singleton shape tuples must include trailing comma",
        ));
    }

    Ok(shape)
}

fn parse_header_keys(dictionary: &str) -> Result<Vec<String>, IOError> {
    let bytes = dictionary.as_bytes();
    let mut keys = Vec::new();
    let mut idx = 0usize;

    while idx < bytes.len() {
        let byte = bytes[idx];
        if byte != b'\'' && byte != b'"' {
            idx += 1;
            continue;
        }

        let quote = byte;
        let start = idx + 1;
        idx += 1;
        while idx < bytes.len() {
            let escaped = idx > start && bytes[idx - 1] == b'\\';
            if bytes[idx] == quote && !escaped {
                break;
            }
            idx += 1;
        }
        if idx >= bytes.len() {
            return Err(IOError::HeaderSchemaInvalid(
                "header key/value quote is not terminated",
            ));
        }

        let token = &dictionary[start..idx];
        idx += 1;

        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < bytes.len() && bytes[idx] == b':' {
            if keys.iter().any(|existing| existing == token) {
                return Err(IOError::HeaderSchemaInvalid(
                    "header dictionary contains duplicate keys",
                ));
            }
            keys.push(token.to_string());
        }
    }

    Ok(keys)
}

fn parse_header_dictionary(header_bytes: &[u8], header_len: usize) -> Result<NpyHeader, IOError> {
    let dictionary = std::str::from_utf8(header_bytes).map_err(|_| {
        IOError::HeaderSchemaInvalid("header bytes must decode as utf-8/ascii dictionary")
    })?;
    let dictionary = dictionary.trim_end();
    if !(dictionary.starts_with('{') && dictionary.ends_with('}')) {
        return Err(IOError::HeaderSchemaInvalid(
            "header dictionary must be wrapped in braces",
        ));
    }
    let keys = parse_header_keys(dictionary)?;
    if keys.len() != NPY_HEADER_REQUIRED_KEYS.len()
        || NPY_HEADER_REQUIRED_KEYS
            .iter()
            .any(|required| !keys.iter().any(|key| key == required))
    {
        return Err(IOError::HeaderSchemaInvalid(
            "header dictionary must contain exactly descr/fortran_order/shape keys",
        ));
    }

    let descr_tail = extract_after_key(dictionary, "descr")?;
    let descr_literal = parse_quoted_value(descr_tail)?;

    let fortran_tail = extract_after_key(dictionary, "fortran_order")?;
    let fortran_order = if fortran_tail.starts_with("True") {
        true
    } else if fortran_tail.starts_with("False") {
        false
    } else {
        return Err(IOError::HeaderSchemaInvalid(
            "fortran_order field must be True or False",
        ));
    };

    let shape_tail = extract_after_key(dictionary, "shape")?;
    let shape_tail = shape_tail
        .strip_prefix('(')
        .ok_or(IOError::HeaderSchemaInvalid(
            "shape field must begin with tuple syntax",
        ))?;
    let shape_end = shape_tail.find(')').ok_or(IOError::HeaderSchemaInvalid(
        "shape tuple missing closing ')'",
    ))?;
    let shape = parse_shape_tuple(&shape_tail[..shape_end])?;

    validate_header_schema(&shape, fortran_order, descr_literal, header_len)
}

fn validate_object_write_payload(shape: &[usize], payload: &[u8]) -> Result<(), IOError> {
    let expected_count = element_count(shape).map_err(|_| {
        IOError::WriteContractViolation("failed to compute element count for object write path")
    })?;
    if expected_count == 0 {
        if payload.is_empty() {
            return Ok(());
        }
        return Err(IOError::WriteContractViolation(
            "zero-sized object payload must be empty",
        ));
    }
    if payload.is_empty() {
        return Err(IOError::WriteContractViolation(
            "object dtype payload requires explicit pickle byte stream",
        ));
    }
    if payload.first().copied() != Some(0x80) {
        return Err(IOError::WriteContractViolation(
            "object dtype payload must start with pickle protocol marker",
        ));
    }
    Ok(())
}

fn validate_object_read_payload(shape: &[usize], payload: &[u8]) -> Result<(), IOError> {
    let expected_count = element_count(shape)
        .map_err(|_| IOError::ReadPayloadIncomplete("failed to compute expected element count"))?;
    if expected_count == 0 {
        if payload.is_empty() {
            return Ok(());
        }
        return Err(IOError::ReadPayloadIncomplete(
            "zero-sized object payload must be empty",
        ));
    }
    if payload.is_empty() {
        return Err(IOError::ReadPayloadIncomplete(
            "object dtype payload requires explicit pickle byte stream",
        ));
    }
    if payload.first().copied() != Some(0x80) {
        return Err(IOError::ReadPayloadIncomplete(
            "object dtype payload must start with pickle protocol marker",
        ));
    }
    Ok(())
}

pub fn write_npy_bytes(
    header: &NpyHeader,
    payload: &[u8],
    allow_pickle: bool,
) -> Result<Vec<u8>, IOError> {
    write_npy_bytes_with_version(header, payload, (1, 0), allow_pickle)
}

pub fn write_npy_bytes_with_version(
    header: &NpyHeader,
    payload: &[u8],
    version: (u8, u8),
    allow_pickle: bool,
) -> Result<Vec<u8>, IOError> {
    validate_npy_version(version)?;
    enforce_pickle_policy(header.descr, allow_pickle)?;
    if header.descr == IOSupportedDType::Object {
        validate_object_write_payload(&header.shape, payload)?;
    } else {
        let item_size = header
            .descr
            .item_size()
            .ok_or(IOError::WriteContractViolation(
                "object dtype requires explicit pickle/object encode path",
            ))?;
        if !payload.len().is_multiple_of(item_size) {
            return Err(IOError::WriteContractViolation(
                "payload bytes must align with dtype item size",
            ));
        }
        let value_count = payload.len() / item_size;
        let _ = validate_write_contract(&header.shape, value_count, header.descr)?;
    }

    let header_bytes = encode_npy_header_bytes(header, version)?;
    let mut encoded = Vec::with_capacity(
        NPY_MAGIC_PREFIX.len()
            + 2
            + npy_length_field_size(version)?
            + header_bytes.len()
            + payload.len(),
    );
    write_npy_preamble(&mut encoded, version, header_bytes.len())?;
    encoded.extend_from_slice(&header_bytes);
    encoded.extend_from_slice(payload);
    Ok(encoded)
}

pub fn read_npy_bytes(payload: &[u8], allow_pickle: bool) -> Result<NpyArrayBytes, IOError> {
    let version = validate_magic_version(payload)?;
    let (header_offset, header_len) = read_header_span(payload, version)?;
    let header_end = header_offset + header_len;
    let header = parse_header_dictionary(&payload[header_offset..header_end], header_len)?;
    let body = &payload[header_end..];

    enforce_pickle_policy(header.descr, allow_pickle)?;
    if header.descr == IOSupportedDType::Object {
        validate_object_read_payload(&header.shape, body)?;
    } else {
        let _ = validate_read_payload(&header.shape, body.len(), header.descr)?;
    }

    Ok(NpyArrayBytes {
        version,
        header,
        payload: body.to_vec(),
    })
}

pub fn validate_magic_version(payload: &[u8]) -> Result<(u8, u8), IOError> {
    if payload.len() < 8 {
        return Err(IOError::MagicInvalid);
    }
    if payload[..6] != NPY_MAGIC_PREFIX {
        return Err(IOError::MagicInvalid);
    }

    let version = (payload[6], payload[7]);
    if version == (1, 0) || version == (2, 0) || version == (3, 0) {
        Ok(version)
    } else {
        Err(IOError::MagicInvalid)
    }
}

pub fn validate_header_schema(
    shape: &[usize],
    fortran_order: bool,
    descr: &str,
    header_len: usize,
) -> Result<NpyHeader, IOError> {
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return Err(IOError::HeaderSchemaInvalid(
            "header bytes must be within bounded budget",
        ));
    }
    if shape.len() > 32 {
        return Err(IOError::HeaderSchemaInvalid(
            "shape rank exceeds packet validation budget",
        ));
    }

    let _ = fortran_order;
    let _ = element_count(shape)?;
    let descr = IOSupportedDType::decode(descr)?;

    Ok(NpyHeader {
        shape: shape.to_vec(),
        fortran_order,
        descr,
    })
}

pub fn validate_descriptor_roundtrip(dtype: IOSupportedDType) -> Result<(), IOError> {
    let encoded = dtype.descr();
    let decoded = IOSupportedDType::decode(encoded)?;
    if decoded == dtype {
        Ok(())
    } else {
        Err(IOError::DTypeDescriptorInvalid)
    }
}

pub fn validate_write_contract(
    shape: &[usize],
    value_count: usize,
    dtype: IOSupportedDType,
) -> Result<usize, IOError> {
    let expected_count = element_count(shape).map_err(|_| {
        IOError::WriteContractViolation("failed to compute element count for write path")
    })?;
    if value_count != expected_count {
        return Err(IOError::WriteContractViolation(
            "value_count does not match shape element count",
        ));
    }

    let item_size = dtype.item_size().ok_or(IOError::WriteContractViolation(
        "object dtype requires explicit pickle/object policy path",
    ))?;

    expected_count
        .checked_mul(item_size)
        .ok_or(IOError::WriteContractViolation(
            "write byte count overflowed",
        ))
}

pub fn validate_read_payload(
    shape: &[usize],
    payload_len_bytes: usize,
    dtype: IOSupportedDType,
) -> Result<usize, IOError> {
    let item_size = dtype.item_size().ok_or(IOError::ReadPayloadIncomplete(
        "object dtype payload requires pickle/object decode path",
    ))?;
    let expected_count = element_count(shape)
        .map_err(|_| IOError::ReadPayloadIncomplete("failed to compute expected element count"))?;
    let expected_bytes =
        expected_count
            .checked_mul(item_size)
            .ok_or(IOError::ReadPayloadIncomplete(
                "expected payload bytes overflowed",
            ))?;

    if payload_len_bytes != expected_bytes {
        return Err(IOError::ReadPayloadIncomplete(
            "payload bytes must exactly match expected shape/dtype footprint",
        ));
    }

    Ok(expected_count)
}

pub fn enforce_pickle_policy(dtype: IOSupportedDType, allow_pickle: bool) -> Result<(), IOError> {
    if dtype == IOSupportedDType::Object && !allow_pickle {
        return Err(IOError::PicklePolicyViolation);
    }
    Ok(())
}

pub fn validate_memmap_contract(
    mode: MemmapMode,
    dtype: IOSupportedDType,
    file_len_bytes: usize,
    expected_bytes: usize,
    validation_retries: usize,
) -> Result<(), IOError> {
    if validation_retries > MAX_MEMMAP_VALIDATION_RETRIES {
        return Err(IOError::MemmapContractViolation(
            "memmap validation retries exceeded bounded budget",
        ));
    }
    if dtype == IOSupportedDType::Object {
        return Err(IOError::MemmapContractViolation(
            "object dtype is invalid for memmap path",
        ));
    }
    if file_len_bytes < expected_bytes {
        return Err(IOError::MemmapContractViolation(
            "backing file is too small for requested mapping",
        ));
    }
    if mode == MemmapMode::Write && expected_bytes == 0 {
        return Err(IOError::MemmapContractViolation(
            "write memmap requires non-empty expected byte footprint",
        ));
    }
    Ok(())
}

pub fn classify_load_dispatch(
    payload_prefix: &[u8],
    allow_pickle: bool,
) -> Result<LoadDispatch, IOError> {
    if payload_prefix.len() >= 4 && payload_prefix[..4] == NPZ_MAGIC_PREFIX {
        return Ok(LoadDispatch::Npz);
    }

    if payload_prefix.len() >= 6 && payload_prefix[..6] == NPY_MAGIC_PREFIX {
        return Ok(LoadDispatch::Npy);
    }

    if allow_pickle && payload_prefix.first().copied() == Some(0x80) {
        return Ok(LoadDispatch::Pickle);
    }

    Err(IOError::LoadDispatchInvalid(
        "payload prefix does not map to allowed npy/npz/pickle branch",
    ))
}

pub fn synthesize_npz_member_names(
    positional_count: usize,
    keyword_names: &[&str],
) -> Result<Vec<String>, IOError> {
    let member_count = positional_count.checked_add(keyword_names.len()).ok_or(
        IOError::NpzArchiveContractViolation("archive member count overflowed"),
    )?;
    if member_count == 0 || member_count > MAX_ARCHIVE_MEMBERS {
        return Err(IOError::NpzArchiveContractViolation(
            "archive member count is outside bounded limits",
        ));
    }

    let mut names = Vec::with_capacity(member_count);
    let mut seen = HashSet::with_capacity(member_count);

    for idx in 0..positional_count {
        let name = format!("arr_{idx}");
        let _ = seen.insert(name.clone());
        names.push(name);
    }

    for &name in keyword_names {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(IOError::NpzArchiveContractViolation(
                "keyword member name cannot be empty",
            ));
        }
        if !seen.insert(trimmed.to_string()) {
            return Err(IOError::NpzArchiveContractViolation(
                "archive member names must be unique",
            ));
        }
        names.push(trimmed.to_string());
    }

    Ok(names)
}

pub fn validate_npz_archive_budget(
    member_count: usize,
    uncompressed_bytes: usize,
    dispatch_retries: usize,
) -> Result<(), IOError> {
    if member_count == 0 || member_count > MAX_ARCHIVE_MEMBERS {
        return Err(IOError::NpzArchiveContractViolation(
            "archive member count is outside bounded limits",
        ));
    }
    if uncompressed_bytes > MAX_ARCHIVE_UNCOMPRESSED_BYTES {
        return Err(IOError::NpzArchiveContractViolation(
            "archive decoded size exceeded bounded budget",
        ));
    }
    if dispatch_retries > MAX_DISPATCH_RETRIES {
        return Err(IOError::LoadDispatchInvalid(
            "dispatch retries exceeded bounded budget",
        ));
    }
    Ok(())
}

// ── loadtxt / savetxt ────────

/// Parsed row-column text data result.
#[derive(Debug, Clone)]
pub struct TextArrayData {
    /// Row-major values.
    pub values: Vec<f64>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
}

/// Load data from a text string (np.loadtxt equivalent).
/// Each line is a row; columns are separated by `delimiter`.
/// Lines starting with `comments` char are skipped.
/// `skiprows` lines are skipped from the start.
/// `max_rows` limits the number of rows read (0 = no limit).
pub fn loadtxt(
    text: &str,
    delimiter: char,
    comments: char,
    skiprows: usize,
    max_rows: usize,
) -> Result<TextArrayData, IOError> {
    let mut values = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut nrows = 0usize;
    for (line_idx, line) in text.lines().enumerate() {
        if line_idx < skiprows { continue; }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) { continue; }
        if max_rows > 0 && nrows >= max_rows { break; }
        let row_vals: Vec<f64> = trimmed
            .split(delimiter)
            .filter(|s| delimiter != ' ' || !s.is_empty())
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| IOError::ReadPayloadIncomplete("loadtxt: parse error in row"))?;
        match ncols {
            None => ncols = Some(row_vals.len()),
            Some(expected) if row_vals.len() != expected => {
                return Err(IOError::ReadPayloadIncomplete("loadtxt: inconsistent number of columns"));
            }
            _ => {}
        }
        values.extend(row_vals);
        nrows += 1;
    }
    Ok(TextArrayData {
        values,
        nrows,
        ncols: ncols.unwrap_or(0),
    })
}

/// Configuration for savetxt.
#[derive(Debug, Clone)]
pub struct SaveTxtConfig<'a> {
    pub delimiter: &'a str,
    pub fmt: &'a str,
    pub header: &'a str,
    pub footer: &'a str,
    pub comments: &'a str,
}

impl Default for SaveTxtConfig<'_> {
    fn default() -> Self {
        Self {
            delimiter: " ",
            fmt: "%g",
            header: "",
            footer: "",
            comments: "#",
        }
    }
}

/// Save data to a text string (np.savetxt equivalent).
/// Writes row-major `values` with shape `(nrows, ncols)`.
pub fn savetxt(
    values: &[f64],
    nrows: usize,
    ncols: usize,
    config: &SaveTxtConfig<'_>,
) -> Result<String, IOError> {
    if values.len() != nrows * ncols {
        return Err(IOError::WriteContractViolation(
            "savetxt: values length != nrows * ncols",
        ));
    }
    let mut output = String::new();
    if !config.header.is_empty() {
        output.push_str(config.comments);
        output.push(' ');
        output.push_str(config.header);
        output.push('\n');
    }
    for r in 0..nrows {
        for c in 0..ncols {
            if c > 0 {
                output.push_str(config.delimiter);
            }
            let v = values[r * ncols + c];
            match config.fmt {
                "%.18e" | "%e" => output.push_str(&format!("{v:e}")),
                "%d" | "%i" => output.push_str(&format!("{}", v as i64)),
                _ => output.push_str(&format!("{v}")),
            }
        }
        output.push('\n');
    }
    if !config.footer.is_empty() {
        output.push_str(config.comments);
        output.push(' ');
        output.push_str(config.footer);
        output.push('\n');
    }
    Ok(output)
}

/// Load data from a text string with more flexible parsing (np.genfromtxt equivalent).
/// Missing values are replaced with `filling_values`.
pub fn genfromtxt(
    text: &str,
    delimiter: char,
    comments: char,
    skip_header: usize,
    filling_values: f64,
) -> Result<TextArrayData, IOError> {
    let mut values = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut nrows = 0usize;
    for (line_idx, line) in text.lines().enumerate() {
        if line_idx < skip_header { continue; }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) { continue; }
        let row_vals: Vec<f64> = trimmed
            .split(delimiter)
            .filter(|s| delimiter != ' ' || !s.is_empty())
            .map(|s| s.trim().parse::<f64>().unwrap_or(filling_values))
            .collect();
        match ncols {
            None => ncols = Some(row_vals.len()),
            Some(expected) if row_vals.len() != expected => {
                // Pad or truncate to match
                let mut padded = row_vals;
                padded.resize(expected, filling_values);
                values.extend(padded);
                nrows += 1;
                continue;
            }
            _ => {}
        }
        values.extend(row_vals);
        nrows += 1;
    }
    Ok(TextArrayData {
        values,
        nrows,
        ncols: ncols.unwrap_or(0),
    })
}

pub fn validate_io_policy_metadata(mode: &str, class: &str) -> Result<(), IOError> {
    let known_mode = mode == "strict" || mode == "hardened";
    let known_class = class == "known_compatible_low_risk"
        || class == "known_compatible_high_risk"
        || class == "known_incompatible_semantics"
        || class == "unknown_semantics";

    if !known_mode || !known_class {
        return Err(IOError::PolicyUnknownMetadata(
            "unknown mode/class metadata rejected fail-closed",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        IO_PACKET_ID, IO_PACKET_REASON_CODES, IOError, IOLogRecord, IORuntimeMode,
        IOSupportedDType, LoadDispatch, MAX_ARCHIVE_MEMBERS, MAX_DISPATCH_RETRIES,
        MAX_HEADER_BYTES, MAX_MEMMAP_VALIDATION_RETRIES, MemmapMode, NPY_MAGIC_PREFIX,
        NPZ_MAGIC_PREFIX, NpyHeader, classify_load_dispatch, encode_npy_header_bytes,
        SaveTxtConfig, enforce_pickle_policy, genfromtxt, loadtxt, read_npy_bytes, savetxt,
        synthesize_npz_member_names,
        validate_descriptor_roundtrip, validate_header_schema, validate_io_policy_metadata,
        validate_magic_version, validate_memmap_contract, validate_npz_archive_budget,
        validate_read_payload, validate_write_contract, write_npy_bytes,
        write_npy_bytes_with_version, write_npy_preamble,
    };

    fn packet009_artifacts() -> Vec<String> {
        vec![
            "artifacts/phase2c/FNP-P2C-009/contract_table.md".to_string(),
            "artifacts/phase2c/FNP-P2C-009/unit_property_evidence.json".to_string(),
        ]
    }

    fn make_manual_npy_payload(header_literal: &str, body: &[u8]) -> Vec<u8> {
        let mut header_bytes = header_literal.as_bytes().to_vec();
        if !header_bytes.ends_with(b"\n") {
            header_bytes.push(b'\n');
        }
        let mut encoded = Vec::new();
        write_npy_preamble(&mut encoded, (1, 0), header_bytes.len()).expect("preamble");
        encoded.extend_from_slice(&header_bytes);
        encoded.extend_from_slice(body);
        encoded
    }

    #[test]
    fn reason_code_registry_matches_packet_contract() {
        assert_eq!(
            IO_PACKET_REASON_CODES,
            [
                "io_magic_invalid",
                "io_header_schema_invalid",
                "io_dtype_descriptor_invalid",
                "io_write_contract_violation",
                "io_read_payload_incomplete",
                "io_pickle_policy_violation",
                "io_memmap_contract_violation",
                "io_load_dispatch_invalid",
                "io_npz_archive_contract_violation",
                "io_policy_unknown_metadata",
            ]
        );
    }

    #[test]
    fn magic_version_accepts_supported_tuples() {
        let mut payload = [0u8; 8];
        payload[..6].copy_from_slice(&NPY_MAGIC_PREFIX);

        for version in [(1, 0), (2, 0), (3, 0)] {
            payload[6] = version.0;
            payload[7] = version.1;
            assert_eq!(
                validate_magic_version(&payload).expect("supported tuple"),
                version
            );
        }
    }

    #[test]
    fn magic_version_rejects_corrupt_prefix_and_unknown_tuple() {
        let err = validate_magic_version(&[0u8; 4]).expect_err("short payload");
        assert_eq!(err.reason_code(), "io_magic_invalid");

        let mut payload = [0u8; 8];
        payload[..6].copy_from_slice(&NPY_MAGIC_PREFIX);
        payload[6] = 9;
        payload[7] = 9;
        let err = validate_magic_version(&payload).expect_err("unsupported tuple");
        assert_eq!(err.reason_code(), "io_magic_invalid");
    }

    #[test]
    fn header_schema_accepts_valid_and_rejects_invalid_budget() {
        let header = validate_header_schema(&[2, 3], false, "<f8", 128).expect("valid header");
        assert_eq!(header.shape, vec![2, 3]);
        assert_eq!(header.descr, IOSupportedDType::F64);

        let err = validate_header_schema(&[2, 3], true, "<f8", MAX_HEADER_BYTES + 1)
            .expect_err("oversized header");
        assert_eq!(err.reason_code(), "io_header_schema_invalid");
    }

    #[test]
    fn descriptor_roundtrip_covers_all_supported_dtypes() {
        let dtypes = [
            IOSupportedDType::Bool,
            IOSupportedDType::I32,
            IOSupportedDType::I64,
            IOSupportedDType::F32,
            IOSupportedDType::F64,
            IOSupportedDType::Object,
        ];

        for dtype in dtypes {
            validate_descriptor_roundtrip(dtype).expect("descriptor roundtrip");
        }

        let err = IOSupportedDType::decode(">i4").expect_err("unsupported descriptor");
        assert_eq!(err.reason_code(), "io_dtype_descriptor_invalid");
    }

    #[test]
    fn write_contract_property_grid_is_deterministic() {
        for seed in 1usize..=128usize {
            let rows = (seed % 17) + 1;
            let cols = (seed % 11) + 1;
            let shape = [rows, cols];
            let value_count = rows * cols;
            let bytes_first =
                validate_write_contract(&shape, value_count, IOSupportedDType::F64).expect("write");
            let bytes_second =
                validate_write_contract(&shape, value_count, IOSupportedDType::F64).expect("write");
            assert_eq!(bytes_first, bytes_second);
            assert_eq!(bytes_first, value_count * 8);
        }
    }

    #[test]
    fn write_contract_rejects_count_mismatch() {
        let err = validate_write_contract(&[3, 3], 8, IOSupportedDType::F64)
            .expect_err("shape/count mismatch");
        assert_eq!(err.reason_code(), "io_write_contract_violation");
    }

    #[test]
    fn read_payload_requires_exact_shape_footprint() {
        let count = validate_read_payload(&[2, 3], 6 * 8, IOSupportedDType::F64).expect("valid");
        assert_eq!(count, 6);

        let short = validate_read_payload(&[2, 3], 5 * 8, IOSupportedDType::F64)
            .expect_err("truncated payload");
        assert_eq!(short.reason_code(), "io_read_payload_incomplete");

        let long = validate_read_payload(&[2, 3], 7 * 8, IOSupportedDType::F64)
            .expect_err("extra trailing bytes");
        assert_eq!(long.reason_code(), "io_read_payload_incomplete");
    }

    #[test]
    fn npy_bytes_roundtrip_preserves_header_and_payload() {
        let header = NpyHeader {
            shape: vec![2, 2],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let payload = [1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>();

        let encoded = write_npy_bytes(&header, &payload, false).expect("encode npy bytes");
        let decoded = read_npy_bytes(&encoded, false).expect("decode npy bytes");
        assert_eq!(decoded.version, (1, 0));
        assert_eq!(decoded.header, header);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn npy_writer_rejects_payload_item_size_misalignment() {
        let header = NpyHeader {
            shape: vec![1],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let err = write_npy_bytes(&header, &[0u8; 7], false)
            .expect_err("payload bytes must align with item size");
        assert_eq!(err.reason_code(), "io_write_contract_violation");
    }

    #[test]
    fn npy_reader_rejects_payload_count_mismatch() {
        let header = NpyHeader {
            shape: vec![2, 2],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let payload = vec![0u8; 4 * 8];
        let mut encoded = write_npy_bytes(&header, &payload, false).expect("encode");
        let _ = encoded.pop();

        let err = read_npy_bytes(&encoded, false).expect_err("payload footprint mismatch");
        assert_eq!(err.reason_code(), "io_read_payload_incomplete");
    }

    #[test]
    fn npy_reader_rejects_truncated_header_region() {
        let header = NpyHeader {
            shape: vec![2, 2],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let payload = vec![0u8; 4 * 8];
        let mut encoded = write_npy_bytes(&header, &payload, false).expect("encode");
        encoded[8] = 0xFF;
        encoded[9] = 0x7F;
        encoded.truncate(64);

        let err = read_npy_bytes(&encoded, false).expect_err("declared header exceeds payload");
        assert_eq!(err.reason_code(), "io_header_schema_invalid");
    }

    #[test]
    fn npy_object_dtype_is_policy_gated_on_read() {
        let header = NpyHeader {
            shape: vec![1],
            fortran_order: false,
            descr: IOSupportedDType::Object,
        };
        let header_bytes = encode_npy_header_bytes(&header, (1, 0)).expect("header bytes");
        let mut encoded = Vec::new();
        write_npy_preamble(&mut encoded, (1, 0), header_bytes.len()).expect("preamble");
        encoded.extend_from_slice(&header_bytes);
        encoded.extend_from_slice(&[0x80, 0x05, 0x4B, 0x01, 0x2E]);

        let err = read_npy_bytes(&encoded, false).expect_err("pickle policy should reject");
        assert_eq!(err.reason_code(), "io_pickle_policy_violation");

        let decoded = read_npy_bytes(&encoded, true).expect("allow_pickle read");
        assert_eq!(decoded.header.descr, IOSupportedDType::Object);
        assert_eq!(decoded.payload, vec![0x80, 0x05, 0x4B, 0x01, 0x2E]);
    }

    #[test]
    fn npy_v2_writer_roundtrip_is_supported() {
        let header = NpyHeader {
            shape: vec![3],
            fortran_order: true,
            descr: IOSupportedDType::I32,
        };
        let payload = [10_i32, 20_i32, 30_i32]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();

        let encoded =
            write_npy_bytes_with_version(&header, &payload, (2, 0), false).expect("write v2");
        let decoded = read_npy_bytes(&encoded, false).expect("read v2");
        assert_eq!(decoded.version, (2, 0));
        assert_eq!(decoded.header, header);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn npy_header_parser_rejects_extra_keys_and_singleton_without_comma() {
        let payload = [10_i32, 20_i32]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();

        let extra_key_header =
            "{'descr': '<i4', 'fortran_order': False, 'shape': (2,), 'extra': 1, }";
        let extra_key_bytes = make_manual_npy_payload(extra_key_header, &payload);
        let extra_key_err =
            read_npy_bytes(&extra_key_bytes, false).expect_err("extra key must be rejected");
        assert_eq!(extra_key_err.reason_code(), "io_header_schema_invalid");

        let singleton_without_comma = "{'descr': '<i4', 'fortran_order': False, 'shape': (2), }";
        let singleton_without_comma_bytes =
            make_manual_npy_payload(singleton_without_comma, &payload);
        let singleton_err = read_npy_bytes(&singleton_without_comma_bytes, false)
            .expect_err("singleton tuple without trailing comma must be rejected");
        assert_eq!(singleton_err.reason_code(), "io_header_schema_invalid");
    }

    #[test]
    fn object_write_path_is_policy_gated_and_requires_pickle_marker() {
        let header = NpyHeader {
            shape: vec![1],
            fortran_order: false,
            descr: IOSupportedDType::Object,
        };

        let policy_err =
            write_npy_bytes(&header, &[0x80, 0x05, 0x4B, 0x01, 0x2E], false).expect_err("policy");
        assert_eq!(policy_err.reason_code(), "io_pickle_policy_violation");

        let marker_err = write_npy_bytes(&header, b"not-pickle", true)
            .expect_err("object payload must carry pickle marker");
        assert_eq!(marker_err.reason_code(), "io_write_contract_violation");

        let encoded =
            write_npy_bytes(&header, &[0x80, 0x05, 0x4B, 0x01, 0x2E], true).expect("object write");
        let decoded = read_npy_bytes(&encoded, true).expect("object read");
        assert_eq!(decoded.header.descr, IOSupportedDType::Object);
    }

    #[test]
    fn zero_sized_object_payload_must_be_empty() {
        let header = NpyHeader {
            shape: vec![0],
            fortran_order: false,
            descr: IOSupportedDType::Object,
        };

        write_npy_bytes(&header, &[], true).expect("zero-sized object payload may be empty");

        let err = write_npy_bytes(&header, &[0x80], true)
            .expect_err("zero-sized object payload must be empty");
        assert_eq!(err.reason_code(), "io_write_contract_violation");
    }

    #[test]
    fn pickle_policy_gate_rejects_object_when_disallowed() {
        let err = enforce_pickle_policy(IOSupportedDType::Object, false)
            .expect_err("object payload must be rejected");
        assert_eq!(err.reason_code(), "io_pickle_policy_violation");
        enforce_pickle_policy(IOSupportedDType::Object, true).expect("explicit allow_pickle");
    }

    #[test]
    fn memmap_contract_enforces_dtype_mode_and_retry_budget() {
        validate_memmap_contract(MemmapMode::ReadOnly, IOSupportedDType::F64, 4096, 1024, 0)
            .expect("valid memmap");
        let parsed = MemmapMode::parse("r+").expect("valid mode parse");
        assert_eq!(parsed, MemmapMode::ReadWrite);

        let object_err = validate_memmap_contract(
            MemmapMode::ReadOnly,
            IOSupportedDType::Object,
            4096,
            1024,
            0,
        )
        .expect_err("object memmap is invalid");
        assert_eq!(object_err.reason_code(), "io_memmap_contract_violation");

        let retry_err = validate_memmap_contract(
            MemmapMode::ReadOnly,
            IOSupportedDType::F64,
            4096,
            1024,
            MAX_MEMMAP_VALIDATION_RETRIES + 1,
        )
        .expect_err("retry budget exceeded");
        assert_eq!(retry_err.reason_code(), "io_memmap_contract_violation");
    }

    #[test]
    fn load_dispatch_selects_expected_branches() {
        let npz = classify_load_dispatch(&NPZ_MAGIC_PREFIX, false).expect("npz branch");
        assert_eq!(npz, LoadDispatch::Npz);

        let npy = classify_load_dispatch(&NPY_MAGIC_PREFIX, false).expect("npy branch");
        assert_eq!(npy, LoadDispatch::Npy);

        let pickle = classify_load_dispatch(&[0x80, 0x05, 0x95], true).expect("pickle branch");
        assert_eq!(pickle, LoadDispatch::Pickle);

        let err = classify_load_dispatch(&[0x80, 0x05, 0x95], false).expect_err("policy reject");
        assert_eq!(err.reason_code(), "io_load_dispatch_invalid");
    }

    #[test]
    fn npz_member_name_contract_enforces_uniqueness_and_budget() {
        let names =
            synthesize_npz_member_names(2, &["weights", "bias"]).expect("valid member names");
        assert_eq!(
            names,
            vec![
                "arr_0".to_string(),
                "arr_1".to_string(),
                "weights".to_string(),
                "bias".to_string()
            ]
        );

        let duplicate = synthesize_npz_member_names(1, &["arr_0"]).expect_err("duplicate name");
        assert_eq!(duplicate.reason_code(), "io_npz_archive_contract_violation");

        let too_many = synthesize_npz_member_names(MAX_ARCHIVE_MEMBERS + 1, &[])
            .expect_err("member budget exceeded");
        assert_eq!(too_many.reason_code(), "io_npz_archive_contract_violation");
    }

    #[test]
    fn npz_archive_budget_enforces_limits() {
        validate_npz_archive_budget(4, 1024, MAX_DISPATCH_RETRIES).expect("budget within limits");

        let huge = validate_npz_archive_budget(4, usize::MAX, 0).expect_err("decoded size too big");
        assert_eq!(huge.reason_code(), "io_npz_archive_contract_violation");

        let retries =
            validate_npz_archive_budget(4, 1024, MAX_DISPATCH_RETRIES + 1).expect_err("retries");
        assert_eq!(retries.reason_code(), "io_load_dispatch_invalid");
    }

    #[test]
    fn policy_metadata_is_fail_closed_for_unknowns() {
        validate_io_policy_metadata("strict", "known_compatible_low_risk").expect("known strict");
        validate_io_policy_metadata("hardened", "unknown_semantics").expect("known hardened");

        let err = validate_io_policy_metadata("mystery", "known_compatible_low_risk")
            .expect_err("unknown mode");
        assert_eq!(err.reason_code(), "io_policy_unknown_metadata");
    }

    #[test]
    fn packet009_log_record_is_replay_complete() {
        let record = IOLogRecord {
            ts_utc: "2026-02-16T00:00:00Z".to_string(),
            suite_id: "fnp-io::tests".to_string(),
            test_id: "UP-009-header-schema".to_string(),
            packet_id: IO_PACKET_ID.to_string(),
            fixture_id: "UP-009-header-schema".to_string(),
            mode: IORuntimeMode::Strict,
            seed: 9009,
            input_digest: "sha256:input".to_string(),
            output_digest: "sha256:output".to_string(),
            env_fingerprint: "fnp-io-unit-tests".to_string(),
            artifact_refs: packet009_artifacts(),
            duration_ms: 2,
            outcome: "pass".to_string(),
            reason_code: "io_header_schema_invalid".to_string(),
        };
        assert!(record.is_replay_complete());
    }

    #[test]
    fn packet009_log_record_rejects_missing_fields() {
        let record = IOLogRecord {
            ts_utc: String::new(),
            suite_id: String::new(),
            test_id: String::new(),
            packet_id: "wrong-packet".to_string(),
            fixture_id: String::new(),
            mode: IORuntimeMode::Hardened,
            seed: 9010,
            input_digest: String::new(),
            output_digest: String::new(),
            env_fingerprint: String::new(),
            artifact_refs: vec![String::new()],
            duration_ms: 0,
            outcome: "unknown".to_string(),
            reason_code: String::new(),
        };
        assert!(!record.is_replay_complete());
    }

    #[test]
    fn packet009_reason_codes_round_trip_into_logs() {
        for (idx, reason_code) in IO_PACKET_REASON_CODES.iter().enumerate() {
            let seed = u64::try_from(idx).expect("small index");
            let record = IOLogRecord {
                ts_utc: "2026-02-16T00:00:00Z".to_string(),
                suite_id: "fnp-io::tests".to_string(),
                test_id: format!("UP-009-{idx}"),
                packet_id: IO_PACKET_ID.to_string(),
                fixture_id: format!("UP-009-{idx}"),
                mode: IORuntimeMode::Strict,
                seed: 20_000 + seed,
                input_digest: "sha256:input".to_string(),
                output_digest: "sha256:output".to_string(),
                env_fingerprint: "fnp-io-unit-tests".to_string(),
                artifact_refs: packet009_artifacts(),
                duration_ms: 1,
                outcome: "pass".to_string(),
                reason_code: (*reason_code).to_string(),
            };
            assert!(record.is_replay_complete());
            assert_eq!(record.reason_code, *reason_code);
        }
    }

    #[test]
    fn io_error_reason_code_mapping_is_stable() {
        let err = IOError::HeaderSchemaInvalid("bad header");
        assert_eq!(err.reason_code(), "io_header_schema_invalid");
        let err = IOError::MagicInvalid;
        assert_eq!(err.reason_code(), "io_magic_invalid");
    }

    // ── loadtxt / savetxt tests ────────

    #[test]
    fn loadtxt_basic() {
        let text = "1.0 2.0 3.0\n4.0 5.0 6.0\n";
        let result = loadtxt(text, ' ', '#', 0, 0).unwrap();
        assert_eq!(result.nrows, 2);
        assert_eq!(result.ncols, 3);
        assert_eq!(result.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn loadtxt_csv() {
        let text = "1,2,3\n4,5,6";
        let result = loadtxt(text, ',', '#', 0, 0).unwrap();
        assert_eq!(result.nrows, 2);
        assert_eq!(result.ncols, 3);
    }

    #[test]
    fn loadtxt_comments_and_skiprows() {
        let text = "# header\n# another comment\n1 2\n3 4\n";
        let result = loadtxt(text, ' ', '#', 0, 0).unwrap();
        assert_eq!(result.nrows, 2);
        assert_eq!(result.values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn loadtxt_skiprows() {
        let text = "header line\n1 2\n3 4\n";
        let result = loadtxt(text, ' ', '#', 1, 0).unwrap();
        assert_eq!(result.nrows, 2);
    }

    #[test]
    fn loadtxt_max_rows() {
        let text = "1 2\n3 4\n5 6\n";
        let result = loadtxt(text, ' ', '#', 0, 2).unwrap();
        assert_eq!(result.nrows, 2);
    }

    #[test]
    fn savetxt_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output = savetxt(&values, 2, 3, &SaveTxtConfig::default()).unwrap();
        assert!(output.contains("1"));
        assert!(output.contains("6"));
        assert_eq!(output.lines().count(), 2);
    }

    #[test]
    fn savetxt_with_header() {
        let values = vec![1.0, 2.0];
        let cfg = SaveTxtConfig { delimiter: ",", header: "x,y", ..SaveTxtConfig::default() };
        let output = savetxt(&values, 1, 2, &cfg).unwrap();
        assert!(output.starts_with("# x,y\n"));
    }

    #[test]
    fn savetxt_roundtrip() {
        let original = vec![1.5, 2.5, 3.5, 4.5];
        let text = savetxt(&original, 2, 2, &SaveTxtConfig::default()).unwrap();
        let loaded = loadtxt(&text, ' ', '#', 0, 0).unwrap();
        assert_eq!(loaded.nrows, 2);
        assert_eq!(loaded.ncols, 2);
        assert_eq!(loaded.values, original);
    }

    #[test]
    fn genfromtxt_with_missing() {
        let text = "1,2,3\n4,,6\n";
        let result = genfromtxt(text, ',', '#', 0, f64::NAN).unwrap();
        assert_eq!(result.nrows, 2);
        assert_eq!(result.ncols, 3);
        assert_eq!(result.values[0], 1.0);
        assert!(result.values[4].is_nan()); // missing value
        assert_eq!(result.values[5], 6.0);
    }

    #[test]
    fn genfromtxt_skip_header() {
        let text = "col1,col2\n1,2\n3,4\n";
        let result = genfromtxt(text, ',', '#', 1, 0.0).unwrap();
        assert_eq!(result.nrows, 2);
        assert_eq!(result.values, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
