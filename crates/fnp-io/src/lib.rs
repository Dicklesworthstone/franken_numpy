#![forbid(unsafe_code)]

use core::fmt;
use std::collections::HashSet;
use std::io::{Read, Write};

use flate2::{Compression, read::DeflateDecoder, write::DeflateEncoder};

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
    I8,
    I16,
    I16Be,
    I32,
    I32Be,
    I64,
    I64Be,
    U8,
    U16,
    U16Be,
    U32,
    U32Be,
    U64,
    U64Be,
    F32,
    F32Be,
    F64,
    F64Be,
    Object,
}

impl IOSupportedDType {
    #[must_use]
    pub const fn descr(self) -> &'static str {
        match self {
            Self::Bool => "|b1",
            Self::I8 => "|i1",
            Self::I16 => "<i2",
            Self::I16Be => ">i2",
            Self::I32 => "<i4",
            Self::I32Be => ">i4",
            Self::I64 => "<i8",
            Self::I64Be => ">i8",
            Self::U8 => "|u1",
            Self::U16 => "<u2",
            Self::U16Be => ">u2",
            Self::U32 => "<u4",
            Self::U32Be => ">u4",
            Self::U64 => "<u8",
            Self::U64Be => ">u8",
            Self::F32 => "<f4",
            Self::F32Be => ">f4",
            Self::F64 => "<f8",
            Self::F64Be => ">f8",
            Self::Object => "|O",
        }
    }

    pub fn decode(descr: &str) -> Result<Self, IOError> {
        match descr {
            "|b1" => Ok(Self::Bool),
            "|i1" => Ok(Self::I8),
            "<i2" => Ok(Self::I16),
            ">i2" => Ok(Self::I16Be),
            "<i4" => Ok(Self::I32),
            ">i4" => Ok(Self::I32Be),
            "<i8" => Ok(Self::I64),
            ">i8" => Ok(Self::I64Be),
            "|u1" => Ok(Self::U8),
            "<u2" => Ok(Self::U16),
            ">u2" => Ok(Self::U16Be),
            "<u4" => Ok(Self::U32),
            ">u4" => Ok(Self::U32Be),
            "<u8" => Ok(Self::U64),
            ">u8" => Ok(Self::U64Be),
            "<f4" => Ok(Self::F32),
            ">f4" => Ok(Self::F32Be),
            "<f8" => Ok(Self::F64),
            ">f8" => Ok(Self::F64Be),
            "|O" => Ok(Self::Object),
            _ => Err(IOError::DTypeDescriptorInvalid),
        }
    }

    #[must_use]
    pub const fn item_size(self) -> Option<usize> {
        match self {
            Self::Bool | Self::I8 | Self::U8 => Some(1),
            Self::I16 | Self::I16Be | Self::U16 | Self::U16Be => Some(2),
            Self::I32 | Self::I32Be | Self::U32 | Self::U32Be | Self::F32 | Self::F32Be => Some(4),
            Self::I64 | Self::I64Be | Self::U64 | Self::U64Be | Self::F64 | Self::F64Be => Some(8),
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

// ── NPZ read/write ────────

/// A named array inside an NPZ archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpzEntry {
    pub name: String,
    pub array: NpyArrayBytes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpzCompression {
    Store,
    Deflate,
}

/// Write multiple named arrays into an uncompressed NPZ archive (np.savez).
///
/// NPZ is a ZIP file containing .npy files. Each entry is stored without
/// compression (STORE method). The entry name gets `.npy` appended if it
/// doesn't already end with it.
pub fn write_npz_bytes(entries: &[(&str, &NpyHeader, &[u8])]) -> Result<Vec<u8>, IOError> {
    write_npz_bytes_with_compression(entries, NpzCompression::Store)
}

/// Write multiple named arrays into an NPZ archive with optional compression.
///
/// Supports ZIP STORE (method 0) and DEFLATE (method 8).
pub fn write_npz_bytes_with_compression(
    entries: &[(&str, &NpyHeader, &[u8])],
    compression: NpzCompression,
) -> Result<Vec<u8>, IOError> {
    if entries.is_empty() {
        return Err(IOError::NpzArchiveContractViolation(
            "npz: cannot write archive with zero entries",
        ));
    }
    if entries.len() > MAX_ARCHIVE_MEMBERS {
        return Err(IOError::NpzArchiveContractViolation(
            "npz: member count exceeds bounded limit",
        ));
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut central_directory: Vec<u8> = Vec::new();
    let mut entry_count: u16 = 0;

    for &(name, header, payload) in entries {
        let npy_data = write_npy_bytes(header, payload, false)?;
        let file_name = if name.ends_with(".npy") {
            name.to_string()
        } else {
            format!("{name}.npy")
        };
        let fname_bytes = file_name.as_bytes();

        let local_offset = buf.len() as u32;
        let crc = crc32_ieee(&npy_data);
        let encoded_data = match compression {
            NpzCompression::Store => npy_data.clone(),
            NpzCompression::Deflate => {
                let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(&npy_data).map_err(|_| {
                    IOError::NpzArchiveContractViolation(
                        "npz: failed to deflate-compress entry payload",
                    )
                })?;
                encoder.finish().map_err(|_| {
                    IOError::NpzArchiveContractViolation(
                        "npz: failed to finalize deflate-compressed entry payload",
                    )
                })?
            }
        };
        let compression_method = match compression {
            NpzCompression::Store => 0_u16,
            NpzCompression::Deflate => 8_u16,
        };
        let compressed_size = encoded_data.len() as u32;
        let uncompressed_size = npy_data.len() as u32;

        // Local file header (30 bytes + filename)
        buf.extend_from_slice(&[0x50, 0x4B, 0x03, 0x04]); // signature
        buf.extend_from_slice(&20_u16.to_le_bytes()); // version needed (2.0)
        buf.extend_from_slice(&0_u16.to_le_bytes()); // flags
        buf.extend_from_slice(&compression_method.to_le_bytes());
        buf.extend_from_slice(&0_u16.to_le_bytes()); // mod time
        buf.extend_from_slice(&0_u16.to_le_bytes()); // mod date
        buf.extend_from_slice(&crc.to_le_bytes()); // crc-32
        buf.extend_from_slice(&compressed_size.to_le_bytes());
        buf.extend_from_slice(&uncompressed_size.to_le_bytes());
        buf.extend_from_slice(&(fname_bytes.len() as u16).to_le_bytes()); // filename len
        buf.extend_from_slice(&0_u16.to_le_bytes()); // extra field len
        buf.extend_from_slice(fname_bytes);
        buf.extend_from_slice(&encoded_data);

        // Central directory entry (46 bytes + filename)
        central_directory.extend_from_slice(&[0x50, 0x4B, 0x01, 0x02]); // signature
        central_directory.extend_from_slice(&20_u16.to_le_bytes()); // version made by
        central_directory.extend_from_slice(&20_u16.to_le_bytes()); // version needed
        central_directory.extend_from_slice(&0_u16.to_le_bytes()); // flags
        central_directory.extend_from_slice(&compression_method.to_le_bytes());
        central_directory.extend_from_slice(&0_u16.to_le_bytes()); // mod time
        central_directory.extend_from_slice(&0_u16.to_le_bytes()); // mod date
        central_directory.extend_from_slice(&crc.to_le_bytes()); // crc-32
        central_directory.extend_from_slice(&compressed_size.to_le_bytes());
        central_directory.extend_from_slice(&uncompressed_size.to_le_bytes());
        central_directory.extend_from_slice(&(fname_bytes.len() as u16).to_le_bytes());
        central_directory.extend_from_slice(&0_u16.to_le_bytes()); // extra field len
        central_directory.extend_from_slice(&0_u16.to_le_bytes()); // comment len
        central_directory.extend_from_slice(&0_u16.to_le_bytes()); // disk number
        central_directory.extend_from_slice(&0_u16.to_le_bytes()); // internal attrs
        central_directory.extend_from_slice(&0_u32.to_le_bytes()); // external attrs
        central_directory.extend_from_slice(&local_offset.to_le_bytes()); // offset
        central_directory.extend_from_slice(fname_bytes);

        entry_count += 1;
    }

    let cd_offset = buf.len() as u32;
    buf.extend_from_slice(&central_directory);
    let cd_size = central_directory.len() as u32;

    // End of central directory record (22 bytes)
    buf.extend_from_slice(&[0x50, 0x4B, 0x05, 0x06]); // signature
    buf.extend_from_slice(&0_u16.to_le_bytes()); // disk number
    buf.extend_from_slice(&0_u16.to_le_bytes()); // disk with CD
    buf.extend_from_slice(&entry_count.to_le_bytes()); // entries on disk
    buf.extend_from_slice(&entry_count.to_le_bytes()); // total entries
    buf.extend_from_slice(&cd_size.to_le_bytes()); // CD size
    buf.extend_from_slice(&cd_offset.to_le_bytes()); // CD offset
    buf.extend_from_slice(&0_u16.to_le_bytes()); // comment length

    Ok(buf)
}

/// Read an NPZ archive and return all named arrays (np.load for .npz files).
///
/// Supports uncompressed STORE (method 0) and DEFLATE-compressed (method 8)
/// entries. Each entry must decode to a valid `.npy` file.
pub fn read_npz_bytes(data: &[u8]) -> Result<Vec<NpzEntry>, IOError> {
    if data.len() < 22 {
        return Err(IOError::NpzArchiveContractViolation(
            "npz: data too short for a ZIP archive",
        ));
    }
    if data[..4] != NPZ_MAGIC_PREFIX {
        return Err(IOError::NpzArchiveContractViolation(
            "npz: not a valid ZIP/NPZ archive",
        ));
    }

    // Find End of Central Directory (scan backwards for signature)
    let mut eocd_pos = None;
    for i in (0..data.len().saturating_sub(21)).rev() {
        if data[i..i + 4] == [0x50, 0x4B, 0x05, 0x06] {
            eocd_pos = Some(i);
            break;
        }
    }
    let eocd = eocd_pos.ok_or(IOError::NpzArchiveContractViolation(
        "npz: cannot find end of central directory",
    ))?;

    let entry_count = u16::from_le_bytes([data[eocd + 10], data[eocd + 11]]) as usize;
    let cd_size = u32::from_le_bytes([
        data[eocd + 12],
        data[eocd + 13],
        data[eocd + 14],
        data[eocd + 15],
    ]) as usize;
    let cd_offset = u32::from_le_bytes([
        data[eocd + 16],
        data[eocd + 17],
        data[eocd + 18],
        data[eocd + 19],
    ]) as usize;

    let cd_end = cd_offset
        .checked_add(cd_size)
        .ok_or(IOError::NpzArchiveContractViolation(
            "npz: central directory bounds overflow",
        ))?;
    if cd_end > data.len() {
        return Err(IOError::NpzArchiveContractViolation(
            "npz: central directory extends beyond archive bounds",
        ));
    }

    validate_npz_archive_budget(entry_count, 0, 0)?;

    let mut entries = Vec::with_capacity(entry_count);
    let mut pos = cd_offset;
    let mut total_uncompressed_bytes = 0usize;

    for _ in 0..entry_count {
        if pos + 46 > data.len() {
            return Err(IOError::NpzArchiveContractViolation(
                "npz: central directory truncated",
            ));
        }
        if data[pos..pos + 4] != [0x50, 0x4B, 0x01, 0x02] {
            return Err(IOError::NpzArchiveContractViolation(
                "npz: invalid central directory entry signature",
            ));
        }

        let compression = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
        if compression != 0 && compression != 8 {
            return Err(IOError::NpzArchiveContractViolation(
                "npz: only STORE (0) and DEFLATE (8) entries are supported",
            ));
        }

        let crc = u32::from_le_bytes([
            data[pos + 16],
            data[pos + 17],
            data[pos + 18],
            data[pos + 19],
        ]);

        let compressed_size = u32::from_le_bytes([
            data[pos + 20],
            data[pos + 21],
            data[pos + 22],
            data[pos + 23],
        ]) as usize;
        let uncompressed_size = u32::from_le_bytes([
            data[pos + 24],
            data[pos + 25],
            data[pos + 26],
            data[pos + 27],
        ]) as usize;
        let fname_len = u16::from_le_bytes([data[pos + 28], data[pos + 29]]) as usize;
        let extra_len = u16::from_le_bytes([data[pos + 30], data[pos + 31]]) as usize;
        let comment_len = u16::from_le_bytes([data[pos + 32], data[pos + 33]]) as usize;
        let local_offset = u32::from_le_bytes([
            data[pos + 42],
            data[pos + 43],
            data[pos + 44],
            data[pos + 45],
        ]) as usize;

        let fname_start = pos + 46;
        if fname_start + fname_len > data.len() {
            return Err(IOError::NpzArchiveContractViolation(
                "npz: filename extends beyond data",
            ));
        }
        let file_name =
            String::from_utf8_lossy(&data[fname_start..fname_start + fname_len]).into_owned();

        // Parse local file header to find data start
        if local_offset + 30 > data.len() {
            return Err(IOError::NpzArchiveContractViolation(
                "npz: local header offset out of bounds",
            ));
        }
        let local_signature = &data[local_offset..local_offset + 4];
        if local_signature != [0x50, 0x4B, 0x03, 0x04] {
            return Err(IOError::NpzArchiveContractViolation(
                "npz: invalid local file header signature",
            ));
        }
        let local_compression =
            u16::from_le_bytes([data[local_offset + 8], data[local_offset + 9]]);
        if local_compression != compression {
            return Err(IOError::NpzArchiveContractViolation(
                "npz: local/central directory compression mismatch",
            ));
        }
        let local_fname_len =
            u16::from_le_bytes([data[local_offset + 26], data[local_offset + 27]]) as usize;
        let local_extra_len =
            u16::from_le_bytes([data[local_offset + 28], data[local_offset + 29]]) as usize;
        let data_start = local_offset + 30 + local_fname_len + local_extra_len;
        let data_end = data_start + compressed_size;

        if data_end > data.len() {
            return Err(IOError::NpzArchiveContractViolation(
                "npz: entry data extends beyond archive",
            ));
        }

        let stored_entry_bytes = &data[data_start..data_end];
        let npy_bytes = match compression {
            0 => {
                if compressed_size != uncompressed_size {
                    return Err(IOError::NpzArchiveContractViolation(
                        "npz: STORE entry has inconsistent compressed/uncompressed sizes",
                    ));
                }
                stored_entry_bytes.to_vec()
            }
            8 => {
                let mut decoder = DeflateDecoder::new(stored_entry_bytes);
                let mut decoded = Vec::new();
                decoder.read_to_end(&mut decoded).map_err(|_| {
                    IOError::NpzArchiveContractViolation(
                        "npz: failed to inflate DEFLATE entry payload",
                    )
                })?;
                decoded
            }
            _ => unreachable!("compression validated before branch"),
        };
        if npy_bytes.len() != uncompressed_size {
            return Err(IOError::NpzArchiveContractViolation(
                "npz: decoded entry length does not match declared uncompressed size",
            ));
        }
        if crc32_ieee(&npy_bytes) != crc {
            return Err(IOError::NpzArchiveContractViolation(
                "npz: decoded entry CRC-32 does not match central directory",
            ));
        }

        total_uncompressed_bytes = total_uncompressed_bytes
            .checked_add(uncompressed_size)
            .ok_or(IOError::NpzArchiveContractViolation(
                "npz: decoded archive size overflowed bounded budget",
            ))?;
        validate_npz_archive_budget(entry_count, total_uncompressed_bytes, 0)?;
        let array = read_npy_bytes(&npy_bytes, false)?;

        // Strip .npy suffix from name for user convenience
        let clean_name = file_name
            .strip_suffix(".npy")
            .unwrap_or(&file_name)
            .to_string();

        entries.push(NpzEntry {
            name: clean_name,
            array,
        });

        pos = fname_start + fname_len + extra_len + comment_len;
    }

    Ok(entries)
}

/// IEEE 802.3 CRC-32 (used by ZIP format).
fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
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
    loadtxt_usecols(text, delimiter, comments, skiprows, max_rows, None)
}

/// Load data from text with optional column selection (np.loadtxt with usecols).
/// `usecols` selects specific columns by zero-based index. When `None`, all columns are loaded.
pub fn loadtxt_usecols(
    text: &str,
    delimiter: char,
    comments: char,
    skiprows: usize,
    max_rows: usize,
    usecols: Option<&[usize]>,
) -> Result<TextArrayData, IOError> {
    let mut values = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut nrows = 0usize;
    for (line_idx, line) in text.lines().enumerate() {
        if line_idx < skiprows {
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) {
            continue;
        }
        if max_rows > 0 && nrows >= max_rows {
            break;
        }
        let all_vals: Vec<f64> = trimmed
            .split(delimiter)
            .filter(|s| delimiter != ' ' || !s.is_empty())
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| IOError::ReadPayloadIncomplete("loadtxt: parse error in row"))?;

        let row_vals = if let Some(cols) = usecols {
            let mut selected = Vec::with_capacity(cols.len());
            for &col in cols {
                if col >= all_vals.len() {
                    return Err(IOError::ReadPayloadIncomplete(
                        "loadtxt: usecols index out of bounds",
                    ));
                }
                selected.push(all_vals[col]);
            }
            selected
        } else {
            all_vals
        };

        match ncols {
            None => ncols = Some(row_vals.len()),
            Some(expected) if row_vals.len() != expected => {
                return Err(IOError::ReadPayloadIncomplete(
                    "loadtxt: inconsistent number of columns",
                ));
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
        if line_idx < skip_header {
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) {
            continue;
        }
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

/// Read raw binary data from bytes into a flat array of f64 values (np.fromfile).
///
/// Interprets `data` as a sequence of elements of the given `dtype`.
/// If `count` is Some, reads at most that many elements.
/// Returns the decoded f64 values.
pub fn fromfile(
    data: &[u8],
    dtype: IOSupportedDType,
    count: Option<usize>,
) -> Result<Vec<f64>, IOError> {
    let item_size = dtype
        .item_size()
        .ok_or(IOError::DTypeDescriptorInvalid)?;
    if item_size == 0 {
        return Ok(Vec::new());
    }
    let max_elems = data.len() / item_size;
    let n = match count {
        Some(c) => c.min(max_elems),
        None => max_elems,
    };

    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * item_size;
        let chunk = &data[offset..offset + item_size];
        let v = decode_element(chunk, dtype)?;
        values.push(v);
    }
    Ok(values)
}

/// Write array values to raw binary bytes (np.ndarray.tofile).
///
/// Encodes each f64 value according to the given `dtype` and writes
/// the binary representation.
pub fn tofile(
    values: &[f64],
    dtype: IOSupportedDType,
) -> Result<Vec<u8>, IOError> {
    let item_size = dtype
        .item_size()
        .ok_or(IOError::DTypeDescriptorInvalid)?;
    let mut buf = Vec::with_capacity(values.len() * item_size);
    for &v in values {
        encode_element(v, dtype, &mut buf)?;
    }
    Ok(buf)
}

/// Decode a single element from raw bytes to f64.
fn decode_element(chunk: &[u8], dtype: IOSupportedDType) -> Result<f64, IOError> {
    match dtype {
        IOSupportedDType::Bool => Ok(if chunk[0] != 0 { 1.0 } else { 0.0 }),
        IOSupportedDType::I8 => Ok(i8::from_le_bytes([chunk[0]]) as f64),
        IOSupportedDType::U8 => Ok(chunk[0] as f64),
        IOSupportedDType::I16 => Ok(i16::from_le_bytes([chunk[0], chunk[1]]) as f64),
        IOSupportedDType::I16Be => Ok(i16::from_be_bytes([chunk[0], chunk[1]]) as f64),
        IOSupportedDType::I32 => {
            Ok(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64)
        }
        IOSupportedDType::I32Be => {
            Ok(i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64)
        }
        IOSupportedDType::I64 => Ok(i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]) as f64),
        IOSupportedDType::I64Be => Ok(i64::from_be_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]) as f64),
        IOSupportedDType::U16 => Ok(u16::from_le_bytes([chunk[0], chunk[1]]) as f64),
        IOSupportedDType::U16Be => Ok(u16::from_be_bytes([chunk[0], chunk[1]]) as f64),
        IOSupportedDType::U32 => {
            Ok(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64)
        }
        IOSupportedDType::U32Be => {
            Ok(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64)
        }
        IOSupportedDType::U64 => Ok(u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]) as f64),
        IOSupportedDType::U64Be => Ok(u64::from_be_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]) as f64),
        IOSupportedDType::F32 => Ok(f32::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
        ]) as f64),
        IOSupportedDType::F32Be => Ok(f32::from_be_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
        ]) as f64),
        IOSupportedDType::F64 => Ok(f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ])),
        IOSupportedDType::F64Be => Ok(f64::from_be_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ])),
        IOSupportedDType::Object => Err(IOError::DTypeDescriptorInvalid),
    }
}

/// Encode a single f64 value to raw bytes according to dtype.
fn encode_element(v: f64, dtype: IOSupportedDType, buf: &mut Vec<u8>) -> Result<(), IOError> {
    match dtype {
        IOSupportedDType::Bool => {
            buf.push(if v != 0.0 { 1 } else { 0 });
        }
        IOSupportedDType::I8 => buf.push((v as i8) as u8),
        IOSupportedDType::U8 => buf.push(v as u8),
        IOSupportedDType::I16 => buf.extend_from_slice(&(v as i16).to_le_bytes()),
        IOSupportedDType::I16Be => buf.extend_from_slice(&(v as i16).to_be_bytes()),
        IOSupportedDType::I32 => buf.extend_from_slice(&(v as i32).to_le_bytes()),
        IOSupportedDType::I32Be => buf.extend_from_slice(&(v as i32).to_be_bytes()),
        IOSupportedDType::I64 => buf.extend_from_slice(&(v as i64).to_le_bytes()),
        IOSupportedDType::I64Be => buf.extend_from_slice(&(v as i64).to_be_bytes()),
        IOSupportedDType::U16 => buf.extend_from_slice(&(v as u16).to_le_bytes()),
        IOSupportedDType::U16Be => buf.extend_from_slice(&(v as u16).to_be_bytes()),
        IOSupportedDType::U32 => buf.extend_from_slice(&(v as u32).to_le_bytes()),
        IOSupportedDType::U32Be => buf.extend_from_slice(&(v as u32).to_be_bytes()),
        IOSupportedDType::U64 => buf.extend_from_slice(&(v as u64).to_le_bytes()),
        IOSupportedDType::U64Be => buf.extend_from_slice(&(v as u64).to_be_bytes()),
        IOSupportedDType::F32 => buf.extend_from_slice(&(v as f32).to_le_bytes()),
        IOSupportedDType::F32Be => buf.extend_from_slice(&(v as f32).to_be_bytes()),
        IOSupportedDType::F64 => buf.extend_from_slice(&v.to_le_bytes()),
        IOSupportedDType::F64Be => buf.extend_from_slice(&v.to_be_bytes()),
        IOSupportedDType::Object => return Err(IOError::DTypeDescriptorInvalid),
    }
    Ok(())
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
        NPZ_MAGIC_PREFIX, NpyHeader, NpzCompression, SaveTxtConfig, classify_load_dispatch,
        encode_npy_header_bytes, enforce_pickle_policy, genfromtxt, loadtxt, loadtxt_usecols,
        read_npy_bytes, read_npz_bytes, savetxt, synthesize_npz_member_names,
        validate_descriptor_roundtrip, validate_header_schema, validate_io_policy_metadata,
        validate_magic_version, validate_memmap_contract, validate_npz_archive_budget,
        validate_read_payload, validate_write_contract, write_npy_bytes,
        write_npy_bytes_with_version, write_npy_preamble, write_npz_bytes,
        write_npz_bytes_with_compression, fromfile, tofile,
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
            IOSupportedDType::I8,
            IOSupportedDType::I16,
            IOSupportedDType::I16Be,
            IOSupportedDType::I32,
            IOSupportedDType::I32Be,
            IOSupportedDType::I64,
            IOSupportedDType::I64Be,
            IOSupportedDType::U8,
            IOSupportedDType::U16,
            IOSupportedDType::U16Be,
            IOSupportedDType::U32,
            IOSupportedDType::U32Be,
            IOSupportedDType::U64,
            IOSupportedDType::U64Be,
            IOSupportedDType::F32,
            IOSupportedDType::F32Be,
            IOSupportedDType::F64,
            IOSupportedDType::F64Be,
            IOSupportedDType::Object,
        ];

        for dtype in dtypes {
            validate_descriptor_roundtrip(dtype).expect("descriptor roundtrip");
        }

        let err = IOSupportedDType::decode(">i3").expect_err("unsupported descriptor");
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
        let cfg = SaveTxtConfig {
            delimiter: ",",
            header: "x,y",
            ..SaveTxtConfig::default()
        };
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

    #[test]
    fn loadtxt_usecols_selects_columns() {
        let text = "1,2,3,4\n5,6,7,8\n";
        let result = loadtxt_usecols(text, ',', '#', 0, 0, Some(&[0, 2])).unwrap();
        assert_eq!(result.nrows, 2);
        assert_eq!(result.ncols, 2);
        assert_eq!(result.values, vec![1.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn loadtxt_usecols_single_column() {
        let text = "10 20 30\n40 50 60\n";
        let result = loadtxt_usecols(text, ' ', '#', 0, 0, Some(&[1])).unwrap();
        assert_eq!(result.nrows, 2);
        assert_eq!(result.ncols, 1);
        assert_eq!(result.values, vec![20.0, 50.0]);
    }

    #[test]
    fn loadtxt_usecols_out_of_bounds() {
        let text = "1,2,3\n4,5,6\n";
        let err =
            loadtxt_usecols(text, ',', '#', 0, 0, Some(&[5])).expect_err("usecols out of bounds");
        assert_eq!(err.reason_code(), "io_read_payload_incomplete");
    }

    #[test]
    fn loadtxt_usecols_none_loads_all() {
        let text = "1,2,3\n4,5,6\n";
        let result = loadtxt_usecols(text, ',', '#', 0, 0, None).unwrap();
        assert_eq!(result.ncols, 3);
        assert_eq!(result.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn npy_i8_u8_roundtrip() {
        let header = NpyHeader {
            shape: vec![3],
            fortran_order: false,
            descr: IOSupportedDType::I8,
        };
        let payload = vec![1u8, 2, 255]; // i8 bytes
        let encoded = write_npy_bytes(&header, &payload, false).expect("write i8");
        let decoded = read_npy_bytes(&encoded, false).expect("read i8");
        assert_eq!(decoded.header.descr, IOSupportedDType::I8);
        assert_eq!(decoded.payload, payload);

        let header_u8 = NpyHeader {
            shape: vec![4],
            fortran_order: false,
            descr: IOSupportedDType::U8,
        };
        let payload_u8 = vec![0u8, 127, 128, 255];
        let encoded_u8 = write_npy_bytes(&header_u8, &payload_u8, false).expect("write u8");
        let decoded_u8 = read_npy_bytes(&encoded_u8, false).expect("read u8");
        assert_eq!(decoded_u8.header.descr, IOSupportedDType::U8);
        assert_eq!(decoded_u8.payload, payload_u8);
    }

    #[test]
    fn npy_i16_u16_roundtrip() {
        let header = NpyHeader {
            shape: vec![2],
            fortran_order: false,
            descr: IOSupportedDType::I16,
        };
        let payload: Vec<u8> = [100_i16, -200_i16]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        let encoded = write_npy_bytes(&header, &payload, false).expect("write i16");
        let decoded = read_npy_bytes(&encoded, false).expect("read i16");
        assert_eq!(decoded.header.descr, IOSupportedDType::I16);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn npy_u32_u64_roundtrip() {
        let header = NpyHeader {
            shape: vec![2],
            fortran_order: false,
            descr: IOSupportedDType::U32,
        };
        let payload: Vec<u8> = [42_u32, 1_000_000_u32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let encoded = write_npy_bytes(&header, &payload, false).expect("write u32");
        let decoded = read_npy_bytes(&encoded, false).expect("read u32");
        assert_eq!(decoded.header.descr, IOSupportedDType::U32);
        assert_eq!(decoded.payload, payload);

        let header_u64 = NpyHeader {
            shape: vec![1],
            fortran_order: false,
            descr: IOSupportedDType::U64,
        };
        let payload_u64: Vec<u8> = u64::MAX.to_le_bytes().to_vec();
        let encoded_u64 = write_npy_bytes(&header_u64, &payload_u64, false).expect("write u64");
        let decoded_u64 = read_npy_bytes(&encoded_u64, false).expect("read u64");
        assert_eq!(decoded_u64.header.descr, IOSupportedDType::U64);
    }

    #[test]
    fn npy_big_endian_descriptor_roundtrip() {
        let header = NpyHeader {
            shape: vec![2],
            fortran_order: false,
            descr: IOSupportedDType::F64Be,
        };
        let payload: Vec<u8> = [1.25_f64, -3.5_f64]
            .into_iter()
            .flat_map(f64::to_be_bytes)
            .collect();
        let encoded = write_npy_bytes(&header, &payload, false).expect("write big-endian f64");
        let decoded = read_npy_bytes(&encoded, false).expect("read big-endian f64");
        assert_eq!(decoded.header.descr, IOSupportedDType::F64Be);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn item_size_matches_dtype_byte_widths() {
        assert_eq!(IOSupportedDType::Bool.item_size(), Some(1));
        assert_eq!(IOSupportedDType::I8.item_size(), Some(1));
        assert_eq!(IOSupportedDType::U8.item_size(), Some(1));
        assert_eq!(IOSupportedDType::I16.item_size(), Some(2));
        assert_eq!(IOSupportedDType::I16Be.item_size(), Some(2));
        assert_eq!(IOSupportedDType::U16.item_size(), Some(2));
        assert_eq!(IOSupportedDType::U16Be.item_size(), Some(2));
        assert_eq!(IOSupportedDType::I32.item_size(), Some(4));
        assert_eq!(IOSupportedDType::I32Be.item_size(), Some(4));
        assert_eq!(IOSupportedDType::U32.item_size(), Some(4));
        assert_eq!(IOSupportedDType::U32Be.item_size(), Some(4));
        assert_eq!(IOSupportedDType::F32.item_size(), Some(4));
        assert_eq!(IOSupportedDType::F32Be.item_size(), Some(4));
        assert_eq!(IOSupportedDType::I64.item_size(), Some(8));
        assert_eq!(IOSupportedDType::I64Be.item_size(), Some(8));
        assert_eq!(IOSupportedDType::U64.item_size(), Some(8));
        assert_eq!(IOSupportedDType::U64Be.item_size(), Some(8));
        assert_eq!(IOSupportedDType::F64.item_size(), Some(8));
        assert_eq!(IOSupportedDType::F64Be.item_size(), Some(8));
        assert_eq!(IOSupportedDType::Object.item_size(), None);
    }

    // ── NPZ tests ──

    #[test]
    fn npz_single_array_roundtrip() {
        let header = NpyHeader {
            shape: vec![3],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let payload: Vec<u8> = [1.0_f64, 2.0, 3.0]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect();
        let npz = write_npz_bytes(&[("arr0", &header, &payload)]).expect("write npz");
        // Verify it starts with ZIP magic
        assert_eq!(&npz[..4], &NPZ_MAGIC_PREFIX);

        let entries = read_npz_bytes(&npz).expect("read npz");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "arr0");
        assert_eq!(entries[0].array.header.shape, vec![3]);
        assert_eq!(entries[0].array.header.descr, IOSupportedDType::F64);
        assert_eq!(entries[0].array.payload, payload);
    }

    #[test]
    fn npz_multiple_arrays_roundtrip() {
        let h1 = NpyHeader {
            shape: vec![2],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let p1: Vec<u8> = [10.0_f64, 20.0]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect();
        let h2 = NpyHeader {
            shape: vec![2, 2],
            fortran_order: false,
            descr: IOSupportedDType::I32,
        };
        let p2: Vec<u8> = [1_i32, 2, 3, 4]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect();

        let npz = write_npz_bytes(&[("x", &h1, &p1), ("matrix", &h2, &p2)]).expect("write npz");
        let entries = read_npz_bytes(&npz).expect("read npz");

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "x");
        assert_eq!(entries[0].array.header.descr, IOSupportedDType::F64);
        assert_eq!(entries[0].array.payload, p1);

        assert_eq!(entries[1].name, "matrix");
        assert_eq!(entries[1].array.header.shape, vec![2, 2]);
        assert_eq!(entries[1].array.header.descr, IOSupportedDType::I32);
        assert_eq!(entries[1].array.payload, p2);
    }

    #[test]
    fn npz_deflate_single_array_roundtrip() {
        let header = NpyHeader {
            shape: vec![3],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let payload: Vec<u8> = [1.0_f64, 2.0, 3.0]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect();
        let npz = write_npz_bytes_with_compression(
            &[("arr0", &header, &payload)],
            NpzCompression::Deflate,
        )
        .expect("write deflate npz");

        let entries = read_npz_bytes(&npz).expect("read deflate npz");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "arr0");
        assert_eq!(entries[0].array.header.descr, IOSupportedDType::F64);
        assert_eq!(entries[0].array.payload, payload);
    }

    #[test]
    fn npz_unsupported_compression_rejected() {
        let header = NpyHeader {
            shape: vec![1],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let payload: Vec<u8> = 1.0_f64.to_le_bytes().to_vec();
        let mut npz = write_npz_bytes(&[("a", &header, &payload)]).expect("write npz");

        // Local header compression field.
        npz[8] = 1;
        npz[9] = 0;

        // Central directory compression field.
        let cd_pos = npz
            .windows(4)
            .position(|window| window == [0x50, 0x4B, 0x01, 0x02])
            .expect("central directory signature");
        npz[cd_pos + 10] = 1;
        npz[cd_pos + 11] = 0;

        let err = read_npz_bytes(&npz).expect_err("unsupported compression must be rejected");
        assert_eq!(err.reason_code(), "io_npz_archive_contract_violation");
    }

    #[test]
    fn npz_deflate_corrupted_payload_rejected() {
        let header = NpyHeader {
            shape: vec![2],
            fortran_order: false,
            descr: IOSupportedDType::I32,
        };
        let payload: Vec<u8> = [123_i32, -5_i32]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect();
        let mut npz =
            write_npz_bytes_with_compression(&[("a", &header, &payload)], NpzCompression::Deflate)
                .expect("write deflate npz");

        let local_fname_len = u16::from_le_bytes([npz[26], npz[27]]) as usize;
        let local_extra_len = u16::from_le_bytes([npz[28], npz[29]]) as usize;
        let data_start = 30 + local_fname_len + local_extra_len;
        assert!(data_start < npz.len(), "expected deflate payload bytes");
        npz[data_start] ^= 0xFF;

        let err = read_npz_bytes(&npz).expect_err("corrupted deflate payload must fail");
        assert_eq!(err.reason_code(), "io_npz_archive_contract_violation");
    }

    #[test]
    fn npz_empty_archive_rejected() {
        let result = write_npz_bytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn npz_dispatch_detection() {
        let h = NpyHeader {
            shape: vec![1],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let p: Vec<u8> = 42.0_f64.to_le_bytes().to_vec();
        let npz = write_npz_bytes(&[("a", &h, &p)]).expect("write");
        let dispatch = classify_load_dispatch(&npz[..8], false).expect("dispatch");
        assert_eq!(dispatch, LoadDispatch::Npz);
    }

    #[test]
    fn npz_boolean_array_roundtrip() {
        let h = NpyHeader {
            shape: vec![3],
            fortran_order: false,
            descr: IOSupportedDType::Bool,
        };
        let p = vec![1u8, 0, 1];
        let npz = write_npz_bytes(&[("flags", &h, &p)]).expect("write");
        let entries = read_npz_bytes(&npz).expect("read");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].array.header.descr, IOSupportedDType::Bool);
        assert_eq!(entries[0].array.payload, vec![1, 0, 1]);
    }

    #[test]
    fn npz_truncated_data_rejected() {
        let h = NpyHeader {
            shape: vec![1],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let p: Vec<u8> = 1.0_f64.to_le_bytes().to_vec();
        let npz = write_npz_bytes(&[("a", &h, &p)]).expect("write");
        // Truncate
        let truncated = &npz[..npz.len() / 2];
        assert!(read_npz_bytes(truncated).is_err());
    }

    // ── fromfile / tofile tests ──

    #[test]
    fn fromfile_f64_basic() {
        let data: Vec<u8> = [1.0f64, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let vals = fromfile(&data, IOSupportedDType::F64, None).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn fromfile_i32() {
        let data: Vec<u8> = [42i32, -7]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let vals = fromfile(&data, IOSupportedDType::I32, None).unwrap();
        assert_eq!(vals, vec![42.0, -7.0]);
    }

    #[test]
    fn fromfile_with_count() {
        let data: Vec<u8> = [1.0f64, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let vals = fromfile(&data, IOSupportedDType::F64, Some(2)).unwrap();
        assert_eq!(vals, vec![1.0, 2.0]);
    }

    #[test]
    fn fromfile_bool() {
        let data = vec![1u8, 0, 1, 1, 0];
        let vals = fromfile(&data, IOSupportedDType::Bool, None).unwrap();
        assert_eq!(vals, vec![1.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn tofile_f64_basic() {
        let vals = vec![1.0, 2.0, 3.0];
        let data = tofile(&vals, IOSupportedDType::F64).unwrap();
        let expected: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(data, expected);
    }

    #[test]
    fn tofile_i32() {
        let vals = vec![42.0, -7.0];
        let data = tofile(&vals, IOSupportedDType::I32).unwrap();
        let expected: Vec<u8> = [42i32, -7]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(data, expected);
    }

    #[test]
    fn fromfile_tofile_roundtrip() {
        let original = vec![1.5, -2.7, 0.0, 1e10];
        let encoded = tofile(&original, IOSupportedDType::F64).unwrap();
        let decoded = fromfile(&encoded, IOSupportedDType::F64, None).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn fromfile_tofile_roundtrip_u16() {
        let original = vec![100.0, 200.0, 65535.0];
        let encoded = tofile(&original, IOSupportedDType::U16).unwrap();
        let decoded = fromfile(&encoded, IOSupportedDType::U16, None).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn fromfile_truncated_data_reads_partial() {
        // 7 bytes: only 3 complete u16 elements (6 bytes), last byte ignored
        let data = vec![1, 0, 2, 0, 3, 0, 99];
        let vals = fromfile(&data, IOSupportedDType::U16, None).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }
}
