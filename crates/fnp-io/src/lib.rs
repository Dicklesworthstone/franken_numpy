#![forbid(unsafe_code)]

use core::fmt;
use std::collections::HashSet;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use flate2::{Compression, read::DeflateDecoder, write::DeflateEncoder};

pub const IO_PACKET_ID: &str = "FNP-P2C-009";
pub const NPY_MAGIC_PREFIX: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];
pub const NPZ_MAGIC_PREFIX: [u8; 4] = [b'P', b'K', 0x03, 0x04];

pub const MAX_HEADER_BYTES: usize = 65_536;
pub const MAX_ARCHIVE_MEMBERS: usize = 4_096;
pub const MAX_ARCHIVE_UNCOMPRESSED_BYTES: usize = 2 * 1024 * 1024 * 1024;
pub const MAX_TEXT_ELEMENTS: usize = 16 * 1024 * 1024; // 128MB of f64s
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
    Complex64,
    Complex64Be,
    Complex128,
    Complex128Be,
    /// Fixed-width byte string, e.g. `|S10` = 10-byte ASCII string.
    Bytes(usize),
    /// Fixed-width Unicode string (UCS-4/UTF-32-LE), e.g. `<U20` = 20-char string.
    Unicode(usize),
    /// Fixed-width Unicode string (UCS-4/UTF-32-BE), e.g. `>U20` = 20-char string.
    UnicodeBe(usize),
    Object,
}

impl IOSupportedDType {
    #[must_use]
    pub fn descr(self) -> String {
        match self {
            Self::Bool => "|b1".to_string(),
            Self::I8 => "|i1".to_string(),
            Self::I16 => "<i2".to_string(),
            Self::I16Be => ">i2".to_string(),
            Self::I32 => "<i4".to_string(),
            Self::I32Be => ">i4".to_string(),
            Self::I64 => "<i8".to_string(),
            Self::I64Be => ">i8".to_string(),
            Self::U8 => "|u1".to_string(),
            Self::U16 => "<u2".to_string(),
            Self::U16Be => ">u2".to_string(),
            Self::U32 => "<u4".to_string(),
            Self::U32Be => ">u4".to_string(),
            Self::U64 => "<u8".to_string(),
            Self::U64Be => ">u8".to_string(),
            Self::F32 => "<f4".to_string(),
            Self::F32Be => ">f4".to_string(),
            Self::F64 => "<f8".to_string(),
            Self::F64Be => ">f8".to_string(),
            Self::Complex64 => "<c8".to_string(),
            Self::Complex64Be => ">c8".to_string(),
            Self::Complex128 => "<c16".to_string(),
            Self::Complex128Be => ">c16".to_string(),
            Self::Bytes(n) => format!("|S{n}"),
            Self::Unicode(n) => format!("<U{n}"),
            Self::UnicodeBe(n) => format!(">U{n}"),
            Self::Object => "|O".to_string(),
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
            "<c8" => Ok(Self::Complex64),
            ">c8" => Ok(Self::Complex64Be),
            "<c16" => Ok(Self::Complex128),
            ">c16" => Ok(Self::Complex128Be),
            "|O" => Ok(Self::Object),
            _ => Self::decode_variable_width(descr),
        }
    }

    fn decode_variable_width(descr: &str) -> Result<Self, IOError> {
        let bytes = descr.as_bytes();
        if bytes.len() < 3 {
            return Err(IOError::DTypeDescriptorInvalid);
        }
        let endian = bytes[0];
        let kind = bytes[1];
        let width_str = &descr[2..];
        let width: usize = width_str
            .parse()
            .map_err(|_| IOError::DTypeDescriptorInvalid)?;
        if width == 0 {
            return Err(IOError::DTypeDescriptorInvalid);
        }
        match (endian, kind) {
            (b'|', b'S') => Ok(Self::Bytes(width)),
            (b'<', b'U') => Ok(Self::Unicode(width)),
            (b'>', b'U') => Ok(Self::UnicodeBe(width)),
            _ => Err(IOError::DTypeDescriptorInvalid),
        }
    }

    #[must_use]
    pub fn item_size(self) -> Option<usize> {
        match self {
            Self::Bool | Self::I8 | Self::U8 => Some(1),
            Self::I16 | Self::I16Be | Self::U16 | Self::U16Be => Some(2),
            Self::I32 | Self::I32Be | Self::U32 | Self::U32Be | Self::F32 | Self::F32Be => Some(4),
            Self::I64
            | Self::I64Be
            | Self::U64
            | Self::U64Be
            | Self::F64
            | Self::F64Be
            | Self::Complex64
            | Self::Complex64Be => Some(8),
            Self::Complex128 | Self::Complex128Be => Some(16),
            Self::Bytes(n) => Some(n),
            Self::Unicode(n) | Self::UnicodeBe(n) => Some(n * 4),
            Self::Object => None,
        }
    }

    #[must_use]
    pub const fn is_complex(self) -> bool {
        matches!(
            self,
            Self::Complex64 | Self::Complex64Be | Self::Complex128 | Self::Complex128Be
        )
    }

    #[must_use]
    pub const fn is_string(self) -> bool {
        matches!(self, Self::Bytes(_) | Self::Unicode(_) | Self::UnicodeBe(_))
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
    pub payload: Arc<[u8]>,
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
            format!("({joined})")
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

fn parse_quoted_value(value: &str) -> Result<String, IOError> {
    let bytes = value.as_bytes();
    if bytes.len() < 2 {
        return Err(IOError::HeaderSchemaInvalid("header quoted value is empty"));
    }
    let quote = bytes[0];
    if quote != b'\'' && quote != b'"' {
        return Err(IOError::HeaderSchemaInvalid(
            "header quoted value must start with quote",
        ));
    }

    let mut result = String::with_capacity(bytes.len());
    let mut escaped = false;
    let mut idx = 1usize;

    while idx < bytes.len() {
        let b = bytes[idx];
        if b == quote && !escaped {
            return Ok(result);
        }

        if b == b'\\' && !escaped {
            escaped = true;
        } else {
            result.push(char::from(b));
            escaped = false;
        }
        idx += 1;
    }

    Err(IOError::HeaderSchemaInvalid(
        "header quoted value missing closing quote",
    ))
}

fn parse_shape_tuple(tuple_literal: &str) -> Result<Vec<usize>, IOError> {
    let inner = tuple_literal.trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    let has_comma = inner.contains(',');

    let mut shape = Vec::new();
    let mut saw_non_empty = false;
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        saw_non_empty = true;
        let dim = part
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

fn parse_header_dictionary_map(
    dictionary: &str,
) -> Result<std::collections::HashMap<String, String>, IOError> {
    let bytes = dictionary.as_bytes();
    let mut map = std::collections::HashMap::new();
    let mut idx = 0usize;

    // Skip leading whitespace and '{'
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    if idx >= bytes.len() || bytes[idx] != b'{' {
        return Err(IOError::HeaderSchemaInvalid(
            "header dictionary must be wrapped in braces",
        ));
    }
    idx += 1;

    let end_idx = dictionary.rfind('}').ok_or(IOError::HeaderSchemaInvalid(
        "header dictionary must be wrapped in braces",
    ))?;

    while idx < end_idx {
        // Find next key
        while idx < end_idx && !matches!(bytes[idx], b'\'' | b'"') {
            idx += 1;
        }
        if idx >= end_idx {
            break;
        }

        let quote = bytes[idx];
        let key_start = idx + 1;
        idx += 1;
        let mut escaped = false;
        while idx < end_idx {
            if bytes[idx] == quote && !escaped {
                break;
            }
            escaped = (bytes[idx] == b'\\') && !escaped;
            idx += 1;
        }
        if idx >= end_idx {
            return Err(IOError::HeaderSchemaInvalid("unterminated header key"));
        }
        let key = &dictionary[key_start..idx];
        idx += 1;

        // Find ':'
        while idx < end_idx && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx >= end_idx || bytes[idx] != b':' {
            return Err(IOError::HeaderSchemaInvalid("missing ':' after header key"));
        }
        idx += 1;

        // Find value
        while idx < end_idx && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        let val_start = idx;

        let mut in_quote = None;
        let mut depth = 0;
        let mut escaped = false;
        while idx < end_idx {
            let b = bytes[idx];
            if let Some(q) = in_quote {
                if b == q && !escaped {
                    in_quote = None;
                }
                escaped = (b == b'\\') && !escaped;
            } else {
                match b {
                    b'\'' | b'"' => in_quote = Some(b),
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' | b']' | b'}' => {
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                    }
                    b',' if depth == 0 => break,
                    _ => {}
                }
            }
            idx += 1;
        }
        let value = dictionary[val_start..idx].trim();
        if map.insert(key.to_string(), value.to_string()).is_some() {
            return Err(IOError::HeaderSchemaInvalid(
                "header dictionary contains duplicate keys",
            ));
        }
        if idx < end_idx && bytes[idx] == b',' {
            idx += 1;
        }
    }
    Ok(map)
}

fn validate_required_header_keys(
    map: &std::collections::HashMap<String, String>,
) -> Result<(), IOError> {
    if map.len() < NPY_HEADER_REQUIRED_KEYS.len() {
        return Err(IOError::HeaderSchemaInvalid(
            "header dictionary must contain at least descr/fortran_order/shape keys",
        ));
    }

    for required in &NPY_HEADER_REQUIRED_KEYS {
        if !map.contains_key(*required) {
            return Err(IOError::HeaderSchemaInvalid(
                "required header field is missing",
            ));
        }
    }

    Ok(())
}

fn parse_fortran_order_value(value: &str) -> Result<bool, IOError> {
    if value == "True" {
        Ok(true)
    } else if value == "False" {
        Ok(false)
    } else {
        Err(IOError::HeaderSchemaInvalid(
            "fortran_order field must be True or False",
        ))
    }
}

fn parse_shape_field(shape_tail: &str) -> Result<Vec<usize>, IOError> {
    let shape_tail = shape_tail
        .strip_prefix('(')
        .ok_or(IOError::HeaderSchemaInvalid(
            "shape field must begin with tuple syntax",
        ))?;
    let shape_end = shape_tail.rfind(')').ok_or(IOError::HeaderSchemaInvalid(
        "shape tuple missing closing ')'",
    ))?;
    parse_shape_tuple(&shape_tail[..shape_end])
}

fn parse_header_dictionary(header_bytes: &[u8], header_len: usize) -> Result<NpyHeader, IOError> {
    let dictionary = std::str::from_utf8(header_bytes).map_err(|_| {
        IOError::HeaderSchemaInvalid("header bytes must decode as utf-8/ascii dictionary")
    })?;
    let dictionary = dictionary.trim_end();

    let map = parse_header_dictionary_map(dictionary)?;

    validate_required_header_keys(&map)?;

    let descr_literal = parse_quoted_value(
        map.get("descr")
            .ok_or(IOError::HeaderSchemaInvalid("descr missing"))?,
    )?;
    let fortran_order = parse_fortran_order_value(
        map.get("fortran_order")
            .ok_or(IOError::HeaderSchemaInvalid("fortran_order missing"))?,
    )?;
    let shape = parse_shape_field(
        map.get("shape")
            .ok_or(IOError::HeaderSchemaInvalid("shape missing"))?,
    )?;

    validate_header_schema(&shape, fortran_order, &descr_literal, header_len)
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
        payload: Arc::from(body),
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
    let decoded = IOSupportedDType::decode(&encoded)?;
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
    // Track covered ranges to prevent overlapping entries
    let mut covered_ranges: Vec<(usize, usize)> = Vec::with_capacity(entry_count);

    for _ in 0..entry_count {
        if pos.checked_add(46).is_none_or(|end| end > data.len()) {
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

        let fname_start = pos
            .checked_add(46)
            .ok_or(IOError::NpzArchiveContractViolation(
                "npz: position overflow",
            ))?;
        let fname_end =
            fname_start
                .checked_add(fname_len)
                .ok_or(IOError::NpzArchiveContractViolation(
                    "npz: filename length overflow",
                ))?;
        if fname_end > data.len() {
            return Err(IOError::NpzArchiveContractViolation(
                "npz: filename extends beyond data",
            ));
        }
        let file_name = String::from_utf8_lossy(&data[fname_start..fname_end]).into_owned();
        // Prevent directory traversal
        if file_name.contains("..") || file_name.starts_with('/') {
            return Err(IOError::NpzArchiveContractViolation(
                "npz: malicious filename detected (directory traversal)",
            ));
        }

        // Parse local file header to find data start
        if local_offset
            .checked_add(30)
            .is_none_or(|end| end > data.len())
        {
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

        let data_start = local_offset
            .checked_add(30)
            .and_then(|s| s.checked_add(local_fname_len))
            .and_then(|s| s.checked_add(local_extra_len))
            .ok_or(IOError::NpzArchiveContractViolation(
                "npz: local data start overflow",
            ))?;
        let data_end =
            data_start
                .checked_add(compressed_size)
                .ok_or(IOError::NpzArchiveContractViolation(
                    "npz: local data end overflow",
                ))?;

        if data_end > data.len() {
            return Err(IOError::NpzArchiveContractViolation(
                "npz: entry data extends beyond archive",
            ));
        }

        // Check for overlapping entries
        let current_range = (local_offset, data_end);
        for &(start, end) in &covered_ranges {
            if (current_range.0 >= start && current_range.0 < end)
                || (current_range.1 > start && current_range.1 <= end)
                || (start >= current_range.0 && start < current_range.1)
            {
                return Err(IOError::NpzArchiveContractViolation(
                    "npz: overlapping zip entries detected",
                ));
            }
        }
        covered_ranges.push(current_range);

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
                // Limit the amount of data we read to uncompressed_size to prevent zip bombs
                let mut decoder =
                    DeflateDecoder::new(stored_entry_bytes).take(uncompressed_size as u64);
                let mut decoded = Vec::with_capacity(uncompressed_size);
                decoder.read_to_end(&mut decoded).map_err(|_| {
                    IOError::NpzArchiveContractViolation(
                        "npz: failed to inflate DEFLATE entry payload",
                    )
                })?;
                decoded
            }
            _ => {
                return Err(IOError::NpzArchiveContractViolation(
                    "npz: unexpected compression method",
                ));
            }
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

        pos = fname_end
            .checked_add(extra_len)
            .and_then(|p| p.checked_add(comment_len))
            .ok_or(IOError::NpzArchiveContractViolation(
                "npz: central directory position overflow",
            ))?;
    }

    Ok(entries)
}

/// IEEE 802.3 CRC-32 (used by ZIP format).
/// Uses a table-based implementation for performance.
fn crc32_ieee(data: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut j = 0;
            while j < 8 {
                if c & 1 != 0 {
                    c = 0xEDB8_8320 ^ (c >> 1);
                } else {
                    c >>= 1;
                }
                j += 1;
            }
            table[i] = c;
            i += 1;
        }
        table
    };

    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc = TABLE[(crc as u8 ^ byte) as usize] ^ (crc >> 8);
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
    // Pre-allocate approximately (15 chars per float + delimiter) * total
    let mut output = String::with_capacity(values.len() * 16);
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
                "%.18e" | "%e" => {
                    use std::fmt::Write;
                    write!(output, "{v:e}").unwrap();
                }
                "%d" | "%i" => {
                    use std::fmt::Write;
                    write!(output, "{}", v as i64).unwrap();
                }
                _ => {
                    use std::fmt::Write;
                    write!(output, "{v}").unwrap();
                }
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

        let current_ncols = row_vals.len();
        let target_ncols = ncols.unwrap_or(current_ncols);

        if values.len() + target_ncols > MAX_TEXT_ELEMENTS {
            return Err(IOError::ReadPayloadIncomplete(
                "genfromtxt: text exceeds MAX_TEXT_ELEMENTS budget",
            ));
        }

        match ncols {
            None => {
                ncols = Some(current_ncols);
                values.extend(row_vals);
            }
            Some(expected) if current_ncols != expected => {
                // Pad or truncate to match
                let mut padded = row_vals;
                padded.resize(expected, filling_values);
                values.extend(padded);
            }
            Some(_) => {
                values.extend(row_vals);
            }
        }
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
    if dtype.is_complex() || dtype.is_string() {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let item_size = dtype.item_size().ok_or(IOError::DTypeDescriptorInvalid)?;
    if item_size == 0 {
        return Ok(Vec::new());
    }
    let max_elems = data.len() / item_size;
    let n = match count {
        Some(c) => {
            if c > max_elems {
                return Err(IOError::ReadPayloadIncomplete(
                    "insufficient data for requested element count",
                ));
            }
            c
        }
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
pub fn tofile(values: &[f64], dtype: IOSupportedDType) -> Result<Vec<u8>, IOError> {
    if dtype.is_complex() || dtype.is_string() {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let item_size = dtype.item_size().ok_or(IOError::DTypeDescriptorInvalid)?;
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
        IOSupportedDType::F32 => {
            Ok(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64)
        }
        IOSupportedDType::F32Be => {
            Ok(f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64)
        }
        IOSupportedDType::F64 => Ok(f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ])),
        IOSupportedDType::F64Be => Ok(f64::from_be_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ])),
        IOSupportedDType::Complex64
        | IOSupportedDType::Complex64Be
        | IOSupportedDType::Complex128
        | IOSupportedDType::Complex128Be
        | IOSupportedDType::Bytes(_)
        | IOSupportedDType::Unicode(_)
        | IOSupportedDType::UnicodeBe(_)
        | IOSupportedDType::Object => Err(IOError::DTypeDescriptorInvalid),
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
        IOSupportedDType::Complex64
        | IOSupportedDType::Complex64Be
        | IOSupportedDType::Complex128
        | IOSupportedDType::Complex128Be
        | IOSupportedDType::Bytes(_)
        | IOSupportedDType::Unicode(_)
        | IOSupportedDType::UnicodeBe(_)
        | IOSupportedDType::Object => return Err(IOError::DTypeDescriptorInvalid),
    }
    Ok(())
}

/// Read raw binary data from bytes into a flat array of complex values.
///
/// Complex values are represented as `(real, imag)` pairs and decoded from the
/// interleaved NPY memory layout used by NumPy complex dtypes.
pub fn fromfile_complex(
    data: &[u8],
    dtype: IOSupportedDType,
    count: Option<usize>,
) -> Result<Vec<(f64, f64)>, IOError> {
    if !dtype.is_complex() {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let item_size = dtype.item_size().ok_or(IOError::DTypeDescriptorInvalid)?;
    let max_elems = data.len() / item_size;
    let n = match count {
        Some(c) => {
            if c > max_elems {
                return Err(IOError::ReadPayloadIncomplete(
                    "insufficient data for requested element count",
                ));
            }
            c
        }
        None => max_elems,
    };

    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * item_size;
        let chunk = &data[offset..offset + item_size];
        values.push(decode_complex_element(chunk, dtype)?);
    }
    Ok(values)
}

/// Write complex values to raw binary bytes using NumPy interleaved layout.
pub fn tofile_complex(values: &[(f64, f64)], dtype: IOSupportedDType) -> Result<Vec<u8>, IOError> {
    if !dtype.is_complex() {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let item_size = dtype.item_size().ok_or(IOError::DTypeDescriptorInvalid)?;
    let mut buf = Vec::with_capacity(values.len() * item_size);
    for &(re, im) in values {
        encode_complex_element(re, im, dtype, &mut buf)?;
    }
    Ok(buf)
}

fn decode_complex_element(chunk: &[u8], dtype: IOSupportedDType) -> Result<(f64, f64), IOError> {
    match dtype {
        IOSupportedDType::Complex64 => Ok((
            f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
            f64::from(f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]])),
        )),
        IOSupportedDType::Complex64Be => Ok((
            f64::from(f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
            f64::from(f32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]])),
        )),
        IOSupportedDType::Complex128 => Ok((
            f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]),
            f64::from_le_bytes([
                chunk[8], chunk[9], chunk[10], chunk[11], chunk[12], chunk[13], chunk[14],
                chunk[15],
            ]),
        )),
        IOSupportedDType::Complex128Be => Ok((
            f64::from_be_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]),
            f64::from_be_bytes([
                chunk[8], chunk[9], chunk[10], chunk[11], chunk[12], chunk[13], chunk[14],
                chunk[15],
            ]),
        )),
        _ => Err(IOError::DTypeDescriptorInvalid),
    }
}

fn encode_complex_element(
    re: f64,
    im: f64,
    dtype: IOSupportedDType,
    buf: &mut Vec<u8>,
) -> Result<(), IOError> {
    match dtype {
        IOSupportedDType::Complex64 => {
            buf.extend_from_slice(&(re as f32).to_le_bytes());
            buf.extend_from_slice(&(im as f32).to_le_bytes());
        }
        IOSupportedDType::Complex64Be => {
            buf.extend_from_slice(&(re as f32).to_be_bytes());
            buf.extend_from_slice(&(im as f32).to_be_bytes());
        }
        IOSupportedDType::Complex128 => {
            buf.extend_from_slice(&re.to_le_bytes());
            buf.extend_from_slice(&im.to_le_bytes());
        }
        IOSupportedDType::Complex128Be => {
            buf.extend_from_slice(&re.to_be_bytes());
            buf.extend_from_slice(&im.to_be_bytes());
        }
        _ => return Err(IOError::DTypeDescriptorInvalid),
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

/// Parse a string of numbers into an array of f64 values (np.fromstring equivalent).
///
/// Supports two modes:
/// - Text mode (`sep` is non-empty): split text on separator, parse as floats
/// - Binary mode (`sep` is empty): interpret raw bytes as binary data of the given dtype
///
/// For text mode, `dtype` is ignored (always parsed as f64).
pub fn fromstring(data: &[u8], dtype: IOSupportedDType, sep: &str) -> Result<Vec<f64>, IOError> {
    if sep.is_empty() {
        // Binary mode: decode bytes as dtype
        fromfile(data, dtype, None)
    } else {
        // Text mode: parse as space/comma/etc separated numbers
        let text = std::str::from_utf8(data).map_err(|_| {
            IOError::ReadPayloadIncomplete("fromstring: invalid UTF-8 in text mode")
        })?;
        let values: Vec<f64> = text
            .split(sep)
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| IOError::ReadPayloadIncomplete("fromstring: parse error"))?;
        Ok(values)
    }
}

/// Serialize array values to a byte buffer (np.ndarray.tobytes equivalent).
///
/// Returns the raw bytes of the array data in the specified dtype encoding.
/// This is an alias for `tofile` providing the NumPy `.tobytes()` method name.
pub fn tobytes(values: &[f64], dtype: IOSupportedDType) -> Result<Vec<u8>, IOError> {
    tofile(values, dtype)
}

/// Serialize array values to a text string with a separator.
///
/// Note: NumPy's `ndarray.tostring()` is a deprecated alias for `tobytes()` (binary).
/// This function is a text formatter analogous to `np.array2string` / `savetxt`.
pub fn tostring(values: &[f64], sep: &str) -> String {
    values
        .iter()
        .map(|v| {
            if v.fract() == 0.0 && v.is_finite() && v.abs() < 1e15 {
                format!("{}", *v as i64)
            } else {
                format!("{v}")
            }
        })
        .collect::<Vec<_>>()
        .join(sep)
}

// ── High-level convenience functions (np.save, np.load, np.savez, np.savez_compressed) ──

/// High-level save: serialize a numeric array to NPY bytes (np.save equivalent).
///
/// Combines header construction and writing into a single call.
/// `shape` and `dtype` describe the array, `values` is the flat row-major data.
pub fn save(shape: &[usize], values: &[f64], dtype: IOSupportedDType) -> Result<Vec<u8>, IOError> {
    let header = NpyHeader {
        shape: shape.to_vec(),
        fortran_order: false,
        descr: dtype,
    };
    let payload = tobytes(values, dtype)?;
    write_npy_bytes(&header, &payload, false)
}

/// High-level load: deserialize NPY bytes back to array data (np.load equivalent for .npy).
///
/// Returns (shape, values, dtype).
pub fn load(data: &[u8]) -> Result<(Vec<usize>, Vec<f64>, IOSupportedDType), IOError> {
    let npy = read_npy_bytes(data, false)?;
    let dtype = npy.header.descr;
    let shape = npy.header.shape;
    let values = fromfile(&npy.payload, dtype, None)?;
    Ok((shape, values, dtype))
}

pub type ComplexValue = (f64, f64);
pub type NpyLoadedComplex = (Vec<usize>, Vec<ComplexValue>, IOSupportedDType);

/// High-level save for complex arrays using NumPy interleaved complex layout.
pub fn save_complex(
    shape: &[usize],
    values: &[ComplexValue],
    dtype: IOSupportedDType,
) -> Result<Vec<u8>, IOError> {
    if !dtype.is_complex() {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let header = NpyHeader {
        shape: shape.to_vec(),
        fortran_order: false,
        descr: dtype,
    };
    let payload = tofile_complex(values, dtype)?;
    write_npy_bytes(&header, &payload, false)
}

/// High-level load for complex-valued NPY arrays.
pub fn load_complex(data: &[u8]) -> Result<NpyLoadedComplex, IOError> {
    let npy = read_npy_bytes(data, false)?;
    let dtype = npy.header.descr;
    if !dtype.is_complex() {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let shape = npy.header.shape;
    let values = fromfile_complex(&npy.payload, dtype, None)?;
    Ok((shape, values, dtype))
}

/// Entry returned by `load_npz`: (name, shape, values, dtype).
pub type NpzLoadedEntry = (String, Vec<usize>, Vec<f64>, IOSupportedDType);

/// High-level load for NPZ archives (np.load equivalent for .npz).
///
/// Returns a vector of `NpzLoadedEntry` tuples.
pub fn load_npz(data: &[u8]) -> Result<Vec<NpzLoadedEntry>, IOError> {
    let entries = read_npz_bytes(data)?;
    let mut results = Vec::with_capacity(entries.len());
    for entry in entries {
        let dtype = entry.array.header.descr;
        let shape = entry.array.header.shape;
        let values = fromfile(&entry.array.payload, dtype, None)?;
        results.push((entry.name, shape, values, dtype));
    }
    Ok(results)
}

/// High-level savez: serialize multiple named arrays to an uncompressed NPZ archive (np.savez).
///
/// Each entry is (name, shape, values, dtype).
pub fn savez(entries: &[(&str, &[usize], &[f64], IOSupportedDType)]) -> Result<Vec<u8>, IOError> {
    let mut payloads = Vec::with_capacity(entries.len());
    let mut headers = Vec::with_capacity(entries.len());

    for (name, shape, values, dtype) in entries {
        let header = NpyHeader {
            shape: shape.to_vec(),
            fortran_order: false,
            descr: *dtype,
        };
        let payload = tobytes(values, *dtype)?;
        headers.push((name, header));
        payloads.push(payload);
    }

    let raw_entries: Vec<(&str, &NpyHeader, &[u8])> = headers
        .iter()
        .zip(payloads.iter())
        .map(|((name, header), payload)| (**name, header, payload.as_slice()))
        .collect();

    write_npz_bytes(&raw_entries)
}

/// High-level savez_compressed: serialize multiple named arrays to a DEFLATE-compressed NPZ
/// archive (np.savez_compressed).
pub fn savez_compressed(
    entries: &[(&str, &[usize], &[f64], IOSupportedDType)],
) -> Result<Vec<u8>, IOError> {
    let mut payloads = Vec::with_capacity(entries.len());
    let mut headers = Vec::with_capacity(entries.len());

    for (name, shape, values, dtype) in entries {
        let header = NpyHeader {
            shape: shape.to_vec(),
            fortran_order: false,
            descr: *dtype,
        };
        let payload = tobytes(values, *dtype)?;
        headers.push((name, header));
        payloads.push(payload);
    }

    let raw_entries: Vec<(&str, &NpyHeader, &[u8])> = headers
        .iter()
        .zip(payloads.iter())
        .map(|((name, header), payload)| (**name, header, payload.as_slice()))
        .collect();

    write_npz_bytes_with_compression(&raw_entries, NpzCompression::Deflate)
}

// ── Memory-mapped file arrays (np.memmap) ──

/// A memory-mapped array backed by a file on disk.
///
/// Provides NumPy-compatible `np.memmap` semantics: the array data lives in a
/// file and is accessed through OS-level memory mapping. Changes to writable
/// mappings are flushed to disk via `flush()`.
///
/// File-backed array that provides the same API surface as `numpy.memmap`.
///
/// Instead of OS-level memory-mapping (which requires `unsafe`), this uses
/// safe standard I/O: the data region is read into an in-memory buffer,
/// and `flush()` writes modifications back to the backing file.
#[derive(Debug)]
pub struct MemmapArray {
    /// Array shape.
    pub shape: Vec<usize>,
    /// Element dtype.
    pub dtype: IOSupportedDType,
    /// Byte offset from file start to the array data region.
    pub offset: usize,
    /// Whether this is a fortran-order mapping.
    pub fortran_order: bool,
    /// In-memory buffer holding the array data.
    buffer: Vec<u8>,
    /// Backing file path (for flush support).
    backing_path: Option<std::path::PathBuf>,
    /// Whether this mapping is writable.
    writable: bool,
}

impl MemmapArray {
    /// Get a read-only byte slice of the array data.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Get a mutable byte slice of the array data.
    ///
    /// Returns `None` if the mapping is read-only.
    pub fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        if self.writable {
            Some(&mut self.buffer)
        } else {
            None
        }
    }

    /// Whether this memmap is writable.
    #[must_use]
    pub fn is_writable(&self) -> bool {
        self.writable
    }

    /// Total number of elements in the array.
    pub fn element_count(&self) -> Result<usize, IOError> {
        element_count(&self.shape)
    }

    /// Total byte size of the mapped array data region.
    pub fn nbytes(&self) -> Result<usize, IOError> {
        let count = self.element_count()?;
        let item_size = self
            .dtype
            .item_size()
            .ok_or(IOError::MemmapContractViolation(
                "unsupported dtype for memmap",
            ))?;
        count
            .checked_mul(item_size)
            .ok_or(IOError::MemmapContractViolation(
                "array byte size overflowed",
            ))
    }

    /// Flush changes to disk for writable mappings.
    ///
    /// No-op for read-only or copy-on-write mappings (no backing path).
    pub fn flush(&mut self) -> Result<(), IOError> {
        let path = match &self.backing_path {
            Some(p) => p.clone(),
            None => return Ok(()),
        };
        if !self.writable {
            return Ok(());
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .map_err(|_| IOError::MemmapContractViolation("flush: failed to open backing file"))?;
        file.seek(SeekFrom::Start(self.offset as u64))
            .map_err(|_| IOError::MemmapContractViolation("flush: failed to seek"))?;
        file.write_all(&self.buffer)
            .map_err(|_| IOError::MemmapContractViolation("flush: failed to write"))?;
        file.flush()
            .map_err(|_| IOError::MemmapContractViolation("flush: failed to sync"))?;
        Ok(())
    }

    /// Decode the mapped bytes as f64 values (for numeric dtypes).
    pub fn to_f64_values(&self) -> Result<Vec<f64>, IOError> {
        fromfile(self.as_bytes(), self.dtype, None)
    }

    /// Decode the mapped bytes as strings (for string dtypes).
    pub fn to_strings(&self) -> Result<Vec<String>, IOError> {
        fromfile_strings(self.as_bytes(), self.dtype, None)
    }
}

/// Open a memory-mapped array backed by a file.
///
/// This is the equivalent of `numpy.memmap(filename, dtype, mode, offset, shape)`.
///
/// # Modes
/// - `ReadOnly` ("r"): Open existing file read-only
/// - `ReadWrite` ("r+"): Open existing file read-write
/// - `Write` ("w+"): Create or truncate file, then map read-write
/// - `CopyOnWrite` ("c"): Map read-only with copy-on-write (changes not written to file)
///
/// # Arguments
/// - `path`: File path
/// - `dtype`: Element data type
/// - `mode`: Access mode
/// - `offset`: Byte offset into the file where array data begins
/// - `shape`: Array dimensions
pub fn memmap(
    path: &std::path::Path,
    dtype: IOSupportedDType,
    mode: MemmapMode,
    offset: usize,
    shape: &[usize],
) -> Result<MemmapArray, IOError> {
    if dtype == IOSupportedDType::Object {
        return Err(IOError::MemmapContractViolation(
            "object dtype is invalid for memmap path",
        ));
    }
    let item_size = dtype.item_size().ok_or(IOError::MemmapContractViolation(
        "unsupported dtype for memmap",
    ))?;
    let count = element_count(shape)?;
    let expected_bytes = count
        .checked_mul(item_size)
        .ok_or(IOError::MemmapContractViolation(
            "array byte size overflowed",
        ))?;

    match mode {
        MemmapMode::Write => {
            if expected_bytes == 0 {
                return Err(IOError::MemmapContractViolation(
                    "write memmap requires non-empty byte footprint",
                ));
            }
            // Create/truncate the file and pre-allocate it to the required size.
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)
                .map_err(|_| IOError::MemmapContractViolation("failed to create file"))?;
            let total_len = offset
                .checked_add(expected_bytes)
                .ok_or(IOError::MemmapContractViolation("file size overflowed"))?;
            file.set_len(total_len as u64)
                .map_err(|_| IOError::MemmapContractViolation("failed to set file length"))?;
            // Start with a zero-filled buffer (matches OS mmap behavior).
            let buffer = vec![0u8; expected_bytes];
            Ok(MemmapArray {
                shape: shape.to_vec(),
                dtype,
                offset,
                fortran_order: false,
                buffer,
                backing_path: Some(path.to_path_buf()),
                writable: true,
            })
        }
        MemmapMode::ReadWrite => {
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .map_err(|_| IOError::MemmapContractViolation("failed to open file for rw"))?;
            let file_len = file
                .metadata()
                .map_err(|_| IOError::MemmapContractViolation("failed to read file metadata"))?
                .len() as usize;
            let end = offset
                .checked_add(expected_bytes)
                .ok_or(IOError::MemmapContractViolation("mapping range overflowed"))?;
            if file_len < end {
                return Err(IOError::MemmapContractViolation(
                    "backing file is too small for requested mapping",
                ));
            }
            file.seek(SeekFrom::Start(offset as u64))
                .map_err(|_| IOError::MemmapContractViolation("failed to seek in file"))?;
            let mut buffer = vec![0u8; expected_bytes];
            file.read_exact(&mut buffer)
                .map_err(|_| IOError::MemmapContractViolation("failed to read file data"))?;
            Ok(MemmapArray {
                shape: shape.to_vec(),
                dtype,
                offset,
                fortran_order: false,
                buffer,
                backing_path: Some(path.to_path_buf()),
                writable: true,
            })
        }
        MemmapMode::ReadOnly | MemmapMode::CopyOnWrite => {
            let mut file = std::fs::File::open(path)
                .map_err(|_| IOError::MemmapContractViolation("failed to open file"))?;
            let file_len = file
                .metadata()
                .map_err(|_| IOError::MemmapContractViolation("failed to read file metadata"))?
                .len() as usize;
            let end = offset
                .checked_add(expected_bytes)
                .ok_or(IOError::MemmapContractViolation("mapping range overflowed"))?;
            if file_len < end {
                return Err(IOError::MemmapContractViolation(
                    "backing file is too small for requested mapping",
                ));
            }
            file.seek(SeekFrom::Start(offset as u64))
                .map_err(|_| IOError::MemmapContractViolation("failed to seek in file"))?;
            let mut buffer = vec![0u8; expected_bytes];
            file.read_exact(&mut buffer)
                .map_err(|_| IOError::MemmapContractViolation("failed to read file data"))?;
            let writable = mode == MemmapMode::CopyOnWrite;
            Ok(MemmapArray {
                shape: shape.to_vec(),
                dtype,
                offset,
                fortran_order: false,
                buffer,
                // ReadOnly and CopyOnWrite: no flush-back to the file.
                backing_path: None,
                writable,
            })
        }
    }
}

/// Open a memory-mapped array from an existing NPY file.
///
/// Reads the NPY header to determine shape/dtype, then memory-maps the
/// data payload region of the file.
pub fn memmap_npy(path: &std::path::Path, mode: MemmapMode) -> Result<MemmapArray, IOError> {
    if mode == MemmapMode::Write {
        return Err(IOError::MemmapContractViolation(
            "write mode is invalid for existing npy-backed memmap",
        ));
    }
    // Read the header to determine shape/dtype/offset
    let header_bytes = std::fs::read(path)
        .map_err(|_| IOError::MemmapContractViolation("failed to read NPY file for header"))?;
    let version = validate_magic_version(&header_bytes)?;
    let (header_offset, header_len) = read_header_span(&header_bytes, version)?;
    let header_end = header_offset + header_len;
    let header = parse_header_dictionary(&header_bytes[header_offset..header_end], header_len)?;
    let mut mapped = memmap(path, header.descr, mode, header_end, &header.shape)?;
    mapped.fortran_order = header.fortran_order;
    Ok(mapped)
}

// ── Structured / Record Dtype NPY support ──

/// A field in a structured/record dtype descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredIOField {
    /// Field name (e.g. "x", "label").
    pub name: String,
    /// Field dtype (e.g. `IOSupportedDType::F64`, `IOSupportedDType::Bytes(5)`).
    pub dtype: IOSupportedDType,
}

/// Descriptor for a structured/record dtype: an ordered list of named, typed fields.
///
/// Corresponds to NumPy's compound dtype descriptors such as
/// `[('x', '<f8'), ('y', '<i4'), ('label', '|S5')]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredIODescriptor {
    pub fields: Vec<StructuredIOField>,
}

impl StructuredIODescriptor {
    /// Total byte size per record (sum of field item sizes).
    pub fn record_size(&self) -> Result<usize, IOError> {
        let mut total = 0usize;
        for field in &self.fields {
            let sz = field
                .dtype
                .item_size()
                .ok_or(IOError::DTypeDescriptorInvalid)?;
            total = total
                .checked_add(sz)
                .ok_or(IOError::HeaderSchemaInvalid("record size overflowed"))?;
        }
        Ok(total)
    }

    /// Byte offset of each field within a record.
    pub fn field_offsets(&self) -> Result<Vec<usize>, IOError> {
        let mut offsets = Vec::with_capacity(self.fields.len());
        let mut offset = 0usize;
        for field in &self.fields {
            offsets.push(offset);
            let sz = field
                .dtype
                .item_size()
                .ok_or(IOError::DTypeDescriptorInvalid)?;
            offset = offset
                .checked_add(sz)
                .ok_or(IOError::HeaderSchemaInvalid("field offset overflowed"))?;
        }
        Ok(offsets)
    }

    /// Format as a NumPy-style structured descriptor string for NPY headers.
    ///
    /// E.g. `[('x', '<f8'), ('y', '<i4'), ('label', '|S5')]`
    pub fn to_descr_string(&self) -> String {
        let parts: Vec<String> = self
            .fields
            .iter()
            .map(|f| format!("('{}', '{}')", f.name, f.dtype.descr()))
            .collect();
        format!("[{}]", parts.join(", "))
    }
}

/// Parse a NumPy structured dtype descriptor string.
///
/// Handles the format: `[('x', '<f8'), ('y', '<i4'), ('label', '|S5')]`
pub fn parse_structured_descr(value: &str) -> Result<StructuredIODescriptor, IOError> {
    let value = value.trim();
    if !value.starts_with('[') || !value.ends_with(']') {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let inner = &value[1..value.len() - 1];
    let inner = inner.trim();
    if inner.is_empty() {
        return Err(IOError::DTypeDescriptorInvalid);
    }

    let mut fields = Vec::new();
    let mut pos = 0;
    let bytes = inner.as_bytes();

    while pos < bytes.len() {
        // Skip whitespace and commas between tuples
        while pos < bytes.len() && (bytes[pos].is_ascii_whitespace() || bytes[pos] == b',') {
            pos += 1;
        }
        if pos >= bytes.len() {
            break;
        }

        // Expect '('
        if bytes[pos] != b'(' {
            return Err(IOError::DTypeDescriptorInvalid);
        }
        pos += 1;

        // Parse field name (quoted string)
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let (name, new_pos) = parse_structured_quoted_string(inner, pos)?;
        pos = new_pos;

        // Skip comma
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= bytes.len() || bytes[pos] != b',' {
            return Err(IOError::DTypeDescriptorInvalid);
        }
        pos += 1;

        // Parse dtype descriptor (quoted string)
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let (dtype_str, new_pos) = parse_structured_quoted_string(inner, pos)?;
        pos = new_pos;

        // Skip to closing ')'
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= bytes.len() || bytes[pos] != b')' {
            return Err(IOError::DTypeDescriptorInvalid);
        }
        pos += 1;

        let dtype = IOSupportedDType::decode(&dtype_str)?;
        fields.push(StructuredIOField { name, dtype });
    }

    if fields.is_empty() {
        return Err(IOError::DTypeDescriptorInvalid);
    }

    Ok(StructuredIODescriptor { fields })
}

/// Parse a single-quoted or double-quoted string starting at `pos`.
/// Returns (string_content, position_after_closing_quote).
fn parse_structured_quoted_string(s: &str, pos: usize) -> Result<(String, usize), IOError> {
    let bytes = s.as_bytes();
    if pos >= bytes.len() {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let quote = bytes[pos];
    if quote != b'\'' && quote != b'"' {
        return Err(IOError::DTypeDescriptorInvalid);
    }

    let mut result = String::new();
    let mut escaped = false;
    let mut idx = pos + 1;

    while idx < bytes.len() {
        let b = bytes[idx];
        if b == quote && !escaped {
            return Ok((result, idx + 1));
        }

        if b == b'\\' && !escaped {
            escaped = true;
        } else {
            result.push(char::from(b));
            escaped = false;
        }
        idx += 1;
    }

    Err(IOError::DTypeDescriptorInvalid)
}

/// Encode a structured NPY header dictionary line.
fn encode_structured_header_dict(
    shape: &[usize],
    fortran_order: bool,
    descriptor: &StructuredIODescriptor,
) -> String {
    let fo = if fortran_order { "True" } else { "False" };
    let shape_str = format_shape_tuple(shape);
    format!(
        "{{'descr': {}, 'fortran_order': {fo}, 'shape': {shape_str}, }}",
        descriptor.to_descr_string()
    )
}

/// Result type for loading structured NPY arrays.
#[derive(Debug, Clone)]
pub struct StructuredNpyData {
    /// Array shape (excluding the structured field dimension).
    pub shape: Vec<usize>,
    /// Structured dtype descriptor (field names, types).
    pub descriptor: StructuredIODescriptor,
    /// Per-field raw bytes. Each entry corresponds to a field; the bytes
    /// are concatenated record-by-record (each chunk is field.item_size bytes).
    pub columns: Vec<Vec<u8>>,
}

/// Read raw binary bytes into structured per-field column data.
///
/// Each record in the binary data has fields packed consecutively
/// (field 0 bytes, then field 1 bytes, ..., then next record).
pub fn fromfile_structured(
    data: &[u8],
    descriptor: &StructuredIODescriptor,
    count: Option<usize>,
) -> Result<StructuredNpyData, IOError> {
    let record_size = descriptor.record_size()?;
    if record_size == 0 {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let max_records = data.len() / record_size;
    let n = match count {
        Some(c) => {
            if c > max_records {
                return Err(IOError::ReadPayloadIncomplete(
                    "insufficient data for requested structured record count",
                ));
            }
            c
        }
        None => max_records,
    };

    let offsets = descriptor.field_offsets()?;
    let mut columns: Vec<Vec<u8>> = descriptor
        .fields
        .iter()
        .map(|f| {
            let sz = f.dtype.item_size().unwrap_or(0);
            Vec::with_capacity(n * sz)
        })
        .collect();

    for record_idx in 0..n {
        let record_start = record_idx * record_size;
        for (field_idx, field) in descriptor.fields.iter().enumerate() {
            let field_size = field
                .dtype
                .item_size()
                .ok_or(IOError::DTypeDescriptorInvalid)?;
            let field_start = record_start + offsets[field_idx];
            let field_end = field_start + field_size;
            if field_end > data.len() {
                return Err(IOError::ReadPayloadIncomplete(
                    "structured record extends past end of data",
                ));
            }
            columns[field_idx].extend_from_slice(&data[field_start..field_end]);
        }
    }

    Ok(StructuredNpyData {
        shape: vec![n],
        descriptor: descriptor.clone(),
        columns,
    })
}

/// Write structured per-field column data to interleaved record bytes.
///
/// All columns must have the same number of records (column.len() / field.item_size).
pub fn tofile_structured(
    descriptor: &StructuredIODescriptor,
    columns: &[Vec<u8>],
) -> Result<Vec<u8>, IOError> {
    if columns.len() != descriptor.fields.len() {
        return Err(IOError::WriteContractViolation(
            "column count does not match structured descriptor field count",
        ));
    }

    let record_size = descriptor.record_size()?;
    if record_size == 0 {
        return Err(IOError::DTypeDescriptorInvalid);
    }

    // Determine record count from first column
    let n = if descriptor.fields.is_empty() {
        0
    } else {
        let first_item_size = descriptor.fields[0]
            .dtype
            .item_size()
            .ok_or(IOError::DTypeDescriptorInvalid)?;
        if first_item_size == 0 {
            return Err(IOError::DTypeDescriptorInvalid);
        }
        if !columns[0].len().is_multiple_of(first_item_size) {
            return Err(IOError::WriteContractViolation(
                "structured column byte length must be an exact multiple of field item size",
            ));
        }
        columns[0].len() / first_item_size
    };

    // Validate all columns have the same record count
    for (i, field) in descriptor.fields.iter().enumerate() {
        let item_size = field
            .dtype
            .item_size()
            .ok_or(IOError::DTypeDescriptorInvalid)?;
        if !columns[i].len().is_multiple_of(item_size) {
            return Err(IOError::WriteContractViolation(
                "structured column byte length must be an exact multiple of field item size",
            ));
        }
        let col_records = columns[i].len() / item_size;
        if col_records != n {
            return Err(IOError::WriteContractViolation(
                "structured columns have inconsistent record counts",
            ));
        }
    }

    let mut buf = Vec::with_capacity(n * record_size);
    for record_idx in 0..n {
        for (field_idx, field) in descriptor.fields.iter().enumerate() {
            let item_size = field
                .dtype
                .item_size()
                .ok_or(IOError::DTypeDescriptorInvalid)?;
            let start = record_idx * item_size;
            let end = start + item_size;
            buf.extend_from_slice(&columns[field_idx][start..end]);
        }
    }

    Ok(buf)
}

/// High-level save for structured arrays to NPY bytes.
pub fn save_structured(
    shape: &[usize],
    descriptor: &StructuredIODescriptor,
    columns: &[Vec<u8>],
) -> Result<Vec<u8>, IOError> {
    let expected_records = element_count(shape)?;
    let record_size = descriptor.record_size()?;

    // Validate record count matches shape
    if !descriptor.fields.is_empty() {
        let first_item_size = descriptor.fields[0]
            .dtype
            .item_size()
            .ok_or(IOError::DTypeDescriptorInvalid)?;
        if first_item_size > 0 {
            let n = columns.first().map_or(0, |c| c.len() / first_item_size);
            if n != expected_records {
                return Err(IOError::WriteContractViolation(
                    "structured record count does not match shape",
                ));
            }
        }
    }

    let payload = tofile_structured(descriptor, columns)?;

    // Build NPY header manually for structured dtype
    let dict_str = encode_structured_header_dict(shape, false, descriptor);
    let dict_bytes = dict_str.as_bytes();
    let version = (1u8, 0u8);
    let length_field_size = 2usize;
    let prefix_len = NPY_MAGIC_PREFIX.len() + 2 + length_field_size;
    let base_header_len = dict_bytes.len() + 1; // +1 for trailing newline
    let padding = (16 - ((prefix_len + base_header_len) % 16)) % 16;
    let header_len = base_header_len + padding;

    if header_len > MAX_HEADER_BYTES {
        return Err(IOError::HeaderSchemaInvalid(
            "structured header exceeds budget",
        ));
    }

    let payload_len = expected_records
        .checked_mul(record_size)
        .ok_or(IOError::WriteContractViolation("payload size overflow"))?;
    let total_size = prefix_len
        .checked_add(header_len)
        .and_then(|h| h.checked_add(payload_len))
        .ok_or(IOError::WriteContractViolation("total size overflow"))?;
    let mut out = Vec::with_capacity(total_size);
    write_npy_preamble(&mut out, version, header_len)?;
    out.extend_from_slice(dict_bytes);
    out.extend(std::iter::repeat_n(b' ', padding));
    out.push(b'\n');
    out.extend_from_slice(&payload);

    Ok(out)
}

/// High-level load for structured NPY arrays.
///
/// Detects structured dtype from the NPY header and returns per-field column data.
pub fn load_structured(data: &[u8]) -> Result<StructuredNpyData, IOError> {
    let version = validate_magic_version(data)?;
    let (header_offset, header_len) = read_header_span(data, version)?;
    let header_end = header_offset + header_len;
    let header_bytes = &data[header_offset..header_end];

    let dictionary = std::str::from_utf8(header_bytes).map_err(|_| {
        IOError::HeaderSchemaInvalid("header bytes must decode as utf-8/ascii dictionary")
    })?;
    let dictionary = dictionary.trim_end();
    let map = parse_header_dictionary_map(dictionary)?;
    validate_required_header_keys(&map)?;

    // Parse shape
    let shape = parse_shape_field(
        map.get("shape")
            .ok_or(IOError::HeaderSchemaInvalid("shape missing"))?,
    )?;
    let _fortran_order = parse_fortran_order_value(
        map.get("fortran_order")
            .ok_or(IOError::HeaderSchemaInvalid("fortran_order missing"))?,
    )?;

    // Parse structured descriptor
    let descr_tail = map
        .get("descr")
        .ok_or(IOError::HeaderSchemaInvalid("descr missing"))?;
    if !descr_tail.starts_with('[') {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let descriptor = parse_structured_descr(descr_tail)?;

    let body = &data[header_end..];
    let expected_records = element_count(&shape)?;
    let mut result = fromfile_structured(body, &descriptor, Some(expected_records))?;
    result.shape = shape;
    Ok(result)
}

// ── String / Unicode NPY support ──

/// Decode a single fixed-width byte string element from raw bytes.
///
/// NumPy `|Sn` stores each element as exactly `n` bytes, null-padded.
/// Returns the string with trailing null bytes stripped.
fn decode_bytes_element(chunk: &[u8]) -> String {
    let end = chunk.iter().rposition(|&b| b != 0).map_or(0, |pos| pos + 1);
    String::from_utf8_lossy(&chunk[..end]).into_owned()
}

/// Decode a single fixed-width Unicode element from raw UCS-4/UTF-32 bytes.
///
/// NumPy `<Un` or `>Un` stores each element as `n` UCS-4 code points (4 bytes each),
/// null-padded. Little-endian if `is_le`, big-endian otherwise.
fn decode_unicode_element(chunk: &[u8], is_le: bool) -> String {
    let mut chars = Vec::with_capacity(chunk.len() / 4);
    for code_unit in chunk.chunks_exact(4) {
        let cp = if is_le {
            u32::from_le_bytes([code_unit[0], code_unit[1], code_unit[2], code_unit[3]])
        } else {
            u32::from_be_bytes([code_unit[0], code_unit[1], code_unit[2], code_unit[3]])
        };
        if cp == 0 {
            break;
        }
        if let Some(c) = char::from_u32(cp) {
            chars.push(c);
        }
    }
    chars.into_iter().collect()
}

/// Encode a string as a fixed-width byte string (NumPy `|Sn`).
///
/// Pads with null bytes to exactly `width` bytes. Truncates if the string is longer.
fn encode_bytes_element(s: &str, width: usize) -> Vec<u8> {
    let mut buf = vec![0u8; width];
    let bytes = s.as_bytes();
    let copy_len = bytes.len().min(width);
    buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
    buf
}

/// Encode a string as a fixed-width Unicode element (NumPy `<Un` or `>Un`).
///
/// Each character is encoded as a 4-byte UCS-4/UTF-32 code point.
/// Pads with null code points to exactly `char_count` characters.
fn encode_unicode_element(s: &str, char_count: usize, is_le: bool) -> Vec<u8> {
    let mut buf = vec![0u8; char_count * 4];
    for (i, c) in s.chars().take(char_count).enumerate() {
        let cp = c as u32;
        let bytes = if is_le {
            cp.to_le_bytes()
        } else {
            cp.to_be_bytes()
        };
        let offset = i * 4;
        buf[offset..offset + 4].copy_from_slice(&bytes);
    }
    buf
}

/// Read raw binary data from bytes into a flat array of strings.
///
/// Handles `|Sn` (byte strings), `<Un` (LE Unicode), and `>Un` (BE Unicode) dtypes.
pub fn fromfile_strings(
    data: &[u8],
    dtype: IOSupportedDType,
    count: Option<usize>,
) -> Result<Vec<String>, IOError> {
    if !dtype.is_string() {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let item_size = dtype.item_size().ok_or(IOError::DTypeDescriptorInvalid)?;
    if item_size == 0 {
        return Ok(Vec::new());
    }
    let max_elems = data.len() / item_size;
    let n = match count {
        Some(c) => {
            if c > max_elems {
                return Err(IOError::ReadPayloadIncomplete(
                    "insufficient data for requested element count",
                ));
            }
            c
        }
        None => max_elems,
    };

    let mut strings = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * item_size;
        let chunk = &data[offset..offset + item_size];
        let s = match dtype {
            IOSupportedDType::Bytes(_) => decode_bytes_element(chunk),
            IOSupportedDType::Unicode(_) => decode_unicode_element(chunk, true),
            IOSupportedDType::UnicodeBe(_) => decode_unicode_element(chunk, false),
            _ => return Err(IOError::DTypeDescriptorInvalid),
        };
        strings.push(s);
    }
    Ok(strings)
}

/// Write string array values to raw binary bytes.
///
/// Encodes each string according to the given string `dtype` (`|Sn`, `<Un`, `>Un`).
pub fn tofile_strings(strings: &[String], dtype: IOSupportedDType) -> Result<Vec<u8>, IOError> {
    if !dtype.is_string() {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let item_size = dtype.item_size().ok_or(IOError::DTypeDescriptorInvalid)?;
    let mut buf = Vec::with_capacity(strings.len() * item_size);
    for s in strings {
        let encoded = match dtype {
            IOSupportedDType::Bytes(w) => encode_bytes_element(s, w),
            IOSupportedDType::Unicode(w) => encode_unicode_element(s, w, true),
            IOSupportedDType::UnicodeBe(w) => encode_unicode_element(s, w, false),
            _ => return Err(IOError::DTypeDescriptorInvalid),
        };
        buf.extend_from_slice(&encoded);
    }
    Ok(buf)
}

/// High-level save for string arrays to NPY bytes (np.save equivalent for string dtypes).
pub fn save_strings(
    shape: &[usize],
    strings: &[String],
    dtype: IOSupportedDType,
) -> Result<Vec<u8>, IOError> {
    if !dtype.is_string() {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let expected = element_count(shape)?;
    if strings.len() != expected {
        return Err(IOError::WriteContractViolation(
            "string count does not match shape element count",
        ));
    }
    let header = NpyHeader {
        shape: shape.to_vec(),
        fortran_order: false,
        descr: dtype,
    };
    let payload = tofile_strings(strings, dtype)?;
    write_npy_bytes(&header, &payload, false)
}

/// Result type for loading string NPY arrays.
pub type NpyLoadedStrings = (Vec<usize>, Vec<String>, IOSupportedDType);

/// High-level load for string-typed NPY arrays (np.load equivalent for string dtypes).
pub fn load_strings(data: &[u8]) -> Result<NpyLoadedStrings, IOError> {
    let npy = read_npy_bytes(data, false)?;
    let dtype = npy.header.descr;
    if !dtype.is_string() {
        return Err(IOError::DTypeDescriptorInvalid);
    }
    let shape = npy.header.shape;
    let strings = fromfile_strings(&npy.payload, dtype, None)?;
    Ok((shape, strings, dtype))
}

#[cfg(test)]
mod tests {
    use super::{
        IO_PACKET_ID, IO_PACKET_REASON_CODES, IOError, IOLogRecord, IORuntimeMode,
        IOSupportedDType, LoadDispatch, MAX_ARCHIVE_MEMBERS, MAX_DISPATCH_RETRIES,
        MAX_HEADER_BYTES, MAX_MEMMAP_VALIDATION_RETRIES, MemmapMode, NPY_MAGIC_PREFIX,
        NPZ_MAGIC_PREFIX, NpyHeader, NpzCompression, SaveTxtConfig, StructuredIODescriptor,
        StructuredIOField, classify_load_dispatch, encode_npy_header_bytes, enforce_pickle_policy,
        fromfile, fromfile_complex, fromfile_strings, fromfile_structured, fromstring, genfromtxt,
        load, load_complex, load_npz, load_strings, load_structured, loadtxt, loadtxt_usecols,
        memmap, memmap_npy, parse_structured_descr, read_npy_bytes, read_npz_bytes, save,
        save_complex, save_strings, save_structured, savetxt, savez, savez_compressed,
        synthesize_npz_member_names, tobytes, tofile, tofile_complex, tofile_strings,
        tofile_structured, tostring, validate_descriptor_roundtrip, validate_header_schema,
        validate_io_policy_metadata, validate_magic_version, validate_memmap_contract,
        validate_npz_archive_budget, validate_read_payload, validate_write_contract,
        write_npy_bytes, write_npy_bytes_with_version, write_npy_preamble, write_npz_bytes,
        write_npz_bytes_with_compression,
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
            IOSupportedDType::Complex64,
            IOSupportedDType::Complex64Be,
            IOSupportedDType::Complex128,
            IOSupportedDType::Complex128Be,
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
        assert_eq!(decoded.payload, payload.into());
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
        assert_eq!(decoded.payload, vec![0x80, 0x05, 0x4B, 0x01, 0x2E].into());
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
        assert_eq!(decoded.payload, payload.into());
    }

    #[test]
    fn npy_header_parser_accepts_extra_keys() {
        let payload = [10_i32, 20_i32]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();

        let extra_key_header =
            "{'descr': '<i4', 'fortran_order': False, 'shape': (2,), 'extra': 1, }";
        let extra_key_bytes = make_manual_npy_payload(extra_key_header, &payload);
        let npy = read_npy_bytes(&extra_key_bytes, false).expect("extra key must be allowed");
        assert_eq!(npy.header.descr, IOSupportedDType::I32);
        assert_eq!(npy.header.shape, vec![2]);
    }

    #[test]
    fn npy_header_parser_rejects_singleton_without_comma() {
        let payload = [10_i32, 20_i32]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();

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
        assert_eq!(decoded.payload, payload.into());

        let header_u8 = NpyHeader {
            shape: vec![4],
            fortran_order: false,
            descr: IOSupportedDType::U8,
        };
        let payload_u8 = vec![0u8, 127, 128, 255];
        let encoded_u8 = write_npy_bytes(&header_u8, &payload_u8, false).expect("write u8");
        let decoded_u8 = read_npy_bytes(&encoded_u8, false).expect("read u8");
        assert_eq!(decoded_u8.header.descr, IOSupportedDType::U8);
        assert_eq!(decoded_u8.payload, payload_u8.into());
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
        assert_eq!(decoded.payload, payload.into());
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
        assert_eq!(decoded.payload, payload.into());

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
        assert_eq!(decoded.payload, payload.into());
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
        assert_eq!(IOSupportedDType::Complex64.item_size(), Some(8));
        assert_eq!(IOSupportedDType::Complex64Be.item_size(), Some(8));
        assert_eq!(IOSupportedDType::Complex128.item_size(), Some(16));
        assert_eq!(IOSupportedDType::Complex128Be.item_size(), Some(16));
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
        assert_eq!(entries[0].array.payload, payload.into());
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
        assert_eq!(entries[0].array.payload, p1.into());

        assert_eq!(entries[1].name, "matrix");
        assert_eq!(entries[1].array.header.shape, vec![2, 2]);
        assert_eq!(entries[1].array.header.descr, IOSupportedDType::I32);
        assert_eq!(entries[1].array.payload, p2.into());
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
        assert_eq!(entries[0].array.payload, payload.into());
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
        assert_eq!(entries[0].array.payload, vec![1, 0, 1].into());
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
        let data: Vec<u8> = [42i32, -7].iter().flat_map(|v| v.to_le_bytes()).collect();
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
        let expected: Vec<u8> = [42i32, -7].iter().flat_map(|v| v.to_le_bytes()).collect();
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
    fn fromfile_tofile_complex128_roundtrip() {
        let original = vec![(1.25, -2.5), (0.0, 3.5), (-4.0, 0.25)];
        let encoded = tofile_complex(&original, IOSupportedDType::Complex128).unwrap();
        let decoded = fromfile_complex(&encoded, IOSupportedDType::Complex128, None).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn fromfile_tofile_complex64_big_endian_roundtrip() {
        let original = vec![(1.5, -2.0), (3.25, 0.5)];
        let encoded = tofile_complex(&original, IOSupportedDType::Complex64Be).unwrap();
        let decoded = fromfile_complex(&encoded, IOSupportedDType::Complex64Be, None).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn scalar_binary_helpers_reject_complex_dtype() {
        let err = tofile(&[1.0, 2.0], IOSupportedDType::Complex128).expect_err("complex dtype");
        assert_eq!(err.reason_code(), "io_dtype_descriptor_invalid");

        let err =
            fromfile(&[0u8; 16], IOSupportedDType::Complex128, None).expect_err("complex dtype");
        assert_eq!(err.reason_code(), "io_dtype_descriptor_invalid");
    }

    #[test]
    fn fromfile_truncated_data_reads_partial() {
        // 7 bytes: only 3 complete u16 elements (6 bytes), last byte ignored
        let data = vec![1, 0, 2, 0, 3, 0, 99];
        let vals = fromfile(&data, IOSupportedDType::U16, None).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    // ── fromstring / tobytes / tostring tests ──

    #[test]
    fn fromstring_text_mode_space_separated() {
        let data = b"1.0 2.5 3.7 4.0";
        let vals = fromstring(data, IOSupportedDType::F64, " ").unwrap();
        assert_eq!(vals, vec![1.0, 2.5, 3.7, 4.0]);
    }

    #[test]
    fn fromstring_text_mode_comma_separated() {
        let data = b"10, 20, 30";
        let vals = fromstring(data, IOSupportedDType::F64, ",").unwrap();
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn fromstring_binary_mode_f64() {
        let original = vec![1.5, -2.7, 3.25];
        let bytes = tobytes(&original, IOSupportedDType::F64).unwrap();
        let decoded = fromstring(&bytes, IOSupportedDType::F64, "").unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn fromstring_binary_mode_i32() {
        let original = vec![1.0, 2.0, 3.0];
        let bytes = tobytes(&original, IOSupportedDType::I32).unwrap();
        let decoded = fromstring(&bytes, IOSupportedDType::I32, "").unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn fromstring_text_empty_input() {
        let data = b"";
        let vals = fromstring(data, IOSupportedDType::F64, " ").unwrap();
        assert!(vals.is_empty());
    }

    #[test]
    fn fromstring_text_invalid_parse() {
        let data = b"1.0 abc 3.0";
        let err = fromstring(data, IOSupportedDType::F64, " ").expect_err("parse should fail");
        assert!(matches!(err, IOError::ReadPayloadIncomplete(_)));
    }

    #[test]
    fn tobytes_f64_roundtrip() {
        let original = vec![42.0, -1.5, 0.0, f64::INFINITY];
        let bytes = tobytes(&original, IOSupportedDType::F64).unwrap();
        assert_eq!(bytes.len(), 4 * 8);
        let decoded = fromstring(&bytes, IOSupportedDType::F64, "").unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn tostring_basic() {
        let vals = vec![1.0, 2.0, 3.0];
        let text = tostring(&vals, ", ");
        assert_eq!(text, "1, 2, 3");
    }

    #[test]
    fn tostring_float_values() {
        let vals = vec![1.5, 2.7, 3.25];
        let text = tostring(&vals, " ");
        assert_eq!(text, "1.5 2.7 3.25");
    }

    // ── High-level convenience function tests ──

    #[test]
    fn save_load_roundtrip() {
        let shape = &[2, 3];
        let values = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes = save(shape, values, IOSupportedDType::F64).unwrap();
        let (loaded_shape, loaded_values, loaded_dtype) = load(&bytes).unwrap();
        assert_eq!(loaded_shape, shape);
        assert_eq!(loaded_values, values);
        assert_eq!(loaded_dtype, IOSupportedDType::F64);
    }

    #[test]
    fn save_load_complex_roundtrip() {
        let shape = &[2];
        let values = &[(1.0, -1.5), (2.5, 3.0)];
        let bytes = save_complex(shape, values, IOSupportedDType::Complex128).unwrap();
        let (loaded_shape, loaded_values, loaded_dtype) = load_complex(&bytes).unwrap();
        assert_eq!(loaded_shape, shape);
        assert_eq!(loaded_values, values);
        assert_eq!(loaded_dtype, IOSupportedDType::Complex128);
    }

    #[test]
    fn load_complex_rejects_non_complex_dtype() {
        let bytes = save(&[2], &[1.0, 2.0], IOSupportedDType::F64).unwrap();
        let err = load_complex(&bytes).expect_err("non-complex dtype should fail");
        assert_eq!(err.reason_code(), "io_dtype_descriptor_invalid");
    }

    #[test]
    fn savez_load_npz_roundtrip() {
        let entries: Vec<(&str, &[usize], &[f64], IOSupportedDType)> = vec![
            ("x", &[3], &[1.0, 2.0, 3.0], IOSupportedDType::F64),
            ("y", &[2], &[4.0, 5.0], IOSupportedDType::F64),
        ];
        let bytes = savez(&entries).unwrap();
        let loaded = load_npz(&bytes).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "x");
        assert_eq!(loaded[0].1, vec![3]);
        assert_eq!(loaded[0].2, vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded[1].0, "y");
        assert_eq!(loaded[1].2, vec![4.0, 5.0]);
    }

    #[test]
    fn savez_compressed_roundtrip() {
        let entries: Vec<(&str, &[usize], &[f64], IOSupportedDType)> = vec![(
            "arr",
            &[4],
            &[10.0, 20.0, 30.0, 40.0],
            IOSupportedDType::F64,
        )];
        let bytes = savez_compressed(&entries).unwrap();
        let loaded = load_npz(&bytes).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "arr");
        assert_eq!(loaded[0].2, vec![10.0, 20.0, 30.0, 40.0]);
    }

    // ── String / Unicode NPY tests ──

    #[test]
    fn bytes_dtype_descriptor_roundtrip() {
        let dt = IOSupportedDType::Bytes(10);
        assert_eq!(dt.descr(), "|S10");
        assert_eq!(IOSupportedDType::decode("|S10").unwrap(), dt);
        assert_eq!(dt.item_size(), Some(10));
        assert!(dt.is_string());
        assert!(!dt.is_complex());
        validate_descriptor_roundtrip(dt).unwrap();
    }

    #[test]
    fn unicode_le_dtype_descriptor_roundtrip() {
        let dt = IOSupportedDType::Unicode(20);
        assert_eq!(dt.descr(), "<U20");
        assert_eq!(IOSupportedDType::decode("<U20").unwrap(), dt);
        assert_eq!(dt.item_size(), Some(80));
        assert!(dt.is_string());
        validate_descriptor_roundtrip(dt).unwrap();
    }

    #[test]
    fn unicode_be_dtype_descriptor_roundtrip() {
        let dt = IOSupportedDType::UnicodeBe(5);
        assert_eq!(dt.descr(), ">U5");
        assert_eq!(IOSupportedDType::decode(">U5").unwrap(), dt);
        assert_eq!(dt.item_size(), Some(20));
        assert!(dt.is_string());
        validate_descriptor_roundtrip(dt).unwrap();
    }

    #[test]
    fn string_dtype_decode_rejects_zero_width() {
        assert!(IOSupportedDType::decode("|S0").is_err());
        assert!(IOSupportedDType::decode("<U0").is_err());
    }

    #[test]
    fn string_dtype_decode_rejects_invalid() {
        assert!(IOSupportedDType::decode("|S").is_err());
        assert!(IOSupportedDType::decode("<Uabc").is_err());
        assert!(IOSupportedDType::decode("|X5").is_err());
    }

    #[test]
    fn fromfile_tofile_bytes_roundtrip() {
        let strings = vec!["hello".to_string(), "world".to_string(), "hi".to_string()];
        let dtype = IOSupportedDType::Bytes(10);
        let encoded = tofile_strings(&strings, dtype).unwrap();
        assert_eq!(encoded.len(), 30); // 3 * 10 bytes
        let decoded = fromfile_strings(&encoded, dtype, None).unwrap();
        assert_eq!(decoded, strings);
    }

    #[test]
    fn fromfile_tofile_unicode_le_roundtrip() {
        let strings = vec!["alpha".to_string(), "beta".to_string()];
        let dtype = IOSupportedDType::Unicode(8);
        let encoded = tofile_strings(&strings, dtype).unwrap();
        assert_eq!(encoded.len(), 64); // 2 * 8 * 4 bytes
        let decoded = fromfile_strings(&encoded, dtype, None).unwrap();
        assert_eq!(decoded, strings);
    }

    #[test]
    fn fromfile_tofile_unicode_be_roundtrip() {
        let strings = vec!["abc".to_string(), "xyz".to_string()];
        let dtype = IOSupportedDType::UnicodeBe(5);
        let encoded = tofile_strings(&strings, dtype).unwrap();
        assert_eq!(encoded.len(), 40); // 2 * 5 * 4 bytes
        let decoded = fromfile_strings(&encoded, dtype, None).unwrap();
        assert_eq!(decoded, strings);
    }

    #[test]
    fn fromfile_strings_with_count() {
        let strings = vec!["aa".to_string(), "bb".to_string(), "cc".to_string()];
        let dtype = IOSupportedDType::Bytes(4);
        let encoded = tofile_strings(&strings, dtype).unwrap();
        let decoded = fromfile_strings(&encoded, dtype, Some(2)).unwrap();
        assert_eq!(decoded, vec!["aa", "bb"]);
    }

    #[test]
    fn bytes_truncation_on_encode() {
        let strings = vec!["toolongstring".to_string()];
        let dtype = IOSupportedDType::Bytes(5);
        let encoded = tofile_strings(&strings, dtype).unwrap();
        let decoded = fromfile_strings(&encoded, dtype, None).unwrap();
        assert_eq!(decoded, vec!["toolo"]);
    }

    #[test]
    fn unicode_truncation_on_encode() {
        let strings = vec!["abcdefgh".to_string()];
        let dtype = IOSupportedDType::Unicode(3);
        let encoded = tofile_strings(&strings, dtype).unwrap();
        let decoded = fromfile_strings(&encoded, dtype, None).unwrap();
        assert_eq!(decoded, vec!["abc"]);
    }

    #[test]
    fn bytes_null_padding() {
        let dtype = IOSupportedDType::Bytes(8);
        let strings = vec!["hi".to_string()];
        let encoded = tofile_strings(&strings, dtype).unwrap();
        assert_eq!(encoded.len(), 8);
        assert_eq!(&encoded[..2], b"hi");
        assert_eq!(&encoded[2..], &[0, 0, 0, 0, 0, 0]);
        let decoded = fromfile_strings(&encoded, dtype, None).unwrap();
        assert_eq!(decoded, vec!["hi"]);
    }

    #[test]
    fn empty_string_roundtrip() {
        let strings = vec!["".to_string(), "x".to_string(), "".to_string()];
        let dtype = IOSupportedDType::Bytes(4);
        let encoded = tofile_strings(&strings, dtype).unwrap();
        let decoded = fromfile_strings(&encoded, dtype, None).unwrap();
        assert_eq!(decoded, strings);
    }

    #[test]
    fn unicode_non_ascii_roundtrip() {
        let strings = vec!["caf\u{00e9}".to_string(), "\u{03b1}\u{03b2}".to_string()];
        let dtype = IOSupportedDType::Unicode(10);
        let encoded = tofile_strings(&strings, dtype).unwrap();
        let decoded = fromfile_strings(&encoded, dtype, None).unwrap();
        assert_eq!(decoded, strings);
    }

    #[test]
    fn save_load_strings_bytes_roundtrip() {
        let strings = vec![
            "foo".to_string(),
            "bar".to_string(),
            "baz".to_string(),
            "qux".to_string(),
        ];
        let dtype = IOSupportedDType::Bytes(6);
        let npy_bytes = save_strings(&[4], &strings, dtype).unwrap();
        let (shape, loaded, loaded_dtype) = load_strings(&npy_bytes).unwrap();
        assert_eq!(shape, vec![4]);
        assert_eq!(loaded, strings);
        assert_eq!(loaded_dtype, dtype);
    }

    #[test]
    fn save_load_strings_unicode_le_roundtrip() {
        let strings = vec!["hello".to_string(), "world".to_string()];
        let dtype = IOSupportedDType::Unicode(10);
        let npy_bytes = save_strings(&[2], &strings, dtype).unwrap();
        let (shape, loaded, loaded_dtype) = load_strings(&npy_bytes).unwrap();
        assert_eq!(shape, vec![2]);
        assert_eq!(loaded, strings);
        assert_eq!(loaded_dtype, dtype);
    }

    #[test]
    fn save_load_strings_2d_roundtrip() {
        let strings = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
            "f".to_string(),
        ];
        let dtype = IOSupportedDType::Bytes(3);
        let npy_bytes = save_strings(&[2, 3], &strings, dtype).unwrap();
        let (shape, loaded, loaded_dtype) = load_strings(&npy_bytes).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(loaded, strings);
        assert_eq!(loaded_dtype, dtype);
    }

    #[test]
    fn save_strings_rejects_non_string_dtype() {
        let strings = vec!["test".to_string()];
        let err = save_strings(&[1], &strings, IOSupportedDType::F64).unwrap_err();
        assert_eq!(err.reason_code(), "io_dtype_descriptor_invalid");
    }

    #[test]
    fn save_strings_rejects_shape_mismatch() {
        let strings = vec!["a".to_string(), "b".to_string()];
        let err = save_strings(&[3], &strings, IOSupportedDType::Bytes(4)).unwrap_err();
        assert_eq!(err.reason_code(), "io_write_contract_violation");
    }

    #[test]
    fn load_strings_rejects_non_string_npy() {
        let npy_bytes = save(&[2], &[1.0, 2.0], IOSupportedDType::F64).unwrap();
        let err = load_strings(&npy_bytes).unwrap_err();
        assert_eq!(err.reason_code(), "io_dtype_descriptor_invalid");
    }

    #[test]
    fn fromfile_rejects_string_dtype() {
        let err = fromfile(&[0; 10], IOSupportedDType::Bytes(5), None).unwrap_err();
        assert_eq!(err.reason_code(), "io_dtype_descriptor_invalid");
    }

    #[test]
    fn tofile_rejects_string_dtype() {
        let err = tofile(&[1.0], IOSupportedDType::Bytes(5)).unwrap_err();
        assert_eq!(err.reason_code(), "io_dtype_descriptor_invalid");
    }

    #[test]
    fn fromfile_strings_rejects_numeric_dtype() {
        let err = fromfile_strings(&[0; 8], IOSupportedDType::F64, None).unwrap_err();
        assert_eq!(err.reason_code(), "io_dtype_descriptor_invalid");
    }

    #[test]
    fn npy_header_with_string_dtype_parses() {
        let header_literal = "{'descr': '|S5', 'fortran_order': False, 'shape': (3,), }          ";
        // 3 elements × 5 bytes each = 15 bytes total
        let body = b"helloworldhi\x00\x00\x00";
        let payload = make_manual_npy_payload(header_literal, body);
        let npy = read_npy_bytes(&payload, false).unwrap();
        assert_eq!(npy.header.descr, IOSupportedDType::Bytes(5));
        assert_eq!(npy.header.shape, vec![3]);
        let strings = fromfile_strings(&npy.payload, npy.header.descr, None).unwrap();
        assert_eq!(strings, vec!["hello", "world", "hi"]);
    }

    #[test]
    fn npy_header_with_unicode_dtype_parses() {
        let header = "{'descr': '<U3', 'fortran_order': False, 'shape': (2,), }          ";
        let mut body = Vec::new();
        for c in "abc".chars() {
            body.extend_from_slice(&(c as u32).to_le_bytes());
        }
        for c in "xy".chars() {
            body.extend_from_slice(&(c as u32).to_le_bytes());
        }
        body.extend_from_slice(&0u32.to_le_bytes()); // null padding for 3rd char
        let payload = make_manual_npy_payload(header, &body);
        let npy = read_npy_bytes(&payload, false).unwrap();
        assert_eq!(npy.header.descr, IOSupportedDType::Unicode(3));
        let strings = fromfile_strings(&npy.payload, npy.header.descr, None).unwrap();
        assert_eq!(strings, vec!["abc", "xy"]);
    }

    // ── Structured / Record Dtype NPY tests ──

    fn make_test_descriptor() -> StructuredIODescriptor {
        StructuredIODescriptor {
            fields: vec![
                StructuredIOField {
                    name: "x".to_string(),
                    dtype: IOSupportedDType::F64,
                },
                StructuredIOField {
                    name: "y".to_string(),
                    dtype: IOSupportedDType::I32,
                },
            ],
        }
    }

    #[test]
    fn structured_descriptor_record_size() {
        let desc = make_test_descriptor();
        assert_eq!(desc.record_size().unwrap(), 12); // 8 + 4
    }

    #[test]
    fn structured_descriptor_field_offsets() {
        let desc = make_test_descriptor();
        let offsets = desc.field_offsets().unwrap();
        assert_eq!(offsets, vec![0, 8]);
    }

    #[test]
    fn structured_descriptor_to_descr_string() {
        let desc = make_test_descriptor();
        assert_eq!(desc.to_descr_string(), "[('x', '<f8'), ('y', '<i4')]");
    }

    #[test]
    fn parse_structured_descr_basic() {
        let desc = parse_structured_descr("[('x', '<f8'), ('y', '<i4')]").unwrap();
        assert_eq!(desc.fields.len(), 2);
        assert_eq!(desc.fields[0].name, "x");
        assert_eq!(desc.fields[0].dtype, IOSupportedDType::F64);
        assert_eq!(desc.fields[1].name, "y");
        assert_eq!(desc.fields[1].dtype, IOSupportedDType::I32);
    }

    #[test]
    fn parse_structured_descr_with_string_field() {
        let desc = parse_structured_descr("[('name', '|S10'), ('value', '<f8')]").unwrap();
        assert_eq!(desc.fields.len(), 2);
        assert_eq!(desc.fields[0].name, "name");
        assert_eq!(desc.fields[0].dtype, IOSupportedDType::Bytes(10));
        assert_eq!(desc.fields[1].name, "value");
        assert_eq!(desc.fields[1].dtype, IOSupportedDType::F64);
    }

    #[test]
    fn parse_structured_descr_single_field() {
        let desc = parse_structured_descr("[('z', '<f4')]").unwrap();
        assert_eq!(desc.fields.len(), 1);
        assert_eq!(desc.fields[0].name, "z");
        assert_eq!(desc.fields[0].dtype, IOSupportedDType::F32);
    }

    #[test]
    fn parse_structured_descr_rejects_empty() {
        assert!(parse_structured_descr("[]").is_err());
        assert!(parse_structured_descr("").is_err());
        assert!(parse_structured_descr("<f8").is_err());
    }

    #[test]
    fn structured_descriptor_roundtrip() {
        let desc = make_test_descriptor();
        let s = desc.to_descr_string();
        let parsed = parse_structured_descr(&s).unwrap();
        assert_eq!(parsed, desc);
    }

    #[test]
    fn fromfile_tofile_structured_roundtrip() {
        let desc = make_test_descriptor();
        // 2 records: (1.5, 10), (2.75, 20)
        let col_x: Vec<u8> = [1.5f64.to_le_bytes(), 2.75f64.to_le_bytes()].concat();
        let col_y: Vec<u8> = [10i32.to_le_bytes(), 20i32.to_le_bytes()].concat();
        let columns = vec![col_x.clone(), col_y.clone()];

        let encoded = tofile_structured(&desc, &columns).unwrap();
        // Each record is 12 bytes (8+4), 2 records = 24 bytes
        assert_eq!(encoded.len(), 24);

        let result = fromfile_structured(&encoded, &desc, None).unwrap();
        assert_eq!(result.columns.len(), 2);
        assert_eq!(result.columns[0], col_x);
        assert_eq!(result.columns[1], col_y);
    }

    #[test]
    fn fromfile_structured_with_count() {
        let desc = make_test_descriptor();
        let col_x: Vec<u8> = [
            1.0f64.to_le_bytes(),
            2.0f64.to_le_bytes(),
            3.0f64.to_le_bytes(),
        ]
        .concat();
        let col_y: Vec<u8> = [
            10i32.to_le_bytes(),
            20i32.to_le_bytes(),
            30i32.to_le_bytes(),
        ]
        .concat();
        let encoded = tofile_structured(&desc, &[col_x, col_y]).unwrap();
        let result = fromfile_structured(&encoded, &desc, Some(2)).unwrap();
        assert_eq!(result.shape, vec![2]);
        // Only first 2 records decoded
        assert_eq!(result.columns[0].len(), 16); // 2 * 8 bytes
        assert_eq!(result.columns[1].len(), 8); // 2 * 4 bytes
    }

    #[test]
    fn tofile_structured_rejects_column_mismatch() {
        let desc = make_test_descriptor();
        let err = tofile_structured(&desc, &[vec![0u8; 8]]).unwrap_err();
        assert_eq!(err.reason_code(), "io_write_contract_violation");
    }

    #[test]
    fn tofile_structured_rejects_inconsistent_records() {
        let desc = make_test_descriptor();
        // col_x has 2 records but col_y has 1
        let col_x = vec![0u8; 16]; // 2 * 8
        let col_y = vec![0u8; 4]; // 1 * 4
        let err = tofile_structured(&desc, &[col_x, col_y]).unwrap_err();
        assert_eq!(err.reason_code(), "io_write_contract_violation");
    }

    #[test]
    fn save_load_structured_roundtrip() {
        let desc = make_test_descriptor();
        let col_x: Vec<u8> = [
            1.5f64.to_le_bytes(),
            2.75f64.to_le_bytes(),
            3.25f64.to_le_bytes(),
        ]
        .concat();
        let col_y: Vec<u8> = [
            10i32.to_le_bytes(),
            20i32.to_le_bytes(),
            30i32.to_le_bytes(),
        ]
        .concat();

        let npy_bytes = save_structured(&[3], &desc, &[col_x.clone(), col_y.clone()]).unwrap();
        let loaded = load_structured(&npy_bytes).unwrap();

        assert_eq!(loaded.shape, vec![3]);
        assert_eq!(loaded.descriptor, desc);
        assert_eq!(loaded.columns.len(), 2);
        assert_eq!(loaded.columns[0], col_x);
        assert_eq!(loaded.columns[1], col_y);
    }

    #[test]
    fn save_load_structured_with_string_field() {
        let desc = StructuredIODescriptor {
            fields: vec![
                StructuredIOField {
                    name: "id".to_string(),
                    dtype: IOSupportedDType::I32,
                },
                StructuredIOField {
                    name: "label".to_string(),
                    dtype: IOSupportedDType::Bytes(5),
                },
            ],
        };
        // 2 records: (1, "hello"), (2, "world")
        let col_id: Vec<u8> = [1i32.to_le_bytes(), 2i32.to_le_bytes()].concat();
        let col_label: Vec<u8> = [b"hello".to_vec(), b"world".to_vec()].concat();

        let npy_bytes = save_structured(&[2], &desc, &[col_id.clone(), col_label.clone()]).unwrap();
        let loaded = load_structured(&npy_bytes).unwrap();

        assert_eq!(loaded.shape, vec![2]);
        assert_eq!(loaded.columns[0], col_id);
        assert_eq!(loaded.columns[1], col_label);
        // Decode string column
        let strings =
            fromfile_strings(&loaded.columns[1], IOSupportedDType::Bytes(5), None).unwrap();
        assert_eq!(strings, vec!["hello", "world"]);
    }

    #[test]
    fn save_structured_rejects_shape_mismatch() {
        let desc = make_test_descriptor();
        let col_x = vec![0u8; 16]; // 2 records
        let col_y = vec![0u8; 8]; // 2 records
        // Shape says 5 but only 2 records
        let err = save_structured(&[5], &desc, &[col_x, col_y]).unwrap_err();
        assert_eq!(err.reason_code(), "io_write_contract_violation");
    }

    #[test]
    fn tofile_structured_rejects_partial_field_bytes() {
        let desc = make_test_descriptor();
        let col_x = vec![0u8; 17];
        let col_y = vec![0u8; 8];

        let err = tofile_structured(&desc, &[col_x, col_y])
            .expect_err("partial field bytes must fail closed");

        assert_eq!(err.reason_code(), "io_write_contract_violation");
    }

    #[test]
    fn structured_three_fields_roundtrip() {
        let desc = StructuredIODescriptor {
            fields: vec![
                StructuredIOField {
                    name: "a".to_string(),
                    dtype: IOSupportedDType::F32,
                },
                StructuredIOField {
                    name: "b".to_string(),
                    dtype: IOSupportedDType::I64,
                },
                StructuredIOField {
                    name: "c".to_string(),
                    dtype: IOSupportedDType::U8,
                },
            ],
        };
        assert_eq!(desc.record_size().unwrap(), 13); // 4 + 8 + 1
        let col_a: Vec<u8> = [1.5f32.to_le_bytes(), 2.5f32.to_le_bytes()].concat();
        let col_b: Vec<u8> = [100i64.to_le_bytes(), 200i64.to_le_bytes()].concat();
        let col_c: Vec<u8> = vec![42u8, 84u8];

        let npy_bytes =
            save_structured(&[2], &desc, &[col_a.clone(), col_b.clone(), col_c.clone()]).unwrap();
        let loaded = load_structured(&npy_bytes).unwrap();
        assert_eq!(loaded.shape, vec![2]);
        assert_eq!(loaded.columns[0], col_a);
        assert_eq!(loaded.columns[1], col_b);
        assert_eq!(loaded.columns[2], col_c);
    }

    #[test]
    fn load_structured_rejects_missing_fortran_order_key() {
        let desc = make_test_descriptor();
        let body = tofile_structured(
            &desc,
            &[
                [1.5f64.to_le_bytes()].concat(),
                [10i32.to_le_bytes()].concat(),
            ],
        )
        .unwrap();
        let header_literal = "{'descr': [('x', '<f8'), ('y', '<i4')], 'shape': (1,), }";
        let payload = make_manual_npy_payload(header_literal, &body);
        let err = load_structured(&payload).expect_err("missing fortran_order must be rejected");
        assert_eq!(err.reason_code(), "io_header_schema_invalid");
    }

    #[test]
    fn load_structured_accepts_extra_header_keys() {
        let desc = make_test_descriptor();
        let body = tofile_structured(
            &desc,
            &[
                [1.5f64.to_le_bytes()].concat(),
                [10i32.to_le_bytes()].concat(),
            ],
        )
        .unwrap();
        let header_literal = "{'descr': [('x', '<f8'), ('y', '<i4')], 'fortran_order': False, 'shape': (1,), 'extra': 1, }";
        let payload = make_manual_npy_payload(header_literal, &body);
        let data = load_structured(&payload).expect("extra keys must be allowed");
        assert_eq!(data.shape, vec![1]);
    }

    // ── Memmap (file-backed array) tests ──

    #[test]
    fn memmap_write_mode_creates_zeroed_file() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_write");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("write_test.bin");
        let shape = [4];
        let mut arr = memmap(&path, IOSupportedDType::F64, MemmapMode::Write, 0, &shape)
            .expect("memmap write");
        assert_eq!(arr.shape, vec![4]);
        assert_eq!(arr.as_bytes().len(), 32); // 4 × 8 bytes
        assert!(arr.as_bytes().iter().all(|&b| b == 0));
        assert!(arr.is_writable());
        // Write some data and flush
        let bytes = arr.as_bytes_mut().unwrap();
        bytes[..8].copy_from_slice(&1.5_f64.to_le_bytes());
        bytes[8..16].copy_from_slice(&2.5_f64.to_le_bytes());
        arr.flush().expect("flush");
        // Verify file contents
        let file_data = std::fs::read(&path).unwrap();
        assert_eq!(file_data.len(), 32);
        assert_eq!(&file_data[..8], &1.5_f64.to_le_bytes());
        assert_eq!(&file_data[8..16], &2.5_f64.to_le_bytes());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_readwrite_mode_reads_existing() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_rw");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("rw_test.bin");
        // Create a file with known data: [1.0, 2.0, 3.0]
        let mut data = Vec::new();
        data.extend_from_slice(&1.0_f64.to_le_bytes());
        data.extend_from_slice(&2.0_f64.to_le_bytes());
        data.extend_from_slice(&3.0_f64.to_le_bytes());
        std::fs::write(&path, &data).unwrap();
        let mut arr = memmap(&path, IOSupportedDType::F64, MemmapMode::ReadWrite, 0, &[3])
            .expect("memmap r+");
        let values = arr.to_f64_values().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
        // Modify second element to 99.0 and flush
        let bytes = arr.as_bytes_mut().unwrap();
        bytes[8..16].copy_from_slice(&99.0_f64.to_le_bytes());
        arr.flush().expect("flush");
        // Re-read file
        let file_data = std::fs::read(&path).unwrap();
        assert_eq!(&file_data[8..16], &99.0_f64.to_le_bytes());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_readonly_mode_prevents_mutation() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_ro");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("ro_test.bin");
        let mut data = Vec::new();
        data.extend_from_slice(&5.0_f64.to_le_bytes());
        data.extend_from_slice(&6.0_f64.to_le_bytes());
        std::fs::write(&path, &data).unwrap();
        let mut arr =
            memmap(&path, IOSupportedDType::F64, MemmapMode::ReadOnly, 0, &[2]).expect("memmap r");
        assert!(!arr.is_writable());
        assert!(arr.as_bytes_mut().is_none());
        let values = arr.to_f64_values().unwrap();
        assert_eq!(values, vec![5.0, 6.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_copy_on_write_does_not_flush_to_file() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_cow");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("cow_test.bin");
        let mut data = Vec::new();
        data.extend_from_slice(&7.0_f64.to_le_bytes());
        data.extend_from_slice(&8.0_f64.to_le_bytes());
        std::fs::write(&path, &data).unwrap();
        let mut arr = memmap(
            &path,
            IOSupportedDType::F64,
            MemmapMode::CopyOnWrite,
            0,
            &[2],
        )
        .expect("memmap c");
        assert!(arr.is_writable());
        // Mutate in-memory buffer
        let bytes = arr.as_bytes_mut().unwrap();
        bytes[..8].copy_from_slice(&999.0_f64.to_le_bytes());
        // Flush is a no-op (no backing path)
        arr.flush().expect("flush noop");
        // File should be unchanged
        let file_data = std::fs::read(&path).unwrap();
        assert_eq!(&file_data[..8], &7.0_f64.to_le_bytes());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_with_offset() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_offset");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("offset_test.bin");
        // 16 bytes header + 24 bytes data
        let mut file_data = vec![0xAA_u8; 16];
        file_data.extend_from_slice(&10.0_f64.to_le_bytes());
        file_data.extend_from_slice(&20.0_f64.to_le_bytes());
        file_data.extend_from_slice(&30.0_f64.to_le_bytes());
        std::fs::write(&path, &file_data).unwrap();
        let arr = memmap(&path, IOSupportedDType::F64, MemmapMode::ReadOnly, 16, &[3])
            .expect("memmap with offset");
        let values = arr.to_f64_values().unwrap();
        assert_eq!(values, vec![10.0, 20.0, 30.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_file_too_small_error() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_small");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("small_test.bin");
        std::fs::write(&path, [0u8; 8]).unwrap();
        let result = memmap(&path, IOSupportedDType::F64, MemmapMode::ReadOnly, 0, &[4]);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_object_dtype_rejected() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_obj");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("obj_test.bin");
        std::fs::write(&path, [0u8; 32]).unwrap();
        let result = memmap(
            &path,
            IOSupportedDType::Object,
            MemmapMode::ReadOnly,
            0,
            &[4],
        );
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_write_empty_rejected() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_empty");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("empty_test.bin");
        let result = memmap(&path, IOSupportedDType::F64, MemmapMode::Write, 0, &[0]);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_i32_dtype() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_i32");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("i32_test.bin");
        let mut data = Vec::new();
        data.extend_from_slice(&42_i32.to_le_bytes());
        data.extend_from_slice(&(-7_i32).to_le_bytes());
        std::fs::write(&path, &data).unwrap();
        let arr = memmap(&path, IOSupportedDType::I32, MemmapMode::ReadOnly, 0, &[2])
            .expect("memmap i32");
        let values = arr.to_f64_values().unwrap();
        assert_eq!(values, vec![42.0, -7.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_nbytes_and_element_count() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_nb");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("nb_test.bin");
        let arr = memmap(&path, IOSupportedDType::F64, MemmapMode::Write, 0, &[3, 2])
            .expect("memmap 3x2");
        assert_eq!(arr.element_count().unwrap(), 6);
        assert_eq!(arr.nbytes().unwrap(), 48);
        assert_eq!(arr.shape, vec![3, 2]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_npy_roundtrip() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_npy");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("npy_test.npy");
        let values = vec![1.25, 2.75, 3.5];
        let npy_bytes = save(&[3], &values, IOSupportedDType::F64).unwrap();
        std::fs::write(&path, &npy_bytes).unwrap();
        let arr = memmap_npy(&path, MemmapMode::ReadOnly).expect("memmap_npy");
        assert_eq!(arr.shape, vec![3]);
        assert_eq!(arr.dtype, IOSupportedDType::F64);
        let loaded = arr.to_f64_values().unwrap();
        assert_eq!(loaded, vec![1.25, 2.75, 3.5]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_npy_preserves_fortran_order_metadata() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_npy_fortran");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("npy_fortran.npy");
        let header = NpyHeader {
            shape: vec![2, 3],
            fortran_order: true,
            descr: IOSupportedDType::F64,
        };
        let payload = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>();
        let npy_bytes = write_npy_bytes(&header, &payload, false).expect("write fortran npy");
        std::fs::write(&path, &npy_bytes).expect("persist npy");

        let arr = memmap_npy(&path, MemmapMode::ReadOnly).expect("memmap_npy");
        assert!(
            arr.fortran_order,
            "header layout metadata should be preserved"
        );
        assert_eq!(arr.shape, vec![2, 3]);
        assert_eq!(arr.dtype, IOSupportedDType::F64);
        assert_eq!(
            arr.to_f64_values().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_npy_readwrite_modify() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_npy_rw");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("npy_rw_test.npy");
        let values = vec![10.0, 20.0];
        let npy_bytes = save(&[2], &values, IOSupportedDType::F64).unwrap();
        std::fs::write(&path, &npy_bytes).unwrap();
        let mut arr = memmap_npy(&path, MemmapMode::ReadWrite).expect("memmap_npy rw");
        assert!(arr.is_writable());
        let bytes = arr.as_bytes_mut().unwrap();
        bytes[..8].copy_from_slice(&42.0_f64.to_le_bytes());
        arr.flush().expect("flush");
        // Re-read via NPY load to verify
        let updated = std::fs::read(&path).unwrap();
        let (_, values, _) = load(&updated).unwrap();
        assert_eq!(values[0], 42.0);
        assert_eq!(values[1], 20.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_npy_rejects_write_mode() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_npy_write_mode");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("npy_write_mode_test.npy");
        let values = vec![7.0, 8.0];
        let npy_bytes = save(&[2], &values, IOSupportedDType::F64).unwrap();
        std::fs::write(&path, &npy_bytes).unwrap();

        let err = memmap_npy(&path, MemmapMode::Write).expect_err("write mode must fail closed");
        assert_eq!(err.reason_code(), "io_memmap_contract_violation");

        let persisted = std::fs::read(&path).expect("file should remain intact");
        assert_eq!(
            persisted, npy_bytes,
            "rejecting write mode must not mutate file"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_string_dtype() {
        let dir = std::env::temp_dir().join("fnp_memmap_test_str");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("str_test.bin");
        // |S5: "hello" "world"
        let mut data = Vec::new();
        data.extend_from_slice(b"hello");
        data.extend_from_slice(b"world");
        std::fs::write(&path, &data).unwrap();
        let arr = memmap(
            &path,
            IOSupportedDType::Bytes(5),
            MemmapMode::ReadOnly,
            0,
            &[2],
        )
        .expect("memmap bytes");
        let strings = arr.to_strings().unwrap();
        assert_eq!(strings, vec!["hello", "world"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memmap_mode_parse() {
        assert_eq!(MemmapMode::parse("r").unwrap(), MemmapMode::ReadOnly);
        assert_eq!(MemmapMode::parse("r+").unwrap(), MemmapMode::ReadWrite);
        assert_eq!(MemmapMode::parse("w+").unwrap(), MemmapMode::Write);
        assert_eq!(MemmapMode::parse("c").unwrap(), MemmapMode::CopyOnWrite);
        assert!(MemmapMode::parse("x").is_err());
    }

    // ── NumPy .npy format oracle tests ──────────────────────────────────
    //
    // Verify write/read roundtrip, magic bytes, header format, and that
    // NumPy-generated payloads parse correctly.

    #[test]
    fn numpy_oracle_roundtrip_f64() {
        let header = NpyHeader {
            shape: vec![4],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let npy_bytes = write_npy_bytes(&header, &payload, false).expect("write");

        let arr = read_npy_bytes(&npy_bytes, false).expect("read");
        assert_eq!(arr.header.shape, vec![4]);
        assert_eq!(arr.header.descr, IOSupportedDType::F64);
        assert!(!arr.header.fortran_order);
        let read_vals: Vec<f64> = arr
            .payload
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(read_vals, values);
    }

    #[test]
    fn numpy_oracle_roundtrip_i32() {
        let header = NpyHeader {
            shape: vec![3],
            fortran_order: false,
            descr: IOSupportedDType::I32,
        };
        let values: Vec<i32> = vec![10, 20, 30];
        let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let npy_bytes = write_npy_bytes(&header, &payload, false).expect("write");

        let arr = read_npy_bytes(&npy_bytes, false).expect("read");
        assert_eq!(arr.header.shape, vec![3]);
        assert_eq!(arr.header.descr, IOSupportedDType::I32);
        let read_vals: Vec<i32> = arr
            .payload
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(read_vals, values);
    }

    #[test]
    fn numpy_oracle_roundtrip_bool() {
        let header = NpyHeader {
            shape: vec![4],
            fortran_order: false,
            descr: IOSupportedDType::Bool,
        };
        let payload = vec![1u8, 0, 1, 0];
        let npy_bytes = write_npy_bytes(&header, &payload, false).expect("write");

        let arr = read_npy_bytes(&npy_bytes, false).expect("read");
        assert_eq!(arr.header.shape, vec![4]);
        assert_eq!(arr.header.descr, IOSupportedDType::Bool);
        assert_eq!(arr.payload, vec![1, 0, 1, 0].into());
    }

    #[test]
    fn numpy_oracle_roundtrip_f64_2d() {
        let header = NpyHeader {
            shape: vec![2, 3],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let npy_bytes = write_npy_bytes(&header, &payload, false).expect("write");

        let arr = read_npy_bytes(&npy_bytes, false).expect("read");
        assert_eq!(arr.header.shape, vec![2, 3]);
        let read_vals: Vec<f64> = arr
            .payload
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(read_vals, values);
    }

    #[test]
    fn numpy_oracle_roundtrip_u8() {
        let header = NpyHeader {
            shape: vec![5],
            fortran_order: false,
            descr: IOSupportedDType::U8,
        };
        let payload = vec![0u8, 127, 128, 200, 255];
        let npy_bytes = write_npy_bytes(&header, &payload, false).expect("write");

        let arr = read_npy_bytes(&npy_bytes, false).expect("read");
        assert_eq!(arr.header.shape, vec![5]);
        assert_eq!(arr.header.descr, IOSupportedDType::U8);
        assert_eq!(arr.payload, payload.into());
    }

    #[test]
    fn numpy_oracle_npy_magic_and_version() {
        let header = NpyHeader {
            shape: vec![2],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let payload = vec![0u8; 16]; // 2 * 8 bytes
        let npy = write_npy_bytes(&header, &payload, false).expect("write");

        // Magic: \x93NUMPY
        assert_eq!(&npy[..6], &[0x93, b'N', b'U', b'M', b'P', b'Y']);
        // Version 1.0
        assert_eq!(npy[6], 1);
        assert_eq!(npy[7], 0);
        // Header length is u16 LE at bytes 8-9
        let header_len = u16::from_le_bytes([npy[8], npy[9]]) as usize;
        // Header dict + padding ends with \n
        assert_eq!(npy[10 + header_len - 1], b'\n');
        // Total = preamble(10) + header + payload(16)
        assert_eq!(npy.len(), 10 + header_len + 16);
    }

    #[test]
    fn numpy_oracle_header_dict_format() {
        // Verify the header dict matches NumPy's format
        let header = NpyHeader {
            shape: vec![4],
            fortran_order: false,
            descr: IOSupportedDType::F64,
        };
        let payload: Vec<u8> = vec![0u8; 32];
        let npy = write_npy_bytes(&header, &payload, false).expect("write");

        let header_len = u16::from_le_bytes([npy[8], npy[9]]) as usize;
        let header_bytes = &npy[10..10 + header_len];
        let header_str = std::str::from_utf8(header_bytes).expect("utf8 header");
        // Must contain the required fields
        assert!(header_str.contains("'descr'"), "missing descr");
        assert!(
            header_str.contains("'fortran_order'"),
            "missing fortran_order"
        );
        assert!(header_str.contains("'shape'"), "missing shape");
        assert!(
            header_str.contains("False"),
            "fortran_order should be False"
        );
        // Total preamble + header must be divisible by 16 (NumPy v1 alignment)
        assert_eq!((10 + header_len) % 16, 0, "header not 16-byte aligned");
    }
}
