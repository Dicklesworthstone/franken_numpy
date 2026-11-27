//! Lossless transfer between numeric NPY payloads and the compute crates' typed storage.

use fnp_dtype::{ArrayStorage, f16};

use super::{
    IOError, IOSupportedDType, MAX_NPY_NDIM, NpyArrayBytes, NpyHeader, NpzCompression,
    NpzNameDecoding, NpzRangePolicy, element_count, npy_v1_encoded_len_for_capacity, parse_npy_bytes_borrowed,
    row_major_values, validate_npz_archive_budget, validate_public_npz_entry_names,
    validate_read_payload, walk_npz_members, write_npy_bytes, write_npz_bytes_with_compression,
};

/// Load a numeric NPY array into native typed storage in logical C order.
/// Byte order is decoded independently for each real or imaginary component.
pub fn load_storage(data: &[u8]) -> Result<(Vec<usize>, ArrayStorage), IOError> {
    let npy = parse_npy_bytes_borrowed(data, false)?;
    let storage = decode_numeric(&npy.header, npy.payload)?;
    Ok((npy.header.shape, storage))
}

impl NpyArrayBytes {
    /// Decode a numeric NPY payload, including an entry returned by `read_npz_bytes`.
    pub fn to_storage(&self) -> Result<ArrayStorage, IOError> {
        validate_read_payload(&self.header.shape, self.payload.len(), self.header.descr)?;
        decode_numeric(&self.header, &self.payload)
    }
}

fn decode_pairs<T, const WIDTH: usize>(
    payload: &[u8],
    component: impl Fn([u8; WIDTH]) -> T,
) -> Vec<(T, T)> {
    // The header footprint was checked before decoding, so every pair is complete.
    payload
        .as_chunks::<WIDTH>()
        .0
        .chunks_exact(2)
        .map(|pair| (component(pair[0]), component(pair[1])))
        .collect()
}

fn decode_numeric(header: &NpyHeader, payload: &[u8]) -> Result<ArrayStorage, IOError> {
    use IOSupportedDType as Io;

    macro_rules! scalars {
        ($variant:ident, $read:path) => {
            ArrayStorage::$variant(row_major_values(
                payload.as_chunks().0.iter().copied().map($read).collect(),
                header,
            ))
        };
    }

    Ok(match header.descr {
        Io::Bool => ArrayStorage::Bool(row_major_values(
            payload.iter().map(|&byte| byte != 0).collect(),
            header,
        )),
        Io::I8 => scalars!(I8, i8::from_le_bytes),
        Io::U8 => ArrayStorage::U8(row_major_values(payload.to_vec(), header)),
        Io::I16 => scalars!(I16, i16::from_le_bytes),
        Io::I16Be => scalars!(I16, i16::from_be_bytes),
        Io::I32 => scalars!(I32, i32::from_le_bytes),
        Io::I32Be => scalars!(I32, i32::from_be_bytes),
        Io::I64 => scalars!(I64, i64::from_le_bytes),
        Io::I64Be => scalars!(I64, i64::from_be_bytes),
        Io::U16 => scalars!(U16, u16::from_le_bytes),
        Io::U16Be => scalars!(U16, u16::from_be_bytes),
        Io::U32 => scalars!(U32, u32::from_le_bytes),
        Io::U32Be => scalars!(U32, u32::from_be_bytes),
        Io::U64 => scalars!(U64, u64::from_le_bytes),
        Io::U64Be => scalars!(U64, u64::from_be_bytes),
        Io::F16 => scalars!(F16, f16::from_le_bytes),
        Io::F16Be => scalars!(F16, f16::from_be_bytes),
        Io::F32 => scalars!(F32, f32::from_le_bytes),
        Io::F32Be => scalars!(F32, f32::from_be_bytes),
        Io::F64 => scalars!(F64, f64::from_le_bytes),
        Io::F64Be => scalars!(F64, f64::from_be_bytes),
        Io::Complex64 => ArrayStorage::Complex64(row_major_values(
            decode_pairs(payload, f32::from_le_bytes),
            header,
        )),
        Io::Complex64Be => ArrayStorage::Complex64(row_major_values(
            decode_pairs(payload, f32::from_be_bytes),
            header,
        )),
        Io::Complex128 => ArrayStorage::Complex128(row_major_values(
            decode_pairs(payload, f64::from_le_bytes),
            header,
        )),
        Io::Complex128Be => ArrayStorage::Complex128(row_major_values(
            decode_pairs(payload, f64::from_be_bytes),
            header,
        )),
        _ => return Err(IOError::DTypeDescriptorInvalid),
    })
}

/// Save numeric compute storage as a C-order NPY array. Multi-byte components
/// are written little-endian. The storage variant determines the file dtype.
pub fn save_storage(shape: &[usize], storage: &ArrayStorage) -> Result<Vec<u8>, IOError> {
    let (header, payload) = encode_numeric(shape, storage)?;
    write_npy_bytes(&header, &payload, false)
}

fn encode_numeric(shape: &[usize], storage: &ArrayStorage) -> Result<(NpyHeader, Vec<u8>), IOError> {
    use IOSupportedDType as Io;

    if shape.len() > MAX_NPY_NDIM {
        return Err(IOError::WriteContractViolation(
            "shape rank exceeds packet validation budget",
        ));
    }
    if element_count(shape)? != storage.len() {
        return Err(IOError::WriteContractViolation(
            "value_count does not match shape element count",
        ));
    }

    macro_rules! scalars {
        ($values:expr, $ty:ident) => {
            $values.iter().copied().flat_map($ty::to_le_bytes).collect()
        };
    }

    let (descr, payload) = match storage {
        ArrayStorage::Bool(v) => (Io::Bool, v.iter().copied().map(u8::from).collect()),
        ArrayStorage::I8(v) => (Io::I8, scalars!(v, i8)),
        ArrayStorage::U8(v) => (Io::U8, v.clone()),
        ArrayStorage::I16(v) => (Io::I16, scalars!(v, i16)),
        ArrayStorage::I32(v) => (Io::I32, scalars!(v, i32)),
        ArrayStorage::I64(v) => (Io::I64, scalars!(v, i64)),
        ArrayStorage::U16(v) => (Io::U16, scalars!(v, u16)),
        ArrayStorage::U32(v) => (Io::U32, scalars!(v, u32)),
        ArrayStorage::U64(v) => (Io::U64, scalars!(v, u64)),
        ArrayStorage::F16(v) => (Io::F16, scalars!(v, f16)),
        ArrayStorage::F32(v) => (Io::F32, scalars!(v, f32)),
        ArrayStorage::F64(v) => (Io::F64, scalars!(v, f64)),
        ArrayStorage::Complex64(v) => (
            Io::Complex64,
            v.iter()
                .flat_map(|&(real, imag)| [real, imag])
                .flat_map(f32::to_le_bytes)
                .collect(),
        ),
        ArrayStorage::Complex128(v) => (
            Io::Complex128,
            v.iter()
                .flat_map(|&(real, imag)| [real, imag])
                .flat_map(f64::to_le_bytes)
                .collect(),
        ),
        _ => return Err(IOError::DTypeDescriptorInvalid),
    };
    let header = NpyHeader {
        shape: shape.to_vec(),
        fortran_order: false,
        descr,
    };
    Ok((header, payload))
}

/// A numeric NPZ member with its public name, logical shape and typed values.
pub type NpzStorageEntry = (String, Vec<usize>, ArrayStorage);

/// Load all numeric members of a stored or deflated archive. Each member
/// retains its own dtype and is decoded directly from its validated NPY bytes.
/// Member names follow ZIP semantics: UTF-8 when flagged, otherwise CP437.
pub fn load_npz_storage(data: &[u8]) -> Result<Vec<NpzStorageEntry>, IOError> {
    let mut entries = Vec::new();
    walk_npz_members(
        data,
        NpzRangePolicy::Adaptive,
        NpzNameDecoding::ZipStandard,
        |name, bytes| {
            let (shape, storage) = load_storage(bytes)?;
            let name = name.strip_suffix(".npy").unwrap_or(&name).to_owned();
            entries.push((name, shape, storage));
            Ok(())
        },
    )?;
    Ok(entries)
}

/// Save heterogeneous numeric arrays to one NPZ archive. Each storage variant
/// chooses its own NPY dtype, with the same representation as `save_storage`.
/// Keys are literal, including any trailing `.npy`, as in `numpy.savez`.
pub fn savez_storage(
    entries: &[(&str, &[usize], &ArrayStorage)],
    compression: NpzCompression,
) -> Result<Vec<u8>, IOError> {
    validate_public_npz_entry_names(entries.iter().map(|entry| format!("{}.npy", entry.0)))?;
    let mut members = Vec::with_capacity(entries.len());
    let mut total = 0usize;
    for &(name, shape, storage) in entries {
        let overflow = || IOError::NpzArchiveContractViolation("archive byte count overflowed");
        let payload_len = storage
            .len()
            .checked_mul(storage.dtype().item_size())
            .ok_or_else(overflow)?;
        validate_npz_archive_budget(
            entries.len(),
            total.checked_add(payload_len).ok_or_else(overflow)?,
            0,
        )?;
        let (header, payload) = encode_numeric(shape, storage)?;
        total = npy_v1_encoded_len_for_capacity(&header, payload.len())
            .and_then(|size| total.checked_add(size))
            .ok_or_else(overflow)?;
        validate_npz_archive_budget(entries.len(), total, 0)?;
        members.push((format!("{name}.npy"), header, payload));
    }
    let raw: Vec<_> = members
        .iter()
        .map(|(name, header, payload)| (name.as_str(), header, payload.as_slice()))
        .collect();
    write_npz_bytes_with_compression(&raw, compression)
}
