use std::sync::Arc;

use fnp_dtype::ArrayStorage;
use fnp_io::{
    IOError, IOSupportedDType, NpyArrayBytes, NpyHeader, load_storage, read_npy_bytes, save_storage,
    write_npy_bytes,
};

#[test]
fn typed_files_preserve_integer_extremes_and_complex_component_bits() {
    let integers = vec![i64::MIN, -(1_i64 << 53) - 1, (1_i64 << 53) + 1, i64::MAX];
    let storage = ArrayStorage::I64(integers.clone());
    let bytes = save_storage(&[2, 2], &storage).unwrap();
    let decoded = read_npy_bytes(&bytes, false).unwrap();
    assert_eq!(decoded.header.descr, IOSupportedDType::I64);
    let expected: Vec<_> = integers.into_iter().flat_map(i64::to_le_bytes).collect();
    assert_eq!(decoded.payload.as_ref(), expected.as_slice());
    assert_eq!(load_storage(&bytes).unwrap(), (vec![2, 2], storage));

    // A noncanonical NaN, negative zero, a subnormal and an infinity.
    let bits = [0x7fa0_4321_u32, 0x8000_0000, 1, 0xff80_0000];
    let big_endian: Vec<_> = bits.iter().copied().flat_map(u32::to_be_bytes).collect();
    let header = NpyHeader {
        shape: vec![2],
        descr: IOSupportedDType::Complex64Be,
        fortran_order: false,
    };
    let bytes = write_npy_bytes(&header, &big_endian, false).unwrap();
    let (shape, values) = load_storage(&bytes).unwrap();
    let little_endian = save_storage(&shape, &values).unwrap();
    let decoded = read_npy_bytes(&little_endian, false).unwrap();
    assert_eq!(decoded.header.descr, IOSupportedDType::Complex64);
    let expected: Vec<_> = bits.into_iter().flat_map(u32::to_le_bytes).collect();
    assert_eq!(decoded.payload.as_ref(), expected.as_slice());
}

#[test]
fn typed_files_validate_shape_and_reorder_complete_fortran_elements() {
    let wide = u64::MAX - 20;
    let header = NpyHeader {
        shape: vec![2, 3],
        descr: IOSupportedDType::U64Be,
        fortran_order: true,
    };
    let physical: Vec<_> = [0, 3, 1, 4, 2, 5]
        .into_iter()
        .flat_map(|n| (wide + n).to_be_bytes())
        .collect();
    let parsed = NpyArrayBytes {
        header,
        version: (1, 0),
        payload: physical.into(),
    };
    assert_eq!(
        parsed.to_storage().unwrap(),
        ArrayStorage::U64((wide..wide + 6).collect())
    );
    let truncated = NpyArrayBytes {
        payload: Arc::from(&parsed.payload[..47]),
        ..parsed
    };
    assert!(matches!(
        truncated.to_storage(),
        Err(IOError::ReadPayloadIncomplete(_))
    ));

    for (shape, storage) in [
        (vec![], ArrayStorage::Bool(vec![true])),
        (vec![2, 0, 3], ArrayStorage::U64(Vec::new())),
    ] {
        let bytes = save_storage(&shape, &storage).unwrap();
        assert_eq!(load_storage(&bytes).unwrap(), (shape, storage));
    }
    assert!(save_storage(&[1; 33], &ArrayStorage::Bool(vec![true])).is_err());
    assert!(save_storage(&[2, 2], &ArrayStorage::F32(vec![1.0])).is_err());
    assert!(save_storage(&[1], &ArrayStorage::String(vec!["abc".into()])).is_err());
}

#[test]
fn numpy_numeric_orders_and_byte_orders_round_trip_in_logical_order() {
    use std::process::Command;
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let script = r#"
import io, sys
try:
    import numpy as np
except ImportError:
    sys.exit(77)
def frame(data):
    sys.stdout.buffer.write(len(data).to_bytes(4, 'little') + data)
for code in ('?', 'i1', 'u1', 'i2', 'i4', 'i8', 'u2', 'u4', 'u8',
             'f2', 'f4', 'f8', 'c8', 'c16'):
    for order in ('C', 'F'):
        for endian in ('<', '>'):
            dt = np.dtype(endian + code)
            x = np.arange(24).reshape(2, 3, 4)
            if dt.kind == 'b':
                x = x % 3 == 0
            elif dt.kind == 'c':
                x = x - 5 + 1j * (2 * x - 17)
            elif code == 'u8':
                x = np.uint64(2**64 - 30) + x.astype(np.uint64)
            elif code == 'i8':
                x = np.int64(-(2**63)) + x
            a = np.array(x, dtype=dt, order=order)
            f = io.BytesIO()
            np.save(f, a, allow_pickle=False)
            frame(f.getvalue())
            frame(a.astype(dt.newbyteorder('<')).tobytes(order='C'))
"#;
    let result = Command::new(python).args(["-c", script]).output().unwrap();
    if result.status.code() == Some(77) {
        return;
    }
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    fn frame<'a>(bytes: &mut &'a [u8]) -> &'a [u8] {
        let (len, rest) = bytes.split_at(4);
        let (data, rest) = rest.split_at(u32::from_le_bytes(len.try_into().unwrap()) as usize);
        *bytes = rest;
        data
    }
    let mut bytes = result.stdout.as_slice();
    for _ in 0..56 {
        let source = frame(&mut bytes);
        let expected = frame(&mut bytes);
        let (shape, storage) = load_storage(source).unwrap();
        assert_eq!(shape, [2, 3, 4]);
        let saved = save_storage(&shape, &storage).unwrap();
        let actual = read_npy_bytes(&saved, false).unwrap();
        let original = read_npy_bytes(source, false).unwrap();
        assert_eq!(
            actual.header.descr.descr(),
            original.header.descr.descr().replace('>', "<")
        );
        assert!(!actual.header.fortran_order);
        assert_eq!(actual.payload.as_ref(), expected);
    }
    assert!(bytes.is_empty());
}
