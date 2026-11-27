use std::sync::Arc;

use fnp_dtype::ArrayStorage;
use fnp_io::{
    IOError, IOSupportedDType, NpyArrayBytes, NpyHeader, NpzCompression, load_npz_storage,
    load_storage, read_npy_bytes, read_npz_bytes, save_storage, savez_storage, write_npy_bytes,
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

#[test]
fn numeric_archives_preserve_member_dtypes_names_and_order() {
    let ids = ArrayStorage::U64(vec![u64::MAX, (1_u64 << 53) + 1]);
    let signal = ArrayStorage::Complex128(vec![(1.0, -2.5)]);
    let flags = ArrayStorage::Bool(Vec::new());
    let members: &[(&str, &[usize], &ArrayStorage)] = &[
        ("ids", &[2], &ids),
        ("signal.npy", &[], &signal),
        ("flags", &[0, 4], &flags),
    ];
    for compression in [NpzCompression::Store, NpzCompression::Deflate] {
        let bytes = savez_storage(members, compression).unwrap();
        assert_eq!(
            load_npz_storage(&bytes).unwrap(),
            [
                ("ids".to_owned(), vec![2], ids.clone()),
                ("signal.npy".to_owned(), vec![], signal.clone()),
                ("flags".to_owned(), vec![0, 4], flags.clone()),
            ]
        );
        let raw = read_npz_bytes(&bytes, false).unwrap();
        assert_eq!(raw[0].array.header.descr, IOSupportedDType::U64);
        assert_eq!(raw[1].array.header.descr, IOSupportedDType::Complex128);
        assert_eq!(raw[2].array.to_storage().unwrap(), flags);
        let empty = savez_storage(&[], compression).unwrap();
        assert!(load_npz_storage(&empty).unwrap().is_empty());
        assert!(savez_storage(&[("x", &[2], &ids), ("x", &[2], &ids)], compression).is_err());
        assert!(savez_storage(&[("../x", &[2], &ids)], compression).is_err());
        assert!(savez_storage(&[("ids", &[1], &ids)], compression).is_err());
    }
}

#[test]
fn zip_member_names_decode_cp437_and_require_valid_flagged_utf8() {
    let value = ArrayStorage::U8(vec![1]);
    let mut bytes = savez_storage(&[("caf!", &[], &value)], NpzCompression::Store).unwrap();
    let central = u32::from_le_bytes(bytes[bytes.len() - 6..bytes.len() - 2].try_into().unwrap())
        as usize;
    // Replace the ASCII placeholder in both headers with CP437's e-acute.
    bytes[30 + 3] = 0x82;
    bytes[central + 46 + 3] = 0x82;
    assert_eq!(load_npz_storage(&bytes).unwrap()[0].0, "café");
    bytes[6..8].copy_from_slice(&(1_u16 << 11).to_le_bytes());
    bytes[central + 8..central + 10].copy_from_slice(&(1_u16 << 11).to_le_bytes());
    assert!(load_npz_storage(&bytes).is_err());
    assert!(read_npz_bytes(&bytes, false).is_err());

    // Older FrankenNumPy writers stored UTF-8 names but left both flags clear.
    // With these bytes the two encodings are genuinely ambiguous.
    let mut old = savez_storage(&[("café", &[], &value)], NpzCompression::Store).unwrap();
    let central = u32::from_le_bytes(old[old.len() - 6..old.len() - 2].try_into().unwrap()) as usize;
    old[6..8].fill(0);
    old[central + 8..central + 10].fill(0);
    assert_eq!(read_npz_bytes(&old, false).unwrap()[0].name, "café");
    assert_eq!(fnp_io::load_npz(&old, false).unwrap()[0].0, "café");
    assert_eq!(load_npz_storage(&old).unwrap()[0].0, "caf├⌐");
}

#[test]
fn numpy_and_rust_exchange_npz_archives_with_literal_unicode_keys() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let script = r#"
import io, sys, zipfile
try:
    import numpy as np
except ImportError:
    sys.exit(77)
keys = ["x", "x.npy", "café_μ"]
data = sys.stdin.buffer.read()
with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
    assert arrays.files == keys, arrays.files
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    assert archive.namelist() == [k + ".npy" for k in keys]
    for i, key in enumerate(keys):
        name = key + ".npy"
        assert bool(archive.getinfo(name).flag_bits & (1 << 11)) == (not key.isascii())
        with archive.open(name) as f:
            assert np.load(f, allow_pickle=False).item() == 2**63 + i
out = io.BytesIO()
save = np.savez if sys.argv[1] == "store" else np.savez_compressed
save(out, **{key: np.uint64(2**63 + i) for i, key in enumerate(keys)})
sys.stdout.buffer.write(out.getvalue())
"#;
    let keys = ["x", "x.npy", "café_μ"];
    let values: Vec<_> = (0..3)
        .map(|i| ArrayStorage::U64(vec![(1_u64 << 63) + i]))
        .collect();
    let entries: Vec<_> = keys
        .iter()
        .zip(&values)
        .map(|(&key, value)| (key, &[][..], value))
        .collect();
    for (compression, argument) in [
        (NpzCompression::Store, "store"),
        (NpzCompression::Deflate, "deflate"),
    ] {
        let bytes = savez_storage(&entries, compression).unwrap();
        let mut child = Command::new(&python)
            .args(["-c", script, argument])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(&bytes).unwrap();
        let result = child.wait_with_output().unwrap();
        if result.status.code() == Some(77) {
            return;
        }
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let loaded = load_npz_storage(&result.stdout).unwrap();
        assert_eq!(loaded.len(), keys.len());
        for ((name, shape, storage), (&key, expected)) in loaded.iter().zip(keys.iter().zip(&values))
        {
            assert_eq!(name, key);
            assert!(shape.is_empty());
            assert_eq!(storage, expected);
        }
    }
}
