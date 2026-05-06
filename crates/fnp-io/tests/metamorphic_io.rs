//! Metamorphic tests for fnp-io.
//!
//! Tests invariants that must hold regardless of specific input values:
//! - Round-trip preservation (write -> read = identity)
//! - Header parsing idempotence
//! - NPZ entry ordering invariance
//! - Dtype descriptor roundtrip stability

use fnp_io::{IOSupportedDType, NpyHeader, read_npy_bytes, write_npy_bytes, read_npz_bytes, write_npz_bytes};

// ─────────────────────────────────────────────────────────────────────────────
// NPY round-trip invariance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_npy_roundtrip_f64_1d() {
    let values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let header = NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: false,
        shape: vec![100],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();

    assert_eq!(loaded.header.shape, header.shape, "shape should roundtrip");
    assert_eq!(loaded.header.descr, header.descr, "descr should roundtrip");
    assert_eq!(loaded.header.fortran_order, header.fortran_order);

    let loaded_values: &[f64] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(loaded_values, &values[..], "values should roundtrip exactly");
}

#[test]
fn mr_npy_roundtrip_f32_2d() {
    let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let header = NpyHeader {
        descr: IOSupportedDType::F32,
        fortran_order: false,
        shape: vec![4, 6],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();

    assert_eq!(loaded.header.shape, vec![4, 6]);
    let loaded_values: &[f32] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(loaded_values, &values[..]);
}

#[test]
fn mr_npy_roundtrip_i32_3d() {
    let values: Vec<i32> = (0..60).collect();
    let header = NpyHeader {
        descr: IOSupportedDType::I32,
        fortran_order: false,
        shape: vec![3, 4, 5],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();

    assert_eq!(loaded.header.shape, vec![3, 4, 5]);
    let loaded_values: &[i32] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(loaded_values, &values[..]);
}

#[test]
fn mr_npy_roundtrip_fortran_order() {
    let values: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let header = NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: true,
        shape: vec![3, 4],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();

    assert!(loaded.header.fortran_order, "fortran_order should roundtrip as true");
    assert_eq!(loaded.header.shape, vec![3, 4]);
}

#[test]
fn mr_npy_roundtrip_empty_array() {
    let values: Vec<f64> = vec![];
    let header = NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: false,
        shape: vec![0],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();

    assert_eq!(loaded.header.shape, vec![0], "empty shape should roundtrip");
    assert!(loaded.payload.is_empty(), "empty payload should roundtrip");
}

#[test]
fn mr_npy_roundtrip_scalar() {
    let values: Vec<f64> = vec![42.0];
    let header = NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: false,
        shape: vec![],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();

    assert!(loaded.header.shape.is_empty(), "scalar shape should roundtrip");
    let loaded_values: &[f64] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(loaded_values[0], 42.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dtype descriptor roundtrip idempotence
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_dtype_descriptor_roundtrip_all_types() {
    let dtypes = [
        IOSupportedDType::F64,
        IOSupportedDType::F32,
        IOSupportedDType::I64,
        IOSupportedDType::I32,
        IOSupportedDType::I16,
        IOSupportedDType::I8,
        IOSupportedDType::U64,
        IOSupportedDType::U32,
        IOSupportedDType::U16,
        IOSupportedDType::U8,
        IOSupportedDType::Bool,
        IOSupportedDType::Complex64,
        IOSupportedDType::Complex128,
    ];

    for dtype in dtypes {
        let descr_str = dtype.descr();
        let parsed = IOSupportedDType::decode(&descr_str);
        assert!(
            parsed.is_ok(),
            "dtype {:?} should roundtrip via descriptor '{}': {:?}",
            dtype,
            descr_str,
            parsed.err()
        );
        assert_eq!(
            parsed.unwrap(),
            dtype,
            "parsed dtype should equal original for '{}'",
            descr_str
        );
    }
}

#[test]
fn mr_dtype_descr_parse_idempotence() {
    let dtypes = [
        IOSupportedDType::F64,
        IOSupportedDType::F32,
        IOSupportedDType::I32,
        IOSupportedDType::Bool,
    ];

    for dtype in dtypes {
        let descr1 = dtype.descr();
        let parsed = IOSupportedDType::decode(&descr1).unwrap();
        let descr2 = parsed.descr();
        assert_eq!(
            descr1, descr2,
            "descr should be idempotent for {:?}: {} vs {}",
            dtype, descr1, descr2
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NPZ roundtrip invariance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_npz_roundtrip_single_entry() {
    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let header = NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: false,
        shape: vec![4],
    };
    let data = bytemuck::cast_slice(&values);
    let bytes = write_npz_bytes(&[("arr_0", &header, data)]).unwrap();
    let entries = read_npz_bytes(&bytes, false).unwrap();

    assert_eq!(entries.len(), 1, "single entry should roundtrip");
    // Name may or may not have .npy extension - accept either
    assert!(
        entries[0].name == "arr_0" || entries[0].name == "arr_0.npy",
        "entry name should be arr_0 or arr_0.npy, got: {}",
        entries[0].name
    );
    assert_eq!(entries[0].array.header.shape, vec![4]);
}

#[test]
fn mr_npz_roundtrip_multiple_entries() {
    let values1: Vec<f64> = vec![1.0, 2.0];
    let values2: Vec<i32> = vec![10, 20, 30];

    let header1 = NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: false,
        shape: vec![2],
    };
    let header2 = NpyHeader {
        descr: IOSupportedDType::I32,
        fortran_order: false,
        shape: vec![3],
    };

    let entries_in = [
        ("x", &header1, bytemuck::cast_slice(&values1)),
        ("y", &header2, bytemuck::cast_slice(&values2)),
    ];
    let bytes = write_npz_bytes(&entries_in).unwrap();
    let entries_out = read_npz_bytes(&bytes, false).unwrap();

    assert_eq!(entries_out.len(), 2, "both entries should roundtrip");

    // Find entries by base name (with or without .npy extension)
    let x_entry = entries_out.iter().find(|e| e.name == "x" || e.name == "x.npy").unwrap();
    let y_entry = entries_out.iter().find(|e| e.name == "y" || e.name == "y.npy").unwrap();

    assert_eq!(x_entry.array.header.shape, vec![2]);
    assert_eq!(y_entry.array.header.shape, vec![3]);
}

// ─────────────────────────────────────────────────────────────────────────────
// NPZ entry order independence
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_npz_entry_order_independence() {
    let values_a: Vec<f64> = vec![1.0];
    let values_b: Vec<f64> = vec![2.0];
    let values_c: Vec<f64> = vec![3.0];

    let header = NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: false,
        shape: vec![1],
    };

    let order1 = [
        ("a", &header, bytemuck::cast_slice(&values_a)),
        ("b", &header, bytemuck::cast_slice(&values_b)),
        ("c", &header, bytemuck::cast_slice(&values_c)),
    ];
    let order2 = [
        ("c", &header, bytemuck::cast_slice(&values_c)),
        ("a", &header, bytemuck::cast_slice(&values_a)),
        ("b", &header, bytemuck::cast_slice(&values_b)),
    ];

    let bytes1 = write_npz_bytes(&order1).unwrap();
    let bytes2 = write_npz_bytes(&order2).unwrap();

    let entries1 = read_npz_bytes(&bytes1, false).unwrap();
    let entries2 = read_npz_bytes(&bytes2, false).unwrap();

    assert_eq!(entries1.len(), entries2.len(), "entry count should match");

    // Match by base name (with or without .npy extension)
    for base in ["a", "b", "c"] {
        let e1 = entries1.iter().find(|e| e.name == base || e.name == format!("{}.npy", base)).unwrap();
        let e2 = entries2.iter().find(|e| e.name == base || e.name == format!("{}.npy", base)).unwrap();

        assert_eq!(
            e1.array.payload.as_ref(),
            e2.array.payload.as_ref(),
            "payload for {} should be independent of entry order",
            base
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Integer type roundtrips
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_npy_roundtrip_i64() {
    let values: Vec<i64> = vec![i64::MIN, -1, 0, 1, i64::MAX];
    let header = NpyHeader {
        descr: IOSupportedDType::I64,
        fortran_order: false,
        shape: vec![5],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();

    let loaded_values: &[i64] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(loaded_values, &values[..], "i64 extreme values should roundtrip");
}

#[test]
fn mr_npy_roundtrip_u64() {
    let values: Vec<u64> = vec![0, 1, u64::MAX / 2, u64::MAX];
    let header = NpyHeader {
        descr: IOSupportedDType::U64,
        fortran_order: false,
        shape: vec![4],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();

    let loaded_values: &[u64] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(loaded_values, &values[..], "u64 values should roundtrip");
}

#[test]
fn mr_npy_roundtrip_bool() {
    let values: Vec<u8> = vec![0, 1, 0, 1, 1, 0];
    let header = NpyHeader {
        descr: IOSupportedDType::Bool,
        fortran_order: false,
        shape: vec![6],
    };
    let bytes = write_npy_bytes(&header, &values, false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();

    assert_eq!(loaded.payload.as_ref(), &values[..], "bool values should roundtrip");
}

// ─────────────────────────────────────────────────────────────────────────────
// Complex number roundtrips
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_npy_roundtrip_complex128() {
    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let header = NpyHeader {
        descr: IOSupportedDType::Complex128,
        fortran_order: false,
        shape: vec![2],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();

    assert_eq!(loaded.header.descr, IOSupportedDType::Complex128);
    let loaded_values: &[f64] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(loaded_values, &values[..], "complex128 should roundtrip");
}

#[test]
fn mr_npy_roundtrip_complex64() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let header = NpyHeader {
        descr: IOSupportedDType::Complex64,
        fortran_order: false,
        shape: vec![2],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();

    assert_eq!(loaded.header.descr, IOSupportedDType::Complex64);
    let loaded_values: &[f32] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(loaded_values, &values[..], "complex64 should roundtrip");
}

// ─────────────────────────────────────────────────────────────────────────────
// Header item size consistency
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_dtype_item_size_consistent() {
    let cases = [
        (IOSupportedDType::F64, 8),
        (IOSupportedDType::F32, 4),
        (IOSupportedDType::I64, 8),
        (IOSupportedDType::I32, 4),
        (IOSupportedDType::I16, 2),
        (IOSupportedDType::I8, 1),
        (IOSupportedDType::U64, 8),
        (IOSupportedDType::U32, 4),
        (IOSupportedDType::U16, 2),
        (IOSupportedDType::U8, 1),
        (IOSupportedDType::Bool, 1),
        (IOSupportedDType::Complex128, 16),
        (IOSupportedDType::Complex64, 8),
    ];

    for (dtype, expected_size) in cases {
        let size = dtype.item_size().unwrap();
        assert_eq!(
            size, expected_size,
            "{:?} item_size should be {}",
            dtype, expected_size
        );
    }
}
