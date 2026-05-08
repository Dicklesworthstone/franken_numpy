//! Metamorphic tests for fnp-io.
//!
//! Tests invariants that must hold regardless of specific input values:
//! - Round-trip preservation (write -> read = identity)
//! - Header parsing idempotence
//! - NPZ entry ordering invariance
//! - Dtype descriptor roundtrip stability

use fnp_io::{
    IOSupportedDType, NpyHeader, read_npy_bytes, read_npz_bytes, write_npy_bytes, write_npz_bytes,
};

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
    assert_eq!(
        loaded_values,
        &values[..],
        "values should roundtrip exactly"
    );
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

    assert!(
        loaded.header.fortran_order,
        "fortran_order should roundtrip as true"
    );
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

    assert!(
        loaded.header.shape.is_empty(),
        "scalar shape should roundtrip"
    );
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
    let x_entry = entries_out
        .iter()
        .find(|e| e.name == "x" || e.name == "x.npy")
        .unwrap();
    let y_entry = entries_out
        .iter()
        .find(|e| e.name == "y" || e.name == "y.npy")
        .unwrap();

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
        let e1 = entries1
            .iter()
            .find(|e| e.name == base || e.name == format!("{}.npy", base))
            .unwrap();
        let e2 = entries2
            .iter()
            .find(|e| e.name == base || e.name == format!("{}.npy", base))
            .unwrap();

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
    assert_eq!(
        loaded_values,
        &values[..],
        "i64 extreme values should roundtrip"
    );
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

    assert_eq!(
        loaded.payload.as_ref(),
        &values[..],
        "bool values should roundtrip"
    );
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

// ─────────────────────────────────────────────────────────────────────────────
// MR: Double roundtrip idempotence (write -> read -> write -> read = identity)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_double_roundtrip_idempotent() {
    let values: Vec<f64> = (0..50).map(|i| i as f64 * 0.01).collect();
    let header = NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: false,
        shape: vec![10, 5],
    };

    // First roundtrip
    let bytes1 = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded1 = read_npy_bytes(&bytes1, false).unwrap();

    // Second roundtrip
    let bytes2 = write_npy_bytes(&loaded1.header, &loaded1.payload, false).unwrap();
    let loaded2 = read_npy_bytes(&bytes2, false).unwrap();

    // Payloads should be identical
    assert_eq!(
        loaded1.payload.as_ref(),
        loaded2.payload.as_ref(),
        "double roundtrip should produce identical payloads"
    );
    assert_eq!(
        loaded1.header.shape, loaded2.header.shape,
        "double roundtrip should preserve shape"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR: Payload size = product(shape) * item_size
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_payload_size_equals_shape_times_itemsize() {
    let shapes: &[&[usize]] = &[&[10], &[4, 5], &[2, 3, 4], &[1, 1, 1, 1], &[100]];

    for dtype in [
        IOSupportedDType::F64,
        IOSupportedDType::F32,
        IOSupportedDType::I32,
    ] {
        let item_size = dtype.item_size().unwrap();

        for &shape in shapes {
            let n_elements: usize = shape.iter().product();
            let expected_payload_size = n_elements * item_size;

            let values: Vec<u8> = vec![0u8; expected_payload_size];
            let header = NpyHeader {
                descr: dtype,
                fortran_order: false,
                shape: shape.to_vec(),
            };

            let bytes = write_npy_bytes(&header, &values, false).unwrap();
            let loaded = read_npy_bytes(&bytes, false).unwrap();

            assert_eq!(
                loaded.payload.len(),
                expected_payload_size,
                "payload size for shape {:?} dtype {:?} should be {}",
                shape,
                dtype,
                expected_payload_size
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR: Shape interpretation invariance (same bytes, different shapes)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_reshape_preserves_raw_bytes() {
    let values: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&values).to_vec();

    // Write as different shapes (same total elements)
    let shapes = vec![
        vec![24],
        vec![2, 12],
        vec![3, 8],
        vec![4, 6],
        vec![2, 3, 4],
        vec![2, 2, 6],
    ];

    for shape in &shapes {
        let header = NpyHeader {
            descr: IOSupportedDType::F64,
            fortran_order: false,
            shape: shape.clone(),
        };

        let bytes = write_npy_bytes(&header, &raw_bytes, false).unwrap();
        let loaded = read_npy_bytes(&bytes, false).unwrap();

        assert_eq!(
            loaded.payload.as_ref(),
            &raw_bytes[..],
            "raw bytes should be preserved regardless of shape {:?}",
            shape
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR: NPZ equivalence (all entries accessible regardless of write order)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_npz_all_entries_recoverable() {
    let n_entries = 5;
    let mut entries_data: Vec<(String, NpyHeader, Vec<u8>)> = Vec::new();

    for i in 0..n_entries {
        let values: Vec<f64> = (0..10).map(|j| (i * 10 + j) as f64).collect();
        let header = NpyHeader {
            descr: IOSupportedDType::F64,
            fortran_order: false,
            shape: vec![10],
        };
        entries_data.push((
            format!("arr_{}", i),
            header,
            bytemuck::cast_slice(&values).to_vec(),
        ));
    }

    let entries_refs: Vec<(&str, &NpyHeader, &[u8])> = entries_data
        .iter()
        .map(|(n, h, d)| (n.as_str(), h, d.as_slice()))
        .collect();

    let bytes = write_npz_bytes(&entries_refs).unwrap();
    let loaded = read_npz_bytes(&bytes, false).unwrap();

    assert_eq!(
        loaded.len(),
        n_entries,
        "all {} entries should be recoverable",
        n_entries
    );

    // Verify each entry's data
    for i in 0..n_entries {
        let name = format!("arr_{}", i);
        let entry = loaded
            .iter()
            .find(|e| e.name == name || e.name == format!("{}.npy", name))
            .unwrap_or_else(|| panic!("entry {name} should exist"));

        let expected_values: Vec<f64> = (0..10).map(|j| (i * 10 + j) as f64).collect();
        let loaded_values: &[f64] = bytemuck::cast_slice(&entry.array.payload);
        assert_eq!(
            loaded_values,
            &expected_values[..],
            "entry {} values should match",
            name
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR: Header invariance across repeated parsing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_header_parse_deterministic() {
    let values: Vec<f64> = vec![1.0, 2.0, 3.0];
    let header = NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: false,
        shape: vec![3],
    };

    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();

    // Parse multiple times
    let loaded1 = read_npy_bytes(&bytes, false).unwrap();
    let loaded2 = read_npy_bytes(&bytes, false).unwrap();
    let loaded3 = read_npy_bytes(&bytes, false).unwrap();

    assert_eq!(loaded1.header.descr, loaded2.header.descr);
    assert_eq!(loaded2.header.descr, loaded3.header.descr);
    assert_eq!(loaded1.header.shape, loaded2.header.shape);
    assert_eq!(loaded2.header.shape, loaded3.header.shape);
    assert_eq!(loaded1.header.fortran_order, loaded2.header.fortran_order);
    assert_eq!(loaded2.header.fortran_order, loaded3.header.fortran_order);
}

// ─────────────────────────────────────────────────────────────────────────────
// MR: Large array roundtrip (stress test)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_large_array_roundtrip() {
    let n = 100_000;
    let values: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let header = NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: false,
        shape: vec![n],
    };

    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();

    let loaded_values: &[f64] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(
        loaded_values.len(),
        n,
        "large array should roundtrip with correct size"
    );

    // Spot check values
    assert_eq!(loaded_values[0], values[0]);
    assert_eq!(loaded_values[n / 2], values[n / 2]);
    assert_eq!(loaded_values[n - 1], values[n - 1]);
}

// ─────────────────────────────────────────────────────────────────────────────
// MR: Integer boundary values roundtrip
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_integer_boundary_roundtrip_i8() {
    let values: Vec<i8> = vec![i8::MIN, i8::MIN + 1, -1, 0, 1, i8::MAX - 1, i8::MAX];
    let header = NpyHeader {
        descr: IOSupportedDType::I8,
        fortran_order: false,
        shape: vec![7],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();
    let loaded_values: &[i8] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(
        loaded_values,
        &values[..],
        "i8 boundary values should roundtrip"
    );
}

#[test]
fn mr_integer_boundary_roundtrip_i16() {
    let values: Vec<i16> = vec![i16::MIN, -1000, -1, 0, 1, 1000, i16::MAX];
    let header = NpyHeader {
        descr: IOSupportedDType::I16,
        fortran_order: false,
        shape: vec![7],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();
    let loaded_values: &[i16] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(
        loaded_values,
        &values[..],
        "i16 boundary values should roundtrip"
    );
}

#[test]
fn mr_integer_boundary_roundtrip_u8() {
    let values: Vec<u8> = vec![0, 1, 127, 128, 254, 255];
    let header = NpyHeader {
        descr: IOSupportedDType::U8,
        fortran_order: false,
        shape: vec![6],
    };
    let bytes = write_npy_bytes(&header, &values, false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();
    assert_eq!(
        loaded.payload.as_ref(),
        &values[..],
        "u8 boundary values should roundtrip"
    );
}

#[test]
fn mr_integer_boundary_roundtrip_u16() {
    let values: Vec<u16> = vec![0, 1, 1000, 32767, 32768, 65534, 65535];
    let header = NpyHeader {
        descr: IOSupportedDType::U16,
        fortran_order: false,
        shape: vec![7],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();
    let loaded_values: &[u16] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(
        loaded_values,
        &values[..],
        "u16 boundary values should roundtrip"
    );
}

#[test]
fn mr_integer_boundary_roundtrip_u32() {
    let values: Vec<u32> = vec![0, 1, u32::MAX / 2, u32::MAX - 1, u32::MAX];
    let header = NpyHeader {
        descr: IOSupportedDType::U32,
        fortran_order: false,
        shape: vec![5],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();
    let loaded_values: &[u32] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(
        loaded_values,
        &values[..],
        "u32 boundary values should roundtrip"
    );
}

#[test]
fn mr_integer_boundary_roundtrip_i32() {
    let values: Vec<i32> = vec![i32::MIN, -1_000_000, -1, 0, 1, 1_000_000, i32::MAX];
    let header = NpyHeader {
        descr: IOSupportedDType::I32,
        fortran_order: false,
        shape: vec![7],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();
    let loaded_values: &[i32] = bytemuck::cast_slice(&loaded.payload);
    assert_eq!(
        loaded_values,
        &values[..],
        "i32 boundary values should roundtrip"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR: Float special values roundtrip
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_float_special_values_roundtrip_f64() {
    let values: Vec<f64> = vec![
        0.0,
        -0.0,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let header = NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: false,
        shape: vec![8],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();
    let loaded_values: &[f64] = bytemuck::cast_slice(&loaded.payload);

    // Check non-NaN values directly
    for i in 0..7 {
        assert_eq!(
            loaded_values[i], values[i],
            "f64 special value at index {} should roundtrip",
            i
        );
    }
    // NaN requires special comparison
    assert!(loaded_values[7].is_nan(), "NaN should roundtrip as NaN");
}

#[test]
fn mr_float_special_values_roundtrip_f32() {
    let values: Vec<f32> = vec![
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let header = NpyHeader {
        descr: IOSupportedDType::F32,
        fortran_order: false,
        shape: vec![8],
    };
    let bytes = write_npy_bytes(&header, bytemuck::cast_slice(&values), false).unwrap();
    let loaded = read_npy_bytes(&bytes, false).unwrap();
    let loaded_values: &[f32] = bytemuck::cast_slice(&loaded.payload);

    for i in 0..7 {
        assert_eq!(
            loaded_values[i], values[i],
            "f32 special value at index {} should roundtrip",
            i
        );
    }
    assert!(loaded_values[7].is_nan(), "f32 NaN should roundtrip as NaN");
}
