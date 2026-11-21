//! Reduced QR contracts for typed real/complex arrays and matrix stacks.
//!
//! QR factors can differ by column signs/phases and by the basis of a null
//! space. Compare reconstruction and orthonormality, along with NumPy's exact
//! dtype and shape contracts, instead of assuming one particular factorization.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::UFuncArray;
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Stdio};

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::I8 => "int8",
        DType::I16 => "int16",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::U8 => "uint8",
        DType::U16 => "uint16",
        DType::U32 => "uint32",
        DType::U64 => "uint64",
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
        _ => panic!("unexpected QR dtype {dtype:?}"),
    }
}

fn serialize(array: &UFuncArray) -> Value {
    let mut shape = array.shape().to_vec();
    if array.dtype().is_complex() {
        assert_eq!(shape.pop(), Some(2));
    }
    if matches!(array.dtype(), DType::F32 | DType::Complex64) {
        for &value in array.values() {
            assert_eq!(value.to_bits(), f64::from(value as f32).to_bits());
        }
    }
    assert_eq!(
        array.to_storage().unwrap().len(),
        shape.iter().product::<usize>()
    );
    json!({"shape": shape, "dtype": dtype_name(array.dtype()), "values": array.values()})
}

fn qr_case(source: &UFuncArray) -> Value {
    let (q, r) = source.qr().unwrap();
    let mut shape = source.shape().to_vec();
    if source.dtype().is_complex() {
        shape.pop();
    }
    // Raw working cells deliberately have more precision than the source
    // dtype in some cases; the oracle must receive the actual typed values.
    let typed_source = UFuncArray::from_storage(shape, source.to_storage().unwrap()).unwrap();
    json!({"source": serialize(&typed_source), "q": serialize(&q), "r": serialize(&r)})
}

fn oracle(cases: &[Value]) {
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let script = r#"
import json, sys
import numpy as np

def array(record):
    values = np.array(record['values'], dtype=np.float64)
    if record['dtype'].startswith('complex'):
        values = values[::2] + 1j * values[1::2]
    return values.astype(record['dtype']).reshape(record['shape'])

for case in json.load(sys.stdin):
    source, q, r = [array(case[key]) for key in ('source', 'q', 'r')]
    expected_q, expected_r = np.linalg.qr(source, mode='reduced')
    for actual, expected in ((q, expected_q), (r, expected_r)):
        assert actual.shape == expected.shape, (case, actual.shape, expected.shape)
        assert actual.dtype == expected.dtype, (case, actual.dtype, expected.dtype)
    m, n = source.shape[-2:]
    k = min(m, n)
    precision = np.float32 if q.dtype in (np.float32, np.complex64) else np.float64
    tolerance = 128 * max(m, n, 1) * np.finfo(precision).eps
    working_dtype = np.complex128 if np.iscomplexobj(q) else np.float64
    q, r, source = [a.astype(working_dtype) for a in (q, r, source)]
    assert np.all(np.isfinite(q)) and np.all(np.isfinite(r)), case
    np.testing.assert_array_equal(np.tril(r, -1), np.zeros_like(r))
    if k:
        # A huge unrelated column must not hide loss of a tiny column.
        scales = np.max(np.maximum(np.abs(source.real), np.abs(source.imag)), axis=-2)
        scales = np.where(scales == 0, 1, scales)[..., None, :]
        np.testing.assert_allclose(q @ (r / scales), source / scales,
                                   rtol=0, atol=tolerance)
    gram = np.swapaxes(q.conj(), -2, -1) @ q
    np.testing.assert_allclose(gram, np.broadcast_to(np.eye(k), gram.shape),
                               rtol=0, atol=tolerance)
print(f'NumPy {np.__version__}: typed QR contracts passed')
"#;
    let mut child = Command::new(python)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("NumPy oracle interpreter must be available");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(cases).unwrap().as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "NumPy QR oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    eprintln!("{}", String::from_utf8_lossy(&output.stdout));
}

#[test]
fn reduced_qr_typed_matrix_stacks_match_numpy() {
    let mut cases = Vec::new();
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for (m, n) in [(1, 1), (1, 5), (5, 1), (3, 2), (2, 3), (4, 4)] {
            for batch in [vec![], vec![2], vec![2, 1, 2]] {
                let mut shape = batch.clone();
                shape.extend([m, n]);
                let mut values = Vec::new();
                for matrix in 0..batch.iter().product::<usize>() {
                    for row in 0..m {
                        for column in 0..n {
                            let seed = (matrix * 13 + row * 11 + column * 7 + 3) % 23;
                            let value = if dtype == DType::Bool {
                                if seed % 2 == 1 { -0.5 } else { 0.0 }
                            } else if dtype.is_integer() {
                                // Narrow integer inputs must be normalized
                                // before the double-precision factorization.
                                seed as f64 + 250.0
                            } else {
                                (seed as f64 - 11.0) / 7.0
                            };
                            values.push(value);
                            if dtype.is_complex() {
                                values.push((seed as f64 - 9.0) / 11.0);
                            }
                        }
                    }
                }
                if dtype.is_complex() {
                    shape.push(2);
                }
                cases.push(qr_case(&UFuncArray::new(shape, values, dtype).unwrap()));
            }
        }
    }
    assert_eq!(cases.len(), 234);
    oracle(&cases);
}

#[test]
fn reduced_qr_rank_deficient_zero_and_scaled_columns_match_numpy() {
    let mut cases = Vec::new();
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        for (m, n) in [(5, 3), (3, 5), (3, 3)] {
            let mut shape = vec![2, m, n];
            let components = if dtype.is_complex() {
                shape.push(2);
                2
            } else {
                1
            };
            let mut values = vec![0.0; 2 * m * n * components];
            for row in 0..m {
                for column in 0..n {
                    // Batch zero is rank one; batch one is exactly zero.
                    let at = components * (row * n + column);
                    values[at] = (row as f64 - 1.5) * (column as f64 + 1.0);
                    if components == 2 {
                        values[at + 1] = (row as f64 + 0.5) * (column as f64 + 1.0);
                    }
                }
            }
            cases.push(qr_case(&UFuncArray::new(shape, values, dtype).unwrap()));
        }
        let scales = if matches!(dtype, DType::F32 | DType::Complex64) {
            [1e30, 1e-30, 1e10]
        } else {
            [1e300, 1e-300, 1e100]
        };
        let mut values = Vec::new();
        for row in [[1.0, 3.0, 2.0], [2.0, -1.0, 3.0], [3.0, 2.0, -2.0]] {
            for (value, scale) in row.into_iter().zip(scales) {
                values.push(value * scale);
                if dtype.is_complex() {
                    values.push(-0.5 * value * scale);
                }
            }
        }
        let mut shape = vec![3, 3];
        if dtype.is_complex() {
            shape.push(2);
        }
        cases.push(qr_case(&UFuncArray::new(shape, values, dtype).unwrap()));
    }
    oracle(&cases);
}

#[test]
fn reduced_qr_preserves_exact_integer_inputs_and_declared_precision() {
    for storage in [
        ArrayStorage::I64(vec![9_007_199_254_740_993, 0, 0, 9_007_199_254_740_995]),
        ArrayStorage::U64(vec![u64::MAX - 1, 0, 0, 9_007_199_254_740_997]),
    ] {
        let source = UFuncArray::from_storage(vec![2, 2], storage.clone()).unwrap();
        assert!(source.has_integer_sidecar());
        let reference =
            UFuncArray::from_storage(vec![2, 2], storage.cast_to(DType::F64).unwrap()).unwrap();
        let (q, r) = source.qr().unwrap();
        let (expected_q, expected_r) = reference.qr().unwrap();
        assert_eq!(q.dtype(), DType::F64);
        assert_eq!(r.dtype(), DType::F64);
        assert_eq!(q.values(), expected_q.values());
        assert_eq!(r.values(), expected_r.values());
        assert_eq!(source.to_storage().unwrap(), storage);
    }
    for dtype in [DType::F32, DType::Complex64] {
        let mut shape = vec![2, 2];
        let mut values = vec![1.0 + 2.0_f64.powi(-30), 1.0, 1.0, 1.0];
        if dtype.is_complex() {
            shape.push(2);
            values = values
                .into_iter()
                .flat_map(|value| [value, 1e-50])
                .collect();
        }
        let source = UFuncArray::new(shape, values, dtype).unwrap();
        let normalized =
            UFuncArray::from_storage(vec![2, 2], source.to_storage().unwrap()).unwrap();
        let (q, r) = source.qr().unwrap();
        let (expected_q, expected_r) = normalized.qr().unwrap();
        assert_eq!(q.dtype(), dtype);
        assert_eq!(r.dtype(), dtype);
        assert_eq!(q.values(), expected_q.values());
        assert_eq!(r.values(), expected_r.values());
        serialize(&q);
        serialize(&r);
    }
}

#[test]
fn reduced_qr_empty_cores_and_batches_do_not_multiply_unused_extents() {
    let mut cases = Vec::new();
    for dtype in [
        DType::Bool,
        DType::I64,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for logical_shape in [
            vec![0, 0],
            vec![4, 0],
            vec![0, 5],
            vec![3, 0, 0],
            vec![0, 2, 3],
            vec![2, 0, 3, 4],
            vec![usize::MAX, 0],
            vec![0, usize::MAX],
            vec![0, usize::MAX, usize::MAX],
            vec![0, usize::MAX, usize::MAX, usize::MAX],
        ] {
            let ndim = logical_shape.len();
            let (m, n) = (logical_shape[ndim - 2], logical_shape[ndim - 1]);
            let mut q_shape = logical_shape[..ndim - 2].to_vec();
            let mut r_shape = q_shape.clone();
            q_shape.extend([m, m.min(n)]);
            r_shape.extend([m.min(n), n]);
            let mut shape = logical_shape.clone();
            if dtype.is_complex() {
                shape.push(2);
                q_shape.push(2);
                r_shape.push(2);
            }
            let source = UFuncArray::new(shape, Vec::new(), dtype).unwrap();
            let (q, r) = source.qr().unwrap();
            assert_eq!(q.shape(), q_shape);
            assert_eq!(r.shape(), r_shape);
            let result_dtype = match dtype {
                DType::Bool | DType::I64 => DType::F64,
                other => other,
            };
            for factor in [&q, &r] {
                assert_eq!(factor.dtype(), result_dtype);
                assert!(factor.values().is_empty());
                assert!(factor.to_storage().unwrap().is_empty());
            }
            if !logical_shape.contains(&usize::MAX) {
                cases.push(qr_case(&source));
            }
        }
    }
    oracle(&cases);
}

#[test]
fn reduced_qr_rejects_invalid_shapes_dtypes_and_nonfinite_values() {
    for source in [
        UFuncArray::new(vec![], vec![1.0], DType::F64).unwrap(),
        UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap(),
        UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::F16).unwrap(),
        UFuncArray::new(vec![0, 0], Vec::new(), DType::F16).unwrap(),
        UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::DateTime64).unwrap(),
        UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::Str).unwrap(),
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
        UFuncArray::new(vec![2], vec![1.0, 0.0], DType::Complex64).unwrap(),
        UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::Complex64).unwrap(),
        UFuncArray::new(vec![0, 1], Vec::new(), DType::Complex128).unwrap(),
    ] {
        assert!(source.qr().is_err());
    }
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut shape = vec![2, 1, 1];
            let values = if dtype.is_complex() {
                shape.push(2);
                vec![1.0, 0.0, 0.0, invalid]
            } else {
                vec![1.0, invalid]
            };
            assert!(UFuncArray::new(shape, values, dtype).unwrap().qr().is_err());
        }
    }
}

#[test]
fn reduced_qr_large_skinny_stacks_return_only_thin_factors() {
    // A full Q would require 80 GB per tall real matrix. The reduced bridge
    // and kernels must finish using storage proportional to the input.
    for dtype in [DType::F64, DType::Complex128] {
        for (m, n) in [(100_000, 1), (1, 100_000)] {
            let components = if dtype.is_complex() { 2 } else { 1 };
            let mut shape = vec![2, m, n];
            if dtype.is_complex() {
                shape.push(2);
            }
            let mut values = Vec::new();
            for index in 0..2 * m * n {
                values.push((index % 7) as f64 - 2.5);
                if dtype.is_complex() {
                    values.push((index % 5) as f64 - 1.5);
                }
            }
            let source = UFuncArray::new(shape, values, dtype).unwrap();
            let (q, r) = source.qr().unwrap();
            assert_eq!(q.values().len(), 2 * m * components);
            assert_eq!(r.values().len(), 2 * n * components);
            for batch in 0..2 {
                let mut norm = 0.0;
                for row in 0..m {
                    let qi = components * (batch * m + row);
                    let qr = q.values()[qi];
                    let qim = if components == 2 {
                        q.values()[qi + 1]
                    } else {
                        0.0
                    };
                    norm += qr * qr + qim * qim;
                    for column in 0..n {
                        let ri = components * (batch * n + column);
                        let rr = r.values()[ri];
                        let rim = if components == 2 {
                            r.values()[ri + 1]
                        } else {
                            0.0
                        };
                        let input = components * ((batch * m + row) * n + column);
                        assert!((qr * rr - qim * rim - source.values()[input]).abs() < 2e-10);
                        if components == 2 {
                            assert!(
                                (qr * rim + qim * rr - source.values()[input + 1]).abs() < 2e-10
                            );
                        }
                    }
                }
                assert!((norm - 1.0).abs() < 2e-10);
            }
        }
    }
}
