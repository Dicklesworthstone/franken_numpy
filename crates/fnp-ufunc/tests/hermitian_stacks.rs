//! Typed Hermitian matrix stacks and explicit triangle selection.
//!
//! Eigenvector signs and bases for repeated eigenvalues are not unique. The
//! oracle therefore checks spectra, eigenpair residuals, and orthonormality.

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
        _ => panic!("unexpected Hermitian dtype {dtype:?}"),
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
    source = array(case['source'])
    expected_values, expected_vectors = np.linalg.eigh(source, UPLO=case['uplo'])
    expected_only = np.linalg.eigvalsh(source, UPLO=case['uplo'])
    values, vectors, only = [array(case[key]) for key in ('values', 'vectors', 'only')]
    for actual, expected in ((values, expected_values), (vectors, expected_vectors),
                             (only, expected_only)):
        assert actual.shape == expected.shape, (case, actual.shape, expected.shape)
        assert actual.dtype == expected.dtype, (case, actual.dtype, expected.dtype)
    n = source.shape[-1]
    precision = np.float32 if values.dtype == np.float32 else np.float64
    tolerance = 256 * max(n, 1) * np.finfo(precision).eps
    source = source.astype(np.complex128 if np.iscomplexobj(source) else np.float64)
    if case['uplo'].upper() == 'L':
        selected = np.tril(source)
        selected = selected + np.swapaxes(np.tril(source, -1).conj(), -2, -1)
    else:
        selected = np.triu(source)
        selected = selected + np.swapaxes(np.triu(source, 1).conj(), -2, -1)
    diagonal = np.arange(n)
    selected[..., diagonal, diagonal] = selected[..., diagonal, diagonal].real
    scale = np.maximum(np.linalg.norm(selected, axis=(-2, -1)), 1)
    for actual, expected in ((values, expected_values), (only, expected_only)):
        assert np.all(np.abs(actual - expected) <= tolerance * scale[..., None]), case
    residual = selected @ vectors - vectors * values[..., None, :]
    assert np.all(np.linalg.norm(residual, axis=(-2, -1)) <= tolerance * scale), case
    gram = np.swapaxes(vectors.conj(), -2, -1) @ vectors
    np.testing.assert_allclose(gram, np.broadcast_to(np.eye(n), gram.shape),
                               rtol=0, atol=tolerance)
print(f'NumPy {np.__version__}: Hermitian stack spectra and eigenpairs passed')
"#;
    let mut child = Command::new(python)
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
        "NumPy oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    eprintln!("{}", String::from_utf8_lossy(&output.stdout));
}

#[test]
fn typed_hermitian_stacks_match_numpy_spectra_and_eigenpairs() {
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
        for n in [1, 2, 4] {
            for batch in [vec![], vec![2], vec![2, 1, 2]] {
                let mut shape = batch.clone();
                shape.extend([n, n]);
                let mut values = Vec::new();
                for matrix in 0..batch.iter().product::<usize>() {
                    for row in 0..n {
                        for column in 0..n {
                            let seed = (matrix * 13 + row * 11 + column * 7 + 3) % 23;
                            let value = if dtype == DType::Bool {
                                if seed % 2 == 1 { 1.0 } else { 0.0 }
                            } else if dtype.is_integer() {
                                seed as f64
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
                let source = UFuncArray::new(shape, values, dtype).unwrap();
                for uplo in ["L", "U", "l", "u"] {
                    let (values, vectors) = source.eigh_with_uplo(uplo).unwrap();
                    let only = source.eigvalsh_with_uplo(uplo).unwrap();
                    // Record the source with its actual storage precision;
                    // raw float32 working cells are deliberately unrounded.
                    let typed_source = UFuncArray::from_storage(
                        if dtype.is_complex() {
                            source.shape()[..source.shape().len() - 1].to_vec()
                        } else {
                            source.shape().to_vec()
                        },
                        source.to_storage().unwrap(),
                    )
                    .unwrap();
                    cases.push(json!({
                        "source": serialize(&typed_source), "uplo": uplo,
                        "values": serialize(&values), "vectors": serialize(&vectors),
                        "only": serialize(&only),
                    }));
                }
            }
        }
    }
    assert_eq!(cases.len(), 468);
    oracle(&cases);
}

#[test]
fn selected_triangle_ignores_nonfinite_unused_entries_and_imaginary_diagonal() {
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        for uplo in ["L", "U", "l", "u"] {
            let lower = uplo.eq_ignore_ascii_case("L");
            let mut values = Vec::new();
            for multiplier in [1.0, 2.0] {
                let real = if lower {
                    [2.0 * multiplier, f64::NAN, multiplier, 3.0 * multiplier]
                } else {
                    [
                        2.0 * multiplier,
                        multiplier,
                        f64::INFINITY,
                        3.0 * multiplier,
                    ]
                };
                for (index, value) in real.into_iter().enumerate() {
                    values.push(value);
                    if dtype.is_complex() {
                        values.push(if index == 0 || index == 3 {
                            f64::NAN
                        } else if (index == 2) == lower {
                            multiplier
                        } else {
                            f64::INFINITY
                        });
                    }
                }
            }
            let mut shape = vec![2, 2, 2];
            if dtype.is_complex() {
                shape.push(2);
            }
            let source = UFuncArray::new(shape, values, dtype).unwrap();
            for actual in [
                source.eigh_with_uplo(uplo).unwrap().0,
                source.eigvalsh_with_uplo(uplo).unwrap(),
            ] {
                let root = if dtype.is_complex() {
                    3.0
                } else {
                    5.0_f64.sqrt()
                };
                for (&value, expected) in actual.values().iter().zip([
                    (5.0 - root) / 2.0,
                    (5.0 + root) / 2.0,
                    5.0 - root,
                    5.0 + root,
                ]) {
                    let tolerance = if matches!(dtype, DType::F32 | DType::Complex64) {
                        2e-6
                    } else {
                        2e-13
                    };
                    assert!((value - expected).abs() <= tolerance);
                }
            }
        }
    }
}

#[test]
fn hermitian_defaults_use_lower_triangle_and_preserve_declared_precision() {
    for dtype in [DType::F32, DType::Complex64] {
        let mut shape = vec![2, 2];
        let mut values = vec![1.0 + 2.0_f64.powi(-30), 77.0, 0.0, -2.0 + 2.0_f64.powi(-28)];
        if dtype.is_complex() {
            shape.push(2);
            values = values
                .into_iter()
                .flat_map(|value| [value, 1e-50])
                .collect();
        }
        let source = UFuncArray::new(shape, values, dtype).unwrap();
        let (spectrum, vectors) = source.eigh().unwrap();
        assert_eq!(spectrum.values(), &[-2.0, 1.0]);
        assert_eq!(spectrum.dtype(), DType::F32);
        assert_eq!(vectors.dtype(), dtype);
        assert_eq!(source.eigvalsh().unwrap().values(), spectrum.values());
        assert_eq!(
            source.eigh_with_uplo("L").unwrap().0.values(),
            spectrum.values()
        );
        serialize(&vectors);
    }

    let exact = vec![9_007_199_254_740_993_i64, 0, 0, 9_007_199_254_740_995];
    let source = UFuncArray::from_storage(vec![2, 2], ArrayStorage::I64(exact.clone())).unwrap();
    assert!(source.has_integer_sidecar());
    let values = source.eigvalsh().unwrap();
    assert_eq!(values.dtype(), DType::F64);
    assert_eq!(values.values(), &[exact[0] as f64, exact[3] as f64]);
    assert_eq!(source.to_storage().unwrap(), ArrayStorage::I64(exact));
}

#[test]
fn hermitian_zero_matrices_keep_batch_shapes_and_return_orthonormal_bases() {
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        let mut shape = vec![2, 3, 3];
        let components = if dtype.is_complex() {
            shape.push(2);
            2
        } else {
            1
        };
        let source = UFuncArray::new(shape, vec![0.0; 18 * components], dtype).unwrap();
        let (values, vectors) = source.eigh().unwrap();
        assert_eq!(values.shape(), &[2, 3]);
        assert!(values.values().iter().all(|&value| value == 0.0));
        for matrix in vectors.values().chunks_exact(9 * components) {
            for left in 0..3 {
                for right in 0..3 {
                    let mut inner = (0.0, 0.0);
                    for row in 0..3 {
                        let left_index = (row * 3 + left) * components;
                        let right_index = (row * 3 + right) * components;
                        let (left_real, right_real) = (matrix[left_index], matrix[right_index]);
                        inner.0 += left_real * right_real;
                        if components == 2 {
                            let (left_imag, right_imag) =
                                (matrix[left_index + 1], matrix[right_index + 1]);
                            inner.0 += left_imag * right_imag;
                            inner.1 += left_real * right_imag - left_imag * right_real;
                        }
                    }
                    let expected = if left == right { 1.0 } else { 0.0 };
                    assert!((inner.0 - expected).abs() < 1e-6);
                    assert!(inner.1.abs() < 1e-6);
                }
            }
        }
    }
}

#[test]
fn hermitian_empty_cores_and_batches_preserve_shapes_without_unused_overflow() {
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
            vec![3, 0, 0],
            vec![0, 2, 2],
            vec![2, 0, 3, 3],
            vec![0, usize::MAX, usize::MAX],
            vec![0, usize::MAX, usize::MAX, usize::MAX],
        ] {
            let mut shape = logical_shape.clone();
            if dtype.is_complex() {
                shape.push(2);
            }
            let source = UFuncArray::new(shape.clone(), Vec::new(), dtype).unwrap();
            let (values, vectors) = source.eigh_with_uplo("U").unwrap();
            let only = source.eigvalsh_with_uplo("U").unwrap();
            let real_dtype = if matches!(dtype, DType::F32 | DType::Complex64) {
                DType::F32
            } else {
                DType::F64
            };
            let vector_dtype = match dtype {
                DType::Bool | DType::I64 => DType::F64,
                other => other,
            };
            assert_eq!(values.shape(), &logical_shape[..logical_shape.len() - 1]);
            assert_eq!(values.dtype(), real_dtype);
            assert_eq!(only.shape(), values.shape());
            assert_eq!(only.dtype(), real_dtype);
            assert_eq!(vectors.shape(), shape);
            assert_eq!(vectors.dtype(), vector_dtype);
            for output in [&values, &vectors, &only] {
                assert!(output.values().is_empty());
                assert!(output.to_storage().unwrap().is_empty());
            }
        }
    }
}

#[test]
fn hermitian_stacks_reject_invalid_triangles_shapes_and_dtypes() {
    let valid = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::F64).unwrap();
    let empty = UFuncArray::new(vec![0, 0], Vec::new(), DType::F64).unwrap();
    for source in [&valid, &empty] {
        for uplo in ["", "lower", "LL", "N", " L"] {
            assert!(source.eigh_with_uplo(uplo).is_err());
            assert!(source.eigvalsh_with_uplo(uplo).is_err());
        }
    }
    for source in [
        UFuncArray::new(vec![], vec![1.0], DType::F64).unwrap(),
        UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap(),
        UFuncArray::new(vec![2, 2, 3], vec![1.0; 12], DType::F64).unwrap(),
        UFuncArray::new(vec![0, 2, 3], Vec::new(), DType::F64).unwrap(),
        UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::F16).unwrap(),
        UFuncArray::new(vec![0, 0], Vec::new(), DType::F16).unwrap(),
        UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::DateTime64).unwrap(),
        UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::Str).unwrap(),
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
        UFuncArray::new(vec![2], vec![1.0, 0.0], DType::Complex64).unwrap(),
        UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::Complex64).unwrap(),
        UFuncArray::new(vec![0, 1], Vec::new(), DType::Complex128).unwrap(),
    ] {
        assert!(source.eigh().is_err());
        assert!(source.eigvalsh().is_err());
    }
}

#[test]
fn hermitian_stacks_reject_nonfinite_values_in_the_selected_matrix() {
    for dtype in [DType::F64, DType::Complex128] {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for selected_index in [0, 2, 3] {
                let mut values = vec![2.0, 99.0, 1.0, 3.0];
                values[selected_index] = invalid;
                let mut shape = vec![2, 2];
                if dtype.is_complex() {
                    shape.push(2);
                    values = values.into_iter().flat_map(|value| [value, 0.0]).collect();
                }
                let source = UFuncArray::new(shape, values, dtype).unwrap();
                assert!(source.eigh().is_err());
                assert!(source.eigvalsh().is_err());
            }
        }
    }
}
