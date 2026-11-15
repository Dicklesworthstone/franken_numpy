//! Complex spectral bridges: algebraic identities and typed storage contracts.
//!
//! Residuals and orthogonality avoid assuming a particular eigenvector phase or
//! a particular basis inside a repeated eigenspace or singular nullspace.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::UFuncArray;
use serde_json::json;
use std::io::Write;
use std::process::{Command, Stdio};

type Complex = (f64, f64);

fn matrix(rows: usize, cols: usize, values: &[Complex], dtype: DType) -> UFuncArray {
    UFuncArray::from_storage(
        vec![rows, cols],
        ArrayStorage::Complex128(values.to_vec())
            .cast_to(dtype)
            .unwrap(),
    )
    .unwrap()
}

fn at(array: &UFuncArray, row: usize, column: usize) -> Complex {
    let offset = 2 * (row * array.shape()[1] + column);
    (array.values()[offset], array.values()[offset + 1])
}

fn multiply(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn conjugate(value: Complex) -> Complex {
    (value.0, -value.1)
}

fn tolerance(dtype: DType) -> f64 {
    if dtype == DType::Complex64 {
        8.0e-6
    } else {
        1.0e-10
    }
}

fn real_dtype(dtype: DType) -> DType {
    if dtype == DType::Complex64 {
        DType::F32
    } else {
        DType::F64
    }
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(actual.is_finite(), "{actual} must be finite");
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} differs from {expected}, tolerance {tolerance}"
    );
}

fn complex_close(actual: Complex, expected: Complex, tolerance: f64) {
    close(actual.0, expected.0, tolerance);
    close(actual.1, expected.1, tolerance);
}

fn assert_declared_precision(array: &UFuncArray) {
    if matches!(array.dtype(), DType::F32 | DType::Complex64) {
        for &value in array.values() {
            assert_eq!(
                value,
                f64::from(value as f32),
                "output cells must have their declared single precision"
            );
        }
    }
}

fn assert_orthonormal(array: &UFuncArray, columns: bool, tolerance: f64) {
    let (rows, cols) = (array.shape()[0], array.shape()[1]);
    let (vectors, terms) = if columns { (cols, rows) } else { (rows, cols) };
    for left in 0..vectors {
        for right in 0..vectors {
            let mut inner = (0.0, 0.0);
            for term in 0..terms {
                let a = if columns {
                    at(array, term, left)
                } else {
                    at(array, left, term)
                };
                let b = if columns {
                    at(array, term, right)
                } else {
                    at(array, right, term)
                };
                let product = multiply(conjugate(a), b);
                inner.0 += product.0;
                inner.1 += product.1;
            }
            complex_close(
                inner,
                (if left == right { 1.0 } else { 0.0 }, 0.0),
                tolerance,
            );
        }
    }
}

fn assert_eigenpairs(input: &UFuncArray, hermitian: &[Complex], expected_values: &[f64]) {
    let n = expected_values.len();
    let dtype = input.dtype();
    let tolerance = tolerance(dtype);
    let (values, vectors) = input.eigh().unwrap();
    let values_only = input.eigvalsh().unwrap();
    assert_eq!(values.shape(), &[n]);
    assert_eq!(values_only.shape(), &[n]);
    assert_eq!(vectors.shape(), &[n, n, 2]);
    assert_eq!(values.dtype(), real_dtype(dtype));
    assert_eq!(values_only.dtype(), real_dtype(dtype));
    assert_eq!(vectors.dtype(), dtype);
    for output in [&values, &values_only, &vectors] {
        assert_declared_precision(output);
    }
    for ((&actual, &only), &expected) in values
        .values()
        .iter()
        .zip(values_only.values())
        .zip(expected_values)
    {
        close(actual, expected, tolerance);
        close(only, expected, tolerance);
    }
    assert_orthonormal(&vectors, true, tolerance);
    for row in 0..n {
        for column in 0..n {
            let mut product = (0.0, 0.0);
            for inner in 0..n {
                let term = multiply(hermitian[row * n + inner], at(&vectors, inner, column));
                product.0 += term.0;
                product.1 += term.1;
            }
            let vector = at(&vectors, row, column);
            let eigenvalue = values.values()[column];
            complex_close(
                product,
                (vector.0 * eigenvalue, vector.1 * eigenvalue),
                tolerance,
            );
        }
    }
}

fn assert_svd(input: &UFuncArray, full: bool) -> Vec<f64> {
    let (m, n) = (input.shape()[0], input.shape()[1]);
    let k = m.min(n);
    let dtype = input.dtype();
    let tolerance = tolerance(dtype);
    let (u, values, vh) = if full {
        input.svd().unwrap()
    } else {
        input.svd_full(false).unwrap()
    };
    assert_eq!(u.shape(), &[m, if full { m } else { k }, 2]);
    assert_eq!(values.shape(), &[k]);
    assert_eq!(vh.shape(), &[if full { n } else { k }, n, 2]);
    assert_eq!(u.dtype(), dtype);
    assert_eq!(vh.dtype(), dtype);
    assert_eq!(values.dtype(), real_dtype(dtype));
    for output in [&u, &values, &vh] {
        assert_declared_precision(output);
    }
    assert!(values.values().iter().all(|&value| value >= 0.0));
    assert!(values.values().windows(2).all(|pair| pair[0] >= pair[1]));
    let values_only = input.svdvals().unwrap();
    assert_eq!(values_only.shape(), &[k]);
    assert_eq!(values_only.dtype(), real_dtype(dtype));
    assert_declared_precision(&values_only);
    for (&actual, &only) in values.values().iter().zip(values_only.values()) {
        close(actual, only, tolerance);
    }
    assert_orthonormal(&u, true, tolerance);
    assert_orthonormal(&vh, false, tolerance);
    for row in 0..m {
        for column in 0..n {
            let mut reconstructed = (0.0, 0.0);
            for inner in 0..k {
                let term = multiply(at(&u, row, inner), at(&vh, inner, column));
                reconstructed.0 += term.0 * values.values()[inner];
                reconstructed.1 += term.1 * values.values()[inner];
            }
            complex_close(reconstructed, at(input, row, column), tolerance);
        }
    }
    values.values().to_vec()
}

#[test]
fn hermitian_bridges_use_lower_triangle_and_ignore_imaginary_diagonal() {
    // The represented Hermitian matrix has trace 5 and determinant 4, hence
    // exact eigenvalues 1 and 4. The upper half and imaginary diagonal are
    // deliberately unrelated to that matrix.
    let hermitian = [(2.0, 0.0), (1.0, 1.0), (1.0, -1.0), (3.0, 0.0)];
    let stored = [(2.0, 99.0), (123.0, -456.0), (1.0, -1.0), (3.0, -77.0)];
    for dtype in [DType::Complex64, DType::Complex128] {
        assert_eigenpairs(&matrix(2, 2, &stored, dtype), &hermitian, &[1.0, 4.0]);
    }
}

#[test]
fn hermitian_repeated_eigenvalues_return_an_orthonormal_basis() {
    // I + 3*u*u^H with u=(1, i, 0)/sqrt(2): eigenvalues 1, 1, 4.
    let hermitian = [
        (2.5, 0.0),
        (0.0, -1.5),
        (0.0, 0.0),
        (0.0, 1.5),
        (2.5, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (1.0, 0.0),
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        assert_eigenpairs(
            &matrix(3, 3, &hermitian, dtype),
            &hermitian,
            &[1.0, 1.0, 4.0],
        );
    }
}

#[test]
fn complex_svd_full_and_thin_reconstruct_tall_wide_and_rank_deficient_inputs() {
    let fixtures = [
        (
            3,
            2,
            vec![
                (1.0, 1.0),
                (2.0, -1.0),
                (0.0, 2.0),
                (3.0, 0.0),
                (-1.0, 0.5),
                (1.0, -1.0),
            ],
        ),
        (
            2,
            3,
            vec![
                (1.0, 1.0),
                (2.0, -1.0),
                (-1.0, 0.5),
                (0.0, 2.0),
                (3.0, 0.0),
                (1.0, -1.0),
            ],
        ),
        (
            3,
            3,
            vec![
                (3.0, 4.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 2.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ],
        ),
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        for (m, n, entries) in &fixtures {
            let input = matrix(*m, *n, entries, dtype);
            for full in [false, true] {
                let values = assert_svd(&input, full);
                if *m == 3 && *n == 3 {
                    for (&actual, expected) in values.iter().zip([5.0, 2.0, 0.0]) {
                        close(actual, expected, tolerance(dtype));
                    }
                }
            }
        }
    }
}

#[test]
fn complex_spectral_bridges_normalize_source_and_output_precision() {
    // At declared single precision this matrix is rank one. Computing from
    // its unnormalized float64 working cells instead leaves a spurious value 1.
    let raw = UFuncArray::new(
        vec![2, 2, 2],
        vec![
            16_777_217.0,
            1.0e-50,
            16_777_216.0,
            0.0,
            16_777_216.0,
            0.0,
            16_777_217.0,
            -1.0e-50,
        ],
        DType::Complex64,
    )
    .unwrap();
    assert_eigenpairs(&raw, &[(16_777_216.0, 0.0); 4], &[0.0, 33_554_432.0]);
    assert_eq!(raw.eigh().unwrap().0.values(), &[0.0, 33_554_432.0]);
    assert_eq!(raw.eigvalsh().unwrap().values(), &[0.0, 33_554_432.0]);
    for full in [false, true] {
        let (_, values, _) = raw.svd_full(full).unwrap();
        assert_eq!(values.dtype(), DType::F32);
        assert_eq!(values.values(), &[33_554_432.0, 0.0]);
    }
    assert_eq!(raw.svdvals().unwrap().values(), &[33_554_432.0, 0.0]);

    let input = matrix(1, 1, &[(3.0, 3.0)], DType::Complex64);
    let expected = f64::from(18.0_f64.sqrt() as f32);
    for full in [false, true] {
        let values = assert_svd(&input, full);
        assert_eq!(values, vec![expected]);
    }
}

#[test]
fn complex_spectral_bridges_handle_empty_shapes_and_reject_invalid_layouts() {
    for dtype in [DType::Complex64, DType::Complex128] {
        assert_eigenpairs(&matrix(0, 0, &[], dtype), &[], &[]);
        for (m, n) in [(0, 0), (0, 3), (3, 0)] {
            for full in [false, true] {
                assert!(assert_svd(&matrix(m, n, &[], dtype), full).is_empty());
            }
        }
        let rectangular = matrix(2, 3, &[(0.0, 0.0); 6], dtype);
        assert!(rectangular.eigh().is_err());
        assert!(rectangular.eigvalsh().is_err());
        for invalid in [
            UFuncArray::new(vec![], vec![1.0], dtype).unwrap(),
            UFuncArray::new(vec![2], vec![1.0, 0.0], dtype).unwrap(),
            UFuncArray::new(vec![2, 3], vec![1.0; 6], dtype).unwrap(),
            UFuncArray::new(vec![0, 1], vec![], dtype).unwrap(),
            UFuncArray::from_storage_with_dtype(
                vec![1, 2, 2],
                ArrayStorage::F64(vec![1.0, 0.0, 0.0, 1.0]),
                dtype,
            )
            .unwrap(),
        ] {
            assert!(invalid.eigh().is_err());
            assert!(invalid.eigvalsh().is_err());
            assert!(invalid.svd_full(false).is_err());
            assert!(invalid.svd_full(true).is_err());
            assert!(invalid.svdvals().is_err());
        }
    }
}

#[test]
fn general_real_eigen_bridges_match_numpy_shapes_dtypes_and_eigenpairs() {
    let fixtures = [
        (vec![2, 2], vec![3.0, 0.0, 0.0, 5.0]),
        (vec![2, 2], vec![2.0, 7.0, 0.0, 5.0]),
        (vec![2, 2], vec![0.0, -1.0, 1.0, 0.0]),
        // Nontrivial eigenvectors for a coupled real Schur block. The core
        // returns columns contiguously; the array bridge must transpose that
        // storage while preserving the real/imaginary pair in each element.
        (
            vec![3, 3],
            vec![0.0, -1.0, 2.0, 1.0, 0.0, 3.0, 0.0, 0.0, 4.0],
        ),
        (vec![2, 2, 2], vec![2.0, 7.0, 0.0, 5.0, 0.0, -1.0, 1.0, 0.0]),
        (
            vec![1, 2, 2, 2],
            vec![2.0, 7.0, 0.0, 5.0, 3.0, 0.0, 0.0, 8.0],
        ),
    ];
    let mut cases = Vec::new();
    for (dtype, name) in [
        (DType::F32, "float32"),
        (DType::F64, "float64"),
        (DType::I32, "int32"),
    ] {
        for (shape, entries) in &fixtures {
            let input = UFuncArray::new(shape.clone(), entries.clone(), dtype).unwrap();
            let (values, vectors) = input.eig().unwrap();
            let values_only = input.eigvals().unwrap();
            let serialize = |output: &UFuncArray| {
                let mut shape = output.shape().to_vec();
                assert_eq!(shape.iter().product::<usize>(), output.values().len());
                if output.dtype().is_complex() {
                    assert_eq!(shape.pop(), Some(2));
                }
                assert_eq!(
                    output.to_storage().unwrap().len(),
                    shape.iter().product::<usize>()
                );
                assert_declared_precision(output);
                let dtype = match output.dtype() {
                    DType::F32 => "float32",
                    DType::F64 => "float64",
                    DType::Complex64 => "complex64",
                    DType::Complex128 => "complex128",
                    other => panic!("unexpected spectral output dtype {other:?}"),
                };
                json!({"shape": shape, "dtype": dtype, "values": output.values()})
            };
            cases.push(json!({
                "shape": shape, "input": entries, "dtype": name,
                "eigenvalues": serialize(&values),
                "eigenvectors": serialize(&vectors),
                "values_only": serialize(&values_only),
            }));
        }
    }
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let script = r#"
import json, sys
import numpy as np

def array(record):
    values = np.array(record['values'], dtype=np.float64)
    if record['dtype'].startswith('complex'):
        values = values[::2] + 1j * values[1::2]
    return values.astype(record['dtype']).reshape(record['shape'])

def spectra_close(actual, expected, tolerance):
    for left, right in zip(actual.reshape(-1, actual.shape[-1]),
                           expected.reshape(-1, expected.shape[-1])):
        remaining = list(right)
        for value in left:
            nearest = min(range(len(remaining)), key=lambda i: abs(value - remaining[i]))
            np.testing.assert_allclose(value, remaining.pop(nearest), rtol=tolerance, atol=tolerance)

cases = json.load(sys.stdin)
for case in cases:
    source = np.array(case['input'], dtype=case['dtype']).reshape(case['shape'])
    expected_values, expected_vectors = np.linalg.eig(source)
    expected_only = np.linalg.eigvals(source)
    values, vectors, only = [array(case[key]) for key in
                            ('eigenvalues', 'eigenvectors', 'values_only')]
    for actual, expected in ((values, expected_values), (vectors, expected_vectors),
                             (only, expected_only)):
        assert actual.shape == expected.shape, (actual.shape, expected.shape)
        assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    tolerance = 2e-5 if case['dtype'] == 'float32' else 2e-10
    spectra_close(values, expected_values, tolerance)
    spectra_close(only, expected_only, tolerance)
    np.testing.assert_allclose(source @ vectors, vectors * values[..., None, :],
                               rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(np.linalg.vector_norm(vectors, axis=-2), 1,
                               rtol=tolerance, atol=tolerance)
print(f"NumPy {np.__version__}: {len(cases)} eigen bridge cases passed")
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
        .write_all(serde_json::to_string(&cases).unwrap().as_bytes())
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
fn general_eigen_bridges_materialize_source_precision_and_support_empty_batches() {
    let input = UFuncArray::new(
        vec![2, 2],
        vec![16_777_217.0, 16_777_216.0, 16_777_216.0, 16_777_217.0],
        DType::F32,
    )
    .unwrap();
    for values in [input.eig().unwrap().0, input.eigvals().unwrap()] {
        assert_eq!(values.dtype(), DType::F32);
        assert_eq!(values.shape(), &[2]);
        let mut sorted = values.values().to_vec();
        sorted.sort_by(f64::total_cmp);
        close(sorted[0], 0.0, 1e-7);
        assert_eq!(sorted[1], 33_554_432.0);
    }
    for dtype in [DType::F32, DType::F64, DType::I64, DType::Bool] {
        for shape in [vec![0, 0], vec![2, 0, 0], vec![0, 2, 2], vec![2, 0, 3, 3]] {
            let input = UFuncArray::new(shape.clone(), vec![], dtype).unwrap();
            let (values, vectors) = input.eig().unwrap();
            let only = input.eigvals().unwrap();
            let expected_dtype = if dtype == DType::F32 {
                DType::F32
            } else {
                DType::F64
            };
            for output in [&values, &vectors, &only] {
                assert_eq!(output.dtype(), expected_dtype);
                assert!(output.values().is_empty());
                assert!(output.to_storage().unwrap().is_empty());
            }
            assert_eq!(values.shape(), &shape[..shape.len() - 1]);
            assert_eq!(only.shape(), values.shape());
            assert_eq!(vectors.shape(), shape);
        }
    }
}

#[test]
fn general_eigen_bridges_reject_invalid_shapes_nonfinite_values_and_unsupported_types() {
    for input in [
        UFuncArray::new(vec![], vec![1.0], DType::F64).unwrap(),
        UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap(),
        UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::F64).unwrap(),
        UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::F16).unwrap(),
        UFuncArray::new(vec![2, 2], vec![f64::NAN, 0.0, 0.0, 1.0], DType::F64).unwrap(),
        UFuncArray::new(vec![2, 2], vec![1.0, f64::INFINITY, 0.0, 1.0], DType::F64).unwrap(),
        matrix(2, 2, &[(1.0, 0.0); 4], DType::Complex128),
        UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::Complex128).unwrap(),
    ] {
        assert!(input.eig().is_err());
        assert!(input.eigvals().is_err());
    }
}
