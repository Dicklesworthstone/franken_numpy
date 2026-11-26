//! Typed matrix and tensor factorizations, including selected-triangle rules.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{UFuncArray, UFuncError};
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
        DType::F16 => "float16",
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
        _ => panic!("unexpected factorization dtype {dtype:?}"),
    }
}

fn input(shape: &[usize], dtype: DType, values: Vec<f64>) -> UFuncArray {
    let mut physical_shape = shape.to_vec();
    if dtype.is_complex() {
        physical_shape.push(2);
    }
    UFuncArray::new(physical_shape, values, dtype).unwrap()
}

fn record(array: &UFuncArray) -> Value {
    let mut shape = array.shape().to_vec();
    if array.dtype().is_complex() {
        assert_eq!(shape.pop(), Some(2));
    }
    // Serialize actual typed storage, including exact wide integers. The
    // caller may deliberately leave extra precision in raw working cells.
    let storage = array.to_storage().unwrap();
    assert_eq!(storage.len(), shape.iter().product::<usize>());
    let values: Vec<String> = match storage {
        ArrayStorage::I64(values) => values.iter().map(ToString::to_string).collect(),
        ArrayStorage::U64(values) => values.iter().map(ToString::to_string).collect(),
        ArrayStorage::Complex64(values) => values
            .into_iter()
            .flat_map(|(real, imaginary)| [f64::from(real), f64::from(imaginary)])
            .map(|value| value.to_string())
            .collect(),
        ArrayStorage::Complex128(values) => values
            .into_iter()
            .flat_map(|(real, imaginary)| [real, imaginary])
            .map(|value| value.to_string())
            .collect(),
        storage => storage
            .to_f64_vec()
            .iter()
            .map(ToString::to_string)
            .collect(),
    };
    json!({"shape": shape, "dtype": dtype_name(array.dtype()), "values": values})
}

fn output_record(result: Result<UFuncArray, UFuncError>) -> Value {
    match result {
        Ok(array) => {
            if matches!(array.dtype(), DType::F32 | DType::Complex64) {
                for &value in array.values() {
                    assert!(value.is_nan() || value.to_bits() == f64::from(value as f32).to_bits());
                }
            }
            record(&array)
        }
        Err(error) => json!({"error": error.to_string()}),
    }
}

fn case(source: &UFuncArray, operation: &str, upper: bool) -> Value {
    let result = if operation == "inv" {
        source.inv()
    } else {
        source.cholesky_with_upper(upper)
    };
    json!({
        "source": record(source), "operation": operation,
        "upper": upper, "actual": output_record(result)
    })
}

fn tensor_solve_case(source: &UFuncArray, rhs: &UFuncArray, axes: Option<&[usize]>) -> Value {
    let result = match axes {
        Some(axes) => source.tensorsolve_with_axes(rhs, axes),
        None => source.tensorsolve(rhs),
    };
    json!({
        "source": record(source), "rhs": record(rhs), "operation": "tensorsolve",
        "axes": axes, "actual": output_record(result)
    })
}

fn tensor_inverse_case(source: &UFuncArray, ind: usize) -> Value {
    json!({
        "source": record(source), "operation": "tensorinv", "ind": ind,
        "actual": output_record(source.tensorinv(ind))
    })
}

fn oracle(cases: &[Value]) {
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let script = r#"
import json, sys, warnings
import numpy as np

def array(record):
    dtype = np.dtype(record['dtype'])
    if dtype.kind in 'iu':
        values = np.array([int(value) for value in record['values']], dtype=dtype)
    else:
        values = np.array([float(value) for value in record['values']], dtype=np.float64)
        if dtype.kind == 'c':
            values = values.view(np.complex128)
        values = values.astype(dtype)
    return values.reshape(record['shape'])

cases = json.load(sys.stdin)
for case in cases:
    source = array(case['source'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            if case['operation'] == 'tensorsolve':
                expected = np.linalg.tensorsolve(source, array(case['rhs']), axes=case['axes'])
            elif case['operation'] == 'tensorinv':
                expected = np.linalg.tensorinv(source, ind=case['ind'])
            elif case['operation'] == 'inv':
                expected = np.linalg.inv(source)
            else:
                expected = np.linalg.cholesky(source, upper=case['upper'])
        except (ValueError, TypeError, np.linalg.LinAlgError) as error:
            assert 'error' in case['actual'], (case, type(error).__name__, str(error))
            continue
    assert 'error' not in case['actual'], case
    actual = array(case['actual'])
    assert actual.shape == expected.shape, (case, expected.shape)
    assert actual.dtype == expected.dtype, (case, expected.dtype)
    if not actual.size:
        continue
    if case['operation'] in ('tensorsolve', 'tensorinv'):
        precision = np.float32 if actual.dtype in (np.float32, np.complex64) else np.float64
        tolerance = 128 * max(source.size, 1) * np.finfo(precision).eps
        scale = max(float(np.max(np.abs(expected))), 1)
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * scale)
        if case['operation'] == 'tensorsolve':
            rhs = array(case['rhs']).ravel()
            if case['axes'] is not None:
                permutation = list(range(source.ndim))
                for axis in case['axes']:
                    permutation.remove(axis)
                    permutation.append(axis)
                source = source.transpose(permutation)
            matrix = source.reshape(rhs.size, rhs.size)
            residual = matrix @ actual.ravel() - rhs
            bound = tolerance * max(np.linalg.norm(matrix) * np.linalg.norm(actual) +
                                    np.linalg.norm(rhs), 1)
        else:
            n = int(np.sqrt(source.size))
            matrix, inverse = source.reshape(n, n), actual.reshape(n, n)
            residual = matrix @ inverse - np.eye(n)
            bound = tolerance * max(np.linalg.norm(matrix) * np.linalg.norm(inverse), 1)
        assert np.linalg.norm(residual) <= bound, case
        continue
    n = source.shape[-1]
    precision = np.float32 if actual.dtype in (np.float32, np.complex64) else np.float64
    tolerance = 128 * max(n, 1) * np.finfo(precision).eps
    scale = np.max(np.abs(expected), axis=(-2, -1), keepdims=True)
    assert np.all(np.isfinite(actual)), case
    assert np.all(np.abs(actual - expected) <= tolerance * scale), case
    working_dtype = np.complex128 if np.iscomplexobj(actual) else np.float64
    source, actual = [value.astype(working_dtype) for value in (source, actual)]
    if case['operation'] == 'inv':
        bound = tolerance * np.maximum(np.linalg.norm(source, axis=(-2, -1)) *
                                       np.linalg.norm(actual, axis=(-2, -1)), 1)
        identity = np.eye(n)
        for residual in (source @ actual - identity, actual @ source - identity):
            assert np.all(np.linalg.norm(residual, axis=(-2, -1)) <= bound), case
    else:
        if case['upper']:
            selected = np.triu(source) + np.swapaxes(np.triu(source, 1).conj(), -2, -1)
            reconstructed = np.swapaxes(actual.conj(), -2, -1) @ actual
            np.testing.assert_array_equal(np.tril(actual, -1), np.zeros_like(actual))
        else:
            selected = np.tril(source) + np.swapaxes(np.tril(source, -1).conj(), -2, -1)
            reconstructed = actual @ np.swapaxes(actual.conj(), -2, -1)
            np.testing.assert_array_equal(np.triu(actual, 1), np.zeros_like(actual))
        diagonal = np.arange(n)
        selected[..., diagonal, diagonal] = source[..., diagonal, diagonal].real
        bound = tolerance * np.linalg.norm(selected, axis=(-2, -1))
        assert np.all(np.linalg.norm(reconstructed - selected, axis=(-2, -1)) <= bound), case
print(f'NumPy {np.__version__}: {len(cases)} matrix/tensor factorization cases passed')
"#;
    let mut child = Command::new(python)
        .args(["-c", script])
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("NumPy oracle interpreter must be available");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(cases).unwrap())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "NumPy factorization oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    eprintln!("{}", String::from_utf8_lossy(&output.stdout));
}

#[test]
fn typed_tensor_solve_and_inverse_match_numpy() {
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
        let mut values = Vec::new();
        for row in 0..6 {
            for column in 0..6 {
                let real = if dtype == DType::Bool {
                    if row == column { 1.0 } else { 0.0 }
                } else if row == column {
                    (12 + row) as f64
                } else if dtype.is_integer() {
                    if column > row { 1.0 } else { 0.0 }
                } else {
                    (row + column + 1) as f64 / 13.0
                };
                values.push(real);
                if dtype.is_complex() {
                    values.push((row as f64 - column as f64 + 1.0) / 11.0);
                }
            }
        }
        for shape in [vec![2, 3, 6], vec![2, 3, 3, 2]] {
            let source = input(&shape, dtype, values.clone());
            cases.push(tensor_inverse_case(&source, 2));
            for rhs_dtype in [
                DType::I64,
                DType::F32,
                DType::F64,
                DType::Complex64,
                DType::Complex128,
            ] {
                let rhs_values = (0..6)
                    .flat_map(|i| {
                        let real = (i + 1) as f64;
                        if rhs_dtype.is_complex() {
                            vec![real, (i as f64 - 2.0) / 7.0]
                        } else {
                            vec![real]
                        }
                    })
                    .collect();
                // Equal products with different leading shapes are legal:
                // NumPy flattens this 3-by-2 RHS for the 2-by-3 tensor prefix.
                let rhs = input(&[3, 2], rhs_dtype, rhs_values);
                cases.push(tensor_solve_case(&source, &rhs, None));
            }
        }
    }
    assert_eq!(cases.len(), 156);
    oracle(&cases);
}

#[test]
fn tensor_solve_axes_move_logical_complex_axes_and_preserve_exact_sources() {
    let mut cases = Vec::new();
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        let mut values = Vec::new();
        for row in 0..6 {
            for column in 0..6 {
                values.push(if row == column { 3.0 + row as f64 } else { 0.0 });
                if dtype.is_complex() {
                    values.push(if row == column { 0.75 } else { 0.0 });
                }
            }
        }
        let source = input(&[2, 3, 2, 3], dtype, values);
        let rhs = input(&[3, 2], DType::F64, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        for axes in [vec![], vec![0, 1], vec![0, 0, 1], vec![2, 3], vec![4]] {
            cases.push(tensor_solve_case(&source, &rhs, Some(&axes)));
        }
    }

    let source = UFuncArray::from_storage(
        vec![2, 1, 2],
        ArrayStorage::U64(vec![9_007_199_254_740_993, 1, 0, 18_446_744_073_709_550_591]),
    )
    .unwrap();
    let rhs = UFuncArray::from_storage(
        vec![1, 2],
        ArrayStorage::I64(vec![9_007_199_254_740_995, -9_007_199_254_740_997]),
    )
    .unwrap();
    cases.push(tensor_solve_case(&source, &rhs, None));
    cases.push(tensor_inverse_case(&source, 1));

    // These raw float64 working cells become exactly singular when read
    // through their declared float32/complex64 storage. Widening the raw
    // cells instead would silently solve a different, nonsingular matrix.
    for dtype in [DType::F32, DType::Complex64] {
        let near_one = 1.0 + 2.0_f64.powi(-26);
        let mut values = Vec::new();
        for real in [near_one, 1.0, 1.0, near_one] {
            values.push(real);
            if dtype.is_complex() {
                values.push(0.0);
            }
        }
        let source = input(&[2, 2], dtype, values);
        let rhs = input(&[2], DType::F32, vec![1.0, 2.0]);
        cases.push(tensor_solve_case(&source, &rhs, None));
        cases.push(tensor_inverse_case(&source, 1));
    }
    oracle(&cases);
}

#[test]
fn tensor_factorization_empty_scalar_singleton_and_error_contracts_match_numpy() {
    let mut cases = Vec::new();
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        let value = if dtype.is_complex() {
            vec![2.0, 1.0]
        } else {
            vec![2.0]
        };
        for shape in [vec![], vec![1], vec![1, 1]] {
            let source = input(&shape, dtype, value.clone());
            for rhs_shape in [vec![], vec![1], vec![1, 1], vec![1, 1, 1, 1, 1]] {
                let rhs = input(&rhs_shape, DType::F32, vec![3.0]);
                cases.push(tensor_solve_case(&source, &rhs, None));
            }
            for ind in [0, 1, 2, 8] {
                cases.push(tensor_inverse_case(&source, ind));
            }
        }
        for (shape, rhs_shape) in [(vec![0, 0], vec![0]), (vec![0, 2, 0], vec![0, 2])] {
            let source = input(&shape, dtype, Vec::new());
            let rhs = input(&rhs_shape, dtype, Vec::new());
            cases.push(tensor_solve_case(&source, &rhs, None));
            for ind in [1, 2, 4] {
                cases.push(tensor_inverse_case(&source, ind));
            }
        }
    }
    for dtype in [DType::F16, DType::F32, DType::Complex128] {
        let values = if dtype.is_complex() {
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        } else {
            vec![1.0, 0.0, 0.0, 1.0]
        };
        let source = input(&[2, 2], dtype, values);
        cases.push(tensor_inverse_case(&source, 1));
        for rhs_dtype in [DType::F16, DType::F64] {
            let rhs = input(&[2], rhs_dtype, vec![1.0, 2.0]);
            cases.push(tensor_solve_case(&source, &rhs, None));
        }
        let wrong_rhs = input(&[3], DType::F64, vec![1.0, 2.0, 3.0]);
        cases.push(tensor_solve_case(&source, &wrong_rhs, None));
    }
    oracle(&cases);

    let malformed = UFuncArray::new(vec![2, 3], vec![0.0; 6], DType::Complex128).unwrap();
    let rhs = input(&[2], DType::F64, vec![1.0, 2.0]);
    assert!(malformed.tensorsolve(&rhs).is_err());
    assert!(malformed.tensorsolve_with_axes(&rhs, &[0]).is_err());
    assert!(malformed.tensorinv(1).is_err());
}

#[test]
fn typed_inverse_and_cholesky_stacks_match_numpy() {
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
        for n in [1, 2, 3, 5, 16] {
            for batch in [vec![], vec![2], vec![2, 1, 2]] {
                let mut shape = batch.clone();
                shape.extend([n, n]);
                let mut values = Vec::new();
                for matrix in 0..batch.iter().product::<usize>() {
                    for row in 0..n {
                        for column in 0..n {
                            let real = if dtype == DType::Bool {
                                if row == column { 1.0 } else { 0.0 }
                            } else if row == column {
                                (4 * n + matrix + row) as f64
                            } else if dtype.is_integer() {
                                1.0
                            } else {
                                ((row + column + matrix) % 3) as f64 / 7.0 + 0.125
                            };
                            values.push(real);
                            if dtype.is_complex() {
                                values.push((row as f64 - column as f64) / (4 * n) as f64);
                            }
                        }
                    }
                }
                // Raw F32/complex64 cells are intentionally not pre-rounded.
                let source = input(&shape, dtype, values);
                cases.push(case(&source, "inv", false));
                cases.push(case(&source, "cholesky", false));
                cases.push(case(&source, "cholesky", true));
            }
        }
    }
    oracle(&cases);
}

#[test]
fn inverse_handles_nonsymmetric_complex_and_real_batches() {
    let mut cases = Vec::new();
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        let mut values = Vec::new();
        for batch in 0..4 {
            for row in 0..3 {
                for column in 0..3 {
                    values.push(if row == column {
                        (4 + batch + row) as f64
                    } else {
                        (2 * row + column) as f64 / 9.0 - 0.25
                    });
                    if dtype.is_complex() {
                        values.push((3 * row + column + batch) as f64 / 13.0 - 0.125);
                    }
                }
            }
        }
        let source = input(&[2, 1, 2, 3, 3], dtype, values);
        cases.push(case(&source, "inv", false));
    }
    oracle(&cases);
}

#[test]
fn cholesky_ignores_unselected_nonfinite_cells_and_imaginary_diagonals() {
    let mut cases = Vec::new();
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        for upper in [false, true] {
            let mut values = Vec::new();
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for row in 0..3 {
                    for column in 0..3 {
                        let unused = if upper { row > column } else { column > row };
                        values.push(if unused {
                            invalid
                        } else if row == column {
                            6.0 + row as f64
                        } else {
                            0.5
                        });
                        if dtype.is_complex() {
                            values.push(if unused || row == column {
                                invalid
                            } else {
                                (row as f64 - column as f64) / 4.0
                            });
                        }
                    }
                }
            }
            let source = input(&[3, 3, 3], dtype, values);
            assert!(source.cholesky_with_upper(upper).is_ok());
            cases.push(case(&source, "cholesky", upper));
        }
    }
    oracle(&cases);
}

#[test]
fn factorizations_use_declared_source_precision_before_deciding_singularity() {
    let mut cases = Vec::new();
    for dtype in [DType::F32, DType::Complex64] {
        let real = [1.0, 1.0, 1.0, 1.0 + 1.0e-8];
        let values = if dtype.is_complex() {
            real.into_iter().flat_map(|value| [value, 0.0]).collect()
        } else {
            real.to_vec()
        };
        let source = input(&[2, 2], dtype, values);
        // The raw f64 matrix is positive definite, but its declared storage
        // rounds to a singular matrix. Computing before rounding is wrong.
        assert!(source.inv().is_err());
        assert!(source.cholesky().is_err());
        assert!(source.cholesky_with_upper(true).is_err());
        cases.push(case(&source, "inv", false));
        cases.push(case(&source, "cholesky", false));
        cases.push(case(&source, "cholesky", true));
    }
    oracle(&cases);
}

#[test]
fn factorizations_preserve_exact_integer_sources_and_input_storage() {
    let mut cases = Vec::new();
    let sources = [
        UFuncArray::from_storage(
            vec![2, 2, 2],
            ArrayStorage::I64(vec![(1i64 << 53) + 1, 1, 1, 3, i64::MAX, 0, 0, 7]),
        )
        .unwrap(),
        UFuncArray::from_storage(
            vec![2, 2, 2],
            ArrayStorage::U64(vec![u64::MAX, 1, 1, 5, (1u64 << 53) + 3, 0, 0, 11]),
        )
        .unwrap(),
    ];
    for source in sources {
        let before = source.to_storage().unwrap();
        cases.push(case(&source, "inv", false));
        cases.push(case(&source, "cholesky", false));
        cases.push(case(&source, "cholesky", true));
        assert_eq!(source.to_storage().unwrap(), before);
    }
    oracle(&cases);
}

#[test]
fn factorizations_preserve_empty_stacks_and_do_not_multiply_unoccupied_extents() {
    let mut cases = Vec::new();
    for dtype in [
        DType::Bool,
        DType::I64,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for shape in [
            vec![0, 0],
            vec![3, 0, 0],
            vec![0, 2, 2],
            vec![2, 0, 3, 3],
            vec![0, usize::MAX, usize::MAX],
            vec![usize::MAX, 0, 0],
        ] {
            let source = input(&shape, dtype, Vec::new());
            let expected_dtype = match dtype {
                DType::Bool | DType::I64 => DType::F64,
                other => other,
            };
            for result in [
                source.inv().unwrap(),
                source.cholesky().unwrap(),
                source.cholesky_with_upper(true).unwrap(),
            ] {
                assert_eq!(result.shape(), source.shape());
                assert_eq!(result.dtype(), expected_dtype);
                assert!(result.values().is_empty());
                assert!(result.to_storage().unwrap().is_empty());
            }
            if !shape.contains(&usize::MAX) {
                cases.push(case(&source, "inv", false));
                cases.push(case(&source, "cholesky", false));
                cases.push(case(&source, "cholesky", true));
            }
        }
    }
    oracle(&cases);
}

#[test]
fn factorizations_reject_invalid_layouts_dtypes_and_bad_later_batches() {
    for source in [
        UFuncArray::new(vec![], vec![1.0], DType::F64).unwrap(),
        UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap(),
        UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::F64).unwrap(),
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
        assert!(source.inv().is_err());
        assert!(source.cholesky().is_err());
        assert!(source.cholesky_with_upper(true).is_err());
    }
    let mut cases = Vec::new();
    for dtype in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for diagonal in [0.0, -1.0] {
            let real = [2.0, 0.0, 0.0, 3.0, diagonal, 0.0, 0.0, 4.0];
            let values = if dtype.is_complex() {
                real.into_iter().flat_map(|value| [value, 0.0]).collect()
            } else {
                real.to_vec()
            };
            let source = input(&[2, 2, 2], dtype, values);
            let before = source.values().to_vec();
            assert!(source.cholesky().is_err());
            assert!(source.cholesky_with_upper(true).is_err());
            if diagonal == 0.0 || dtype == DType::F16 {
                assert!(source.inv().is_err());
            }
            cases.push(case(&source, "inv", false));
            cases.push(case(&source, "cholesky", false));
            cases.push(case(&source, "cholesky", true));
            assert_eq!(source.values(), before);
        }
    }
    oracle(&cases);
}
