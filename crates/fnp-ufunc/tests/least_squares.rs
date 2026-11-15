//! Array-level least-squares contracts, including complex and matrix RHS inputs.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::UFuncArray;
use serde_json::{Value, json};
use std::process::{Command, Stdio};

fn array(shape: &[usize], values: &[(f64, f64)], dtype: DType) -> UFuncArray {
    UFuncArray::from_storage(
        shape.to_vec(),
        ArrayStorage::Complex128(values.to_vec())
            .cast_to(dtype)
            .unwrap(),
    )
    .unwrap()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "actual {actual}, expected {expected}, tolerance {tolerance}"
    );
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::I8 => "int8",
        DType::I64 => "int64",
        DType::U64 => "uint64",
        DType::F16 => "float16",
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
        _ => panic!("unexpected test dtype {dtype:?}"),
    }
}

fn oracle_input(array: &UFuncArray) -> Value {
    let storage = array.to_storage().unwrap();
    let mut shape = array.shape().to_vec();
    if array.dtype().is_complex() {
        assert_eq!(shape.pop(), Some(2));
    }
    let values: Vec<_> = (0..storage.len())
        .map(|index| {
            if array.dtype().is_integer() || array.dtype() == DType::Bool {
                json!([storage.get_i128(index).unwrap().to_string(), "0"])
            } else {
                let (real, imaginary) = storage.get_complex128(index).unwrap();
                json!([real.to_string(), imaginary.to_string()])
            }
        })
        .collect();
    json!({"shape": shape, "dtype": dtype_name(array.dtype()), "values": values})
}

#[test]
fn matrix_right_hand_sides_return_independent_solutions_and_residuals() {
    // The third equation cannot be fitted. Each column has an independently
    // known solution and a nonzero residual, so transposing B or reusing the
    // first RHS cannot accidentally satisfy the assertion.
    let a = array(
        &[3, 2],
        &[
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ],
        DType::F64,
    );
    let b = array(
        &[3, 2],
        &[
            (2.0, 1.0),
            (-3.0, 2.0),
            (4.0, -2.0),
            (5.0, 1.0),
            (3.0, 4.0),
            (0.0, -2.0),
        ],
        DType::Complex128,
    );
    let (x, residuals, rank, spectrum) = a.lstsq(&b).unwrap();
    assert_eq!(x.shape(), &[2, 2, 2]);
    assert_eq!(x.dtype(), DType::Complex128);
    assert_eq!(rank, 2);
    assert_eq!(residuals.shape(), &[2]);
    assert_eq!(residuals.dtype(), DType::F64);
    for (&actual, expected) in x
        .values()
        .iter()
        .zip([2.0, 1.0, -3.0, 2.0, 4.0, -2.0, 5.0, 1.0])
    {
        close(actual, expected, 1.0e-12);
    }
    close(residuals.values()[0], 25.0, 1.0e-12);
    close(residuals.values()[1], 4.0, 1.0e-12);
    assert_eq!(spectrum.shape(), &[2]);
    for &value in spectrum.values() {
        close(value, 1.0, 1.0e-12);
    }
}

#[test]
fn underdetermined_complex_problem_returns_the_minimum_norm_solution() {
    let a = array(&[1, 2], &[(1.0, 0.0), (0.0, 1.0)], DType::Complex128);
    let b = array(&[1], &[(2.0, 2.0)], DType::Complex128);
    let (x, residuals, rank, spectrum) = a.lstsq(&b).unwrap();
    assert_eq!(x.shape(), &[2, 2]);
    for (&actual, expected) in x.values().iter().zip([1.0, 1.0, 1.0, -1.0]) {
        close(actual, expected, 1.0e-12);
    }
    assert_eq!(residuals.shape(), &[0]);
    assert_eq!(rank, 1);
    close(spectrum.values()[0], 2.0_f64.sqrt(), 1.0e-12);
}

#[test]
fn explicit_cutoff_controls_rank_without_changing_the_reported_spectrum() {
    for dtype in [DType::F64, DType::Complex128] {
        let a = array(
            &[2, 2],
            &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0e-8, 0.0)],
            dtype,
        );
        let b = array(&[2], &[(2.0, 0.0), (3.0e-8, 0.0)], dtype);
        let (full, _, full_rank, full_spectrum) = a.lstsq(&b).unwrap();
        let (truncated, _, rank, spectrum) = a.lstsq_with_rcond(&b, Some(1.0e-6)).unwrap();
        assert_eq!(full_rank, 2);
        assert_eq!(rank, 1);
        let step = if dtype.is_complex() { 2 } else { 1 };
        close(full.values()[step], 3.0, 1.0e-12);
        close(truncated.values()[0], 2.0, 1.0e-12);
        assert_eq!(truncated.values()[step], 0.0);
        assert_eq!(spectrum.values(), full_spectrum.values());
        assert!(a.lstsq_with_rcond(&b, Some(f64::NAN)).is_err());
    }
}

#[test]
fn least_squares_reads_declared_source_precision_before_widening() {
    for dtype in [DType::F32, DType::Complex64] {
        let (shape, values) = if dtype.is_complex() {
            (vec![1, 1, 2], vec![1.0 + 2.0_f64.powi(-24), 0.0])
        } else {
            (vec![1, 1], vec![1.0 + 2.0_f64.powi(-24)])
        };
        let a = UFuncArray::new(shape, values, dtype).unwrap();
        let b = UFuncArray::from_vec(vec![1.0]);
        let (x, _, _, _) = a.lstsq(&b).unwrap();
        assert_eq!(x.values()[0], 1.0);
        assert_eq!(
            x.dtype(),
            if dtype.is_complex() {
                DType::Complex128
            } else {
                DType::F64
            }
        );
    }
}

#[test]
fn least_squares_rejects_invalid_logical_shapes_and_unsupported_dtypes() {
    let a = UFuncArray::from_2d(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
    for shape in [vec![], vec![1], vec![1, 2], vec![2, 1, 1]] {
        let count: usize = shape.iter().product();
        let b = UFuncArray::new(shape, vec![1.0; count], DType::F64).unwrap();
        assert!(a.lstsq(&b).is_err());
    }
    let b = UFuncArray::from_vec(vec![1.0, 2.0]);
    assert!(b.lstsq(&b).is_err());
    assert!(a.astype(DType::F16).lstsq(&b).is_err());
    assert!(a.lstsq(&b.astype(DType::F16)).is_err());
    let malformed = UFuncArray::new(vec![2, 3], vec![0.0; 6], DType::Complex128).unwrap();
    assert!(malformed.lstsq(&b).is_err());
    let nonfinite = UFuncArray::from_vec(vec![f64::INFINITY, 1.0]);
    assert!(a.lstsq(&nonfinite).is_err());
}

#[test]
fn real_and_complex_least_squares_match_live_numpy() {
    let mut cases = Vec::new();
    let mut results = Vec::new();
    let dtype_pairs = [
        (DType::F32, DType::F32),
        (DType::F64, DType::F64),
        (DType::Complex64, DType::Complex64),
        (DType::Complex128, DType::Complex128),
        (DType::F32, DType::Complex64),
        (DType::Complex64, DType::F64),
        (DType::Complex64, DType::I8),
        (DType::I64, DType::F32),
    ];
    for (a_dtype, b_dtype) in dtype_pairs {
        for (m, n) in [(4, 2), (2, 4), (3, 3), (0, 3), (3, 0), (0, 0)] {
            let a_values: Vec<_> = (0..m)
                .flat_map(|row| {
                    (0..n).map(move |column| {
                        (
                            if row == column {
                                4.0
                            } else {
                                ((row + 2 * column) % 3) as f64 - 1.0
                            },
                            ((2 * row + column) % 3) as f64 - 1.0,
                        )
                    })
                })
                .collect();
            let a = array(&[m, n], &a_values, a_dtype);
            for columns in [None, Some(0), Some(1), Some(3)] {
                let nrhs = columns.unwrap_or(1);
                let b_values: Vec<_> = (0..m * nrhs)
                    .map(|index| {
                        (
                            ((3 * index + 1) % 7) as f64 - 3.0,
                            ((index + 2) % 5) as f64 - 2.0,
                        )
                    })
                    .collect();
                let shape = if columns.is_some() {
                    vec![m, nrhs]
                } else {
                    vec![m]
                };
                let b = array(&shape, &b_values, b_dtype);
                cases.push(json!({"a": oracle_input(&a), "b": oracle_input(&b), "rcond": null}));
                results.push(a.lstsq(&b).unwrap());
            }
        }
        // Deliberately rank-deficient system, with and without an explicit
        // cutoff, checks that residuals remain empty even when m > n.
        let a = array(
            &[3, 2],
            &[
                (1.0, 1.0),
                (2.0, 2.0),
                (2.0, 2.0),
                (4.0, 4.0),
                (3.0, 3.0),
                (6.0, 6.0),
            ],
            a_dtype,
        );
        let b = array(&[3], &[(2.0, 1.0), (1.0, -1.0), (3.0, 2.0)], b_dtype);
        for rcond in [None, Some(0.1)] {
            cases.push(json!({"a": oracle_input(&a), "b": oracle_input(&b), "rcond": rcond}));
            results.push(a.lstsq_with_rcond(&b, rcond).unwrap());
        }
        // A dense rank-one matrix's rounding-sized trailing singular value can
        // fall on either side of the explicit -1 cutoff in different SVD
        // backends. Use a diagonal spectrum to distinguish the machine cutoff
        // from the dimension-scaled default without that numerical ambiguity.
        let a = array(
            &[2, 2],
            &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (2.0e-16, 0.0)],
            a_dtype,
        );
        let b = array(&[2], &[(1.0, 0.0), (2.0, 0.0)], b_dtype);
        for rcond in [None, Some(-1.0), Some(1.0e-12)] {
            cases.push(json!({"a": oracle_input(&a), "b": oracle_input(&b), "rcond": rcond}));
            results.push(a.lstsq_with_rcond(&b, rcond).unwrap());
        }
    }
    let script = r#"
import json, sys
import numpy as np

def load(item):
    dtype = np.dtype(item['dtype'])
    if dtype.kind == 'c':
        values = [complex(float(real), float(imag)) for real, imag in item['values']]
    else:
        convert = int if dtype.kind in 'biu' else float
        values = [convert(real) for real, imag in item['values']]
    return np.array(values, dtype=dtype).reshape(item['shape'])

def pack(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                values=[[float(x.real), float(x.imag)] for x in array.ravel()])

rows = []
for item in json.load(sys.stdin):
    x, residuals, rank, spectrum = np.linalg.lstsq(load(item['a']), load(item['b']), rcond=item['rcond'])
    rows.append(dict(x=pack(x), residuals=pack(residuals), rank=int(rank), spectrum=pack(spectrum)))
json.dump(rows, sys.stdout, allow_nan=False)
"#;
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let mut child = Command::new(python)
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("NumPy oracle must start");
    serde_json::to_writer(child.stdin.take().unwrap(), &cases).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(expected.len(), results.len());
    for (case_index, ((x, residuals, rank, spectrum), expected)) in
        results.iter().zip(expected).enumerate()
    {
        assert_eq!(
            *rank,
            expected["rank"].as_u64().unwrap() as usize,
            "case {case_index}: {}",
            cases[case_index]
        );
        for (name, actual) in [("x", x), ("residuals", residuals), ("spectrum", spectrum)] {
            let expected = &expected[name];
            let mut shape = actual.shape().to_vec();
            if actual.dtype().is_complex() {
                assert_eq!(shape.pop(), Some(2));
            }
            assert_eq!(json!(shape), expected["shape"], "case {case_index} {name}");
            assert_eq!(
                dtype_name(actual.dtype()),
                expected["dtype"].as_str().unwrap(),
                "case {case_index} {name}"
            );
            let storage = actual.to_storage().unwrap();
            let values = expected["values"].as_array().unwrap();
            assert_eq!(storage.len(), values.len());
            let tolerance = if matches!(actual.dtype(), DType::F32 | DType::Complex64) {
                8.0e-6
            } else {
                2.0e-10
            };
            for (index, expected) in values.iter().enumerate() {
                let (real, imag) = storage.get_complex128(index).unwrap();
                close(real, expected[0].as_f64().unwrap(), tolerance);
                close(imag, expected[1].as_f64().unwrap(), tolerance);
            }
        }
    }
}
