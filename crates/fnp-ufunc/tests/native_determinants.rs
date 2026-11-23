//! Typed scalar and stacked determinants, with a live NumPy behavioral oracle.

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
        _ => panic!("unexpected determinant dtype {dtype:?}"),
    }
}

fn input(shape: &[usize], values: Vec<f64>, dtype: DType) -> UFuncArray {
    let mut physical_shape = shape.to_vec();
    if dtype.is_complex() {
        physical_shape.push(2);
    }
    UFuncArray::new(physical_shape, values, dtype).unwrap()
}

fn output_record(array: &UFuncArray) -> Value {
    let mut shape = array.shape().to_vec();
    if array.dtype().is_complex() {
        shape.pop();
    }
    if matches!(array.dtype(), DType::F32 | DType::Complex64) {
        for &value in array.values() {
            assert!(
                value.is_nan() || value == f64::from(value as f32),
                "output must already have its declared single precision"
            );
        }
    }
    json!({
        "shape": shape,
        "dtype": dtype_name(array.dtype()),
        "values": array.values().iter().map(f64::to_string).collect::<Vec<_>>()
    })
}

fn case(array: &UFuncArray) -> Value {
    let mut shape = array.shape().to_vec();
    if array.dtype().is_complex() {
        shape.pop();
    }
    let mut exact_integers = None;
    if matches!(array.dtype(), DType::I64 | DType::U64) {
        exact_integers = Some(match array.to_storage().unwrap() {
            ArrayStorage::I64(values) => values.iter().map(i64::to_string).collect::<Vec<_>>(),
            ArrayStorage::U64(values) => values.iter().map(u64::to_string).collect::<Vec<_>>(),
            _ => unreachable!(),
        });
    }
    let determinant = match array.det_array() {
        Ok(result) => output_record(&result),
        Err(error) => json!({"error": error.to_string()}),
    };
    let slogdet = match array.slogdet_array() {
        Ok((sign, logarithm)) => json!([output_record(&sign), output_record(&logarithm)]),
        Err(error) => json!({"error": error.to_string()}),
    };
    json!({
        "shape": shape,
        "dtype": dtype_name(array.dtype()),
        "values": array.values().iter().map(f64::to_string).collect::<Vec<_>>(),
        "exact_integers": exact_integers,
        "det": determinant,
        "slogdet": slogdet
    })
}

fn check_numpy(cases: &[Value]) {
    let script = r#"
import json, sys, warnings
import numpy as np

def check_output(actual, expected, context):
    expected = np.asarray(expected)
    assert 'error' not in actual, (context, actual)
    assert actual['shape'] == list(expected.shape), (context, actual, expected.shape)
    assert actual['dtype'] == expected.dtype.name, (context, actual, expected.dtype)
    values = np.array([float(value) for value in actual['values']], dtype=np.float64)
    if expected.dtype.kind == 'c':
        values = values.view(np.complex128)
    values = values.reshape(expected.shape)
    precision = np.empty((), dtype=expected.dtype).real.dtype
    tolerance = 64 * np.finfo(precision).eps
    np.testing.assert_allclose(values, expected, rtol=tolerance, atol=tolerance,
                               err_msg=str(context), equal_nan=True)

cases = json.load(sys.stdin)
for index, case in enumerate(cases):
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        if case['exact_integers'] is not None:
            source = np.array([int(value) for value in case['exact_integers']], dtype=case['dtype'])
        else:
            source = np.array([float(value) for value in case['values']], dtype=np.float64)
            if case['dtype'].startswith('complex'):
                source = source.view(np.complex128)
            source = source.astype(case['dtype'])
        source = source.reshape(case['shape'])
        for operation in ('det', 'slogdet'):
            context = (index, case['dtype'], case['shape'], operation)
            try:
                expected = getattr(np.linalg, operation)(source)
            except Exception:
                assert 'error' in case[operation], (context, case[operation])
            else:
                if operation == 'det':
                    check_output(case[operation], expected, context)
                else:
                    assert not isinstance(case[operation], dict), (context, case[operation])
                    for actual_part, expected_part in zip(case[operation], expected):
                        check_output(actual_part, expected_part, context)
print(len(cases))
"#;
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let mut child = Command::new(python)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("native determinant parity requires a working NumPy oracle");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(cases).unwrap().as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "NumPy determinant oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        cases.len().to_string()
    );
}

#[test]
fn determinants_match_numpy_for_numeric_dtypes_and_matrix_stacks() {
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
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for n in [0, 1, 2, 3, 5] {
            for batch_shape in [&[][..], &[2], &[2, 3], &[0], &[2, 0]] {
                let mut shape = batch_shape.to_vec();
                shape.extend([n, n]);
                let batches = batch_shape.iter().product::<usize>();
                let mut values = Vec::new();
                for lane in 0..batches {
                    for row in 0..n {
                        for column in 0..n {
                            let real = if row == column {
                                (n * 3 + lane + 4) as f64
                            } else {
                                ((row * 7 + column * 11 + lane) % 3) as f64
                            };
                            values.push(if dtype.is_float() || dtype.is_complex() {
                                real + 0.1250000001
                            } else {
                                real
                            });
                            if dtype.is_complex() {
                                values.push(
                                    ((row * 5 + column * 13 + lane) % 5) as f64 - 2.3125000003,
                                );
                            }
                        }
                    }
                }
                cases.push(case(&input(&shape, values, dtype)));
            }
        }
    }
    check_numpy(&cases);
}

#[test]
fn determinants_preserve_singular_lanes_and_complex_phases() {
    let mut cases = Vec::new();
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        let complex = dtype.is_complex();
        let real_values = [
            1.0, 2.0, 1.0, 2.0, // singular
            0.0, 2.0, 3.0, 0.0, // odd permutation
            2.0, 1.0, 0.0, 3.0, // positive determinant
        ];
        let values = real_values
            .iter()
            .flat_map(|&real| {
                if complex {
                    vec![real, 0.5 * real]
                } else {
                    vec![real]
                }
            })
            .collect();
        let array = input(&[3, 2, 2], values, dtype);
        let determinant = array.det_array().unwrap();
        let (sign, logarithm) = array.slogdet_array().unwrap();
        assert_eq!(determinant.values()[0], 0.0);
        assert_eq!(sign.values()[0], 0.0);
        assert_eq!(logarithm.values()[0], f64::NEG_INFINITY);
        if complex {
            assert_eq!(determinant.values()[1], 0.0);
            assert_eq!(sign.values()[1], 0.0);
            assert_eq!(logarithm.shape(), &[3]);
            assert_ne!(sign.values()[3], 0.0, "phase must retain imaginary part");
        }
        cases.push(case(&array));
    }
    check_numpy(&cases);
}

#[test]
fn determinant_source_precision_and_scalar_conveniences_match_numpy() {
    let mut cases = Vec::new();
    for (dtype, raw, expected) in [
        (DType::Bool, 7.0, 1.0),
        (DType::I8, 255.0, -1.0),
        (DType::U8, 257.0, 1.0),
        (DType::F32, 1.0 + f64::from(f32::EPSILON) / 4.0, 1.0),
        (DType::F64, 1.125, 1.125),
    ] {
        let array = input(&[1, 1], vec![raw], dtype);
        assert_eq!(array.det().unwrap(), expected);
        let (sign, logarithm) = array.slogdet().unwrap();
        assert_eq!(sign, expected.signum());
        assert!((logarithm - expected.abs().ln()).abs() < 1e-12);
        cases.push(case(&array));
    }
    for dtype in [DType::Complex64, DType::Complex128] {
        let raw = [1.0 + f64::from(f32::EPSILON) / 4.0, -0.75];
        let array = input(&[1, 1], raw.to_vec(), dtype);
        let output = array.det_array().unwrap();
        let expected_real = if dtype == DType::Complex64 {
            1.0
        } else {
            raw[0]
        };
        for (&actual, expected) in output.values().iter().zip([expected_real, raw[1]]) {
            assert!((actual - expected).abs() <= 4.0 * f64::EPSILON);
        }
        assert_eq!(output.shape(), &[2]);
        assert!(array.det().is_err());
        assert!(array.slogdet().is_err());
        cases.push(case(&array));
    }
    for array in [
        UFuncArray::from_storage(vec![2, 2], ArrayStorage::I64(vec![i64::MAX, 1, 0, -3])).unwrap(),
        UFuncArray::from_storage(vec![2, 2], ArrayStorage::U64(vec![u64::MAX, 7, 0, 3])).unwrap(),
    ] {
        cases.push(case(&array));
    }
    check_numpy(&cases);
}

#[test]
fn empty_determinants_distinguish_empty_cores_from_empty_batches() {
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        let empty = input(&[2, 3, 0, 0], Vec::new(), dtype);
        let determinant = empty.det_array().unwrap();
        let (sign, logarithm) = empty.slogdet_array().unwrap();
        let components = if dtype.is_complex() { 2 } else { 1 };
        assert_eq!(determinant.values().len(), 6 * components);
        assert_eq!(logarithm.shape(), &[2, 3]);
        assert_eq!(logarithm.values(), &[0.0; 6]);
        for output in [&determinant, &sign] {
            for value in output.values().chunks_exact(components) {
                assert_eq!(value[0], 1.0);
                if dtype.is_complex() {
                    assert_eq!(value[1], 0.0);
                }
            }
        }
        // Never multiply the enormous unoccupied core dimensions.
        let vacant = input(&[0, usize::MAX, usize::MAX], Vec::new(), dtype);
        assert!(vacant.det_array().unwrap().values().is_empty());
        let (sign, logarithm) = vacant.slogdet_array().unwrap();
        assert!(sign.values().is_empty());
        assert!(logarithm.values().is_empty());
    }
}

#[test]
fn determinant_rejects_bad_shapes_dtypes_and_unrepresentable_outputs() {
    for shape in [&[][..], &[3], &[2, 3], &[0, 2, 3]] {
        for dtype in [DType::F64, DType::Complex128] {
            let components = if dtype.is_complex() { 2 } else { 1 };
            let count = shape.iter().product::<usize>() * components;
            let array = input(shape, vec![1.0; count], dtype);
            assert!(array.det_array().is_err());
            assert!(array.slogdet_array().is_err());
        }
    }
    for dtype in [DType::F16, DType::DateTime64, DType::TimeDelta64] {
        for shape in [&[2, 2][..], &[0, 2, 2], &[0, 0]] {
            let array = input(shape, vec![0.0; shape.iter().product()], dtype);
            assert!(array.det_array().is_err());
            assert!(array.slogdet_array().is_err());
        }
    }
    let malformed = UFuncArray::new(vec![2, 3], vec![0.0; 6], DType::Complex128).unwrap();
    assert!(malformed.det_array().is_err());
    assert!(malformed.slogdet_array().is_err());
    for dtype in [DType::F64, DType::Complex128] {
        let too_many = input(&[usize::MAX, 0, 0], Vec::new(), dtype);
        for error in [
            too_many.det_array().unwrap_err(),
            too_many.slogdet_array().unwrap_err(),
        ] {
            assert!(matches!(error, UFuncError::Shape(_)));
        }
    }
    let stack = input(&[1, 1, 1], vec![2.0], DType::F64);
    assert!(stack.det().is_err());
    assert!(stack.slogdet().is_err());
}

#[test]
fn real_determinants_survive_intermediate_overflow_and_underflow() {
    let mut cases = Vec::new();
    for diagonal in [
        [1e200, 1e200, 1e-200, 1e-200],
        [1e-200, 1e-200, 1e200, 1e200],
        [-1e200, 1e200, 1e-200, 1e-200],
        [-1e-200, 1e-200, 1e200, 1e200],
    ] {
        let mut values = vec![0.0; 16];
        for (index, &value) in diagonal.iter().enumerate() {
            values[index * 5] = value;
        }
        let array = input(&[4, 4], values, DType::F64);
        let determinant = array.det().unwrap();
        assert!((determinant - diagonal[0].signum()).abs() < 8.0 * f64::EPSILON);
        cases.push(case(&array));
    }
    // A truly extreme final determinant may overflow/underflow while slogdet
    // remains usable and carries the correct sign.
    for scale in [1e-200, 1e200] {
        let array = input(&[2, 2], vec![-scale, 0.0, 0.0, scale], DType::F64);
        let (sign, logarithm) = array.slogdet().unwrap();
        assert_eq!(sign, -1.0);
        assert!((logarithm - 2.0 * scale.ln()).abs() < 1e-12);
        cases.push(case(&array));
    }
    check_numpy(&cases);
}

#[test]
fn complex_logdet_retains_phase_when_determinant_magnitude_is_extreme() {
    let mut cases = Vec::new();
    for scale in [1e-200, 1e200] {
        let array = input(
            &[2, 2],
            vec![scale, scale, 0.0, 0.0, 0.0, 0.0, -scale, scale],
            DType::Complex128,
        );
        let (phase, logarithm) = array.slogdet_array().unwrap();
        assert!((phase.values()[0] + 1.0).abs() < 1e-14);
        assert!(phase.values()[1].abs() < 1e-14);
        assert!((logarithm.values()[0] - (2.0 * scale.ln() + 2.0_f64.ln())).abs() < 1e-12);
        cases.push(case(&array));
    }
    for (shape, values) in [
        (vec![2, 2], vec![1e200, 0.0, 0.0, 0.0, 0.0, 0.0, 1e200, 0.0]),
        (vec![1, 1], vec![f64::MAX, f64::MAX]),
    ] {
        cases.push(case(&input(&shape, values, DType::Complex128)));
    }
    check_numpy(&cases);
}
