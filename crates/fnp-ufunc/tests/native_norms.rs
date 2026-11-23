//! Native norm contracts: logical complex elements, typed arithmetic and batches.

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
        _ => unreachable!("numeric norm test dtype"),
    }
}

fn input(shape: &[usize], dtype: DType, data: &[f64]) -> UFuncArray {
    let mut physical_shape = shape.to_vec();
    if dtype.is_complex() {
        physical_shape.push(2);
    }
    // Deliberately retain unrounded working values: the operation must read
    // them at their declared source precision, just as NumPy reads storage.
    UFuncArray::new(physical_shape, data.to_vec(), dtype).unwrap()
}

fn result_record(result: Result<UFuncArray, UFuncError>) -> Value {
    match result {
        Ok(array) => {
            let storage = array
                .to_storage()
                .expect("norm output is valid typed storage");
            assert_eq!(storage.len(), array.values().len());
            if array.dtype() == DType::F32 {
                for &value in array.values() {
                    assert!(value.is_nan() || value == f64::from(value as f32));
                }
            }
            json!({
                "shape": array.shape(),
                "dtype": dtype_name(array.dtype()),
                "values": array.values().iter().map(|v| v.to_string()).collect::<Vec<_>>()
            })
        }
        Err(error) => json!({"error": error.to_string()}),
    }
}

fn vector_case(
    cases: &mut Vec<Value>,
    array: &UFuncArray,
    order: Option<f64>,
    axis: Option<isize>,
    axes: Option<&[isize]>,
    keepdims: bool,
) {
    let mut shape = array.shape().to_vec();
    if array.dtype().is_complex() {
        shape.pop();
    }
    let actual = if let Some(axes) = axes {
        array.vector_norm_axes(order, axes, keepdims)
    } else {
        array.vector_norm(order, axis, keepdims)
    };
    cases.push(json!({
        "operation": "vector_norm", "shape": shape,
        "input_dtype": dtype_name(array.dtype()),
        "input": array.values().iter().map(|v| v.to_string()).collect::<Vec<_>>(),
        "order": order.map(|v| v.to_string()), "axis": axis, "axes": axes,
        "keepdims": keepdims, "actual": result_record(actual)
    }));
}

fn matrix_case(cases: &mut Vec<Value>, array: &UFuncArray, order: &str, keepdims: bool) {
    let mut shape = array.shape().to_vec();
    if array.dtype().is_complex() {
        shape.pop();
    }
    cases.push(json!({
        "operation": "matrix_norm", "shape": shape,
        "input_dtype": dtype_name(array.dtype()),
        "input": array.values().iter().map(|v| v.to_string()).collect::<Vec<_>>(),
        "order": order, "keepdims": keepdims,
        "actual": result_record(array.matrix_norm(Some(order), keepdims))
    }));
}

fn check_numpy(cases: &[Value]) {
    let script = r#"
import json, sys, warnings
import numpy as np
cases = json.load(sys.stdin)
for case in cases:
    raw = np.array([float(x) for x in case['input']], dtype=np.float64)
    if case['input_dtype'].startswith('complex'):
        raw = raw.view(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        source = raw.astype(case['input_dtype']).reshape(case['shape'])
        order = case['order']
        if order is not None and order not in ('fro', 'f', 'nuc'):
            order = float(order)
        try:
            if case['operation'] == 'vector_norm':
                axis = tuple(case['axes']) if case['axes'] is not None else case['axis']
                expected = np.linalg.vector_norm(source, axis=axis, keepdims=case['keepdims'], ord=order)
            else:
                expected = np.linalg.matrix_norm(source, keepdims=case['keepdims'], ord=order)
        except (ValueError, TypeError, np.linalg.LinAlgError) as error:
            assert 'error' in case['actual'], (case, type(error).__name__, str(error))
            continue
    actual = case['actual']
    assert 'error' not in actual, (case, expected)
    assert actual['shape'] == list(expected.shape), (case, expected.shape)
    assert actual['dtype'] == str(expected.dtype), (case, expected.dtype)
    values = np.array([float(x) for x in actual['values']]).reshape(actual['shape'])
    precision = expected.dtype.itemsize
    tolerance = 0.004 if precision == 2 else 3e-6 if precision == 4 else 5e-11
    np.testing.assert_allclose(values, expected, rtol=tolerance, atol=tolerance * 1e-3,
                               equal_nan=True, err_msg=str(case))
print(f'NumPy {np.__version__}: {len(cases)} native norm cases passed')
"#;
    let interpreter = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let mut child = Command::new(interpreter)
        .args(["-c", script])
        .env("OPENBLAS_NUM_THREADS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch native norm NumPy oracle");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(cases).unwrap())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn complex_norms_reduce_whole_elements_and_preserve_component_precision() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let a = input(&[2], dtype, &[3.0, 4.0, -5.0, 12.0]);
        assert_eq!(a.norm(Some("1")).unwrap(), 18.0);
        assert_eq!(a.norm(Some("inf")).unwrap(), 13.0);
        assert_eq!(a.norm(Some("-inf")).unwrap(), 5.0);
        let squared = a.vector_norm(None, None, true).unwrap();
        assert_eq!(squared.shape(), &[1]);
        assert_eq!(
            squared.dtype(),
            if dtype == DType::Complex64 {
                DType::F32
            } else {
                DType::F64
            }
        );
        assert!((squared.values()[0] - 194.0_f64.sqrt()).abs() < 1e-6);
        let count = a.vector_norm(Some(0.0), None, false).unwrap();
        assert_eq!(count.values(), &[2.0]);
    }
    let scalar = input(&[], DType::Complex128, &[3.0, 4.0]);
    assert_eq!(scalar.norm(None).unwrap(), 5.0);
    assert_eq!(scalar.vector_norm(None, None, true).unwrap().shape(), &[]);
    assert!(scalar.norm(Some("2")).is_err());
    assert!(scalar.vector_norm(None, Some(0), false).is_err());
}

#[test]
fn vector_norms_dtypes_axes_orders_and_source_rounding_match_numpy() {
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
        for shape in [vec![], vec![6], vec![2, 3], vec![2, 3, 2]] {
            let count = shape.iter().product::<usize>() * if dtype.is_complex() { 2 } else { 1 };
            let data: Vec<f64> = (0..count)
                .map(|i| ((i * 17 + 3) % 23) as f64 / 7.0)
                .collect();
            let a = input(&shape, dtype, &data);
            let axes =
                std::iter::once(None).chain((0..shape.len()).map(|axis| Some(axis as isize)));
            for axis in axes {
                for order in [
                    None,
                    Some(0.0),
                    Some(1.0),
                    Some(2.0),
                    Some(3.5),
                    Some(-1.0),
                    Some(-2.0),
                    Some(0.5),
                    Some(f64::INFINITY),
                    Some(f64::NEG_INFINITY),
                ] {
                    for keepdims in [false, true] {
                        vector_case(&mut cases, &a, order, axis, None, keepdims);
                    }
                }
            }
        }
    }
    check_numpy(&cases);
}

#[test]
fn vector_tuple_axes_follow_supplied_order_and_keep_logical_dimensions() {
    let mut cases = Vec::new();
    for dtype in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        let components = if dtype.is_complex() { 2 } else { 1 };
        let data: Vec<f64> = (0..24 * components)
            .map(|i| (i as f64 - 17.0) / 9.0)
            .collect();
        let a = input(&[2, 3, 4], dtype, &data);
        for axes in [
            &[][..],
            &[0],
            &[-1],
            &[0, 2],
            &[2, 0],
            &[-1, -3],
            &[2, 1, 0],
            &[0, 0],
            &[0, -3],
            &[3],
        ] {
            for order in [
                None,
                Some(0.0),
                Some(1.0),
                Some(-1.0),
                Some(3.0),
                Some(f64::INFINITY),
                Some(f64::NEG_INFINITY),
            ] {
                for keepdims in [false, true] {
                    vector_case(&mut cases, &a, order, None, Some(axes), keepdims);
                }
            }
        }
    }
    check_numpy(&cases);
}

#[test]
fn empty_vector_lanes_distinguish_zero_identity_negative_powers_and_minimum_errors() {
    let mut cases = Vec::new();
    for dtype in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for shape in [vec![0], vec![0, 2], vec![2, 0], vec![2, 0, 3], vec![0, 0]] {
            let a = input(&shape, dtype, &[]);
            for order in [
                None,
                Some(0.0),
                Some(1.0),
                Some(-1.0),
                Some(f64::INFINITY),
                Some(f64::NEG_INFINITY),
            ] {
                for keepdims in [false, true] {
                    vector_case(&mut cases, &a, order, None, None, keepdims);
                    for axis in 0..shape.len() {
                        vector_case(&mut cases, &a, order, Some(axis as isize), None, keepdims);
                    }
                    let axes: Vec<isize> = (0..shape.len()).rev().map(|i| i as isize).collect();
                    vector_case(&mut cases, &a, order, None, Some(&axes), keepdims);
                    vector_case(&mut cases, &a, order, None, Some(&[]), keepdims);
                }
            }
        }
    }
    check_numpy(&cases);
}

#[test]
fn vector_norms_nonfinite_values_and_low_precision_overflow_match_numpy() {
    let mut cases = Vec::new();
    for dtype in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for data in [
            vec![0.0, -0.0, 0.0, -0.0],
            vec![f64::NAN, 0.0, 3.0, 4.0],
            vec![f64::INFINITY, f64::NAN, 3.0, 4.0],
            vec![300.0, 0.0, 400.0, 0.0],
            vec![1e30, 0.0, -1e30, 0.0],
            vec![1e-30, 0.0, 1e-30, 0.0],
        ] {
            let a = input(&[if dtype.is_complex() { 2 } else { 4 }], dtype, &data);
            for order in [
                None,
                Some(0.0),
                Some(1.0),
                Some(-1.0),
                Some(0.5),
                Some(f64::NAN),
                Some(f64::INFINITY),
                Some(f64::NEG_INFINITY),
            ] {
                vector_case(&mut cases, &a, order, None, None, false);
            }
        }
    }
    check_numpy(&cases);
}

#[test]
fn stacked_real_and_complex_matrix_norms_match_numpy() {
    let mut cases = Vec::new();
    for dtype in [
        DType::Bool,
        DType::I32,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for shape in [
            vec![2, 3],
            vec![3, 2],
            vec![2, 2, 3],
            vec![2, 3, 2, 2],
            vec![0, 3],
            vec![3, 0],
            vec![0, 0],
            vec![0, 2, 3],
            vec![2, 0, 3],
        ] {
            let count = shape.iter().product::<usize>() * if dtype.is_complex() { 2 } else { 1 };
            let data: Vec<f64> = (0..count)
                .map(|i| ((i * 13 + 5) % 29) as f64 / 11.0)
                .collect();
            let a = input(&shape, dtype, &data);
            for order in ["fro", "f", "1", "-1", "inf", "-inf", "2", "-2", "nuc", "0"] {
                for keepdims in [false, true] {
                    matrix_case(&mut cases, &a, order, keepdims);
                }
            }
        }
    }
    check_numpy(&cases);
}

#[test]
fn matrix_spectral_and_induced_norms_obey_complex_diagonal_identities() {
    let a = input(
        &[2, 2, 2],
        DType::Complex128,
        &[
            3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, -12.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0,
        ],
    );
    for (order, expected) in [
        ("2", [5.0, 12.0]),
        ("-2", [2.0, 5.0]),
        ("nuc", [7.0, 17.0]),
        ("1", [5.0, 12.0]),
        ("-1", [2.0, 5.0]),
        ("inf", [5.0, 12.0]),
        ("-inf", [2.0, 5.0]),
    ] {
        let actual = a.matrix_norm(Some(order), true).unwrap();
        assert_eq!(actual.shape(), &[2, 1, 1]);
        assert_eq!(actual.dtype(), DType::F64);
        for (&actual, expected) in actual.values().iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "{order}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn norms_normalize_storage_reject_invalid_inputs_and_check_empty_output_capacity() {
    let wide =
        UFuncArray::from_storage(vec![2], ArrayStorage::U64(vec![u64::MAX, 1 << 63])).unwrap();
    let expected = u64::MAX as f64 + (1u64 << 63) as f64;
    assert_eq!(wide.norm(Some("1")).unwrap(), expected);
    let rounded = input(&[1], DType::F32, &[16_777_217.0]);
    assert_eq!(rounded.norm(Some("1")).unwrap(), 16_777_216.0);
    let scalar = input(&[], DType::F32, &[4.0]);
    assert_eq!(scalar.norm(None).unwrap(), 4.0);
    assert!(scalar.matrix_norm(None, false).is_err());
    let rank_three = input(&[1, 1, 1], DType::F64, &[2.0]);
    assert_eq!(rank_three.norm(None).unwrap(), 2.0);
    assert!(rank_three.norm(Some("2")).is_err());
    let bad_complex = UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::Complex128).unwrap();
    assert!(bad_complex.norm(None).is_err());
    assert!(bad_complex.vector_norm(None, None, false).is_err());
    assert!(bad_complex.matrix_norm(None, false).is_err());
    let temporal = UFuncArray::new(vec![2], vec![0.0; 2], DType::DateTime64).unwrap();
    assert!(temporal.vector_norm(None, None, false).is_err());
    let impossible_output = input(&[usize::MAX, 0], DType::F64, &[]);
    assert!(impossible_output.vector_norm(None, Some(1), false).is_err());
    let impossible_batch = input(&[usize::MAX, 0, 0], DType::F64, &[]);
    assert!(impossible_batch.matrix_norm(None, false).is_err());
    let empty_batch = input(&[0, usize::MAX, usize::MAX], DType::F64, &[]);
    assert!(
        empty_batch
            .matrix_norm(None, true)
            .unwrap()
            .values()
            .is_empty()
    );
}

#[test]
fn half_scalar_norm_retains_float32_dot_accumulation_before_the_final_cast() {
    let a = input(&[1024], DType::F16, &[0.0001; 1024]);
    // Every squared input is smaller than float16's smallest subnormal. The
    // scalar dot path sums those products in float32 before narrowing; the
    // vector_norm ufunc path intentionally squares in float16 first.
    let vector = a.vector_norm(None, None, false).unwrap();
    assert_eq!(vector.values(), &[0.0]);
    assert!(a.norm(None).unwrap() > 0.003);
    assert_eq!(a.norm(None).unwrap(), a.norm(Some("2")).unwrap());
    assert_eq!(a.norm(None).unwrap(), a.norm(Some("2.0")).unwrap());
    assert_eq!(a.norm(None).unwrap(), a.norm(Some("2e0")).unwrap());
    let matrix = input(&[32, 32], DType::F16, &[0.0001; 1024]);
    assert_eq!(a.norm(None).unwrap(), matrix.norm(Some("fro")).unwrap());
    let script = "import numpy as n; a=n.full(1024, 0.0001, dtype=n.float16); print(float(n.linalg.norm(a)))";
    let interpreter = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = Command::new(interpreter)
        .args(["-c", script])
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .unwrap();
    assert!(output.status.success());
    let expected: f64 = String::from_utf8(output.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(a.norm(None).unwrap(), expected);
}
