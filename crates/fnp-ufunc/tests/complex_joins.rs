//! Complex joins preserve logical axes, component pairs, and casting policy.

use fnp_dtype::{ArrayStorage, DType};
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
use fnp_ufunc::{FloatErrorCallback, seterrcall};
use fnp_ufunc::{
    FloatErrorKind, FloatErrorMode, UFuncArray, UFuncError, errstate, take_float_error_events,
};
use serde_json::{Value, json};
use std::process::{Command, Stdio};
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn complex(shape: &[usize], values: &[(f64, f64)], dtype: DType) -> UFuncArray {
    let mut shape = shape.to_vec();
    shape.push(2);
    UFuncArray::new(
        shape,
        values.iter().flat_map(|&(re, im)| [re, im]).collect(),
        dtype,
    )
    .unwrap()
}

fn real(shape: &[usize], values: &[f64], dtype: DType) -> UFuncArray {
    UFuncArray::new(shape.to_vec(), values.to_vec(), dtype).unwrap()
}

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
        _ => panic!("unexpected join dtype {dtype:?}"),
    }
}

fn logical_shape(array: &UFuncArray) -> Vec<usize> {
    let mut shape = array.shape().to_vec();
    if array.dtype().is_complex() {
        assert_eq!(shape.pop(), Some(2), "complex components must stay paired");
    }
    shape
}

fn values(array: &UFuncArray) -> Value {
    if array.dtype().is_complex() {
        return json!(
            array
                .values()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| [pair[0].to_string(), pair[1].to_string()])
                .collect::<Vec<_>>()
        );
    }
    let strings = match array.to_storage().unwrap() {
        ArrayStorage::I64(values) => values.iter().map(ToString::to_string).collect(),
        ArrayStorage::U64(values) => values.iter().map(ToString::to_string).collect(),
        _ => array
            .values()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
    };
    json!(strings)
}

fn invoke(
    op: &str,
    arrays: &[UFuncArray],
    axis: Option<isize>,
    dtype: Option<DType>,
    casting: &str,
) -> Result<UFuncArray, UFuncError> {
    let refs: Vec<_> = arrays.iter().collect();
    match op {
        "concatenate" => match (axis, dtype) {
            (Some(axis), Some(dtype)) => {
                UFuncArray::concatenate_with_dtype(&refs, axis, dtype, casting)
            }
            (Some(axis), None) => UFuncArray::concatenate(&refs, axis),
            (None, None) => UFuncArray::concatenate_flat(&refs),
            _ => unreachable!("flat explicit-dtype join has no native entrypoint"),
        },
        "stack" => match dtype {
            Some(dtype) => UFuncArray::stack_with_dtype(&refs, axis.unwrap(), dtype, casting),
            None => UFuncArray::stack(&refs, axis.unwrap()),
        },
        "hstack" => UFuncArray::hstack(arrays),
        "vstack" => UFuncArray::vstack(arrays),
        "dstack" => UFuncArray::dstack(arrays),
        "column_stack" => UFuncArray::column_stack(arrays),
        "append" => arrays[0].append(&arrays[1], axis),
        "atleast_1d" => Ok(arrays[0].atleast_1d()),
        "atleast_2d" => Ok(arrays[0].atleast_2d()),
        "atleast_3d" => Ok(arrays[0].atleast_3d()),
        "ravel" => Ok(arrays[0].ravel()),
        _ => unreachable!("join operation"),
    }
}

#[derive(Default)]
struct OracleBatch {
    cases: Vec<Value>,
    actual: Vec<Result<UFuncArray, UFuncError>>,
}

impl OracleBatch {
    fn push(
        &mut self,
        op: &str,
        arrays: &[UFuncArray],
        axis: Option<isize>,
        dtype: Option<DType>,
        casting: &str,
    ) {
        self.cases.push(json!({
            "op": op,
            "arrays": arrays.iter().map(|array| json!({
                "shape": logical_shape(array), "dtype": dtype_name(array.dtype()),
                "values": values(array),
            })).collect::<Vec<_>>(),
            "axis": axis, "dtype": dtype.map(dtype_name), "casting": casting,
        }));
        let _errors = errstate(Some(FloatErrorMode::Ignore), None, None, None, None);
        self.actual.push(invoke(op, arrays, axis, dtype, casting));
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np
warnings.simplefilter('ignore')
np.seterr(all='ignore')
rows = []
for case in json.load(sys.stdin):
    arrays = []
    for item in case['arrays']:
        dtype = np.dtype(item['dtype'])
        if dtype.kind == 'c':
            array = np.empty(len(item['values']), dtype=dtype)
            array.real = [float(pair[0]) for pair in item['values']]
            array.imag = [float(pair[1]) for pair in item['values']]
        else:
            convert = int if dtype.kind in 'iub' else float
            array = np.array([convert(value) for value in item['values']], dtype=dtype)
        arrays.append(array.reshape(item['shape']))
    try:
        op = case['op']
        if op in ('concatenate', 'stack'):
            kw = dict(axis=case['axis'])
            if case['dtype'] is not None:
                kw.update(dtype=case['dtype'], casting=case['casting'])
            result = getattr(np, op)(arrays, **kw)
        elif op == 'append':
            result = np.append(*arrays, axis=case['axis'])
        elif op in ('atleast_1d', 'atleast_2d', 'atleast_3d', 'ravel'):
            result = getattr(np, op)(arrays[0])
        else:
            result = getattr(np, op)(arrays)
        flat = result.reshape(-1)
        if result.dtype.kind == 'c':
            values = [[str(float(value.real)), str(float(value.imag))] for value in flat]
        elif result.dtype.kind in 'iub':
            values = [str(int(value)) for value in flat]
        else:
            values = [str(float(value)) for value in flat]
        rows.append(dict(shape=list(result.shape), dtype=str(result.dtype), values=values))
    except Exception as error:
        rows.append(dict(error=type(error).__name__))
json.dump(rows, sys.stdout)
"#;
        let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(python)
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("NumPy oracle must be available");
        serde_json::to_writer(child.stdin.take().unwrap(), &self.cases).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.actual.len());
        for ((case, actual), expected) in self.cases.iter().zip(self.actual).zip(expected) {
            if expected.get("error").is_some() {
                assert!(
                    actual.is_err(),
                    "{case}: expected {expected}, got {actual:?}"
                );
                continue;
            }
            let actual = actual.unwrap_or_else(|error| panic!("{case}: {error:?}"));
            assert_eq!(
                dtype_name(actual.dtype()),
                expected["dtype"].as_str().unwrap(),
                "{case}"
            );
            assert_eq!(json!(logical_shape(&actual)), expected["shape"], "{case}");
            let actual_values = values(&actual);
            let actual_values = actual_values.as_array().unwrap();
            let expected_values = expected["values"].as_array().unwrap();
            assert_eq!(actual_values.len(), expected_values.len(), "{case}");
            for (actual, expected) in actual_values.iter().zip(expected_values) {
                let components = if actual.is_array() {
                    actual
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(expected.as_array().unwrap())
                        .collect::<Vec<_>>()
                } else {
                    vec![(actual, expected)]
                };
                for (actual, expected) in components {
                    let actual = actual.as_str().unwrap();
                    let expected = expected.as_str().unwrap();
                    if expected.parse::<i128>().is_ok() && actual.parse::<i128>().is_ok() {
                        assert_eq!(
                            actual.parse::<i128>().unwrap(),
                            expected.parse::<i128>().unwrap(),
                            "{case}"
                        );
                    } else {
                        let actual: f64 = actual.parse().unwrap();
                        let expected: f64 = expected.parse().unwrap();
                        if expected.is_nan() {
                            assert!(actual.is_nan(), "{case}");
                        } else {
                            assert_eq!(actual.to_bits(), expected.to_bits(), "{case}");
                        }
                    }
                }
            }
        }
        eprintln!("validated {} NumPy join cases", self.cases.len());
    }
}

#[test]
fn complex_concatenate_and_stack_use_every_logical_axis() {
    let mut batch = OracleBatch::default();
    for shape in [&[][..], &[3][..], &[2, 3][..], &[2, 1, 3][..]] {
        let count: usize = shape.iter().product();
        let pairs: Vec<_> = (0..count)
            .map(|i| (i as f64 + 0.25, -(i as f64) - 0.5))
            .collect();
        for dtype in [DType::Complex64, DType::Complex128] {
            let a = complex(shape, &pairs, dtype);
            let b = real(
                shape,
                &(0..count).map(|i| 100.0 + i as f64).collect::<Vec<_>>(),
                DType::F32,
            );
            for arrays in [
                vec![a.clone(), a.clone()],
                vec![a.clone(), b.clone()],
                vec![b, a],
            ] {
                for axis in -(shape.len() as isize)..shape.len() as isize {
                    batch.push("concatenate", &arrays, Some(axis), None, "same_kind");
                }
                for axis in -(shape.len() as isize + 1)..=shape.len() as isize {
                    batch.push("stack", &arrays, Some(axis), None, "same_kind");
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_join_dtype_overrides_cast_original_inputs_once() {
    let a = complex(
        &[2, 2],
        &[(1.75, -2.5), (-3.75, 4.0), (0.0, 5.0), (6.25, -0.0)],
        DType::Complex128,
    );
    let b = real(&[2, 2], &[8.75, -9.25, 10.5, 11.125], DType::F64);
    let mut batch = OracleBatch::default();
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for axis in [-2, -1, 0, 1] {
            batch.push(
                "concatenate",
                &[a.clone(), b.clone()],
                Some(axis),
                Some(dtype),
                "unsafe",
            );
        }
        for axis in [-3, -1, 0, 2] {
            batch.push(
                "stack",
                &[a.clone(), b.clone()],
                Some(axis),
                Some(dtype),
                "unsafe",
            );
        }
    }
    let integers =
        UFuncArray::from_storage(vec![2], ArrayStorage::I64(vec![(1i64 << 53) + 1, i64::MAX]))
            .unwrap();
    let complex_integers = complex(&[2], &[(5.0, 1.0), (-9.0, 0.0)], DType::Complex128);
    for op in ["concatenate", "stack"] {
        for dtype in [DType::I64, DType::Complex64, DType::Complex128] {
            batch.push(
                op,
                &[integers.clone(), complex_integers.clone()],
                Some(-1),
                Some(dtype),
                "unsafe",
            );
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_convenience_joins_preserve_pairs_and_promote_dimensions() {
    let mut batch = OracleBatch::default();
    for shape in [&[][..], &[3][..], &[2, 3][..], &[1, 2, 3][..]] {
        let count: usize = shape.iter().product();
        let a = complex(
            shape,
            &(0..count)
                .map(|i| (i as f64, 10.0 + i as f64))
                .collect::<Vec<_>>(),
            DType::Complex128,
        );
        let b = real(
            shape,
            &(0..count).map(|i| -1.0 - i as f64).collect::<Vec<_>>(),
            DType::F64,
        );
        for op in ["hstack", "vstack", "dstack", "column_stack"] {
            for arrays in [
                vec![a.clone(), a.clone()],
                vec![a.clone(), b.clone()],
                vec![b.clone(), a.clone()],
            ] {
                batch.push(op, &arrays, None, None, "same_kind");
            }
        }
        for op in ["atleast_1d", "atleast_2d", "atleast_3d", "ravel"] {
            batch.push(op, std::slice::from_ref(&a), None, None, "same_kind");
        }
        batch.push("append", &[a.clone(), b.clone()], None, None, "same_kind");
        batch.push("concatenate", &[a, b], None, None, "same_kind");
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_joins_support_empty_populations_and_signed_zero() {
    let mut batch = OracleBatch::default();
    for shape in [&[0][..], &[0, 3][..], &[2, 0][..]] {
        let a = complex(shape, &[], DType::Complex64);
        let b = real(shape, &[], DType::I64);
        for op in ["hstack", "vstack", "dstack", "column_stack"] {
            batch.push(op, &[a.clone(), b.clone()], None, None, "same_kind");
        }
        for axis in -(shape.len() as isize)..shape.len() as isize {
            batch.push(
                "concatenate",
                &[a.clone(), b.clone()],
                Some(axis),
                None,
                "same_kind",
            );
            batch.push(
                "stack",
                &[a.clone(), b.clone()],
                Some(axis),
                Some(DType::Complex128),
                "safe",
            );
        }
        batch.push(
            "concatenate",
            &[a.clone(), b.clone()],
            None,
            None,
            "same_kind",
        );
        batch.push("append", &[a, b], None, None, "same_kind");
    }
    let a = complex(&[2], &[(0.0, -0.0), (-0.0, 0.0)], DType::Complex64);
    batch.push(
        "stack",
        &[a.clone(), a.clone()],
        Some(-1),
        Some(DType::Complex128),
        "safe",
    );
    batch.push("concatenate", &[a.clone(), a], Some(-1), None, "same_kind");
    batch.assert_matches_numpy();
}

#[test]
fn complex_join_validation_precedes_value_conversion() {
    let _errors = errstate(None, None, None, None, Some(FloatErrorMode::Raise));
    let _ = take_float_error_events();
    let a = complex(&[2, 2], &[(f64::NAN, 1.0); 4], DType::Complex128);
    let b = complex(&[3, 2], &[(1.0, 0.0); 6], DType::Complex128);
    assert!(matches!(
        UFuncArray::concatenate_with_dtype(&[&a, &b], 1, DType::I64, "safe"),
        Err(UFuncError::Shape(_))
    ));
    assert!(matches!(
        UFuncArray::concatenate_with_dtype(&[&a, &b], 5, DType::I64, "unsafe"),
        Err(UFuncError::AxisOutOfBounds { .. })
    ));
    assert!(matches!(
        UFuncArray::stack_with_dtype(&[&a, &b], 5, DType::I64, "unsafe"),
        Err(UFuncError::Shape(_))
    ));
    assert!(take_float_error_events().is_empty());
    for op in ["concatenate", "stack"] {
        assert!(matches!(
            invoke(
                op,
                &[a.clone(), a.clone()],
                Some(-1),
                Some(DType::I64),
                "unsafe"
            ),
            Err(UFuncError::FloatingPoint {
                kind: FloatErrorKind::Invalid,
                ..
            })
        ));
    }
    let _ = take_float_error_events();
}

#[test]
fn complex_join_cast_rules_and_axis_errors_match_numpy() {
    let a = complex(&[2], &[(1.0, 2.0), (3.0, 4.0)], DType::Complex128);
    let b = real(&[2], &[5.0, 6.0], DType::F64);
    let mut batch = OracleBatch::default();
    for op in ["concatenate", "stack"] {
        batch.push(op, &[], Some(0), None, "same_kind");
        for casting in ["no", "equiv", "safe", "same_kind", "unsafe", "invalid"] {
            for dtype in [DType::I64, DType::Complex64, DType::Complex128] {
                batch.push(op, &[a.clone(), b.clone()], Some(-1), Some(dtype), casting);
            }
        }
        for axis in [-4, 3] {
            batch.push(op, &[a.clone(), b.clone()], Some(axis), None, "same_kind");
        }
    }
    batch.push("concatenate", &[a.clone(), b], Some(1), None, "same_kind");
    let scalar = complex(&[], &[(1.0, 2.0)], DType::Complex128);
    batch.push(
        "concatenate",
        &[scalar.clone(), scalar],
        Some(0),
        None,
        "same_kind",
    );
    batch.assert_matches_numpy();
}

#[test]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn uint32_join_casts_follow_destination_vector_runs() {
    let mut batch = OracleBatch::default();
    for shape in [
        &[6][..],
        &[2, 3][..],
        &[3, 2][..],
        &[2, 8][..],
        &[8, 1][..],
        &[1, 8][..],
        &[2, 1, 4][..],
    ] {
        let count: usize = shape.iter().product();
        for dtype in [
            DType::F16,
            DType::F32,
            DType::F64,
            DType::Complex64,
            DType::Complex128,
        ] {
            for pattern in [
                &[-4_294_967_296.0][..],
                &[
                    f64::NAN,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    -4_294_967_296.0,
                    -2_147_483_649.0,
                    2_147_483_648.0,
                    4_294_967_296.0,
                    -1.0,
                ][..],
            ] {
                let data: Vec<_> = (0..count).map(|i| pattern[i % pattern.len()]).collect();
                let source = if dtype.is_complex() {
                    complex(
                        shape,
                        &data
                            .iter()
                            .map(|&real| (real, f64::NAN))
                            .collect::<Vec<_>>(),
                        dtype,
                    )
                } else {
                    real(shape, &data, dtype)
                };
                for axis in 0..shape.len() {
                    batch.push(
                        "concatenate",
                        &[source.clone(), source.clone()],
                        Some(axis as isize),
                        Some(DType::U32),
                        "unsafe",
                    );
                    let mut empty_shape = shape.to_vec();
                    empty_shape[axis] = 0;
                    let empty = if dtype.is_complex() {
                        complex(&empty_shape, &[], dtype)
                    } else {
                        real(&empty_shape, &[], dtype)
                    };
                    batch.push(
                        "concatenate",
                        &[source.clone(), empty],
                        Some(axis as isize),
                        Some(DType::U32),
                        "unsafe",
                    );
                }
                for axis in 0..=shape.len() {
                    batch.push(
                        "stack",
                        &[source.clone(), source.clone()],
                        Some(axis as isize),
                        Some(DType::U32),
                        "unsafe",
                    );
                    batch.push(
                        "stack",
                        std::slice::from_ref(&source),
                        Some(axis as isize),
                        Some(DType::U32),
                        "unsafe",
                    );
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
struct RestoreCallback(Option<FloatErrorCallback>);

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
impl Drop for RestoreCallback {
    fn drop(&mut self) {
        seterrcall(self.0.take());
    }
}

#[test]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn uint32_join_invalid_status_is_aggregated_per_operand() {
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&calls);
    let _callback = RestoreCallback(seterrcall(Some(Arc::new(move |event| {
        assert_eq!(event.kind, FloatErrorKind::Invalid);
        assert_eq!(event.op, "cast");
        callback_calls.fetch_add(1, Ordering::SeqCst);
    }))));
    let _errors = errstate(None, None, None, None, Some(FloatErrorMode::Call));
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        let source = if dtype.is_complex() {
            complex(&[2, 3], &[(-4_294_967_296.0, f64::NAN); 6], dtype)
        } else {
            real(&[2, 3], &[-4_294_967_296.0; 6], dtype)
        };
        calls.store(0, Ordering::SeqCst);
        let scalar_runs =
            UFuncArray::concatenate_with_dtype(&[&source, &source], 1, DType::U32, "unsafe")
                .unwrap();
        assert!(scalar_runs.values().iter().all(|&value| value == 0.0));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        UFuncArray::concatenate_with_dtype(&[&source, &source], 0, DType::U32, "unsafe").unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        UFuncArray::stack_with_dtype(&[&source, &source], -1, DType::U32, "unsafe").unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
    calls.store(0, Ordering::SeqCst);
    let bad = complex(&[8, 2], &[(f64::NAN, 0.0); 16], DType::Complex128);
    let good = real(&[8, 1], &[3.0; 8], DType::F64);
    UFuncArray::concatenate_with_dtype(&[&bad, &good], 1, DType::U32, "unsafe").unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    UFuncArray::concatenate_with_dtype(&[&bad, &bad], 1, DType::U32, "unsafe").unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[test]
fn joins_handle_huge_empty_shapes_without_iteration_or_overflow() {
    let real_empty = real(&[usize::MAX, 0], &[], DType::F64);
    let complex_empty = complex(&[usize::MAX, 0], &[], DType::Complex128);
    for array in [&real_empty, &complex_empty] {
        let joined = UFuncArray::concatenate(&[array, array], 1).unwrap();
        assert_eq!(joined.shape(), array.shape());
        assert_eq!(joined.dtype(), array.dtype());
        assert!(joined.values().is_empty());
        assert!(matches!(
            UFuncArray::concatenate(&[array, array], 0),
            Err(UFuncError::Shape(fnp_ndarray::ShapeError::Overflow))
        ));
    }
    let converted =
        UFuncArray::concatenate_with_dtype(&[&complex_empty, &real_empty], 1, DType::U32, "unsafe")
            .unwrap();
    assert_eq!(converted.shape(), &[usize::MAX, 0]);
    assert!(converted.values().is_empty());
}
