//! Discrete differences preserve logical elements, source precision, and integer bits.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
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
        _ => panic!("test requires a numeric dtype"),
    }
}

fn complex(shape: &[usize], values: &[(f64, f64)], dtype: DType) -> UFuncArray {
    let mut physical_shape = shape.to_vec();
    physical_shape.push(2);
    UFuncArray::new(
        physical_shape,
        values
            .iter()
            .flat_map(|&(real, imaginary)| [real, imaginary])
            .collect(),
        dtype,
    )
    .unwrap()
}

fn pattern_array(shape: &[usize], dtype: DType) -> UFuncArray {
    let count: usize = shape.iter().product();
    if dtype.is_complex() {
        let values = (0..count)
            .map(|index| {
                let x = index as f64;
                (x * x + 0.25 * (x % 3.0), 0.5 - x * (x + 0.5))
            })
            .collect::<Vec<_>>();
        complex(shape, &values, dtype)
    } else {
        let values = (0..count)
            .map(|index| (index * index % 29) as f64)
            .collect();
        UFuncArray::from_storage_with_dtype(shape.to_vec(), ArrayStorage::F64(values), dtype)
            .unwrap()
    }
}

fn oracle_input(array: &UFuncArray) -> Value {
    let mut shape = array.shape().to_vec();
    let values = if array.dtype().is_complex() {
        assert_eq!(shape.pop(), Some(2));
        json!(
            array
                .values()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|parts| [parts[0].to_string(), parts[1].to_string()])
                .collect::<Vec<_>>()
        )
    } else if array.dtype().is_integer() || array.dtype() == DType::Bool {
        let storage = array.to_storage().unwrap();
        json!(
            (0..storage.len())
                .map(|index| storage.get_i128(index).unwrap().to_string())
                .collect::<Vec<_>>()
        )
    } else {
        json!(
            array
                .values()
                .iter()
                .map(f64::to_string)
                .collect::<Vec<_>>()
        )
    };
    json!({"shape": shape, "dtype": dtype_name(array.dtype()), "values": values})
}

#[derive(Default)]
struct OracleBatch {
    inputs: Vec<Value>,
    cases: Vec<Value>,
    actual: Vec<Result<UFuncArray, UFuncError>>,
}

impl OracleBatch {
    fn input(&mut self, array: &UFuncArray) -> usize {
        let index = self.inputs.len();
        self.inputs.push(oracle_input(array));
        index
    }

    fn case(
        &mut self,
        input: usize,
        op: &str,
        args: Value,
        actual: Result<UFuncArray, UFuncError>,
    ) {
        self.cases
            .push(json!({"input": input, "op": op, "args": args}));
        self.actual.push(actual);
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np

warnings.simplefilter('ignore')
np.seterr(all='ignore')

def load(item):
    dtype = np.dtype(item['dtype'])
    if dtype.kind == 'c':
        result = np.empty(len(item['values']), dtype=dtype)
        result.real = [float(parts[0]) for parts in item['values']]
        result.imag = [float(parts[1]) for parts in item['values']]
    else:
        convert = int if dtype.kind in 'biu' else float
        result = np.array([convert(value) for value in item['values']], dtype=dtype)
    return result.reshape(item['shape'])

data = json.load(sys.stdin)
inputs = [load(item) for item in data['inputs']]
rows = []
for case in data['cases']:
    try:
        array, args = inputs[case['input']], case['args']
        if case['op'] == 'diff':
            kwargs = {}
            for name in ('prepend', 'append'):
                if args.get(name) is not None:
                    kwargs[name] = inputs[args[name]]
            axis = args['axis'] if args['axis'] is not None else -1
            result = np.diff(array, n=args['n'], axis=axis, **kwargs)
        elif case['op'] == 'ediff1d':
            kwargs = {}
            for name in ('to_begin', 'to_end'):
                if args.get(name) is not None:
                    kwargs[name] = inputs[args[name]]
            result = np.ediff1d(array, **kwargs)
        else:
            raise AssertionError(case['op'])
        result = np.asarray(result)
        if result.dtype.kind == 'c':
            parts = np.empty(result.size * 2, dtype=np.float64)
            parts[0::2] = result.real.ravel()
            parts[1::2] = result.imag.ravel()
        else:
            parts = result.astype(np.float64).ravel()
        rows.append(dict(shape=list(result.shape), dtype=str(result.dtype),
                         bits=parts.view(np.uint64).tolist(),
                         integers=[str(int(value)) for value in result.flat]
                             if result.dtype.kind in 'biu' else None))
    except (IndexError, TypeError, ValueError, OverflowError) as error:
        rows.append(dict(error=type(error).__name__, message=str(error)))
print(json.dumps(rows))
"#;
        let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(python)
            .args(["-c", script])
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("launch live NumPy discrete difference oracle");
        let data = json!({"inputs": self.inputs, "cases": self.cases});
        serde_json::to_writer(child.stdin.take().unwrap(), &data).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy discrete difference oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.actual.len());
        for ((case, actual), expected) in data["cases"]
            .as_array()
            .unwrap()
            .iter()
            .zip(self.actual)
            .zip(expected)
        {
            let source = &data["inputs"][case["input"].as_u64().unwrap() as usize];
            let context = format!("{case}, source={source}");
            if expected.get("error").is_some() {
                assert!(
                    actual.is_err(),
                    "{context}: expected {expected}, got {actual:?}"
                );
                continue;
            }
            let actual = actual.unwrap_or_else(|error| panic!("{context}: {error}"));
            let mut shape = actual.shape().to_vec();
            if actual.dtype().is_complex() {
                assert_eq!(shape.pop(), Some(2), "{context}: component dimension");
            }
            assert_eq!(json!(shape), expected["shape"], "{context}: logical shape");
            assert_eq!(
                dtype_name(actual.dtype()),
                expected["dtype"],
                "{context}: dtype"
            );
            if actual.dtype().is_integer() || actual.dtype() == DType::Bool {
                let storage = actual.to_storage().unwrap();
                let integers = (0..storage.len())
                    .map(|index| storage.get_i128(index).unwrap().to_string())
                    .collect::<Vec<_>>();
                assert_eq!(
                    json!(integers),
                    expected["integers"],
                    "{context}: exact integer bits"
                );
            }
            let bits = expected["bits"].as_array().unwrap();
            assert_eq!(
                actual.values().len(),
                bits.len(),
                "{context}: component count"
            );
            for (index, (&actual, expected)) in actual.values().iter().zip(bits).enumerate() {
                let expected = f64::from_bits(expected.as_u64().unwrap());
                if expected.is_nan() {
                    assert!(actual.is_nan(), "{context}: component {index} expected NaN");
                } else {
                    // Subtraction has no reassociation freedom. Compare exact
                    // bits, including signed zero and infinity, at every dtype.
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{context}: component {index}, got {actual:?}, expected {expected:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn complex_differences_keep_component_pairs_and_logical_axes() {
    let array = complex(
        &[2, 3],
        &[
            (1.0, 10.0),
            (4.0, 20.0),
            (9.0, 40.0),
            (2.0, 3.0),
            (8.0, 11.0),
            (18.0, 27.0),
        ],
        DType::Complex128,
    );
    let first = array.diff(1, None).unwrap();
    assert_eq!(first.shape(), &[2, 2, 2]);
    assert_eq!(
        first.values(),
        &[3.0, 10.0, 5.0, 20.0, 6.0, 8.0, 10.0, 16.0]
    );
    let second = array.diff(2, Some(-1)).unwrap();
    assert_eq!(second.shape(), &[2, 1, 2]);
    assert_eq!(second.values(), &[2.0, 10.0, 4.0, 8.0]);
    let rows = array.diff(1, Some(0)).unwrap();
    assert_eq!(rows.shape(), &[1, 3, 2]);
    assert_eq!(rows.values(), &[1.0, -7.0, 4.0, -9.0, 9.0, -13.0]);
    let flat = array.ediff1d().unwrap();
    assert_eq!(flat.shape(), &[5, 2]);
    assert_eq!(
        flat.values(),
        &[3.0, 10.0, 5.0, 20.0, -7.0, -37.0, 6.0, 8.0, 10.0, 16.0]
    );
    assert!(array.diff(1, Some(2)).is_err());
    assert!(array.diff(1, Some(-3)).is_err());
}

#[test]
fn discrete_difference_shapes_axes_orders_and_scalars_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128, DType::F64, DType::Bool] {
        for shape in [
            vec![],
            vec![0],
            vec![1],
            vec![2],
            vec![5],
            vec![2, 3],
            vec![2, 3, 4],
            vec![2, 0, 3],
            vec![0, 2, 3],
            vec![3, 0],
        ] {
            let array = pattern_array(&shape, dtype);
            let input = batch.input(&array);
            let rank = shape.len() as isize;
            let axes = std::iter::once(None)
                .chain((-rank - 1..=rank).map(Some))
                .collect::<Vec<_>>();
            for axis in axes {
                for n in [0, 1, 2, 4, 7] {
                    batch.case(
                        input,
                        "diff",
                        json!({"axis": axis, "n": n}),
                        array.diff(n, axis),
                    );
                }
            }
            batch.case(input, "ediff1d", json!({}), array.ediff1d());
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn differences_round_source_components_and_each_successive_pass() {
    let mut batch = OracleBatch::default();
    let values = [
        (16_777_217.0, -16_777_219.0),
        (16_777_216.0, -16_777_220.0),
        (0.1, -0.3),
        (0.2, 0.7),
        (-1.0e-50, 1.0e-50),
        (1.0 + 2.0_f64.powi(-24), -1.0 - 2.0_f64.powi(-24)),
        (-0.0, 0.0),
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [vec![7], vec![1, 7], vec![7, 1]] {
            let array = complex(&shape, &values, dtype);
            let input = batch.input(&array);
            for axis in std::iter::once(None).chain((0..shape.len() as isize).map(Some)) {
                for n in [1, 2, 3] {
                    batch.case(
                        input,
                        "diff",
                        json!({"axis": axis, "n": n}),
                        array.diff(n, axis),
                    );
                }
            }
            batch.case(input, "ediff1d", json!({}), array.ediff1d());
        }
    }
    for dtype in [DType::F16, DType::F32, DType::F64] {
        let values = vec![2049.0, 2048.0, 0.1, -0.3, 0.7, 1.0e-50, -0.0];
        let array = UFuncArray::new(vec![7], values, dtype).unwrap();
        let input = batch.input(&array);
        for n in [1, 2, 3] {
            batch.case(
                input,
                "diff",
                json!({"axis": null, "n": n}),
                array.diff(n, None),
            );
        }
        batch.case(input, "ediff1d", json!({}), array.ediff1d());
    }
    batch.assert_matches_numpy();
}

#[test]
fn difference_nonfinite_components_and_signed_zero_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for values in [
            vec![(-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0), (0.0, 0.0)],
            vec![(f64::NAN, 1.0), (2.0, 3.0), (4.0, f64::NAN), (5.0, 8.0)],
            vec![
                (f64::INFINITY, 3.0),
                (f64::INFINITY, 4.0),
                (f64::NEG_INFINITY, 5.0),
                (2.0, f64::INFINITY),
            ],
            vec![
                (1.0e300, -1.0e300),
                (1.0e300, -1.0e300),
                (1.0e-300, -1.0e-300),
                (-0.0, 0.0),
            ],
        ] {
            for shape in [vec![4], vec![2, 2]] {
                let array = complex(&shape, &values, dtype);
                let input = batch.input(&array);
                for axis in [None, Some(0), Some(-1)] {
                    for n in [1, 2] {
                        batch.case(
                            input,
                            "diff",
                            json!({"axis": axis, "n": n}),
                            array.diff(n, axis),
                        );
                    }
                }
                batch.case(input, "ediff1d", json!({}), array.ediff1d());
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn diff_scalar_endpoints_broadcast_and_promote_original_dtypes() {
    let endpoints = [
        UFuncArray::from_storage(vec![], ArrayStorage::F64(vec![0.1])).unwrap(),
        complex(&[], &[(0.2, -0.3)], DType::Complex64),
        complex(&[], &[(1.25, -2.5)], DType::Complex128),
        UFuncArray::from_storage(vec![], ArrayStorage::I64(vec![(1_i64 << 53) + 3])).unwrap(),
        UFuncArray::from_storage(vec![], ArrayStorage::U64(vec![u64::MAX - 7])).unwrap(),
        UFuncArray::from_storage(vec![], ArrayStorage::Bool(vec![true])).unwrap(),
    ];
    let mut batch = OracleBatch::default();
    let endpoint_ids = endpoints
        .iter()
        .map(|array| batch.input(array))
        .collect::<Vec<_>>();
    for dtype in [
        DType::Complex64,
        DType::Complex128,
        DType::F32,
        DType::I16,
        DType::U64,
        DType::Bool,
    ] {
        for shape in [vec![3], vec![2, 3], vec![2, 2, 3], vec![0, 3], vec![2, 0]] {
            let array = pattern_array(&shape, dtype);
            let input = batch.input(&array);
            for axis in std::iter::once(None).chain((0..shape.len() as isize).map(Some)) {
                for (before, after) in [(0, 1), (1, 2), (3, 4), (5, 0)] {
                    for n in [0, 1, 2] {
                        batch.case(
                            input,
                            "diff",
                            json!({
                                "axis": axis, "n": n,
                                "prepend": endpoint_ids[before], "append": endpoint_ids[after],
                            }),
                            array.diff_with(
                                n,
                                axis,
                                Some(&endpoints[before]),
                                Some(&endpoints[after]),
                            ),
                        );
                    }
                }
                batch.case(
                    input,
                    "diff",
                    json!({"axis": axis, "n": 1, "prepend": endpoint_ids[1]}),
                    array.diff_with(1, axis, Some(&endpoints[1]), None),
                );
                batch.case(
                    input,
                    "diff",
                    json!({"axis": axis, "n": 1, "append": endpoint_ids[3]}),
                    array.diff_with(1, axis, None, Some(&endpoints[3])),
                );
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn diff_array_endpoint_shapes_and_validation_match_numpy() {
    let mut batch = OracleBatch::default();
    let array = pattern_array(&[2, 3], DType::Complex64);
    let input = batch.input(&array);
    let scalar = pattern_array(&[], DType::F64);
    let scalar_id = batch.input(&scalar);
    for shape in [
        vec![2, 1],
        vec![1, 3],
        vec![2, 0],
        vec![2, 2],
        vec![3],
        vec![1, 1, 3],
    ] {
        let endpoint = pattern_array(&shape, DType::Complex128);
        let endpoint_id = batch.input(&endpoint);
        for axis in [None, Some(0), Some(-1), Some(2), Some(-3)] {
            for n in [0, 1, 2] {
                batch.case(
                    input,
                    "diff",
                    json!({
                        "axis": axis, "n": n, "prepend": endpoint_id, "append": scalar_id,
                    }),
                    array.diff_with(n, axis, Some(&endpoint), Some(&scalar)),
                );
                batch.case(
                    input,
                    "diff",
                    json!({"axis": axis, "n": n, "append": endpoint_id}),
                    array.diff_with(n, axis, None, Some(&endpoint)),
                );
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn integer_differences_preserve_exact_values_and_wrapping_arithmetic() {
    let storages = [
        ArrayStorage::Bool(vec![true, false, true, true, false, false]),
        ArrayStorage::I8(vec![i8::MIN, i8::MAX, -1, 0, 1, 17]),
        ArrayStorage::I16(vec![i16::MIN, i16::MAX, -1, 0, 1, 17]),
        ArrayStorage::I32(vec![i32::MIN, i32::MAX, -1, 0, 1, 17]),
        ArrayStorage::I64(vec![
            i64::MIN,
            i64::MAX,
            (1_i64 << 53) + 3,
            -(1_i64 << 53) - 7,
            0,
            1,
        ]),
        ArrayStorage::U8(vec![u8::MAX, 0, 1, 127, 128, 17]),
        ArrayStorage::U16(vec![u16::MAX, 0, 1, 32767, 32768, 17]),
        ArrayStorage::U32(vec![u32::MAX, 0, 1, (1 << 31) - 1, 1 << 31, 17]),
        ArrayStorage::U64(vec![
            u64::MAX,
            0,
            1,
            (1_u64 << 53) + 3,
            1_u64 << 63,
            u64::MAX - 7,
        ]),
    ];
    let mut batch = OracleBatch::default();
    for storage in storages {
        for shape in [vec![6], vec![2, 3], vec![3, 2], vec![1, 2, 3]] {
            let array = UFuncArray::from_storage(shape.clone(), storage.clone()).unwrap();
            let input = batch.input(&array);
            for axis in std::iter::once(None).chain((0..shape.len() as isize).map(Some)) {
                for n in [0, 1, 2, 3] {
                    batch.case(
                        input,
                        "diff",
                        json!({"axis": axis, "n": n}),
                        array.diff(n, axis),
                    );
                }
            }
            batch.case(input, "ediff1d", json!({}), array.ediff1d());
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn typed_ediff1d_endpoints_flatten_cast_and_preserve_exact_integers() {
    let endpoints = [
        complex(&[], &[(0.2, -0.3)], DType::Complex64),
        complex(&[1, 2], &[(1.25, -2.5), (-0.0, 0.0)], DType::Complex128),
        complex(&[0], &[], DType::Complex128),
        UFuncArray::from_storage(vec![], ArrayStorage::F32(vec![0.1])).unwrap(),
        UFuncArray::from_storage(vec![0], ArrayStorage::F64(vec![])).unwrap(),
        UFuncArray::from_storage(vec![2], ArrayStorage::I64(vec![i64::MIN + 3, i64::MAX - 3]))
            .unwrap(),
        UFuncArray::from_storage(
            vec![1, 2],
            ArrayStorage::U64(vec![u64::MAX - 1, (1_u64 << 53) + 7]),
        )
        .unwrap(),
        UFuncArray::from_storage(vec![], ArrayStorage::Bool(vec![true])).unwrap(),
    ];
    let mut batch = OracleBatch::default();
    let endpoint_ids = endpoints
        .iter()
        .map(|array| batch.input(array))
        .collect::<Vec<_>>();
    for dtype in [
        DType::Complex64,
        DType::Complex128,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::I64,
        DType::U64,
        DType::Bool,
    ] {
        for shape in [vec![], vec![0], vec![4], vec![2, 2]] {
            let array = pattern_array(&shape, dtype);
            let input = batch.input(&array);
            for (endpoint, &endpoint_id) in endpoints.iter().zip(&endpoint_ids) {
                batch.case(
                    input,
                    "ediff1d",
                    json!({"to_begin": endpoint_id}),
                    array.ediff1d_with_arrays(Some(endpoint), None),
                );
                batch.case(
                    input,
                    "ediff1d",
                    json!({"to_end": endpoint_id}),
                    array.ediff1d_with_arrays(None, Some(endpoint)),
                );
                batch.case(
                    input,
                    "ediff1d",
                    json!({"to_begin": endpoint_id, "to_end": endpoint_id}),
                    array.ediff1d_with_arrays(Some(endpoint), Some(endpoint)),
                );
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn float_slice_ediff1d_endpoints_obey_numpy_casting_even_when_empty() {
    let mut batch = OracleBatch::default();
    for dtype in [
        DType::Complex64,
        DType::Complex128,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::I64,
        DType::U64,
        DType::Bool,
    ] {
        for shape in [vec![], vec![0], vec![4], vec![2, 2]] {
            let array = pattern_array(&shape, dtype);
            let input = batch.input(&array);
            for values in [
                vec![],
                vec![-0.0],
                vec![0.1, -0.3, 1.0e-50],
                vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
            ] {
                let endpoint =
                    UFuncArray::from_storage(vec![values.len()], ArrayStorage::F64(values.clone()))
                        .unwrap();
                let endpoint_id = batch.input(&endpoint);
                batch.case(
                    input,
                    "ediff1d",
                    json!({"to_begin": endpoint_id, "to_end": endpoint_id}),
                    array.ediff1d_ext(Some(&values), Some(&values)),
                );
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn diff_zero_order_ignores_axis_and_endpoints_without_converting_source() {
    let incompatible = UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::Complex64).unwrap();
    for array in [
        complex(&[], &[(16_777_217.0, -0.0)], DType::Complex64),
        UFuncArray::from_storage(vec![], ArrayStorage::I64(vec![i64::MAX - 1])).unwrap(),
        pattern_array(&[2, 3], DType::Complex128),
    ] {
        for axis in [None, Some(isize::MIN), Some(isize::MAX)] {
            for result in [
                array.diff(0, axis),
                array.diff_with(0, axis, Some(&incompatible), Some(&incompatible)),
            ] {
                let result = result.unwrap();
                assert_eq!(result.shape(), array.shape());
                assert_eq!(result.dtype(), array.dtype());
                assert_eq!(
                    result
                        .values()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    array
                        .values()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(result.to_storage().unwrap(), array.to_storage().unwrap());
            }
        }
    }
}

#[test]
fn differences_reject_malformed_storage_and_handle_huge_empty_dimensions() {
    let valid = pattern_array(&[3], DType::Complex128);
    for malformed in [
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
        UFuncArray::new(vec![3], vec![1.0; 3], DType::Complex64).unwrap(),
        UFuncArray::new(vec![0, 1], vec![], DType::Complex128).unwrap(),
    ] {
        assert!(malformed.diff(1, None).is_err());
        assert!(malformed.diff_with(1, None, Some(&valid), None).is_err());
        assert!(malformed.ediff1d().is_err());
        assert!(valid.diff_with(1, None, Some(&malformed), None).is_err());
        assert!(valid.ediff1d_with_arrays(Some(&malformed), None).is_err());
    }
    for dtype in [DType::F64, DType::Complex64, DType::Complex128] {
        let mut shape = vec![0, usize::MAX, usize::MAX];
        if dtype.is_complex() {
            shape.push(2);
        }
        let array = UFuncArray::new(shape.clone(), vec![], dtype).unwrap();
        let diff = array.diff(1, Some(1)).unwrap();
        let mut expected_shape = shape.clone();
        expected_shape[1] -= 1;
        assert_eq!(diff.shape(), expected_shape);
        assert!(diff.values().is_empty());
        let diff = array.diff(usize::MAX, Some(-1)).unwrap();
        let mut expected_shape = shape;
        expected_shape[2] = 0;
        assert_eq!(diff.shape(), expected_shape);
        assert!(diff.values().is_empty());
        assert!(array.diff(1, Some(3)).is_err());
        let nonempty = pattern_array(&[2, 3], dtype)
            .diff(usize::MAX, None)
            .unwrap();
        assert!(nonempty.values().is_empty());
        let expected = if dtype.is_complex() {
            vec![2, 0, 2]
        } else {
            vec![2, 0]
        };
        assert_eq!(nonempty.shape(), expected);
    }
}
