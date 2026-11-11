//! Indexed assignment preserves logical elements, typed values, and error timing.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{
    FloatErrorCallback, FloatErrorMode, UFuncArray, UFuncError, errstate, seterrcall,
    take_float_error_events,
};
use serde_json::{Value, json};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

const NUMERIC: [DType; 14] = [
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
];

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
        _ => panic!("non-numeric scatter test input"),
    }
}

fn count(shape: &[usize]) -> usize {
    if shape.contains(&0) {
        0
    } else {
        shape.iter().product()
    }
}

fn complex(shape: &[usize], dtype: DType, values: &[(f64, f64)]) -> UFuncArray {
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

fn sample(shape: &[usize], dtype: DType) -> UFuncArray {
    let pattern = [
        (-0.0, 0.0),
        (1.0 + 2.0_f64.powi(-24), -1.0 - 2.0_f64.powi(-24)),
        (1.0e-50, -1.0e-50),
        (-3.5, 2.25),
        (f64::NAN, 4.0),
        (5.0, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY),
        (1.0e100, -1.0e100),
    ];
    let length = count(shape);
    if dtype.is_complex() {
        return complex(
            shape,
            dtype,
            &(0..length)
                .map(|index| pattern[index % pattern.len()])
                .collect::<Vec<_>>(),
        );
    }
    let storage = match dtype {
        DType::I64 => ArrayStorage::I64(
            (0..length)
                .map(|index| [i64::MIN, (1_i64 << 53) + 1, i64::MAX, -7][index % 4])
                .collect(),
        ),
        DType::U64 => ArrayStorage::U64(
            (0..length)
                .map(|index| [0, (1_u64 << 53) + 1, u64::MAX, 17][index % 4])
                .collect(),
        ),
        _ => ArrayStorage::F64(
            (0..length)
                .map(|index| {
                    if dtype.is_integer() || dtype == DType::Bool {
                        index as f64 - 3.0
                    } else {
                        pattern[index % pattern.len()].0
                    }
                })
                .collect(),
        )
        .cast_to(dtype)
        .unwrap(),
    };
    UFuncArray::from_storage(shape.to_vec(), storage).unwrap()
}

fn destination(shape: &[usize], dtype: DType) -> UFuncArray {
    if dtype.is_complex() {
        complex(shape, dtype, &vec![(17.0, -19.0); count(shape)])
    } else {
        UFuncArray::from_storage(
            shape.to_vec(),
            ArrayStorage::F64(vec![17.0; count(shape)])
                .cast_to(dtype)
                .unwrap(),
        )
        .unwrap()
    }
}

fn indices(shape: &[usize], pattern: &[i64]) -> UFuncArray {
    UFuncArray::from_storage(
        shape.to_vec(),
        ArrayStorage::I64(
            (0..count(shape))
                .map(|index| pattern[index % pattern.len()])
                .collect(),
        ),
    )
    .unwrap()
}

fn input(array: &UFuncArray) -> Value {
    let mut shape = array.shape().to_vec();
    let values = if array.dtype().is_complex() {
        assert_eq!(shape.pop(), Some(2));
        json!(
            array
                .values()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|parts| { [parts[0].to_string(), parts[1].to_string()] })
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

fn error_category(error: UFuncError) -> String {
    match error {
        UFuncError::FloatingPoint { kind, .. } => return format!("float:{}", kind.as_str()),
        UFuncError::AxisOutOfBounds { .. } => return "axis".into(),
        _ => {}
    }
    let message = error.to_string();
    if message.contains("indices must be an integer") {
        "index-dtype"
    } else if message.contains("indices ndim") {
        "index-rank"
    } else if message.contains("non-axis dimension") {
        "index-shape"
    } else if message.contains("broadcast values") {
        "value-shape"
    } else if message.contains("empty array") || message.contains("out of bounds") {
        "bounds"
    } else {
        panic!("unexpected scatter error: {message}");
    }
    .into()
}

struct Outcome {
    array: UFuncArray,
    error: Option<String>,
    warnings: Vec<String>,
    calls: Vec<String>,
}

#[derive(Default)]
struct OracleBatch {
    cases: Vec<Value>,
    outcomes: Vec<Outcome>,
}

impl OracleBatch {
    fn record(
        &mut self,
        dst: &UFuncArray,
        mut case: Value,
        mode: FloatErrorMode,
        operation: impl FnOnce(&mut UFuncArray) -> Result<(), UFuncError>,
    ) {
        case["dst"] = input(dst);
        case["mode"] = json!(mode.as_str());
        self.cases.push(case);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let callback_calls = Arc::clone(&calls);
        let callback: FloatErrorCallback = Arc::new(move |event| {
            assert_eq!(event.op, "cast");
            callback_calls
                .lock()
                .unwrap()
                .push(event.kind.as_str().to_string());
        });
        let previous = seterrcall(Some(callback));
        take_float_error_events();
        let mut array = dst.clone();
        let error = {
            let _state = errstate(Some(mode), None, None, None, None);
            operation(&mut array).err().map(error_category)
        };
        let warnings = take_float_error_events()
            .into_iter()
            .map(|event| {
                assert_eq!(event.op, "cast");
                event.kind.as_str().to_string()
            })
            .collect();
        let _ = seterrcall(previous);
        let calls = calls.lock().unwrap().clone();
        self.outcomes.push(Outcome {
            array,
            error,
            warnings,
            calls,
        });
    }

    fn put(&mut self, dst: &UFuncArray, idx: &[i64], src: &UFuncArray, mode: FloatErrorMode) {
        self.record(
            dst,
            json!({"op": "put", "indices": idx, "src": input(src)}),
            mode,
            |array| array.put(idx, src),
        );
    }

    fn along(
        &mut self,
        dst: &UFuncArray,
        idx: &UFuncArray,
        src: &UFuncArray,
        axis: isize,
        mode: FloatErrorMode,
    ) {
        self.record(
            dst,
            json!({"op": "along", "indices": input(idx), "src": input(src), "axis": axis}),
            mode,
            |array| array.put_along_axis(idx, src, axis),
        );
    }

    fn assert_matches_numpy(self) {
        assert!(!self.cases.is_empty());
        let script = r#"
import json, sys, warnings
import numpy as np

def load(item):
    dtype = np.dtype(item['dtype'])
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        if dtype.kind == 'c':
            array = np.empty(len(item['values']), dtype=dtype)
            array.real = [float(pair[0]) for pair in item['values']]
            array.imag = [float(pair[1]) for pair in item['values']]
        else:
            convert = int if dtype.kind in 'biu' else float
            array = np.array([convert(value) for value in item['values']], dtype=dtype)
    return array.reshape(item['shape'])

def kind(message):
    return {'overflow':'over', 'underflow':'under', 'invalid':'invalid', 'divide':'divide'}[message.split()[0]]

def category(error):
    text = str(error)
    if isinstance(error, FloatingPointError): return 'float:' + kind(text)
    if isinstance(error, np.exceptions.AxisError): return 'axis'
    if 'must be an integer array' in text: return 'index-dtype'
    if 'same number of dimensions' in text: return 'index-rank'
    if 'indexing arrays could not be broadcast' in text: return 'index-shape'
    if 'value array of shape' in text: return 'value-shape'
    if isinstance(error, IndexError): return 'bounds'
    raise AssertionError((type(error).__name__, text))

rows = []
for case in json.load(sys.stdin):
    dst, src = load(case['dst']), load(case['src'])
    idx = case['indices'] if case['op'] == 'put' else load(case['indices'])
    calls = []
    previous = np.seterrcall(lambda message, flags: calls.append(kind(message)))
    failure = None
    with warnings.catch_warnings(record=True) as caught, np.errstate(all=case['mode']):
        warnings.simplefilter('always')
        try:
            if case['op'] == 'put': np.put(dst, idx, src)
            else: np.put_along_axis(dst, idx, src, axis=case['axis'])
        except (TypeError, ValueError, IndexError, FloatingPointError) as error:
            failure = category(error)
    np.seterrcall(previous)
    messages = [kind(str(item.message)) for item in caught
                if issubclass(item.category, RuntimeWarning)
                and not issubclass(item.category, np.exceptions.ComplexWarning)]
    if dst.dtype.kind == 'c':
        parts = np.empty(2 * dst.size, dtype=np.float64)
        parts[0::2], parts[1::2] = dst.real.ravel(), dst.imag.ravel()
    else:
        parts = dst.astype(np.float64).ravel()
    rows.append(dict(shape=list(dst.shape), dtype=str(dst.dtype), error=failure,
                     warnings=messages, calls=calls, bits=parts.view(np.uint64).tolist(),
                     integers=[str(int(value)) for value in dst.flat] if dst.dtype.kind in 'biu' else None))
print(json.dumps(rows))
"#;
        let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(python)
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("launch live NumPy scatter oracle");
        let mut stdin = child.stdin.take().unwrap();
        serde_json::to_writer(&mut stdin, &self.cases).unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy scatter oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            expected.len(),
            self.cases.len(),
            "oracle returned every case"
        );
        for ((case, outcome), expected) in self.cases.iter().zip(self.outcomes).zip(expected) {
            let Outcome {
                array,
                error,
                warnings,
                calls,
            } = outcome;
            assert_eq!(json!(error), expected["error"], "{case}: error precedence");
            assert_eq!(
                json!(warnings),
                expected["warnings"],
                "{case}: floating warnings"
            );
            assert_eq!(
                json!(calls),
                expected["calls"],
                "{case}: floating callbacks"
            );
            let mut shape = array.shape().to_vec();
            if array.dtype().is_complex() {
                assert_eq!(shape.pop(), Some(2));
            }
            assert_eq!(json!(shape), expected["shape"], "{case}: logical shape");
            assert_eq!(
                dtype_name(array.dtype()),
                expected["dtype"],
                "{case}: dtype"
            );
            if array.dtype().is_integer() || array.dtype() == DType::Bool {
                let storage = array.to_storage().unwrap();
                let values = (0..storage.len())
                    .map(|index| storage.get_i128(index).unwrap().to_string())
                    .collect::<Vec<_>>();
                assert_eq!(
                    json!(values),
                    expected["integers"],
                    "{case}: exact integers"
                );
            }
            let bits = expected["bits"].as_array().unwrap();
            assert_eq!(
                array.values().len(),
                bits.len(),
                "{case}: physical component count"
            );
            for (index, (&actual, expected)) in array.values().iter().zip(bits).enumerate() {
                let expected = f64::from_bits(expected.as_u64().unwrap());
                if expected.is_nan() {
                    assert!(actual.is_nan(), "{case}: component {index} expected NaN");
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{case}: component {index}, got {actual}, expected {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn put_numeric_casts_complex_pairs_and_exact_integers_match_numpy() {
    let mut batch = OracleBatch::default();
    for dst_dtype in NUMERIC {
        let dst = destination(&[3, 4], dst_dtype);
        for src_dtype in NUMERIC {
            batch.put(
                &dst,
                &[0, 11, -1, 5, 0, -12, 4, 5, 3, 2, 6, 10, 7],
                &sample(&[2, 4], src_dtype),
                FloatErrorMode::Ignore,
            );
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn put_scalar_empty_values_and_partial_index_errors_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [
        DType::F64,
        DType::I64,
        DType::U64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for shape in [&[][..], &[3], &[2, 0]] {
            let dst = destination(shape, dtype);
            for src_shape in [&[][..], &[0], &[2]] {
                let src = sample(src_shape, DType::Complex128);
                for idx in [
                    &[][..],
                    &[0],
                    &[-1],
                    &[0, 1, 99],
                    &[0, i64::MIN],
                    &[i64::MAX],
                ] {
                    batch.put(&dst, idx, &src, FloatErrorMode::Ignore);
                }
            }
        }
    }
    // Exercise the existing duplicate-run F64 fast path, including its error
    // boundary: only writes before the invalid run may affect the destination.
    batch.put(
        &destination(&[3], DType::F64),
        &[0, 0, 0, 1, 1, 1, -1, -1, 9, 9],
        &sample(&[3], DType::F64),
        FloatErrorMode::Ignore,
    );
    batch.assert_matches_numpy();
}

#[test]
fn along_axis_numeric_dtypes_and_broadcasted_logical_shapes_match_numpy() {
    let mut batch = OracleBatch::default();
    for dst_dtype in NUMERIC {
        for src_dtype in NUMERIC {
            for dst_shape in [&[2, 4][..], &[1, 4]] {
                batch.along(
                    &destination(dst_shape, dst_dtype),
                    &indices(&[2, 3], &[0, -1, 1, 2, 0, 2]),
                    &sample(&[1, 3], src_dtype),
                    -1,
                    FloatErrorMode::Ignore,
                );
            }
        }
    }
    for dtype in [DType::Complex64, DType::Complex128] {
        for src_shape in [&[][..], &[2, 1], &[1, 2, 1], &[1, 1, 2, 1]] {
            batch.along(
                &destination(&[2, 3, 4], dtype),
                &indices(&[1, 2, 4], &[0, -1, 1, 0]),
                &sample(src_shape, DType::Complex128),
                -2,
                FloatErrorMode::Ignore,
            );
        }
    }
    // Broadcasting the destination repeats physical rows. Different source
    // rows must overwrite them in C order, including duplicate indices.
    for dtype in [DType::I64, DType::Complex64, DType::Complex128] {
        for source_dtype in [DType::I64, DType::U64, DType::Complex128] {
            batch.along(
                &destination(&[1, 4], dtype),
                &indices(&[2, 3], &[0, 3, 1, 2, 0, 2]),
                &sample(&[2, 3], source_dtype),
                1,
                FloatErrorMode::Ignore,
            );
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn along_axis_empty_outputs_and_validation_precedence_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::F32, DType::I64, DType::Complex64] {
        let src = sample(&[], DType::Complex128);
        for axis in [-3, -2, -1, 0, 1, 2] {
            for idx in [
                indices(&[2, 2], &[0, 99]),
                indices(&[0], &[0]),
                sample(&[2, 2], DType::F64),
                sample(&[2, 2], DType::Bool),
            ] {
                batch.along(
                    &destination(&[2, 3], dtype),
                    &idx,
                    &src,
                    axis,
                    FloatErrorMode::Ignore,
                );
            }
        }
        batch.along(
            &destination(&[2, 3], dtype),
            &indices(&[3, 2], &[0]),
            &src,
            1,
            FloatErrorMode::Ignore,
        );
        batch.along(
            &destination(&[2, 3], dtype),
            &indices(&[1, 2], &[0, 99]),
            &sample(&[3], DType::F64),
            -1,
            FloatErrorMode::Ignore,
        );
        batch.along(
            &destination(&[2, 0, 3], dtype),
            &indices(&[1, 0, 3], &[99]),
            &sample(&[1, 3], DType::Complex128),
            1,
            FloatErrorMode::Ignore,
        );
        batch.along(
            &destination(&[2, 3], dtype),
            &indices(&[2, 0], &[99]),
            &sample(&[0], DType::Complex128),
            -1,
            FloatErrorMode::Ignore,
        );
        batch.along(
            &destination(&[1, 0], dtype),
            &indices(&[3, 1], &[0]),
            &src,
            -1,
            FloatErrorMode::Ignore,
        );
        batch.along(
            &destination(&[], dtype),
            &indices(&[], &[0]),
            &src,
            -1,
            FloatErrorMode::Ignore,
        );
    }
    batch.assert_matches_numpy();
}

#[test]
fn scatter_cast_errors_preserve_numpy_mutation_timing() {
    let mut batch = OracleBatch::default();
    for mode in [
        FloatErrorMode::Ignore,
        FloatErrorMode::Warn,
        FloatErrorMode::Raise,
        FloatErrorMode::Call,
    ] {
        for dtype in [DType::I64, DType::F16, DType::F32, DType::Complex64] {
            for value in [f64::NAN, f64::INFINITY, 1.0e100, 1.0e-50] {
                for shape in [&[][..], &[1], &[2]] {
                    let source =
                        UFuncArray::new(shape.to_vec(), vec![value; count(shape)], DType::F64)
                            .unwrap();
                    let dst = destination(&[3], dtype);
                    for idx in [&[][..], &[0, 2], &[0, 3]] {
                        batch.put(&dst, idx, &source, mode);
                        batch.along(&dst, &indices(&[idx.len()], idx), &source, -1, mode);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn along_axis_uint32_cast_runs_follow_broadcast_source_layout() {
    let witness = [
        f64::NAN,
        f64::INFINITY,
        -1.0,
        -2_147_483_649.0,
        2_147_483_648.0,
        4_294_967_295.0,
        4_294_967_296.0,
        1.0e100,
    ];
    let mut batch = OracleBatch::default();
    for source_dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        for (dst_shape, idx_shape, src_shape, axis) in [
            (vec![8], vec![8], vec![8], 0),
            (vec![2, 8], vec![2, 8], vec![1, 8], 0),
            (vec![8, 2], vec![8, 2], vec![8, 1], 1),
            (vec![2, 4, 4], vec![2, 4, 4], vec![2, 1, 4], 1),
            (vec![2, 4, 8], vec![2, 1, 1], vec![1, 4, 8], 0),
            (vec![2, 4, 8], vec![2, 4, 8], vec![4, 8], 0),
            (vec![1, 4], vec![2, 3], vec![2, 3], 1),
            (vec![2, 8], vec![3, 1], vec![3, 8], 0),
            (vec![2, 8], vec![1, 2], vec![2, 2], 1),
            (vec![1, 2, 8], vec![1, 2, 1], vec![1, 2, 8], 0),
            (vec![2, 2, 8], vec![2, 2, 1], vec![2, 2, 8], 0),
            (vec![2, 1, 4, 8], vec![2, 3, 1, 1], vec![1, 1, 4, 8], 1),
            (vec![1, 1, 1, 32], vec![2, 3, 1, 16], vec![2, 3, 1, 16], 3),
        ] {
            let source_values = (0..count(&src_shape))
                .map(|index| witness[index % witness.len()])
                .collect::<Vec<_>>();
            let src = if source_dtype.is_complex() {
                complex(
                    &src_shape,
                    source_dtype,
                    &source_values
                        .iter()
                        .map(|&real| (real, f64::NAN))
                        .collect::<Vec<_>>(),
                )
            } else {
                UFuncArray::from_storage(
                    src_shape,
                    ArrayStorage::F64(source_values)
                        .cast_to(source_dtype)
                        .unwrap(),
                )
                .unwrap()
            };
            let idx = if dst_shape.len() == 1 {
                indices(&idx_shape, &[2, 4, 5, 6, 1, 7, 0, 3])
            } else {
                indices(
                    &idx_shape,
                    &(0..count(&idx_shape))
                        .map(|index| (index % dst_shape[axis as usize]) as i64)
                        .collect::<Vec<_>>(),
                )
            };
            for mode in [
                FloatErrorMode::Ignore,
                FloatErrorMode::Warn,
                FloatErrorMode::Raise,
                FloatErrorMode::Call,
            ] {
                batch.along(&destination(&dst_shape, DType::U32), &idx, &src, axis, mode);
                // Converting an index operand to intp changes the buffered
                // iterator's cost, which can change its value-cast run size.
                batch.along(
                    &destination(&dst_shape, DType::U32),
                    &idx.astype(DType::U64),
                    &src,
                    axis,
                    mode,
                );
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn along_axis_uint32_casts_keep_buffer_and_scalar_tail_boundaries() {
    let witness = [
        f64::NAN,
        f64::INFINITY,
        -2_147_483_649.0,
        4_294_967_295.0,
        -1.0,
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        for (shape, index_shape, axis) in [
            (vec![8195], vec![8195], 0),
            (vec![2, 3, 1367], vec![2, 3, 1367], 1),
        ] {
            let values = (0..count(&shape))
                .map(|index| witness[index % witness.len()])
                .collect::<Vec<_>>();
            let source = if dtype.is_complex() {
                complex(
                    &shape,
                    dtype,
                    &values.iter().map(|&value| (value, 2.0)).collect::<Vec<_>>(),
                )
            } else {
                UFuncArray::from_storage(
                    shape.clone(),
                    ArrayStorage::F64(values).cast_to(dtype).unwrap(),
                )
                .unwrap()
            };
            let idx = indices(
                &index_shape,
                &(0..count(&index_shape))
                    .map(|index| (index % shape[axis as usize]) as i64)
                    .collect::<Vec<_>>(),
            );
            batch.along(
                &destination(&shape, DType::U32),
                &idx,
                &source,
                axis,
                FloatErrorMode::Raise,
            );
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn along_axis_unsigned_indices_use_exact_intp_conversion() {
    let mut batch = OracleBatch::default();
    let idx =
        UFuncArray::from_storage(vec![1, 3], ArrayStorage::U64(vec![0, u64::MAX, 1])).unwrap();
    batch.along(
        &destination(&[2, 3], DType::Complex128),
        &idx,
        &sample(&[2, 3], DType::U64),
        -1,
        FloatErrorMode::Ignore,
    );
    let invalid =
        UFuncArray::from_storage(vec![1, 2], ArrayStorage::U64(vec![0, u64::MAX - 4])).unwrap();
    batch.along(
        &destination(&[2, 3], DType::Complex64),
        &invalid,
        &sample(&[], DType::I64),
        -1,
        FloatErrorMode::Ignore,
    );
    batch.assert_matches_numpy();
}

#[test]
fn scatter_rejects_malformed_complex_layouts_without_writes() {
    let malformed = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::Complex128).unwrap();
    let mut dst = destination(&[3], DType::Complex128);
    let before = dst.values().to_vec();
    assert!(dst.put(&[0], &malformed).is_err());
    assert_eq!(dst.values(), before);
    assert!(
        dst.put_along_axis(&indices(&[1], &[0]), &malformed, 0)
            .is_err()
    );
    assert_eq!(dst.values(), before);
    let mut malformed_dst = malformed;
    let before = malformed_dst.values().to_vec();
    assert!(malformed_dst.put(&[0], &sample(&[], DType::F64)).is_err());
    assert_eq!(malformed_dst.values(), before);
    assert!(
        malformed_dst
            .put_along_axis(&indices(&[1], &[0]), &sample(&[], DType::F64), 0)
            .is_err()
    );
    assert_eq!(malformed_dst.values(), before);
}
