//! place/putmask must cast and cycle complete logical array elements.

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

#[derive(Clone, Copy)]
enum Operation {
    Place,
    Putmask,
}

impl Operation {
    fn name(self) -> &'static str {
        match self {
            Self::Place => "place",
            Self::Putmask => "putmask",
        }
    }

    fn apply(
        self,
        dst: &mut UFuncArray,
        mask: &UFuncArray,
        values: &UFuncArray,
    ) -> Result<(), UFuncError> {
        match self {
            Self::Place => dst.place(mask, values),
            Self::Putmask => dst.putmask(mask, values),
        }
    }
}

const OPERATIONS: [Operation; 2] = [Operation::Place, Operation::Putmask];

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
        _ => panic!("non-numeric mask-mutation fixture"),
    }
}

fn count(shape: &[usize]) -> usize {
    if shape.contains(&0) {
        0
    } else {
        shape.iter().product()
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

fn source(shape: &[usize], dtype: DType) -> UFuncArray {
    let pattern = [
        (-0.0, 0.0),
        (1.0 + 2.0_f64.powi(-24), -1.0 - 2.0_f64.powi(-24)),
        (1.0e-50, -1.0e-50),
        (f64::NAN, -3.5),
        (-2.25, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY),
        (0.0, -0.0),
    ];
    if dtype.is_complex() {
        return complex(
            shape,
            &(0..count(shape))
                .map(|index| pattern[index % pattern.len()])
                .collect::<Vec<_>>(),
            dtype,
        );
    }
    let storage = match dtype {
        DType::I64 => ArrayStorage::I64(
            (0..count(shape))
                .map(|index| [i64::MIN, (1_i64 << 53) + 1, i64::MAX, -7][index % 4])
                .collect(),
        ),
        DType::U64 => ArrayStorage::U64(
            (0..count(shape))
                .map(|index| [0, (1_u64 << 53) + 1, u64::MAX, 7][index % 4])
                .collect(),
        ),
        _ => ArrayStorage::F64(
            (0..count(shape))
                .map(|index| {
                    if dtype.is_integer() || dtype == DType::Bool {
                        [-3.0, 0.0, 1.0, 7.0][index % 4]
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
        complex(shape, &vec![(17.0, -19.0); count(shape)], dtype)
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

fn bool_mask(shape: &[usize], pattern: &[bool]) -> UFuncArray {
    UFuncArray::from_storage(
        shape.to_vec(),
        ArrayStorage::Bool(
            (0..count(shape))
                .map(|index| pattern[index % pattern.len()])
                .collect(),
        ),
    )
    .unwrap()
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

fn error_category(error: UFuncError) -> String {
    if let UFuncError::FloatingPoint { kind, .. } = error {
        return format!("float:{}", kind.as_str());
    }
    let message = error.to_string();
    if message.contains("mask size") || message.contains("same size") {
        "mask-size"
    } else if message.contains("cannot cast") {
        "cast-dtype"
    } else if message.contains("must not be empty") {
        "empty-values"
    } else {
        panic!("unexpected mask-mutation error: {message}");
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
    fn push(&mut self, op: Operation, dst: &UFuncArray, mask: &UFuncArray, values: &UFuncArray) {
        self.push_policy(op, dst, mask, values, FloatErrorMode::Ignore);
    }

    fn push_policy(
        &mut self,
        op: Operation,
        dst: &UFuncArray,
        mask: &UFuncArray,
        values: &UFuncArray,
        mode: FloatErrorMode,
    ) {
        let mode_name = match mode {
            FloatErrorMode::Ignore => "ignore",
            FloatErrorMode::Warn => "warn",
            FloatErrorMode::Raise => "raise",
            FloatErrorMode::Call => "call",
            _ => panic!("unsupported oracle policy"),
        };
        self.cases.push(json!({
            "op": op.name(), "dst": oracle_input(dst), "mask": oracle_input(mask),
            "values": oracle_input(values), "mode": mode_name,
        }));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let callback_calls = Arc::clone(&calls);
        let callback: FloatErrorCallback = Arc::new(move |event| {
            callback_calls
                .lock()
                .unwrap()
                .push(event.kind.as_str().to_string());
        });
        let previous_callback = seterrcall(Some(callback));
        take_float_error_events();
        let mut array = dst.clone();
        let error = {
            let _errors = errstate(Some(mode), None, None, None, None);
            op.apply(&mut array, mask, values).err().map(error_category)
        };
        let warnings = take_float_error_events()
            .into_iter()
            .map(|event| event.kind.as_str().to_string())
            .collect();
        let _ = seterrcall(previous_callback);
        let calls = calls.lock().unwrap().clone();
        self.outcomes.push(Outcome {
            array,
            error,
            warnings,
            calls,
        });
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
            result = np.empty(len(item['values']), dtype=dtype)
            result.real = [float(parts[0]) for parts in item['values']]
            result.imag = [float(parts[1]) for parts in item['values']]
        else:
            convert = int if dtype.kind in 'biu' else float
            result = np.array([convert(value) for value in item['values']], dtype=dtype)
    return result.reshape(item['shape'])

def float_kind(message):
    return {'overflow': 'over', 'underflow': 'under', 'invalid': 'invalid', 'divide': 'divide'}[message.split()[0]]

def category(error):
    text = str(error)
    if isinstance(error, FloatingPointError): return 'float:' + float_kind(text)
    if 'mask and data must be the same size' in text: return 'mask-size'
    if (text.startswith(('Cannot cast array data from ', 'Cannot cast scalar from '))
            and "according to the rule 'safe'" in text): return 'cast-dtype'
    if 'Cannot insert from an empty array' in text: return 'empty-values'
    raise AssertionError((type(error).__name__, text))

rows = []
for case in json.load(sys.stdin):
    dst, mask, values = load(case['dst']), load(case['mask']), load(case['values'])
    calls = []
    previous_callback = np.seterrcall(lambda message, flags: calls.append(float_kind(message)))
    failure = None
    with warnings.catch_warnings(record=True) as caught, np.errstate(all=case['mode']):
        warnings.simplefilter('always')
        try: getattr(np, case['op'])(dst, mask, values)
        except (TypeError, ValueError, FloatingPointError) as error: failure = category(error)
    np.seterrcall(previous_callback)
    messages = [float_kind(str(warning.message)) for warning in caught
                if issubclass(warning.category, RuntimeWarning)
                and not issubclass(warning.category, np.exceptions.ComplexWarning)]
    if dst.dtype.kind == 'c':
        parts = np.empty(dst.size * 2, dtype=np.float64)
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
            .expect("launch live NumPy place/putmask oracle");
        let mut stdin = child.stdin.take().unwrap();
        serde_json::to_writer(&mut stdin, &self.cases).unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy mask-mutation oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.cases.len());
        assert_eq!(expected.len(), self.outcomes.len());
        for ((case, outcome), expected) in self.cases.iter().zip(self.outcomes).zip(expected) {
            assert_eq!(
                json!(outcome.error),
                expected["error"],
                "{case}: error precedence"
            );
            assert_eq!(
                json!(outcome.warnings),
                expected["warnings"],
                "{case}: warnings"
            );
            assert_eq!(json!(outcome.calls), expected["calls"], "{case}: callbacks");
            let actual = outcome.array;
            let mut shape = actual.shape().to_vec();
            if actual.dtype().is_complex() {
                assert_eq!(shape.pop(), Some(2));
            }
            assert_eq!(json!(shape), expected["shape"], "{case}: logical shape");
            assert_eq!(
                dtype_name(actual.dtype()),
                expected["dtype"],
                "{case}: dtype"
            );
            if actual.dtype().is_integer() || actual.dtype() == DType::Bool {
                let storage = actual.to_storage().unwrap();
                let integers = (0..storage.len())
                    .map(|index| storage.get_i128(index).unwrap().to_string())
                    .collect::<Vec<_>>();
                assert_eq!(
                    json!(integers),
                    expected["integers"],
                    "{case}: exact integers"
                );
            }
            let bits = expected["bits"].as_array().unwrap();
            assert_eq!(
                actual.values().len(),
                bits.len(),
                "{case}: stored components"
            );
            for (index, (&value, bits)) in actual.values().iter().zip(bits).enumerate() {
                let expected = f64::from_bits(bits.as_u64().unwrap());
                if expected.is_nan() {
                    assert!(value.is_nan(), "{case}: component {index} expected NaN");
                } else {
                    assert_eq!(
                        value.to_bits(),
                        expected.to_bits(),
                        "{case}: component {index}, got {value}, expected {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn masked_numeric_mutations_obey_safe_casting_and_exact_values() {
    let mut batch = OracleBatch::default();
    for op in OPERATIONS {
        for target in NUMERIC {
            for input in NUMERIC {
                let dst = destination(&[2, 5], target);
                let values = source(&[7], input);
                for pattern in [&[true][..], &[false][..], &[false, true, true, false][..]] {
                    batch.push(op, &dst, &bool_mask(&[10], pattern), &values);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn masked_complex_mutations_flatten_logical_shapes_and_keep_distinct_cycles() {
    let mut batch = OracleBatch::default();
    let shapes: &[&[usize]] = &[&[], &[1], &[7], &[2, 3], &[2, 1, 3], &[0, 3]];
    let value_shapes: &[&[usize]] = &[&[], &[1], &[2], &[2, 2], &[1, 7], &[0]];
    for op in OPERATIONS {
        for target in [DType::Complex64, DType::Complex128] {
            for input in [DType::F32, DType::Complex64, DType::Complex128] {
                for shape in shapes {
                    let dst = destination(shape, target);
                    for mask_shape in [shape.to_vec(), vec![count(shape)], vec![1, count(shape)]] {
                        let mask = bool_mask(&mask_shape, &[false, true, true, false, true]);
                        for values_shape in value_shapes {
                            batch.push(op, &dst, &mask, &source(values_shape, input));
                        }
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn masked_mutations_use_mask_truth_at_the_declared_source_precision() {
    let mut batch = OracleBatch::default();
    let parts = [
        (0.0, 1.0e-50),
        (1.0e-50, 0.0),
        (0.0, -2.0),
        (-0.0, 0.0),
        (f64::NAN, 0.0),
        (0.0, f64::NAN),
        (f64::INFINITY, 0.0),
        (0.0, f64::NEG_INFINITY),
    ];
    let masks = [
        complex(&[2, 4], &parts, DType::Complex64),
        complex(&[8], &parts, DType::Complex128),
        UFuncArray::new(
            vec![8],
            parts.iter().map(|&(real, _)| real).collect(),
            DType::F32,
        )
        .unwrap(),
        source(&[8], DType::I64),
        source(&[2, 2, 2], DType::U64),
    ];
    for op in OPERATIONS {
        for dtype in [
            DType::Complex64,
            DType::Complex128,
            DType::F64,
            DType::I64,
            DType::U64,
        ] {
            for mask in &masks {
                batch.push(op, &destination(&[4, 2], dtype), mask, &source(&[3], dtype));
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn masked_mutations_validate_size_then_cast_before_empty_or_false_noops() {
    let mut batch = OracleBatch::default();
    for op in OPERATIONS {
        for dtype in [DType::F32, DType::I64, DType::Complex64, DType::Complex128] {
            for shape in [&[2][..], &[0][..], &[][..]] {
                let dst = destination(shape, dtype);
                for mask_shape in [&[0][..], &[1][..], &[2][..], &[3][..]] {
                    for selected in [false, true] {
                        let mask = bool_mask(mask_shape, &[selected]);
                        for source_dtype in [DType::F64, DType::Complex128, dtype] {
                            for value_shape in [&[0][..], &[1][..]] {
                                batch.push(op, &dst, &mask, &source(value_shape, source_dtype));
                            }
                        }
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn masked_safe_casts_do_not_raise_or_dispatch_floating_errors() {
    let mut batch = OracleBatch::default();
    for op in OPERATIONS {
        for mode in [
            FloatErrorMode::Warn,
            FloatErrorMode::Raise,
            FloatErrorMode::Call,
        ] {
            for dtype in NUMERIC {
                for selected in [false, true] {
                    batch.push_policy(
                        op,
                        &destination(&[7], DType::Complex128),
                        &bool_mask(&[7], &[selected]),
                        &source(&[7], dtype),
                        mode,
                    );
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn masked_mutations_reject_malformed_complex_layouts_before_writing() {
    let malformed = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::Complex128).unwrap();
    let valid = destination(&[3], DType::Complex128);
    let mask = bool_mask(&[3], &[true]);
    let values = source(&[1], DType::Complex128);
    for op in OPERATIONS {
        let mut dst = malformed.clone();
        assert!(op.apply(&mut dst, &mask, &values).is_err());
        assert_eq!(dst.values(), malformed.values());
        let mut dst = valid.clone();
        assert!(op.apply(&mut dst, &malformed, &values).is_err());
        assert_eq!(dst.values(), valid.values());
        assert!(op.apply(&mut dst, &mask, &malformed).is_err());
        assert_eq!(dst.values(), valid.values());
    }
}
