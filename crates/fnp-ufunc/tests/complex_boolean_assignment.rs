//! Scalar Boolean assignment uses logical masks and Python-float packing.

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

const MODES: [FloatErrorMode; 6] = [
    FloatErrorMode::Ignore,
    FloatErrorMode::Warn,
    FloatErrorMode::Raise,
    FloatErrorMode::Call,
    FloatErrorMode::Print,
    FloatErrorMode::Log,
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
        _ => panic!("non-numeric Boolean assignment fixture"),
    }
}

fn count(shape: &[usize]) -> usize {
    if shape.contains(&0) {
        0
    } else {
        shape.iter().product()
    }
}

fn destination(shape: &[usize], dtype: DType) -> UFuncArray {
    let count = count(shape);
    if dtype.is_complex() {
        let mut physical_shape = shape.to_vec();
        physical_shape.push(2);
        let parts = [(-0.0, -0.0), (17.0, -19.0), (23.0, 29.0)];
        return UFuncArray::new(
            physical_shape,
            (0..count)
                .flat_map(|index| {
                    let (real, imaginary) = parts[index % parts.len()];
                    [real, imaginary]
                })
                .collect(),
            dtype,
        )
        .unwrap();
    }
    let storage = match dtype {
        DType::I64 => ArrayStorage::I64(
            (0..count)
                .map(|index| [i64::MIN, (1_i64 << 53) + 1, i64::MAX, -7][index % 4])
                .collect(),
        ),
        DType::U64 => ArrayStorage::U64(
            (0..count)
                .map(|index| [0, (1_u64 << 53) + 1, u64::MAX, 7][index % 4])
                .collect(),
        ),
        _ => ArrayStorage::F64(
            (0..count)
                .map(|index| [-0.0, 17.0, 23.0][index % 3])
                .collect(),
        )
        .cast_to(dtype)
        .unwrap(),
    };
    UFuncArray::from_storage(shape.to_vec(), storage).unwrap()
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
    if message.contains("mask rank") {
        "mask-rank"
    } else if message.contains("mask size/shape") {
        "mask-shape"
    } else if message.contains("float NaN") {
        "scalar-nan"
    } else if message.contains("float infinity") {
        "scalar-infinity"
    } else if message.contains("out of bounds") {
        "scalar-range"
    } else {
        panic!("unexpected scalar Boolean assignment error: {message}");
    }
    .into()
}

struct Outcome {
    array: UFuncArray,
    error: Option<String>,
    reports: Vec<String>,
    calls: Vec<String>,
}

#[derive(Default)]
struct OracleBatch {
    cases: Vec<Value>,
    outcomes: Vec<Outcome>,
}

impl OracleBatch {
    fn push(&mut self, dst: &UFuncArray, mask: &UFuncArray, value: f64) {
        self.push_policy(dst, mask, value, FloatErrorMode::Ignore);
    }

    fn push_policy(
        &mut self,
        dst: &UFuncArray,
        mask: &UFuncArray,
        value: f64,
        mode: FloatErrorMode,
    ) {
        self.cases.push(json!({
            "dst": oracle_input(dst), "mask": oracle_input(mask),
            "value": value.to_string(), "mode": mode.as_str(),
        }));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let callback_calls = Arc::clone(&calls);
        let callback: FloatErrorCallback = Arc::new(move |event| {
            assert_eq!(event.op, "cast");
            assert_eq!(event.mode, FloatErrorMode::Call);
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
            array.boolean_set(mask, value).err().map(error_category)
        };
        let reports = take_float_error_events()
            .into_iter()
            .map(|event| {
                assert_eq!(event.op, "cast");
                assert_eq!(event.mode, mode);
                event.kind.as_str().to_string()
            })
            .collect();
        let _ = seterrcall(previous_callback);
        let calls = calls.lock().unwrap().clone();
        self.outcomes.push(Outcome {
            array,
            error,
            reports,
            calls,
        });
    }

    fn assert_matches_numpy(self) {
        assert!(!self.cases.is_empty());
        let script = r#"
import json, os, sys, warnings
import numpy as np

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

def float_kind(message):
    first = message.removeprefix('Warning: ').split()[0]
    return {'overflow': 'over', 'underflow': 'under', 'invalid': 'invalid', 'divide': 'divide'}[first]

def category(error):
    message = str(error)
    if isinstance(error, FloatingPointError): return 'float:' + float_kind(message)
    if 'too many indices' in message: return 'mask-rank'
    if 'boolean index did not match' in message: return 'mask-shape'
    if 'float NaN' in message: return 'scalar-nan'
    if 'float infinity' in message: return 'scalar-infinity'
    if 'out of bounds' in message or 'Python int too large' in message: return 'scalar-range'
    raise AssertionError((type(error).__name__, message))

class Sink:
    def __init__(self): self.calls, self.logs = [], []
    def __call__(self, message, flags): self.calls.append(float_kind(message))
    def write(self, message): self.logs.append(float_kind(message))

rows = []
for case in json.load(sys.stdin):
    dst, mask = load(case['dst']), load(case['mask'])
    # This API accepts a Python float, not a NumPy scalar or an ndarray.
    # Those other source types have different underflow/range semantics.
    value = float(case['value'])
    sink = Sink()
    previous_callback = np.seterrcall(sink)
    failure, printed = None, ''
    if case['mode'] == 'print':
        readfd, writefd = os.pipe()
        saved_stderr = os.dup(2)
        os.dup2(writefd, 2)
        os.close(writefd)
    try:
        with warnings.catch_warnings(record=True) as caught, np.errstate(all=case['mode']):
            warnings.simplefilter('always')
            try: dst[mask] = value
            except (TypeError, ValueError, IndexError, OverflowError, FloatingPointError) as error:
                failure = category(error)
    finally:
        if case['mode'] == 'print':
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
            printed = os.read(readfd, 65536).decode()
            os.close(readfd)
        np.seterrcall(previous_callback)
    reports = [float_kind(str(warning.message)) for warning in caught
               if issubclass(warning.category, RuntimeWarning)]
    reports.extend(sink.logs)
    reports.extend(float_kind(line) for line in printed.splitlines() if line)
    if dst.dtype.kind == 'c':
        parts = np.empty(dst.size * 2, dtype=np.float64)
        parts[0::2], parts[1::2] = dst.real.ravel(), dst.imag.ravel()
    else:
        parts = dst.astype(np.float64).ravel()
    rows.append(dict(shape=list(dst.shape), dtype=str(dst.dtype), error=failure,
                     reports=reports, calls=sink.calls, bits=parts.view(np.uint64).tolist(),
                     integers=[str(int(item)) for item in dst.flat] if dst.dtype.kind in 'biu' else None))
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
            .expect("launch live NumPy scalar Boolean assignment oracle");
        let mut stdin = child.stdin.take().unwrap();
        serde_json::to_writer(&mut stdin, &self.cases).unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy Boolean assignment oracle failed: {}",
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
                json!(outcome.reports),
                expected["reports"],
                "{case}: reports"
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
                "{case}: physical components"
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
fn scalar_boolean_assignment_handles_logical_prefix_and_scalar_masks() {
    let mut batch = OracleBatch::default();
    let shapes: &[&[usize]] = &[&[], &[1], &[3], &[2, 3], &[2, 1, 3], &[2, 0, 3], &[0, 3]];
    for dtype in NUMERIC {
        for &shape in shapes {
            let dst = destination(shape, dtype);
            for rank in 0..=shape.len() {
                for pattern in [&[true][..], &[false][..], &[false, true, true][..]] {
                    let mask = bool_mask(&shape[..rank], pattern);
                    for value in [-0.0, 2.5, -1.5] {
                        batch.push(&dst, &mask, value);
                    }
                }
            }
            // NumPy also accepts empty mask dimensions on nonempty axes.
            for axis in 0..shape.len() {
                let mut mask_shape = shape.to_vec();
                mask_shape[axis] = 0;
                let mask = bool_mask(&mask_shape, &[false]);
                batch.push(&dst, &mask, 2.5);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn integer_scalar_assignment_truncates_then_checks_exact_range() {
    let mut batch = OracleBatch::default();
    let bounds = [
        (DType::I8, -128.0_f64, 128.0_f64),
        (DType::I16, -32_768.0, 32_768.0),
        (DType::I32, -2_147_483_648.0, 2_147_483_648.0),
        (DType::I64, -2.0_f64.powi(63), 2.0_f64.powi(63)),
        (DType::U8, 0.0, 256.0),
        (DType::U16, 0.0, 65_536.0),
        (DType::U32, 0.0, 4_294_967_296.0),
        (DType::U64, 0.0, 2.0_f64.powi(64)),
    ];
    for (dtype, lower, upper) in bounds {
        let dst = destination(&[2, 3], dtype);
        let masks = [
            bool_mask(&[2, 3], &[false, true, false]),
            bool_mask(&[2, 3], &[false]),
            bool_mask(&[], &[true]),
            bool_mask(&[], &[false]),
            bool_mask(&[0, 3], &[false]),
        ];
        for value in [
            lower.next_down(),
            lower,
            lower.next_up(),
            upper.next_down(),
            upper,
            upper.next_up(),
            -1.1,
            -0.9,
            1.9,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            2.0_f64.powi(53).next_down(),
            2.0_f64.powi(53),
            2.0_f64.powi(53).next_up(),
        ] {
            for mask in &masks {
                batch.push(&dst, mask, value);
            }
            batch.push(
                &destination(&[0, 3], dtype),
                &bool_mask(&[0], &[false]),
                value,
            );
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn scalar_packing_reports_overflow_without_underflow_in_every_mode() {
    let mut batch = OracleBatch::default();
    for dtype in NUMERIC {
        for mode in MODES {
            for value in [
                1.0e100,
                -1.0e100,
                1.0e-50,
                -1.0e-50,
                1.0e-9,
                1.0e-6,
                65_519.0,
                65_520.0,
                -0.0,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ] {
                let dst = destination(&[2, 3], dtype);
                for mask in [
                    bool_mask(&[2], &[false, true]),
                    bool_mask(&[2, 3], &[false]),
                    bool_mask(&[], &[false]),
                    bool_mask(&[2, 0], &[false]),
                ] {
                    batch.push_policy(&dst, &mask, value, mode);
                }
                batch.push_policy(
                    &destination(&[0, 3], dtype),
                    &bool_mask(&[], &[true]),
                    value,
                    mode,
                );
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn invalid_boolean_mask_wins_before_scalar_packing_without_mutation() {
    let mut batch = OracleBatch::default();
    let mask_shapes: &[&[usize]] = &[&[3], &[1], &[6], &[2, 2], &[0, 4], &[2, 0, 2]];
    for dtype in NUMERIC {
        for &mask_shape in mask_shapes {
            for pattern in [&[false][..], &[true][..]] {
                let dst = destination(&[2, 3], dtype);
                let mask = bool_mask(mask_shape, pattern);
                for value in [f64::NAN, f64::INFINITY, 1.0e100] {
                    batch.push_policy(&dst, &mask, value, FloatErrorMode::Raise);
                }
            }
        }
        batch.push_policy(
            &destination(&[], dtype),
            &bool_mask(&[0], &[false]),
            f64::NAN,
            FloatErrorMode::Raise,
        );
    }
    batch.assert_matches_numpy();
}

#[test]
fn boolean_assignment_preserves_unselected_wide_integer_sidecars() {
    let mut batch = OracleBatch::default();
    let destinations = [
        UFuncArray::from_storage(
            vec![2, 3],
            ArrayStorage::I64(vec![i64::MIN, i64::MAX, (1_i64 << 53) + 1, -7, 9, 11]),
        )
        .unwrap(),
        UFuncArray::from_storage(
            vec![2, 3],
            ArrayStorage::U64(vec![
                u64::MAX,
                (1_u64 << 53) + 1,
                (1_u64 << 63) + 7,
                7,
                9,
                11,
            ]),
        )
        .unwrap(),
    ];
    for dst in destinations {
        for mask in [
            bool_mask(&[2], &[false, true]),
            bool_mask(&[2, 3], &[false, true]),
        ] {
            for value in [
                2.0_f64.powi(63).next_down(),
                2.0_f64.powi(63),
                2.0_f64.powi(64).next_down(),
            ] {
                batch.push(&dst, &mask, value);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn boolean_assignment_rejects_invalid_layout_and_nonboolean_masks() {
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [vec![], vec![1], vec![1, 1]] {
            let mut dst = UFuncArray::new(shape, vec![7.0], dtype).unwrap();
            let before = dst.values().to_vec();
            assert!(dst.boolean_set(&bool_mask(&[], &[true]), -0.0).is_err());
            assert_eq!(dst.values(), before);
        }
    }
    for dtype in NUMERIC {
        let mut dst = destination(&[2, 3], dtype);
        let before = oracle_input(&dst);
        let mask = UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::F64).unwrap();
        let error = dst.boolean_set(&mask, 2.5).unwrap_err();
        assert!(error.to_string().contains("mask must be a boolean array"));
        assert_eq!(oracle_input(&dst), before);
    }
}

#[test]
fn boolean_assignment_handles_huge_empty_logical_shapes_without_products() {
    for dtype in NUMERIC {
        let mut dst = destination(&[0, usize::MAX, usize::MAX], dtype);
        let before = oracle_input(&dst);
        for mask in [
            bool_mask(&[0], &[false]),
            bool_mask(&[], &[true]),
            bool_mask(&[], &[false]),
        ] {
            dst.boolean_set(&mask, 7.0).unwrap();
            assert_eq!(oracle_input(&dst), before);
        }
    }
}
