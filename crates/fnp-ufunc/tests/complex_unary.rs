//! Logical complex unary operations preserve pairs, source precision and output shapes.

use fnp_dtype::DType;
use fnp_ufunc::{
    FloatErrorKind, FloatErrorMode, UFuncArray, UFuncError, UnaryOp, errstate, logical_not,
    seterrcall, take_float_error_events,
};
use serde_json::{Value, json};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

fn complex(shape: &[usize], parts: &[(f64, f64)], dtype: DType) -> UFuncArray {
    let mut physical_shape = shape.to_vec();
    physical_shape.push(2);
    UFuncArray::new(
        physical_shape,
        parts.iter().flat_map(|&(re, im)| [re, im]).collect(),
        dtype,
    )
    .unwrap()
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::Bool => "bool",
        _ => panic!("unary fixture dtype"),
    }
}

#[test]
fn unary_signs_preserve_complete_complex_elements() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let input = complex(
            &[2, 2],
            &[(1.0, 2.0), (-3.0, 4.0), (0.0, -2.0), (5.0, 0.0)],
            dtype,
        );
        for (op, expected) in [
            (
                UnaryOp::Positive,
                vec![1.0, 2.0, -3.0, 4.0, 0.0, -2.0, 5.0, 0.0],
            ),
            (
                UnaryOp::Negative,
                vec![-1.0, -2.0, 3.0, -4.0, -0.0, 2.0, -5.0, -0.0],
            ),
        ] {
            let output = input.try_elementwise_unary(op).unwrap();
            assert_eq!(output.dtype(), dtype);
            assert_eq!(output.shape(), &[2, 2, 2]);
            assert_eq!(output.values(), expected);
            let infallible = input.elementwise_unary(op);
            assert_eq!(infallible.dtype(), dtype);
            assert_eq!(infallible.shape(), output.shape());
            assert_eq!(infallible.values(), output.values());
        }
    }
}

#[test]
fn logical_not_and_predicates_observe_complex64_source_precision() {
    let input = complex(
        &[4],
        &[(0.0, -0.0), (0.0, 1.0e-50), (1.0e40, 0.0), (f64::NAN, 1.0)],
        DType::Complex64,
    );
    for (op, expected) in [
        (UnaryOp::LogicalNot, vec![1.0, 1.0, 0.0, 0.0]),
        (UnaryOp::Isnan, vec![0.0, 0.0, 0.0, 1.0]),
        (UnaryOp::Isinf, vec![0.0, 0.0, 1.0, 0.0]),
        (UnaryOp::Isfinite, vec![1.0, 1.0, 0.0, 0.0]),
    ] {
        let output = input.try_elementwise_unary(op).unwrap();
        assert_eq!(output.shape(), &[4]);
        assert_eq!(output.dtype(), DType::Bool);
        assert_eq!(output.values(), expected);
    }
    let direct = logical_not(&input).unwrap();
    assert_eq!(direct.shape(), &[4]);
    assert_eq!(direct.values(), &[1.0, 1.0, 0.0, 0.0]);

    let rounded = complex(&[], &[(1.0 + 2.0_f64.powi(-24), 1.0e-50)], DType::Complex64);
    let positive = rounded.try_elementwise_unary(UnaryOp::Positive).unwrap();
    assert_eq!(positive.shape(), &[2]);
    assert_eq!(positive.dtype(), DType::Complex64);
    assert_eq!(positive.values(), &[1.0, 0.0]);
    let negative = rounded.try_elementwise_unary(UnaryOp::Negative).unwrap();
    assert_eq!(negative.values()[0], -1.0);
    assert_eq!(negative.values()[1].to_bits(), (-0.0f64).to_bits());
}

#[test]
fn complex_unary_scalar_empty_and_signed_zero_shapes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let scalar = complex(&[], &[(3.0, -4.0)], dtype);
        let magnitude = scalar.try_elementwise_unary(UnaryOp::Abs).unwrap();
        assert!(magnitude.shape().is_empty());
        assert_eq!(magnitude.values(), &[5.0]);
        assert_eq!(
            magnitude.dtype(),
            if dtype == DType::Complex64 {
                DType::F32
            } else {
                DType::F64
            }
        );
        assert!(logical_not(&scalar).unwrap().shape().is_empty());

        let zeros = complex(&[2], &[(0.0, -0.0), (-0.0, 0.0)], dtype);
        for op in [UnaryOp::Positive, UnaryOp::Negative] {
            let output = zeros.try_elementwise_unary(op).unwrap();
            for (&before, &after) in zeros.values().iter().zip(output.values()) {
                let expected = if op == UnaryOp::Negative {
                    -before
                } else {
                    before
                };
                assert_eq!(after.to_bits(), expected.to_bits());
            }
        }

        let empty = complex(&[2, 0, 3], &[], dtype);
        for op in [
            UnaryOp::Positive,
            UnaryOp::Negative,
            UnaryOp::Abs,
            UnaryOp::LogicalNot,
        ] {
            let output = empty.try_elementwise_unary(op).unwrap();
            assert!(output.values().is_empty());
            assert_eq!(
                output.shape(),
                if matches!(op, UnaryOp::Abs | UnaryOp::LogicalNot) {
                    &[2, 0, 3][..]
                } else {
                    &[2, 0, 3, 2][..]
                }
            );
        }
    }
}

#[test]
fn complex_unary_rejects_malformed_component_layout() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let malformed = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], dtype).unwrap();
        for op in [
            UnaryOp::Positive,
            UnaryOp::Negative,
            UnaryOp::Abs,
            UnaryOp::LogicalNot,
            UnaryOp::Isnan,
            UnaryOp::Isinf,
            UnaryOp::Isfinite,
        ] {
            assert!(malformed.try_elementwise_unary(op).is_err(), "{op:?}");
        }
        assert!(logical_not(&malformed).is_err());
    }
}

#[test]
fn complex_unary_values_shapes_and_precision_match_numpy() {
    let mut cases = Vec::new();
    let mut actual = Vec::new();
    let parts = [
        (3.0, 4.0),
        (-2.5, 1.5),
        (0.0, -0.0),
        (-0.0, 0.0),
        (1.0 + 2.0_f64.powi(-24), 1.0e-50),
        (f64::NAN, 2.0),
        (f64::INFINITY, 1.0),
        (1.0, f64::NEG_INFINITY),
    ];
    let ops = [
        UnaryOp::Positive,
        UnaryOp::Negative,
        UnaryOp::Abs,
        UnaryOp::LogicalNot,
        UnaryOp::Isnan,
        UnaryOp::Isinf,
        UnaryOp::Isfinite,
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [vec![8], vec![2, 2, 2], vec![], vec![2, 0, 3]] {
            let count: usize = shape.iter().product();
            let input = complex(&shape, &parts[..count], dtype);
            for op in ops {
                cases.push(json!({
                    "dtype": dtype_name(dtype),
                    "shape": shape,
                    "parts": parts[..count].iter().map(|&(re, im)| [re.to_string(), im.to_string()]).collect::<Vec<_>>(),
                    "op": op.name(),
                }));
                actual.push(input.try_elementwise_unary(op).unwrap());
            }
        }
    }
    let script = r#"
import json, sys, warnings
import numpy as np

rows = []
for case in json.load(sys.stdin):
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        a = np.empty(len(case['parts']), dtype=case['dtype'])
        a.real = [float(pair[0]) for pair in case['parts']]
        a.imag = [float(pair[1]) for pair in case['parts']]
        a = a.reshape(case['shape'])
        result = np.asarray(getattr(np, case['op'])(a))
    flat = result.ravel()
    if result.dtype.kind == 'c':
        values = [str(v) for pair in zip(flat.real, flat.imag) for v in pair]
    else:
        values = [str(float(v)) for v in flat]
    rows.append(dict(shape=list(result.shape), dtype=str(result.dtype), values=values))
print(json.dumps(rows))
"#;
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let mut child = Command::new(python)
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch live NumPy unary oracle");
    let mut stdin = child.stdin.take().unwrap();
    serde_json::to_writer(&mut stdin, &cases).unwrap();
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "NumPy unary oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(expected.len(), actual.len());
    for ((case, actual), expected) in cases.iter().zip(actual).zip(expected) {
        let mut shape = actual.shape().to_vec();
        if actual.dtype().is_complex() {
            assert_eq!(shape.pop(), Some(2));
        }
        assert_eq!(json!(shape), expected["shape"], "{case}: logical shape");
        assert_eq!(
            dtype_name(actual.dtype()),
            expected["dtype"].as_str().unwrap(),
            "{case}: dtype"
        );
        let values = expected["values"].as_array().unwrap();
        assert_eq!(actual.values().len(), values.len(), "{case}: element count");
        for (&got, expected) in actual.values().iter().zip(values) {
            let want: f64 = expected.as_str().unwrap().parse().unwrap();
            if want.is_nan() {
                assert!(got.is_nan(), "{case}: got {got}, expected NaN");
            } else if want.is_infinite() || want == 0.0 {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{case}: exceptional/signed-zero result"
                );
            } else {
                let tolerance = if matches!(actual.dtype(), DType::F32 | DType::Complex64) {
                    4.0 * f64::from(f32::EPSILON)
                } else {
                    4.0 * f64::EPSILON
                };
                assert!(
                    (got - want).abs() <= tolerance * want.abs().max(1.0),
                    "{case}: got {got}, expected {want}"
                );
            }
        }
    }
}

#[derive(Default)]
struct ExponentialOracle {
    cases: Vec<Value>,
    actual: Vec<(UFuncArray, Vec<String>)>,
}

impl ExponentialOracle {
    fn push(&mut self, input: &UFuncArray, op: UnaryOp) {
        let mut shape = input.shape().to_vec();
        assert_eq!(shape.pop(), Some(2));
        self.cases.push(json!({
            "shape": shape,
            "dtype": dtype_name(input.dtype()),
            "parts": input.values().as_chunks::<2>().0.iter()
                .map(|pair| [pair[0].to_string(), pair[1].to_string()])
                .collect::<Vec<_>>(),
            "op": op.name(),
        }));
        let _errors = errstate(Some(FloatErrorMode::Warn), None, None, None, None);
        take_float_error_events();
        let output = input.try_elementwise_unary(op).unwrap();
        let reports = take_float_error_events()
            .into_iter()
            .map(|event| {
                assert_eq!(event.op, op.name());
                assert_eq!(event.mode, FloatErrorMode::Warn);
                event.kind.as_str().to_string()
            })
            .collect();
        self.actual.push((output, reports));
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np

kinds = {'divide': 'divide', 'overflow': 'over', 'underflow': 'under', 'invalid': 'invalid'}
rows = []
for case in json.load(sys.stdin):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.empty(len(case['parts']), dtype=case['dtype'])
        a.real = [float(pair[0]) for pair in case['parts']]
        a.imag = [float(pair[1]) for pair in case['parts']]
        a = a.reshape(case['shape'])
    with warnings.catch_warnings(record=True) as caught, np.errstate(all='warn'):
        warnings.simplefilter('always')
        output = np.asarray(getattr(np, case['op'])(a))
    values = [repr(float(part)) for value in output.ravel()
              for part in (value.real, value.imag)]
    rows.append(dict(shape=list(output.shape), dtype=str(output.dtype), values=values,
                     reports=[kinds[str(item.message).split()[0]] for item in caught]))
print(json.dumps(rows))
"#;
        let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(python)
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("launch NumPy complex exponential oracle");
        let mut stdin = child.stdin.take().unwrap();
        serde_json::to_writer(&mut stdin, &self.cases).unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy complex exponential oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.actual.len());
        for ((case, (actual, reports)), expected) in
            self.cases.iter().zip(self.actual).zip(expected)
        {
            let mut shape = actual.shape().to_vec();
            assert_eq!(shape.pop(), Some(2), "{case}: component dimension");
            assert_eq!(json!(shape), expected["shape"], "{case}: logical shape");
            assert_eq!(
                dtype_name(actual.dtype()),
                expected["dtype"],
                "{case}: dtype"
            );
            let values = expected["values"].as_array().unwrap();
            assert_eq!(values.len(), actual.values().len(), "{case}: element count");
            let (epsilon, smallest) = if actual.dtype() == DType::Complex64 {
                (f64::from(f32::EPSILON), f64::from(f32::from_bits(1)))
            } else {
                (f64::EPSILON, f64::from_bits(1))
            };
            for (component, (&actual, expected)) in actual.values().iter().zip(values).enumerate() {
                let expected: f64 = expected.as_str().unwrap().parse().unwrap();
                if expected.is_nan() {
                    assert!(
                        actual.is_nan(),
                        "{case}: component {component}: expected NaN, got {actual}"
                    );
                } else if expected.is_infinite() || expected == 0.0 {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{case}: component {component}: exceptional value or signed zero"
                    );
                } else {
                    let tolerance = 8.0 * epsilon * expected.abs() + 4.0 * smallest;
                    assert!(
                        actual.is_finite() && (actual - expected).abs() <= tolerance,
                        "{case}: component {component}: {actual} != {expected}, tolerance {tolerance}"
                    );
                }
            }
            assert_eq!(
                json!(reports),
                expected["reports"],
                "{case}: floating-point reports"
            );
        }
    }
}

#[test]
fn complex_exp2_expm1_log1p_match_numpy_values_shapes_and_source_precision() {
    let ops = [UnaryOp::Exp2, UnaryOp::Expm1, UnaryOp::Log1p];
    let parts = [
        (3.0, 4.0),
        (-3.0, 4.0),
        (3.0, -4.0),
        (-3.0, -4.0),
        (0.0, 2.0),
        (-2.0, 0.0),
        (1.0e-16, 1.0e-16),
        (-1.0e-8, 1.0e-8),
        (-1.0, 0.25),
        (-1.0, -0.25),
        (1.0 + 2.0_f64.powi(-24), 1.0e-50),
        (0.5, 10000.25),
    ];
    let mut batch = ExponentialOracle::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [
            vec![],
            vec![12],
            vec![3, 4],
            vec![2, 1, 2, 3],
            vec![0, 3],
            vec![2, 0, 1],
        ] {
            let count: usize = shape.iter().product();
            let input = complex(&shape, &parts[..count], dtype);
            for op in ops {
                batch.push(&input, op);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_exp2_expm1_log1p_match_numpy_special_values_and_error_categories() {
    let mut batch = ExponentialOracle::default();
    let components = [
        f64::NEG_INFINITY,
        -4.0,
        -1.0,
        -0.0,
        0.0,
        4.0,
        f64::INFINITY,
        f64::NAN,
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        for re in components {
            for im in components {
                let input = complex(&[], &[(re, im)], dtype);
                for op in [UnaryOp::Exp2, UnaryOp::Expm1, UnaryOp::Log1p] {
                    batch.push(&input, op);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_exp2_expm1_log1p_match_numpy_at_extreme_magnitudes() {
    let mut batch = ExponentialOracle::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let (tiny, minimum, largest) = if dtype == DType::Complex64 {
            (
                f64::from(f32::from_bits(1)),
                f64::from(f32::MIN_POSITIVE),
                f64::from(f32::MAX),
            )
        } else {
            (f64::from_bits(1), f64::MIN_POSITIVE, f64::MAX)
        };
        let parts = [
            (0.0, tiny),
            (tiny, 0.0),
            (tiny, tiny),
            (-tiny, -tiny),
            (0.0, minimum),
            (minimum, 0.0),
            (minimum, minimum),
            (minimum.sqrt(), minimum.sqrt()),
            (1.0, tiny),
            (-1.0, tiny),
            (-1.0, -tiny),
            (-2.0, tiny),
            (-4.0, tiny),
            (largest, largest),
            (largest, tiny),
            (tiny, largest),
            (f64::NAN, tiny),
            (f64::INFINITY, tiny),
            (1000.0, 0.0),
            (-1000.0, 0.0),
            (2000.0, 1.0),
            (-2000.0, 1.0),
            (88.0, 1.0),
            (710.0, 1.0),
        ];
        for pair in parts {
            let input = complex(&[], &[pair], dtype);
            for op in [UnaryOp::Exp2, UnaryOp::Expm1, UnaryOp::Log1p] {
                batch.push(&input, op);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_exp2_expm1_log1p_obey_error_policy_and_infallible_pair_semantics() {
    for dtype in [DType::Complex64, DType::Complex128] {
        for (op, value, kind) in [
            (UnaryOp::Exp2, (2000.0, 1.0), FloatErrorKind::Over),
            (
                UnaryOp::Expm1,
                (f64::INFINITY, 0.0),
                FloatErrorKind::Invalid,
            ),
            (UnaryOp::Log1p, (-1.0, -0.0), FloatErrorKind::Divide),
        ] {
            let input = complex(&[2], &[value, value], dtype);
            let expected = {
                let _errors = errstate(Some(FloatErrorMode::Ignore), None, None, None, None);
                take_float_error_events();
                let result = input.try_elementwise_unary(op).unwrap();
                assert!(take_float_error_events().is_empty());
                result
            };
            {
                let _errors = errstate(Some(FloatErrorMode::Raise), None, None, None, None);
                assert!(matches!(input.try_elementwise_unary(op),
                    Err(UFuncError::FloatingPoint { kind: actual, .. }) if actual == kind));
                let output = input.elementwise_unary(op);
                assert_eq!(output.dtype(), dtype);
                assert_eq!(output.shape(), input.shape());
                for (&actual, &expected) in output.values().iter().zip(expected.values()) {
                    assert!(
                        actual.to_bits() == expected.to_bits()
                            || actual.is_nan() && expected.is_nan()
                    );
                }
                let events = take_float_error_events();
                assert_eq!(events.len(), 1);
                assert_eq!(events[0].kind, kind);
                assert_eq!(events[0].mode, FloatErrorMode::Raise);
                assert_eq!(events[0].op, op.name());
            }
            let calls = Arc::new(Mutex::new(Vec::new()));
            let callback_calls = Arc::clone(&calls);
            let previous = seterrcall(Some(Arc::new(move |event| {
                callback_calls.lock().unwrap().push(event.kind);
            })));
            {
                let _errors = errstate(Some(FloatErrorMode::Call), None, None, None, None);
                input.try_elementwise_unary(op).unwrap();
            }
            let _ = seterrcall(previous);
            assert_eq!(*calls.lock().unwrap(), [kind]);
            take_float_error_events();
        }
        for shape in [vec![4], vec![0, 4]] {
            let count: usize = shape.iter().product();
            let malformed = UFuncArray::new(shape, vec![1.0; count], dtype).unwrap();
            for op in [UnaryOp::Exp2, UnaryOp::Expm1, UnaryOp::Log1p] {
                assert!(malformed.try_elementwise_unary(op).is_err());
            }
        }
    }
}
