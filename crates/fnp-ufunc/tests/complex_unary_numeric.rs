//! Generic complex unary operations preserve logical elements and source precision.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{
    FloatErrorKind, FloatErrorMode, UFuncArray, UFuncError, UnaryOp, errstate, logical_not,
    seterrcall, take_float_error_events,
};
use serde_json::{Value, json};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

const OPS: [UnaryOp; 19] = [
    UnaryOp::Negative,
    UnaryOp::Positive,
    UnaryOp::Square,
    UnaryOp::Sign,
    UnaryOp::LogicalNot,
    UnaryOp::Sqrt,
    UnaryOp::Exp,
    UnaryOp::Log,
    UnaryOp::Reciprocal,
    UnaryOp::Log2,
    UnaryOp::Log10,
    UnaryOp::Round,
    UnaryOp::Rint,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Tan,
    UnaryOp::Sinh,
    UnaryOp::Cosh,
    UnaryOp::Tanh,
];

const TRIG_OPS: [UnaryOp; 6] = [
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Tan,
    UnaryOp::Sinh,
    UnaryOp::Cosh,
    UnaryOp::Tanh,
];

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
        DType::Bool => "bool",
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
        _ => panic!("complex unary fixture requires a complex or Boolean dtype"),
    }
}

#[derive(Default)]
struct OracleBatch {
    cases: Vec<Value>,
    actual: Vec<(UFuncArray, Vec<String>)>,
}

impl OracleBatch {
    fn push(&mut self, input: &UFuncArray, op: UnaryOp) {
        let mut shape = input.shape().to_vec();
        assert_eq!(shape.pop(), Some(2));
        self.cases.push(json!({
            "shape": shape,
            "dtype": dtype_name(input.dtype()),
            "values": input.values().as_chunks::<2>().0.iter()
                .map(|parts| [parts[0].to_string(), parts[1].to_string()])
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

def kind(message):
    return {'divide': 'divide', 'overflow': 'over', 'underflow': 'under',
            'invalid': 'invalid'}[message.split()[0]]

rows = []
for case in json.load(sys.stdin):
    # Source narrowing is array construction, outside the ufunc's seterr scope.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        source = np.empty(len(case['values']), dtype=case['dtype'])
        source.real = [float(value[0]) for value in case['values']]
        source.imag = [float(value[1]) for value in case['values']]
        source = source.reshape(case['shape'])
    with warnings.catch_warnings(record=True) as caught, np.errstate(all='warn'):
        warnings.simplefilter('always')
        result = np.asarray(getattr(np, case['op'])(source))
    if result.dtype.kind == 'c':
        values = [repr(float(part)) for value in result.ravel()
                  for part in (value.real, value.imag)]
    else:
        values = [str(int(value)) for value in result.ravel()]
    rows.append(dict(shape=list(result.shape), dtype=str(result.dtype), values=values,
                     reports=[kind(str(item.message)) for item in caught]))
print(json.dumps(rows))
"#;
        let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(python)
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("launch NumPy complex unary oracle");
        let mut stdin = child.stdin.take().unwrap();
        serde_json::to_writer(&mut stdin, &self.cases).unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "complex unary oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.actual.len());
        for ((case, (actual, reports)), expected) in
            self.cases.iter().zip(self.actual).zip(expected)
        {
            let mut shape = actual.shape().to_vec();
            if actual.dtype().is_complex() {
                assert_eq!(shape.pop(), Some(2), "{case}: physical component axis");
            }
            assert_eq!(
                json!(shape),
                expected["shape"],
                "{case}: logical output shape"
            );
            assert_eq!(
                dtype_name(actual.dtype()),
                expected["dtype"],
                "{case}: output dtype"
            );
            let expected_values = expected["values"].as_array().unwrap();
            assert_eq!(
                actual.values().len(),
                expected_values.len(),
                "{case}: output size"
            );
            for (index, (&actual_value, expected_value)) in
                actual.values().iter().zip(expected_values).enumerate()
            {
                let expected_value: f64 = expected_value.as_str().unwrap().parse().unwrap();
                if expected_value.is_nan() {
                    assert!(
                        actual_value.is_nan(),
                        "{case}: component {index}: {actual_value} != NaN"
                    );
                } else if expected_value.is_infinite() || expected_value == 0.0 {
                    assert_eq!(
                        actual_value.to_bits(),
                        expected_value.to_bits(),
                        "{case}: component {index}: special value {actual_value} != {expected_value}"
                    );
                } else if matches!(
                    case["op"].as_str().unwrap(),
                    "negative" | "positive" | "square" | "logical_not"
                ) {
                    assert_eq!(
                        actual_value.to_bits(),
                        expected_value.to_bits(),
                        "{case}: component {index}: {actual_value} != {expected_value}"
                    );
                } else {
                    // Native libm implementations can differ by a few ulps.
                    // Keep dtype, signs, classifications, and error policy exact;
                    // allow a small absolute tolerance for subnormal components.
                    let (epsilon, smallest) = if actual.dtype() == DType::Complex64 {
                        (f64::from(f32::EPSILON), f64::from(f32::from_bits(1)))
                    } else {
                        (f64::EPSILON, f64::from_bits(1))
                    };
                    let tolerance = 8.0 * epsilon * expected_value.abs() + 4.0 * smallest;
                    assert!(
                        actual_value.is_finite()
                            && (actual_value - expected_value).abs() <= tolerance,
                        "{case}: component {index}: {actual_value} != {expected_value}, tolerance {tolerance}"
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
fn complex_unary_arithmetic_and_logical_not_use_complex_elements() {
    let _errors = errstate(Some(FloatErrorMode::Ignore), None, None, None, None);
    for dtype in [DType::Complex64, DType::Complex128] {
        let input = complex(
            &[2, 2],
            &[(3.0, 4.0), (0.0, -0.0), (-2.0, 0.0), (0.0, 7.0)],
            dtype,
        );
        let square = input.try_elementwise_unary(UnaryOp::Square).unwrap();
        assert_eq!(square.dtype(), dtype);
        assert_eq!(square.shape(), &[2, 2, 2]);
        assert_eq!(
            square.values(),
            &[-7.0, 24.0, 0.0, -0.0, 4.0, -0.0, -49.0, 0.0]
        );
        assert_eq!(square.values()[3].to_bits(), (-0.0_f64).to_bits());
        let negative = input.try_elementwise_unary(UnaryOp::Negative).unwrap();
        assert_eq!(negative.dtype(), dtype);
        assert_eq!(
            negative.values(),
            &[-3.0, -4.0, -0.0, 0.0, 2.0, -0.0, -0.0, -7.0]
        );
        let truth = input.try_elementwise_unary(UnaryOp::LogicalNot).unwrap();
        assert_eq!(truth.dtype(), DType::Bool);
        assert_eq!(truth.shape(), &[2, 2]);
        assert_eq!(truth.values(), &[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(logical_not(&input).unwrap(), truth);
        assert_eq!(input.elementwise_unary(UnaryOp::LogicalNot), truth);
        let scalar = complex(&[], &[(0.0, -0.0)], dtype);
        assert!(logical_not(&scalar).unwrap().shape().is_empty());
        for shape in [&[0, 3][..], &[2, 0, 1][..]] {
            let empty = complex(shape, &[], dtype);
            for op in OPS {
                let result = empty.try_elementwise_unary(op).unwrap();
                let mut expected_shape = shape.to_vec();
                if op != UnaryOp::LogicalNot {
                    expected_shape.push(2);
                }
                assert_eq!(result.shape(), expected_shape);
                assert!(result.values().is_empty());
            }
        }
    }
}

#[test]
fn complex_unary_rounds_source_components_before_computation() {
    let _errors = errstate(Some(FloatErrorMode::Ignore), None, None, None, None);
    let input = complex(
        &[3],
        &[
            (1.0 + 2.0_f64.powi(-24), 1.0e-50),
            (1.0e-50, -1.0e-50),
            (f64::MAX, 0.0),
        ],
        DType::Complex64,
    );
    let positive = input.try_elementwise_unary(UnaryOp::Positive).unwrap();
    assert_eq!(
        positive.values(),
        &[1.0, 0.0, 0.0, -0.0, f64::INFINITY, 0.0]
    );
    assert_eq!(positive.values()[3].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(logical_not(&input).unwrap().values(), &[0.0, 1.0, 0.0]);
    let sign = input.try_elementwise_unary(UnaryOp::Sign).unwrap();
    assert_eq!(sign.values(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let mut batch = OracleBatch::default();
    for op in OPS {
        batch.push(&input, op);
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_unary_matches_numpy_across_shapes_and_finite_values() {
    let pattern = [
        (3.0, 4.0),
        (-3.0, 4.0),
        (3.0, -4.0),
        (-3.0, -4.0),
        (0.0, 2.0),
        (-2.0, 0.0),
        (1.0, 1.0e-8),
        (1.0e-8, -1.0),
        (0.75, -0.25),
        (8.0, 0.125),
        (-7.0, -1.5),
        (0.125, 6.0),
    ];
    let shapes: &[&[usize]] = &[&[], &[12], &[3, 4], &[2, 1, 2, 3], &[0, 3], &[2, 0, 1]];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &shape in shapes {
            let count: usize = shape.iter().product();
            let parts = (0..count)
                .map(|index| pattern[index % pattern.len()])
                .collect::<Vec<_>>();
            let input = complex(shape, &parts, dtype);
            for op in OPS {
                batch.push(&input, op);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_unary_matches_numpy_signed_zero_nan_and_infinity_branches() {
    let components = [
        f64::NEG_INFINITY,
        -4.0,
        -0.0,
        0.0,
        4.0,
        f64::INFINITY,
        f64::NAN,
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for re in components {
            for im in components {
                let input = complex(&[], &[(re, im)], dtype);
                for op in OPS {
                    batch.push(&input, op);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_log_preserves_small_results_near_the_unit_circle() {
    let cases = [
        (0.6, 0.8),
        (0.5_f64.sqrt(), 0.5_f64.sqrt()),
        (0.5, 0.75_f64.sqrt()),
        (1.0 - 2.0_f64.powi(-53), 2.0_f64.powi(-26)),
        (1.0, 1.0e-8),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for (re, im) in cases {
            for (re, im) in [(re, im), (-re, im), (im, -re), (-im, -re)] {
                batch.push(&complex(&[], &[(re, im)], dtype), UnaryOp::Log);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_sqrt_rounds_tiny_minor_components_in_source_precision() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let tiny = if dtype == DType::Complex64 {
            f64::from(f32::from_bits(1))
        } else {
            f64::from_bits(1)
        };
        for real in [0.0625, 0.25, 0.5, 0.9, 1.0, 2.0, 4.0, 16.0] {
            for multiple in [1.0, 2.0, 3.0, 4.0, 5.0, 7.0] {
                for (re, im) in [
                    (real, multiple * tiny),
                    (-real, multiple * tiny),
                    (real, -multiple * tiny),
                    (-real, -multiple * tiny),
                ] {
                    batch.push(&complex(&[], &[(re, im)], dtype), UnaryOp::Sqrt);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_unary_retains_finite_roots_logs_and_scaled_exponential_components() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let (largest, normal, tiny) = if dtype == DType::Complex64 {
            (
                f64::from(f32::MAX),
                f64::from(f32::MIN_POSITIVE),
                f64::from(f32::from_bits(1)),
            )
        } else {
            (f64::MAX, f64::MIN_POSITIVE, f64::from_bits(1))
        };
        let cases = [
            (largest, largest),
            (-largest, largest),
            (largest, tiny),
            (tiny, largest),
            (tiny, tiny),
            (3.0 * tiny, 4.0 * tiny),
            (normal / 2.0, normal / 2.0),
            (normal, tiny),
            (tiny, 0.0),
            (0.0, tiny),
            (normal.sqrt(), normal.sqrt()),
            (1.0, tiny),
            (-1.0, tiny),
            (tiny, -1.0),
            (0.25, tiny),
            (-0.9, tiny),
            (2.0, tiny),
            (0.5, tiny),
            (710.0, 1.0),
            (1000.0, tiny),
            (-1000.0, tiny),
        ];
        for pair in cases {
            let input = complex(&[], &[pair], dtype);
            for op in OPS {
                batch.push(&input, op);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_unary_dispatches_aggregate_float_errors_and_preserves_infallible_values() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let input = complex(
            &[4],
            &[
                (1000.0, 1.0),
                (-1000.0, 1.0),
                (0.0, f64::INFINITY),
                (1000.0, 1.0),
            ],
            dtype,
        );
        let expected = {
            let _errors = errstate(Some(FloatErrorMode::Ignore), None, None, None, None);
            take_float_error_events();
            let result = input.try_elementwise_unary(UnaryOp::Exp).unwrap();
            assert!(take_float_error_events().is_empty());
            result
        };
        {
            let _errors = errstate(Some(FloatErrorMode::Warn), None, None, None, None);
            let result = input.try_elementwise_unary(UnaryOp::Exp).unwrap();
            assert_eq!(result.dtype(), dtype);
            let reports = take_float_error_events();
            assert_eq!(
                reports.iter().map(|event| event.kind).collect::<Vec<_>>(),
                [
                    FloatErrorKind::Over,
                    FloatErrorKind::Under,
                    FloatErrorKind::Invalid
                ]
            );
            assert!(
                reports
                    .iter()
                    .all(|event| event.op == "exp" && event.mode == FloatErrorMode::Warn)
            );
        }
        {
            let _errors = errstate(Some(FloatErrorMode::Raise), None, None, None, None);
            assert!(matches!(
                input.try_elementwise_unary(UnaryOp::Exp),
                Err(UFuncError::FloatingPoint {
                    kind: FloatErrorKind::Over,
                    ..
                })
            ));
            let result = input.elementwise_unary(UnaryOp::Exp);
            assert_eq!(result.dtype(), dtype);
            assert_eq!(result.shape(), input.shape());
            for (&actual, &expected) in result.values().iter().zip(expected.values()) {
                assert!(
                    actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan()
                );
            }
            assert!(
                take_float_error_events()
                    .iter()
                    .any(|event| event.kind == FloatErrorKind::Over
                        && event.mode == FloatErrorMode::Raise)
            );
        }
        let calls = Arc::new(Mutex::new(Vec::new()));
        let callback_calls = Arc::clone(&calls);
        let previous = seterrcall(Some(Arc::new(move |event| {
            assert_eq!(event.op, "exp");
            callback_calls.lock().unwrap().push(event.kind);
        })));
        {
            let _errors = errstate(Some(FloatErrorMode::Call), None, None, None, None);
            input.try_elementwise_unary(UnaryOp::Exp).unwrap();
        }
        let _ = seterrcall(previous);
        assert_eq!(
            *calls.lock().unwrap(),
            [
                FloatErrorKind::Over,
                FloatErrorKind::Under,
                FloatErrorKind::Invalid
            ]
        );
        take_float_error_events();
        let zero = complex(&[], &[(0.0, -0.0)], dtype);
        let _errors = errstate(Some(FloatErrorMode::Raise), None, None, None, None);
        assert!(matches!(
            zero.try_elementwise_unary(UnaryOp::Log),
            Err(UFuncError::FloatingPoint {
                kind: FloatErrorKind::Divide,
                ..
            })
        ));
    }
}

#[test]
fn complex_unary_rejects_malformed_layouts_and_retains_real_logical_not() {
    for dtype in [DType::Complex64, DType::Complex128] {
        for malformed in [
            UFuncArray::new(vec![4], vec![1.0; 4], dtype).unwrap(),
            UFuncArray::new(vec![0, 4], vec![], dtype).unwrap(),
        ] {
            for op in OPS {
                assert!(malformed.try_elementwise_unary(op).is_err());
            }
            assert!(logical_not(&malformed).is_err());
        }
    }
    let input =
        UFuncArray::from_storage(vec![3], ArrayStorage::I64(vec![0, i64::MAX, i64::MIN])).unwrap();
    let result = logical_not(&input).unwrap();
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.dtype(), DType::Bool);
    assert_eq!(result.values(), &[1.0, 0.0, 0.0]);
}

#[test]
fn complex_reciprocal_distinguishes_exact_subnormal_results_from_underflow() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let (power, largest, smallest) = if dtype == DType::Complex64 {
            (
                f64::from(2.0_f32.powi(127)),
                f64::from(f32::MAX),
                f64::from(f32::from_bits(1)),
            )
        } else {
            (2.0_f64.powi(1023), f64::MAX, f64::from_bits(1))
        };
        for (real, imag) in [
            (power, 0.0),
            (-power, -0.0),
            (0.0, power),
            (-0.0, -power),
            (largest, 0.0),
            (0.0, largest),
            (largest, largest),
            (largest, smallest),
            (smallest, largest),
            (smallest, smallest),
        ] {
            batch.push(&complex(&[], &[(real, imag)], dtype), UnaryOp::Reciprocal);
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_base_logarithms_retain_status_from_the_natural_logarithm() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let minimum = if dtype == DType::Complex64 {
            f64::from(f32::MIN_POSITIVE)
        } else {
            f64::MIN_POSITIVE
        };
        for pair in [
            (1.0, 0.8 * minimum),
            (1.0, -0.8 * minimum),
            (-1.0, 0.8 * minimum),
            (0.0, -0.0),
        ] {
            for op in [UnaryOp::Log2, UnaryOp::Log10] {
                batch.push(&complex(&[], &[pair], dtype), op);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_reciprocal_reports_invalid_zero_and_retains_infallible_pairs() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let input = complex(&[2], &[(0.0, -0.0), (3.0, 4.0)], dtype);
        let _errors = errstate(Some(FloatErrorMode::Raise), None, None, None, None);
        assert!(matches!(
            input.try_elementwise_unary(UnaryOp::Reciprocal),
            Err(UFuncError::FloatingPoint {
                kind: FloatErrorKind::Invalid,
                ..
            })
        ));
        let output = input.elementwise_unary(UnaryOp::Reciprocal);
        assert_eq!(output.dtype(), dtype);
        assert_eq!(output.shape(), input.shape());
        assert!(output.values()[0].is_nan() && output.values()[1].is_nan());
        let expected = if dtype == DType::Complex64 {
            [f64::from(0.12_f32), f64::from(-0.16_f32)]
        } else {
            [0.12, -0.16]
        };
        assert_eq!(&output.values()[2..], &expected);
        let events = take_float_error_events();
        assert!(events.iter().any(|event| event.op == "reciprocal"
            && event.kind == FloatErrorKind::Invalid
            && event.mode == FloatErrorMode::Raise));
    }
}

#[test]
fn complex_trigonometric_functions_match_numpy_near_poles_and_across_quadrants() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let epsilon = if dtype == DType::Complex64 {
            f64::from(f32::EPSILON)
        } else {
            f64::EPSILON
        };
        let tiny = if dtype == DType::Complex64 {
            f64::from(f32::from_bits(1))
        } else {
            f64::from_bits(1)
        };
        for n in -3..=3 {
            let pole = (f64::from(n) + 0.5) * std::f64::consts::PI;
            for delta in [-8.0 * epsilon, -epsilon, 0.0, epsilon, 8.0 * epsilon] {
                for minor in [-0.25, -epsilon, -tiny, -0.0, 0.0, tiny, epsilon, 0.25] {
                    let pairs = [(pole + delta, minor), (minor, pole + delta)];
                    for op in TRIG_OPS {
                        batch.push(&complex(&[2], &pairs, dtype), op);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_trigonometric_functions_preserve_tiny_and_scaled_components() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let (tiny, normal, large) = if dtype == DType::Complex64 {
            (
                f64::from(f32::from_bits(1)),
                f64::from(f32::MIN_POSITIVE),
                90.0,
            )
        } else {
            (f64::from_bits(1), f64::MIN_POSITIVE, 710.0)
        };
        for magnitude in [tiny, 3.0 * tiny, 0.5 * normal, normal, normal.sqrt()] {
            for major in [
                0.0,
                1.0,
                4.0,
                22.0,
                40.0,
                88.0,
                89.0,
                large,
                large + 20.0,
                1000.0,
            ] {
                for pair in [
                    (major, magnitude),
                    (major, -magnitude),
                    (-major, magnitude),
                    (magnitude, major),
                    (-magnitude, -major),
                ] {
                    for op in TRIG_OPS {
                        batch.push(&complex(&[], &[pair], dtype), op);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_trigonometric_errors_aggregate_and_infallible_results_keep_pairs() {
    for dtype in [DType::Complex64, DType::Complex128] {
        for op in TRIG_OPS {
            let input = complex(
                &[4],
                &[
                    (1000.0, 1000.0),
                    (1000.0, 1000.0),
                    (f64::INFINITY, f64::INFINITY),
                    (1.0, 0.5),
                ],
                dtype,
            );
            let expected = {
                let _errors = errstate(Some(FloatErrorMode::Ignore), None, None, None, None);
                input.try_elementwise_unary(op).unwrap()
            };
            let reports = {
                let _errors = errstate(Some(FloatErrorMode::Warn), None, None, None, None);
                take_float_error_events();
                input.try_elementwise_unary(op).unwrap();
                take_float_error_events()
            };
            assert!(
                !reports.is_empty(),
                "{op:?}: exercise a real floating error"
            );
            for kind in [
                FloatErrorKind::Divide,
                FloatErrorKind::Over,
                FloatErrorKind::Under,
                FloatErrorKind::Invalid,
            ] {
                assert!(reports.iter().filter(|event| event.kind == kind).count() <= 1);
            }
            let _errors = errstate(Some(FloatErrorMode::Raise), None, None, None, None);
            assert!(matches!(
                input.try_elementwise_unary(op),
                Err(UFuncError::FloatingPoint { .. })
            ));
            let actual = input.elementwise_unary(op);
            assert_eq!(actual.shape(), input.shape());
            assert_eq!(actual.dtype(), dtype);
            for (&actual, &expected) in actual.values().iter().zip(expected.values()) {
                assert!(
                    actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan()
                );
            }
            take_float_error_events();
        }
    }
}
