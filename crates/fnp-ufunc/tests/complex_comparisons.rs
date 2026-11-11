//! Complex comparison and logical ufuncs use logical elements and NumPy ordering.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{
    BinaryDispatchMethod, BinaryOp, FloatErrorKind, FloatErrorMode, UFuncArray, UFuncError,
    UFuncLoopRegistry, errstate, parse_fixed_signature_string, plan_binary_dispatch_with_registry,
    seterrcall, take_float_error_events,
};
use serde_json::{Value, json};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

const OPS: [BinaryOp; 9] = [
    BinaryOp::Equal,
    BinaryOp::NotEqual,
    BinaryOp::Less,
    BinaryOp::LessEqual,
    BinaryOp::Greater,
    BinaryOp::GreaterEqual,
    BinaryOp::LogicalAnd,
    BinaryOp::LogicalOr,
    BinaryOp::LogicalXor,
];

const NUMERIC_DTYPES: [DType; 14] = [
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
        _ => panic!("comparison fixture requires a numeric dtype"),
    }
}

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

fn sample(shape: &[usize], dtype: DType, offset: usize) -> UFuncArray {
    let count = if shape.contains(&0) {
        0
    } else {
        shape.iter().product()
    };
    let pattern = [
        (0.0, -0.0),
        (0.0, 2.0),
        (-3.0, 0.0),
        (1.0 + 2.0_f64.powi(-24), -1.0 - 2.0_f64.powi(-24)),
        (1.0e-50, -1.0e-50),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY),
    ];
    if dtype.is_complex() {
        let parts = (0..count)
            .map(|i| pattern[(i + offset) % pattern.len()])
            .collect::<Vec<_>>();
        return complex(shape, &parts, dtype);
    }
    if dtype.is_integer() || dtype == DType::Bool {
        return UFuncArray::from_storage(
            shape.to_vec(),
            ArrayStorage::F64((0..count).map(|i| ((i + offset) % 5) as f64).collect())
                .cast_to(dtype)
                .unwrap(),
        )
        .unwrap();
    }
    UFuncArray::new(
        shape.to_vec(),
        (0..count)
            .map(|i| pattern[(i + offset) % pattern.len()].0)
            .collect(),
        dtype,
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
                .map(|parts| { [parts[0].to_string(), parts[1].to_string()] })
                .collect::<Vec<_>>()
        )
    } else if array.dtype().is_integer() || array.dtype() == DType::Bool {
        let storage = array.to_storage().unwrap();
        json!(
            (0..storage.len())
                .map(|i| storage.get_i128(i).unwrap().to_string())
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
    cases: Vec<Value>,
    actual: Vec<(Result<UFuncArray, UFuncError>, Vec<String>)>,
}

impl OracleBatch {
    fn push(&mut self, lhs: &UFuncArray, rhs: &UFuncArray, op: BinaryOp) {
        self.cases
            .push(json!({"lhs": oracle_input(lhs), "rhs": oracle_input(rhs), "op": op.name()}));
        let _guard = errstate(Some(FloatErrorMode::Warn), None, None, None, None);
        take_float_error_events();
        let output = lhs.elementwise_binary(rhs, op);
        let events = take_float_error_events()
            .into_iter()
            .map(|event| event.message)
            .collect();
        self.actual.push((output, events));
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np

def load(item):
    dtype = np.dtype(item['dtype'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if dtype.kind == 'c':
            a = np.empty(len(item['values']), dtype=dtype)
            a.real = [float(pair[0]) for pair in item['values']]
            a.imag = [float(pair[1]) for pair in item['values']]
        else:
            convert = int if dtype.kind in 'biu' else float
            a = np.array([convert(value) for value in item['values']], dtype=dtype)
    return a.reshape(item['shape'])

rows = []
for case in json.load(sys.stdin):
    lhs, rhs = load(case['lhs']), load(case['rhs'])
    try:
        with warnings.catch_warnings(record=True) as messages, np.errstate(all='warn'):
            warnings.simplefilter('always')
            result = np.asarray(getattr(np, case['op'])(lhs, rhs))
        rows.append(dict(shape=list(result.shape), dtype=str(result.dtype),
                         values=result.ravel().tolist(), warnings=[str(m.message) for m in messages]))
    except (TypeError, ValueError) as error:
        rows.append(dict(error=type(error).__name__))
print(json.dumps(rows))
"#;
        let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(python)
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("launch live NumPy complex comparison oracle");
        let mut stdin = child.stdin.take().unwrap();
        serde_json::to_writer(&mut stdin, &self.cases).unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "comparison oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.actual.len());
        for ((case, (actual, events)), expected) in self.cases.iter().zip(self.actual).zip(expected)
        {
            if expected.get("error").is_some() {
                assert!(
                    actual.is_err(),
                    "{case}: expected {expected}, got {actual:?}"
                );
                continue;
            }
            let actual = actual.unwrap_or_else(|error| panic!("{case}: {error}"));
            assert_eq!(actual.dtype(), DType::Bool, "{case}: output dtype");
            assert_eq!(expected["dtype"], "bool", "{case}: oracle output dtype");
            assert_eq!(
                json!(actual.shape()),
                expected["shape"],
                "{case}: logical shape"
            );
            assert!(
                actual
                    .values()
                    .iter()
                    .all(|&value| value == 0.0 || value == 1.0)
            );
            assert_eq!(
                json!(
                    actual
                        .values()
                        .iter()
                        .map(|&v| v != 0.0)
                        .collect::<Vec<_>>()
                ),
                expected["values"],
                "{case}: values"
            );
            assert_eq!(
                json!(events),
                expected["warnings"],
                "{case}: floating-point events"
            );
        }
    }
}

#[test]
fn comparisons_return_one_boolean_per_complex_element() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let lhs = complex(&[3], &[(1.0, 2.0), (1.0, 4.0), (0.0, 2.0)], dtype);
        let rhs = complex(&[3], &[(1.0, 3.0), (1.0, 4.0), (0.0, 0.0)], dtype);
        for (op, expected) in [
            (BinaryOp::Equal, [0.0, 1.0, 0.0]),
            (BinaryOp::NotEqual, [1.0, 0.0, 1.0]),
            (BinaryOp::Less, [1.0, 0.0, 0.0]),
            (BinaryOp::LessEqual, [1.0, 1.0, 0.0]),
            (BinaryOp::Greater, [0.0, 0.0, 1.0]),
            (BinaryOp::GreaterEqual, [0.0, 1.0, 1.0]),
            (BinaryOp::LogicalAnd, [1.0, 1.0, 0.0]),
            (BinaryOp::LogicalOr, [1.0, 1.0, 1.0]),
            (BinaryOp::LogicalXor, [0.0, 0.0, 1.0]),
        ] {
            let result = lhs.elementwise_binary(&rhs, op).unwrap();
            assert_eq!(result.shape(), &[3], "{op:?}");
            assert_eq!(result.dtype(), DType::Bool, "{op:?}");
            assert_eq!(result.values(), &expected, "{op:?}");
        }
    }
}

#[test]
fn comparisons_match_numpy_across_numeric_dtypes_and_logical_broadcasts() {
    let mut batch = OracleBatch::default();
    let shapes: &[(&[usize], &[usize])] = &[
        (&[8], &[8]),
        (&[2, 1, 3], &[1, 4, 1]),
        (&[], &[2, 3]),
        (&[2, 1], &[]),
        (&[0, 3], &[1, 3]),
        (&[2, 0, 1], &[0, 3]),
        (&[2, 3], &[2]),
    ];
    for complex_dtype in [DType::Complex64, DType::Complex128] {
        for other_dtype in NUMERIC_DTYPES {
            for &(lhs_shape, rhs_shape) in shapes {
                let lhs = sample(lhs_shape, complex_dtype, 0);
                let rhs = sample(rhs_shape, other_dtype, 1);
                for op in OPS {
                    batch.push(&lhs, &rhs, op);
                    batch.push(&rhs, &lhs, op);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_ordering_nan_short_circuits_match_numpy_values_and_warnings() {
    let mut batch = OracleBatch::default();
    let points = [
        (0.0, 0.0),
        (-0.0, -0.0),
        (0.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (f64::INFINITY, f64::NEG_INFINITY),
        (f64::NEG_INFINITY, f64::INFINITY),
        (f64::NAN, 0.0),
        (0.0, f64::NAN),
        (1.0, f64::NAN),
        (f64::NAN, f64::NAN),
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        for lhs in points {
            for rhs in points {
                let lhs = complex(&[], &[lhs], dtype);
                let rhs = complex(&[], &[rhs], dtype);
                for op in OPS {
                    batch.push(&lhs, &rhs, op);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn comparisons_round_source_precision_before_promotion_and_keep_exact_integer_inputs() {
    let mut batch = OracleBatch::default();
    let real_inputs = [
        UFuncArray::from_storage(
            vec![5],
            ArrayStorage::I64(vec![
                0,
                9_007_199_254_740_993,
                i64::MAX,
                i64::MIN,
                -9_007_199_254_740_993,
            ]),
        )
        .unwrap(),
        UFuncArray::from_storage(
            vec![5],
            ArrayStorage::U64(vec![
                0,
                9_007_199_254_740_993,
                u64::MAX,
                1 << 63,
                (1 << 63) + 1,
            ]),
        )
        .unwrap(),
        UFuncArray::new(
            vec![5],
            vec![1.0e-50, 1.0 + 2.0_f64.powi(-24), f64::MAX, 1.0e-8, -0.0],
            DType::F32,
        )
        .unwrap(),
        UFuncArray::new(
            vec![5],
            vec![1.0e-50, 1.0 + 2.0_f64.powi(-11), 65_520.0, 1.0e-8, -0.0],
            DType::F16,
        )
        .unwrap(),
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        let input = complex(
            &[5, 1],
            &[
                (1.0e-50, -1.0e-50),
                (1.0 + 2.0_f64.powi(-24), 0.0),
                (9_007_199_254_740_992.0, 0.0),
                (i64::MAX as f64, -0.0),
                (u64::MAX as f64, 1.0),
            ],
            dtype,
        );
        for other in &real_inputs {
            for op in OPS {
                batch.push(&input, other, op);
                batch.push(other, &input, op);
            }
        }
        let wider = complex(&[5], &[(1.0, 0.0); 5], DType::Complex128);
        for op in OPS {
            batch.push(&input, &wider, op);
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_comparisons_validate_shapes_layouts_and_dtypes_before_execution() {
    let valid = complex(&[3], &[(1.0, 2.0); 3], DType::Complex128);
    for malformed in [
        UFuncArray::new(vec![3], vec![1.0; 3], DType::Complex128).unwrap(),
        UFuncArray::new(vec![], vec![1.0], DType::Complex64).unwrap(),
        UFuncArray::new(vec![0, 3], vec![], DType::Complex64).unwrap(),
    ] {
        for op in OPS {
            assert!(malformed.elementwise_binary(&valid, op).is_err());
            assert!(valid.elementwise_binary(&malformed, op).is_err());
        }
    }
    for dtype in [
        DType::DateTime64,
        DType::TimeDelta64,
        DType::Str,
        DType::Object,
        DType::Void,
    ] {
        let other = UFuncArray::new(vec![3], vec![0.0; 3], dtype).unwrap();
        for op in OPS[..6].iter().copied() {
            assert!(matches!(
                valid.elementwise_binary(&other, op),
                Err(UFuncError::TypeResolutionInvalid { .. })
            ));
            assert!(matches!(
                other.elementwise_binary(&valid, op),
                Err(UFuncError::TypeResolutionInvalid { .. })
            ));
        }
    }
    let empty = complex(&[0, usize::MAX], &[], DType::Complex128);
    let scalar = UFuncArray::scalar(1.0, DType::F64);
    for op in OPS {
        let result = empty.elementwise_binary(&scalar, op).unwrap();
        assert_eq!(result.shape(), &[0, usize::MAX]);
        assert!(result.values().is_empty());
    }
}

#[test]
fn complex_boolean_dispatch_plans_preserve_registry_and_fixed_signature_contracts() {
    let lhs = complex(&[2, 1], &[(1.0, 0.0), (2.0, 3.0)], DType::Complex64);
    let rhs = UFuncArray::new(vec![3], vec![0.0, 1.0, 2.0], DType::F32).unwrap();
    let mut registry = UFuncLoopRegistry::new();
    for op in OPS {
        let selection =
            plan_binary_dispatch_with_registry(op.name(), &lhs, &rhs, &registry, None).unwrap();
        assert_eq!(selection.method, BinaryDispatchMethod::BuiltinPromoted);
        assert_eq!(selection.plan.out_shape, vec![2, 3]);
        assert_eq!(selection.plan.out_count, 6);
        assert_eq!(selection.plan.out_dtype, DType::Bool);
    }
    let signature = parse_fixed_signature_string("Ff->?", 2, 1).unwrap();
    let selection =
        plan_binary_dispatch_with_registry("equal", &lhs, &rhs, &registry, Some(&signature))
            .unwrap();
    assert_eq!(selection.method, BinaryDispatchMethod::FixedSignature);
    assert_eq!(selection.plan.out_shape, vec![2, 3]);
    assert_eq!(selection.plan.out_dtype, DType::Bool);

    registry
        .register(
            "complex_compare",
            "equal",
            vec![DType::Complex64, DType::F32],
            vec![DType::Bool],
        )
        .unwrap();
    let selection =
        plan_binary_dispatch_with_registry("equal", &lhs, &rhs, &registry, Some(&signature))
            .unwrap();
    assert_eq!(selection.method, BinaryDispatchMethod::CustomLoop);
    assert_eq!(
        selection.custom_loop_name.as_deref(),
        Some("complex_compare")
    );
    let actual = lhs
        .elementwise_binary_with_registry(&rhs, BinaryOp::Equal, &registry)
        .unwrap();
    assert_eq!(actual.shape(), &[2, 3]);
    assert_eq!(actual.values(), &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    let mut invalid = UFuncLoopRegistry::new();
    invalid
        .register(
            "non_boolean_compare",
            "equal",
            vec![DType::Complex64, DType::F32],
            vec![DType::F64],
        )
        .unwrap();
    assert!(matches!(
        lhs.elementwise_binary_with_registry(&rhs, BinaryOp::Equal, &invalid),
        Err(UFuncError::TypeResolutionInvalid { .. })
    ));
    assert!(matches!(
        plan_binary_dispatch_with_registry("equal", &lhs, &rhs, &invalid, Some(&signature)),
        Err(UFuncError::TypeResolutionInvalid { .. })
    ));
}

#[test]
fn complex_ordering_aggregates_invalid_events_and_obeys_error_policy() {
    let lhs = complex(
        &[3],
        &[(f64::NAN, 0.0), (1.0, f64::NAN), (2.0, 0.0)],
        DType::Complex128,
    );
    let rhs = complex(
        &[3],
        &[(0.0, 0.0), (1.0, 0.0), (2.0, f64::NAN)],
        DType::Complex128,
    );
    for mode in [
        FloatErrorMode::Ignore,
        FloatErrorMode::Warn,
        FloatErrorMode::Print,
        FloatErrorMode::Log,
        FloatErrorMode::Raise,
    ] {
        let _guard = errstate(Some(mode), None, None, None, None);
        take_float_error_events();
        let result = lhs.elementwise_binary(&rhs, BinaryOp::Less);
        let events = take_float_error_events();
        if mode == FloatErrorMode::Raise {
            assert!(matches!(
                result,
                Err(UFuncError::FloatingPoint {
                    kind: FloatErrorKind::Invalid,
                    ..
                })
            ));
            assert!(events.is_empty());
        } else {
            assert_eq!(result.unwrap().values(), &[0.0; 3]);
            assert_eq!(events.len(), usize::from(mode != FloatErrorMode::Ignore));
            if let Some(event) = events.first() {
                assert_eq!(event.kind, FloatErrorKind::Invalid);
                assert_eq!(event.op, "less");
                assert_eq!(event.mode, mode);
            }
        }
    }
    let events = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&events);
    let previous = seterrcall(Some(Arc::new(move |event| {
        observed.lock().unwrap().push(event)
    })));
    {
        let _guard = errstate(Some(FloatErrorMode::Call), None, None, None, None);
        take_float_error_events();
        let result = lhs.elementwise_binary(&rhs, BinaryOp::GreaterEqual);
        let _ = seterrcall(previous);
        assert_eq!(result.unwrap().values(), &[0.0; 3]);
        assert!(take_float_error_events().is_empty());
    }
    let events = events.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, FloatErrorKind::Invalid);
    assert_eq!(events[0].op, "greater_equal");
}
