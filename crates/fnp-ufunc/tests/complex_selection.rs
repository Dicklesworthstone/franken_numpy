//! Conditional selection acts on logical complex elements, never components.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{
    FloatErrorCallback, FloatErrorKind, FloatErrorMode, UFuncArray, UFuncError, errstate,
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
        parts
            .iter()
            .flat_map(|&(real, imaginary)| [real, imaginary])
            .collect(),
        dtype,
    )
    .expect("valid interleaved complex input")
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
        _ => panic!("non-numeric selection dtype"),
    }
}

fn sample(shape: &[usize], dtype: DType, offset: usize) -> UFuncArray {
    let count = if shape.contains(&0) {
        0
    } else {
        shape.iter().product()
    };
    let pattern = [
        (0.0, -0.0),
        (0.0, 2.25),
        (-3.5, 0.0),
        (1.0 + 2.0_f64.powi(-24), -1.0 - 2.0_f64.powi(-24)),
        (1.0e-50, -1.0e-50),
        (f64::NAN, 4.5),
        (0.0, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY),
    ];
    if dtype.is_complex() {
        let parts = (0..count)
            .map(|index| pattern[(index + offset) % pattern.len()])
            .collect::<Vec<_>>();
        return complex(shape, &parts, dtype);
    }
    let values: Vec<f64> = (0..count)
        .map(|index| {
            if dtype.is_integer() || dtype == DType::Bool {
                ((index + offset) % 7) as f64 - 3.0
            } else {
                pattern[(index + offset) % pattern.len()].0
            }
        })
        .collect();
    UFuncArray::from_storage(
        shape.to_vec(),
        ArrayStorage::F64(values).cast_to(dtype).unwrap(),
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

#[derive(Default)]
struct OracleBatch {
    cases: Vec<Value>,
    actual: Vec<Result<UFuncArray, UFuncError>>,
}

impl OracleBatch {
    fn push(&mut self, condition: &UFuncArray, x: &UFuncArray, y: &UFuncArray) {
        self.cases.push(json!({
            "condition": oracle_input(condition), "x": oracle_input(x), "y": oracle_input(y),
        }));
        self.actual.push(UFuncArray::where_select(condition, x, y));
    }

    fn push_select(&mut self, conditions: &[&UFuncArray], choices: &[&UFuncArray], default: f64) {
        self.cases.push(json!({
            "conditions": conditions.iter().map(|array| oracle_input(array)).collect::<Vec<_>>(),
            "choices": choices.iter().map(|array| oracle_input(array)).collect::<Vec<_>>(),
            "default": default.to_string(),
        }));
        self.actual
            .push(UFuncArray::select(conditions, choices, default));
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np

warnings.simplefilter('ignore')
def load(item):
    dtype = np.dtype(item['dtype'])
    if dtype.kind == 'c':
        array = np.empty(len(item['values']), dtype=dtype)
        array.real = [float(pair[0]) for pair in item['values']]
        array.imag = [float(pair[1]) for pair in item['values']]
    else:
        convert = int if dtype.kind in 'biu' else float
        array = np.array([convert(value) for value in item['values']], dtype=dtype)
    return array.reshape(item['shape'])
rows = []
for case in json.load(sys.stdin):
    try:
        if 'conditions' in case:
            result = np.asarray(np.select([load(item) for item in case['conditions']],
                                          [load(item) for item in case['choices']],
                                          float(case['default'])))
        else:
            result = np.asarray(np.where(load(case['condition']), load(case['x']), load(case['y'])))
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
            .expect("launch live NumPy selection oracle");
        let mut stdin = child.stdin.take().unwrap();
        serde_json::to_writer(&mut stdin, &self.cases).unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy selection oracle failed: {}",
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
            let actual = actual.unwrap_or_else(|error| panic!("{case}: {error}"));
            let mut shape = actual.shape().to_vec();
            if actual.dtype().is_complex() {
                assert_eq!(shape.pop(), Some(2), "{case}: component dimension");
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
fn where_select_keeps_complex_pairs_and_condition_axes_together() {
    let condition = UFuncArray::new(vec![2], vec![1.0, 0.0], DType::Bool).unwrap();
    for dtype in [DType::Complex64, DType::Complex128] {
        let x = complex(&[2], &[(1.0, 2.0), (3.0, 4.0)], dtype);
        let y = complex(&[2], &[(5.0, 6.0), (7.0, 8.0)], dtype);
        let result = UFuncArray::where_select(&condition, &x, &y).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.dtype(), dtype);
        assert_eq!(result.values(), &[1.0, 2.0, 7.0, 8.0]);

        let condition = complex(&[2], &[(0.0, 1.0), (0.0, 0.0)], dtype);
        let x = UFuncArray::new(vec![2], vec![10.0, 20.0], DType::F64).unwrap();
        let y = UFuncArray::new(vec![2], vec![30.0, 40.0], DType::F64).unwrap();
        let result = UFuncArray::where_select(&condition, &x, &y).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.dtype(), DType::F64);
        assert_eq!(result.values(), &[10.0, 40.0]);
    }
}

#[test]
fn where_select_complex_broadcast_and_promotion_match_numpy() {
    let shapes: &[(&[usize], &[usize], &[usize])] = &[
        (&[], &[], &[]),
        (&[8], &[8], &[8]),
        (&[2, 1], &[1, 3], &[]),
        (&[1, 3], &[2, 1], &[2, 3]),
        (&[2, 1, 3], &[], &[1, 4, 1]),
        (&[0, 1], &[1, 3], &[3]),
        (&[1, 0, 3], &[2, 1, 1], &[1, 1, 3]),
    ];
    let mut batch = OracleBatch::default();
    for condition_dtype in [DType::Bool, DType::F64, DType::Complex64, DType::Complex128] {
        for x_dtype in [DType::Complex64, DType::Complex128] {
            for y_dtype in [
                DType::Bool,
                DType::I16,
                DType::F16,
                DType::F32,
                DType::F64,
                DType::Complex64,
                DType::Complex128,
            ] {
                for &(condition_shape, x_shape, y_shape) in shapes {
                    let condition = sample(condition_shape, condition_dtype, 0);
                    let x = sample(x_shape, x_dtype, 1);
                    let y = sample(y_shape, y_dtype, 3);
                    batch.push(&condition, &x, &y);
                    batch.push(&condition, &y, &x);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn where_select_complex_output_tests_conditions_at_declared_precision() {
    let mut batch = OracleBatch::default();
    for (condition_dtype, minimum_subnormal) in [
        (DType::F16, 2.0_f64.powi(-24)),
        (DType::F32, 2.0_f64.powi(-149)),
    ] {
        // `new` retains the working f64 payload: truth must first round it to
        // the declared source dtype, where the first two values become zero.
        let condition = UFuncArray::new(
            vec![6],
            vec![
                minimum_subnormal / 4.0,
                -minimum_subnormal / 4.0,
                minimum_subnormal,
                f64::NAN,
                f64::INFINITY,
                0.0,
            ],
            condition_dtype,
        )
        .unwrap();
        for dtype in [DType::Complex64, DType::Complex128] {
            let x = complex(&[], &[(1.0, 2.0)], dtype);
            let y = complex(&[], &[(3.0, 4.0)], dtype);
            let result = UFuncArray::where_select(&condition, &x, &y).unwrap();
            assert_eq!(result.shape(), &[6, 2]);
            assert_eq!(
                result.values(),
                &[3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0]
            );
            batch.push(&condition, &x, &y);
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn where_select_complex_conditions_preserve_real_branch_source_precision() {
    let mut batch = OracleBatch::default();
    for condition_dtype in [DType::Complex64, DType::Complex128] {
        let condition = complex(
            &[4],
            &[(0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0)],
            condition_dtype,
        );
        for (dtype, midpoint, tiny) in [
            (DType::F16, 2049.0, 2.0_f64.powi(-26)),
            (DType::F32, 16777217.0, 2.0_f64.powi(-151)),
        ] {
            let source =
                UFuncArray::new(vec![4], vec![midpoint, -midpoint, tiny, -tiny], dtype).unwrap();
            for other_dtype in [dtype, DType::F64] {
                let other =
                    UFuncArray::new(vec![4], vec![-midpoint, midpoint, -tiny, tiny], other_dtype)
                        .unwrap();
                batch.push(&condition, &source, &other);
                batch.push(&condition, &other, &source);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn where_select_complex_conditions_preserve_exact_integer_branches() {
    let mut batch = OracleBatch::default();
    for condition_dtype in [DType::Bool, DType::Complex64, DType::Complex128] {
        let condition = sample(&[4], condition_dtype, 0);
        let signed = UFuncArray::from_storage(
            vec![4],
            ArrayStorage::I64(vec![
                i64::MIN,
                -(1_i64 << 53) - 1,
                (1_i64 << 53) + 1,
                i64::MAX,
            ]),
        )
        .unwrap();
        let signed_other = UFuncArray::from_storage(
            vec![4],
            ArrayStorage::I64(vec![-1, i64::MIN + 1, i64::MAX - 1, 0]),
        )
        .unwrap();
        let unsigned = UFuncArray::from_storage(
            vec![4],
            ArrayStorage::U64(vec![u64::MAX, (1_u64 << 53) + 1, (1_u64 << 63) + 1, 0]),
        )
        .unwrap();
        let unsigned_other = UFuncArray::from_storage(
            vec![4],
            ArrayStorage::U64(vec![0, u64::MAX - 1, 1, (1_u64 << 53) + 3]),
        )
        .unwrap();
        batch.push(&condition, &signed, &signed_other);
        batch.push(&condition, &unsigned, &unsigned_other);
        let complex_branch = sample(&[4], DType::Complex64, 1);
        for integer in [&signed, &unsigned] {
            batch.push(&condition, integer, &complex_branch);
            batch.push(&condition, &complex_branch, integer);
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn where_select_checks_complex_layout_and_logical_broadcast_before_selection() {
    let scalar = UFuncArray::scalar(1.0, DType::F64);
    let false_condition = UFuncArray::scalar(0.0, DType::Bool);
    let valid_complex = complex(&[], &[(2.0, 3.0)], DType::Complex128);
    for malformed in [
        UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::Complex64).unwrap(),
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
    ] {
        for result in [
            UFuncArray::where_select(&malformed, &scalar, &scalar),
            UFuncArray::where_select(&false_condition, &malformed, &scalar),
            UFuncArray::where_select(&scalar, &valid_complex, &malformed),
        ] {
            assert!(matches!(result, Err(UFuncError::Msg(message))
                if message.contains("interleaved trailing-dimension shape")));
        }
    }
    let condition = sample(&[3], DType::Bool, 0);
    let incompatible = sample(&[2], DType::Complex128, 1);
    assert!(matches!(
        UFuncArray::where_select(&condition, &incompatible, &valid_complex),
        Err(UFuncError::Shape(_))
    ));

    // A zero dimension keeps the output empty even when other dimensions would
    // overflow a physical component count. No unused stride product may panic.
    let empty = complex(&[0, usize::MAX], &[], DType::Complex128);
    let result = UFuncArray::where_select(&false_condition, &empty, &valid_complex).unwrap();
    assert_eq!(result.shape(), &[0, usize::MAX, 2]);
    assert!(result.values().is_empty());
}

#[test]
fn select_preserves_complex_pairs_with_broadcasting_and_first_match_precedence() {
    let first = UFuncArray::new(vec![2, 1], vec![1.0, 0.0], DType::Bool).unwrap();
    let second = UFuncArray::new(vec![1, 3], vec![1.0, 0.0, 1.0], DType::Bool).unwrap();
    let third = UFuncArray::scalar(0.0, DType::Bool);
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let first_choice = complex(
            &[1, 3],
            &[
                (1.25, -0.0),
                (-0.0, 2.5),
                (f64::INFINITY, f64::NEG_INFINITY),
            ],
            dtype,
        );
        let second_choice = complex(&[], &[(7.5, -4.25)], dtype);
        let third_choice = sample(&[2, 1], DType::F32, 2);
        for default in [-0.0, -1.25, f64::NAN, f64::INFINITY] {
            batch.push_select(
                &[&first, &second, &third],
                &[&first_choice, &second_choice, &third_choice],
                default,
            );
            batch.push_select(
                &[&second, &first, &third],
                &[&second_choice, &first_choice, &third_choice],
                default,
            );
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn select_numeric_dtype_pairs_and_float_defaults_match_numpy() {
    let dtypes = [
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
    let first = UFuncArray::new(vec![3], vec![1.0, 0.0, 0.0], DType::Bool).unwrap();
    let second = UFuncArray::new(vec![3], vec![1.0, 1.0, 0.0], DType::Bool).unwrap();
    let mut batch = OracleBatch::default();
    for left_dtype in dtypes {
        for right_dtype in dtypes {
            let left = sample(&[3], left_dtype, 0);
            let right = sample(&[3], right_dtype, 2);
            for default in [0.0, -0.0, 0.1, 1.0e100, f64::NAN, f64::NEG_INFINITY] {
                batch.push_select(&[&first, &second], &[&left, &right], default);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn select_mixed_signed_unsigned_and_float_choices_use_common_dtype() {
    let condition = UFuncArray::scalar(1.0, DType::Bool);
    let mut batch = OracleBatch::default();
    for (signed, unsigned, floating) in [
        (DType::I8, DType::U8, DType::F16),
        (DType::I16, DType::U16, DType::F32),
        (DType::I16, DType::U16, DType::Complex64),
        (DType::I64, DType::U64, DType::F32),
        (DType::I64, DType::U64, DType::Complex64),
    ] {
        let inputs = [
            sample(&[], signed, 0),
            sample(&[], unsigned, 4),
            sample(&[], floating, 2),
        ];
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            batch.push_select(
                &[&condition, &condition, &condition],
                &[&inputs[order[0]], &inputs[order[1]], &inputs[order[2]]],
                0.0,
            );
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn select_casts_source_precision_and_exact_integer_sidecars_before_selection() {
    let condition = UFuncArray::new(vec![4], vec![1.0, 0.0, 1.0, 0.0], DType::Bool).unwrap();
    let fallback_condition = UFuncArray::scalar(1.0, DType::Bool);
    let signed = UFuncArray::from_storage(
        vec![4],
        ArrayStorage::I64(vec![
            i64::MIN,
            -(1_i64 << 53) - 1,
            (1_i64 << 53) + 1,
            i64::MAX,
        ]),
    )
    .unwrap();
    let unsigned = UFuncArray::from_storage(
        vec![4],
        ArrayStorage::U64(vec![u64::MAX, (1_u64 << 53) + 1, (1_u64 << 63) + 1, 0]),
    )
    .unwrap();
    let mut batch = OracleBatch::default();
    for (dtype, midpoint, tiny) in [
        (DType::F16, 2049.0, 2.0_f64.powi(-26)),
        (DType::F32, 16777217.0, 2.0_f64.powi(-151)),
    ] {
        let narrow =
            UFuncArray::new(vec![4], vec![midpoint, -midpoint, tiny, -tiny], dtype).unwrap();
        for wide in [&signed, &unsigned] {
            batch.push_select(&[&condition, &fallback_condition], &[&narrow, wide], -0.0);
            batch.push_select(&[&condition, &fallback_condition], &[wide, &narrow], -0.0);
        }
        for other_dtype in [DType::F64, DType::Complex64, DType::Complex128] {
            let other = sample(&[4], other_dtype, 1);
            batch.push_select(&[&condition, &fallback_condition], &[&narrow, &other], -0.0);
        }
    }
    let complex_narrow = sample(&[4], DType::Complex64, 3);
    for other in [&signed, &unsigned] {
        batch.push_select(
            &[&condition, &fallback_condition],
            &[&complex_narrow, other],
            0.0,
        );
        batch.push_select(
            &[&condition, &fallback_condition],
            &[other, &complex_narrow],
            0.0,
        );
    }
    batch.assert_matches_numpy();
}

#[test]
fn select_scalar_and_empty_logical_shapes_match_numpy() {
    let mut batch = OracleBatch::default();
    for shape in [&[][..], &[0][..], &[2, 0, 3][..]] {
        let scalar_condition = UFuncArray::scalar(1.0, DType::Bool);
        let condition = sample(shape, DType::Bool, 0);
        for dtype in [DType::F16, DType::F64, DType::Complex64, DType::Complex128] {
            let choice = sample(shape, dtype, 1);
            let scalar_choice = sample(&[], dtype, 3);
            batch.push_select(
                &[&condition, &scalar_condition],
                &[&choice, &scalar_choice],
                -0.0,
            );
        }
    }
    batch.assert_matches_numpy();

    let scalar_condition = UFuncArray::scalar(0.0, DType::Bool);
    for dtype in [DType::F64, DType::Complex128] {
        let empty = sample(&[0, usize::MAX], dtype, 0);
        let scalar = sample(&[], dtype, 0);
        let result = UFuncArray::select(
            &[&scalar_condition, &scalar_condition],
            &[&empty, &scalar],
            0.0,
        )
        .unwrap();
        assert!(result.values().is_empty());
        assert_eq!(result.shape(), empty.shape());
    }
}

#[test]
fn select_checks_broadcast_output_capacity_before_allocation() {
    let condition = UFuncArray::scalar(0.0, DType::Bool);
    for ndim in [usize::BITS as usize - 4, usize::BITS as usize] {
        for dtype in [DType::F64, DType::Complex128] {
            // Each input stores only two logical elements, but together their
            // independent axes request more output bytes than an allocation can
            // represent. Even never-selected choices participate in the shape.
            let choices: Vec<UFuncArray> = (0..ndim)
                .map(|axis| {
                    let mut shape = vec![1; ndim];
                    shape[axis] = 2;
                    sample(&shape, dtype, axis)
                })
                .collect();
            let conditions = vec![&condition; ndim];
            let choices = choices.iter().collect::<Vec<_>>();
            let result = UFuncArray::select(&conditions, &choices, 0.0);
            assert!(matches!(result, Err(UFuncError::Shape(_))));
        }
    }
}

#[test]
fn select_rejects_invalid_conditions_and_unselected_malformed_choices() {
    let valid = complex(&[], &[(1.0, 2.0)], DType::Complex128);
    let bool_condition = UFuncArray::scalar(0.0, DType::Bool);
    let mut batch = OracleBatch::default();
    for dtype in [
        DType::I8,
        DType::U64,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        let condition = sample(&[], dtype, 1);
        batch.push_select(&[&condition], &[&valid], 0.0);
    }
    let condition = sample(&[3], DType::Bool, 0);
    let incompatible = sample(&[2], DType::Complex128, 0);
    batch.push_select(&[&condition], &[&incompatible], 0.0);
    batch.push_select(&[], &[], 0.0);
    batch.push_select(&[&bool_condition], &[], 0.0);
    batch.assert_matches_numpy();

    for malformed in [
        UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::Complex64).unwrap(),
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
    ] {
        let result = UFuncArray::select(&[&bool_condition], &[&malformed], 0.0);
        assert!(matches!(result, Err(UFuncError::Msg(message))
            if message.contains("interleaved trailing-dimension shape")));
    }
}

struct SelectCallbackGuard(Option<FloatErrorCallback>);

impl Drop for SelectCallbackGuard {
    fn drop(&mut self) {
        let _ = seterrcall(self.0.take());
    }
}

#[test]
fn select_default_cast_errors_match_numpy_warn_call_and_raise() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let callback_calls = Arc::clone(&calls);
    let _callback = SelectCallbackGuard(seterrcall(Some(Arc::new(move |event| {
        callback_calls.lock().unwrap().push(event);
    }))));
    let mut cases = Vec::new();
    let mut actual = Vec::new();
    for dtype in [
        DType::F16,
        DType::F32,
        DType::Complex64,
        DType::F64,
        DType::Complex128,
    ] {
        let (tiny, subnormal) = if dtype == DType::F16 {
            (2.0_f64.powi(-14), 2.0_f64.powi(-24))
        } else {
            (f64::from(f32::MIN_POSITIVE), f64::from(f32::from_bits(1)))
        };
        let quarter_boundary = tiny - subnormal * 0.25;
        let defaults = [
            0.0,
            -0.0,
            1.0e100,
            -1.0e100,
            1.0e-50,
            -1.0e-50,
            1.0e-40,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            subnormal,
            -subnormal,
            subnormal * 0.5,
            subnormal * 1.5,
            tiny - subnormal,
            tiny - subnormal * 0.5,
            quarter_boundary.next_down(),
            quarter_boundary,
            quarter_boundary.next_up(),
            tiny - subnormal * 0.125,
            tiny,
            tiny + subnormal * 0.25,
        ];
        for shape in [&[][..], &[8][..], &[0][..]] {
            let choice = sample(shape, dtype, 0);
            for truth in [false, true] {
                let condition = UFuncArray::scalar(if truth { 1.0 } else { 0.0 }, DType::Bool);
                for default in defaults {
                    for mode in [
                        FloatErrorMode::Warn,
                        FloatErrorMode::Call,
                        FloatErrorMode::Raise,
                    ] {
                        cases.push(json!({
                            "dtype": dtype_name(dtype), "shape": shape,
                            "truth": truth, "default": default.to_string(), "mode": mode.as_str(),
                        }));
                        take_float_error_events();
                        calls.lock().unwrap().clear();
                        let result = {
                            let _errors = errstate(
                                Some(FloatErrorMode::Ignore),
                                None,
                                Some(mode),
                                Some(mode),
                                None,
                            );
                            UFuncArray::select(&[&condition], &[&choice], default)
                        };
                        let error = match result {
                            Ok(_) => None,
                            Err(UFuncError::FloatingPoint { kind, detail }) => {
                                assert!(matches!(
                                    kind,
                                    FloatErrorKind::Over | FloatErrorKind::Under
                                ));
                                Some(detail)
                            }
                            Err(error) => panic!("unexpected select error: {error}"),
                        };
                        let messages = take_float_error_events()
                            .into_iter()
                            .map(|event| {
                                assert_eq!(event.mode, FloatErrorMode::Warn);
                                assert_eq!(event.op, "cast");
                                event.message
                            })
                            .collect::<Vec<_>>();
                        let callback_messages = calls
                            .lock()
                            .unwrap()
                            .iter()
                            .map(|event| {
                                assert_eq!(event.mode, FloatErrorMode::Call);
                                assert_eq!(event.op, "cast");
                                let flag = match event.kind {
                                    FloatErrorKind::Over => 2,
                                    FloatErrorKind::Under => 4,
                                    _ => panic!("unexpected select callback kind"),
                                };
                                json!([event.message, flag])
                            })
                            .collect::<Vec<_>>();
                        actual.push(json!({"error": error, "messages": messages, "calls": callback_messages}));
                    }
                }
            }
        }
    }
    let script = r#"
import json, sys, warnings
import numpy as np

rows = []
for case in json.load(sys.stdin):
    choice = np.zeros(case['shape'], dtype=case['dtype'])
    calls = []
    def record(error, flag):
        calls.append([error + ' encountered in cast', flag])
    np.seterrcall(record)
    error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with np.errstate(all='ignore', over=case['mode'], under=case['mode']):
            try:
                np.select([case['truth']], [choice], float(case['default']))
            except FloatingPointError as failure:
                error = str(failure)
    rows.append(dict(error=error, messages=[str(warning.message) for warning in caught], calls=calls))
print(json.dumps(rows))
"#;
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let mut child = Command::new(python)
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch live NumPy select default cast oracle");
    let mut stdin = child.stdin.take().unwrap();
    serde_json::to_writer(&mut stdin, &cases).unwrap();
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(expected.len(), actual.len());
    for ((case, actual), expected) in cases.iter().zip(actual).zip(expected) {
        assert_eq!(actual, expected, "select default cast policy: {case}");
    }
}

#[test]
fn select_validates_inputs_before_dispatching_default_cast_errors() {
    let _errors = errstate(
        Some(FloatErrorMode::Ignore),
        None,
        Some(FloatErrorMode::Raise),
        Some(FloatErrorMode::Raise),
        None,
    );
    let condition = UFuncArray::scalar(0.0, DType::Bool);
    let invalid_condition = UFuncArray::scalar(1.0, DType::F64);
    let valid_choice = sample(&[], DType::Complex64, 0);
    let malformed = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::Complex64).unwrap();
    let incompatible_condition = sample(&[3], DType::Bool, 0);
    let incompatible_choice = sample(&[2], DType::Complex64, 0);
    for default in [1.0e100, 1.0e-50] {
        assert!(matches!(
            UFuncArray::select(&[&invalid_condition], &[&valid_choice], default),
            Err(UFuncError::TypeResolutionInvalid { .. })
        ));
        assert!(matches!(
            UFuncArray::select(&[&condition], &[&malformed], default),
            Err(UFuncError::Msg(_))
        ));
        assert!(matches!(
            UFuncArray::select(&[&incompatible_condition], &[&incompatible_choice], default),
            Err(UFuncError::Shape(_))
        ));
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn select_allocation_failure_precedes_default_cast_errors() {
    let condition = UFuncArray::scalar(0.0, DType::Bool);
    const NDIM: usize = 29;
    for dtype in [DType::F16, DType::F32, DType::Complex64] {
        // Only four logical elements per input, but their broadcast result
        // requires 2^61 bytes (2^62 for complex storage). This fits usize and
        // isize while exceeding supported process virtual address spaces, so
        // reservation must fail without allocating or touching physical pages.
        let choices: Vec<UFuncArray> = (0..NDIM)
            .map(|axis| {
                let mut shape = vec![1; NDIM];
                shape[axis] = 4;
                sample(&shape, dtype, 0)
            })
            .collect();
        let conditions = vec![&condition; NDIM];
        let choices = choices.iter().collect::<Vec<_>>();
        for default in [1.0e100, 1.0e-50] {
            for mode in [FloatErrorMode::Warn, FloatErrorMode::Raise] {
                take_float_error_events();
                let _errors = errstate(
                    Some(FloatErrorMode::Ignore),
                    None,
                    Some(mode),
                    Some(mode),
                    None,
                );
                let result = UFuncArray::select(&conditions, &choices, default);
                assert!(matches!(result, Err(UFuncError::Msg(message))
                    if message.contains("unable to allocate output array")));
                assert!(take_float_error_events().is_empty());
            }
        }
    }
}
