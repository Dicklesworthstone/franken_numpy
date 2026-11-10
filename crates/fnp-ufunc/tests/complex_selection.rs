//! Conditional selection acts on logical complex elements, never components.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
use std::process::{Command, Stdio};

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
