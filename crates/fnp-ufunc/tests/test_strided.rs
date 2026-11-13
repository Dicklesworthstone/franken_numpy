use fnp_dtype::DType;
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn default_sum_preserves_numpy_contiguous_and_strided_order() {
    let mut cases = Vec::new();
    let mut actual = Vec::new();
    for len in [0, 1, 2, 8, 9, 127, 128, 129, 8193] {
        for pattern in [
            [1e16, 1.0, -1e16, 1.0],
            [1e308, 1e308, -1e308, -1e308],
            [-0.0; 4],
            [f64::INFINITY, 1.0, f64::NEG_INFINITY, 2.0],
        ] {
            let values: Vec<f64> = (0..len * 2).map(|i| pattern[i % 4]).collect();
            for shape in [vec![2, len], vec![len, 2]] {
                let array = UFuncArray::new(shape.clone(), values.clone(), DType::F64).unwrap();
                for axis in [None, Some(0), Some(-1)] {
                    for keepdims in [false, true] {
                        cases.push(json!({
                            "shape": shape, "axis": axis, "keepdims": keepdims,
                            "values": values.iter().map(f64::to_string).collect::<Vec<_>>(),
                        }));
                        actual.push(array.reduce_sum(axis, keepdims).unwrap());
                    }
                }
            }
        }
    }
    let script = r#"
import json, sys, warnings
import numpy as np
out = []
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for case in json.load(sys.stdin):
        a = np.array(case['values'], dtype=np.float64).reshape(case['shape'])
        result = np.asarray(np.sum(a, axis=case['axis'], keepdims=case['keepdims']))
        out.append({'shape': list(result.shape), 'bits': result.reshape(-1).view(np.uint64).tolist()})
print(json.dumps(out))
"#;
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mut child = Command::new(python)
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch NumPy sum oracle");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&cases).unwrap())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            json!(actual.shape()),
            expected["shape"],
            "case {index} shape"
        );
        assert_eq!(actual.dtype(), DType::F64);
        let bits = expected["bits"].as_array().unwrap();
        assert_eq!(actual.values().len(), bits.len(), "case {index} length");
        for (&value, bits) in actual.values().iter().zip(bits) {
            let bits = bits.as_u64().unwrap();
            assert!(
                value.to_bits() == bits || (value.is_nan() && f64::from_bits(bits).is_nan()),
                "case {index}: sum {value:?}, NumPy {:?}",
                f64::from_bits(bits)
            );
        }
    }
}

#[test]
fn default_arithmetic_reductions_accept_scalar_axis_aliases() {
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::U32,
        DType::I64,
        DType::F16,
        DType::F32,
        DType::F64,
    ] {
        let scalar = UFuncArray::scalar(1.0, dtype);
        for axis in [Some(0), Some(-1)] {
            for keepdims in [false, true] {
                for result in [
                    scalar.reduce_sum(axis, keepdims),
                    scalar.reduce_prod(axis, keepdims),
                ] {
                    let result = result.expect("NumPy accepts axis 0 and -1 for scalar sum/prod");
                    assert!(result.shape().is_empty());
                    assert_eq!(result.values(), &[1.0]);
                }
            }
        }
        assert!(scalar.reduce_sum(Some(1), false).is_err());
        assert!(scalar.reduce_prod(Some(-2), false).is_err());
    }
}

#[test]
fn default_arithmetic_reductions_validate_empty_output_shapes() {
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::U64,
        DType::F16,
        DType::F32,
        DType::F64,
    ] {
        let impossible = UFuncArray::new(vec![0, usize::MAX], Vec::new(), dtype).unwrap();
        for result in [
            impossible.reduce_sum(Some(0), false),
            impossible.reduce_prod(Some(0), true),
        ] {
            assert!(matches!(
                result,
                Err(UFuncError::Shape(fnp_ndarray::ShapeError::Overflow))
            ));
        }
        let empty = UFuncArray::new(vec![0, 1, usize::MAX, 2], Vec::new(), dtype).unwrap();
        for keepdims in [false, true] {
            for result in [
                empty.reduce_sum(Some(1), keepdims),
                empty.reduce_prod(Some(1), keepdims),
            ] {
                let result =
                    result.expect("empty output does not calculate unused stride products");
                assert!(result.values().is_empty());
                assert_eq!(
                    result.shape(),
                    if keepdims {
                        &[0, 1, usize::MAX, 2][..]
                    } else {
                        &[0, usize::MAX, 2][..]
                    }
                );
            }
        }
    }
}

#[test]
fn test_strided_backwards() -> Result<(), UFuncError> {
    // Create [5, 6, 7, 8, 9], then slice to get a view starting at offset 4
    // so that stride -1 can traverse backwards: offsets 4, 3, 2 → values 9, 8, 7
    let a = UFuncArray::arange(0.0, 10.0, 1.0, DType::F64)?;
    let reversed = a.slice_axis(0, None, None, -1)?;
    let view = reversed.shared_view()?;
    // The reversed view sees [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] via negative stride
    let sub = view.slice_axis(0, None, Some(3), 1)?;
    // sub = [9, 8, 7] — 3 elements
    assert_eq!(sub.shape(), &[3]);
    assert_eq!(sub.item(&[0])?, 9.0);
    assert_eq!(sub.item(&[1])?, 8.0);
    assert_eq!(sub.item(&[2])?, 7.0);
    Ok(())
}

#[test]
fn default_integer_modular_arithmetic_avoids_float_casts() {
    use fnp_dtype::ArrayStorage;
    use fnp_ufunc::BinaryOp;

    for (lhs, rhs, op, expected) in [
        (
            ArrayStorage::I32(vec![i32::MAX]),
            ArrayStorage::I32(vec![i32::MAX]),
            BinaryOp::Add,
            ArrayStorage::I32(vec![-2]),
        ),
        (
            ArrayStorage::I32(vec![i32::MAX]),
            ArrayStorage::I32(vec![i32::MAX]),
            BinaryOp::Mul,
            ArrayStorage::I32(vec![1]),
        ),
        (
            ArrayStorage::U16(vec![u16::MAX]),
            ArrayStorage::U16(vec![u16::MAX]),
            BinaryOp::Mul,
            ArrayStorage::U16(vec![1]),
        ),
        (
            ArrayStorage::U32(vec![u32::MAX]),
            ArrayStorage::U32(vec![u32::MAX]),
            BinaryOp::Mul,
            ArrayStorage::U32(vec![1]),
        ),
        (
            ArrayStorage::I64(vec![3_002_399_751_580_331]),
            ArrayStorage::I64(vec![3]),
            BinaryOp::Mul,
            ArrayStorage::I64(vec![9_007_199_254_740_993]),
        ),
    ] {
        let lhs = UFuncArray::from_storage(vec![1], lhs).unwrap();
        let rhs = UFuncArray::from_storage(vec![1], rhs).unwrap();
        let actual = lhs.elementwise_binary(&rhs, op).unwrap();
        assert_eq!(actual.to_storage().unwrap(), expected, "{op:?}");
    }
}

#[test]
fn integer_arithmetic_reports_integer_loop_errors() {
    use fnp_dtype::ArrayStorage;
    use fnp_ufunc::{BinaryOp, FloatErrorKind, FloatErrorMode, errstate, take_float_error_events};

    let _guard = errstate(Some(FloatErrorMode::Warn), None, None, None, None);
    for (dtype, minimum) in [
        (DType::I8, i128::from(i8::MIN)),
        (DType::I16, i128::from(i16::MIN)),
        (DType::I32, i128::from(i32::MIN)),
        (DType::I64, i128::from(i64::MIN)),
    ] {
        let mut lhs = ArrayStorage::zeros(dtype, 3);
        let mut rhs = ArrayStorage::zeros(dtype, 3);
        lhs.set_i128(0, minimum).unwrap();
        lhs.set_i128(1, 3).unwrap();
        rhs.set_i128(0, -1).unwrap();
        let lhs = UFuncArray::from_storage(vec![3], lhs).unwrap();
        let rhs = UFuncArray::from_storage(vec![3], rhs).unwrap();
        for op in [BinaryOp::FloorDivide, BinaryOp::Remainder, BinaryOp::Fmod] {
            take_float_error_events();
            let result = lhs.elementwise_binary(&rhs, op).unwrap();
            let values = result.to_storage().unwrap();
            let first = if op == BinaryOp::FloorDivide {
                minimum
            } else {
                0
            };
            assert_eq!(values.get_i128(0).unwrap(), first, "{dtype:?} {op:?}");
            assert_eq!(values.get_i128(1).unwrap(), 0);
            assert_eq!(values.get_i128(2).unwrap(), 0);
            let kinds: Vec<_> = take_float_error_events()
                .iter()
                .map(|event| event.kind)
                .collect();
            let expected = if op == BinaryOp::FloorDivide {
                vec![FloatErrorKind::Divide, FloatErrorKind::Over]
            } else {
                vec![FloatErrorKind::Divide]
            };
            assert_eq!(kinds, expected, "{dtype:?} {op:?}");
        }
    }

    let _raise = errstate(Some(FloatErrorMode::Raise), None, None, None, None);
    let minimum = UFuncArray::from_storage(vec![1], ArrayStorage::I32(vec![i32::MIN])).unwrap();
    let negative_one = UFuncArray::from_storage(vec![1], ArrayStorage::I32(vec![-1])).unwrap();
    assert!(matches!(
        minimum.elementwise_binary(&negative_one, BinaryOp::FloorDivide),
        Err(UFuncError::FloatingPoint {
            kind: FloatErrorKind::Over,
            ..
        })
    ));
    let zero = UFuncArray::from_storage(vec![1], ArrayStorage::I32(vec![0])).unwrap();
    for op in [BinaryOp::FloorDivide, BinaryOp::Remainder, BinaryOp::Fmod] {
        assert!(matches!(
            minimum.elementwise_binary(&zero, op),
            Err(UFuncError::FloatingPoint {
                kind: FloatErrorKind::Divide,
                ..
            })
        ));
    }
}

#[test]
fn integer_power_rejects_negative_exponents_only_for_evaluated_elements() {
    use fnp_dtype::ArrayStorage;
    use fnp_ufunc::{BinaryOp, FloatErrorMode, errstate};

    let _guard = errstate(Some(FloatErrorMode::Raise), None, None, None, None);
    for dtype in [DType::I8, DType::I16, DType::I32, DType::I64] {
        let lhs = UFuncArray::from_storage(vec![2, 1], ArrayStorage::zeros(dtype, 2)).unwrap();
        let mut powers = ArrayStorage::zeros(dtype, 2);
        powers.set_i128(0, -1).unwrap();
        powers.set_i128(1, 2).unwrap();
        let rhs = UFuncArray::from_storage(vec![2], powers).unwrap();
        let error = lhs.elementwise_binary(&rhs, BinaryOp::Power).unwrap_err();
        assert!(
            matches!(&error, UFuncError::Msg(message) if message.contains("negative integer powers")),
            "{dtype:?}: {error}"
        );
        let empty = UFuncArray::from_storage(vec![0, 1], ArrayStorage::zeros(dtype, 0)).unwrap();
        let result = empty.elementwise_binary(&rhs, BinaryOp::Power).unwrap();
        assert_eq!(result.shape(), &[0, 2]);
        assert_eq!(result.dtype(), dtype);
        assert!(result.values().is_empty());
    }
}

#[test]
fn masked_f64_sum_preserves_true_run_boundaries() -> Result<(), UFuncError> {
    // NumPy sums each contiguous true-mask run separately. The first
    // seven-element run is linear; the second eight-element run is pairwise.
    let row = [1e16, 1.0, -1e16, 1.0, 1.0, 1.0, 1.0, 1.0];
    let array = UFuncArray::new(vec![2, 8], row.repeat(2), DType::F64)?;
    let mut selected = vec![1.0; 16];
    selected[7] = 0.0;
    let mask = UFuncArray::new(vec![2, 8], selected, DType::Bool)?;
    for keepdims in [false, true] {
        let all = array.reduce_sum_where(&mask, None, keepdims)?;
        assert_eq!(all.values(), &[8.0]);
        let rows = array.reduce_sum_where(&mask, Some(-1), keepdims)?;
        assert_eq!(rows.values(), &[4.0, 4.0]);
        let columns = array.reduce_sum_where(&mask, Some(0), keepdims)?;
        assert_eq!(
            columns.values(),
            &[2e16, 2.0, -2e16, 2.0, 2.0, 2.0, 2.0, 1.0]
        );
        assert_eq!(all.shape(), if keepdims { &[1, 1][..] } else { &[][..] });
        assert_eq!(rows.shape(), if keepdims { &[2, 1][..] } else { &[2][..] });
    }
    let empty = UFuncArray::new(vec![0, 2], vec![], DType::F64)?;
    let include = UFuncArray::scalar(1.0, DType::Bool);
    assert_eq!(
        empty.reduce_sum_where(&include, Some(0), false)?.values(),
        &[0.0, 0.0]
    );
    assert!(
        empty
            .reduce_sum_where(&include, Some(1), false)?
            .values()
            .is_empty()
    );
    let scalar = UFuncArray::scalar(3.0, DType::F64);
    for axis in [None, Some(0), Some(-1)] {
        assert_eq!(
            scalar.reduce_sum_where(&include, axis, false)?.values(),
            &[3.0]
        );
    }
    let exclude = UFuncArray::scalar(0.0, DType::Bool);
    assert_eq!(
        array.reduce_sum_where(&exclude, None, false)?.values(),
        &[0.0]
    );
    Ok(())
}
