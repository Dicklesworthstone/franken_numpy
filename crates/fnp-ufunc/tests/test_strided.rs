use fnp_dtype::DType;
use fnp_ufunc::{UFuncArray, UFuncError};

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
