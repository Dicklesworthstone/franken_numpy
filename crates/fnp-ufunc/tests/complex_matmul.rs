//! Logical complex matrix contractions, typed promotion, and vector conventions.
//!
//! Numerical checks use componentwise forward-error bounds: portable typed
//! accumulation and vendor BLAS may use different reduction orders or FMA.
//! Exceptional-value behavior of vendor BLAS is not claimed to be bit-identical.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ndarray::ShapeError;
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
use std::process::{Command, Stdio};

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
        _ => panic!("unsupported oracle dtype"),
    }
}

fn source(shape: &[usize], dtype: DType, seed: usize) -> UFuncArray {
    let count: usize = shape.iter().product();
    if dtype.is_complex() {
        let mut physical_shape = shape.to_vec();
        physical_shape.push(2);
        let values = (0..count)
            .flat_map(|index| {
                let real = match (index + seed) % 13 {
                    0 => 1.0 + 2.0_f64.powi(-24),
                    1 => 1.0e-50,
                    _ => ((index * 3 + seed) % 17) as f64 / 4.0 - 2.0,
                };
                let imaginary = ((index * 5 + seed) % 19) as f64 / 4.0 - 2.25;
                [real, imaginary]
            })
            .collect();
        UFuncArray::new(physical_shape, values, dtype).unwrap()
    } else {
        let values = (0..count)
            .map(|index| ((index * 3 + seed) % 17) as f64 / 4.0 - 2.0)
            .collect();
        UFuncArray::from_storage(
            shape.to_vec(),
            ArrayStorage::F64(values).cast_to(dtype).unwrap(),
        )
        .unwrap()
    }
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
        json!(array.values().iter().map(f64::to_string).collect::<Vec<_>>())
    };
    json!({"shape": shape, "dtype": dtype_name(array.dtype()), "values": values})
}

#[derive(Default)]
struct OracleBatch {
    cases: Vec<Value>,
    actual: Vec<Result<UFuncArray, UFuncError>>,
}

impl OracleBatch {
    fn case(&mut self, op: &str, lhs: &UFuncArray, rhs: &UFuncArray, axis: Option<isize>) {
        let actual = match op {
            "matmul" => lhs.matmul(rhs),
            "matvec" => UFuncArray::matvec(lhs, rhs),
            "vecmat" => UFuncArray::vecmat(lhs, rhs),
            "vdot" => lhs.vdot(rhs),
            "vecdot" => lhs.vecdot(rhs, axis),
            _ => panic!("unsupported contraction"),
        };
        self.cases.push(json!({
            "op": op, "lhs": oracle_input(lhs), "rhs": oracle_input(rhs), "axis": axis
        }));
        self.actual.push(actual);
    }

    fn finish(self) {
        let script = r#"
import json, sys, warnings
import numpy as np

warnings.simplefilter('ignore')
np.seterr(all='ignore')

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

def operate(op, left, right, axis):
    if op == 'vecdot':
        return np.vecdot(left, right, axis=-1 if axis is None else axis)
    return getattr(np, op)(left, right)

rows = []
for case in json.load(sys.stdin):
    try:
        left, right = load(case['lhs']), load(case['rhs'])
        op, axis = case['op'], case['axis']
        result = np.asarray(operate(op, left, right, axis))
        # Promote each input before deriving the rounding bound, matching the
        # ufunc's typed conversion rather than its original working cells.
        left = left.astype(result.dtype)
        right = right.astype(result.dtype)
        ar, ai = np.abs(left.real.astype(np.float64)), np.abs(left.imag.astype(np.float64))
        br, bi = np.abs(right.real.astype(np.float64)), np.abs(right.imag.astype(np.float64))
        real_bound = operate(op, ar, br, axis) + operate(op, ai, bi, axis)
        imag_bound = operate(op, ar, bi, axis) + operate(op, ai, br, axis)
        if op == 'vdot':
            contracted = left.size
        else:
            contracted = left.shape[(-1 if axis is None else axis) if op == 'vecdot' else -1]
        parts = np.empty(result.size * 2, dtype=np.float64)
        parts[0::2] = result.real.ravel()
        parts[1::2] = result.imag.ravel()
        bounds = np.empty(result.size * 2, dtype=np.float64)
        bounds[0::2] = np.asarray(real_bound).ravel()
        bounds[1::2] = np.asarray(imag_bound).ravel()
        rows.append(dict(shape=list(result.shape), dtype=str(result.dtype),
                         bits=parts.view(np.uint64).tolist(),
                         bounds=bounds.tolist(), contracted=int(contracted)))
    except (ValueError, TypeError, IndexError, OverflowError) as error:
        rows.append(dict(error=type(error).__name__, message=str(error)))
print(json.dumps(rows))
"#;
        let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(python)
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("launch live NumPy contraction oracle");
        serde_json::to_writer(child.stdin.take().unwrap(), &self.cases).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy contraction oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(self.cases.len(), expected.len());
        for ((case, actual), expected) in self.cases.iter().zip(self.actual).zip(expected) {
            if expected.get("error").is_some() {
                assert!(actual.is_err(), "{case}: expected {expected}, got {actual:?}");
                continue;
            }
            let actual = actual.unwrap_or_else(|error| panic!("{case}: {error}"));
            assert!(actual.dtype().is_complex(), "{case}: expected complex output");
            let mut shape = actual.shape().to_vec();
            assert_eq!(shape.pop(), Some(2), "{case}: component axis");
            assert_eq!(json!(shape), expected["shape"], "{case}: logical shape");
            assert_eq!(dtype_name(actual.dtype()), expected["dtype"], "{case}: dtype");
            let bits = expected["bits"].as_array().unwrap();
            let bounds = expected["bounds"].as_array().unwrap();
            assert_eq!(actual.values().len(), bits.len(), "{case}: component count");
            let contracted = expected["contracted"].as_u64().unwrap() as f64;
            let (epsilon, smallest) = if actual.dtype() == DType::Complex64 {
                (f64::from(f32::EPSILON), f64::from(f32::from_bits(1)))
            } else {
                (f64::EPSILON, f64::from_bits(1))
            };
            for (index, ((&actual, expected), bound)) in actual
                .values()
                .iter()
                .zip(bits)
                .zip(bounds)
                .enumerate()
            {
                let expected = f64::from_bits(expected.as_u64().unwrap());
                if expected.is_nan() {
                    assert!(actual.is_nan(), "{case}: component {index} expected NaN");
                } else if expected.is_infinite() {
                    assert_eq!(actual, expected, "{case}: component {index} infinity");
                } else {
                    assert!(actual.is_finite(), "{case}: component {index} must be finite");
                    let scale = bound.as_f64().unwrap();
                    let tolerance = 8.0 * (contracted + 1.0) * (epsilon * scale + smallest);
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{case}: component {index}: {actual} != {expected}; bound {tolerance}"
                    );
                }
            }
        }
    }
}

#[test]
fn complex_matmul_logical_shapes_and_broadcasts_match_numpy() {
    let shapes: &[(&[usize], &[usize])] = &[
        (&[3], &[3]), (&[3], &[3, 4]), (&[2, 3], &[3]),
        (&[2, 3], &[3, 4]), (&[1, 3], &[3, 1]),
        (&[2, 1, 3, 4], &[1, 5, 4, 2]),
        (&[4], &[2, 3, 4, 5]), (&[2, 3, 4, 5], &[5]),
        (&[2, 1, 3, 4], &[4, 2]), (&[3, 4], &[2, 1, 4, 2]),
        (&[0], &[0]), (&[2, 0], &[0, 3]), (&[0, 3], &[3, 4]),
        (&[2, 3], &[3, 0]), (&[0, 2, 3], &[1, 3, 4]),
        (&[2, 1, 3, 0], &[1, 5, 0, 2]),
        (&[], &[3]), (&[3], &[]), (&[], &[]),
        (&[2, 3], &[4, 2]), (&[2, 3, 4], &[3, 4, 2]),
    ];
    let mut batch = OracleBatch::default();
    for lhs_dtype in [DType::Complex64, DType::Complex128] {
        for rhs_dtype in [DType::Complex64, DType::Complex128] {
            for &(lhs_shape, rhs_shape) in shapes {
                batch.case(
                    "matmul",
                    &source(lhs_shape, lhs_dtype, 0),
                    &source(rhs_shape, rhs_dtype, 4),
                    None,
                );
            }
        }
    }
    batch.finish();
}

#[test]
fn complex_matmul_mixed_numeric_types_match_numpy() {
    let dtypes = [
        DType::Bool, DType::I8, DType::I16, DType::I32, DType::I64,
        DType::U8, DType::U16, DType::U32, DType::U64,
        DType::F16, DType::F32, DType::F64, DType::Complex64, DType::Complex128,
    ];
    let mut batch = OracleBatch::default();
    for lhs_dtype in dtypes {
        for rhs_dtype in dtypes {
            if !(lhs_dtype.is_complex() || rhs_dtype.is_complex()) {
                continue;
            }
            for (op, lhs_shape, rhs_shape) in [
                ("matmul", &[2, 3][..], &[3, 4][..]),
                ("matmul", &[3][..], &[3][..]),
                ("matvec", &[2, 3][..], &[3][..]),
                ("vecmat", &[3][..], &[3, 4][..]),
            ] {
                batch.case(
                    op,
                    &source(lhs_shape, lhs_dtype, 1),
                    &source(rhs_shape, rhs_dtype, 5),
                    None,
                );
            }
        }
    }
    batch.finish();
}

#[test]
fn complex_matrix_vector_batches_preserve_singleton_axes() {
    let matvec_shapes: &[(&[usize], &[usize])] = &[
        (&[2, 3], &[3]), (&[2, 3], &[1, 3]),
        (&[1, 2, 3], &[3]), (&[2, 1, 4, 3], &[1, 5, 3]),
        (&[4, 3], &[2, 1, 3]), (&[2, 1, 4, 3], &[5, 3]),
        (&[0, 2, 3], &[1, 3]), (&[2, 0], &[0]),
        (&[3], &[3]), (&[], &[3]), (&[2, 3], &[]),
        (&[2, 3], &[4]), (&[2, 4, 3], &[3, 3]),
    ];
    let vecmat_shapes: &[(&[usize], &[usize])] = &[
        (&[3], &[3, 2]), (&[1, 3], &[3, 2]),
        (&[3], &[1, 3, 2]), (&[2, 1, 3], &[1, 5, 3, 4]),
        (&[2, 1, 3], &[3, 4]), (&[5, 3], &[2, 1, 3, 4]),
        (&[1, 3], &[0, 3, 2]), (&[0], &[0, 2]),
        (&[3], &[3]), (&[], &[3, 2]), (&[3], &[]),
        (&[4], &[3, 2]), (&[2, 3], &[3, 3, 4]),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for (op, shapes) in [("matvec", matvec_shapes), ("vecmat", vecmat_shapes)] {
            for &(lhs_shape, rhs_shape) in shapes {
                batch.case(
                    op,
                    &source(lhs_shape, dtype, 2),
                    &source(rhs_shape, dtype, 7),
                    None,
                );
            }
        }
    }
    batch.finish();
}

#[test]
fn complex_matrix_products_cast_exact_wide_integers() {
    let mut batch = OracleBatch::default();
    for storage in [
        ArrayStorage::I64(vec![i64::MAX, i64::MIN, (1_i64 << 53) + 1, -(1_i64 << 53) - 1]),
        ArrayStorage::U64(vec![u64::MAX, u64::MAX - 1, (1_u64 << 63) + 1, (1_u64 << 53) + 1]),
    ] {
        let matrix = UFuncArray::from_storage(vec![2, 2], storage.clone()).unwrap();
        let vector = UFuncArray::from_storage(vec![4], storage).unwrap();
        for dtype in [DType::Complex64, DType::Complex128] {
            let complex_matrix = source(&[2, 2], dtype, 3);
            let complex_vector = source(&[4], dtype, 8);
            batch.case("matmul", &matrix, &complex_matrix, None);
            batch.case("matmul", &complex_matrix, &matrix, None);
            batch.case("matmul", &vector, &complex_vector, None);
            batch.case("matmul", &complex_vector, &vector, None);
        }
    }
    batch.finish();
}

#[test]
fn vecmat_conjugates_the_vector_but_matmul_and_matvec_do_not() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let vector = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, -4.0], dtype).unwrap();
        let matrix = UFuncArray::new(
            vec![2, 2, 2],
            vec![2.0, -1.0, 1.0, 3.0, -2.0, 4.0, 3.0, -2.0],
            dtype,
        )
        .unwrap();
        let direct = vector.matmul(&matrix).unwrap();
        assert_eq!(direct.shape(), &[2, 2]);
        assert_eq!(direct.values(), &[14.0, 23.0, -4.0, -13.0]);
        let conjugated = UFuncArray::vecmat(&vector, &matrix).unwrap();
        assert_eq!(conjugated.shape(), &[2, 2]);
        assert_eq!(conjugated.values(), &[-22.0, -1.0, 24.0, 7.0]);
        let matrix_vector = UFuncArray::matvec(&matrix, &vector).unwrap();
        assert_eq!(matrix_vector.shape(), &[2, 2]);
        assert_eq!(matrix_vector.values(), &[19.0, 8.0, -9.0, -18.0]);
    }
}

#[test]
fn complex64_accumulates_in_its_component_type() {
    // The f32 accumulation loses the middle unit before cancellation. A mixed
    // complex128 operand promotes the accumulator and retains it.
    let values = [16_777_216.0, 1.0, -16_777_216.0]
        .into_iter()
        .cycle()
        .take(6)
        .flat_map(|value| [value, 0.0])
        .collect();
    let lhs = UFuncArray::new(vec![2, 3, 2], values, DType::Complex64).unwrap();
    for (dtype, expected) in [(DType::Complex64, 0.0), (DType::Complex128, 1.0)] {
        let rhs = UFuncArray::new(
            vec![3, 2, 2],
            (0..6).flat_map(|_| [1.0, 0.0]).collect(),
            dtype,
        )
        .unwrap();
        let output = lhs.matmul(&rhs).unwrap();
        assert_eq!(output.dtype(), dtype);
        assert_eq!(output.shape(), &[2, 2, 2]);
        assert_eq!(output.values(), &[expected, 0.0, expected, 0.0, expected, 0.0, expected, 0.0]);
    }
}

#[test]
fn complex_matmul_rejects_malformed_layouts_and_impossible_outputs() {
    let valid = source(&[2, 2], DType::Complex128, 0);
    for malformed in [
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
        UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::Complex64).unwrap(),
        UFuncArray::new(vec![0, 1], vec![], DType::Complex128).unwrap(),
    ] {
        for result in [
            malformed.matmul(&valid), valid.matmul(&malformed),
            UFuncArray::matvec(&malformed, &valid), UFuncArray::matvec(&valid, &malformed),
            UFuncArray::vecmat(&malformed, &valid), UFuncArray::vecmat(&valid, &malformed),
        ] {
            assert!(result.is_err(), "malformed complex layout must be rejected");
        }
    }
    let huge_empty = UFuncArray::new(
        vec![usize::MAX, 0, 2], vec![], DType::Complex128,
    )
    .unwrap();
    let empty = source(&[0, 0], DType::Complex128, 0);
    let output = huge_empty.matmul(&empty).unwrap();
    assert_eq!(output.shape(), &[usize::MAX, 0, 2]);
    assert!(output.values().is_empty());
    let nonempty_columns = source(&[0, 2], DType::Complex128, 0);
    assert!(matches!(
        huge_empty.matmul(&nonempty_columns),
        Err(UFuncError::Shape(ShapeError::Overflow))
    ));
    let max_count = isize::MAX as usize / std::mem::size_of::<[f64; 2]>();
    let lhs = UFuncArray::new(vec![max_count + 1, 0, 2], vec![], DType::Complex128).unwrap();
    let rhs = source(&[0, 1], DType::Complex128, 0);
    assert!(matches!(lhs.matmul(&rhs), Err(UFuncError::Shape(ShapeError::Overflow))));
}

#[test]
fn complex_vdot_flattens_logical_elements_and_conjugates_left() {
    let mut batch = OracleBatch::default();
    let shapes: &[(&[usize], &[usize])] = &[
        (&[], &[]),
        (&[], &[1]),
        (&[1, 1, 1], &[]),
        (&[6], &[2, 3]),
        (&[2, 3], &[3, 2]),
        (&[0], &[0, 3]),
        (&[2, 0, 3], &[0]),
        (&[2], &[3]),
    ];
    for complex_dtype in [DType::Complex64, DType::Complex128] {
        for other_dtype in [
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
        ] {
            for &(left_shape, right_shape) in shapes {
                let left = source(left_shape, complex_dtype, 1);
                let right = source(right_shape, other_dtype, 3);
                batch.case("vdot", &left, &right, None);
                batch.case("vdot", &right, &left, None);
            }
        }
    }
    // Both components distinguish conjugating the left operand from either
    // an unconjugated product or conjugating the right operand instead.
    let left = UFuncArray::new(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, -4.0],
        DType::Complex128,
    )
    .unwrap();
    let right = UFuncArray::new(
        vec![2, 2],
        vec![5.0, -6.0, 7.0, 8.0],
        DType::Complex128,
    )
    .unwrap();
    batch.case("vdot", &left, &right, None);
    batch.case("vdot", &right, &left, None);
    batch.finish();
}

#[test]
fn complex_vecdot_contracts_independent_logical_axes_and_broadcasts_batches() {
    let layouts: &[(&[usize], &[usize], Option<isize>)] = &[
        (&[3], &[3], None),
        (&[2, 3], &[3], None),
        (&[2, 1, 3], &[1, 4, 3], None),
        (&[2, 1, 3], &[1, 1, 3], Some(-1)),
        (&[3, 2], &[3, 1], Some(0)),
        (&[2, 3, 4], &[3, 4], Some(-2)),
        (&[2, 3, 4], &[1, 3], Some(1)),
        (&[2, 3, 4], &[1, 3], Some(-2)),
        (&[0], &[0], None),
        (&[2, 0], &[1, 0], None),
        (&[2, 0, 3], &[1, 0, 1], Some(1)),
        (&[0, 3], &[1, 3], Some(-1)),
        (&[1, 0, 3], &[2, 1, 3], None),
        (&[2, 3], &[2, 4], None),
        (&[2, 3], &[4, 3], None),
        (&[], &[1], None),
        (&[1], &[], None),
        (&[], &[], None),
    ];
    let mut batch = OracleBatch::default();
    for complex_dtype in [DType::Complex64, DType::Complex128] {
        for other_dtype in [
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
        ] {
            for &(left_shape, right_shape, axis) in layouts {
                let left = source(left_shape, complex_dtype, 1);
                let right = source(right_shape, other_dtype, 3);
                batch.case("vecdot", &left, &right, axis);
                batch.case("vecdot", &right, &left, axis);
            }
        }
        let left = source(&[2, 3], complex_dtype, 1);
        let right = source(&[2, 3], complex_dtype, 3);
        for axis in [Some(-3), Some(2), Some(isize::MIN), Some(isize::MAX)] {
            batch.case("vecdot", &left, &right, axis);
        }
    }
    batch.finish();
}

#[test]
fn complex_vector_dots_preserve_exact_integer_inputs_before_promotion() {
    let mut batch = OracleBatch::default();
    for storage in [
        ArrayStorage::I64(vec![i64::MAX, i64::MIN, (1_i64 << 53) + 1, -1, 0, 1]),
        ArrayStorage::U64(vec![u64::MAX, u64::MAX - 1, (1_u64 << 63) + 1, 0, 1, 2]),
    ] {
        let integers = UFuncArray::from_storage(vec![2, 3], storage).unwrap();
        for dtype in [DType::Complex64, DType::Complex128] {
            let complex = source(&[2, 3], dtype, 2);
            batch.case("vdot", &integers, &complex, None);
            batch.case("vdot", &complex, &integers, None);
            for axis in [None, Some(0), Some(1), Some(-2)] {
                batch.case("vecdot", &integers, &complex, axis);
                batch.case("vecdot", &complex, &integers, axis);
            }
        }
    }
    batch.finish();
}

#[test]
fn complex_vector_dots_reject_malformed_storage() {
    let valid = source(&[2], DType::Complex128, 0);
    for malformed in [
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
        UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::Complex64).unwrap(),
        UFuncArray::new(vec![0, 1], vec![], DType::Complex128).unwrap(),
    ] {
        assert!(malformed.vdot(&valid).is_err());
        assert!(valid.vdot(&malformed).is_err());
        for axis in [None, Some(0), Some(-1)] {
            assert!(malformed.vecdot(&valid, axis).is_err());
            assert!(valid.vecdot(&malformed, axis).is_err());
        }
    }
}
