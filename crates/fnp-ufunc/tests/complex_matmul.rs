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
    fn case(&mut self, op: &str, lhs: &UFuncArray, rhs: &UFuncArray, axis: Option<isize>) {
        let actual = match op {
            "matmul" => lhs.matmul(rhs),
            "matvec" => UFuncArray::matvec(lhs, rhs),
            "vecmat" => UFuncArray::vecmat(lhs, rhs),
            "vdot" => lhs.vdot(rhs),
            "vecdot" => lhs.vecdot(rhs, axis),
            "dot" => lhs.dot(rhs),
            "inner" => lhs.inner(rhs),
            "outer" => lhs.outer(rhs),
            "kron" => lhs.kron(rhs),
            "tensordot" => lhs.tensordot(rhs, axis.unwrap_or(2) as usize),
            _ => panic!("unsupported contraction"),
        };
        self.cases.push(json!({
            "op": op, "lhs": oracle_input(lhs), "rhs": oracle_input(rhs), "axis": axis
        }));
        self.actual.push(actual);
    }

    fn axes_case(&mut self, lhs: &UFuncArray, rhs: &UFuncArray, a: &[isize], b: &[isize]) {
        self.cases.push(json!({
            "op": "tensordot_axes", "lhs": oracle_input(lhs), "rhs": oracle_input(rhs),
            "axis": null, "axes": [a, b]
        }));
        self.actual.push(lhs.tensordot_axes(rhs, a, b));
    }

    fn chain_case(&mut self, arrays: &[&UFuncArray]) {
        self.cases.push(json!({
            "op": "multi_dot", "arrays": arrays.iter().map(|array| oracle_input(array)).collect::<Vec<_>>()
        }));
        self.actual.push(UFuncArray::multi_dot(arrays));
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

def operate(op, left, right, axis, axes=None):
    if op == 'vecdot':
        return np.vecdot(left, right, axis=-1 if axis is None else axis)
    if op == 'tensordot':
        return np.tensordot(left, right, axes=2 if axis is None else axis)
    if op == 'tensordot_axes':
        return np.tensordot(left, right, axes=axes)
    return getattr(np, op)(left, right)

def chain_bounds(arrays):
    matrices = list(arrays)
    if matrices[0].ndim == 1:
        matrices[0] = matrices[0][None, :]
    if matrices[-1].ndim == 1:
        matrices[-1] = matrices[-1][:, None]
    order = np.linalg._linalg._multi_dot_matrix_chain_order(matrices)
    def visit(i, j):
        if i == j:
            value = matrices[i]
            return (np.abs(value.real.astype(np.float64)),
                    np.abs(value.imag.astype(np.float64)), value.dtype, 0, 0.0)
        k = int(order[i, j])
        ar, ai, adtype, awork, aeps = visit(i, k)
        br, bi, bdtype, bwork, beps = visit(k + 1, j)
        dtype = np.result_type(adtype, bdtype)
        epsilon = np.finfo(dtype).eps if dtype.kind in 'fc' else 0.0
        return (ar @ br + ai @ bi, ar @ bi + ai @ br, dtype,
                awork + bwork + ar.shape[-1], max(aeps, beps, epsilon))
    return visit(0, len(matrices) - 1)

rows = []
for case in json.load(sys.stdin):
    try:
        if case['op'] == 'multi_dot':
            arrays = [load(item) for item in case['arrays']]
            result = np.asarray(np.linalg.multi_dot(arrays))
            if len(arrays) > 2:
                real_bound, imag_bound, _, contracted, epsilon = chain_bounds(arrays)
                parts = np.empty(result.size * 2, dtype=np.float64)
                parts[0::2] = result.real.ravel()
                parts[1::2] = result.imag.ravel()
                bounds = np.empty(result.size * 2, dtype=np.float64)
                bounds[0::2] = real_bound.ravel()
                bounds[1::2] = imag_bound.ravel()
                rows.append(dict(shape=list(result.shape), dtype=str(result.dtype),
                                 bits=parts.view(np.uint64).tolist(), bounds=bounds.tolist(),
                                 contracted=int(contracted), epsilon=float(epsilon),
                                 integers=[str(int(x)) for x in result.ravel()] if result.dtype.kind in 'biu' else None))
                continue
            left, right = arrays
            op, axis, axes = 'dot', None, None
        else:
            left, right = load(case['lhs']), load(case['rhs'])
            op, axis, axes = case['op'], case['axis'], case.get('axes')
        result = np.asarray(operate(op, left, right, axis, axes))
        # Promote each input before deriving the rounding bound, matching the
        # ufunc's typed conversion rather than its original working cells.
        left = left.astype(result.dtype)
        right = right.astype(result.dtype)
        ar, ai = np.abs(left.real.astype(np.float64)), np.abs(left.imag.astype(np.float64))
        br, bi = np.abs(right.real.astype(np.float64)), np.abs(right.imag.astype(np.float64))
        real_bound = operate(op, ar, br, axis, axes) + operate(op, ai, bi, axis, axes)
        imag_bound = operate(op, ar, bi, axis, axes) + operate(op, ai, br, axis, axes)
        if op == 'vdot':
            contracted = left.size
        elif op in ('outer', 'kron') or (op in ('dot', 'inner') and (left.ndim == 0 or right.ndim == 0)):
            contracted = 1
        elif op == 'tensordot':
            n = 2 if axis is None else axis
            contracted = int(np.prod(left.shape[left.ndim-n:], dtype=np.int64))
        elif op == 'tensordot_axes':
            contracted = int(np.prod([left.shape[i] for i in axes[0]], dtype=np.int64))
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
                         bounds=bounds.tolist(), contracted=int(contracted),
                         integers=[str(int(x)) for x in result.ravel()] if result.dtype.kind in 'biu' else None))
    except (ValueError, TypeError, IndexError, OverflowError, np.linalg.LinAlgError) as error:
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
                assert!(
                    actual.is_err(),
                    "{case}: expected {expected}, got {actual:?}"
                );
                continue;
            }
            let actual = actual.unwrap_or_else(|error| panic!("{case}: {error}"));
            let mut shape = actual.shape().to_vec();
            if actual.dtype().is_complex() {
                assert_eq!(shape.pop(), Some(2), "{case}: component axis");
            }
            assert_eq!(json!(shape), expected["shape"], "{case}: logical shape");
            assert_eq!(
                dtype_name(actual.dtype()),
                expected["dtype"],
                "{case}: dtype"
            );
            if let Some(integers) = expected["integers"].as_array() {
                let storage = actual.to_storage().unwrap();
                let exact = (0..storage.len())
                    .map(|index| storage.get_i128(index).unwrap().to_string())
                    .collect::<Vec<_>>();
                assert_eq!(
                    json!(exact),
                    json!(integers),
                    "{case}: exact integer values"
                );
                continue;
            }
            let components = if actual.dtype().is_complex() {
                actual.values().to_vec()
            } else {
                actual
                    .values()
                    .iter()
                    .flat_map(|&value| [value, 0.0])
                    .collect()
            };
            let bits = expected["bits"].as_array().unwrap();
            let bounds = expected["bounds"].as_array().unwrap();
            assert_eq!(components.len(), bits.len(), "{case}: component count");
            let contracted = expected["contracted"].as_u64().unwrap() as f64;
            let (epsilon, smallest) = if matches!(actual.dtype(), DType::Complex64 | DType::F32) {
                (f64::from(f32::EPSILON), f64::from(f32::from_bits(1)))
            } else if actual.dtype() == DType::F16 {
                (2.0_f64.powi(-10), 2.0_f64.powi(-24))
            } else {
                (f64::EPSILON, f64::from_bits(1))
            };
            // A mixed chain can round an early complex64/f32 product before a
            // later operand widens the final result to complex128/f64.
            let epsilon = expected["epsilon"].as_f64().unwrap_or(epsilon).max(epsilon);
            for (index, ((&actual, expected), bound)) in
                components.iter().zip(bits).zip(bounds).enumerate()
            {
                let expected = f64::from_bits(expected.as_u64().unwrap());
                if expected.is_nan() {
                    assert!(actual.is_nan(), "{case}: component {index} expected NaN");
                } else if expected.is_infinite() {
                    assert_eq!(actual, expected, "{case}: component {index} infinity");
                } else {
                    assert!(
                        actual.is_finite(),
                        "{case}: component {index} must be finite"
                    );
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
        (&[3], &[3]),
        (&[3], &[3, 4]),
        (&[2, 3], &[3]),
        (&[2, 3], &[3, 4]),
        (&[1, 3], &[3, 1]),
        (&[2, 1, 3, 4], &[1, 5, 4, 2]),
        (&[4], &[2, 3, 4, 5]),
        (&[2, 3, 4, 5], &[5]),
        (&[2, 1, 3, 4], &[4, 2]),
        (&[3, 4], &[2, 1, 4, 2]),
        (&[0], &[0]),
        (&[2, 0], &[0, 3]),
        (&[0, 3], &[3, 4]),
        (&[2, 3], &[3, 0]),
        (&[0, 2, 3], &[1, 3, 4]),
        (&[2, 1, 3, 0], &[1, 5, 0, 2]),
        (&[], &[3]),
        (&[3], &[]),
        (&[], &[]),
        (&[2, 3], &[4, 2]),
        (&[2, 3, 4], &[3, 4, 2]),
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
fn complex_dot_and_inner_contract_logical_axes_without_batch_broadcasting() {
    let dot_shapes: &[(&[usize], &[usize])] = &[
        (&[], &[]),
        (&[], &[2, 3]),
        (&[2, 3], &[]),
        (&[3], &[3]),
        (&[3], &[3, 4]),
        (&[2, 3], &[3]),
        (&[2, 3, 4], &[5, 4, 6]),
        (&[2, 3, 4], &[4, 6]),
        (&[4], &[2, 3, 4, 5]),
        (&[2, 1, 3], &[5, 3, 2]),
        (&[0], &[0]),
        (&[2, 0], &[0, 3]),
        (&[0, 3], &[3, 4]),
        (&[2, 3], &[3, 0]),
        (&[2, 0, 3], &[4, 3, 5]),
        (&[2, 3], &[4, 2]),
    ];
    let inner_shapes: &[(&[usize], &[usize])] = &[
        (&[], &[]),
        (&[], &[2, 3]),
        (&[2, 3], &[]),
        (&[3], &[3]),
        (&[3], &[4, 3]),
        (&[2, 3], &[3]),
        (&[2, 3, 4], &[5, 6, 4]),
        (&[2, 3, 4], &[6, 4]),
        (&[0], &[0]),
        (&[2, 0], &[3, 0]),
        (&[0, 3], &[4, 3]),
        (&[2, 3], &[0, 3]),
        (&[2, 3], &[2, 4]),
    ];
    let mut batch = OracleBatch::default();
    for complex_dtype in [DType::Complex64, DType::Complex128] {
        for other_dtype in [DType::Complex64, DType::Complex128, DType::F32, DType::I16] {
            for (op, shapes) in [("dot", dot_shapes), ("inner", inner_shapes)] {
                for &(a, b) in shapes {
                    batch.case(
                        op,
                        &source(a, complex_dtype, 1),
                        &source(b, other_dtype, 4),
                        None,
                    );
                    batch.case(
                        op,
                        &source(a, other_dtype, 7),
                        &source(b, complex_dtype, 2),
                        None,
                    );
                }
            }
        }
    }
    batch.finish();
}

#[test]
fn complex_outer_and_kron_preserve_complete_elements_and_numeric_promotion() {
    let shapes: &[(&[usize], &[usize])] = &[
        (&[], &[]),
        (&[], &[2, 3]),
        (&[2, 3], &[]),
        (&[3], &[2]),
        (&[2, 3], &[3, 2]),
        (&[2, 1, 3], &[3, 2]),
        (&[1, 2], &[2, 1, 3]),
        (&[2, 1, 2, 1], &[3, 1, 2]),
        (&[0], &[3]),
        (&[2, 0], &[0, 3]),
        (&[2, 3], &[0]),
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
            for &(a, b) in shapes {
                for op in ["outer", "kron"] {
                    batch.case(
                        op,
                        &source(a, complex_dtype, 0),
                        &source(b, other_dtype, 4),
                        None,
                    );
                    batch.case(
                        op,
                        &source(a, other_dtype, 7),
                        &source(b, complex_dtype, 1),
                        None,
                    );
                }
            }
        }
    }
    batch.finish();
}

#[test]
fn complex_tensordot_counts_and_explicit_axis_lists_match_numpy() {
    let cases: &[(&[usize], &[usize], usize)] = &[
        (&[2, 3, 4], &[3, 4, 5], 2),
        (&[2, 3], &[3, 4], 1),
        (&[2, 3, 4], &[2, 3, 4], 3),
        (&[2, 3], &[4, 5], 0),
        (&[], &[2, 3], 0),
        (&[2, 3], &[], 0),
        (&[], &[], 0),
        (&[2, 0], &[0, 3], 1),
        (&[0, 2, 3], &[2, 3, 4], 2),
        (&[2, 3], &[3, 0], 1),
        (&[0], &[0], 1),
        (&[2, 3], &[2, 3], 1),
        (&[2, 3], &[2, 3], 3),
        (&[], &[], 1),
    ];
    type AxisCase<'a> = (&'a [usize], &'a [usize], &'a [isize], &'a [isize]);
    let axis_cases: &[AxisCase<'_>] = &[
        (&[2, 3, 4], &[4, 5, 2], &[2, 0], &[0, 2]),
        (&[2, 3, 4], &[4, 5, 2], &[-1, -3], &[-3, -1]),
        (&[2, 3, 4], &[5, 4, 3], &[1, 2], &[2, 1]),
        (&[2, 3, 4], &[5, 4, 3], &[2, 1], &[1, 2]),
        (&[2, 3, 4], &[4, 3, 2], &[2, 1, 0], &[0, 1, 2]),
        (&[2, 3], &[3, 2], &[], &[]),
        (&[], &[3], &[], &[]),
        (&[2, 0, 3], &[3, 0, 4], &[1, 2], &[1, 0]),
        (&[0, 2, 3], &[4, 3], &[-1], &[-1]),
        (&[2, 3], &[2, 3], &[0, 0], &[0, 0]),
        (&[2, 3], &[2, 3], &[0, -2], &[0, -2]),
        (&[2, 3], &[2, 3], &[0], &[0, 1]),
        (&[2, 3], &[2, 3], &[2], &[1]),
        (&[2, 3], &[2, 3], &[-3], &[1]),
        (&[2, 3], &[2, 3], &[0], &[1]),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for other_dtype in [DType::Complex64, DType::Complex128, DType::F16, DType::I64] {
            for &(a, b, axes) in cases {
                batch.case(
                    "tensordot",
                    &source(a, dtype, 0),
                    &source(b, other_dtype, 6),
                    Some(axes as isize),
                );
            }
            for &(a, b, axes_a, axes_b) in axis_cases {
                batch.axes_case(
                    &source(a, dtype, 2),
                    &source(b, other_dtype, 5),
                    axes_a,
                    axes_b,
                );
            }
        }
    }
    batch.finish();
}

#[test]
fn complex_tensor_products_preserve_exact_integer_conversion_and_source_precision() {
    let mut batch = OracleBatch::default();
    for storage in [
        ArrayStorage::I64(vec![
            i64::MAX,
            i64::MIN,
            (1_i64 << 53) + 1,
            -(1_i64 << 53) - 1,
        ]),
        ArrayStorage::U64(vec![
            u64::MAX,
            u64::MAX - 1,
            (1_u64 << 63) + 1,
            (1_u64 << 53) + 1,
        ]),
    ] {
        let matrix = UFuncArray::from_storage(vec![2, 2], storage).unwrap();
        for dtype in [DType::Complex64, DType::Complex128] {
            let complex = source(&[2, 2], dtype, 0);
            for op in ["dot", "inner", "outer", "kron", "tensordot"] {
                batch.case(op, &matrix, &complex, Some(1));
                batch.case(op, &complex, &matrix, Some(1));
            }
            batch.axes_case(&matrix, &complex, &[0], &[1]);
            batch.axes_case(&complex, &matrix, &[1], &[0]);
        }
    }
    for dtype in [DType::Complex64, DType::Complex128] {
        let scalar = source(&[], dtype, 0);
        for other in [DType::F16, DType::F32, DType::Complex64] {
            let input = source(&[2, 3], other, 0);
            batch.case("dot", &scalar, &input, None);
            batch.case("inner", &input, &scalar, None);
        }
    }
    batch.finish();
}

#[test]
fn complex_tensor_products_do_not_conjugate_inputs() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, -4.0], dtype).unwrap();
        let b = UFuncArray::new(vec![2, 2], vec![5.0, -6.0, 7.0, 8.0], dtype).unwrap();
        for result in [
            a.dot(&b),
            a.inner(&b),
            a.tensordot(&b, 1),
            a.tensordot_axes(&b, &[0], &[-1]),
        ] {
            let result = result.unwrap();
            assert_eq!(result.shape(), &[2]);
            assert_eq!(result.values(), &[70.0, 0.0]);
        }
        let outer = a.outer(&b).unwrap();
        assert_eq!(outer.shape(), &[2, 2, 2]);
        assert_eq!(
            outer.values(),
            &[17.0, 4.0, -9.0, 22.0, -9.0, -38.0, 53.0, -4.0]
        );
        let kron = a.kron(&b).unwrap();
        assert_eq!(kron.shape(), &[4, 2]);
        assert_eq!(kron.values(), outer.values());
    }
}

#[test]
fn explicit_tensordot_axes_preserve_real_dtypes_and_exact_integers() {
    let mut batch = OracleBatch::default();
    for dtype in [
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
    ] {
        batch.axes_case(
            &source(&[2, 3, 4], dtype, 1),
            &source(&[4, 5, 2], dtype, 4),
            &[2, 0],
            &[0, 2],
        );
        batch.axes_case(
            &source(&[2, 0], dtype, 0),
            &source(&[0, 3], dtype, 0),
            &[1],
            &[0],
        );
        batch.axes_case(&source(&[], dtype, 2), &source(&[3], dtype, 4), &[], &[]);
    }
    for storage in [
        ArrayStorage::I64(vec![
            i64::MAX,
            i64::MIN,
            (1_i64 << 53) + 1,
            -(1_i64 << 53) - 1,
        ]),
        ArrayStorage::U64(vec![
            u64::MAX,
            u64::MAX - 1,
            (1_u64 << 63) + 1,
            (1_u64 << 53) + 1,
        ]),
    ] {
        let array = UFuncArray::from_storage(vec![2, 2], storage).unwrap();
        let other = source(&[2, 3], array.dtype(), 3);
        batch.axes_case(&array, &other, &[0], &[0]);
        batch.axes_case(&array, &array, &[1, 0], &[0, 1]);
    }
    let narrow = UFuncArray::new(vec![1], vec![1.0 + 2.0_f64.powi(-24)], DType::F32).unwrap();
    let wide = UFuncArray::new(vec![1], vec![1.0], DType::F64).unwrap();
    let output = narrow.tensordot_axes(&wide, &[0], &[0]).unwrap();
    assert_eq!(output.dtype(), DType::F64);
    assert_eq!(output.values(), &[1.0]);
    batch.axes_case(&narrow, &wide, &[0], &[0]);
    batch.axes_case(&wide, &narrow, &[0], &[0]);
    let overflowing =
        UFuncArray::from_storage(vec![2], ArrayStorage::F32(vec![f32::MAX; 2])).unwrap();
    let signs = UFuncArray::from_storage(vec![2], ArrayStorage::F32(vec![2.0, -2.0])).unwrap();
    assert!(
        overflowing
            .tensordot_axes(&signs, &[0], &[0])
            .unwrap()
            .values()[0]
            .is_nan()
    );
    batch.axes_case(&overflowing, &signs, &[0], &[0]);
    let half = UFuncArray::new(vec![3], vec![4096.0, 1.0, -4096.0], DType::F16).unwrap();
    let factors = UFuncArray::new(vec![3], vec![4096.0, 1.0, 4096.0], DType::F16).unwrap();
    assert_eq!(
        half.tensordot_axes(&factors, &[0], &[0]).unwrap().values(),
        &[0.0]
    );
    batch.axes_case(&half, &factors, &[0], &[0]);
    batch.finish();
}

#[test]
fn complex_tensor_products_reject_invalid_layouts_axes_and_output_capacities() {
    let valid = source(&[2, 2], DType::Complex128, 0);
    for malformed in [
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
        UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::Complex64).unwrap(),
        UFuncArray::new(vec![0, 1], vec![], DType::Complex128).unwrap(),
    ] {
        for (a, b) in [(&malformed, &valid), (&valid, &malformed)] {
            for result in [
                a.dot(b),
                a.inner(b),
                a.outer(b),
                a.kron(b),
                a.tensordot(b, 1),
                a.tensordot_axes(b, &[0], &[0]),
            ] {
                assert!(
                    result.is_err(),
                    "malformed complex storage must be rejected"
                );
            }
        }
    }
    for axis in [isize::MIN, isize::MAX] {
        assert!(valid.tensordot_axes(&valid, &[axis], &[0]).is_err());
        assert!(valid.tensordot_axes(&valid, &[0], &[axis]).is_err());
    }
    let max_count = isize::MAX as usize / std::mem::size_of::<[f64; 2]>();
    let huge = UFuncArray::new(vec![max_count + 1, 0, 2], vec![], DType::Complex128).unwrap();
    let rhs = source(&[0, 1], DType::Complex128, 0);
    for result in [
        huge.dot(&rhs),
        huge.tensordot(&rhs, 1),
        huge.tensordot_axes(&rhs, &[1], &[0]),
    ] {
        assert!(matches!(
            result,
            Err(UFuncError::Shape(ShapeError::Overflow))
        ));
    }
    let huge_empty = UFuncArray::new(vec![usize::MAX, 0, 2], vec![], DType::Complex128).unwrap();
    let empty = source(&[0, 0], DType::Complex128, 0);
    assert_eq!(
        huge_empty.tensordot(&empty, 1).unwrap().shape(),
        &[usize::MAX, 0, 2]
    );
    assert!(huge_empty.outer(&valid).unwrap().values().is_empty());
    let scalar = source(&[], DType::Complex64, 0);
    assert_eq!(
        huge_empty.kron(&scalar).unwrap().shape(),
        &[usize::MAX, 0, 2]
    );
    assert!(matches!(
        huge_empty.kron(&valid),
        Err(UFuncError::Shape(ShapeError::Overflow))
    ));
    let opaque = UFuncArray::new(vec![2, 2], vec![1.0; 4], DType::Str).unwrap();
    assert!(valid.outer(&opaque).is_err());
    assert!(valid.dot(&opaque).is_err());
    assert!(valid.kron(&opaque).is_err());
}

#[test]
fn complex_matmul_mixed_numeric_types_match_numpy() {
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
        (&[2, 3], &[3]),
        (&[2, 3], &[1, 3]),
        (&[1, 2, 3], &[3]),
        (&[2, 1, 4, 3], &[1, 5, 3]),
        (&[4, 3], &[2, 1, 3]),
        (&[2, 1, 4, 3], &[5, 3]),
        (&[0, 2, 3], &[1, 3]),
        (&[2, 0], &[0]),
        (&[3], &[3]),
        (&[], &[3]),
        (&[2, 3], &[]),
        (&[2, 3], &[4]),
        (&[2, 4, 3], &[3, 3]),
    ];
    let vecmat_shapes: &[(&[usize], &[usize])] = &[
        (&[3], &[3, 2]),
        (&[1, 3], &[3, 2]),
        (&[3], &[1, 3, 2]),
        (&[2, 1, 3], &[1, 5, 3, 4]),
        (&[2, 1, 3], &[3, 4]),
        (&[5, 3], &[2, 1, 3, 4]),
        (&[1, 3], &[0, 3, 2]),
        (&[0], &[0, 2]),
        (&[3], &[3]),
        (&[], &[3, 2]),
        (&[3], &[]),
        (&[4], &[3, 2]),
        (&[2, 3], &[3, 3, 4]),
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
        ArrayStorage::I64(vec![
            i64::MAX,
            i64::MIN,
            (1_i64 << 53) + 1,
            -(1_i64 << 53) - 1,
        ]),
        ArrayStorage::U64(vec![
            u64::MAX,
            u64::MAX - 1,
            (1_u64 << 63) + 1,
            (1_u64 << 53) + 1,
        ]),
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
        assert_eq!(
            output.values(),
            &[expected, 0.0, expected, 0.0, expected, 0.0, expected, 0.0]
        );
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
            malformed.matmul(&valid),
            valid.matmul(&malformed),
            UFuncArray::matvec(&malformed, &valid),
            UFuncArray::matvec(&valid, &malformed),
            UFuncArray::vecmat(&malformed, &valid),
            UFuncArray::vecmat(&valid, &malformed),
        ] {
            assert!(result.is_err(), "malformed complex layout must be rejected");
        }
    }
    let huge_empty = UFuncArray::new(vec![usize::MAX, 0, 2], vec![], DType::Complex128).unwrap();
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
    assert!(matches!(
        lhs.matmul(&rhs),
        Err(UFuncError::Shape(ShapeError::Overflow))
    ));
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
    let left = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, -4.0], DType::Complex128).unwrap();
    let right = UFuncArray::new(vec![2, 2], vec![5.0, -6.0, 7.0, 8.0], DType::Complex128).unwrap();
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
fn complex_multi_dot_chains_and_vector_endpoints_match_numpy() {
    let layouts: &[&[&[usize]]] = &[
        &[&[2, 3], &[3, 4], &[4, 2]],
        &[&[2, 2], &[2, 2], &[2, 2]],
        &[&[2, 3], &[3, 1], &[1, 4], &[4, 2]],
        &[&[3, 1], &[1, 4], &[4, 2], &[2, 5], &[5, 1]],
        &[&[3], &[3, 4], &[4, 2]],
        &[&[2, 3], &[3, 4], &[4]],
        &[&[3], &[3, 4], &[4]],
        &[&[3], &[3, 2], &[2, 4], &[4]],
        &[&[2, 0], &[0, 3], &[3, 4]],
        &[&[0, 2], &[2, 3], &[3, 4]],
        &[&[2, 3], &[3, 0], &[0, 4]],
        &[&[2, 3], &[3, 4], &[4, 0]],
        &[&[2, 0], &[0, 3], &[3, 0], &[0, 4]],
        &[&[0], &[0, 3], &[3, 0], &[0]],
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shapes in layouts {
            let arrays: Vec<_> = shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| source(shape, dtype, index + 1))
                .collect();
            batch.chain_case(&arrays.iter().collect::<Vec<_>>());
        }
    }
    batch.finish();
}

#[test]
fn complex_multi_dot_preserves_pairwise_promotion_and_exact_integer_sources() {
    let mut batch = OracleBatch::default();
    for dtypes in [
        [DType::Complex64, DType::F32, DType::Complex64],
        [DType::F64, DType::Complex64, DType::Complex64],
        [DType::Complex64, DType::Complex128, DType::F32],
        [DType::F32, DType::Complex64, DType::Complex128],
        [DType::Complex64, DType::I16, DType::Complex128],
        [DType::Bool, DType::Complex64, DType::Complex64],
    ] {
        for shapes in [
            [vec![2, 3], vec![3, 4], vec![4, 2]],
            [vec![3], vec![3, 4], vec![4]],
        ] {
            let arrays: Vec<_> = shapes
                .iter()
                .zip(dtypes)
                .enumerate()
                .map(|(index, (shape, dtype))| source(shape, dtype, index + 2))
                .collect();
            batch.chain_case(&arrays.iter().collect::<Vec<_>>());
        }
    }
    for storage in [
        ArrayStorage::I64(vec![i64::MAX, i64::MIN, (1_i64 << 53) + 1, -1, 0, 1]),
        ArrayStorage::U64(vec![u64::MAX, u64::MAX - 1, (1_u64 << 63) + 1, 0, 1, 2]),
    ] {
        let integers = UFuncArray::from_storage(vec![2, 3], storage).unwrap();
        for dtype in [DType::Complex64, DType::Complex128] {
            let left = source(&[2, 2], dtype, 1);
            let right = source(&[3, 2], dtype, 3);
            batch.chain_case(&[&left, &integers, &right]);
            batch.chain_case(&[&integers, &right, &left]);
        }
    }
    batch.finish();
}

#[test]
fn complex_multi_dot_does_not_conjugate_vector_endpoints() {
    let left =
        UFuncArray::new(vec![1, 2, 2], vec![1.0, 2.0, 3.0, -1.0], DType::Complex128)
            .unwrap();
    let middle = UFuncArray::new(
        vec![2, 2, 2],
        vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        DType::Complex128,
    )
    .unwrap();
    let right =
        UFuncArray::new(vec![2, 1, 2], vec![1.0, -1.0, 2.0, 1.0], DType::Complex128)
            .unwrap();
    let matrix = UFuncArray::multi_dot(&[&left, &middle, &right]).unwrap();
    assert_eq!(matrix.shape(), &[1, 1, 2]);
    assert_eq!(matrix.values(), &[5.0, 9.0]);
    let row = left.reshape(&[2]).unwrap();
    let column = right.reshape(&[2]).unwrap();
    let scalar = UFuncArray::multi_dot(&[&row, &middle, &column]).unwrap();
    assert_eq!(scalar.shape(), &[2]);
    assert_eq!(scalar.values(), &[5.0, 9.0]);
}

#[test]
fn complex_multi_dot_rejects_invalid_chains_like_numpy() {
    let layouts: &[&[&[usize]]] = &[
        &[],
        &[&[2, 2]],
        &[&[], &[2, 2], &[2, 2]],
        &[&[2, 2], &[2, 2], &[]],
        &[&[2, 2], &[2], &[2, 2]],
        &[&[2, 2], &[1, 2, 2], &[2, 2]],
        &[&[2, 3], &[4, 2], &[2, 2]],
        &[&[2, 3], &[3, 2], &[3, 2]],
        &[&[2, 3], &[4, 0], &[0, 2]],
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shapes in layouts {
            let arrays: Vec<_> = shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| source(shape, dtype, index))
                .collect();
            batch.chain_case(&arrays.iter().collect::<Vec<_>>());
        }
    }
    batch.finish();
}

#[test]
fn complex_multi_dot_validates_layouts_and_checked_empty_output_capacity() {
    let valid = source(&[2, 2], DType::Complex128, 0);
    for malformed in [
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
        UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::Complex64).unwrap(),
        UFuncArray::new(vec![0, 1], vec![], DType::Complex128).unwrap(),
    ] {
        assert!(UFuncArray::multi_dot(&[&malformed, &valid, &valid]).is_err());
        assert!(UFuncArray::multi_dot(&[&valid, &malformed, &valid]).is_err());
        assert!(UFuncArray::multi_dot(&[&valid, &valid, &malformed]).is_err());
    }
    let tall = UFuncArray::new(vec![usize::MAX, 0, 2], vec![], DType::Complex128).unwrap();
    let wide = UFuncArray::new(vec![0, usize::MAX, 2], vec![], DType::Complex128).unwrap();
    let empty = UFuncArray::multi_dot(&[&tall, &wide, &tall]).unwrap();
    assert_eq!(empty.shape(), &[usize::MAX, 0, 2]);
    assert!(empty.values().is_empty());
    assert!(matches!(
        UFuncArray::multi_dot(&[&tall, &wide, &tall, &wide]),
        Err(UFuncError::Shape(ShapeError::Overflow))
    ));
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

#[test]
fn complex_multi_dot_expanded_chains_and_vector_endpoints_match_numpy() {
    let dimensions: &[&[usize]] = &[
        &[2, 3, 4, 5],
        &[4, 2, 5, 3],
        &[2, 2, 2, 2],
        &[2, 3, 4, 2, 5],
        &[4, 1, 5, 2, 3],
        &[1, 1, 1, 1, 1],
        &[3, 2, 5, 1, 4, 2],
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &dims in dimensions {
            for first_vector in [false, true] {
                for last_vector in [false, true] {
                    let arrays = dims
                        .windows(2)
                        .enumerate()
                        .map(|(index, pair)| {
                            let shape = if first_vector && index == 0 {
                                vec![pair[1]]
                            } else if last_vector && index == dims.len() - 2 {
                                vec![pair[0]]
                            } else {
                                pair.to_vec()
                            };
                            source(&shape, dtype, index * 3 + 1)
                        })
                        .collect::<Vec<_>>();
                    batch.chain_case(&arrays.iter().collect::<Vec<_>>());
                }
            }
        }
    }
    batch.finish();
}

#[test]
fn complex_multi_dot_mixed_dtype_chains_match_numpy() {
    let mut batch = OracleBatch::default();
    let dimensions = [2, 3, 2, 4, 2];
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
            for complex_position in 0..4 {
                let arrays = dimensions
                    .windows(2)
                    .enumerate()
                    .map(|(index, shape)| {
                        source(
                            shape,
                            if index == complex_position {
                                complex_dtype
                            } else {
                                other_dtype
                            },
                            index * 2 + 1,
                        )
                    })
                    .collect::<Vec<_>>();
                batch.chain_case(&arrays.iter().collect::<Vec<_>>());
            }
        }
        for storage in [
            ArrayStorage::I64(vec![i64::MAX, i64::MIN, (1_i64 << 53) + 1, -1]),
            ArrayStorage::U64(vec![u64::MAX, (1_u64 << 63) + 1, (1_u64 << 53) + 1, 1]),
        ] {
            let integers = UFuncArray::from_storage(vec![2, 2], storage).unwrap();
            let complex = source(&[2, 2], complex_dtype, 2);
            let real = source(&[2, 2], DType::F64, 4);
            batch.chain_case(&[&integers, &complex, &real]);
            batch.chain_case(&[&complex, &integers, &real]);
            batch.chain_case(&[&complex, &real, &integers]);
        }
    }
    batch.finish();
}

#[test]
fn complex_multi_dot_preserves_selected_product_dtype_and_source_precision() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let complex_one =
            UFuncArray::from_storage_with_dtype(vec![1, 1], ArrayStorage::F64(vec![1.0]), dtype)
                .unwrap();
        let integer_large =
            UFuncArray::from_storage(vec![1, 1], ArrayStorage::I8(vec![127])).unwrap();
        let integer_two = UFuncArray::from_storage(vec![1, 1], ArrayStorage::I8(vec![2])).unwrap();
        let integer_one = UFuncArray::from_storage(vec![1, 1], ArrayStorage::I8(vec![1])).unwrap();
        // Equal costs choose the first split, A @ (B @ C). Its intermediate
        // int8 product must wrap before promotion to the complex result dtype.
        for chain in [
            vec![&complex_one, &integer_large, &integer_two],
            vec![&complex_one, &integer_large, &integer_two, &integer_one],
        ] {
            let output = UFuncArray::multi_dot(&chain).unwrap();
            assert_eq!(output.dtype(), dtype);
            assert_eq!(output.shape(), &[1, 1, 2]);
            assert_eq!(output.values(), &[-2.0, 0.0]);
            batch.chain_case(&chain);
        }
        let output = UFuncArray::multi_dot(&[&integer_large, &integer_two, &complex_one]).unwrap();
        assert_eq!(output.values(), &[254.0, 0.0]);
        batch.chain_case(&[&integer_large, &integer_two, &complex_one]);

        // These unequal shapes select (A @ B) @ C; a whole-chain complex cast
        // would incorrectly replace the int8 result -6 with the value 762.
        let a = UFuncArray::from_storage(vec![2, 3], ArrayStorage::I8(vec![127; 6])).unwrap();
        let b = UFuncArray::from_storage(vec![3, 1], ArrayStorage::I8(vec![2; 3])).unwrap();
        let c =
            UFuncArray::from_storage_with_dtype(vec![1, 2], ArrayStorage::F64(vec![1.0; 2]), dtype)
                .unwrap();
        let output = UFuncArray::multi_dot(&[&a, &b, &c]).unwrap();
        assert_eq!(output.shape(), &[2, 2, 2]);
        assert_eq!(
            output.values(),
            &[-6.0, 0.0, -6.0, 0.0, -6.0, 0.0, -6.0, 0.0]
        );
        batch.chain_case(&[&a, &b, &c]);
    }
    let raw_single = UFuncArray::new(
        vec![1, 1, 2],
        vec![16_777_217.0, 1.0 + 2.0_f64.powi(-24)],
        DType::Complex64,
    )
    .unwrap();
    let wide_one =
        UFuncArray::from_storage(vec![1, 1], ArrayStorage::Complex128(vec![(1.0, 0.0)])).unwrap();
    let output = UFuncArray::multi_dot(&[&raw_single, &wide_one, &wide_one]).unwrap();
    assert_eq!(output.dtype(), DType::Complex128);
    assert_eq!(output.values(), &[16_777_216.0, 1.0]);
    batch.chain_case(&[&raw_single, &wide_one, &wide_one]);
    batch.finish();
}

#[test]
fn complex_multi_dot_rounds_real_subtrees_before_complex_promotion() {
    let complex_one =
        UFuncArray::from_storage(vec![1, 1], ArrayStorage::Complex128(vec![(1.0, 0.0)])).unwrap();
    let mut batch = OracleBatch::default();
    for (dtype, values, expected) in [
        (
            DType::F16,
            [1.9423828125, 1.666015625, 1.1337890625],
            3.66796875,
        ),
        (
            DType::F32,
            [1.4978675842285156, 1.4936197996139526, 1.5002261400222778],
            3.3563730716705322,
        ),
    ] {
        let arrays = values
            .into_iter()
            .map(|value| UFuncArray::new(vec![1, 1], vec![value], dtype).unwrap())
            .collect::<Vec<_>>();
        let chain = [&complex_one, &arrays[0], &arrays[1], &arrays[2]];
        let output = UFuncArray::multi_dot(&chain).unwrap();
        assert_eq!(output.dtype(), DType::Complex128);
        assert_eq!(output.values(), &[expected, 0.0]);
        batch.chain_case(&chain);
    }
    let wide_one = UFuncArray::new(vec![1, 1], vec![1.0], DType::F64).unwrap();
    for (dtype, raw_value, expected) in [
        (DType::F16, 1.0 + 2.0_f64.powi(-11), 1.0),
        (DType::F32, 16_777_217.0, 16_777_216.0),
    ] {
        let raw = UFuncArray::new(vec![1, 1], vec![raw_value], dtype).unwrap();
        let chain = [&complex_one, &raw, &wide_one, &wide_one];
        let output = UFuncArray::multi_dot(&chain).unwrap();
        assert_eq!(output.values(), &[expected, 0.0]);
        batch.chain_case(&chain);
    }
    batch.finish();
}

#[test]
fn complex_multi_dot_empty_chains_and_invalid_logical_shapes_match_numpy() {
    let mut batch = OracleBatch::default();
    let empty_dimensions: &[&[usize]] = &[
        &[2, 0, 3, 4],
        &[0, 3, 4, 0],
        &[0, 0, 0, 0],
        &[2, 3, 0, 4, 5],
        &[2, 0, 0, 3, 4],
    ];
    let invalid_shapes: &[&[&[usize]]] = &[
        &[&[], &[1, 1], &[1, 1]],
        &[&[1, 1], &[1], &[1, 1]],
        &[&[1, 1], &[1, 1], &[]],
        &[&[1, 1, 1], &[1, 1], &[1, 1]],
        &[&[1, 1], &[1, 1, 1], &[1, 1]],
        &[&[1, 1], &[1, 1], &[1, 1, 1]],
        &[&[2, 3], &[4, 5], &[5, 2]],
        &[&[2, 3], &[3, 4], &[5, 2]],
        &[&[3], &[4, 2], &[2]],
        &[&[2, 3], &[3, 4], &[5]],
        &[&[0, 2], &[3, 0], &[0, 0]],
    ];
    batch.chain_case(&[]);
    for dtype in [DType::Complex64, DType::Complex128] {
        for &dims in empty_dimensions {
            for first_vector in [false, true] {
                for last_vector in [false, true] {
                    let arrays = dims
                        .windows(2)
                        .enumerate()
                        .map(|(index, pair)| {
                            let shape = if first_vector && index == 0 {
                                vec![pair[1]]
                            } else if last_vector && index == dims.len() - 2 {
                                vec![pair[0]]
                            } else {
                                pair.to_vec()
                            };
                            source(&shape, dtype, index + 1)
                        })
                        .collect::<Vec<_>>();
                    batch.chain_case(&arrays.iter().collect::<Vec<_>>());
                }
            }
        }
        for &shapes in invalid_shapes {
            let arrays = shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| source(shape, dtype, index + 1))
                .collect::<Vec<_>>();
            batch.chain_case(&arrays.iter().collect::<Vec<_>>());
        }
        let scalar = source(&[], dtype, 0);
        let matrix = source(&[2, 3], dtype, 1);
        let tensor = source(&[2, 3, 4], dtype, 2);
        let vector = source(&[4], dtype, 3);
        batch.chain_case(&[&matrix]);
        // NumPy keeps ordinary dot semantics for exactly two operands,
        // including scalars and higher-rank operands rejected in longer chains.
        batch.chain_case(&[&scalar, &matrix]);
        batch.chain_case(&[&tensor, &vector]);
    }
    batch.finish();
}

#[test]
fn complex_multi_dot_rejects_malformed_storage_and_impossible_intermediates() {
    let valid = source(&[2, 2], DType::Complex128, 1);
    for malformed in [
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
        UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::Complex64).unwrap(),
        UFuncArray::new(vec![0, 1], vec![], DType::Complex128).unwrap(),
    ] {
        for chain in [
            [&malformed, &valid, &valid],
            [&valid, &malformed, &valid],
            [&valid, &valid, &malformed],
        ] {
            assert!(UFuncArray::multi_dot(&chain).is_err());
        }
    }
    let huge_empty =
        UFuncArray::from_storage(vec![usize::MAX, 0], ArrayStorage::Complex128(vec![])).unwrap();
    let empty = UFuncArray::from_storage(vec![0, 0], ArrayStorage::Complex128(vec![])).unwrap();
    let output = UFuncArray::multi_dot(&[&huge_empty, &empty, &empty, &empty]).unwrap();
    assert_eq!(output.shape(), &[usize::MAX, 0, 2]);
    assert!(output.values().is_empty());

    let columns = UFuncArray::from_storage(vec![0, 1], ArrayStorage::Complex128(vec![])).unwrap();
    let unit =
        UFuncArray::from_storage(vec![1, 1], ArrayStorage::Complex128(vec![(1.0, 0.0)])).unwrap();
    assert!(matches!(
        UFuncArray::multi_dot(&[&huge_empty, &columns, &unit]),
        Err(UFuncError::Shape(ShapeError::Overflow))
    ));
    let max_count = isize::MAX as usize / std::mem::size_of::<[f64; 2]>();
    let too_many =
        UFuncArray::from_storage(vec![max_count + 1, 0], ArrayStorage::Complex64(vec![])).unwrap();
    assert!(matches!(
        UFuncArray::multi_dot(&[&too_many, &columns, &unit]),
        Err(UFuncError::Shape(ShapeError::Overflow))
    ));

    // A complex chain can first multiply two real operands. Check their
    // physical output capacity before the real dot kernel allocates its Vec.
    let extent = 1_usize << (usize::BITS / 2 - 1);
    let row = UFuncArray::new(vec![0, extent], vec![], DType::F64).unwrap();
    let column = UFuncArray::new(vec![extent, 0], vec![], DType::F64).unwrap();
    assert!(matches!(
        UFuncArray::multi_dot(&[&empty, &row, &column, &row]),
        Err(UFuncError::Shape(ShapeError::Overflow))
    ));
}
