//! Complex reduction contracts for the interleaved Rust array representation.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
use std::process::{Command, Stdio};

fn complex_array(shape: &[usize], values: &[(f64, f64)], dtype: DType) -> UFuncArray {
    let mut physical_shape = shape.to_vec();
    physical_shape.push(2);
    UFuncArray::new(
        physical_shape,
        values.iter().flat_map(|&(re, im)| [re, im]).collect(),
        dtype,
    )
    .expect("valid interleaved complex array")
}

fn reduce(
    array: &UFuncArray,
    operation: &str,
    dtype: Option<DType>,
    axis: Option<isize>,
    keepdims: bool,
    ddof: usize,
) -> Result<UFuncArray, UFuncError> {
    match (operation, dtype) {
        ("sum", None) => array.reduce_sum(axis, keepdims),
        ("prod", None) => array.reduce_prod(axis, keepdims),
        ("mean", None) => array.reduce_mean(axis, keepdims),
        ("var", None) => array.reduce_var(axis, keepdims, ddof),
        ("std", None) => array.reduce_std(axis, keepdims, ddof),
        ("nansum", None) => array.nansum(axis, keepdims),
        ("nanprod", None) => array.nanprod(axis, keepdims),
        ("nanmean", None) => array.nanmean(axis, keepdims),
        ("nanvar", None) => array.nanvar(axis, keepdims, ddof),
        ("nanstd", None) => array.nanstd(axis, keepdims, ddof),
        ("cumsum", None) => array.cumsum(axis),
        ("cumprod", None) => array.cumprod(axis),
        ("nancumsum", None) => array.nancumsum(axis),
        ("nancumprod", None) => array.nancumprod(axis),
        ("cumulative_sum", None) => array.cumulative_sum(axis),
        ("cumulative_prod", None) => array.cumulative_prod(axis),
        ("sum", Some(dtype)) => array.reduce_sum_with_dtype(axis, keepdims, dtype),
        ("prod", Some(dtype)) => array.reduce_prod_with_dtype(axis, keepdims, dtype),
        ("mean", Some(dtype)) => array.reduce_mean_with_dtype(axis, keepdims, dtype),
        ("var", Some(dtype)) => array.reduce_var_with_dtype(axis, keepdims, ddof, dtype),
        ("std", Some(dtype)) => array.reduce_std_with_dtype(axis, keepdims, ddof, dtype),
        _ => unreachable!("test reduction operation"),
    }
}

fn numpy_dtype(dtype: DType) -> &'static str {
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
        _ => panic!("test requires a numeric dtype"),
    }
}

#[derive(Default)]
struct OracleBatch {
    inputs: Vec<Value>,
    cases: Vec<Value>,
    actual: Vec<Result<UFuncArray, UFuncError>>,
}

impl OracleBatch {
    fn add_input(&mut self, array: &UFuncArray) -> usize {
        let mut shape = array.shape().to_vec();
        let values: Value = if array.dtype().is_complex() {
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
        } else {
            let values: Vec<String> = match array.to_storage().expect("real input storage") {
                ArrayStorage::I64(values) => values.into_iter().map(|v| v.to_string()).collect(),
                ArrayStorage::U64(values) => values.into_iter().map(|v| v.to_string()).collect(),
                _ => array.values().iter().map(|v| v.to_string()).collect(),
            };
            json!(values)
        };
        let index = self.inputs.len();
        self.inputs.push(json!({
            "shape": shape,
            "dtype": numpy_dtype(array.dtype()),
            "values": values,
        }));
        index
    }

    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        input: usize,
        array: &UFuncArray,
        operation: &str,
        dtype: Option<DType>,
        axis: Option<isize>,
        keepdims: bool,
        ddof: usize,
    ) {
        self.cases.push(json!({
            "input": input,
            "op": operation,
            "dtype": dtype.map(numpy_dtype),
            "axis": axis,
            "keepdims": keepdims,
            "ddof": ddof,
        }));
        self.actual
            .push(reduce(array, operation, dtype, axis, keepdims, ddof));
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np

warnings.simplefilter('ignore')
payload = json.load(sys.stdin)
inputs = []
for item in payload['inputs']:
    dtype = np.dtype(item['dtype'])
    if dtype.kind == 'c':
        a = np.empty(len(item['values']), dtype=dtype)
        a.real = [float(pair[0]) for pair in item['values']]
        a.imag = [float(pair[1]) for pair in item['values']]
    else:
        convert = int if dtype.kind in 'iu' else float
        a = np.array([convert(v) for v in item['values']], dtype=dtype)
    inputs.append(a.reshape(item['shape']))
rows = []
for case in payload['cases']:
    kw = dict(axis=case['axis'], keepdims=case['keepdims'])
    if 'cum' in case['op']:
        del kw['keepdims']
    if case['dtype'] is not None:
        kw['dtype'] = case['dtype']
    if case['op'] in ('var', 'std', 'nanvar', 'nanstd'):
        kw['ddof'] = case['ddof']
    try:
        result = np.asarray(getattr(np, case['op'])(inputs[case['input']], **kw))
        if result.dtype.kind == 'c':
            parts = np.empty(result.size * 2, dtype=np.float64)
            parts[0::2] = result.real.ravel()
            parts[1::2] = result.imag.ravel()
        else:
            parts = result.astype(np.float64).ravel()
        rows.append(dict(shape=list(result.shape), dtype=str(result.dtype),
                         bits=parts.view(np.uint64).tolist()))
    except (TypeError, ValueError, IndexError) as error:
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
            .expect("launch the live NumPy complex reduction oracle");
        let mut stdin = child.stdin.take().expect("oracle stdin");
        serde_json::to_writer(
            &mut stdin,
            &json!({"inputs": self.inputs, "cases": self.cases}),
        )
        .expect("write oracle inputs");
        drop(stdin);
        let output = child.wait_with_output().expect("collect oracle outputs");
        assert!(
            output.status.success(),
            "NumPy complex reduction oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
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
                assert_eq!(shape.pop(), Some(2), "{case}: missing interleaved lane");
            }
            assert_eq!(json!(shape), expected["shape"], "{case}: shape");
            assert_eq!(
                numpy_dtype(actual.dtype()),
                expected["dtype"],
                "{case}: dtype"
            );
            let expected_bits = expected["bits"].as_array().expect("result bits");
            assert_eq!(actual.values().len(), expected_bits.len(), "{case}: length");
            for (index, (&value, bits)) in actual.values().iter().zip(expected_bits).enumerate() {
                let expected = f64::from_bits(bits.as_u64().expect("float bits"));
                if value.is_nan() && expected.is_nan() {
                    continue;
                }
                assert_eq!(
                    value.to_bits(),
                    expected.to_bits(),
                    "{case}: component {index}, actual={value:?}, expected={expected:?}"
                );
            }
        }
    }
}

#[test]
fn complex_default_reductions_preserve_components_and_real_variance() {
    let a = complex_array(&[2], &[(1.0, 2.0), (3.0, 4.0)], DType::Complex128);
    let sum = a.reduce_sum(None, false).unwrap();
    assert_eq!(sum.shape(), &[2]);
    assert_eq!(sum.values(), &[4.0, 6.0]);
    assert_eq!(a.reduce_prod(None, false).unwrap().values(), &[-5.0, 10.0]);
    assert_eq!(a.reduce_mean(None, false).unwrap().values(), &[2.0, 3.0]);
    let variance = a.reduce_var(None, false, 0).unwrap();
    assert_eq!(variance.shape(), &[] as &[usize]);
    assert_eq!(variance.dtype(), DType::F64);
    assert_eq!(variance.values(), &[2.0]);
    assert_eq!(
        a.reduce_std(None, false, 0).unwrap().values(),
        &[2.0f64.sqrt()]
    );
}

#[test]
fn complex_reductions_validate_logical_axes_and_storage() {
    let scalar = complex_array(&[], &[(3.0, 4.0)], DType::Complex64);
    for axis in [None, Some(0), Some(-1)] {
        assert_eq!(scalar.reduce_sum(axis, true).unwrap().values(), &[3.0, 4.0]);
        assert_eq!(
            scalar.reduce_prod(axis, true).unwrap().values(),
            &[3.0, 4.0]
        );
    }
    for axis in [Some(0), Some(-1), Some(1)] {
        assert!(scalar.reduce_mean(axis, false).is_err());
        assert!(scalar.reduce_var(axis, false, 0).is_err());
        assert!(scalar.reduce_std(axis, false, 0).is_err());
    }
    for a in [
        UFuncArray::new(vec![3], vec![1.0; 3], DType::Complex64).unwrap(),
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
    ] {
        for operation in ["sum", "prod", "mean", "var", "std"] {
            assert!(reduce(&a, operation, None, None, false, 0).is_err());
        }
    }
}

#[test]
fn complex_reduction_dtype_axis_and_empty_matrix_matches_numpy() {
    let mut batch = OracleBatch::default();
    for source_dtype in [DType::Complex64, DType::Complex128] {
        for shape in [vec![], vec![2, 3], vec![2, 0, 3], vec![0, 2]] {
            let count = shape.iter().product();
            let pattern = [(1.25, 2.5), (-3.0, 0.5), (0.0, -4.25), (5.5, 1.0)];
            let values = (0..count)
                .map(|i| pattern[i % pattern.len()])
                .collect::<Vec<_>>();
            let a = complex_array(&shape, &values, source_dtype);
            let input = batch.add_input(&a);
            for operation in ["sum", "prod", "mean", "var", "std"] {
                for dtype in [
                    None,
                    Some(DType::Complex64),
                    Some(DType::Complex128),
                    Some(DType::F16),
                    Some(DType::F32),
                    Some(DType::F64),
                ] {
                    for axis in [None, Some(0), Some(-1), Some(1), Some(-3)] {
                        for keepdims in [false, true] {
                            for ddof in if matches!(operation, "var" | "std") {
                                &[0, 1, 7][..]
                            } else {
                                &[0][..]
                            } {
                                batch.push(input, &a, operation, dtype, axis, keepdims, *ddof);
                            }
                        }
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_real_integer_and_boolean_accumulators_match_numpy() {
    let mut batch = OracleBatch::default();
    let a = complex_array(
        &[2, 3],
        &[
            (0.0, 2.0),
            (3.0, 4.0),
            (-2.0, 0.0),
            (1.5, -0.5),
            (5.0, 1.0),
            (-1.0, 3.0),
        ],
        DType::Complex64,
    );
    let input = batch.add_input(&a);
    for operation in ["sum", "prod", "mean", "var", "std"] {
        for dtype in [
            DType::Bool,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
        ] {
            for axis in [None, Some(0), Some(-1)] {
                for keepdims in [false, true] {
                    batch.push(input, &a, operation, Some(dtype), axis, keepdims, 0);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_accumulators_accept_real_inputs_without_losing_source_precision() {
    let mut batch = OracleBatch::default();
    let midpoint = (1_i64 << 62) + (1_i64 << 38);
    let arrays = [
        UFuncArray::new(
            vec![2, 3],
            vec![1.001, 2.003, 3.005, -1.007, 0.0, 4.009],
            DType::F16,
        )
        .unwrap(),
        UFuncArray::new(
            vec![2, 3],
            vec![16_777_217.0, 1.0, -16_777_216.0, 3.0, -2.0, 1.0],
            DType::F32,
        )
        .unwrap(),
        UFuncArray::new(
            vec![2, 3],
            vec![16_777_217.0, 1.0, -16_777_216.0, 3.0, -2.0, 1.0],
            DType::F64,
        )
        .unwrap(),
        UFuncArray::from_storage(vec![2], ArrayStorage::I64(vec![midpoint - 1, midpoint + 1]))
            .unwrap(),
        UFuncArray::from_storage(
            vec![2, 3],
            ArrayStorage::Bool(vec![true, false, true, false, false, true]),
        )
        .unwrap(),
    ];
    for a in arrays {
        let input = batch.add_input(&a);
        for operation in ["sum", "prod", "mean", "var", "std"] {
            for dtype in [DType::Complex64, DType::Complex128] {
                for axis in [None, Some(0), Some(-1)] {
                    for keepdims in [false, true] {
                        batch.push(input, &a, operation, Some(dtype), axis, keepdims, 0);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_pairwise_order_conversion_buffers_and_strided_lanes_match_numpy() {
    let mut batch = OracleBatch::default();
    for source_dtype in [DType::Complex64, DType::Complex128] {
        for len in [3, 4, 63, 64, 65, 127, 128, 129, 8191, 8192, 8193] {
            let large = if source_dtype == DType::Complex64 {
                1e8
            } else {
                1e16
            };
            let values = (0..len * 2)
                .map(|i| {
                    let re = [large, 1.0, -large, 0.25, 3.0, -0.5, -2.0][i % 7];
                    let im = [-large, 2.0, large, -0.125, 1.0][i % 5];
                    (re, im)
                })
                .collect::<Vec<_>>();
            for shape in [[2, len], [len, 2]] {
                let a = complex_array(&shape, &values, source_dtype);
                let input = batch.add_input(&a);
                for operation in ["sum", "mean", "var", "std"] {
                    for dtype in [None, Some(DType::Complex64), Some(DType::Complex128)] {
                        for axis in [None, Some(0), Some(1)] {
                            batch.push(input, &a, operation, dtype, axis, false, 0);
                        }
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_to_real_reductions_keep_conversion_buffer_boundaries() {
    let mut batch = OracleBatch::default();
    for (source_dtype, target_dtype, large) in [
        (DType::Complex64, DType::F32, 1e8),
        (DType::Complex128, DType::F64, 1e16),
    ] {
        let values = (0..8193)
            .map(|index| ([large, -large, 1.0][index % 3], 0.0))
            .collect::<Vec<_>>();
        let a = complex_array(&[8193], &values, source_dtype);
        // The final element begins a new conversion buffer. Reducing a copied
        // real component as one unbuffered lane would instead produce zero.
        assert_eq!(
            a.reduce_sum_with_dtype(None, false, target_dtype)
                .unwrap()
                .values(),
            &[1.0]
        );
        let input = batch.add_input(&a);
        for operation in ["sum", "prod", "mean", "var", "std"] {
            for axis in [None, Some(0), Some(-1)] {
                for keepdims in [false, true] {
                    batch.push(input, &a, operation, Some(target_dtype), axis, keepdims, 0);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_signed_zeros_and_nonfinite_values_match_numpy() {
    let mut batch = OracleBatch::default();
    for source_dtype in [DType::Complex64, DType::Complex128] {
        for values in [
            vec![(-0.0, -0.0), (-0.0, 0.0)],
            vec![(f64::INFINITY, 1.0), (1.0, 2.0)],
            vec![(1.0, f64::INFINITY), (2.0, 3.0)],
            vec![(f64::NAN, 1.0), (1.0, 2.0)],
            vec![(1.0, f64::NAN), (1.0, 2.0)],
        ] {
            let a = complex_array(&[2], &values, source_dtype);
            let input = batch.add_input(&a);
            for operation in ["sum", "prod", "mean", "var", "std"] {
                for dtype in [
                    None,
                    Some(DType::Complex64),
                    Some(DType::Complex128),
                    Some(DType::F64),
                ] {
                    batch.push(input, &a, operation, dtype, None, false, 0);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_nan_reductions_skip_whole_elements_and_keep_logical_shapes() {
    let a = complex_array(
        &[2, 2],
        &[(1.0, 2.0), (f64::NAN, 100.0), (50.0, f64::NAN), (3.0, 4.0)],
        DType::Complex128,
    );
    let sum = a.nansum(None, false).unwrap();
    assert_eq!(sum.shape(), &[2]);
    assert_eq!(sum.values(), &[4.0, 6.0]);
    assert_eq!(a.nanprod(None, false).unwrap().values(), &[-5.0, 10.0]);
    assert_eq!(a.nanmean(None, false).unwrap().values(), &[2.0, 3.0]);
    let variance = a.nanvar(None, true, 0).unwrap();
    assert_eq!(variance.shape(), &[1, 1]);
    assert_eq!(variance.dtype(), DType::F64);
    assert_eq!(variance.values(), &[2.0]);
    let rows = a.nanmean(Some(-1), true).unwrap();
    assert_eq!(rows.shape(), &[2, 1, 2]);
    assert_eq!(rows.values(), &[1.0, 2.0, 3.0, 4.0]);
    assert!(a.nanvar(None, false, 2).unwrap().values()[0].is_nan());
    for operation in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
        assert!(reduce(&a, operation, None, Some(2), false, 0).is_err());
        let malformed = UFuncArray::new(vec![3], vec![1.0; 3], DType::Complex128).unwrap();
        assert!(reduce(&malformed, operation, None, None, false, 0).is_err());
    }
}

#[test]
fn complex_nan_reductions_match_numpy_across_axes_empty_lanes_and_ddof() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [
            vec![],
            vec![7],
            vec![2, 3],
            vec![2, 3, 4],
            vec![0],
            vec![2, 0, 3],
            vec![0, 2],
        ] {
            for pattern in [
                vec![
                    (1.25, 2.5),
                    (f64::NAN, 100.0),
                    (-3.0, 0.5),
                    (50.0, f64::NAN),
                    (5.5, -1.0),
                ],
                vec![(f64::NAN, 1.0), (2.0, f64::NAN)],
                vec![(-0.0, -0.0), (-0.0, 0.0)],
                vec![(f64::INFINITY, 1.0), (1.0, f64::INFINITY), (f64::NAN, 0.0)],
            ] {
                let count = shape.iter().product();
                let values = (0..count)
                    .map(|i| pattern[i % pattern.len()])
                    .collect::<Vec<_>>();
                let a = complex_array(&shape, &values, dtype);
                let input = batch.add_input(&a);
                for operation in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
                    for axis in [None, Some(0), Some(-1), Some(1), Some(2), Some(-3)] {
                        for keepdims in [false, true] {
                            for ddof in if matches!(operation, "nanvar" | "nanstd") {
                                &[0, 1, 7][..]
                            } else {
                                &[0][..]
                            } {
                                batch.push(input, &a, operation, None, axis, keepdims, *ddof);
                            }
                        }
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_nan_reductions_preserve_pairwise_positions_and_component_precision() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let large = if dtype == DType::Complex64 { 1e8 } else { 1e16 };
        for len in [3, 4, 63, 64, 65, 127, 129, 8193] {
            let pattern = [
                (large, -large),
                (f64::NAN, 5.0),
                (1.0, 2.0),
                (-large, large),
                (3.0, f64::NAN),
                (0.25, -0.125),
            ];
            let values = (0..len * 2)
                .map(|i| pattern[i % pattern.len()])
                .collect::<Vec<_>>();
            for shape in [[2, len], [len, 2]] {
                let a = complex_array(&shape, &values, dtype);
                let input = batch.add_input(&a);
                for operation in ["nansum", "nanmean", "nanvar", "nanstd"] {
                    for axis in [None, Some(0), Some(1)] {
                        batch.push(input, &a, operation, None, axis, false, 0);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_cumulative_operations_preserve_pairs_and_first_element_bits() {
    let a = complex_array(&[2], &[(1.0, 2.0), (3.0, 4.0)], DType::Complex128);
    let sums = a.cumsum(None).unwrap();
    assert_eq!(sums.shape(), &[2, 2]);
    assert_eq!(sums.values(), &[1.0, 2.0, 4.0, 6.0]);
    assert_eq!(a.cumprod(None).unwrap().values(), &[1.0, 2.0, -5.0, 10.0]);
    for first in [(-0.0, -0.0), (f64::INFINITY, 1.0), (1.0, f64::INFINITY)] {
        let a = complex_array(&[1], &[first], DType::Complex128);
        for result in [a.cumsum(None).unwrap(), a.cumprod(None).unwrap()] {
            assert_eq!(result.values()[0].to_bits(), first.0.to_bits());
            assert_eq!(result.values()[1].to_bits(), first.1.to_bits());
        }
    }
    let a = complex_array(
        &[3],
        &[(f64::NAN, 10.0), (3.0, 4.0), (5.0, f64::NAN)],
        DType::Complex128,
    );
    assert_eq!(
        a.nancumsum(None).unwrap().values(),
        &[0.0, 0.0, 3.0, 4.0, 3.0, 4.0]
    );
    assert_eq!(
        a.nancumprod(None).unwrap().values(),
        &[1.0, 0.0, 3.0, 4.0, 3.0, 4.0]
    );

    let empty = complex_array(&[0, usize::MAX, usize::MAX], &[], DType::Complex64);
    for operation in ["cumsum", "cumprod", "nancumsum", "nancumprod"] {
        let flattened = reduce(&empty, operation, None, None, false, 0).unwrap();
        assert_eq!(flattened.shape(), &[0, 2]);
        assert!(
            reduce(&empty, operation, None, Some(0), false, 0)
                .unwrap()
                .values()
                .is_empty()
        );
        assert!(reduce(&empty, operation, None, Some(3), false, 0).is_err());
        let malformed = UFuncArray::new(vec![3], vec![1.0; 3], DType::Complex128).unwrap();
        assert!(reduce(&malformed, operation, None, None, false, 0).is_err());
    }
    let matrix = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
    assert!(matrix.cumulative_sum(None).is_err());
    assert!(matrix.cumulative_prod(None).is_err());
    assert_eq!(
        matrix.cumsum(None).unwrap().values(),
        &[1.0, 3.0, 6.0, 10.0]
    );
}

#[test]
fn complex_cumulative_operations_match_numpy_on_logical_axes_and_missing_values() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [
            vec![],
            vec![7],
            vec![2, 3],
            vec![2, 3, 4],
            vec![0],
            vec![2, 0, 3],
            vec![0, 2],
        ] {
            for pattern in [
                vec![(1.25, 2.5), (-3.0, 0.5), (0.0, -4.25), (5.5, 1.0)],
                vec![(f64::NAN, 1.0), (2.0, 3.0), (50.0, f64::NAN)],
                vec![(-0.0, -0.0), (0.0, 0.0), (-0.0, 0.0)],
                vec![(f64::INFINITY, 1.0), (1.0, f64::INFINITY), (f64::NAN, 0.0)],
                vec![
                    (16_777_217.0, -16_777_217.0),
                    (1.0, 1.0),
                    (-16_777_216.0, 16_777_216.0),
                ],
            ] {
                let count = shape.iter().product();
                let values = (0..count)
                    .map(|i| pattern[i % pattern.len()])
                    .collect::<Vec<_>>();
                let a = complex_array(&shape, &values, dtype);
                let input = batch.add_input(&a);
                for operation in [
                    "cumsum",
                    "cumprod",
                    "nancumsum",
                    "nancumprod",
                    "cumulative_sum",
                    "cumulative_prod",
                ] {
                    for axis in [None, Some(0), Some(-1), Some(1), Some(2), Some(-3)] {
                        batch.push(input, &a, operation, None, axis, false, 0);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn cumulative_scalar_axes_match_numpy_across_numeric_dtypes() {
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
        let a = UFuncArray::from_storage(
            vec![],
            ArrayStorage::F64(vec![3.25]).cast_to(dtype).unwrap(),
        )
        .unwrap();
        let input = batch.add_input(&a);
        for operation in [
            "cumsum",
            "cumprod",
            "nancumsum",
            "nancumprod",
            "cumulative_sum",
            "cumulative_prod",
        ] {
            for axis in [None, Some(0), Some(-1), Some(1), Some(-2)] {
                batch.push(input, &a, operation, None, axis, false, 0);
            }
        }
    }
    batch.assert_matches_numpy();
}
