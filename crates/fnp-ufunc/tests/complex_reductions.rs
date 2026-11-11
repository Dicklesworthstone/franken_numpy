//! Complex casting and reduction contracts for the interleaved Rust array representation.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{
    FloatErrorKind, FloatErrorMode, UFuncArray, UFuncError, errstate, seterrcall,
    take_float_error_events,
};
use serde_json::{Value, json};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

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
        ("nansum", Some(dtype)) => array.nansum_with_dtype(axis, keepdims, dtype),
        ("nanprod", Some(dtype)) => array.nanprod_with_dtype(axis, keepdims, dtype),
        ("nanmean", Some(dtype)) => array.nanmean_with_dtype(axis, keepdims, dtype),
        ("nanvar", Some(dtype)) => array.nanvar_with_dtype(axis, keepdims, ddof, dtype),
        ("nanstd", Some(dtype)) => array.nanstd_with_dtype(axis, keepdims, ddof, dtype),
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

    fn push_cast(&mut self, input: usize, array: &UFuncArray, dtype: DType) {
        self.cases.push(json!({
            "input": input,
            "op": "astype",
            "dtype": numpy_dtype(dtype),
            "axis": null,
            "keepdims": false,
            "ddof": 0,
        }));
        let _guard = errstate(None, None, None, None, Some(FloatErrorMode::Ignore));
        self.actual.push(array.astype_with_casting(dtype, "unsafe"));
    }

    fn push_storage_cast(
        &mut self,
        input: usize,
        shape: &[usize],
        storage: &ArrayStorage,
        dtype: DType,
    ) {
        self.cases.push(json!({
            "input": input,
            "op": "astype",
            "dtype": numpy_dtype(dtype),
            "axis": null,
            "keepdims": false,
            "ddof": 0,
        }));
        self.actual.push(UFuncArray::from_storage_with_dtype(
            shape.to_vec(),
            storage.clone(),
            dtype,
        ));
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
        if case['op'] == 'astype':
            result = inputs[case['input']].astype(case['dtype'], casting='unsafe')
        else:
            result = np.asarray(getattr(np, case['op'])(inputs[case['input']], **kw))
        if result.dtype.kind == 'c':
            parts = np.empty(result.size * 2, dtype=np.float64)
            parts[0::2] = result.real.ravel()
            parts[1::2] = result.imag.ravel()
        else:
            parts = result.astype(np.float64).ravel()
        row = dict(shape=list(result.shape), dtype=str(result.dtype),
                   bits=parts.view(np.uint64).tolist())
        if case['op'] == 'astype' and result.dtype.kind in 'iu':
            row['integers'] = [str(int(value)) for value in result.ravel()]
        rows.append(row)
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
            if let Some(integers) = expected.get("integers") {
                let storage = actual.to_storage().expect("exact integer cast output");
                let actual_integers = (0..storage.len())
                    .map(|index| storage.get_i128(index).unwrap().to_string())
                    .collect::<Vec<_>>();
                assert_eq!(json!(actual_integers), *integers, "{case}: integer bits");
            }
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
fn nan_default_f64_preserves_mask_positions_in_contiguous_reductions() {
    let cancellation = vec![1e16, 1.0, 1.0, f64::NAN, 1.0, 1.0, 1.0, -1e16];
    let a = UFuncArray::new(vec![8], cancellation.clone(), DType::F64).unwrap();
    assert_eq!(a.nansum(None, false).unwrap().values(), &[2.0]);
    assert_eq!(a.nanmean(None, false).unwrap().values(), &[2.0 / 7.0]);
    let mut batch = OracleBatch::default();
    for values in [
        cancellation,
        vec![
            0.057571064743431044,
            12.915675752486784,
            0.03249502120569186,
            f64::NAN,
            -645.4586539400414,
            2029.5216975049639,
            0.016698446653827605,
            -0.9616396122752112,
            -2116.1740398670127,
        ],
    ] {
        let a = UFuncArray::new(vec![values.len()], values, DType::F64).unwrap();
        let input = batch.add_input(&a);
        for operation in ["nansum", "nanmean", "nanvar", "nanstd"] {
            for keepdims in [false, true] {
                for ddof in [0, 2] {
                    batch.push(input, &a, operation, None, None, keepdims, ddof);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn nan_dtype_preserves_source_precision_masks_and_exact_integers() {
    let a = UFuncArray::new(vec![3], vec![1024.5, f64::NAN, 1025.5], DType::F64).unwrap();
    // The half accumulator rounds the mean, but deviations stay in float64.
    // Casting the input to half before subtracting would yield variance 1.
    assert_eq!(
        a.nanvar_with_dtype(None, false, 0, DType::F16)
            .unwrap()
            .values(),
        &[0.25]
    );
    let a = UFuncArray::new(vec![3], vec![500.0, f64::NAN, -500.0], DType::F16).unwrap();
    // Conversely, a wide accumulator cannot prevent the source half-precision
    // squared-deviation buffer from overflowing.
    assert_eq!(
        a.nanvar_with_dtype(None, false, 0, DType::F64)
            .unwrap()
            .values(),
        &[f64::INFINITY]
    );
    let a = complex_array(
        &[4],
        &[(1.0, 2.0), (f64::NAN, 100.0), (50.0, f64::NAN), (3.0, 4.0)],
        DType::Complex128,
    );
    assert_eq!(
        a.nansum_with_dtype(None, false, DType::F64)
            .unwrap()
            .values(),
        &[4.0]
    );
    assert_eq!(
        a.nanprod_with_dtype(None, false, DType::F64)
            .unwrap()
            .values(),
        &[3.0]
    );
    assert_eq!(
        a.nanvar_with_dtype(None, false, 0, DType::F64)
            .unwrap()
            .values(),
        &[11.0]
    );
    let bad = a
        .nanvar_with_dtype(None, true, 2, DType::Complex64)
        .unwrap();
    assert_eq!(bad.shape(), &[1, 2]);
    assert!(bad.values()[0].is_nan());
    assert_eq!(bad.values()[1].to_bits(), 0.0_f64.to_bits());
    let bad_std = a
        .nanstd_with_dtype(None, false, 2, DType::Complex64)
        .unwrap();
    assert!(bad_std.values().iter().all(|value| value.is_nan()));

    let large = (1_u64 << 53) + 1;
    let a = UFuncArray::from_storage(vec![2], ArrayStorage::U64(vec![large, 3])).unwrap();
    assert_eq!(
        a.nansum_with_dtype(None, false, DType::U64)
            .unwrap()
            .to_storage()
            .unwrap(),
        ArrayStorage::U64(vec![large + 3])
    );
    assert_eq!(
        a.nanprod_with_dtype(None, false, DType::U64)
            .unwrap()
            .to_storage()
            .unwrap(),
        ArrayStorage::U64(vec![large * 3])
    );
}

#[test]
fn nan_dtype_rejects_invalid_accumulators_and_complex_storage() {
    for source in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        let a = if source.is_complex() {
            complex_array(&[1], &[(1.0, 2.0)], source)
        } else {
            UFuncArray::new(vec![1], vec![1.0], source).unwrap()
        };
        // The restriction applies to inexact sources even with no NaN values.
        for dtype in [
            DType::Bool,
            DType::I64,
            DType::U8,
            DType::Str,
            DType::Structured,
        ] {
            for operation in ["nanmean", "nanvar", "nanstd"] {
                assert!(reduce(&a, operation, Some(dtype), None, false, 0).is_err());
            }
        }
    }
    for a in [
        UFuncArray::new(vec![3], vec![1.0; 3], DType::Complex64).unwrap(),
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
        UFuncArray::new(vec![1], vec![1.0], DType::Str).unwrap(),
    ] {
        for operation in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
            assert!(reduce(&a, operation, Some(DType::F64), None, false, 0).is_err());
        }
    }
}

#[test]
fn nan_dtype_real_complex_integer_axes_and_empty_results_match_numpy() {
    let mut batch = OracleBatch::default();
    for source in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        let a = if source.is_complex() {
            complex_array(
                &[2, 3],
                &[
                    (f64::NAN, 7.0),
                    (2.0, f64::NAN),
                    (f64::NAN, f64::NAN),
                    (1.001, 2.003),
                    (3.0, -0.5),
                    (-2.0, 1.25),
                ],
                source,
            )
        } else {
            UFuncArray::new(
                vec![2, 3],
                vec![f64::NAN, f64::NAN, f64::NAN, 1.001, 3.0, -2.0],
                source,
            )
            .unwrap()
        };
        let input = batch.add_input(&a);
        for dtype in [
            DType::F16,
            DType::F32,
            DType::F64,
            DType::Complex64,
            DType::Complex128,
            DType::I8,
            DType::Bool,
        ] {
            for operation in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
                for (axis, keepdims, ddof) in
                    [(None, false, 0), (Some(0), true, 1), (Some(-1), false, 7)]
                {
                    batch.push(input, &a, operation, Some(dtype), axis, keepdims, ddof);
                }
            }
        }
    }
    for source in [DType::F32, DType::Complex128] {
        for shape in [vec![], vec![2, 0, 3], vec![0, 2]] {
            let count = shape.iter().product();
            let a = if source.is_complex() {
                complex_array(&shape, &vec![(f64::NAN, 1.0); count], source)
            } else {
                UFuncArray::new(shape, vec![f64::NAN; count], source).unwrap()
            };
            let input = batch.add_input(&a);
            for dtype in [DType::F16, DType::F64, DType::Complex128] {
                for operation in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
                    for (axis, keepdims, ddof) in [
                        (None, true, 0),
                        (Some(0), false, 0),
                        (Some(-1), true, 1),
                        (Some(2), false, 0),
                    ] {
                        batch.push(input, &a, operation, Some(dtype), axis, keepdims, ddof);
                    }
                }
            }
        }
    }
    for storage in [
        ArrayStorage::Bool(vec![true, false, true, false, true, false]),
        ArrayStorage::I8(vec![-127, 126, 4, -3, 2, 5]),
        ArrayStorage::I64(vec![(1_i64 << 53) + 1, -2, 3, 5, -7, 11]),
        ArrayStorage::U64(vec![(1_u64 << 53) + 1, 2, 3, 5, 7, 11]),
    ] {
        let a = UFuncArray::from_storage(vec![2, 3], storage).unwrap();
        let input = batch.add_input(&a);
        for dtype in [
            DType::F16,
            DType::F32,
            DType::F64,
            DType::Complex64,
            DType::Complex128,
            DType::I8,
            DType::U64,
            DType::Bool,
        ] {
            for operation in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
                for (axis, keepdims) in [(None, false), (Some(0), false), (Some(-1), true)] {
                    batch.push(input, &a, operation, Some(dtype), axis, keepdims, 0);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn nan_dtype_conversion_buffers_and_large_valid_counts_match_numpy() {
    let mut batch = OracleBatch::default();
    for source in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        let large = match source {
            DType::F16 => 1e3,
            DType::F32 | DType::Complex64 => 1e8,
            _ => 1e16,
        };
        let targets = if matches!(source, DType::F64 | DType::Complex128) {
            [DType::F32, DType::Complex64]
        } else {
            [DType::F64, DType::Complex128]
        };
        for len in [129, 8193] {
            let pattern = [
                (large, -large),
                (f64::NAN, 7.0),
                (1.001, 2.003),
                (-large, large),
                (0.25, -0.125),
                (3.0, 1.0),
            ];
            let values = (0..2 * len)
                .map(|index| pattern[index % pattern.len()])
                .collect::<Vec<_>>();
            for (shape, axis) in [([2, len], 1), ([len, 2], 0)] {
                let a = if source.is_complex() {
                    complex_array(&shape, &values, source)
                } else {
                    UFuncArray::new(
                        shape.to_vec(),
                        values.iter().map(|pair| pair.0).collect(),
                        source,
                    )
                    .unwrap()
                };
                let input = batch.add_input(&a);
                for dtype in targets {
                    for operation in ["nansum", "nanmean", "nanvar", "nanstd"] {
                        batch.push(input, &a, operation, Some(dtype), Some(axis), false, 0);
                    }
                }
            }
        }
    }
    // A valid-element count above float16's finite range must remain integer,
    // while the half sums and squared deviations are still representable.
    let a = UFuncArray::new(
        vec![70_003],
        (0..70_003)
            .map(|index| {
                if index % 101 == 0 {
                    f64::NAN
                } else if index % 2 == 0 {
                    0.25
                } else {
                    -0.25
                }
            })
            .collect(),
        DType::F16,
    )
    .unwrap();
    let input = batch.add_input(&a);
    for dtype in [DType::F16, DType::F64, DType::Complex64] {
        for operation in ["nanmean", "nanvar", "nanstd"] {
            for ddof in [0, 3] {
                batch.push(input, &a, operation, Some(dtype), None, false, ddof);
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

#[test]
fn complex_nan_reductions_discard_entire_elements_and_preserve_logical_shape() {
    let a = complex_array(
        &[2, 3],
        &[
            (f64::NAN, 100.0),
            (100.0, f64::NAN),
            (1.0, 2.0),
            (3.0, 4.0),
            (f64::NAN, f64::NAN),
            (5.0, 6.0),
        ],
        DType::Complex128,
    );
    let sum = a.nansum(Some(1), true).unwrap();
    assert_eq!(sum.shape(), &[2, 1, 2]);
    assert_eq!(sum.values(), &[1.0, 2.0, 8.0, 10.0]);
    assert_eq!(a.nanprod(None, false).unwrap().values(), &[-85.0, 20.0]);
    let mean = a.nanmean(Some(1), false).unwrap();
    assert_eq!(mean.shape(), &[2, 2]);
    assert_eq!(mean.values(), &[1.0, 2.0, 4.0, 5.0]);
    let variance = a.nanvar(Some(1), true, 0).unwrap();
    assert_eq!(variance.shape(), &[2, 1]);
    assert_eq!(variance.dtype(), DType::F64);
    assert_eq!(variance.values(), &[0.0, 2.0]);
    assert!(a.nanvar(Some(1), false, 1).unwrap().values()[0].is_nan());
    assert_eq!(a.nanvar(Some(1), false, 1).unwrap().values()[1], 4.0);

    let all_nan = complex_array(&[1], &[(1.0, f64::NAN)], DType::Complex64);
    assert_eq!(all_nan.nansum(None, false).unwrap().values(), &[0.0, 0.0]);
    assert_eq!(all_nan.nanprod(None, false).unwrap().values(), &[1.0, 0.0]);
    for operation in ["nanmean", "nanvar", "nanstd"] {
        assert!(
            reduce(&all_nan, operation, None, None, false, 0)
                .unwrap()
                .values()
                .iter()
                .all(|value| value.is_nan())
        );
    }
}

#[test]
fn complex_nan_reductions_validate_storage_and_logical_axes() {
    for malformed in [
        UFuncArray::new(vec![3], vec![1.0; 3], DType::Complex64).unwrap(),
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
    ] {
        for operation in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
            assert!(reduce(&malformed, operation, None, None, false, 0).is_err());
        }
    }
    let scalar = complex_array(&[], &[(3.0, 4.0)], DType::Complex64);
    for operation in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
        for axis in [None, Some(0), Some(-1)] {
            let output = reduce(&scalar, operation, None, axis, true, 0).unwrap();
            if matches!(operation, "nanvar" | "nanstd") {
                assert_eq!(output.shape(), &[] as &[usize]);
                assert_eq!(output.values(), &[0.0]);
            } else {
                assert_eq!(output.shape(), &[2]);
                assert_eq!(output.values(), &[3.0, 4.0]);
            }
        }
        assert!(reduce(&scalar, operation, None, Some(1), false, 0).is_err());
        let vector = complex_array(&[2], &[(1.0, 2.0), (3.0, 4.0)], DType::Complex64);
        assert!(reduce(&vector, operation, None, Some(1), false, 0).is_err());
    }
}

#[test]
fn complex_nan_reduction_axes_empty_and_ddof_matrix_matches_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [vec![], vec![2, 3], vec![2, 0, 3], vec![0, 2], vec![2, 2, 3]] {
            let count = shape.iter().product();
            let pattern = [
                (1.25, 2.5),
                (f64::NAN, 3.0),
                (2.0, f64::NAN),
                (f64::NAN, f64::NAN),
                (-3.0, 0.5),
                (5.5, 1.0),
            ];
            let values = (0..count)
                .map(|i| pattern[i % pattern.len()])
                .collect::<Vec<_>>();
            let a = complex_array(&shape, &values, dtype);
            let input = batch.add_input(&a);
            for operation in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
                for axis in [None, Some(0), Some(-1), Some(1), Some(-3)] {
                    for keepdims in [false, true] {
                        for &ddof in if matches!(operation, "nanvar" | "nanstd") {
                            &[0, 1, 7][..]
                        } else {
                            &[0][..]
                        } {
                            batch.push(input, &a, operation, None, axis, keepdims, ddof);
                        }
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_nan_reduction_precision_and_pairwise_order_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let large = if dtype == DType::Complex64 { 1e8 } else { 1e16 };
        for len in [3, 4, 63, 64, 65, 127, 128, 129, 8191, 8192, 8193] {
            let pattern = [
                (large, -large),
                (1.001, 2.003),
                (f64::NAN, 7.0),
                (-large, large),
                (0.25, -0.125),
                (7.0, f64::NAN),
                (3.0, 1.0),
            ];
            let values = (0..2 * len)
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
fn complex_nan_reduction_infinities_and_signed_zeros_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for values in [
            vec![(-0.0, -0.0), (-0.0, 0.0)],
            vec![(f64::INFINITY, 1.0), (1.0, 2.0)],
            vec![(1.0, f64::INFINITY), (2.0, 3.0)],
            vec![(f64::INFINITY, f64::NAN), (2.0, 3.0)],
            vec![(f64::NAN, f64::INFINITY), (2.0, 3.0)],
            vec![(f64::NAN, 1.0), (1.0, f64::NAN)],
        ] {
            let a = complex_array(&[2], &values, dtype);
            let input = batch.add_input(&a);
            for operation in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
                for keepdims in [false, true] {
                    batch.push(input, &a, operation, None, None, keepdims, 0);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_casts_keep_logical_shape_and_use_both_components_for_truth() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let array = complex_array(
            &[2, 3],
            &[
                (1.75, 200.0),
                (-3.75, -100.0),
                (0.0, 4.0),
                (-0.0, -0.0),
                (0.0, f64::NAN),
                (0.0, f64::INFINITY),
            ],
            dtype,
        );
        let _guard = errstate(None, None, None, None, Some(FloatErrorMode::Raise));
        let integer = array.astype_with_casting(DType::I32, "unsafe").unwrap();
        assert_eq!(integer.shape(), &[2, 3]);
        assert_eq!(integer.values(), &[1.0, -3.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            integer.to_storage().unwrap(),
            ArrayStorage::I32(vec![1, -3, 0, 0, 0, 0])
        );
        let truth = array.astype(DType::Bool);
        assert_eq!(truth.shape(), &[2, 3]);
        assert_eq!(
            truth.to_storage().unwrap(),
            ArrayStorage::Bool(vec![true, true, true, false, true, true])
        );
        let real = array.astype(DType::F64);
        assert_eq!(real.shape(), &[2, 3]);
        assert_eq!(real.values(), &[1.75, -3.75, 0.0, -0.0, 0.0, 0.0]);
        assert_eq!(real.values()[3].to_bits(), (-0.0_f64).to_bits());
        let scalar = complex_array(&[], &[(3.75, -4.0)], dtype).astype(DType::I16);
        assert_eq!(scalar.shape(), &[] as &[usize]);
        assert_eq!(scalar.values(), &[3.0]);
    }
}

#[test]
fn complex_cast_matrix_matches_numpy_shapes_dtypes_and_exact_values() {
    let mut batch = OracleBatch::default();
    for source in [DType::Complex64, DType::Complex128] {
        for shape in [vec![], vec![2, 3], vec![0], vec![2, 0, 3]] {
            let count = shape.iter().product();
            let pattern = [
                (16_777_217.0, -16_777_219.0),
                (-257.9, 255.5),
                (255.9, -128.5),
                (-0.0, 0.0),
                (0.0, -0.25),
                (1.000_488_281_250_000_2, -1.000_488_281_250_000_2),
            ];
            let values = (0..count)
                .map(|index| pattern[index % pattern.len()])
                .collect::<Vec<_>>();
            let array = complex_array(&shape, &values, source);
            let input = batch.add_input(&array);
            for target in [
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
                if target != source {
                    batch.push_cast(input, &array, target);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn real_and_exact_integer_casts_to_complex_match_numpy_without_double_rounding() {
    let midpoint = (1_i64 << 62) + (1_i64 << 38);
    let arrays = [
        UFuncArray::new(vec![], vec![-0.0], DType::F64).unwrap(),
        UFuncArray::new(vec![2, 0, 3], Vec::new(), DType::F32).unwrap(),
        UFuncArray::new(vec![3], vec![1.001, -2.003, -0.0], DType::F16).unwrap(),
        UFuncArray::new(vec![2], vec![16_777_217.0, -16_777_219.0], DType::F32).unwrap(),
        UFuncArray::new(
            vec![2, 2],
            vec![1.25, -0.0, f64::NAN, f64::INFINITY],
            DType::F64,
        )
        .unwrap(),
        UFuncArray::from_storage(
            vec![2, 2],
            ArrayStorage::I64(vec![midpoint - 1, midpoint + 1, i64::MIN, i64::MAX]),
        )
        .unwrap(),
        UFuncArray::from_storage(
            vec![2, 2],
            ArrayStorage::U64(vec![midpoint as u64 - 1, midpoint as u64 + 1, 0, u64::MAX]),
        )
        .unwrap(),
        UFuncArray::from_storage(vec![2], ArrayStorage::Bool(vec![true, false])).unwrap(),
    ];
    let mut batch = OracleBatch::default();
    for array in arrays {
        let input = batch.add_input(&array);
        for target in [DType::Complex64, DType::Complex128] {
            batch.push_cast(input, &array, target);
        }
    }
    batch.assert_matches_numpy();
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn complex_integer_casts_keep_exact_sidecars_and_scalar_vector_boundaries() {
    let mut batch = OracleBatch::default();
    let two_to_63 = 9_223_372_036_854_775_808.0_f64;
    let two_to_64 = 18_446_744_073_709_551_616.0_f64;
    for source in [DType::Complex64, DType::Complex128] {
        for len in [0, 1, 3, 4, 5, 6, 7, 8, 9, 8191, 8192, 8193] {
            let pattern = [
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                -4_294_967_295.0,
                4_294_967_297.0,
                -1.9,
                two_to_63.next_down(),
                two_to_63,
                two_to_64.next_down(),
                two_to_64,
                -two_to_63,
            ];
            let values = (0..len)
                .map(|index| (pattern[index % pattern.len()], f64::NAN))
                .collect::<Vec<_>>();
            let array = complex_array(&[len], &values, source);
            let input = batch.add_input(&array);
            for target in [
                DType::I8,
                DType::I16,
                DType::I32,
                DType::I64,
                DType::U8,
                DType::U16,
                DType::U32,
                DType::U64,
            ] {
                batch.push_cast(input, &array, target);
            }
        }
    }
    batch.assert_matches_numpy();
    let array = complex_array(
        &[3],
        &[
            (-1.9, f64::NAN),
            (two_to_63, 7.0),
            (two_to_64.next_down(), 9.0),
        ],
        DType::Complex128,
    );
    assert_eq!(
        array.astype(DType::U64).to_storage().unwrap(),
        ArrayStorage::U64(vec![u64::MAX, 1_u64 << 63, u64::MAX - 2047])
    );
}

#[test]
fn complex_cast_invalid_policy_ignores_discarded_imaginary_values() {
    take_float_error_events();
    let _guard = errstate(None, None, None, None, Some(FloatErrorMode::Raise));
    for source in [DType::Complex64, DType::Complex128] {
        let array = complex_array(
            &[3],
            &[
                (1.9, f64::NAN),
                (-2.9, f64::INFINITY),
                (0.0, f64::NEG_INFINITY),
            ],
            source,
        );
        assert_eq!(
            array
                .astype_with_casting(DType::I32, "unsafe")
                .unwrap()
                .values(),
            &[1.0, -2.0, 0.0]
        );
        assert_eq!(
            array
                .astype_with_casting(DType::Bool, "unsafe")
                .unwrap()
                .values(),
            &[1.0, 1.0, 1.0]
        );
        let empty = complex_array(&[2, 0], &[], source);
        let output = empty.astype_with_casting(DType::U64, "unsafe").unwrap();
        assert_eq!(output.shape(), &[2, 0]);
        assert!(output.values().is_empty());
        let invalid = complex_array(&[2], &[(f64::NAN, 0.0), (f64::INFINITY, 0.0)], source);
        for target in [
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
        ] {
            assert!(matches!(
                invalid.astype_with_casting(target, "unsafe"),
                Err(UFuncError::FloatingPoint {
                    kind: FloatErrorKind::Invalid,
                    ..
                })
            ));
        }
    }
    assert!(take_float_error_events().is_empty());
}

#[test]
fn complex_cast_invalid_events_are_per_call_and_infallible_cast_stays_value_only() {
    let array = complex_array(
        &[4],
        &[
            (f64::NAN, 0.0),
            (f64::INFINITY, 1.0),
            (3.9, f64::NAN),
            (-4.9, 0.0),
        ],
        DType::Complex128,
    );
    for mode in [
        FloatErrorMode::Warn,
        FloatErrorMode::Print,
        FloatErrorMode::Log,
    ] {
        take_float_error_events();
        let _guard = errstate(None, None, None, None, Some(mode));
        let output = array.astype_with_casting(DType::I64, "unsafe").unwrap();
        assert_eq!(output.shape(), &[4]);
        assert_eq!(&output.values()[2..], &[3.0, -4.0]);
        let events = take_float_error_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, FloatErrorKind::Invalid);
        assert_eq!(events[0].mode, mode);
        assert_eq!(events[0].op, "cast");
        assert_eq!(events[0].message, "invalid value encountered in cast");
    }
    let expected = {
        let _guard = errstate(None, None, None, None, Some(FloatErrorMode::Ignore));
        array
            .astype_with_casting(DType::U64, "unsafe")
            .unwrap()
            .to_storage()
            .unwrap()
    };
    {
        let _guard = errstate(None, None, None, None, Some(FloatErrorMode::Raise));
        assert_eq!(array.astype(DType::U64).to_storage().unwrap(), expected);
    }
    assert!(take_float_error_events().is_empty());
}

#[test]
fn complex_cast_invalid_callback_runs_once_and_policy_rejection_precedes_status() {
    let array = complex_array(
        &[2],
        &[(f64::NAN, 0.0), (f64::INFINITY, 1.0)],
        DType::Complex64,
    );
    let sink = Arc::new(Mutex::new(Vec::new()));
    let callback_sink = Arc::clone(&sink);
    let previous = seterrcall(Some(Arc::new(move |event| {
        callback_sink.lock().unwrap().push(event);
    })));
    take_float_error_events();
    let result = {
        let _guard = errstate(None, None, None, None, Some(FloatErrorMode::Call));
        for policy in ["safe", "same_kind", "no", "equiv", "unknown"] {
            assert!(matches!(
                array.astype_with_casting(DType::I32, policy),
                Err(UFuncError::Msg(_))
            ));
        }
        assert!(sink.lock().unwrap().is_empty());
        array.astype_with_casting(DType::I32, "unsafe")
    };
    let _ = seterrcall(previous);
    result.unwrap();
    let events = sink.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, FloatErrorKind::Invalid);
    assert_eq!(events[0].mode, FloatErrorMode::Call);
    assert_eq!(events[0].op, "cast");
    assert!(take_float_error_events().is_empty());
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn complex_uint32_invalid_status_tracks_complex_layout_and_source_precision() {
    let _guard = errstate(None, None, None, None, Some(FloatErrorMode::Raise));
    for (source, len, index, invalid) in [
        (DType::Complex64, 4, 0, false),
        (DType::Complex64, 5, 0, true),
        (DType::Complex64, 5, 4, false),
        (DType::Complex128, 5, 0, false),
        (DType::Complex128, 6, 0, true),
        (DType::Complex128, 6, 4, false),
        (DType::Complex128, 8192, 8187, true),
        (DType::Complex128, 8192, 8188, false),
        (DType::Complex128, 8193, 8191, true),
        (DType::Complex128, 8193, 8192, false),
    ] {
        let mut values = vec![(0.0, f64::NAN); len];
        values[index].0 = -4_294_967_296.0;
        let array = complex_array(&[len], &values, source);
        let result = array.astype_with_casting(DType::U32, "unsafe");
        assert_eq!(
            result.is_err(),
            invalid,
            "{source:?}, length={len}, index={index}"
        );
        if let Ok(output) = result {
            assert_eq!(output.shape(), &[len]);
            assert!(output.values().iter().all(|&value| value == 0.0));
        }
    }
    for (source, invalid) in [(DType::Complex64, true), (DType::Complex128, false)] {
        let array = complex_array(&[1], &[(2_147_483_647.0, 0.0)], source);
        assert_eq!(
            array.astype_with_casting(DType::I32, "unsafe").is_err(),
            invalid
        );
    }
}

#[test]
fn complex_casts_reject_malformed_interleaved_layouts() {
    let _guard = errstate(None, None, None, None, Some(FloatErrorMode::Raise));
    for array in [
        UFuncArray::new(vec![], vec![f64::NAN], DType::Complex64).unwrap(),
        UFuncArray::new(vec![3], vec![f64::NAN; 3], DType::Complex128).unwrap(),
        UFuncArray::new(vec![2, 0], Vec::new(), DType::Complex64).unwrap(),
    ] {
        assert!(matches!(
            array.astype_with_casting(DType::I32, "unsafe"),
            Err(UFuncError::Msg(message)) if message.contains("interleaved")
        ));
    }
}

#[test]
fn complex_same_dtype_casts_preserve_values_after_policy_and_layout_validation() {
    take_float_error_events();
    let _guard = errstate(None, None, None, None, Some(FloatErrorMode::Raise));
    for dtype in [DType::Complex64, DType::Complex128] {
        let array = complex_array(
            &[2],
            &[
                (16_777_217.0, -16_777_219.0),
                (f64::from_bits(0x7ff8_0000_0000_0042), -0.0),
            ],
            dtype,
        );
        let expected_bits = array
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        for policy in ["no", "equiv", "safe", "same_kind", "unsafe"] {
            let output = array.astype_with_casting(dtype, policy).unwrap();
            assert_eq!(output.shape(), array.shape());
            assert_eq!(output.dtype(), dtype);
            assert_eq!(
                output
                    .values()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_bits,
                "{dtype:?}, casting={policy}: unchanged dtype must not requantize"
            );
        }
        assert!(matches!(
            array.astype_with_casting(dtype, "unknown"),
            Err(UFuncError::Msg(message)) if message.contains("rule 'unknown'")
        ));
        let malformed = UFuncArray::new(vec![3], vec![f64::NAN; 3], dtype).unwrap();
        assert!(matches!(
            malformed.astype_with_casting(dtype, "no"),
            Err(UFuncError::Msg(message)) if message.contains("interleaved")
        ));
        assert!(matches!(
            malformed.astype_with_casting(dtype, "unknown"),
            Err(UFuncError::Msg(message)) if message.contains("rule 'unknown'")
        ));
    }
    assert!(take_float_error_events().is_empty());
}

#[test]
fn narrowing_cast_error_modes_report_each_retained_component_kind_once() {
    let real = UFuncArray::new(
        vec![4],
        vec![1.0e300, -1.0e300, 1.0e-300, -1.0e-300],
        DType::F64,
    )
    .unwrap();
    let complex = complex_array(
        &[2],
        &[(1.0e300, 1.0e-300), (-1.0e300, -1.0e-300)],
        DType::Complex128,
    );
    for array in [&real, &complex] {
        for target in [DType::F16, DType::F32, DType::Complex64] {
            // The real target discards the complex fixture's tiny imaginary
            // components, so those casts report overflow alone.
            let expected = if array.dtype().is_complex() && !target.is_complex() {
                vec![FloatErrorKind::Over]
            } else {
                vec![FloatErrorKind::Over, FloatErrorKind::Under]
            };
            for mode in [
                FloatErrorMode::Ignore,
                FloatErrorMode::Warn,
                FloatErrorMode::Print,
                FloatErrorMode::Log,
            ] {
                take_float_error_events();
                let _guard = errstate(Some(mode), None, None, None, None);
                let result = array.astype_with_casting(target, "unsafe").unwrap();
                assert_eq!(result.dtype(), target);
                assert!(result.values().iter().any(|value| value.is_infinite()));
                let events = take_float_error_events();
                let kinds = events.iter().map(|event| event.kind).collect::<Vec<_>>();
                if mode == FloatErrorMode::Ignore {
                    assert!(kinds.is_empty());
                } else {
                    assert_eq!(kinds, expected, "{:?} -> {target:?}", array.dtype());
                    assert!(
                        events
                            .iter()
                            .all(|event| event.mode == mode && event.op == "cast")
                    );
                }
            }
            let sink = Arc::new(Mutex::new(Vec::new()));
            let callback_sink = Arc::clone(&sink);
            let previous = seterrcall(Some(Arc::new(move |event| {
                callback_sink.lock().unwrap().push(event);
            })));
            let result = {
                let _guard = errstate(Some(FloatErrorMode::Call), None, None, None, None);
                array.astype_with_casting(target, "unsafe")
            };
            let _ = seterrcall(previous);
            result.unwrap();
            assert_eq!(
                sink.lock()
                    .unwrap()
                    .iter()
                    .map(|event| event.kind)
                    .collect::<Vec<_>>(),
                expected
            );
            assert!(take_float_error_events().is_empty());
        }
    }
}

#[test]
fn narrowing_cast_raise_preserves_validation_precedence_and_value_only_api() {
    take_float_error_events();
    let _guard = errstate(Some(FloatErrorMode::Raise), None, None, None, None);
    for source in [DType::F64, DType::Complex128] {
        let array = if source.is_complex() {
            complex_array(&[2], &[(1.0e300, 0.0), (1.0e-300, 0.0)], source)
        } else {
            UFuncArray::new(vec![2], vec![1.0e300, 1.0e-300], source).unwrap()
        };
        for target in [DType::F16, DType::F32, DType::Complex64] {
            for policy in ["safe", "no", "equiv", "unknown"] {
                assert!(matches!(
                    array.astype_with_casting(target, policy),
                    Err(UFuncError::Msg(_))
                ));
            }
            assert!(matches!(
                array.astype_with_casting(target, "unsafe"),
                Err(UFuncError::FloatingPoint {
                    kind: FloatErrorKind::Over,
                    ..
                })
            ));
            let _overflow_ignored = errstate(None, None, Some(FloatErrorMode::Ignore), None, None);
            assert!(matches!(
                array.astype_with_casting(target, "unsafe"),
                Err(UFuncError::FloatingPoint {
                    kind: FloatErrorKind::Under,
                    ..
                })
            ));
            let value_only = array.astype(target);
            assert_eq!(value_only.dtype(), target);
            assert!(value_only.values()[0].is_infinite());
        }
    }
    let malformed = UFuncArray::new(vec![3], vec![1.0e300; 3], DType::Complex128).unwrap();
    assert!(matches!(
        malformed.astype_with_casting(DType::Complex64, "unsafe"),
        Err(UFuncError::Msg(message)) if message.contains("interleaved")
    ));
    assert!(take_float_error_events().is_empty());
}

#[test]
fn narrowing_cast_status_uses_source_precision_and_ignores_nonfinite_inputs() {
    take_float_error_events();
    let _guard = errstate(Some(FloatErrorMode::Raise), None, None, None, None);
    for source in [DType::F16, DType::F32, DType::Complex64] {
        // These unquantized working values are already infinity/zero in the
        // declared source dtype. Casting them again must not report new flags.
        let array = if source.is_complex() {
            complex_array(&[2], &[(1.0e300, 1.0e-300), (-1.0e300, -1.0e-300)], source)
        } else {
            UFuncArray::new(
                vec![4],
                vec![1.0e300, -1.0e300, 1.0e-300, -1.0e-300],
                source,
            )
            .unwrap()
        };
        for target in [DType::F16, DType::F32, DType::Complex64] {
            array.astype_with_casting(target, "unsafe").unwrap();
        }
    }
    for source in [DType::F64, DType::Complex128] {
        let array = if source.is_complex() {
            complex_array(
                &[2],
                &[(f64::NAN, f64::INFINITY), (f64::NEG_INFINITY, f64::NAN)],
                source,
            )
        } else {
            UFuncArray::new(
                vec![3],
                vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
                source,
            )
            .unwrap()
        };
        for target in [DType::F16, DType::F32, DType::Complex64] {
            array.astype_with_casting(target, "unsafe").unwrap();
        }
    }
    let discarded = complex_array(&[2], &[(1.0, 1.0e300), (-0.0, 1.0e-300)], DType::Complex128);
    for target in [DType::F16, DType::F32] {
        let output = discarded.astype_with_casting(target, "unsafe").unwrap();
        assert_eq!(output.values(), &[1.0, -0.0]);
        assert!(output.values()[1].is_sign_negative());
    }
    assert!(take_float_error_events().is_empty());
}

#[test]
fn narrowing_integer_to_half_reports_overflow_without_losing_wide_input_bits() {
    let array = UFuncArray::from_storage(
        vec![3],
        ArrayStorage::U64(vec![u64::MAX, (1_u64 << 63) + 1, 65_504]),
    )
    .unwrap();
    take_float_error_events();
    {
        let _guard = errstate(Some(FloatErrorMode::Warn), None, None, None, None);
        let output = array.astype_with_casting(DType::F16, "same_kind").unwrap();
        assert_eq!(output.values(), &[f64::INFINITY, f64::INFINITY, 65_504.0]);
        let events = take_float_error_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, FloatErrorKind::Over);
    }
    let _guard = errstate(Some(FloatErrorMode::Raise), None, None, None, None);
    assert!(matches!(
        array.astype_with_casting(DType::F16, "same_kind"),
        Err(UFuncError::FloatingPoint {
            kind: FloatErrorKind::Over,
            ..
        })
    ));
    assert_eq!(
        array
            .astype_with_casting(DType::U64, "no")
            .unwrap()
            .to_storage()
            .unwrap(),
        ArrayStorage::U64(vec![u64::MAX, (1_u64 << 63) + 1, 65_504])
    );
    // Raw integer working cells retain the same conversions as astype, even
    // when no exact-integer sidecar was supplied to the constructor.
    for source in [DType::I64, DType::U64] {
        let raw = UFuncArray::new(vec![1], vec![2.0_f64.powi(54)], source).unwrap();
        for target in [DType::F32, DType::F64] {
            assert_eq!(
                raw.astype_with_casting(target, "unsafe").unwrap().values(),
                raw.astype(target).values()
            );
        }
        assert!(matches!(
            raw.astype_with_casting(DType::F16, "unsafe"),
            Err(UFuncError::FloatingPoint {
                kind: FloatErrorKind::Over,
                ..
            })
        ));
    }
    let empty = complex_array(&[2, 0], &[], DType::Complex128);
    assert!(
        empty
            .astype_with_casting(DType::F16, "unsafe")
            .unwrap()
            .values()
            .is_empty()
    );
    assert!(take_float_error_events().is_empty());
}

#[test]
fn narrowing_cast_flags_and_target_bytes_match_numpy() {
    let _guard = errstate(Some(FloatErrorMode::Warn), None, None, None, None);
    let mut batch = OracleBatch::default();
    let mut expected_cases = Vec::new();
    let mut actual = Vec::new();
    let half_tiny = f64::from(fnp_dtype::f16::MIN_POSITIVE);
    let single_tiny = f64::from(f32::MIN_POSITIVE);
    let single_step = f64::from(f32::from_bits(1));
    let mut values = vec![
        0.0,
        -0.0,
        1.0,
        1.0e300,
        -1.0e300,
        1.0e-300,
        -1.0e-300,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        65_504.0,
        65_519.0,
        65_520.0,
        f64::from(f32::MAX),
        f64::from(f32::MAX) + 2.0_f64.powi(103),
    ];
    for fraction in [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5] {
        values.push(half_tiny - 2.0_f64.powi(-24) * fraction);
        values.push(single_tiny - single_step * fraction);
    }
    for source in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for &value in &values {
            // Scalar cases isolate each boundary instead of letting one
            // overflow/underflow elsewhere in a vector hide a missing flag.
            let array = if source.is_complex() {
                // The JSON fixture encodes NaN as text, which does not retain
                // its sign. Keep both source NaNs positive in this byte test.
                let imaginary = if value.is_nan() { value } else { -value };
                complex_array(&[1], &[(value, imaginary)], source)
            } else {
                UFuncArray::new(vec![1], vec![value], source).unwrap()
            };
            let input = batch.add_input(&array);
            for target in [DType::F16, DType::F32, DType::Complex64] {
                if source == target {
                    continue;
                }
                take_float_error_events();
                let output = array.astype_with_casting(target, "unsafe").unwrap();
                let flags = take_float_error_events()
                    .into_iter()
                    .map(|event| event.message)
                    .collect::<Vec<_>>();
                let bytes = match target {
                    DType::F16 => output
                        .values()
                        .iter()
                        .flat_map(|&value| fnp_dtype::f16_from_f64(value).to_bits().to_le_bytes())
                        .collect::<Vec<_>>(),
                    DType::F32 | DType::Complex64 => output
                        .values()
                        .iter()
                        .flat_map(|&value| (value as f32).to_bits().to_le_bytes())
                        .collect::<Vec<_>>(),
                    _ => unreachable!("narrowing cast target"),
                };
                expected_cases.push(json!({"input": input, "dtype": numpy_dtype(target)}));
                actual.push(json!({"flags": flags, "bytes": bytes}));
            }
        }
    }
    let script = r#"
import json, sys, warnings
import numpy as np
payload = json.load(sys.stdin)
inputs = []
with np.errstate(all='ignore'), warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for item in payload['inputs']:
        dtype = np.dtype(item['dtype'])
        if dtype.kind == 'c':
            values = [complex(float(pair[0]), float(pair[1])) for pair in item['values']]
        else:
            values = [float(value) for value in item['values']]
        inputs.append(np.array(values, dtype=dtype).reshape(item['shape']))
rows = []
for case in payload['cases']:
    with np.errstate(all='warn'), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        output = inputs[case['input']].astype(case['dtype'])
    flags = [str(item.message) for item in caught
             if issubclass(item.category, RuntimeWarning)
             and not issubclass(item.category, np.exceptions.ComplexWarning)]
    rows.append(dict(flags=flags, bytes=list(output.astype(output.dtype.newbyteorder('<')).tobytes())))
print(json.dumps(rows))
"#;
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let mut child = Command::new(python)
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch NumPy narrowing cast oracle");
    let mut stdin = child.stdin.take().expect("oracle stdin");
    serde_json::to_writer(
        &mut stdin,
        &json!({"inputs": batch.inputs, "cases": expected_cases}),
    )
    .expect("write narrowing cast cases");
    drop(stdin);
    let output = child
        .wait_with_output()
        .expect("collect narrowing cast oracle");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<Value> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
    assert_eq!(actual.len(), expected.len());
    for ((case, actual), expected) in expected_cases.iter().zip(actual).zip(expected) {
        assert_eq!(
            actual, expected,
            "{case}: narrowing flags and exact target bytes"
        );
    }
}

fn assert_typed_complex_storage_bits(actual: &ArrayStorage, expected: &ArrayStorage) {
    let bits = |storage: &ArrayStorage| -> Vec<[u64; 2]> {
        match storage {
            ArrayStorage::Complex64(values) => values
                .iter()
                .map(|&(real, imaginary)| {
                    [u64::from(real.to_bits()), u64::from(imaginary.to_bits())]
                })
                .collect(),
            ArrayStorage::Complex128(values) => values
                .iter()
                .map(|&(real, imaginary)| [real.to_bits(), imaginary.to_bits()])
                .collect(),
            _ => panic!("expected typed complex storage"),
        }
    };
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(bits(actual), bits(expected));
}

#[test]
fn typed_complex_storage_round_trips_logical_shapes_and_component_bits() {
    let values64 = [
        (-0.0_f32, 0.0_f32),
        (f32::from_bits(1), -f32::from_bits(1)),
        (f32::from_bits(0x7fc0_1234), f32::from_bits(0xffc0_5678)),
        (f32::INFINITY, f32::NEG_INFINITY),
        (f32::MAX, f32::MIN_POSITIVE),
        (1.25, -16_777_216.0),
    ];
    let values128 = [
        (-0.0_f64, 0.0_f64),
        (f64::from_bits(1), -f64::from_bits(1)),
        (
            f64::from_bits(0x7ff8_0000_0000_1234),
            f64::from_bits(0xfff8_0000_0000_5678),
        ),
        (f64::INFINITY, f64::NEG_INFINITY),
        (f64::MAX, f64::MIN_POSITIVE),
        (1.0 + f64::EPSILON, -16_777_217.0),
    ];
    for shape in [
        vec![],
        vec![1],
        vec![2],
        vec![2, 2],
        vec![2, 1, 3],
        vec![0],
        vec![2, 0, 3],
        vec![usize::MAX, 0],
    ] {
        let count = shape.iter().product::<usize>();
        for storage in [
            ArrayStorage::Complex64(values64[..count].to_vec()),
            ArrayStorage::Complex128(values128[..count].to_vec()),
        ] {
            let mut physical_shape = shape.clone();
            physical_shape.push(2);
            for array in [
                UFuncArray::from_storage(shape.clone(), storage.clone()).unwrap(),
                UFuncArray::from_storage_with_dtype(
                    shape.clone(),
                    storage.clone(),
                    storage.dtype(),
                )
                .unwrap(),
            ] {
                assert_eq!(array.shape(), physical_shape);
                assert_eq!(array.dtype(), storage.dtype());
                assert_eq!(array.values().len(), count * 2);
                assert!(!array.has_integer_sidecar());
                assert_typed_complex_storage_bits(&array.to_storage().unwrap(), &storage);
            }
        }
    }
}

#[test]
fn typed_complex_storage_exports_raw_cells_at_declared_component_precision() {
    let values = [
        (16_777_217.0, 1.0 + 2.0_f64.powi(-24)),
        (-0.0, -f64::from(f32::from_bits(1)) * 0.75),
        (f64::from(f32::MAX) * 2.0, f64::NAN),
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        let array = complex_array(&[3], &values, dtype);
        let expected = if dtype == DType::Complex64 {
            ArrayStorage::Complex64(
                values
                    .iter()
                    .map(|&(real, imaginary)| (real as f32, imaginary as f32))
                    .collect(),
            )
        } else {
            ArrayStorage::Complex128(values.to_vec())
        };
        let storage = array.to_storage().unwrap();
        assert_typed_complex_storage_bits(&storage, &expected);
        let restored = UFuncArray::from_storage(vec![3], storage).unwrap();
        assert_typed_complex_storage_bits(&restored.to_storage().unwrap(), &expected);
        assert_eq!(restored.shape(), &[3, 2]);
    }
}

#[test]
fn typed_complex_storage_casts_match_numpy_without_losing_source_precision() {
    let mut batch = OracleBatch::default();
    let values = [
        (-0.0, 0.0),
        (0.0, -0.0),
        (0.0, 2.0),
        (-0.0, -3.0),
        (1.5, -4.5),
        (-1.75, 2.25),
        (16_777_217.0, 0.1),
        (65_520.0, 65_519.0),
        (
            f64::from(f32::MAX) * 2.0,
            -f64::from(f32::MIN_POSITIVE) * 0.5,
        ),
        (2.0_f64.powi(54), 0.0),
        (i64::MIN as f64, 0.0),
        (2.0_f64.powi(64), 0.0),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY),
        (f64::NEG_INFINITY, f64::INFINITY),
    ];
    let targets = [
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
    for source in [DType::Complex64, DType::Complex128] {
        for shape in [vec![16], vec![2, 8], vec![2, 2, 4]] {
            let array = complex_array(&shape, &values, source);
            let storage = if source == DType::Complex64 {
                ArrayStorage::Complex64(
                    values
                        .iter()
                        .map(|&(real, imaginary)| (real as f32, imaginary as f32))
                        .collect(),
                )
            } else {
                ArrayStorage::Complex128(values.to_vec())
            };
            let input = batch.add_input(&array);
            for target in targets {
                batch.push_storage_cast(input, &shape, &storage, target);
            }
        }
        for value in values {
            let array = complex_array(&[], &[value], source);
            let storage = if source == DType::Complex64 {
                ArrayStorage::Complex64(vec![(value.0 as f32, value.1 as f32)])
            } else {
                ArrayStorage::Complex128(vec![value])
            };
            let input = batch.add_input(&array);
            for target in targets {
                batch.push_storage_cast(input, &[], &storage, target);
            }
        }
    }

    // These integers straddle an f32 midpoint but collapse to the same f64.
    // Real -> complex64 must convert the original typed integer exactly once.
    let midpoint = (1_i64 << 54) + (1_i64 << 30);
    for storage in [
        ArrayStorage::Bool(vec![false, true, true, false]),
        ArrayStorage::I8(vec![i8::MIN, -1, 0, i8::MAX]),
        ArrayStorage::I16(vec![i16::MIN, -1, 0, i16::MAX]),
        ArrayStorage::I32(vec![i32::MIN, -1, 0, i32::MAX]),
        ArrayStorage::I64(vec![
            -midpoint - 1,
            -midpoint + 1,
            midpoint - 1,
            midpoint + 1,
        ]),
        ArrayStorage::U8(vec![0, 1, 2, u8::MAX]),
        ArrayStorage::U16(vec![0, 1, 2, u16::MAX]),
        ArrayStorage::U32(vec![0, 1, 2, u32::MAX]),
        ArrayStorage::U64(vec![midpoint as u64 - 1, midpoint as u64 + 1, 0, u64::MAX]),
        ArrayStorage::F16(
            [-0.0, 0.1, 2.0_f64.powi(-24), f64::INFINITY]
                .into_iter()
                .map(fnp_dtype::f16_from_f64)
                .collect(),
        ),
        ArrayStorage::F32(vec![-0.0, 0.1, f32::from_bits(1), f32::INFINITY]),
        ArrayStorage::F64(vec![-0.0, 1.0 + 2.0_f64.powi(-24), 1e-46, f64::NAN]),
    ] {
        for shape in [vec![4], vec![2, 2], vec![1, 4, 1]] {
            let array = UFuncArray::from_storage(shape.clone(), storage.clone()).unwrap();
            let input = batch.add_input(&array);
            for target in [DType::Complex64, DType::Complex128] {
                batch.push_storage_cast(input, &shape, &storage, target);
            }
        }
    }
    for source in [DType::Complex64, DType::Complex128] {
        let storage = if source == DType::Complex64 {
            ArrayStorage::Complex64(vec![])
        } else {
            ArrayStorage::Complex128(vec![])
        };
        let array = complex_array(&[2, 0, 3], &[], source);
        let input = batch.add_input(&array);
        for target in targets {
            batch.push_storage_cast(input, &[2, 0, 3], &storage, target);
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn typed_complex_storage_rejects_invalid_shapes_and_unsupported_casts() {
    for storage in [
        ArrayStorage::Complex64(vec![(1.0, 2.0)]),
        ArrayStorage::Complex128(vec![(1.0, 2.0)]),
    ] {
        for shape in [vec![2], vec![1, 2], vec![0], vec![1, 0]] {
            assert!(matches!(
                UFuncArray::from_storage(shape.clone(), storage.clone()),
                Err(UFuncError::InvalidInputLength { actual: 1, .. })
            ));
            for target in [DType::Bool, DType::F64, DType::Complex64, DType::Complex128] {
                assert!(matches!(
                    UFuncArray::from_storage_with_dtype(shape.clone(), storage.clone(), target),
                    Err(UFuncError::InvalidInputLength { actual: 1, .. })
                ));
            }
        }
        for target in [
            DType::DateTime64,
            DType::TimeDelta64,
            DType::Str,
            DType::Structured,
        ] {
            assert!(UFuncArray::from_storage_with_dtype(vec![], storage.clone(), target).is_err());
        }
    }
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [
            vec![],
            vec![1],
            vec![3],
            vec![1, 3],
            vec![2, 1],
            vec![0],
            vec![2, 0],
        ] {
            let count = shape.iter().product::<usize>();
            let malformed = UFuncArray::new(shape, vec![0.0; count], dtype).unwrap();
            assert!(matches!(
                malformed.to_storage(),
                Err(UFuncError::Msg(message)) if message.contains("interleaved")
            ));
        }
        assert!(
            UFuncArray::from_storage_with_dtype(
                vec![1],
                ArrayStorage::String(vec!["1+2j".into()]),
                dtype,
            )
            .is_err()
        );
        assert!(matches!(
            UFuncArray::from_storage_with_dtype(
                vec![usize::MAX, 2],
                ArrayStorage::F64(vec![]),
                dtype
            ),
            Err(UFuncError::Shape(_))
        ));
    }
    assert!(matches!(
        UFuncArray::from_storage(vec![usize::MAX, 2], ArrayStorage::Complex128(vec![])),
        Err(UFuncError::Shape(_))
    ));
}

#[test]
fn typed_complex_storage_preserves_real_sidecars_and_value_only_cast_behavior() {
    for storage in [
        ArrayStorage::I64(vec![i64::MIN, i64::MAX]),
        ArrayStorage::U64(vec![0, u64::MAX]),
    ] {
        let imported = UFuncArray::from_storage(vec![2], storage.clone()).unwrap();
        let same_dtype =
            UFuncArray::from_storage_with_dtype(vec![2], storage.clone(), storage.dtype()).unwrap();
        for array in [imported, same_dtype] {
            assert!(array.has_integer_sidecar());
            assert_eq!(array.to_storage().unwrap(), storage);
            assert_eq!(array.shape(), &[2]);
        }
        // The established real-only bridge remains conservative about these
        // wide inputs; accepting complex targets must not change that contract.
        assert!(UFuncArray::from_storage_with_dtype(vec![2], storage, DType::F64).is_err());
    }
    let raw_integer = UFuncArray::new(vec![1], vec![2.0_f64.powi(54)], DType::I64).unwrap();
    assert!(raw_integer.to_storage().is_err());

    let _guard = errstate(None, None, None, None, Some(FloatErrorMode::Raise));
    take_float_error_events();
    let narrowed = UFuncArray::from_storage_with_dtype(
        vec![1],
        ArrayStorage::Complex128(vec![(1e100, 1e-100)]),
        DType::Complex64,
    )
    .unwrap();
    assert_eq!(narrowed.values(), &[f64::INFINITY, 0.0]);
    let integer = UFuncArray::from_storage_with_dtype(
        vec![],
        ArrayStorage::Complex128(vec![(f64::NAN, 1.0)]),
        DType::I64,
    )
    .unwrap();
    assert_eq!(integer.shape(), &[] as &[usize]);
    assert!(take_float_error_events().is_empty());
}

#[test]
fn typed_complex_storage_round_trips_reduction_and_cast_results() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let pairs = [(1.0, 2.0), (3.0, -4.0), (-5.0, 6.0), (7.0, 8.0)];
        let storage = if dtype == DType::Complex64 {
            ArrayStorage::Complex64(
                pairs
                    .iter()
                    .map(|&(re, im)| (re as f32, im as f32))
                    .collect(),
            )
        } else {
            ArrayStorage::Complex128(pairs.to_vec())
        };
        let array = UFuncArray::from_storage(vec![2, 2], storage).unwrap();
        let sum = array.reduce_sum(Some(0), true).unwrap();
        assert_eq!(sum.shape(), &[1, 2, 2]);
        assert_eq!(sum.values(), &[-4.0, 8.0, 10.0, 4.0]);
        let restored = UFuncArray::from_storage(vec![1, 2], sum.to_storage().unwrap()).unwrap();
        assert_eq!(restored, sum);
        let real = UFuncArray::from_storage_with_dtype(
            vec![1, 2],
            restored.to_storage().unwrap(),
            DType::F64,
        )
        .unwrap();
        assert_eq!(real.shape(), &[1, 2]);
        assert_eq!(real.values(), &[-4.0, 10.0]);
        let variance = array.reduce_var(Some(-1), false, 0).unwrap();
        assert_eq!(variance.shape(), &[2]);
        assert_eq!(variance.values(), &[10.0, 37.0]);
        assert_eq!(
            UFuncArray::from_storage(vec![2], variance.to_storage().unwrap()).unwrap(),
            variance,
        );
    }
}
