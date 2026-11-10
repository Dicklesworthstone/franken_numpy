//! Joint-axis complex reductions must follow logical shapes and NumPy's loop order.

use fnp_dtype::DType;
use fnp_ufunc::{UFuncArray, UFuncError, where_nonzero};
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
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
        DType::F32 => "float32",
        DType::F64 => "float64",
        _ => panic!("unexpected complex reduction dtype {dtype:?}"),
    }
}

fn reduce_tuple(
    input: &UFuncArray,
    op: &str,
    axes: &[isize],
    keepdims: bool,
    ddof: usize,
) -> Result<UFuncArray, UFuncError> {
    match op {
        "sum" => input.reduce_sum_axes(axes, keepdims),
        "prod" => input.reduce_prod_axes(axes, keepdims),
        "mean" => input.reduce_mean_axes(axes, keepdims),
        "var" => input.reduce_var_axes(axes, keepdims, ddof),
        "std" => input.reduce_std_axes(axes, keepdims, ddof),
        "nansum" => input.nansum_axes(axes, keepdims),
        "nanprod" => input.nanprod_axes(axes, keepdims),
        "nanmean" => input.nanmean_axes(axes, keepdims),
        "nanvar" => input.nanvar_axes(axes, keepdims, ddof),
        "nanstd" => input.nanstd_axes(axes, keepdims, ddof),
        "any" => input.any_axes(axes),
        "all" => input.all_axes(axes),
        "count_nonzero" => input.count_nonzero_axes(axes, keepdims),
        _ => unreachable!("tuple reduction operation"),
    }
}

fn axis_subsets(ndim: usize) -> Vec<Vec<isize>> {
    let mut result = Vec::new();
    for mask in 0..1usize << ndim {
        let axes: Vec<isize> = (0..ndim)
            .filter(|axis| mask & (1 << axis) != 0)
            .map(|axis| axis as isize)
            .collect();
        result.push(axes.clone());
        if axes.len() > 1 {
            result.push(axes.iter().rev().copied().collect());
        }
        if !axes.is_empty() {
            result.push(axes.iter().map(|&axis| axis - ndim as isize).collect());
        }
    }
    result
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
        assert_eq!(shape.pop(), Some(2));
        let input = self.inputs.len();
        self.inputs.push(json!({
            "shape": shape,
            "dtype": dtype_name(array.dtype()),
            "values": array.values().as_chunks::<2>().0.iter()
                .map(|parts| [parts[0].to_string(), parts[1].to_string()])
                .collect::<Vec<_>>(),
        }));
        input
    }

    fn push(
        &mut self,
        input: usize,
        array: &UFuncArray,
        op: &str,
        axes: &[isize],
        keepdims: bool,
        ddof: usize,
    ) {
        self.cases.push(json!({
            "input": input, "op": op, "axes": axes,
            "keepdims": keepdims, "ddof": ddof,
        }));
        self.actual
            .push(reduce_tuple(array, op, axes, keepdims, ddof));
    }

    fn push_single_product(
        &mut self,
        input: usize,
        array: &UFuncArray,
        axis: Option<isize>,
        keepdims: bool,
        dtype: Option<DType>,
    ) {
        self.cases.push(json!({
            "input": input, "op": "prod", "axis": axis,
            "keepdims": keepdims, "dtype": dtype.map(dtype_name),
        }));
        self.actual.push(match dtype {
            Some(dtype) => array.reduce_prod_with_dtype(axis, keepdims, dtype),
            None => array.reduce_prod(axis, keepdims),
        });
    }

    fn push_with_dtype(
        &mut self,
        input: usize,
        array: &UFuncArray,
        operation: (&str, usize),
        axes: &[isize],
        keepdims: bool,
        dtype: DType,
    ) {
        let (op, ddof) = operation;
        self.cases.push(json!({
            "input": input, "op": op, "axes": axes,
            "keepdims": keepdims, "ddof": ddof, "dtype": dtype_name(dtype),
        }));
        self.actual.push(match op {
            "sum" => array.reduce_sum_axes_with_dtype(axes, keepdims, dtype),
            "prod" => array.reduce_prod_axes_with_dtype(axes, keepdims, dtype),
            "mean" => array.reduce_mean_axes_with_dtype(axes, keepdims, dtype),
            "var" => array.reduce_var_axes_with_dtype(axes, keepdims, ddof, dtype),
            "std" => array.reduce_std_axes_with_dtype(axes, keepdims, ddof, dtype),
            "nansum" => array.nansum_axes_with_dtype(axes, keepdims, dtype),
            "nanprod" => array.nanprod_axes_with_dtype(axes, keepdims, dtype),
            "nanmean" => array.nanmean_axes_with_dtype(axes, keepdims, dtype),
            "nanvar" => array.nanvar_axes_with_dtype(axes, keepdims, ddof, dtype),
            "nanstd" => array.nanstd_axes_with_dtype(axes, keepdims, ddof, dtype),
            _ => unreachable!("typed complex tuple operation"),
        });
    }

    fn push_truth_single(
        &mut self,
        input: usize,
        array: &UFuncArray,
        op: &str,
        axis: Option<isize>,
        keepdims: bool,
    ) {
        self.cases.push(json!({
            "input": input, "op": op, "axis": axis, "keepdims": keepdims,
        }));
        self.actual.push(match op {
            "any" => array.any(axis),
            "all" => array.all(axis),
            "count_nonzero" => array.count_nonzero(axis, keepdims),
            _ => unreachable!("truth reduction operation"),
        });
    }

    fn push_truth_indices(&mut self, input: usize, array: &UFuncArray) {
        for (op, native, result) in [
            ("flatnonzero", "nonzero", array.nonzero()),
            ("flatnonzero", "flatnonzero", array.flatnonzero()),
            ("argwhere", "argwhere", array.argwhere()),
        ] {
            self.cases
                .push(json!({"input": input, "op": op, "native": native}));
            self.actual.push(Ok(result));
        }
        match where_nonzero(array) {
            Ok(coordinates) => {
                for (axis, values) in coordinates.into_iter().enumerate() {
                    self.cases.push(json!({
                        "input": input, "op": "nonzero", "coordinate": axis,
                    }));
                    self.actual.push(Ok(values));
                }
            }
            Err(error) => {
                self.cases.push(json!({
                    "input": input, "op": "nonzero", "coordinate": 0,
                }));
                self.actual.push(Err(error));
            }
        }
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np

warnings.simplefilter('ignore')
payload = json.load(sys.stdin)
inputs = []
for item in payload['inputs']:
    a = np.empty(len(item['values']), dtype=item['dtype'])
    a.real = [float(pair[0]) for pair in item['values']]
    a.imag = [float(pair[1]) for pair in item['values']]
    inputs.append(a.reshape(item['shape']))
rows = []
for case in payload['cases']:
    try:
        if case['op'] in ('flatnonzero', 'argwhere', 'nonzero'):
            result = getattr(np, case['op'])(inputs[case['input']])
            if 'coordinate' in case:
                result = result[case['coordinate']]
            result = np.asarray(result)
        else:
            axis = tuple(case['axes']) if 'axes' in case else case['axis']
            kw = dict(axis=axis, keepdims=case['keepdims'])
            if case.get('dtype') is not None:
                kw['dtype'] = case['dtype']
            if case['op'] in ('var', 'std', 'nanvar', 'nanstd'):
                kw['ddof'] = case['ddof']
            result = np.asarray(getattr(np, case['op'])(inputs[case['input']], **kw))
        if result.dtype.kind == 'c':
            parts = np.empty(result.size * 2, dtype=np.float64)
            parts[0::2] = result.real.ravel()
            parts[1::2] = result.imag.ravel()
        else:
            parts = result.astype(np.float64).ravel()
        integers = [str(int(x)) for x in result.flat] if result.dtype.kind in 'biu' else None
        rows.append(dict(shape=list(result.shape), dtype=str(result.dtype),
                         bits=parts.view(np.uint64).tolist(), integers=integers))
    except (TypeError, ValueError, IndexError) as error:
        rows.append(dict(error=type(error).__name__, axis=getattr(error, 'axis', None),
                         ndim=getattr(error, 'ndim', None)))
print(json.dumps(rows))
"#;
        let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(python)
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("launch live NumPy complex tuple oracle");
        let mut stdin = child.stdin.take().expect("oracle stdin");
        serde_json::to_writer(
            &mut stdin,
            &json!({"inputs": self.inputs, "cases": self.cases}),
        )
        .expect("write tuple oracle inputs");
        drop(stdin);
        let output = child
            .wait_with_output()
            .expect("collect tuple oracle results");
        assert!(
            output.status.success(),
            "NumPy complex tuple oracle: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
        assert_eq!(expected.len(), self.actual.len());
        for ((case, actual), expected) in self.cases.iter().zip(self.actual).zip(expected) {
            if expected["error"] == "AxisError" {
                assert!(
                    matches!(actual, Err(UFuncError::AxisOutOfBounds { axis, ndim })
                        if json!(axis) == expected["axis"] && json!(ndim) == expected["ndim"]),
                    "{case}: expected logical-axis error {expected}, got {actual:?}"
                );
                continue;
            }
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
                assert_eq!(shape.pop(), Some(2), "{case}: complex component dimension");
            }
            assert_eq!(json!(shape), expected["shape"], "{case}: shape");
            assert_eq!(
                dtype_name(actual.dtype()),
                expected["dtype"],
                "{case}: dtype"
            );
            if actual.dtype().is_integer() || actual.dtype() == DType::Bool {
                let storage = actual.to_storage().expect("exact integer result storage");
                let integers: Vec<String> = (0..storage.len())
                    .map(|index| storage.get_i128(index).unwrap().to_string())
                    .collect();
                assert_eq!(
                    json!(integers),
                    expected["integers"],
                    "{case}: exact integers"
                );
            }
            let bits = expected["bits"].as_array().expect("result bits");
            assert_eq!(actual.values().len(), bits.len(), "{case}: length");
            for (index, (&got, expected)) in actual.values().iter().zip(bits).enumerate() {
                let expected = f64::from_bits(expected.as_u64().expect("float bits"));
                if expected.is_nan() {
                    assert!(
                        got.is_nan(),
                        "{case}: element {index} expected NaN, got {got}"
                    );
                } else {
                    assert_eq!(
                        got.to_bits(),
                        expected.to_bits(),
                        "{case}: element {index}: got {got}, expected {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn complex_nan_tuple_masks_counts_and_real_variance_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        // Each NaN in either component excludes its entire logical element.
        // The retained middle axis includes an all-NaN lane and unequal counts.
        let array = complex_array(
            &[2, 3, 2],
            &[
                (1.25, 2.5),
                (f64::NAN, 100.0),
                (f64::NAN, 4.0),
                (5.0, f64::NAN),
                (-0.0, -0.0),
                (3.75, 1.5),
                (100.0, f64::NAN),
                (8.5, -2.0),
                (f64::NAN, f64::NAN),
                (1.0, f64::NAN),
                (2.5, 3.0),
                (f64::NAN, 2.0),
            ],
            dtype,
        );
        let input = batch.add_input(&array);
        for axes in [
            vec![],
            vec![0, 2],
            vec![2, 0],
            vec![-1, -3],
            vec![0, 1],
            vec![1, 2],
            vec![0, 1, 2],
        ] {
            for keepdims in [false, true] {
                for op in ["nansum", "nanprod", "nanmean"] {
                    batch.push(input, &array, op, &axes, keepdims, 0);
                }
                for ddof in [0, 1, 2, 12] {
                    for op in ["nanvar", "nanstd"] {
                        batch.push(input, &array, op, &axes, keepdims, ddof);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_nan_tuple_explicit_dtypes_and_statistical_rejections_match_numpy() {
    let mut batch = OracleBatch::default();
    for source in [DType::Complex64, DType::Complex128] {
        for has_nan in [false, true] {
            let values = (0..12)
                .map(|i| {
                    let re = [0.0, 1.25, -0.0, -3.0, 2.75, 1.0004882812500002][i % 6];
                    let im = [1.0, -2.5, 3.0, -0.0, 4.0][i % 5];
                    if has_nan && i % 4 == 1 {
                        (f64::NAN, im)
                    } else if has_nan && i % 4 == 2 {
                        (re, f64::NAN)
                    } else {
                        (re, im)
                    }
                })
                .collect::<Vec<_>>();
            let array = complex_array(&[2, 3, 2], &values, source);
            let input = batch.add_input(&array);
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
                DType::Complex64,
                DType::Complex128,
            ] {
                for axes in [vec![], vec![0, 2], vec![0, 1, 2]] {
                    for keepdims in [false, true] {
                        for op in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
                            batch.push_with_dtype(input, &array, (op, 1), &axes, keepdims, dtype);
                        }
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_nan_tuple_empty_scalar_nonfinite_and_axis_errors_match_numpy() {
    let mut batch = OracleBatch::default();
    for source in [DType::Complex64, DType::Complex128] {
        for shape in [vec![], vec![0], vec![2, 0, 3], vec![2, 3, 0], vec![2, 2]] {
            for pattern in [
                [(f64::NAN, 2.0), (3.0, f64::NAN), (-0.0, -0.0), (0.0, 0.0)],
                [
                    (f64::INFINITY, 1.0),
                    (1.0, -f64::INFINITY),
                    (f64::NAN, 3.0),
                    (4.0, 5.0),
                ],
            ] {
                let values = (0..shape.iter().product())
                    .map(|i| pattern[i % 4])
                    .collect::<Vec<_>>();
                let array = complex_array(&shape, &values, source);
                let input = batch.add_input(&array);
                let mut tuples = axis_subsets(shape.len());
                tuples.extend([
                    vec![shape.len() as isize],
                    vec![-(shape.len() as isize) - 1],
                    vec![0, 0],
                    vec![-1, shape.len() as isize - 1],
                    vec![0, 0, shape.len() as isize],
                ]);
                for axes in tuples {
                    for keepdims in [false, true] {
                        for op in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
                            batch.push(input, &array, op, &axes, keepdims, 0);
                            for dtype in [DType::I8, DType::F32, DType::Complex128] {
                                batch.push_with_dtype(
                                    input,
                                    &array,
                                    (op, 0),
                                    &axes,
                                    keepdims,
                                    dtype,
                                );
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
fn complex_nan_tuple_reduction_order_and_conversion_buffers_match_numpy() {
    let mut batch = OracleBatch::default();
    for (source, dtype) in [
        (DType::Complex64, DType::Complex64),
        (DType::Complex64, DType::Complex128),
        (DType::Complex64, DType::F64),
        (DType::Complex128, DType::Complex64),
        (DType::Complex128, DType::F32),
    ] {
        let large = if source == DType::Complex64 {
            1e8
        } else {
            1e16
        };
        for n in [63, 64, 65, 8191, 8192, 8193] {
            for shape in [vec![2, 3, n], vec![n, 3, 2]] {
                let values = (0..shape.iter().product())
                    .map(|i| {
                        let re = if i % 17 == 3 {
                            f64::NAN
                        } else {
                            [large, 1.0, -large, 0.25, 3.1, -0.7, -2.0][i % 7]
                        };
                        let im = if i % 19 == 5 {
                            f64::NAN
                        } else {
                            [-large, 2.0, large, -0.125, 1.25][i % 5]
                        };
                        (re, im)
                    })
                    .collect::<Vec<_>>();
                let array = complex_array(&shape, &values, source);
                let input = batch.add_input(&array);
                for axes in [vec![0, 2], vec![2, 0], vec![0, 1, 2]] {
                    for op in ["nansum", "nanmean", "nanvar", "nanstd"] {
                        batch.push(input, &array, op, &axes, false, 1);
                        batch.push_with_dtype(input, &array, (op, 1), &axes, false, dtype);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_nan_tuple_products_preserve_identity_positions_and_joint_rounding() {
    let mut batch = OracleBatch::default();
    for source in [DType::Complex64, DType::Complex128] {
        for shape in [vec![2, 3, 4], vec![4, 3, 2], vec![2, 1, 3, 2]] {
            let values = (0..shape.iter().product())
                .map(|i| {
                    let re = if i % 7 == 1 {
                        f64::NAN
                    } else {
                        ((i * 17 % 97) as f64 - 48.0) / 130.0
                    };
                    let im = if i % 11 == 3 {
                        f64::NAN
                    } else {
                        ((i * 11 % 61) as f64 - 30.0) / 170.0
                    };
                    (re, im)
                })
                .collect::<Vec<_>>();
            let array = complex_array(&shape, &values, source);
            let input = batch.add_input(&array);
            for axes in axis_subsets(shape.len()) {
                for keepdims in [false, true] {
                    batch.push(input, &array, "nanprod", &axes, keepdims, 0);
                    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
                        batch.push_with_dtype(
                            input,
                            &array,
                            ("nanprod", 0),
                            &axes,
                            keepdims,
                            dtype,
                        );
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_tuple_variance_is_real_and_negative_axes_are_logical() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let a = complex_array(&[2], &[(1.0, 2.0), (3.0, 4.0)], dtype);
        assert_eq!(
            a.reduce_sum_axes(&[-1], false).unwrap().values(),
            &[4.0, 6.0]
        );
        assert_eq!(
            a.reduce_mean_axes(&[-1], false).unwrap().values(),
            &[2.0, 3.0]
        );
        let variance = a.reduce_var_axes(&[0], false, 0).unwrap();
        assert_eq!(variance.shape(), &[] as &[usize]);
        assert_eq!(variance.values(), &[2.0]);
        assert_eq!(
            variance.dtype(),
            if dtype == DType::Complex64 {
                DType::F32
            } else {
                DType::F64
            }
        );
        let large = if dtype == DType::Complex64 { 1e8 } else { 1e16 };
        let a = complex_array(
            &[2, 3],
            &[
                (large, 0.0),
                (1.0, 0.0),
                (-large, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
            ],
            dtype,
        );
        assert_eq!(
            a.reduce_sum_axes(&[0, 1], false).unwrap().values(),
            &[2.0, 0.0]
        );
    }
}

#[test]
fn complex_tuple_empty_shapes_do_not_overflow_before_zero_dimensions() {
    let input = complex_array(&[0, usize::MAX, usize::MAX], &[], DType::Complex128);
    for op in ["sum", "prod", "mean", "var", "std"] {
        for keepdims in [false, true] {
            let reduced = reduce_tuple(&input, op, &[0, 1, 2], keepdims, 0)
                .unwrap_or_else(|error| panic!("{op}: zero-sized reduction: {error}"));
            if op == "sum" {
                assert_eq!(reduced.values(), &[0.0, 0.0]);
            } else if op == "prod" {
                assert_eq!(reduced.values(), &[1.0, 0.0]);
            } else {
                assert!(reduced.values().iter().all(|value| value.is_nan()));
            }
            let mut shape = if keepdims { vec![1, 1, 1] } else { vec![] };
            if reduced.dtype().is_complex() {
                shape.push(2);
            }
            assert_eq!(reduced.shape(), shape);

            // Variance validates the selected count even without output lanes;
            // other reductions can return the empty output without forming it.
            let empty = reduce_tuple(&input, op, &[1, 2], keepdims, 0);
            if matches!(op, "var" | "std") {
                assert!(matches!(
                    empty,
                    Err(UFuncError::Shape(fnp_ndarray::ShapeError::Overflow))
                ));
                continue;
            }
            let empty = empty.unwrap_or_else(|error| panic!("{op}: zero-sized output: {error}"));
            assert!(empty.values().is_empty());
            let mut shape = if keepdims { vec![0, 1, 1] } else { vec![0] };
            if empty.dtype().is_complex() {
                shape.push(2);
            }
            assert_eq!(empty.shape(), shape);
        }
    }
}

#[test]
fn complex_tuple_shapes_empty_axes_and_axis_errors_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [
            vec![],
            vec![2],
            vec![2, 3],
            vec![2, 1, 3],
            vec![2, 3, 2, 2],
            vec![0],
            vec![0, 3],
            vec![2, 0, 3],
            vec![2, 3, 0],
            vec![0, 0],
            vec![1, 0, 2, 1],
            vec![1, 1, 1],
        ] {
            let values = (0..shape.iter().product())
                .map(|i| {
                    (
                        [0.25, -1.0, 2.5, 3.0, -0.5][i % 5],
                        [0.75, -2.0, 1.5, 0.0, 0.25][i % 5],
                    )
                })
                .collect::<Vec<_>>();
            let array = complex_array(&shape, &values, dtype);
            let input = batch.add_input(&array);
            let mut tuples = axis_subsets(shape.len());
            tuples.extend([
                vec![shape.len() as isize],
                vec![-(shape.len() as isize) - 1],
                vec![0, 0],
                vec![-1, shape.len() as isize - 1],
                vec![0, 0, shape.len() as isize],
            ]);
            for axes in tuples {
                for keepdims in [false, true] {
                    for op in ["sum", "prod", "mean", "var", "std"] {
                        batch.push(input, &array, op, &axes, keepdims, 0);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_tuple_joint_order_and_pairwise_boundaries_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let large = if dtype == DType::Complex64 { 1e8 } else { 1e16 };
        for n in [3, 4, 63, 64, 65, 127, 128, 129] {
            for shape in [[2, 3, n], [n, 3, 2]] {
                let values = (0..shape.iter().product())
                    .map(|i| {
                        (
                            [large, 1.0, -large, 0.25, 3.0, -0.5, -2.0][i % 7],
                            [-large, 2.0, large, -0.125, 1.0][i % 5],
                        )
                    })
                    .collect::<Vec<_>>();
                let array = complex_array(&shape, &values, dtype);
                let input = batch.add_input(&array);
                for axes in [
                    vec![0, 2],
                    vec![2, 0],
                    vec![-3, -1],
                    vec![0, 1],
                    vec![1, 2],
                    vec![0, 1, 2],
                ] {
                    for keepdims in [false, true] {
                        for op in ["sum", "mean", "var", "std"] {
                            batch.push(input, &array, op, &axes, keepdims, 0);
                        }
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_tuple_product_preserves_joint_fold_and_strided_precision() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [
            vec![5, 3],
            vec![3, 5],
            vec![2, 3, 4],
            vec![2, 4, 3],
            vec![2, 1, 3, 2],
            vec![1, 5, 3],
            vec![5, 1],
        ] {
            let values = (0..shape.iter().product())
                .map(|i| {
                    (
                        ((i * 17 % 97) as f64 - 48.0) / 13.0,
                        ((i * 29 % 89) as f64 - 44.0) / 19.0,
                    )
                })
                .collect::<Vec<_>>();
            let array = complex_array(&shape, &values, dtype);
            let input = batch.add_input(&array);
            for axes in axis_subsets(shape.len()) {
                for keepdims in [false, true] {
                    batch.push(input, &array, "prod", &axes, keepdims, 0);
                }
            }
            for axis in [None, Some(0), Some(-1)] {
                for keepdims in [false, true] {
                    for accumulator in [None, Some(DType::Complex64), Some(DType::Complex128)] {
                        batch.push_single_product(input, &array, axis, keepdims, accumulator);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_tuple_nonfinite_signed_zero_and_ddof_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for values in [
            vec![(-0.0, -0.0), (-0.0, 0.0), (0.0, -0.0), (0.0, 0.0)],
            vec![
                (f64::INFINITY, 1.0),
                (1.0, 2.0),
                (-f64::INFINITY, 3.0),
                (4.0, -1.0),
            ],
            vec![
                (1.0, f64::INFINITY),
                (2.0, -1.0),
                (f64::NAN, 1.0),
                (4.0, 3.0),
            ],
            vec![(1.0, f64::NAN), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0)],
            vec![(1.0, 2.0), (3.0, 4.0), (-1.0, -2.0), (1.0, -2.0)],
        ] {
            for shape in [vec![2, 2], vec![1, 2, 1, 2]] {
                let array = complex_array(&shape, &values, dtype);
                let input = batch.add_input(&array);
                for axes in axis_subsets(shape.len()) {
                    for keepdims in [false, true] {
                        for op in ["sum", "prod", "mean"] {
                            batch.push(input, &array, op, &axes, keepdims, 0);
                        }
                        for ddof in [0, 1, 2, 4, 5] {
                            for op in ["var", "std"] {
                                batch.push(input, &array, op, &axes, keepdims, ddof);
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
fn complex_tuple_explicit_dtypes_match_numpy_and_preserve_imaginary_deviations() {
    let mut batch = OracleBatch::default();
    for source in [DType::Complex64, DType::Complex128] {
        let values = [
            (0.0, 1.0),
            (1.25, -2.5),
            (0.0, -1.0),
            (-3.0, 0.0),
            (2.75, 4.0),
            (-0.0, -0.0),
            (1.0004882812500002, 0.5),
            (2.5, -1.25),
            (4.0, 3.0),
            (-1.5, -2.0),
            (3.5, 1.0),
            (0.25, -0.75),
        ];
        let array = complex_array(&[2, 3, 2], &values, source);
        let input = batch.add_input(&array);
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
            DType::Complex64,
            DType::Complex128,
        ] {
            for axes in [vec![], vec![0, 2], vec![2, 0], vec![-1, -3], vec![0, 1, 2]] {
                for keepdims in [false, true] {
                    for op in ["sum", "prod", "mean"] {
                        batch.push_with_dtype(input, &array, (op, 0), &axes, keepdims, dtype);
                    }
                    for ddof in [0, 1, 12] {
                        for op in ["var", "std"] {
                            batch.push_with_dtype(
                                input,
                                &array,
                                (op, ddof),
                                &axes,
                                keepdims,
                                dtype,
                            );
                        }
                    }
                }
            }
        }

        let array = complex_array(
            &[2, 2],
            &[(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)],
            source,
        );
        let real = array
            .reduce_var_axes_with_dtype(&[0, 1], false, 0, DType::F32)
            .unwrap();
        assert_eq!(real.dtype(), DType::F32);
        assert_eq!(real.values(), &[14.75]);
        let complex = array
            .reduce_var_axes_with_dtype(&[0, 1], false, 0, DType::Complex64)
            .unwrap();
        assert_eq!(complex.dtype(), DType::Complex64);
        assert_eq!(complex.values(), &[2.5, 0.0]);
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_tuple_explicit_dtype_empty_scalar_and_axis_errors_match_numpy() {
    let mut batch = OracleBatch::default();
    for shape in [vec![], vec![0], vec![2, 0, 3], vec![2, 3, 0], vec![1, 1, 1]] {
        let values = vec![(-0.0, 2.0); shape.iter().product()];
        let array = complex_array(&shape, &values, DType::Complex64);
        let input = batch.add_input(&array);
        let mut tuples = axis_subsets(shape.len());
        tuples.extend([
            vec![shape.len() as isize],
            vec![-(shape.len() as isize) - 1],
            vec![0, 0],
            vec![0, 0, shape.len() as isize],
        ]);
        for dtype in [DType::Bool, DType::I8, DType::F16, DType::Complex128] {
            for axes in &tuples {
                for keepdims in [false, true] {
                    for op in ["sum", "prod", "mean", "var", "std"] {
                        batch.push_with_dtype(input, &array, (op, 0), axes, keepdims, dtype);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_tuple_rejects_unrepresentable_physical_output_capacity() {
    // The second extent fits a real output's byte bound but exceeds it once
    // each logical result needs two components. Neither case allocates data.
    let too_many_pairs = isize::MAX as usize / std::mem::size_of::<(f64, f64)>() + 1;
    for columns in [usize::MAX, too_many_pairs] {
        let array = complex_array(&[0, columns], &[], DType::Complex64);
        for keepdims in [false, true] {
            for dtype in [DType::Complex64, DType::Complex128] {
                for result in [
                    array.reduce_sum_axes_with_dtype(&[0], keepdims, dtype),
                    array.reduce_prod_axes_with_dtype(&[0], keepdims, dtype),
                    array.reduce_mean_axes_with_dtype(&[0], keepdims, dtype),
                    array.reduce_var_axes_with_dtype(&[0], keepdims, 0, dtype),
                    array.reduce_std_axes_with_dtype(&[0], keepdims, 0, dtype),
                    array.nansum_axes_with_dtype(&[0], keepdims, dtype),
                    array.nanprod_axes_with_dtype(&[0], keepdims, dtype),
                    array.nanmean_axes_with_dtype(&[0], keepdims, dtype),
                    array.nanvar_axes_with_dtype(&[0], keepdims, 0, dtype),
                    array.nanstd_axes_with_dtype(&[0], keepdims, 0, dtype),
                ] {
                    assert!(matches!(
                        result,
                        Err(UFuncError::Shape(fnp_ndarray::ShapeError::Overflow))
                    ));
                }
            }
            // Default complex nanvar/nanstd have real results, but their
            // intermediate complex mean must also fit before counts allocate.
            for op in [
                "sum", "prod", "mean", "var", "std", "nansum", "nanprod", "nanmean", "nanvar",
                "nanstd",
            ] {
                assert!(matches!(
                    reduce_tuple(&array, op, &[0], keepdims, 0),
                    Err(UFuncError::Shape(fnp_ndarray::ShapeError::Overflow))
                ));
            }
        }
    }
}

#[test]
fn complex_tuple_explicit_dtype_conversion_buffers_match_numpy() {
    let mut batch = OracleBatch::default();
    for (source, dtype) in [
        (DType::Complex64, DType::Complex128),
        (DType::Complex128, DType::Complex64),
        (DType::Complex64, DType::F64),
        (DType::Complex128, DType::F32),
        (DType::Complex128, DType::F16),
    ] {
        let large = if dtype == DType::F16 {
            2048.0
        } else if matches!(dtype, DType::F32 | DType::Complex64) {
            1e8
        } else {
            1e16
        };
        for n in [63, 64, 65, 8191, 8192, 8193] {
            for shape in [vec![2, 3, n], vec![n, 3, 2]] {
                let values = (0..shape.iter().product())
                    .map(|i| {
                        (
                            [large, 1.0, -large, 0.25, 3.1, -0.7, -2.0][i % 7],
                            [-large, 2.0, large, -0.125, 1.25][i % 5],
                        )
                    })
                    .collect::<Vec<_>>();
                let array = complex_array(&shape, &values, source);
                let input = batch.add_input(&array);
                for axes in [vec![0, 2], vec![2, 0], vec![0, 1, 2]] {
                    for op in ["sum", "mean", "var", "std"] {
                        batch.push_with_dtype(input, &array, (op, 0), &axes, false, dtype);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_tuple_explicit_product_dtypes_preserve_joint_rounding() {
    let mut batch = OracleBatch::default();
    for source in [DType::Complex64, DType::Complex128] {
        for shape in [vec![2, 3], vec![3, 2], vec![2, 1, 3], vec![2, 3, 2]] {
            let values = (0..shape.iter().product())
                .map(|i| {
                    (
                        ((i * 17 % 97) as f64 - 48.0) / 13.0,
                        ((i * 11 % 61) as f64 - 30.0) / 17.0,
                    )
                })
                .collect::<Vec<_>>();
            let array = complex_array(&shape, &values, source);
            let input = batch.add_input(&array);
            for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
                for axes in axis_subsets(shape.len()) {
                    for keepdims in [false, true] {
                        batch.push_with_dtype(input, &array, ("prod", 0), &axes, keepdims, dtype);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_truth_reductions_and_indices_match_logical_numpy_elements() {
    let mut batch = OracleBatch::default();
    let pattern = [
        (0.0, -0.0),
        (0.0, 3.0),
        (-2.0, 0.0),
        (1.0, -4.0),
        (f64::NAN, 0.0),
        (0.0, f64::NAN),
        (f64::INFINITY, -0.0),
        (0.0, f64::NEG_INFINITY),
        (1.0e-50, -1.0e-50),
        (0.0, f64::from(f32::from_bits(1))),
        (-0.0, 0.0),
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [
            vec![],
            vec![pattern.len()],
            vec![3, 4],
            vec![2, 3, 2],
            vec![2, 0, 3],
            vec![0, 2],
        ] {
            // Rotations make the scalar cases cover each truth state as well
            // as changing which logical axes contain the nonzero elements.
            for offset in 0..pattern.len() {
                let values = (0..shape.iter().product())
                    .map(|index| pattern[(index + offset) % pattern.len()])
                    .collect::<Vec<_>>();
                let array = complex_array(&shape, &values, dtype);
                let input = batch.add_input(&array);
                batch.push_truth_indices(input, &array);
                for op in ["any", "all", "count_nonzero"] {
                    for axis in [None, Some(0), Some(-1), Some(1), Some(-3), Some(3)] {
                        batch.push_truth_single(input, &array, op, axis, false);
                        if op == "count_nonzero" {
                            batch.push_truth_single(input, &array, op, axis, true);
                        }
                    }
                    let mut axes = axis_subsets(shape.len());
                    axes.extend([vec![0, 0], vec![shape.len() as isize], vec![-4]]);
                    for axis_tuple in axes {
                        batch.push(input, &array, op, &axis_tuple, false, 0);
                        if op == "count_nonzero" {
                            batch.push(input, &array, op, &axis_tuple, true, 0);
                        }
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_truth_counts_each_pair_once_and_checks_layout_before_axes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let array = complex_array(
            &[2, 2],
            &[(0.0, 1.0), (2.0, 0.0), (3.0, 4.0), (0.0, -0.0)],
            dtype,
        );
        assert_eq!(array.all(Some(0)).unwrap().values(), &[1.0, 0.0]);
        assert_eq!(array.any(Some(-1)).unwrap().values(), &[1.0, 1.0]);
        assert_eq!(array.count_nonzero(None, false).unwrap().values(), &[3.0]);
        assert_eq!(array.flatnonzero().values(), &[0.0, 1.0, 2.0]);
        assert_eq!(array.nonzero().values(), &[0.0, 1.0, 2.0]);
        assert_eq!(array.argwhere().shape(), &[3, 2]);
        assert_eq!(array.argwhere().values(), &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        let coordinates = where_nonzero(&array).unwrap();
        assert_eq!(coordinates.len(), 2);
        assert_eq!(coordinates[0].values(), &[0.0, 0.0, 1.0]);
        assert_eq!(coordinates[1].values(), &[0.0, 1.0, 0.0]);
        for result in [
            array.any(Some(2)),
            array.all(Some(2)),
            array.count_nonzero(Some(2), false),
        ] {
            assert!(matches!(
                result,
                Err(UFuncError::AxisOutOfBounds { axis: 2, ndim: 2 })
            ));
        }

        let malformed = UFuncArray::new(vec![3], vec![0.0, 1.0, 2.0], dtype).unwrap();
        for result in [
            malformed.any(None),
            malformed.all(None),
            malformed.any_axes(&[]),
            malformed.all_axes(&[]),
            malformed.count_nonzero(None, false),
            malformed.count_nonzero_axes(&[], false),
        ] {
            assert!(matches!(result, Err(UFuncError::Msg(message))
                if message.contains("interleaved trailing-dimension shape")));
        }
        assert!(where_nonzero(&malformed).is_err());
        // The infallible flat/index methods retain their prior malformed-input
        // behavior instead of gaining a panic or changing their public API.
        assert_eq!(malformed.nonzero().values(), &[1.0, 2.0]);
        assert_eq!(malformed.flatnonzero().values(), &[1.0, 2.0]);
        assert_eq!(malformed.argwhere().shape(), &[2, 1]);
        assert_eq!(malformed.argwhere().values(), &[1.0, 2.0]);
    }
}
