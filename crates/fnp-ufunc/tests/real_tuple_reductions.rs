//! Floating tuple reductions preserve NumPy's joint traversal and source precision.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{BinaryOp, UFuncArray, UFuncError};
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
        _ => panic!("unexpected real reduction dtype {dtype:?}"),
    }
}

#[test]
fn uint64_boolean_arithmetic_preserves_exact_unsigned_sidecars() {
    let values = vec![(1_u64 << 53) + 1, (1_u64 << 63) + 1, u64::MAX];
    let array = UFuncArray::from_storage(vec![3], ArrayStorage::U64(values.clone())).unwrap();
    let truth = UFuncArray::scalar(1.0, DType::Bool);
    let extract = |array: UFuncArray| {
        assert_eq!(array.dtype(), DType::U64);
        match array.to_storage().unwrap() {
            ArrayStorage::U64(values) => values,
            other => panic!("expected unsigned sidecar, got {:?}", other.dtype()),
        }
    };
    let added = vec![(1_u64 << 53) + 2, (1_u64 << 63) + 2, 0];
    for (lhs, rhs) in [(&array, &truth), (&truth, &array)] {
        assert_eq!(
            extract(lhs.elementwise_binary(rhs, BinaryOp::Add).unwrap()),
            added
        );
        assert_eq!(
            extract(lhs.elementwise_binary(rhs, BinaryOp::Mul).unwrap()),
            values
        );
    }
    assert_eq!(
        extract(array.elementwise_binary(&truth, BinaryOp::Sub).unwrap()),
        vec![1_u64 << 53, 1_u64 << 63, u64::MAX - 1]
    );
    assert_eq!(
        extract(truth.elementwise_binary(&array, BinaryOp::Sub).unwrap()),
        vec![u64::MAX - (1_u64 << 53) + 1, 1_u64 << 63, 2]
    );
}

fn reduce_tuple(
    array: &UFuncArray,
    op: &str,
    axes: &[isize],
    keepdims: bool,
    ddof: usize,
) -> Result<UFuncArray, UFuncError> {
    match op {
        "sum" => array.reduce_sum_axes(axes, keepdims),
        "prod" => array.reduce_prod_axes(axes, keepdims),
        "mean" => array.reduce_mean_axes(axes, keepdims),
        "var" => array.reduce_var_axes(axes, keepdims, ddof),
        "std" => array.reduce_std_axes(axes, keepdims, ddof),
        "nansum" => array.nansum_axes(axes, keepdims),
        "nanprod" => array.nanprod_axes(axes, keepdims),
        "nanmean" => array.nanmean_axes(axes, keepdims),
        "nanvar" => array.nanvar_axes(axes, keepdims, ddof),
        "nanstd" => array.nanstd_axes(axes, keepdims, ddof),
        _ => unreachable!("real tuple operation"),
    }
}

fn axis_subsets(ndim: usize) -> Vec<Vec<isize>> {
    (0..1usize << ndim)
        .map(|mask| {
            (0..ndim)
                .filter(|axis| mask & (1 << axis) != 0)
                .map(|axis| axis as isize)
                .collect()
        })
        .collect()
}

fn reduction_dtypes() -> [DType; 14] {
    [
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
    ]
}

#[derive(Default)]
struct OracleBatch {
    inputs: Vec<Value>,
    cases: Vec<Value>,
    actual: Vec<Result<UFuncArray, UFuncError>>,
}

impl OracleBatch {
    fn add_input(&mut self, array: &UFuncArray) -> usize {
        let index = self.inputs.len();
        let values: Vec<String> = match array.to_storage().expect("typed oracle source") {
            ArrayStorage::I64(values) => values.iter().map(i64::to_string).collect(),
            ArrayStorage::U64(values) => values.iter().map(u64::to_string).collect(),
            _ => array.values().iter().map(f64::to_string).collect(),
        };
        self.inputs.push(json!({
            "shape": array.shape(), "dtype": dtype_name(array.dtype()),
            "values": values,
        }));
        index
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

    fn push_single_mean(
        &mut self,
        input: usize,
        array: &UFuncArray,
        axis: Option<isize>,
        keepdims: bool,
    ) {
        self.cases.push(json!({
            "input": input, "op": "mean", "axis": axis, "keepdims": keepdims,
        }));
        self.actual.push(array.reduce_mean(axis, keepdims));
    }

    fn push_single_statistic(
        &mut self,
        input: usize,
        array: &UFuncArray,
        operation: (&str, usize),
        axis: Option<isize>,
        keepdims: bool,
    ) {
        let (op, ddof) = operation;
        self.cases.push(json!({
            "input": input, "op": op, "axis": axis,
            "keepdims": keepdims, "ddof": ddof,
        }));
        self.actual.push(match op {
            "var" => array.reduce_var(axis, keepdims, ddof),
            "std" => array.reduce_std(axis, keepdims, ddof),
            _ => unreachable!("single-axis statistic"),
        });
    }

    fn push_nanarg(
        &mut self,
        input: usize,
        array: &UFuncArray,
        op: &str,
        axis: Option<isize>,
    ) {
        self.cases.push(json!({
            "input": input, "op": op, "axis": axis, "keepdims": false,
        }));
        self.actual.push(match op {
            "nanargmin" => array.nanargmin(axis),
            "nanargmax" => array.nanargmax(axis),
            _ => unreachable!("NaN-aware argument reduction"),
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
            _ => unreachable!("typed real tuple operation"),
        });
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np

warnings.simplefilter('ignore')
payload = json.load(sys.stdin)
inputs = []
for item in payload['inputs']:
    parse = int if item['dtype'] in ('int64', 'uint64') else float
    inputs.append(np.asarray([parse(x) for x in item['values']], dtype=item['dtype'])
                  .reshape(item['shape']))
rows = []
for case in payload['cases']:
    axis = tuple(case['axes']) if 'axes' in case else case['axis']
    kw = dict(axis=axis, keepdims=case['keepdims'])
    if case.get('dtype') is not None:
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
            .expect("launch live NumPy real tuple oracle");
        let mut stdin = child.stdin.take().expect("oracle stdin");
        serde_json::to_writer(
            &mut stdin,
            &json!({"inputs": self.inputs, "cases": self.cases}),
        )
        .expect("write real tuple oracle inputs");
        drop(stdin);
        let output = child.wait_with_output().expect("collect real tuple oracle");
        assert!(
            output.status.success(),
            "NumPy real tuple oracle: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
        assert_eq!(expected.len(), self.actual.len());
        for ((case, actual), expected) in self.cases.iter().zip(self.actual).zip(expected) {
            if expected["error"] == "AxisError" {
                assert!(
                    matches!(actual, Err(UFuncError::AxisOutOfBounds { axis, ndim })
                        if json!(axis) == expected["axis"] && json!(ndim) == expected["ndim"]),
                    "{case}: expected axis error {expected}, got {actual:?}"
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
fn default_mean_preserves_numpy_pairwise_order_and_source_dtypes() {
    let mut batch = OracleBatch::default();
    let mut overflow_cancellation = vec![1.0e308; 8];
    overflow_cancellation.extend_from_slice(&[-1.0e308; 8]);
    for values in [
        vec![1.0e16, 1.0, -1.0e16, 1.0, 1.0e16, 1.0, -1.0e16, 1.0],
        overflow_cancellation,
        vec![1.0e20; 129],
        vec![-0.0, -0.0, 0.0, 0.0],
        vec![f64::INFINITY, 1.0, f64::NEG_INFINITY, 4.0],
        vec![1.0, f64::NAN, -2.0, 3.0],
    ] {
        let n = values.len();
        for shape in [vec![n], vec![1, n], vec![n, 1]] {
            let array = UFuncArray::new(shape, values.clone(), DType::F64).unwrap();
            let input = batch.add_input(&array);
            for axis in [None, Some(0), Some(1), Some(-1)] {
                for keepdims in [false, true] {
                    batch.push_single_mean(input, &array, axis, keepdims);
                }
            }
        }
    }

    // Exercise both contiguous pairwise lanes and strided output updates,
    // including source rounding and both sides of the pairwise boundaries.
    for dtype in [DType::F16, DType::F32, DType::F64] {
        let large = match dtype {
            DType::F16 => 2048.0,
            DType::F32 => 1.0e8,
            _ => 1.0e16,
        };
        let pattern = [
            large,
            1.000_488_281_250_000_2,
            -large,
            0.25,
            3.1,
            -0.7,
            -2.0,
        ];
        for n in [7, 8, 9, 127, 128, 129, 257] {
            let values: Vec<f64> = (0..2 * n)
                .map(|index| pattern[index % pattern.len()])
                .collect();
            for shape in [vec![2, n], vec![n, 2], vec![2, n, 1]] {
                let array = UFuncArray::new(shape, values.clone(), dtype).unwrap();
                let input = batch.add_input(&array);
                for axis in [None, Some(0), Some(1), Some(-1)] {
                    for keepdims in [false, true] {
                        batch.push_single_mean(input, &array, axis, keepdims);
                    }
                }
            }
        }
    }

    // A numeric constructor can retain an unnormalized working buffer.
    // Mean must read the declared source storage before accumulating.
    for dtype in [DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16] {
        let array = UFuncArray::new(vec![2, 3], vec![0.0, 1.2, 2.7, 0.0, 3.9, 4.1], dtype).unwrap();
        let input = batch.add_input(&array);
        for axis in [None, Some(0), Some(1), Some(-1)] {
            for keepdims in [false, true] {
                batch.push_single_mean(input, &array, axis, keepdims);
            }
        }
    }

    // Integer sidecars preserve values that cannot be represented by the
    // working f64 buffer until NumPy's accumulator conversion takes place.
    for storage in [
        ArrayStorage::I64(vec![
            (1_i64 << 53) + 1,
            (1_i64 << 53) + 3,
            -((1_i64 << 53) + 1),
            -((1_i64 << 53) + 3),
            i64::MIN + 1,
            i64::MAX - 1,
            17,
            -7,
        ]),
        ArrayStorage::U64(vec![
            (1_u64 << 53) + 1,
            (1_u64 << 53) + 3,
            (1_u64 << 63) + 1,
            (1_u64 << 63) + 3,
            u64::MAX,
            u64::MAX - 2,
            17,
            7,
        ]),
    ] {
        for shape in [vec![8], vec![2, 4], vec![4, 2]] {
            let array = UFuncArray::from_storage(shape, storage.clone()).unwrap();
            let input = batch.add_input(&array);
            for axis in [None, Some(0), Some(1), Some(-1)] {
                for keepdims in [false, true] {
                    batch.push_single_mean(input, &array, axis, keepdims);
                }
            }
        }
    }

    for dtype in [
        DType::Bool,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::F32,
        DType::F64,
    ] {
        for (shape, values) in [
            (vec![], vec![1.25]),
            (vec![0], vec![]),
            (vec![0, 2], vec![]),
            (vec![2, 0], vec![]),
        ] {
            let array = UFuncArray::new(shape, values, dtype).unwrap();
            let input = batch.add_input(&array);
            for axis in [None, Some(0), Some(1), Some(-1), Some(-2)] {
                for keepdims in [false, true] {
                    batch.push_single_mean(input, &array, axis, keepdims);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn default_float32_mean_keeps_large_reduction_counts_exact() {
    let n = (1_usize << 24) + 1;
    // Generate this one bounded large fixture inside NumPy instead of sending
    // millions of values through JSON. The exact integer divisor differs
    // from its float32 rounding at this length.
    let script = r#"
import json, sys
import numpy as np

values = np.zeros(int(sys.argv[1]), dtype=np.float32)
values[0] = 1.0
rows = []
for axis in (None, 0):
    result = np.asarray(np.mean(values, axis=axis))
    rows.append(dict(shape=list(result.shape), dtype=str(result.dtype),
                     bits=int(result.astype(np.float64).view(np.uint64).item())))
print(json.dumps(rows))
"#;
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = Command::new(python)
        .args(["-c", script])
        .arg(n.to_string())
        .output()
        .expect("launch live NumPy large-count mean oracle");
    assert!(
        output.status.success(),
        "NumPy large-count mean oracle: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<Value> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
    assert_eq!(expected.len(), 2);

    let mut values = vec![0.0; n];
    values[0] = 1.0;
    let array = UFuncArray::new(vec![n], values, DType::F32).unwrap();
    for (axis, expected) in [None, Some(0)].into_iter().zip(expected) {
        let actual = array.reduce_mean(axis, false).unwrap();
        assert_eq!(actual.dtype(), DType::F32);
        assert_eq!(dtype_name(actual.dtype()), expected["dtype"]);
        assert!(actual.shape().is_empty());
        assert_eq!(json!(actual.shape()), expected["shape"]);
        assert_eq!(actual.values().len(), 1);
        assert_eq!(
            actual.values()[0].to_bits(),
            expected["bits"].as_u64().expect("float bits"),
            "axis {axis:?}: mean must divide by the exact reduction count"
        );
    }
}

#[test]
fn default_variance_and_std_preserve_numpy_reduction_order() {
    // Sequentially summing the mean used to invent a standard deviation of
    // 262144 for this constant array; NumPy's pairwise mean is exactly 1e20.
    let constant = UFuncArray::new(vec![129], vec![1.0e20; 129], DType::F64).unwrap();
    assert_eq!(
        constant.reduce_var(None, false, 0).unwrap().values(),
        &[0.0]
    );
    assert_eq!(
        constant.reduce_std(None, false, 0).unwrap().values(),
        &[0.0]
    );

    let mut batch = OracleBatch::default();
    for dtype in [DType::F16, DType::F32, DType::F64] {
        for n in [7, 8, 127, 128, 129, 257] {
            let offset = if dtype == DType::F16 { 16.0 } else { 1.0e5 };
            let values: Vec<f64> = (0..2 * n)
                .map(|index| offset + ((index * 17 % 31) as f64 - 15.0) * 0.03125)
                .collect();
            for shape in [vec![2, n], vec![n, 2], vec![2, n, 1]] {
                let array = UFuncArray::new(shape.clone(), values.clone(), dtype).unwrap();
                let input = batch.add_input(&array);
                let all_axes: Vec<isize> = (0..shape.len()).map(|axis| axis as isize).collect();
                for axis in [None, Some(0), Some(1), Some(-1)] {
                    let axes = axis.map_or_else(|| all_axes.clone(), |axis| vec![axis]);
                    for keepdims in [false, true] {
                        for ddof in [0, 1, 2, 2 * n] {
                            for op in ["var", "std"] {
                                batch.push_single_statistic(
                                    input,
                                    &array,
                                    (op, ddof),
                                    axis,
                                    keepdims,
                                );
                                batch.push(input, &array, op, &axes, keepdims, ddof);
                            }
                        }
                    }
                }
            }
        }
    }
    let input = batch.add_input(&constant);
    for axis in [None, Some(0), Some(-1)] {
        for keepdims in [false, true] {
            for op in ["var", "std"] {
                batch.push_single_statistic(input, &constant, (op, 0), axis, keepdims);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn default_integer_variance_preserves_source_casts_and_conversion_buffers() {
    let mut batch = OracleBatch::default();
    for storage in [
        ArrayStorage::I64(
            (0..16_386)
                .map(|index| (1_i64 << 53) + (index * 17 % 31) - 15)
                .collect(),
        ),
        ArrayStorage::U64(
            (0..16_386)
                .map(|index| u64::MAX - (index * 17 % 31))
                .collect(),
        ),
    ] {
        for shape in [vec![2, 8193], vec![8193, 2]] {
            let array = UFuncArray::from_storage(shape, storage.clone()).unwrap();
            let input = batch.add_input(&array);
            for axis in [None, Some(0), Some(1), Some(-1)] {
                for keepdims in [false, true] {
                    for ddof in [0, 1] {
                        for op in ["var", "std"] {
                            batch.push_single_statistic(input, &array, (op, ddof), axis, keepdims);
                        }
                    }
                }
            }
        }
    }
    // Numeric construction can retain an unnormalized working buffer. Both
    // the mean and deviations must observe the declared storage, even for
    // narrow integers and Boolean values without a wide-integer sidecar.
    for dtype in [DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16] {
        let array = UFuncArray::new(vec![2, 3], vec![0.0, 1.2, 2.7, 0.0, 3.9, 4.1], dtype).unwrap();
        let input = batch.add_input(&array);
        for axis in [None, Some(0), Some(1)] {
            let axes = axis.map_or_else(|| vec![0, 1], |axis| vec![axis]);
            for keepdims in [false, true] {
                for op in ["var", "std"] {
                    batch.push_single_statistic(input, &array, (op, 0), axis, keepdims);
                    batch.push(input, &array, op, &axes, keepdims, 0);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn nanarg_extrema_infinities_nan_ties_and_scalar_axes_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::F16, DType::F32, DType::F64] {
        for values in [
            vec![],
            vec![f64::INFINITY],
            vec![f64::NEG_INFINITY],
            vec![f64::NAN],
            vec![f64::NAN, f64::INFINITY],
            vec![f64::INFINITY, f64::NAN],
            vec![f64::NAN, f64::NEG_INFINITY],
            vec![f64::NEG_INFINITY, f64::NAN],
            vec![f64::NAN, f64::NAN],
            vec![f64::INFINITY, f64::INFINITY, f64::NAN],
            vec![f64::NEG_INFINITY, f64::NAN, f64::NEG_INFINITY],
            vec![f64::NAN, -0.0, 0.0, -0.0],
        ] {
            let n = values.len();
            for shape in [vec![n], vec![1, n], vec![n, 1]] {
                let array = UFuncArray::new(shape, values.clone(), dtype).unwrap();
                let input = batch.add_input(&array);
                for axis in [None, Some(0), Some(1), Some(-1), Some(-2)] {
                    for op in ["nanargmin", "nanargmax"] {
                        batch.push_nanarg(input, &array, op, axis);
                    }
                }
            }
        }
        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -0.0, 0.0] {
            let array = UFuncArray::scalar(value, dtype);
            let input = batch.add_input(&array);
            for axis in [None, Some(0), Some(-1), Some(1), Some(-2)] {
                for op in ["nanargmin", "nanargmax"] {
                    batch.push_nanarg(input, &array, op, axis);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn nanarg_extrema_observe_declared_source_precision() {
    let mut batch = OracleBatch::default();
    for (dtype, values) in [
        (DType::F16, vec![1.0001, 1.0, f64::NAN, -0.0]),
        (DType::F16, vec![70000.0, 80000.0, f64::NAN, f64::INFINITY]),
        (DType::F32, vec![1.0 + 2.0_f64.powi(-25), 1.0, f64::NAN, -0.0]),
        (DType::F32, vec![1.0e100, 1.0e90, f64::NAN, f64::INFINITY]),
        (DType::Bool, vec![1.9, 1.1, 0.0, -0.0]),
        (DType::I8, vec![1.9, 1.1, 3.1, 3.9]),
        (DType::U8, vec![1.9, 1.1, 3.1, 3.9]),
        (DType::I16, vec![1.9, 1.1, 3.1, 3.9]),
        (DType::U16, vec![1.9, 1.1, 3.1, 3.9]),
        (DType::I32, vec![1.9, 1.1, 3.1, 3.9]),
        (DType::U32, vec![1.9, 1.1, 3.1, 3.9]),
    ] {
        for shape in [vec![4], vec![2, 2]] {
            let array = UFuncArray::new(shape, values.clone(), dtype).unwrap();
            let input = batch.add_input(&array);
            for axis in [None, Some(0), Some(-1)] {
                for op in ["nanargmin", "nanargmax"] {
                    batch.push_nanarg(input, &array, op, axis);
                }
            }
        }
    }
    for storage in [
        ArrayStorage::I64(vec![
            (1_i64 << 53) + 1,
            (1_i64 << 53) + 2,
            i64::MIN,
            i64::MIN + 1,
        ]),
        ArrayStorage::U64(vec![
            (1_u64 << 53) + 1,
            (1_u64 << 53) + 2,
            u64::MAX,
            u64::MAX - 1,
        ]),
    ] {
        let array = UFuncArray::from_storage(vec![2, 2], storage).unwrap();
        let input = batch.add_input(&array);
        for axis in [None, Some(0), Some(1)] {
            for op in ["nanargmin", "nanargmax"] {
                batch.push_nanarg(input, &array, op, axis);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn nanarg_extrema_simd_tails_and_parallel_lanes_match_numpy() {
    let mut batch = OracleBatch::default();
    // Width nine exercises an eight-lane SIMD block and its scalar tail.
    // The two large shapes cross the parallel threshold with outer == 1
    // and outer > 1 respectively, using the same infinity/NaN contracts.
    for (outer, rows, columns) in [(1, 3, 9), (2, 3, 9), (1, 5, 4097), (5, 5, 1025)] {
        let shape = if outer == 1 {
            vec![rows, columns]
        } else {
            vec![outer, rows, columns]
        };
        let axis = if outer == 1 { 0 } else { 1 };
        let values: Vec<f64> = (0..outer * rows * columns)
            .map(|index| {
                let row = index / columns % rows;
                match index % columns % 6 {
                    0 => f64::INFINITY,
                    1 => f64::NEG_INFINITY,
                    2 if row == 0 => f64::NAN,
                    2 => f64::INFINITY,
                    3 if row == 0 => f64::NAN,
                    3 => f64::NEG_INFINITY,
                    4 if row == 0 => f64::NAN,
                    4 => (row % 2) as f64,
                    _ if row % 2 == 0 => -0.0,
                    _ => 0.0,
                }
            })
            .collect();
        let array = UFuncArray::new(shape.clone(), values.clone(), DType::F64).unwrap();
        let input = batch.add_input(&array);
        for selected_axis in [None, Some(axis), Some(-2)] {
            for op in ["nanargmin", "nanargmax"] {
                batch.push_nanarg(input, &array, op, selected_axis);
            }
        }
        // All-NaN lanes in both the first vector block and the final scalar
        // tail must still fail, even when every other lane has valid data.
        for column in [0, columns - 1] {
            let mut all_nan = values.clone();
            for row in 0..rows {
                all_nan[row * columns + column] = f64::NAN;
            }
            let array = UFuncArray::new(shape.clone(), all_nan, DType::F64).unwrap();
            let input = batch.add_input(&array);
            for op in ["nanargmin", "nanargmax"] {
                batch.push_nanarg(input, &array, op, Some(axis));
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn real_nan_tuple_masks_counts_and_ddof_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::F16, DType::F32, DType::F64] {
        // Reducing axes 0 and 2 leaves lanes with two, zero, and three valid
        // elements. A mean of intermediate means would weight them incorrectly.
        let array = UFuncArray::new(
            vec![2, 3, 2],
            vec![
                1.25,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                -0.0,
                3.75,
                f64::NAN,
                8.5,
                f64::NAN,
                f64::NAN,
                2.5,
                f64::NAN,
            ],
            dtype,
        )
        .unwrap();
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
fn real_nan_tuple_explicit_dtypes_and_statistical_rejections_match_numpy() {
    let mut batch = OracleBatch::default();
    for source in [DType::F16, DType::F32, DType::F64] {
        for has_nan in [false, true] {
            let values = (0..12)
                .map(|i| {
                    if has_nan && i % 4 == 1 {
                        f64::NAN
                    } else {
                        [0.25, -1.1, 2.5, 3.7, -0.5, 1.0004882812500002][i % 6]
                    }
                })
                .collect();
            let array = UFuncArray::new(vec![2, 3, 2], values, source).unwrap();
            let input = batch.add_input(&array);
            for dtype in reduction_dtypes() {
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
fn real_nan_tuple_empty_scalar_nonfinite_and_axis_errors_match_numpy() {
    let mut batch = OracleBatch::default();
    for shape in [
        vec![],
        vec![0],
        vec![2, 0, 3],
        vec![2, 3, 0],
        vec![1, 1, 1],
        vec![2, 2],
    ] {
        for pattern in [
            [f64::NAN, -0.0, 0.0, f64::NAN],
            [f64::INFINITY, 1.0, -f64::INFINITY, f64::NAN],
        ] {
            let array = UFuncArray::new(
                shape.clone(),
                (0..shape.iter().product())
                    .map(|i| pattern[i % 4])
                    .collect(),
                DType::F64,
            )
            .unwrap();
            let input = batch.add_input(&array);
            let mut tuples = axis_subsets(shape.len());
            tuples.extend([
                vec![shape.len() as isize],
                vec![-(shape.len() as isize) - 1],
                vec![0, 0],
                vec![0, 0, shape.len() as isize],
            ]);
            for axes in tuples {
                for keepdims in [false, true] {
                    for op in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
                        batch.push(input, &array, op, &axes, keepdims, 0);
                        for dtype in [DType::I8, DType::F32, DType::Complex64] {
                            batch.push_with_dtype(input, &array, (op, 0), &axes, keepdims, dtype);
                        }
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn real_nan_tuple_integer_inputs_preserve_exact_storage() {
    let mut batch = OracleBatch::default();
    for storage in [
        ArrayStorage::Bool(vec![true, false, true, true, false, false]),
        ArrayStorage::I64(vec![
            9_007_199_254_740_993,
            9_007_199_254_740_995,
            -9_007_199_254_740_993,
            -9_007_199_254_740_995,
            i64::MIN + 1,
            i64::MAX - 1,
        ]),
        ArrayStorage::U64(vec![
            9_007_199_254_740_993,
            9_007_199_254_740_995,
            9_007_199_254_740_997,
            u64::MAX,
            u64::MAX - 2,
            u64::MAX - 4,
        ]),
    ] {
        let array = UFuncArray::from_storage(vec![2, 1, 3], storage).unwrap();
        let input = batch.add_input(&array);
        for axes in [vec![], vec![0, 2], vec![-1, -3], vec![1, 2]] {
            for keepdims in [false, true] {
                for op in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"] {
                    batch.push(input, &array, op, &axes, keepdims, 0);
                    for dtype in [
                        DType::Bool,
                        DType::I8,
                        DType::I64,
                        DType::U64,
                        DType::F32,
                        DType::F64,
                        DType::Complex64,
                    ] {
                        batch.push_with_dtype(input, &array, (op, 0), &axes, keepdims, dtype);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn real_nan_tuple_reduction_order_and_conversion_buffers_match_numpy() {
    let mut batch = OracleBatch::default();
    for (source, dtype) in [
        (DType::F16, DType::F16),
        (DType::F16, DType::F32),
        (DType::F32, DType::F64),
        (DType::F64, DType::F32),
        (DType::F64, DType::Complex64),
    ] {
        let large = if source == DType::F16 {
            2048.0
        } else if dtype == DType::F64 {
            1e16
        } else {
            1e8
        };
        for n in [127, 128, 129, 8191, 8192, 8193] {
            for shape in [vec![2, 3, n], vec![n, 3, 2]] {
                let values = (0..shape.iter().product())
                    .map(|i| {
                        if i % 19 == 3 {
                            f64::NAN
                        } else {
                            [large, 1.0, -large, 0.25, 3.1, -0.7, -2.0][i % 7]
                        }
                    })
                    .collect();
                let array = UFuncArray::new(shape, values, source).unwrap();
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
fn real_nan_tuple_half_source_rounding_and_large_counts_match_numpy() {
    let mut batch = OracleBatch::default();
    let above_midpoint = f64::from_bits(1.00048828125_f64.to_bits() + 1);
    for (shape, values) in [
        (vec![1, 3], vec![above_midpoint, 1.0, f64::NAN]),
        (vec![1, 3], vec![65504.0, 65504.0, f64::NAN]),
        (
            vec![2, 32769],
            (0..65538)
                .map(|i| match i {
                    0 | 65537 => f64::NAN,
                    _ if i % 2 == 0 => 0.25,
                    _ => -0.25,
                })
                .collect(),
        ),
    ] {
        let array = UFuncArray::new(shape, values, DType::F16).unwrap();
        let input = batch.add_input(&array);
        for keepdims in [false, true] {
            for op in ["nansum", "nanmean", "nanvar", "nanstd"] {
                batch.push(input, &array, op, &[0, 1], keepdims, 0);
                for dtype in [DType::F16, DType::F32, DType::F64] {
                    batch.push_with_dtype(input, &array, (op, 0), &[0, 1], keepdims, dtype);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn real_tuple_joint_mean_and_zero_extents_regressions() {
    for dtype in [DType::F32, DType::F64] {
        let large = if dtype == DType::F32 { 1e8 } else { 1e16 };
        let array = UFuncArray::new(
            vec![2, 4],
            vec![large, 1.0, -large, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype,
        )
        .unwrap();
        assert_eq!(
            array.reduce_sum_axes(&[0, 1], false).unwrap().values(),
            &[4.0]
        );
        assert_eq!(
            array.reduce_mean_axes(&[1, 0], false).unwrap().values(),
            &[0.5]
        );

        let empty = UFuncArray::new(vec![0, usize::MAX, usize::MAX], vec![], dtype).unwrap();
        for op in ["sum", "prod", "mean", "var", "std"] {
            for keepdims in [false, true] {
                let result = reduce_tuple(&empty, op, &[0, 1, 2], keepdims, 0).unwrap();
                let shape = if keepdims { vec![1, 1, 1] } else { vec![] };
                assert_eq!(result.shape(), shape, "{op}: empty reduction shape");
                assert_eq!(result.dtype(), dtype);
                match op {
                    "sum" => assert_eq!(result.values(), &[0.0]),
                    "prod" => assert_eq!(result.values(), &[1.0]),
                    _ => assert!(result.values()[0].is_nan()),
                }
                let result = reduce_tuple(&empty, op, &[1, 2], keepdims, 0);
                if matches!(op, "var" | "std") {
                    assert!(matches!(
                        result,
                        Err(UFuncError::Shape(fnp_ndarray::ShapeError::Overflow))
                    ));
                    continue;
                }
                let result = result.unwrap();
                let shape = if keepdims { vec![0, 1, 1] } else { vec![0] };
                assert_eq!(result.shape(), shape, "{op}: empty output shape");
                assert!(result.values().is_empty());
            }
        }
    }
}

#[test]
fn real_tuple_shapes_empty_axes_and_axis_errors_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::F16, DType::F32, DType::F64] {
        for shape in [
            vec![],
            vec![2, 3],
            vec![2, 1, 3],
            vec![2, 3, 2, 2],
            vec![0],
            vec![2, 0, 3],
            vec![2, 3, 0],
            vec![1, 1, 1],
        ] {
            let values = (0..shape.iter().product())
                .map(|i| [0.25, -1.1, 2.5, 3.7, -0.5][i % 5])
                .collect();
            let array = UFuncArray::new(shape.clone(), values, dtype).unwrap();
            let input = batch.add_input(&array);
            let mut tuples = axis_subsets(shape.len());
            tuples.extend([
                vec![shape.len() as isize],
                vec![-(shape.len() as isize) - 1],
                vec![0, 0],
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
fn real_tuple_explicit_dtypes_preserve_exact_integers_and_wrapping() {
    let mut batch = OracleBatch::default();
    for storage in [
        ArrayStorage::Bool(vec![
            true, false, true, true, false, false, true, true, true, false, true, false,
        ]),
        ArrayStorage::I8(vec![-128, -127, -17, -1, 0, 1, 2, 3, 63, 126, 127, 7]),
        ArrayStorage::I64(vec![
            i64::MIN,
            i64::MAX,
            9_007_199_254_740_993,
            -9_007_199_254_740_995,
            (1_i64 << 62) + (1_i64 << 38) + 1,
            -((1_i64 << 62) + (1_i64 << 38) + 1),
            127,
            128,
            257,
            -1,
            0,
            3,
        ]),
        ArrayStorage::U8(vec![0, 1, 127, 128, 254, 255, 3, 7, 16, 63, 129, 251]),
        ArrayStorage::U64(vec![
            u64::MAX,
            u64::MAX - 2,
            9_007_199_254_740_993,
            9_007_199_254_740_995,
            (1_u64 << 63) + (1_u64 << 39) + 1,
            (1_u64 << 63) - 1,
            127,
            128,
            257,
            1,
            0,
            3,
        ]),
    ] {
        let array = UFuncArray::from_storage(vec![2, 2, 3], storage).unwrap();
        let input = batch.add_input(&array);
        for dtype in reduction_dtypes() {
            for axes in [vec![], vec![0, 2], vec![2, 0], vec![-1, -3], vec![0, 1, 2]] {
                for keepdims in [false, true] {
                    for op in ["sum", "prod"] {
                        batch.push_with_dtype(input, &array, (op, 0), &axes, keepdims, dtype);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn real_tuple_explicit_statistical_dtypes_match_numpy() {
    let mut batch = OracleBatch::default();
    for source in [
        DType::Bool,
        DType::I8,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::F32,
        DType::F64,
    ] {
        let pattern = if source == DType::Bool {
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        } else if source == DType::U64 {
            [0.0, 1.0, 3.0, 7.0, 2.0, 4.0]
        } else {
            [0.25, -1.1, 2.5, 3.7, -0.5, 1.0004882812500002]
        };
        let values = (0..12).map(|i| pattern[i % pattern.len()]).collect();
        let storage = ArrayStorage::F64(values).cast_to(source).unwrap();
        let array = UFuncArray::from_storage(vec![2, 3, 2], storage).unwrap();
        let input = batch.add_input(&array);
        for dtype in reduction_dtypes() {
            for axes in [vec![], vec![0, 2], vec![1, 2], vec![-1, -3], vec![0, 1, 2]] {
                for keepdims in [false, true] {
                    batch.push_with_dtype(input, &array, ("mean", 0), &axes, keepdims, dtype);
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
    }
    batch.assert_matches_numpy();
}

#[test]
fn real_tuple_explicit_dtype_empty_scalar_and_invalid_axes_match_numpy() {
    let mut batch = OracleBatch::default();
    for shape in [vec![], vec![0], vec![2, 0, 3], vec![2, 3, 0], vec![1, 1, 1]] {
        let array = UFuncArray::new(
            shape.clone(),
            vec![-0.0; shape.iter().product()],
            DType::F64,
        )
        .unwrap();
        let input = batch.add_input(&array);
        let mut tuples = axis_subsets(shape.len());
        tuples.extend([
            vec![shape.len() as isize],
            vec![-(shape.len() as isize) - 1],
            vec![0, 0],
            vec![0, 0, shape.len() as isize],
        ]);
        for dtype in [DType::Bool, DType::I8, DType::F16, DType::Complex64] {
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
fn real_tuple_explicit_dtypes_skip_unused_empty_extent_products() {
    for (shape, zero_axis) in [
        (vec![0, usize::MAX, usize::MAX], 0),
        (vec![usize::MAX, 0, usize::MAX], 1),
    ] {
        let array = UFuncArray::from_storage(shape, ArrayStorage::I64(vec![])).unwrap();
        let retained_zero_axes: Vec<isize> = (0..3)
            .filter(|&axis| axis != zero_axis)
            .map(|axis| axis as isize)
            .collect();
        for dtype in reduction_dtypes() {
            for keepdims in [false, true] {
                for product in [false, true] {
                    let reduced = if product {
                        array.reduce_prod_axes_with_dtype(&[0, 1, 2], keepdims, dtype)
                    } else {
                        array.reduce_sum_axes_with_dtype(&[0, 1, 2], keepdims, dtype)
                    }
                    .unwrap();
                    let mut shape = if keepdims { vec![1; 3] } else { vec![] };
                    if dtype.is_complex() {
                        shape.push(2);
                    }
                    assert_eq!(reduced.shape(), shape);
                    assert_eq!(reduced.dtype(), dtype);
                    let identity = if product { 1.0 } else { 0.0 };
                    let values = if dtype.is_complex() {
                        vec![identity, 0.0]
                    } else {
                        vec![identity]
                    };
                    assert_eq!(reduced.values(), values);

                    let empty = if product {
                        array.reduce_prod_axes_with_dtype(&retained_zero_axes, keepdims, dtype)
                    } else {
                        array.reduce_sum_axes_with_dtype(&retained_zero_axes, keepdims, dtype)
                    }
                    .unwrap();
                    let mut shape = if keepdims { vec![1; 3] } else { vec![0] };
                    if keepdims {
                        shape[zero_axis] = 0;
                    }
                    if dtype.is_complex() {
                        shape.push(2);
                    }
                    assert_eq!(empty.shape(), shape);
                    assert_eq!(empty.dtype(), dtype);
                    assert!(empty.values().is_empty());
                }
            }
        }
    }
    // This ordering exceeds the constructor's checked prefix product before
    // reaching zero, so it cannot be presented to a reduction API.
    assert!(matches!(
        UFuncArray::new(vec![usize::MAX, usize::MAX, 0], vec![], DType::I64),
        Err(UFuncError::Shape(fnp_ndarray::ShapeError::Overflow))
    ));
}

#[test]
fn real_tuple_explicit_dtypes_reject_unrepresentable_output_capacity() {
    let array = UFuncArray::from_storage(vec![0, usize::MAX], ArrayStorage::I64(vec![])).unwrap();
    for dtype in reduction_dtypes() {
        for keepdims in [false, true] {
            for result in [
                array.reduce_sum_axes_with_dtype(&[0], keepdims, dtype),
                array.reduce_prod_axes_with_dtype(&[0], keepdims, dtype),
                array.reduce_mean_axes_with_dtype(&[0], keepdims, dtype),
                array.reduce_var_axes_with_dtype(&[0], keepdims, 0, dtype),
                array.reduce_std_axes_with_dtype(&[0], keepdims, 0, dtype),
            ] {
                assert!(matches!(
                    result,
                    Err(UFuncError::Shape(fnp_ndarray::ShapeError::Overflow))
                ));
            }
        }
    }

    // This count fits the real plan's capacity bound but not a complex
    // accumulator. NaN statistics must reject it before allocating counts.
    let too_many_pairs = isize::MAX as usize / std::mem::size_of::<(f64, f64)>() + 1;
    let array = UFuncArray::new(vec![0, too_many_pairs], vec![], DType::F64).unwrap();
    for dtype in [DType::Complex64, DType::Complex128] {
        for keepdims in [false, true] {
            for result in [
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
    }
}

#[test]
fn real_tuple_explicit_dtype_conversion_buffers_and_source_rounding_match_numpy() {
    let mut batch = OracleBatch::default();
    for (source, dtype) in [
        (DType::I64, DType::F32),
        (DType::I64, DType::F64),
        (DType::I64, DType::Complex64),
        (DType::F16, DType::F16),
        (DType::F16, DType::F32),
        (DType::F32, DType::F64),
        (DType::F64, DType::F16),
        (DType::F64, DType::F32),
        (DType::F64, DType::Complex64),
    ] {
        let pattern = if source == DType::I64 {
            [
                (1_u64 << 60) as f64,
                1.0,
                -((1_u64 << 60) as f64),
                3.0,
                5.0,
                -2.0,
                7.0,
            ]
        } else if source == DType::F16 || dtype == DType::F16 {
            [
                2048.0,
                0.25,
                -2048.0,
                1.0004882812500002,
                -0.5,
                0.125,
                -0.125,
            ]
        } else {
            let large = if dtype == DType::F64 { 1e16 } else { 1e8 };
            [large, 1.0, -large, 0.25, 3.1, -0.7, -2.0]
        };
        for n in [8191, 8192, 8193] {
            for shape in [vec![2, 3, n], vec![n, 3, 2]] {
                let values = (0..shape.iter().product())
                    .map(|i| pattern[i % pattern.len()])
                    .collect();
                let storage = ArrayStorage::F64(values).cast_to(source).unwrap();
                let array = UFuncArray::from_storage(shape, storage).unwrap();
                let input = batch.add_input(&array);
                for axes in [vec![0, 2], vec![2, 0], vec![0, 1, 2]] {
                    for op in ["sum", "mean", "var", "std"] {
                        batch.push_with_dtype(input, &array, (op, 0), &axes, false, dtype);
                    }
                }
            }
        }
    }
    // Explicit float16 mean uses a float16 sum; the default float16 mean
    // instead accumulates in float32 before rounding the final quotient.
    let array = UFuncArray::new(vec![1, 2], vec![65504.0, 65504.0], DType::F16).unwrap();
    let input = batch.add_input(&array);
    for dtype in [DType::F16, DType::F32, DType::F64] {
        batch.push_with_dtype(input, &array, ("mean", 0), &[0, 1], false, dtype);
    }
    batch.push(input, &array, "mean", &[0, 1], false, 0);
    // A raw f64 just above a half midpoint must first become the upper half.
    // Rounding it through f32 would instead lose the source's nonzero variance.
    let above_midpoint = f64::from_bits(1.00048828125_f64.to_bits() + 1);
    let array = UFuncArray::new(vec![1, 2], vec![above_midpoint, 1.0], DType::F16).unwrap();
    assert_eq!(
        array.reduce_var_axes(&[0, 1], false, 0).unwrap().values(),
        &[4.76837158203125e-7]
    );
    assert_eq!(
        array.reduce_var_axes(&[], false, 0).unwrap().values(),
        &[0.0, 0.0]
    );
    let vector = UFuncArray::new(vec![2], vec![above_midpoint, 1.0], DType::F16).unwrap();
    for axis in [None, Some(0)] {
        assert_eq!(
            vector.reduce_var(axis, false, 0).unwrap().values(),
            &[4.76837158203125e-7]
        );
        assert_eq!(
            vector.reduce_std(axis, false, 0).unwrap().values(),
            &[0.000690460205078125]
        );
    }
    let input = batch.add_input(&array);
    for axes in [&[0, 1][..], &[0][..], &[][..]] {
        for keepdims in [false, true] {
            for op in ["sum", "prod", "mean", "var", "std"] {
                batch.push(input, &array, op, axes, keepdims, 0);
            }
            for op in ["var", "std"] {
                batch.push_with_dtype(input, &array, (op, 0), axes, keepdims, DType::F16);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn real_tuple_cancellation_and_pairwise_boundaries_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::F32, DType::F64] {
        let large = if dtype == DType::F32 { 1e8 } else { 1e16 };
        for n in [7, 8, 127, 128, 129] {
            for shape in [[2, 3, n], [n, 3, 2]] {
                let values = (0..shape.iter().product())
                    .map(|i| [large, 1.0, -large, 0.25, 3.1, -0.7, -2.0][i % 7])
                    .collect();
                let array = UFuncArray::new(shape.to_vec(), values, dtype).unwrap();
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
fn real_tuple_products_preserve_joint_left_fold() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::F32, DType::F64] {
        for shape in [
            vec![5, 3],
            vec![3, 5],
            vec![2, 3, 4],
            vec![2, 1, 3, 2],
            vec![3, 5, 1],
        ] {
            let values = (0..shape.iter().product())
                .map(|i| ((i * 17 % 97) as f64 - 48.0) / 13.0)
                .collect();
            let array = UFuncArray::new(shape.clone(), values, dtype).unwrap();
            let input = batch.add_input(&array);
            for axes in axis_subsets(shape.len()) {
                for keepdims in [false, true] {
                    batch.push(input, &array, "prod", &axes, keepdims, 0);
                    if axes.len() > 1 {
                        let permuted: Vec<isize> = axes
                            .iter()
                            .rev()
                            .map(|&axis| axis - shape.len() as isize)
                            .collect();
                        batch.push(input, &array, "prod", &permuted, keepdims, 0);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn real_tuple_nonfinite_signed_zero_and_ddof_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::F32, DType::F64] {
        for values in [
            vec![-0.0, -0.0, 0.0, 0.0],
            vec![f64::INFINITY, 1.0, -f64::INFINITY, 4.0],
            vec![1.0, f64::NAN, -2.0, 3.0],
            vec![1.1, 2.3, -1.7, 4.1],
        ] {
            let array = UFuncArray::new(vec![2, 2], values, dtype).unwrap();
            let input = batch.add_input(&array);
            for axes in axis_subsets(2) {
                for keepdims in [false, true] {
                    for op in ["sum", "prod", "mean"] {
                        batch.push(input, &array, op, &axes, keepdims, 0);
                    }
                    for ddof in [0, 1, 4, 5] {
                        for op in ["var", "std"] {
                            batch.push(input, &array, op, &axes, keepdims, ddof);
                        }
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn real_tuple_helpers_preserve_half_mean_and_integer_variance() {
    let mut batch = OracleBatch::default();
    for (shape, values) in [
        (vec![], vec![2048.9]),
        (vec![2], vec![2048.9, -2048.0]),
        (vec![2], vec![1.000_488_281_250_000_2, -1.0]),
        (vec![1, 2], vec![2048.9, -2048.0]),
        (vec![2, 1], vec![2048.9, -2048.0]),
        (vec![3], vec![2048.9, -2048.0, 0.01]),
        (vec![1, 3], vec![65519.9, -65504.0, 0.125]),
        (vec![3, 1], vec![65519.9, -65504.0, 0.125]),
    ] {
        let array = UFuncArray::new(shape.clone(), values, DType::F16).unwrap();
        let input = batch.add_input(&array);
        let mut tuples = axis_subsets(shape.len());
        tuples.push(
            (0..shape.len())
                .rev()
                .map(|axis| axis as isize - shape.len() as isize)
                .collect(),
        );
        for keepdims in [false, true] {
            for axes in &tuples {
                batch.push(input, &array, "mean", axes, keepdims, 0);
            }
            for axis in [None, Some(0), Some(-1)] {
                batch.push_single_mean(input, &array, axis, keepdims);
            }
        }
    }
    for dtype in [DType::Bool, DType::I8, DType::I64] {
        let values = if dtype == DType::Bool {
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        } else {
            vec![1.0, 3.0, -1.0, 5.0, 7.0, 9.0]
        };
        let array = UFuncArray::new(vec![2, 3], values, dtype).unwrap();
        let input = batch.add_input(&array);
        for axes in axis_subsets(2) {
            for keepdims in [false, true] {
                batch.push(input, &array, "mean", &axes, keepdims, 0);
                for ddof in [0, 1, 6, 7] {
                    for op in ["var", "std"] {
                        batch.push(input, &array, op, &axes, keepdims, ddof);
                    }
                }
            }
        }
    }
    for storage in [
        ArrayStorage::I64(vec![
            4_503_599_627_370_492,
            4_503_599_627_370_498,
            4_503_599_627_370_490,
            4_503_599_627_370_499,
            4_503_599_627_370_498,
            4_503_599_627_370_495,
        ]),
        ArrayStorage::I64(vec![
            9_007_199_254_740_993,
            9_007_199_254_740_995,
            -9_007_199_254_740_993,
            -9_007_199_254_740_995,
            i64::MIN + 1,
            i64::MAX - 1,
        ]),
        ArrayStorage::U64(vec![
            9_007_199_254_740_993,
            9_007_199_254_740_995,
            9_007_199_254_740_997,
            u64::MAX,
            u64::MAX - 2,
            u64::MAX - 4,
        ]),
    ] {
        let array = UFuncArray::from_storage(vec![2, 3], storage).unwrap();
        let input = batch.add_input(&array);
        for axes in [vec![], vec![0], vec![1], vec![0, 1], vec![-1, -2]] {
            for keepdims in [false, true] {
                batch.push(input, &array, "mean", &axes, keepdims, 0);
                for ddof in [0, 1, 6, 7] {
                    for op in ["var", "std"] {
                        batch.push(input, &array, op, &axes, keepdims, ddof);
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn real_tuple_mean_conversion_buffers_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::I64, DType::F16] {
        let pattern = if dtype == DType::I64 {
            [
                (1_u64 << 60) as f64,
                1.0,
                -((1_u64 << 60) as f64),
                3.0,
                5.0,
                -2.0,
                7.0,
            ]
        } else {
            [65504.0, 0.25, -65504.0, 1.0, -0.5, 0.125, -0.125]
        };
        for n in [8191, 8192, 8193, 16385] {
            for shape in [vec![2, n], vec![n, 2]] {
                let values: Vec<f64> = (0..shape.iter().product())
                    .map(|i| pattern[i % 7])
                    .collect();
                let array = if dtype == DType::I64 {
                    UFuncArray::from_storage(
                        shape,
                        ArrayStorage::I64(values.into_iter().map(|value| value as i64).collect()),
                    )
                } else {
                    UFuncArray::new(shape, values, dtype)
                }
                .unwrap();
                let input = batch.add_input(&array);
                for axes in [vec![0], vec![1], vec![0, 1], vec![-1, -2]] {
                    for keepdims in [false, true] {
                        batch.push(input, &array, "mean", &axes, keepdims, 0);
                    }
                }
                for axis in [None, Some(0), Some(1), Some(-1)] {
                    batch.push_single_mean(input, &array, axis, false);
                }
            }
        }
        // A retained middle dimension splits contiguous runs, and each cast
        // buffer must continue the output accumulated by an earlier run.
        let shape = vec![2, 3, 8193];
        let values: Vec<f64> = (0..shape.iter().product())
            .map(|i| pattern[i % 7])
            .collect();
        let array = if dtype == DType::I64 {
            UFuncArray::from_storage(
                shape,
                ArrayStorage::I64(values.into_iter().map(|value| value as i64).collect()),
            )
        } else {
            UFuncArray::new(shape, values, dtype)
        }
        .unwrap();
        let input = batch.add_input(&array);
        for axes in [vec![0, 2], vec![0, 1], vec![1, 2], vec![0, 1, 2]] {
            for keepdims in [false, true] {
                batch.push(input, &array, "mean", &axes, keepdims, 0);
            }
        }
    }
    for block_len in [4096, 8192] {
        let n = 2 * block_len + 1;
        let values = (0..n)
            .map(|i| {
                if i < block_len {
                    65504.0
                } else if i < 2 * block_len {
                    -65504.0
                } else {
                    1.0
                }
            })
            .collect();
        let array = UFuncArray::new(vec![n], values, DType::F16).unwrap();
        assert_eq!(
            array.reduce_mean(None, false).unwrap().values(),
            &[1.0 / (2 * block_len) as f64],
            "half mean must preserve the final cast-buffer remainder"
        );
        let input = batch.add_input(&array);
        for keepdims in [false, true] {
            for axes in [vec![0], vec![-1]] {
                batch.push(input, &array, "mean", &axes, keepdims, 0);
            }
            for axis in [None, Some(0), Some(-1)] {
                batch.push_single_mean(input, &array, axis, keepdims);
            }
        }
    }
    let n = 16385;
    let values = (0..2 * 3 * n)
        .map(|i| match i % n {
            0..8192 => 65504.0,
            8192..16384 => -65504.0,
            _ => 1.0,
        })
        .collect();
    let array = UFuncArray::new(vec![2, 3, n], values, DType::F16).unwrap();
    assert_eq!(
        array.reduce_mean_axes(&[0, 2], false).unwrap().values(),
        &[1.0 / 32768.0; 3],
        "cast buffers must add directly to the previous output, not a run total"
    );
    let input = batch.add_input(&array);
    for axes in [vec![0, 2], vec![2, 0], vec![-3, -1], vec![0, 1, 2]] {
        for keepdims in [false, true] {
            batch.push(input, &array, "mean", &axes, keepdims, 0);
        }
    }
    batch.assert_matches_numpy();
}
