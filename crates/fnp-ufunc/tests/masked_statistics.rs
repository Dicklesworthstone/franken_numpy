//! Statistical reductions combine logical axes, typed accumulators, and masks.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{
    FloatErrorMode, ReductionOptions, StatisticalReductionOptions, UFuncArray, UFuncError, errstate,
};
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
        _ => panic!("non-numeric statistics fixture: {dtype:?}"),
    }
}

fn complex(shape: &[usize], values: &[(f64, f64)], dtype: DType) -> UFuncArray {
    let mut shape = shape.to_vec();
    shape.push(2);
    UFuncArray::new(
        shape,
        values
            .iter()
            .flat_map(|&(real, imag)| [real, imag])
            .collect(),
        dtype,
    )
    .unwrap()
}

fn boolean(shape: &[usize], values: &[bool]) -> UFuncArray {
    UFuncArray::new(
        shape.to_vec(),
        values
            .iter()
            .map(|&value| if value { 1.0 } else { 0.0 })
            .collect(),
        DType::Bool,
    )
    .unwrap()
}

fn reduce(
    array: &UFuncArray,
    operation: (&str, usize),
    options: StatisticalReductionOptions<'_>,
) -> Result<UFuncArray, UFuncError> {
    match operation {
        ("mean", _) => array.reduce_mean_with_options(options),
        ("var", ddof) => array.reduce_var_with_options(options, ddof),
        ("std", ddof) => array.reduce_std_with_options(options, ddof),
        ("nanmean", _) => array.nanmean_with_options(options),
        ("nanvar", ddof) => array.nanvar_with_options(options, ddof),
        ("nanstd", ddof) => array.nanstd_with_options(options, ddof),
        _ => panic!("unknown statistics fixture operation: {operation:?}"),
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
        let index = self.inputs.len();
        let mut shape = array.shape().to_vec();
        if array.dtype().is_complex() {
            assert_eq!(shape.pop(), Some(2));
        }
        let values: Vec<String> = if array.dtype().is_integer() || array.dtype() == DType::Bool {
            let storage = array.to_storage().unwrap();
            (0..storage.len())
                .map(|index| storage.get_i128(index).unwrap().to_string())
                .collect()
        } else {
            // Preserve wider working values: both implementations must round
            // to the declared source precision before reducing them.
            array.values().iter().map(f64::to_string).collect()
        };
        self.inputs.push(json!({
            "shape": shape, "dtype": dtype_name(array.dtype()), "values": values,
        }));
        index
    }

    fn push(
        &mut self,
        input: usize,
        array: &UFuncArray,
        operation: (&str, usize),
        options: StatisticalReductionOptions<'_>,
        mask: Option<usize>,
    ) {
        assert_eq!(mask.is_some(), options.where_mask.is_some());
        self.cases.push(json!({
            "input": input, "op": operation.0, "ddof": operation.1,
            "axes": options.axes, "keepdims": options.keepdims,
            "dtype": options.dtype.map(dtype_name), "mask": mask,
        }));
        let _errors = errstate(Some(FloatErrorMode::Ignore), None, None, None, None);
        self.actual.push(reduce(array, operation, options));
    }

    fn push_corrected(
        &mut self,
        input: usize,
        array: &UFuncArray,
        operation: (&str, f64),
        options: StatisticalReductionOptions<'_>,
        auxiliary: (Option<usize>, Option<(usize, &UFuncArray)>),
    ) {
        let (mask, mean) = auxiliary;
        let (op, correction) = operation;
        self.cases.push(json!({
            "input": input, "op": op, "correction": correction.to_string(),
            "axes": options.axes, "keepdims": options.keepdims,
            "dtype": options.dtype.map(dtype_name), "mask": mask,
            "mean": mean.map(|(id, _)| id),
        }));
        let mean = mean.map(|(_, array)| array);
        let _errors = errstate(Some(FloatErrorMode::Ignore), None, None, None, None);
        self.actual.push(match op {
            "var" => array.reduce_var_with_correction(options, correction, mean),
            "std" => array.reduce_std_with_correction(options, correction, mean),
            "nanvar" => array.nanvar_with_correction(options, correction, mean),
            "nanstd" => array.nanstd_with_correction(options, correction, mean),
            _ => panic!("unknown corrected statistics operation: {op}"),
        });
    }

    fn push_arithmetic(
        &mut self,
        input: usize,
        array: &UFuncArray,
        operation: &str,
        options: ReductionOptions<'_>,
        auxiliary: (Option<usize>, Option<usize>),
    ) {
        let (mask, initial) = auxiliary;
        assert_eq!(mask.is_some(), options.where_mask.is_some());
        assert_eq!(initial.is_some(), options.initial.is_some());
        self.cases.push(json!({
            "input": input, "op": operation, "axes": options.axes,
            "keepdims": options.keepdims, "dtype": options.dtype.map(dtype_name),
            "mask": mask, "initial": initial,
        }));
        let _errors = errstate(Some(FloatErrorMode::Ignore), None, None, None, None);
        self.actual.push(match operation {
            "nansum" => array.nansum_with_options(options),
            "nanprod" => array.nanprod_with_options(options),
            _ => panic!("unknown NaN arithmetic fixture operation: {operation}"),
        });
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np

warnings.simplefilter('ignore')
np.seterr(all='ignore')
payload = json.load(sys.stdin)
inputs = []
for item in payload['inputs']:
    dtype = np.dtype(item['dtype'])
    if dtype.kind == 'c':
        values = [float(value) for value in item['values']]
        array = np.empty(len(values) // 2, dtype=dtype)
        array.real = values[0::2]
        array.imag = values[1::2]
    else:
        parse = int if dtype.kind in 'biu' else float
        array = np.asarray([parse(value) for value in item['values']], dtype=dtype)
    inputs.append(array.reshape(item['shape']))
rows = []
for case in payload['cases']:
    axis = None if case['axes'] is None else tuple(case['axes'])
    kw = dict(axis=axis, keepdims=case['keepdims'])
    if case['dtype'] is not None:
        kw['dtype'] = case['dtype']
    if case['mask'] is not None:
        kw['where'] = inputs[case['mask']]
    if case.get('initial') is not None:
        kw['initial'] = inputs[case['initial']]
    if case.get('mean') is not None:
        kw['mean'] = inputs[case['mean']]
    if 'correction' in case:
        kw['correction'] = float(case['correction'])
    elif case['op'] in ('var', 'std', 'nanvar', 'nanstd'):
        kw['ddof'] = case['ddof']
    try:
        result = np.asarray(getattr(np, case['op'])(inputs[case['input']], **kw))
        if result.dtype.kind == 'c':
            parts = np.empty(result.size * 2, dtype=np.float64)
            parts[0::2] = result.real.ravel()
            parts[1::2] = result.imag.ravel()
        else:
            parts = result.astype(np.float64).ravel()
        integers = [str(int(value)) for value in result.flat] if result.dtype.kind in 'biu' else None
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
            .expect("launch NumPy masked statistics oracle");
        let mut stdin = child.stdin.take().unwrap();
        serde_json::to_writer(
            &mut stdin,
            &json!({"inputs": self.inputs, "cases": self.cases}),
        )
        .unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy masked statistics oracle: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.actual.len());
        let case_count = expected.len();
        let nan_case_count = self
            .cases
            .iter()
            .filter(|case| case["op"].as_str().is_some_and(|op| op.starts_with("nan")))
            .count();
        for ((case, actual), expected) in self.cases.iter().zip(self.actual).zip(expected) {
            if expected["error"] == "AxisError" {
                assert!(
                    matches!(actual, Err(UFuncError::AxisOutOfBounds { axis, ndim })
                        if json!(axis) == expected["axis"] && json!(ndim) == expected["ndim"]),
                    "{case}: expected {expected}, got {actual:?}"
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
                let integers: Vec<String> = (0..storage.len())
                    .map(|index| storage.get_i128(index).unwrap().to_string())
                    .collect();
                assert_eq!(
                    json!(integers),
                    expected["integers"],
                    "{case}: exact integers"
                );
            }
            let bits = expected["bits"].as_array().unwrap();
            assert_eq!(actual.values().len(), bits.len(), "{case}: component count");
            for (index, (&got, expected)) in actual.values().iter().zip(bits).enumerate() {
                let expected = f64::from_bits(expected.as_u64().unwrap());
                if expected.is_nan() {
                    assert!(
                        got.is_nan(),
                        "{case}: element {index}: got {got}, expected NaN"
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
        eprintln!(
            "NUMPY_REDUCTION_ORACLE verified_cases={case_count} nan_cases={nan_case_count} test={}",
            std::thread::current().name().unwrap_or("unnamed")
        );
    }
}

#[test]
fn variance_supplied_means_and_real_corrections_match_numpy() {
    let mut batch = OracleBatch::default();
    let real = UFuncArray::new(
        vec![2, 2, 3],
        (0..12).map(|i| i as f64 - 4.0).collect(),
        DType::F32,
    )
    .unwrap();
    let complex_input = complex(
        &[2, 2, 3],
        &(0..12)
            .map(|i| (i as f64 - 3.0, (i % 4) as f64 + 0.5))
            .collect::<Vec<_>>(),
        DType::Complex64,
    );
    let integer =
        UFuncArray::from_storage(vec![2, 2, 3], ArrayStorage::I8(vec![110; 12])).unwrap();
    let mask = boolean(&[1, 2, 3], &[true, false, true, false, false, false]);
    let mask_id = batch.add_input(&mask);
    let means = [
        UFuncArray::scalar(4.0, DType::I8),
        UFuncArray::new(vec![1, 2, 1], vec![1.25, 2.75], DType::F64).unwrap(),
        complex(
            &[3],
            &[(1.0, -2.0), (0.5, 4.0), (3.0, 2.5)],
            DType::Complex64,
        ),
        UFuncArray::new(vec![4], vec![0.0; 4], DType::F32).unwrap(),
    ];
    let ids: Vec<_> = means.iter().map(|mean| batch.add_input(mean)).collect();
    for array in [&real, &complex_input, &integer] {
        let input = batch.add_input(array);
        for mean in std::iter::once(None).chain(ids.iter().copied().zip(&means).map(Some)) {
            for correction in [
                -1.5,
                0.5,
                4.0,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ] {
                for op in ["var", "std"] {
                    for keepdims in [false, true] {
                        let options = StatisticalReductionOptions {
                            axes: Some(&[0, -1]),
                            where_mask: Some(&mask),
                            keepdims,
                            dtype: None,
                        };
                        batch.push_corrected(
                            input,
                            array,
                            (op, correction),
                            options,
                            (Some(mask_id), mean),
                        );
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn masked_statistics_broadcast_masks_and_count_joint_axes() {
    let mut batch = OracleBatch::default();
    let masks = [
        boolean(&[], &[true]),
        boolean(&[], &[false]),
        boolean(&[4], &[true, false, true, true]),
        boolean(&[1, 3, 1], &[true, false, true]),
        boolean(
            &[2, 1, 4],
            &[true, false, true, false, false, true, true, true],
        ),
        boolean(
            &[2, 3, 4],
            &(0..24).map(|index| index % 5 < 2).collect::<Vec<_>>(),
        ),
    ];
    let mask_ids: Vec<usize> = masks.iter().map(|mask| batch.add_input(mask)).collect();
    let axes: &[Option<&[isize]>] = &[
        None,
        Some(&[]),
        Some(&[0]),
        Some(&[1]),
        Some(&[2]),
        Some(&[0, 2]),
        Some(&[2, 0]),
        Some(&[-3, -1]),
        Some(&[0, 1, 2]),
    ];
    for dtype in [
        DType::Bool,
        DType::I16,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        let values: Vec<f64> = (0..24).map(|index| (index % 9) as f64 - 4.0).collect();
        let array = if dtype.is_complex() {
            complex(
                &[2, 3, 4],
                &values
                    .iter()
                    .enumerate()
                    .map(|(index, &value)| (value, (index % 5) as f64 - 2.0))
                    .collect::<Vec<_>>(),
                dtype,
            )
        } else {
            UFuncArray::new(vec![2, 3, 4], values, dtype).unwrap()
        };
        let input = batch.add_input(&array);
        for (&mask_id, mask) in mask_ids.iter().zip(&masks) {
            for &axes in axes {
                for keepdims in [false, true] {
                    for operation in [("mean", 0), ("var", 0), ("var", 1), ("std", 0)] {
                        batch.push(
                            input,
                            &array,
                            operation,
                            StatisticalReductionOptions {
                                axes,
                                keepdims,
                                dtype: None,
                                where_mask: Some(mask),
                            },
                            Some(mask_id),
                        );
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn masked_statistics_explicit_accumulators_preserve_source_precision() {
    let mut batch = OracleBatch::default();
    let masks = [
        boolean(&[2, 1], &[true, false]),
        boolean(&[4], &[true, false, true, true]),
    ];
    let mask_ids: Vec<usize> = masks.iter().map(|mask| batch.add_input(mask)).collect();
    let mut arrays = Vec::new();
    for dtype in [DType::Bool, DType::I16, DType::F16, DType::F32, DType::F64] {
        arrays.push(
            UFuncArray::new(
                vec![2, 4],
                vec![1.0003, 4.75, -3.125, 8.5, 2.001, 7.0, 9.25, -1.5],
                dtype,
            )
            .unwrap(),
        );
    }
    for dtype in [DType::Complex64, DType::Complex128] {
        arrays.push(complex(
            &[2, 4],
            &[
                (1.00000003, 2.0),
                (4.75, -8.5),
                (-3.125, 6.5),
                (8.5, -1.0),
                (2.00000006, -3.5),
                (7.0, 5.0),
                (9.25, 1.5),
                (-1.5, -4.0),
            ],
            dtype,
        ));
    }
    for array in arrays {
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
            for (&mask_id, mask) in mask_ids.iter().zip(&masks) {
                for axes in [None, Some(&[0][..]), Some(&[1][..]), Some(&[][..])] {
                    for keepdims in [false, true] {
                        for operation in [("mean", 0), ("var", 0), ("std", 0)] {
                            batch.push(
                                input,
                                &array,
                                operation,
                                StatisticalReductionOptions {
                                    axes,
                                    keepdims,
                                    dtype: Some(dtype),
                                    where_mask: Some(mask),
                                },
                                Some(mask_id),
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
fn masked_statistics_retain_exact_integer_sources_before_conversion() {
    let mut batch = OracleBatch::default();
    let masks = [
        boolean(&[4], &[true, false, true, true]),
        boolean(&[2, 1], &[false, true]),
    ];
    let mask_ids: Vec<usize> = masks.iter().map(|mask| batch.add_input(mask)).collect();
    let arrays = [
        UFuncArray::from_storage(
            vec![2, 4],
            ArrayStorage::I64(vec![
                (1_i64 << 53) + 1,
                (1_i64 << 53) + 3,
                (1_i64 << 53) + 5,
                (1_i64 << 53) + 7,
                i64::MAX,
                i64::MIN,
                -3,
                9,
            ]),
        )
        .unwrap(),
        UFuncArray::from_storage(
            vec![2, 4],
            ArrayStorage::U64(vec![
                (1_u64 << 53) + 1,
                (1_u64 << 53) + 3,
                (1_u64 << 63) + 5,
                u64::MAX,
                1,
                3,
                5,
                7,
            ]),
        )
        .unwrap(),
    ];
    for array in arrays {
        let input = batch.add_input(&array);
        for dtype in [
            None,
            Some(DType::I8),
            Some(DType::I64),
            Some(DType::U64),
            Some(DType::F32),
            Some(DType::F64),
            Some(DType::Complex128),
        ] {
            for (&mask_id, mask) in mask_ids.iter().zip(&masks) {
                for axes in [None, Some(&[0][..]), Some(&[1][..]), Some(&[][..])] {
                    for operation in [
                        ("mean", 0),
                        ("var", 0),
                        ("std", 1),
                        ("nanmean", 0),
                        ("nanvar", 0),
                        ("nanstd", 1),
                    ] {
                        batch.push(
                            input,
                            &array,
                            operation,
                            StatisticalReductionOptions {
                                axes,
                                keepdims: false,
                                dtype,
                                where_mask: Some(mask),
                            },
                            Some(mask_id),
                        );
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn masked_statistics_keep_selected_runs_and_conversion_buffer_boundaries() {
    let mut batch = OracleBatch::default();
    for width in [129, 8193] {
        let mask = boolean(
            &[1, width],
            &(0..width)
                .map(|index| !matches!(index % 137, 7 | 8 | 126 | 127 | 128))
                .collect::<Vec<_>>(),
        );
        let mask_id = batch.add_input(&mask);
        for (source, accumulator, large) in [
            (DType::F16, None, 16.0),
            (DType::F16, Some(DType::F32), 16.0),
            (DType::F32, None, 16_777_216.0),
            (DType::F32, Some(DType::F64), 16_777_216.0),
            (DType::F64, Some(DType::F32), 16_777_216.0),
            (DType::F64, None, 1.0e16),
        ] {
            let values: Vec<f64> = (0..2 * width)
                .map(|index| match index % 7 {
                    0 => large,
                    1 => 1.0,
                    2 => -large,
                    3 => -0.0,
                    4 => 0.125,
                    5 => -0.0625,
                    _ => 2.0,
                })
                .collect();
            let array = UFuncArray::new(vec![2, width], values, source).unwrap();
            let input = batch.add_input(&array);
            for axes in [None, Some(&[0][..]), Some(&[1][..]), Some(&[1, 0][..])] {
                for operation in [("mean", 0), ("var", 0), ("std", 1)] {
                    batch.push(
                        input,
                        &array,
                        operation,
                        StatisticalReductionOptions {
                            axes,
                            keepdims: false,
                            dtype: accumulator,
                            where_mask: Some(&mask),
                        },
                        Some(mask_id),
                    );
                    batch.push(
                        input,
                        &array,
                        operation,
                        StatisticalReductionOptions {
                            axes,
                            keepdims: false,
                            dtype: accumulator,
                            where_mask: None,
                        },
                        None,
                    );
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn masked_statistics_exclude_nonfinite_values_without_changing_ddof_rules() {
    let mut batch = OracleBatch::default();
    let masks = [
        boolean(
            &[2, 4],
            &[true, false, false, true, true, true, false, false],
        ),
        boolean(&[], &[false]),
        boolean(&[], &[true]),
    ];
    let mask_ids: Vec<usize> = masks.iter().map(|mask| batch.add_input(mask)).collect();
    let mut arrays = Vec::new();
    for dtype in [DType::F16, DType::F32, DType::F64] {
        arrays.push(
            UFuncArray::new(
                vec![2, 4],
                vec![
                    1.0,
                    f64::NAN,
                    f64::INFINITY,
                    3.0,
                    -0.0,
                    -0.0,
                    f64::NEG_INFINITY,
                    1.0,
                ],
                dtype,
            )
            .unwrap(),
        );
    }
    for dtype in [DType::Complex64, DType::Complex128] {
        arrays.push(complex(
            &[2, 4],
            &[
                (1.0, 2.0),
                (f64::NAN, 1.0),
                (1.0, f64::INFINITY),
                (3.0, 4.0),
                (-0.0, -0.0),
                (-0.0, -0.0),
                (f64::NEG_INFINITY, 2.0),
                (1.0, f64::NAN),
            ],
            dtype,
        ));
    }
    for array in arrays {
        let input = batch.add_input(&array);
        for dtype in [None, Some(DType::F32), Some(DType::Complex128)] {
            for (&mask_id, mask) in mask_ids.iter().zip(&masks) {
                for axes in [None, Some(&[0][..]), Some(&[1][..]), Some(&[][..])] {
                    for keepdims in [false, true] {
                        for operation in [
                            ("mean", 0),
                            ("var", 0),
                            ("var", 2),
                            ("var", 9),
                            ("std", 0),
                            ("std", 2),
                            ("nanmean", 0),
                            ("nanvar", 0),
                            ("nanvar", 2),
                            ("nanvar", 9),
                            ("nanstd", 0),
                            ("nanstd", 2),
                        ] {
                            batch.push(
                                input,
                                &array,
                                operation,
                                StatisticalReductionOptions {
                                    axes,
                                    keepdims,
                                    dtype,
                                    where_mask: Some(mask),
                                },
                                Some(mask_id),
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
fn masked_statistics_empty_scalar_and_invalid_axes_match_numpy() {
    let mut batch = OracleBatch::default();
    let masks = [boolean(&[], &[true]), boolean(&[], &[false])];
    let mask_ids: Vec<usize> = masks.iter().map(|mask| batch.add_input(mask)).collect();
    for shape in [vec![], vec![0], vec![2, 0, 3], vec![0, 2, 3]] {
        for dtype in [DType::I64, DType::F16, DType::F64, DType::Complex64] {
            let array = if dtype.is_complex() {
                complex(
                    &shape,
                    if shape.is_empty() { &[(2.0, 3.0)] } else { &[] },
                    dtype,
                )
            } else {
                UFuncArray::new(
                    shape.clone(),
                    if shape.is_empty() { vec![2.0] } else { vec![] },
                    dtype,
                )
                .unwrap()
            };
            let input = batch.add_input(&array);
            for (&mask_id, mask) in mask_ids.iter().zip(&masks) {
                for axes in [
                    None,
                    Some(&[][..]),
                    Some(&[0][..]),
                    Some(&[-1][..]),
                    Some(&[0, 2][..]),
                    Some(&[0, 0][..]),
                    Some(&[6][..]),
                ] {
                    for keepdims in [false, true] {
                        for operation in [
                            ("mean", 0),
                            ("var", 0),
                            ("std", 1),
                            ("nanmean", 0),
                            ("nanvar", 0),
                            ("nanstd", 1),
                        ] {
                            batch.push(
                                input,
                                &array,
                                operation,
                                StatisticalReductionOptions {
                                    axes,
                                    keepdims,
                                    dtype: None,
                                    where_mask: Some(mask),
                                },
                                Some(mask_id),
                            );
                        }
                        for operation in ["nansum", "nanprod"] {
                            batch.push_arithmetic(
                                input,
                                &array,
                                operation,
                                ReductionOptions {
                                    axes,
                                    keepdims,
                                    where_mask: Some(mask),
                                    ..Default::default()
                                },
                                (Some(mask_id), None),
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
fn masked_statistics_reject_nonboolean_and_nonbroadcast_masks() {
    let mut batch = OracleBatch::default();
    let array = UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::F64).unwrap();
    let input = batch.add_input(&array);
    let masks = [
        UFuncArray::new(vec![3], vec![1.0; 3], DType::I64).unwrap(),
        UFuncArray::new(vec![], vec![1.0], DType::F32).unwrap(),
        complex(&[3], &[(1.0, 0.0); 3], DType::Complex128),
        boolean(&[4], &[true; 4]),
        boolean(&[2, 3, 1], &[true; 6]),
    ];
    for mask in masks {
        let mask_id = batch.add_input(&mask);
        for axes in [None, Some(&[1][..]), Some(&[][..])] {
            for operation in [
                ("mean", 0),
                ("var", 0),
                ("std", 0),
                ("nanmean", 0),
                ("nanvar", 0),
                ("nanstd", 0),
            ] {
                batch.push(
                    input,
                    &array,
                    operation,
                    StatisticalReductionOptions {
                        axes,
                        where_mask: Some(&mask),
                        ..Default::default()
                    },
                    Some(mask_id),
                );
            }
            for operation in ["nansum", "nanprod"] {
                batch.push_arithmetic(
                    input,
                    &array,
                    operation,
                    ReductionOptions {
                        axes,
                        where_mask: Some(&mask),
                        ..Default::default()
                    },
                    (Some(mask_id), None),
                );
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn masked_statistics_validate_complex_layout_and_impossible_output_sizes() {
    let mask = boolean(&[], &[true]);
    let malformed = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::Complex128).unwrap();
    for operation in [
        ("mean", 0),
        ("var", 0),
        ("std", 0),
        ("nanmean", 0),
        ("nanvar", 0),
        ("nanstd", 0),
    ] {
        assert!(
            reduce(
                &malformed,
                operation,
                StatisticalReductionOptions {
                    where_mask: Some(&mask),
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    for dtype in [DType::F64, DType::Complex128] {
        let array = if dtype.is_complex() {
            complex(&[usize::MAX, 0, 3], &[], dtype)
        } else {
            UFuncArray::new(vec![usize::MAX, 0, 3], vec![], dtype).unwrap()
        };
        for operation in [
            ("mean", 0),
            ("var", 0),
            ("std", 0),
            ("nanmean", 0),
            ("nanvar", 0),
            ("nanstd", 0),
        ] {
            assert!(
                reduce(
                    &array,
                    operation,
                    StatisticalReductionOptions {
                        axes: Some(&[1]),
                        where_mask: Some(&mask),
                        ..Default::default()
                    }
                )
                .is_err()
            );
            let empty = reduce(
                &array,
                operation,
                StatisticalReductionOptions {
                    axes: Some(&[0]),
                    where_mask: Some(&mask),
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(empty.values().is_empty());
            let expected = if dtype.is_complex() && matches!(operation.0, "mean" | "nanmean") {
                vec![0, 3, 2]
            } else {
                vec![0, 3]
            };
            assert_eq!(empty.shape(), expected);
        }
    }
}

#[test]
fn nan_statistics_combine_valid_counts_masks_and_accumulator_dtypes() {
    let mut batch = OracleBatch::default();
    let masks = [
        boolean(&[], &[true]),
        boolean(&[], &[false]),
        boolean(&[4], &[true, false, true, true]),
        boolean(&[1, 3, 1], &[true, false, true]),
        boolean(
            &[2, 1, 4],
            &[true, false, true, false, false, true, true, true],
        ),
    ];
    let mask_ids: Vec<usize> = masks.iter().map(|mask| batch.add_input(mask)).collect();
    for source_dtype in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        let parts: Vec<(f64, f64)> = (0..24)
            .map(|index| {
                let real = if index % 7 == 0 {
                    f64::NAN
                } else {
                    (index % 9) as f64 * 1.00000003 - 4.0
                };
                let imag = if index % 5 == 0 {
                    f64::NAN
                } else {
                    (index % 5) as f64 - 2.0
                };
                (real, imag)
            })
            .collect();
        let array = if source_dtype.is_complex() {
            complex(&[2, 3, 4], &parts, source_dtype)
        } else {
            UFuncArray::new(
                vec![2, 3, 4],
                parts.iter().map(|value| value.0).collect(),
                source_dtype,
            )
            .unwrap()
        };
        let input = batch.add_input(&array);
        for dtype in [
            None,
            Some(DType::F16),
            Some(DType::F32),
            Some(DType::F64),
            Some(DType::Complex64),
            Some(DType::Complex128),
            Some(DType::I64),
            Some(DType::Bool),
        ] {
            for (&mask_id, mask) in mask_ids.iter().zip(&masks) {
                for axes in [
                    None,
                    Some(&[][..]),
                    Some(&[0][..]),
                    Some(&[1][..]),
                    Some(&[2][..]),
                    Some(&[2, 0][..]),
                    Some(&[-3, -1][..]),
                ] {
                    for keepdims in [false, true] {
                        for operation in
                            [("nanmean", 0), ("nanvar", 0), ("nanvar", 2), ("nanstd", 1)]
                        {
                            batch.push(
                                input,
                                &array,
                                operation,
                                StatisticalReductionOptions {
                                    axes,
                                    keepdims,
                                    dtype,
                                    where_mask: Some(mask),
                                },
                                Some(mask_id),
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
fn nan_variance_supplied_means_and_real_corrections_match_numpy() {
    let mut batch = OracleBatch::default();
    let mask = boolean(&[3], &[true, true, false]);
    let mask_id = batch.add_input(&mask);
    let means = [
        // Distinct from the measured lane means so silently recomputing a
        // supplied mean would change the expected result.
        UFuncArray::scalar(3.125, DType::F64),
        UFuncArray::new(vec![2, 1], vec![2.00000003, -1.25], DType::F32).unwrap(),
        complex(
            &[3],
            &[(0.0, 0.5), (2.0, 3.0), (-1.0, 0.0)],
            DType::Complex64,
        ),
        UFuncArray::new(vec![4], vec![0.0; 4], DType::F32).unwrap(),
    ];
    let mean_refs: Vec<_> = means
        .iter()
        .map(|mean| Some((batch.add_input(mean), mean)))
        .chain(std::iter::once(None))
        .collect();
    let parts = [
        (f64::NAN, 7.0),
        (6.0, f64::NAN),
        (1.0, -3.0),
        (1.25000003, 1.5),
        (2.75, -0.5),
        (f64::NAN, 0.0),
    ];
    for source in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        let array = if source.is_complex() {
            complex(&[2, 3], &parts, source)
        } else {
            UFuncArray::new(
                vec![2, 3],
                parts.iter().map(|pair| pair.0).collect(),
                source,
            )
            .unwrap()
        };
        let input = batch.add_input(&array);
        for &mean in &mean_refs {
            for axes in [None, Some(&[-1][..]), Some(&[1, 0][..]), Some(&[][..])] {
                for correction in [
                    -0.5,
                    0.5,
                    1.0,
                    2.0,
                    3.25,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    f64::NAN,
                ] {
                    for dtype in [None, Some(DType::F64), Some(DType::Complex64)] {
                        for (op, keepdims) in [("nanvar", false), ("nanstd", true)] {
                            batch.push_corrected(
                                input,
                                &array,
                                (op, correction),
                                StatisticalReductionOptions {
                                    axes,
                                    keepdims,
                                    dtype,
                                    where_mask: Some(&mask),
                                },
                                (Some(mask_id), mean),
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
fn nan_statistics_keep_nan_slots_in_selected_runs_and_conversion_buffers() {
    let mut batch = OracleBatch::default();
    for width in [129, 8193] {
        let mask = boolean(
            &[1, width],
            &(0..width)
                .map(|index| !matches!(index % 137, 7 | 8 | 126 | 127 | 128))
                .collect::<Vec<_>>(),
        );
        let mask_id = batch.add_input(&mask);
        for (source, large) in [
            (DType::F16, 16.0),
            (DType::F32, 16_777_216.0),
            (DType::F64, 1.0e16),
            (DType::Complex64, 16_777_216.0),
        ] {
            let parts: Vec<(f64, f64)> = (0..2 * width)
                .map(|index| {
                    let real = match index % 11 {
                        0 => large,
                        1 => f64::NAN,
                        2 => 1.0,
                        3 => -large,
                        4 => -0.0,
                        _ => 0.125,
                    };
                    (
                        real,
                        if index % 17 == 0 {
                            f64::NAN
                        } else {
                            (index % 5) as f64 - 2.0
                        },
                    )
                })
                .collect();
            let array = if source.is_complex() {
                complex(&[2, width], &parts, source)
            } else {
                UFuncArray::new(
                    vec![2, width],
                    parts.iter().map(|value| value.0).collect(),
                    source,
                )
                .unwrap()
            };
            let input = batch.add_input(&array);
            for dtype in [None, Some(DType::F64)] {
                for axes in [None, Some(&[0][..]), Some(&[1][..]), Some(&[1, 0][..])] {
                    for (mask, mask_id) in [(None, None), (Some(&mask), Some(mask_id))] {
                        for operation in [("nanmean", 0), ("nanvar", 0), ("nanstd", 1)] {
                            batch.push(
                                input,
                                &array,
                                operation,
                                StatisticalReductionOptions {
                                    axes,
                                    keepdims: false,
                                    dtype,
                                    where_mask: mask,
                                },
                                mask_id,
                            );
                        }
                        batch.push_arithmetic(
                            input,
                            &array,
                            "nansum",
                            ReductionOptions {
                                axes,
                                dtype,
                                where_mask: mask,
                                ..Default::default()
                            },
                            (mask_id, None),
                        );
                    }
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn nan_arithmetic_combines_typed_initials_masks_and_joint_axes() {
    let mut batch = OracleBatch::default();
    let masks = [
        boolean(&[], &[false]),
        boolean(&[4], &[true, false, true, true]),
        boolean(&[2, 1], &[false, true]),
    ];
    let mask_ids: Vec<usize> = masks.iter().map(|mask| batch.add_input(mask)).collect();
    let initials = [
        UFuncArray::from_storage(vec![], ArrayStorage::U64(vec![(1_u64 << 53) + 3])).unwrap(),
        UFuncArray::scalar(-0.0, DType::F64),
        complex(&[], &[(1.25, -2.5)], DType::Complex128),
        UFuncArray::scalar(f64::NAN, DType::F64),
    ];
    let initial_ids: Vec<usize> = initials
        .iter()
        .map(|initial| batch.add_input(initial))
        .collect();
    let mut arrays = vec![
        UFuncArray::from_storage(
            vec![2, 4],
            ArrayStorage::I64(vec![(1_i64 << 53) + 1, 3, -7, 1, 2, i64::MAX, 0, -1]),
        )
        .unwrap(),
        UFuncArray::from_storage(
            vec![2, 4],
            ArrayStorage::U64(vec![(1_u64 << 53) + 1, 3, u64::MAX, 1, 2, 5, 0, 7]),
        )
        .unwrap(),
    ];
    for dtype in [DType::F16, DType::F32, DType::F64] {
        arrays.push(
            UFuncArray::new(
                vec![2, 4],
                vec![1.0003, f64::NAN, -3.0, -0.0, 2.0, 0.5, f64::NAN, 4.0],
                dtype,
            )
            .unwrap(),
        );
    }
    for dtype in [DType::Complex64, DType::Complex128] {
        arrays.push(complex(
            &[2, 4],
            &[
                (1.00000003, 2.0),
                (f64::NAN, 7.0),
                (-3.0, 2.5),
                (-0.0, -0.0),
                (2.0, f64::NAN),
                (0.5, -1.5),
                (f64::NAN, f64::NAN),
                (4.0, 1.0),
            ],
            dtype,
        ));
    }
    for array in arrays {
        let input = batch.add_input(&array);
        for dtype in [
            None,
            Some(DType::Bool),
            Some(DType::I8),
            Some(DType::I64),
            Some(DType::U64),
            Some(DType::F16),
            Some(DType::F32),
            Some(DType::F64),
            Some(DType::Complex64),
            Some(DType::Complex128),
        ] {
            for (&mask_id, mask) in mask_ids.iter().zip(&masks) {
                for axes in [
                    None,
                    Some(&[][..]),
                    Some(&[0][..]),
                    Some(&[1][..]),
                    Some(&[-1, -2][..]),
                ] {
                    for keepdims in [false, true] {
                        for (initial, initial_id) in std::iter::once((None, None)).chain(
                            initials
                                .iter()
                                .zip(&initial_ids)
                                .map(|(initial, &id)| (Some(initial), Some(id))),
                        ) {
                            for operation in ["nansum", "nanprod"] {
                                batch.push_arithmetic(
                                    input,
                                    &array,
                                    operation,
                                    ReductionOptions {
                                        axes,
                                        keepdims,
                                        dtype,
                                        where_mask: Some(mask),
                                        initial,
                                    },
                                    (Some(mask_id), initial_id),
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
