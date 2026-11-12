//! Logical complex unique values, axis slices, and reconstruction metadata.
//! The differential oracle targets NumPy 2.4 or later.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{
    UFuncArray, UFuncError, UniqueInfo, unique_all, unique_counts, unique_inverse, unique_values,
};
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Stdio};

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::I64 => "int64",
        DType::U64 => "uint64",
        DType::F64 => "float64",
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
        _ => panic!("unsupported unique test dtype"),
    }
}

fn complex_array(shape: &[usize], values: &[(f64, f64)], dtype: DType) -> UFuncArray {
    let mut physical_shape = shape.to_vec();
    physical_shape.push(2);
    UFuncArray::new(
        physical_shape,
        values.iter().flat_map(|&(re, im)| [re, im]).collect(),
        dtype,
    )
    .unwrap()
}

fn input_spec(array: &UFuncArray) -> Value {
    let mut shape = array.shape().to_vec();
    let values = if array.dtype().is_complex() {
        assert_eq!(shape.pop(), Some(2));
        json!(
            array
                .values()
                .chunks_exact(2)
                .map(|parts| [parts[0].to_string(), parts[1].to_string()])
                .collect::<Vec<_>>()
        )
    } else {
        let values: Vec<String> = match array.to_storage().unwrap() {
            ArrayStorage::I64(values) => values.iter().map(ToString::to_string).collect(),
            ArrayStorage::U64(values) => values.iter().map(ToString::to_string).collect(),
            _ => array.values().iter().map(ToString::to_string).collect(),
        };
        json!(values)
    };
    json!({"shape": shape, "dtype": dtype_name(array.dtype()), "values": values})
}

fn output_arrays(info: UniqueInfo) -> Vec<UFuncArray> {
    let (values, indices, inverse, counts) = info;
    let mut output = vec![values];
    output.extend(indices);
    output.extend(inverse);
    output.extend(counts);
    output
}

#[derive(Default)]
struct OracleBatch {
    cases: Vec<Value>,
    actual: Vec<Result<Vec<UFuncArray>, UFuncError>>,
}

impl OracleBatch {
    fn options(
        &mut self,
        array: &UFuncArray,
        axis: Option<isize>,
        flags: u8,
        equal_nan: bool,
    ) {
        self.cases.push(json!({
            "op": "unique", "array": input_spec(array), "axis": axis,
            "flags": flags, "equal_nan": equal_nan,
        }));
        self.actual.push(
            array
                .unique_with_options(
                    axis,
                    flags & 1 != 0,
                    flags & 2 != 0,
                    flags & 4 != 0,
                    equal_nan,
                )
                .map(output_arrays),
        );
    }

    fn legacy(&mut self, array: &UFuncArray, axis: Option<isize>, flags: u8) {
        self.cases.push(json!({
            "op": "unique", "array": input_spec(array), "axis": axis,
            "flags": flags, "equal_nan": true,
        }));
        self.actual.push(match axis {
            Some(axis) => array
                .unique_axis_with_info(axis, flags & 1 != 0, flags & 2 != 0, flags & 4 != 0)
                .map(output_arrays),
            None => Ok(output_arrays(array.unique_with_info(
                flags & 1 != 0,
                flags & 2 != 0,
                flags & 4 != 0,
            ))),
        });
    }

    fn variants(&mut self, array: &UFuncArray) {
        let values = unique_values(array);
        let (count_values, counts) = unique_counts(array);
        let (inverse_values, inverse) = unique_inverse(array);
        let (all_values, indices, all_inverse, all_counts) = unique_all(array);
        for (op, actual) in [
            ("unique_values", vec![values]),
            ("unique_counts", vec![count_values, counts]),
            ("unique_inverse", vec![inverse_values, inverse]),
            ("unique_all", vec![all_values, indices, all_inverse, all_counts]),
        ] {
            self.cases.push(json!({"op": op, "array": input_spec(array)}));
            self.actual.push(Ok(actual));
        }
    }

    fn check(self) {
        const SCRIPT: &str = r#"
import json, sys, warnings
import numpy as np

def array(spec):
    dtype = np.dtype(spec['dtype'])
    if dtype.kind == 'c':
        values = [complex(float(re), float(im)) for re, im in spec['values']]
    elif dtype.kind in 'iu':
        values = [int(value) for value in spec['values']]
    else:
        values = [float(value) for value in spec['values']]
    return np.array(values, dtype=dtype).reshape(spec['shape'])

def encode(result, normalize_zero=False):
    result = np.asarray(result)
    if result.dtype.kind == 'c':
        parts = np.column_stack((result.real.ravel(), result.imag.ravel()))
        components = np.asarray(parts, dtype=np.float64).ravel()
    else:
        components = result.astype(np.float64).ravel()
    if normalize_zero:
        components[components == 0] = 0.0
    return {
        'shape': list(result.shape), 'dtype': result.dtype.name,
        'bits': [int(value) for value in components.view(np.uint64)],
    }

outputs = []
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for case in json.load(sys.stdin):
        try:
            source = array(case['array'])
            if case['op'] == 'unique':
                flags = case['flags']
                result = np.unique(
                    source, axis=case['axis'], return_index=bool(flags & 1),
                    return_inverse=bool(flags & 2), return_counts=bool(flags & 4),
                    equal_nan=case['equal_nan'],
                )
                arrays = result if isinstance(result, tuple) else (result,)
            elif case['op'] == 'unique_values':
                # NumPy 2.4 may return hash order. The Array API does not
                # prescribe ordering or the signed-zero representative.
                arrays = (np.sort(np.unique_values(source)),)
            else:
                arrays = tuple(getattr(np, case['op'])(source))
            outputs.append({
                'arrays': [encode(value, case['op'] == 'unique_values') for value in arrays],
            })
        except Exception as error:
            outputs.append({'error': type(error).__name__})
json.dump(outputs, sys.stdout)
"#;
        let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(python)
            .args(["-c", SCRIPT])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("NumPy oracle must be available");
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(&serde_json::to_vec(&self.cases).unwrap()).unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.actual.len());
        for (case_index, (expected, actual)) in expected.iter().zip(&self.actual).enumerate() {
            let context = format!("case {case_index}: {}", self.cases[case_index]);
            if expected.get("error").is_some() {
                assert!(actual.is_err(), "{context}: expected error, got {actual:?}");
                continue;
            }
            let actual = actual.as_ref().unwrap_or_else(|error| {
                panic!("{context}: unexpected error {error:?}")
            });
            let expected = expected["arrays"].as_array().unwrap();
            assert_eq!(actual.len(), expected.len(), "{context}: metadata presence");
            let operation = self.cases[case_index]["op"].as_str().unwrap();
            let flags = self.cases[case_index]["flags"].as_u64().unwrap_or(0);
            let has_index = operation == "unique_all" || (operation == "unique" && flags & 1 != 0);
            let distinct_nan_inverse = if operation == "unique_inverse" {
                Some(1)
            } else if operation == "unique"
                && flags & 2 != 0 && !has_index
                && self.cases[case_index]["equal_nan"] == false
            {
                Some(1)
            } else {
                None
            };
            if let Some(inverse_output) = distinct_nan_inverse {
                // Unstable sorting may permute identical NaNs. Each NaN is
                // nevertheless a distinct group with exactly one source entry.
                let components = if actual[0].dtype().is_complex() { 2 } else { 1 };
                let mut frequencies = vec![0_usize; actual[0].values().len() / components];
                for &index in actual[inverse_output].values() {
                    assert!(index.is_finite() && index >= 0.0 && index.fract() == 0.0, "{context}");
                    assert!((index as usize) < frequencies.len(), "{context}: inverse bounds");
                    frequencies[index as usize] += 1;
                }
                for (group, parts) in actual[0].values().chunks_exact(components).enumerate() {
                    if parts.iter().any(|part| part.is_nan()) {
                        assert_eq!(frequencies[group], 1, "{context}: distinct NaN group");
                    }
                }
            }
            for (array_index, (actual_array, expected_array)) in actual.iter().zip(expected).enumerate() {
                let actual_values = &actual[0];
                let expected_values = &expected[0];
                let actual = actual_array;
                let expected = expected_array;
                let context = format!("{context}, output {array_index}");
                let mut shape = actual.shape().to_vec();
                if actual.dtype().is_complex() {
                    assert_eq!(shape.pop(), Some(2), "{context}: component axis");
                }
                assert_eq!(json!(shape), expected["shape"], "{context}: shape");
                assert_eq!(dtype_name(actual.dtype()), expected["dtype"], "{context}: dtype");
                let bits = expected["bits"].as_array().unwrap();
                assert_eq!(actual.values().len(), bits.len(), "{context}: component count");
                for (component, (&actual, bits)) in actual.values().iter().zip(bits).enumerate() {
                    let expected = f64::from_bits(bits.as_u64().unwrap());
                    if distinct_nan_inverse == Some(array_index) && actual != expected {
                        let components = if actual_values.dtype().is_complex() { 2 } else { 1 };
                        let actual_start = actual as usize * components;
                        let expected_start = expected as usize * components;
                        let expected_bits = expected_values["bits"].as_array().unwrap();
                        let actual_parts = &actual_values.values()[actual_start..actual_start + components];
                        let expected_parts: Vec<f64> = expected_bits[expected_start..expected_start + components]
                            .iter().map(|bits| f64::from_bits(bits.as_u64().unwrap())).collect();
                        assert!(actual_parts.iter().any(|part| part.is_nan()), "{context}");
                        for (&actual, &expected) in actual_parts.iter().zip(&expected_parts) {
                            assert!(
                                (actual.is_nan() && expected.is_nan()) || actual == expected,
                                "{context}: an unstable inverse must still identify the same NaN value"
                            );
                        }
                        continue;
                    }
                    // Without return_index, the representative among equal
                    // signed-zero keys is not specified by NumPy's sort.
                    if !has_index && actual == 0.0 && expected == 0.0 {
                        continue;
                    }
                    if expected.is_nan() {
                        assert!(actual.is_nan(), "{context}: component {component}");
                    } else {
                        assert_eq!(
                            actual.to_bits(), expected.to_bits(),
                            "{context}: component {component}, {actual:?} != {expected:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn complex_unique_preserves_pairs_and_logical_inverse_shapes() {
    let values = [(3.0, 4.0), (1.0, 5.0), (1.0, -2.0), (3.0, 4.0), (1.0, 5.0), (2.0, 0.0)];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [vec![6], vec![2, 3], vec![1, 2, 3]] {
            let array = complex_array(&shape, &values, dtype);
            for flags in 0..8 {
                batch.options(&array, None, flags, true);
            }
            batch.legacy(&array, None, 0);
            batch.legacy(&array, None, 7);
            batch.variants(&array);
        }
    }
    batch.check();
}

#[test]
fn complex_unique_nan_policy_and_array_api_variants_match_numpy() {
    let nan = f64::NAN;
    let values = [
        (nan, 2.0), (1.0, nan), (nan, 1.0), (1.0, nan),
        (0.0, nan), (nan, nan), (nan, 2.0), (1.0, 2.0),
        (1.0, 2.0), (0.0, -0.0), (-0.0, 0.0), (f64::INFINITY, f64::NEG_INFINITY),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [vec![12], vec![3, 4]] {
            let array = complex_array(&shape, &values, dtype);
            for equal_nan in [false, true] {
                for flags in 0..8 {
                    batch.options(&array, None, flags, equal_nan);
                }
                for axis in [0, -1] {
                    batch.options(&array, Some(axis), 7, equal_nan);
                }
            }
            batch.legacy(&array, None, 7);
            batch.legacy(&array, Some(0), 7);
            batch.variants(&array);
        }
    }
    batch.check();
}

#[test]
fn complex_unique_keeps_the_minimum_nan_class_and_first_tied_representative() {
    let first_nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let second_nan = f64::from_bits(0xfff8_0000_0000_0002);
    let array = complex_array(
        &[7],
        &[
            (first_nan, 0.0), (2.0, first_nan), (-1.0, second_nan),
            (-1.0, first_nan), (0.0, -0.0), (-0.0, 0.0), (first_nan, first_nan),
        ],
        DType::Complex128,
    );
    let (values, indices, inverse, counts) = array.unique_with_info(true, true, true);
    assert_eq!(values.shape(), &[2, 2]);
    assert_eq!(values.values()[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(values.values()[1].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(values.values()[2], -1.0);
    assert_eq!(values.values()[3].to_bits(), second_nan.to_bits());
    assert_eq!(indices.unwrap().values(), &[4.0, 2.0]);
    assert_eq!(inverse.unwrap().values(), &[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    assert_eq!(counts.unwrap().values(), &[2.0, 5.0]);

    // NumPy 2.4's metadata-free hash route keeps the first source NaN.
    let values = array.unique();
    assert_eq!(values.shape(), &[2, 2]);
    assert_eq!(values.values()[2].to_bits(), first_nan.to_bits());
    assert_eq!(values.values()[3].to_bits(), 0.0_f64.to_bits());
}

#[test]
fn complex_unique_axis_groups_whole_slices_and_preserves_nan_rows() {
    let nan = f64::NAN;
    let values = [
        (2.0, 1.0), (nan, 1.0), (0.0, -0.0),
        (1.0, 1.0), (5.0, 2.0), (0.0, 0.0),
        (1.0, 1.0), (5.0, 2.0), (-0.0, -0.0),
        (2.0, 1.0), (nan, 1.0), (0.0, -0.0),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [vec![4, 3], vec![2, 3, 2]] {
            let array = complex_array(&shape, &values, dtype);
            let ndim = shape.len() as isize;
            for axis in -ndim..ndim {
                for equal_nan in [false, true] {
                    batch.options(&array, Some(axis), 0, equal_nan);
                    batch.options(&array, Some(axis), 7, equal_nan);
                }
                batch.legacy(&array, Some(axis), 7);
            }
            batch.options(&array, Some(ndim), 7, true);
            batch.options(&array, Some(-ndim - 1), 7, true);
        }
    }
    batch.check();
}

#[test]
fn complex_unique_rounds_source_components_before_grouping() {
    let values = [
        (1.0 + 2.0_f64.powi(-25), 2.0 + 2.0_f64.powi(-24)),
        (1.0, 2.0),
        (16_777_217.0, 0.0), (16_777_216.0, 0.0),
        (1e-100, -1e-100), (0.0, -0.0),
        (1e100, 1.0), (f64::INFINITY, 1.0),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [vec![8], vec![2, 4]] {
            let array = complex_array(&shape, &values, dtype);
            batch.options(&array, None, 7, true);
            batch.options(&array, Some(0), 7, true);
            batch.options(&array, Some(-1), 7, false);
            batch.variants(&array);
        }
    }
    batch.check();
}

#[test]
fn complex_unique_scalar_empty_arrays_and_empty_slices_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [vec![], vec![0], vec![0, 3], vec![3, 0], vec![2, 0, 3], vec![2, 3, 0]] {
            let values = if shape.is_empty() { vec![(-0.0, 2.0)] } else { vec![] };
            let array = complex_array(&shape, &values, dtype);
            for flags in 0..8 {
                batch.options(&array, None, flags, true);
            }
            batch.legacy(&array, None, 7);
            batch.variants(&array);
            let ndim = shape.len() as isize;
            for axis in -ndim - 1..=ndim {
                batch.options(&array, Some(axis), 7, true);
                batch.legacy(&array, Some(axis), 7);
            }
        }
    }
    batch.check();
}

#[test]
fn complex_unique_checked_errors_preserve_legacy_input_without_losing_components() {
    for dtype in [DType::Complex64, DType::Complex128] {
        for (shape, values) in [
            (vec![], vec![1.0]),
            (vec![3], vec![3.0, -0.0, 1.0]),
            (vec![1, 4], vec![3.0, 4.0, 1.0, 2.0]),
            (vec![0], vec![]),
        ] {
            let array = UFuncArray::new(shape, values, dtype).unwrap();
            assert!(array.unique_with_options(None, true, true, true, true).is_err());
            assert!(array.unique_axis_with_info(0, true, true, true).is_err());
            let (preserved, indices, inverse, counts) = array.unique_with_info(true, true, true);
            assert_eq!(preserved, array);
            assert!(indices.is_none() && inverse.is_none() && counts.is_none());
            assert_eq!(array.unique(), array);
            assert_eq!(unique_values(&array), array);
            let (preserved, counts) = unique_counts(&array);
            assert_eq!(preserved, array);
            assert!(counts.values().is_empty());
            let (preserved, inverse) = unique_inverse(&array);
            assert_eq!(preserved, array);
            assert!(inverse.values().is_empty());
            let (preserved, indices, inverse, counts) = unique_all(&array);
            assert_eq!(preserved, array);
            assert!(indices.values().is_empty());
            assert!(inverse.values().is_empty());
            assert!(counts.values().is_empty());
        }

        let huge = complex_array(&[0, usize::MAX, usize::MAX], &[], dtype);
        let (values, indices, inverse, counts) =
            huge.unique_with_options(Some(1), true, false, false, true).unwrap();
        assert_eq!(values.shape(), &[0, 1, usize::MAX, 2]);
        assert!(values.values().is_empty());
        assert_eq!(indices.unwrap().values(), &[0.0]);
        assert!(inverse.is_none() && counts.is_none());
        assert!(huge.unique_with_options(Some(1), false, true, false, true).is_err());
        if (usize::MAX as u128) > i64::MAX as u128 {
            assert!(huge.unique_with_options(Some(1), false, false, true, true).is_err());
        }
        let flat = huge.unique_with_options(None, true, true, true, true).unwrap();
        assert_eq!(flat.0.shape(), &[0, 2]);
        assert_eq!(flat.2.unwrap().shape(), &[0, usize::MAX, usize::MAX]);
    }

    #[cfg(target_pointer_width = "64")]
    {
        let axis_len = (1_usize << 53) + 1;
        let array = complex_array(&[0, axis_len], &[], DType::Complex128);
        let (_, _, _, counts) =
            array.unique_with_options(Some(1), false, false, true, true).unwrap();
        assert_eq!(
            counts.unwrap().to_storage().unwrap(),
            ArrayStorage::I64(vec![axis_len as i64]),
        );
    }
}

#[test]
fn checked_unique_keeps_real_nan_options_and_exact_integer_sidecars() {
    let mut batch = OracleBatch::default();
    let real = UFuncArray::new(
        vec![2, 3],
        vec![f64::NAN, -0.0, 2.0, f64::NAN, 0.0, 2.0],
        DType::F64,
    ).unwrap();
    for equal_nan in [false, true] {
        for flags in 0..8 {
            batch.options(&real, None, flags, equal_nan);
        }
    }
    for storage in [
        ArrayStorage::I64(vec![-9_007_199_254_740_993, -9_007_199_254_740_992,
                              -9_007_199_254_740_993, 1, 0, 1]),
        ArrayStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_992,
                              9_007_199_254_740_993, 1, 0, 1]),
    ] {
        let array = UFuncArray::from_storage(vec![2, 3], storage).unwrap();
        for equal_nan in [false, true] {
            batch.options(&array, None, 7, equal_nan);
            batch.options(&array, Some(0), 7, equal_nan);
        }
        let (unique, _, _, _) = array.unique_with_options(None, true, true, true, true).unwrap();
        assert!(unique.has_integer_sidecar());
        match unique.to_storage().unwrap() {
            ArrayStorage::I64(values) => {
                assert!(values.contains(&-9_007_199_254_740_993));
                assert!(values.contains(&-9_007_199_254_740_992));
            }
            ArrayStorage::U64(values) => {
                assert!(values.contains(&9_007_199_254_740_993));
                assert!(values.contains(&9_007_199_254_740_992));
            }
            _ => panic!("wide integer storage was lost"),
        }
    }
    batch.check();
}

#[test]
fn complex_unique_generated_axis_and_metadata_cases_match_numpy() {
    let mut batch = OracleBatch::default();
    let mut state = 0x9e37_79b9_u64;
    for case in 0..24 {
        let shape = match case % 3 {
            0 => vec![12],
            1 => vec![3, 4],
            _ => vec![2, 3, 2],
        };
        let dtype = if case % 2 == 0 { DType::Complex64 } else { DType::Complex128 };
        let values: Vec<(f64, f64)> = (0..12).map(|index| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let re = ((state >> 32) % 5) as f64 - 2.0;
            let im = ((state >> 40) % 5) as f64 - 2.0;
            if case % 4 == 0 && index % 7 == 0 { (re, f64::NAN) } else { (re, im) }
        }).collect();
        let array = complex_array(&shape, &values, dtype);
        batch.options(&array, None, 7, case % 2 == 0);
        for axis in 0..shape.len() {
            batch.options(&array, Some(axis as isize), 7, true);
        }
    }
    batch.check();
}
