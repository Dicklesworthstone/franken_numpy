//! Complex sorting, selection, and search use logical elements and NumPy ordering.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
use std::io::Write;
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
        _ => panic!("unsupported ordering test dtype"),
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

#[derive(Default)]
struct OracleBatch {
    cases: Vec<Value>,
    actual: Vec<Result<UFuncArray, UFuncError>>,
}

impl OracleBatch {
    fn order(&mut self, array: &UFuncArray, axis: Option<isize>, kind: Option<&str>) {
        for operation in ["sort", "argsort"] {
            self.cases.push(json!({
                "op": operation, "array": input_spec(array), "axis": axis, "kind": kind,
            }));
            self.actual.push(if operation == "sort" {
                array.sort(axis, kind)
            } else {
                array.argsort(axis, kind)
            });
        }
    }

    fn search(
        &mut self,
        array: &UFuncArray,
        needles: &UFuncArray,
        side: Option<&str>,
        sorter: Option<&[usize]>,
    ) {
        self.cases.push(json!({
            "op": "searchsorted", "array": input_spec(array),
            "needles": input_spec(needles), "side": side, "sorter": sorter,
        }));
        self.actual.push(array.searchsorted(needles, side, sorter));
    }

    fn partition(&mut self, array: &UFuncArray, kth: usize, axis: Option<isize>) {
        for operation in ["partition", "argpartition"] {
            let actual = if operation == "partition" {
                array.partition(kth, axis)
            } else {
                array.argpartition(kth, axis)
            };
            self.cases.push(json!({
                "op": operation, "array": input_spec(array), "axis": axis, "kth": kth,
                "actual": actual.as_ref().ok().map(input_spec),
            }));
            self.actual.push(actual);
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

def checked_partition(case, source, reference):
    # Partition permutations and ordering ties are unspecified. Validate the
    # complete-pair multiset, relative-index permutation, and NumPy order rank
    # instead of requiring the same incidental selection permutation.
    if case['actual'] is None:
        return reference
    actual = array(case['actual'])
    assert actual.shape == reference.shape, 'partition shape'
    assert actual.dtype == reference.dtype, 'partition dtype'
    if source.size == 0:
        return actual
    axis = case['axis']
    if axis is None or source.ndim == 0:
        source = source.ravel()
        axis = -1
    length = source.shape[axis]
    if case['op'] == 'argpartition':
        expected_indices = np.broadcast_to(np.arange(length),
                                           np.moveaxis(actual, axis, -1).shape)
        assert np.array_equal(np.sort(np.moveaxis(actual, axis, -1), axis=-1),
                              expected_indices), 'relative-index permutation'
        candidate = np.take_along_axis(source, actual, axis=axis)
        reference = np.take_along_axis(source, reference, axis=axis)
    else:
        candidate = actual
    lanes = lambda value: np.moveaxis(value, axis, -1).reshape(-1, length)
    def pair_bits(row):
        parts = np.column_stack((row.real, row.imag)).astype(np.float64)
        return sorted(tuple(int(part) for part in pair)
                      for pair in parts.view(np.uint64))
    kth = case['kth']
    for original, selected, expected in zip(lanes(source), lanes(candidate), lanes(reference)):
        assert pair_bits(selected) == pair_bits(original), 'complete-pair permutation'
        ordered = np.sort(original)
        lower = np.searchsorted(ordered, selected, side='left')
        upper = np.searchsorted(ordered, selected, side='right')
        assert lower[kth] <= kth < upper[kth], 'selected order statistic'
        assert lower[kth] == np.searchsorted(ordered, expected[kth]), 'NumPy kth rank'
        assert np.all(lower[:kth] <= lower[kth]), 'lower partition'
        assert np.all(lower[kth + 1:] >= lower[kth]), 'upper partition'
    return actual

outputs = []
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for case in json.load(sys.stdin):
        try:
            source = array(case['array'])
            if case['op'] == 'searchsorted':
                options = {}
                if case['side'] is not None:
                    options['side'] = case['side']
                if case['sorter'] is not None:
                    # Conversion to intp mirrors the integer-index ABI.
                    options['sorter'] = np.array(case['sorter'], dtype=np.uint64).astype(np.intp)
                result = np.asarray(np.searchsorted(source, array(case['needles']), **options))
            elif case['op'] in ('partition', 'argpartition'):
                reference = np.asarray(getattr(np, case['op'])(source, case['kth'], axis=case['axis']))
                result = checked_partition(case, source, reference)
            else:
                options = {'axis': case['axis']}
                if case['kind'] is not None:
                    options['kind'] = case['kind']
                result = np.asarray(getattr(np, case['op'])(source, **options))
            if result.dtype.kind == 'c':
                parts = np.column_stack((result.real.ravel(), result.imag.ravel()))
                components = np.asarray(parts, dtype=np.float64).ravel()
            else:
                components = result.astype(np.float64).ravel()
            outputs.append({
                'shape': list(result.shape), 'dtype': result.dtype.name,
                'bits': [int(value) for value in components.view(np.uint64)],
            })
        except Exception as error:
            outputs.append({'error': type(error).__name__, 'message': str(error)})
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
        stdin
            .write_all(&serde_json::to_vec(&self.cases).unwrap())
            .unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.actual.len());
        for (index, (expected, actual)) in expected.iter().zip(&self.actual).enumerate() {
            let context = format!("case {index}: {}", self.cases[index]);
            if expected.get("error").is_some() {
                assert!(
                    actual.is_err(),
                    "{context}: expected {expected}, got {actual:?}"
                );
                continue;
            }
            let actual = actual.as_ref().unwrap_or_else(|error| {
                panic!("{context}: unexpected error {error:?}")
            });
            let mut shape = actual.shape().to_vec();
            if actual.dtype().is_complex() {
                assert_eq!(shape.pop(), Some(2), "{context}");
            }
            assert_eq!(json!(shape), expected["shape"], "{context}: shape");
            assert_eq!(dtype_name(actual.dtype()), expected["dtype"], "{context}: dtype");
            let bits = expected["bits"].as_array().unwrap();
            assert_eq!(actual.values().len(), bits.len(), "{context}: component count");
            for (component, (&actual, bits)) in actual.values().iter().zip(bits).enumerate() {
                let expected = f64::from_bits(bits.as_u64().unwrap());
                if expected.is_nan() {
                    assert!(actual.is_nan(), "{context}: component {component}");
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{context}: component {component}, {actual:?} != {expected:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn complex_ordering_preserves_pairs_and_logical_indices() {
    let array = complex_array(
        &[3],
        &[(3.0, 4.0), (1.0, 2.0), (3.0, -1.0)],
        DType::Complex128,
    );
    let sorted = array.sort(None, None).unwrap();
    assert_eq!(sorted.shape(), &[3, 2]);
    assert_eq!(sorted.values(), &[1.0, 2.0, 3.0, -1.0, 3.0, 4.0]);
    let indices = array.argsort(Some(-1), Some("stable")).unwrap();
    assert_eq!(indices.shape(), &[3]);
    assert_eq!(indices.values(), &[1.0, 2.0, 0.0]);
    let gathered = array.take_along_axis(&indices, -1).unwrap();
    assert_eq!(gathered.values(), sorted.values());
}

#[test]
fn complex_ordering_axes_kinds_scalars_and_empty_arrays_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [
            vec![],
            vec![1],
            vec![2, 3],
            vec![2, 3, 2],
            vec![0],
            vec![2, 0, 3],
            vec![0, 2, 3],
        ] {
            let count: usize = shape.iter().product();
            let values: Vec<_> = (0..count)
                .map(|index| (((index * 7) % 13) as f64 - 4.0, index as f64 - 5.0))
                .collect();
            let array = complex_array(&shape, &values, dtype);
            let mut axes = vec![None, Some(-1), Some(0)];
            axes.extend((0..shape.len()).map(|axis| Some(axis as isize)));
            axes.push(Some(shape.len() as isize));
            axes.push(Some(-(shape.len() as isize) - 1));
            for axis in axes {
                for kind in [
                    None,
                    Some("quicksort"),
                    Some("stable"),
                    Some("mergesort"),
                    Some("heapsort"),
                ] {
                    batch.order(&array, axis, kind);
                }
            }
        }
    }
    batch.check();
}

#[test]
fn complex_stable_ordering_preserves_nan_categories_zero_signs_and_source_precision() {
    let mut batch = OracleBatch::default();
    let values = [
        (f64::NAN, f64::NAN),
        (2.0, f64::NAN),
        (f64::NAN, -2.0),
        (f64::INFINITY, 4.0),
        (0.0, -0.0),
        (-0.0, 0.0),
        (f64::NEG_INFINITY, f64::NAN),
        (f64::NAN, f64::NEG_INFINITY),
        (f64::NEG_INFINITY, 7.0),
        (1.0, 2.0),
        (1.0 + 2.0_f64.powi(-24), 1.0),
        (1.0, 1.0),
        (f64::NAN, 2.0),
        (f64::INFINITY, f64::NAN),
        (1e-50, -1e-50),
        (-0.0, -0.0),
        (2.0, f64::NAN),
        (f64::NAN, -2.0),
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        let array = complex_array(&[3, 6], &values, dtype);
        for axis in [None, Some(0), Some(1), Some(-1)] {
            for kind in [Some("stable"), Some("mergesort")] {
                batch.order(&array, axis, kind);
            }
        }
    }
    batch.check();
}

#[test]
fn complex_searchsorted_shapes_nan_order_and_sorter_match_numpy() {
    let mut batch = OracleBatch::default();
    let sorted_pairs = [
        (f64::NEG_INFINITY, 0.0),
        (-2.0, 3.0),
        (0.0, -0.0),
        (0.0, 0.0),
        (1.0, -4.0),
        (1.0, 2.0),
        (f64::INFINITY, 7.0),
        (f64::NEG_INFINITY, f64::NAN),
        (2.0, f64::NAN),
        (f64::INFINITY, f64::NAN),
        (f64::NAN, f64::NEG_INFINITY),
        (f64::NAN, 2.0),
        (f64::NAN, f64::INFINITY),
        (f64::NAN, f64::NAN),
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        let sorted = complex_array(&[sorted_pairs.len()], &sorted_pairs, dtype);
        let reversed: Vec<_> = sorted_pairs.iter().copied().rev().collect();
        let unsorted = complex_array(&[reversed.len()], &reversed, dtype);
        let sorter: Vec<_> = (0..reversed.len()).rev().collect();
        for needle_dtype in [DType::Complex64, DType::Complex128] {
            for (shape, pairs) in [
                (vec![], vec![(1.0, 0.0)]),
                (
                    vec![2, 3],
                    vec![
                        (0.0, 0.0),
                        (1.0, 0.0),
                        (1.0, 2.0),
                        (f64::NAN, 1.0),
                        (1.0, f64::NAN),
                        (f64::NAN, f64::NAN),
                    ],
                ),
                (vec![2, 0], vec![]),
                (vec![sorted_pairs.len()], sorted_pairs.to_vec()),
            ] {
                let needles = complex_array(&shape, &pairs, needle_dtype);
                for side in [None, Some("left"), Some("right")] {
                    batch.search(&sorted, &needles, side, None);
                    batch.search(&unsorted, &needles, side, Some(&sorter));
                    // The no-sorter call also preserves NumPy's carried bounds
                    // when the user supplies an unsorted input.
                    batch.search(&unsorted, &needles, side, None);
                }
            }
        }
        let empty = complex_array(&[0], &[], dtype);
        batch.search(&empty, &sorted, None, None);
    }
    batch.check();
}

#[test]
fn complex_searchsorted_mixed_types_preserve_promotion_and_exact_integer_sources() {
    let mut batch = OracleBatch::default();
    let complex = complex_array(
        &[4],
        &[(0.0, 0.0), (1.0, -1.0), (1.0, 1.0), (2.0, 0.0)],
        DType::Complex64,
    );
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::F16,
        DType::F32,
        DType::F64,
    ] {
        let real = UFuncArray::new(vec![4], vec![0.0, 1.0, 1.0, 2.0], dtype).unwrap();
        for side in [None, Some("right")] {
            batch.search(&complex, &real, side, None);
            batch.search(&real, &complex, side, None);
        }
    }
    let signed = UFuncArray::from_storage(
        vec![4],
        ArrayStorage::I64(vec![i64::MIN, -1, 9_007_199_254_740_993, i64::MAX]),
    )
    .unwrap();
    let unsigned = UFuncArray::from_storage(
        vec![4],
        ArrayStorage::U64(vec![0, 9_007_199_254_740_993, 9_007_199_254_740_994, u64::MAX]),
    )
    .unwrap();
    for real in [&signed, &unsigned] {
        let complex = complex_array(
            &[4],
            &[
                (-9_223_372_036_854_775_808.0, 0.0),
                (0.0, 0.0),
                (9_007_199_254_740_994.0, 0.0),
                (18_446_744_073_709_551_616.0, 0.0),
            ],
            DType::Complex128,
        );
        for side in [None, Some("right")] {
            batch.search(real, &complex, side, None);
            batch.search(&complex, real, side, None);
        }
    }
    let rounded = complex_array(&[1], &[(1.0 + 2.0_f64.powi(-24), 0.0)], DType::Complex64);
    let needles = complex_array(
        &[2],
        &[(1.0, 0.0), (1.0 + 2.0_f64.powi(-24), 0.0)],
        DType::Complex128,
    );
    batch.search(&rounded, &needles, None, None);
    batch.search(&rounded, &needles, Some("right"), None);
    batch.check();
}

#[test]
fn complex_ordering_validation_and_visited_sorter_errors_match_numpy() {
    let mut batch = OracleBatch::default();
    let array = complex_array(
        &[3],
        &[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        DType::Complex128,
    );
    let scalar = complex_array(&[], &[(2.0, 0.0)], DType::Complex128);
    let empty = complex_array(&[0], &[], DType::Complex128);
    batch.order(&array, None, Some("invalid"));
    batch.search(&array, &scalar, Some("invalid"), None);
    batch.search(&scalar, &array, None, None);
    batch.search(&array, &scalar, None, Some(&[0, 1]));
    batch.search(&array, &scalar, None, Some(&[0, usize::MAX, 2]));
    batch.search(&array, &empty, None, Some(&[0, usize::MAX, 2]));
    let large = complex_array(&[], &[(100.0, 0.0)], DType::Complex128);
    batch.search(&array, &large, None, Some(&[usize::MAX, 1, 2]));
    batch.check();

    let malformed = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::Complex128).unwrap();
    assert!(malformed.sort(None, None).is_err());
    assert!(malformed.argsort(None, None).is_err());
    assert!(malformed.searchsorted(&scalar, None, None).is_err());
    assert!(array.searchsorted(&malformed, None, None).is_err());
}

#[test]
fn complex_empty_ordering_does_not_overflow_unvisited_stride_products() {
    let array = complex_array(&[usize::MAX, 0, usize::MAX], &[], DType::Complex128);
    for axis in [0, 1, 2, -1] {
        let sorted = array.sort(Some(axis), Some("stable")).unwrap();
        assert_eq!(sorted.shape(), array.shape());
        assert!(sorted.values().is_empty());
        let indices = array.argsort(Some(axis), Some("stable")).unwrap();
        assert_eq!(indices.shape(), &[usize::MAX, 0, usize::MAX]);
        assert!(indices.values().is_empty());
    }
    assert_eq!(array.sort(None, None).unwrap().shape(), &[0, 2]);
    assert_eq!(array.argsort(None, None).unwrap().shape(), &[0]);
}

#[test]
fn complex_partition_preserves_pairs_and_logical_indices() {
    let array = complex_array(
        &[3],
        &[(3.0, 4.0), (1.0, 2.0), (3.0, -1.0)],
        DType::Complex128,
    );
    let selected = array.partition(1, None).unwrap();
    assert_eq!(selected.shape(), &[3, 2]);
    assert_eq!(selected.dtype(), DType::Complex128);
    assert_eq!(selected.values(), &[1.0, 2.0, 3.0, -1.0, 3.0, 4.0]);
    let indices = array.argpartition(1, Some(-1)).unwrap();
    assert_eq!(indices.shape(), &[3]);
    assert_eq!(indices.dtype(), DType::I64);
    assert_eq!(indices.values(), &[1.0, 2.0, 0.0]);
    assert_eq!(
        array.take_along_axis(&indices, -1).unwrap().values(),
        selected.values()
    );
    assert!(array.partition(3, None).is_err());
    assert!(array.argpartition(3, None).is_err());
    assert!(array.partition(0, Some(1)).is_err());
    assert!(array.argpartition(0, Some(1)).is_err());
}

#[test]
fn complex_partition_axes_kth_scalars_and_empty_arrays_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [
            vec![],
            vec![1],
            vec![5],
            vec![2, 3],
            vec![2, 3, 4],
            vec![0],
            vec![2, 0, 3],
            vec![0, 2, 3],
        ] {
            let count: usize = shape.iter().product();
            let pairs: Vec<_> = (0..count)
                .map(|index| (((index * 7) % 13) as f64 - 4.0, 5.0 - index as f64))
                .collect();
            let array = complex_array(&shape, &pairs, dtype);
            let rank = shape.len() as isize;
            let mut axes = vec![None, Some(0), Some(-1), Some(rank), Some(-rank - 1)];
            axes.extend((0..rank).map(Some));
            axes.extend((-rank..0).map(Some));
            axes.sort_unstable();
            axes.dedup();
            let mut positions = vec![
                0,
                1,
                2,
                3,
                count / 2,
                count.saturating_sub(1),
                count,
                count + 1,
            ];
            positions.sort_unstable();
            positions.dedup();
            for axis in axes {
                for &kth in &positions {
                    batch.partition(&array, kth, axis);
                }
            }
        }
    }
    batch.check();
}

#[test]
fn complex_partition_nan_categories_ties_and_source_precision_match_numpy() {
    let mut batch = OracleBatch::default();
    let pairs = [
        (f64::NAN, f64::NAN),
        (2.0, f64::NAN),
        (f64::NAN, -2.0),
        (f64::INFINITY, 4.0),
        (0.0, -0.0),
        (-0.0, 0.0),
        (f64::NEG_INFINITY, f64::NAN),
        (f64::NAN, f64::NEG_INFINITY),
        (f64::NEG_INFINITY, 7.0),
        (1.0, 2.0),
        (1.0 + 2.0_f64.powi(-24), 1.0),
        (1.0, 1.0),
        (f64::NAN, 2.0),
        (f64::INFINITY, f64::NAN),
        (1e-50, -1e-50),
        (-0.0, -0.0),
        (2.0, f64::NAN),
        (f64::NAN, -2.0),
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        let array = complex_array(&[3, 6], &pairs, dtype);
        for (axis, length) in [(None, 18), (Some(0), 3), (Some(1), 6), (Some(-1), 6)] {
            for kth in 0..length {
                batch.partition(&array, kth, axis);
            }
        }
        // Exercise selection beyond tiny lanes, with repeated values and NaNs
        // at both contiguous and strided logical positions.
        let pairs: Vec<_> = (0..195)
            .map(|index| {
                let real = if index % 17 == 0 {
                    f64::NAN
                } else {
                    ((index * 73) % 97) as f64 - 48.0
                };
                let imag = if index % 19 == 0 {
                    f64::NAN
                } else {
                    ((index * 29) % 31) as f64 - 15.0
                };
                (real, imag)
            })
            .collect();
        let array = complex_array(&[3, 65], &pairs, dtype);
        for (axis, length) in [(None, 195), (Some(0), 3), (Some(1), 65), (Some(-1), 65)] {
            for kth in [0, length / 2, length - 1] {
                batch.partition(&array, kth, axis);
            }
        }
    }
    batch.check();
}

#[test]
fn complex_partition_preserves_nan_payloads_and_validates_empty_and_malformed_layouts() {
    let array = complex_array(
        &[2, 4],
        &[
            (f64::from_bits(0x7ff8_0000_0000_0123), -0.0),
            (-0.0, 0.0),
            (2.0, f64::from_bits(0xfff8_0000_0000_0456)),
            (3.0, -2.0),
            (0.0, -0.0),
            (f64::INFINITY, 1.0),
            (f64::from_bits(0x7ff8_0000_0000_0789), f64::NAN),
            (f64::NEG_INFINITY, -1.0),
        ],
        DType::Complex128,
    );
    let pair_bits = |values: &[f64]| {
        let mut pairs: Vec<_> = values
            .chunks_exact(2)
            .map(|parts| (parts[0].to_bits(), parts[1].to_bits()))
            .collect();
        pairs.sort_unstable();
        pairs
    };
    for (axis, length) in [(0, 2), (1, 4)] {
        for kth in 0..length {
            let selected = array.partition(kth, Some(axis)).unwrap();
            let indices = array.argpartition(kth, Some(axis)).unwrap();
            let gathered = array.take_along_axis(&indices, axis).unwrap();
            assert_eq!(pair_bits(selected.values()), pair_bits(array.values()));
            assert_eq!(
                selected
                    .values()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                gathered
                    .values()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

    for (shape, values) in [
        (vec![], vec![1.0]),
        (vec![3], vec![1.0, 2.0, 3.0]),
        (vec![2, 1], vec![1.0, 2.0]),
    ] {
        let malformed = UFuncArray::new(shape, values, DType::Complex128).unwrap();
        for axis in [None, Some(0)] {
            assert!(malformed.partition(0, axis).is_err());
            assert!(malformed.argpartition(0, axis).is_err());
        }
    }
    let empty = complex_array(&[usize::MAX, 0, usize::MAX], &[], DType::Complex128);
    for axis in [None, Some(0), Some(1), Some(2), Some(-1)] {
        for kth in [0, isize::MAX as usize] {
            let selected = empty.partition(kth, axis).unwrap();
            let indices = empty.argpartition(kth, axis).unwrap();
            let expected_shape: &[usize] = if axis.is_none() {
                &[0, 2]
            } else {
                empty.shape()
            };
            let expected_indices: &[usize] = if axis.is_none() {
                &[0]
            } else {
                &[usize::MAX, 0, usize::MAX]
            };
            assert_eq!(selected.shape(), expected_shape);
            assert_eq!(indices.shape(), expected_indices);
            assert!(selected.values().is_empty());
            assert!(indices.values().is_empty());
        }
    }
    assert!(empty.partition(0, Some(3)).is_err());
    assert!(empty.argpartition(0, Some(-4)).is_err());
}
