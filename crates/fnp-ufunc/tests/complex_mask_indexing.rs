//! Boolean indexing and extraction operate on complete logical elements.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
use std::process::{Command, Stdio};

fn complex(shape: &[usize], parts: &[(f64, f64)], dtype: DType) -> UFuncArray {
    let mut shape = shape.to_vec();
    shape.push(2);
    UFuncArray::new(
        shape,
        parts.iter().flat_map(|&(re, im)| [re, im]).collect(),
        dtype,
    )
    .unwrap()
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::I8 => "int8",
        DType::I64 => "int64",
        DType::U64 => "uint64",
        DType::F16 => "float16",
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
        _ => panic!("unsupported test dtype"),
    }
}

fn sample(shape: &[usize], dtype: DType) -> UFuncArray {
    let count = if shape.contains(&0) {
        0
    } else {
        shape.iter().product()
    };
    let pattern = [
        (-0.0, 0.0),
        (0.0, -2.5),
        (1.0 + 2.0_f64.powi(-24), -1.0 - 2.0_f64.powi(-24)),
        (1.0e-50, -1.0e-50),
        (f64::NAN, 4.0),
        (5.0, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY),
    ];
    if dtype.is_complex() {
        let parts = (0..count)
            .map(|index| pattern[index % pattern.len()])
            .collect::<Vec<_>>();
        return complex(shape, &parts, dtype);
    }
    let storage = match dtype {
        DType::I64 => ArrayStorage::I64(
            (0..count)
                .map(|index| [i64::MIN, (1_i64 << 53) + 1, -1, i64::MAX][index % 4])
                .collect(),
        ),
        DType::U64 => ArrayStorage::U64(
            (0..count)
                .map(|index| [0, (1_u64 << 53) + 1, u64::MAX, 7][index % 4])
                .collect(),
        ),
        _ => ArrayStorage::F64(
            (0..count)
                .map(|index| pattern[index % pattern.len()].0)
                .collect(),
        )
        .cast_to(dtype)
        .unwrap(),
    };
    UFuncArray::from_storage(shape.to_vec(), storage).unwrap()
}

fn input(array: &UFuncArray) -> Value {
    let mut shape = array.shape().to_vec();
    let values = if array.dtype().is_complex() {
        assert_eq!(shape.pop(), Some(2));
        json!(
            array
                .values()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| [pair[0].to_string(), pair[1].to_string()])
                .collect::<Vec<_>>()
        )
    } else if array.dtype().is_integer() {
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
    json!({"shape":shape, "dtype":dtype_name(array.dtype()), "values":values})
}

#[derive(Default)]
struct OracleBatch {
    cases: Vec<Value>,
    actual: Vec<Result<UFuncArray, UFuncError>>,
}

impl OracleBatch {
    fn push(&mut self, op: &str, array: &UFuncArray, mask: &UFuncArray, axis: Option<isize>) {
        self.cases
            .push(json!({"op":op,"array":input(array),"mask":input(mask),"axis":axis}));
        self.actual.push(match op {
            "extract" => UFuncArray::extract(mask, array),
            "boolean_index" => array.boolean_index(mask),
            "compress" => array.compress(
                &mask
                    .values()
                    .iter()
                    .map(|&value| value != 0.0)
                    .collect::<Vec<_>>(),
                axis,
            ),
            _ => unreachable!(),
        });
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np
warnings.simplefilter('ignore')
def load(item):
    dtype = np.dtype(item['dtype'])
    if dtype.kind == 'c':
        a = np.empty(len(item['values']), dtype=dtype)
        a.real = [float(pair[0]) for pair in item['values']]
        a.imag = [float(pair[1]) for pair in item['values']]
    else:
        convert = int if dtype.kind in 'iu' else float
        a = np.array([convert(value) for value in item['values']], dtype=dtype)
    return a.reshape(item['shape'])
rows = []
for case in json.load(sys.stdin):
    a, mask = load(case['array']), load(case['mask'])
    try:
        if case['op'] == 'extract':
            result = np.extract(mask, a)
        elif case['op'] == 'compress':
            result = np.compress(mask, a, axis=case['axis'])
        else:
            result = a[mask]
        result = np.asarray(result)
        if result.dtype.kind == 'c':
            values = np.empty(result.size * 2, dtype=np.float64)
            values[0::2] = result.real.ravel()
            values[1::2] = result.imag.ravel()
        else:
            values = result.astype(np.float64).ravel()
        rows.append(dict(shape=list(result.shape), dtype=str(result.dtype),
                         bits=values.view(np.uint64).tolist(),
                         integers=[str(int(v)) for v in result.flat] if result.dtype.kind in 'iu' else None))
    except (IndexError, ValueError, TypeError) as error:
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
            .expect("launch live NumPy mask oracle");
        let mut stdin = child.stdin.take().unwrap();
        serde_json::to_writer(&mut stdin, &self.cases).unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
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
                assert_eq!(shape.pop(), Some(2), "{case}: component dimension");
            }
            assert_eq!(json!(shape), expected["shape"], "{case}: logical shape");
            assert_eq!(
                dtype_name(actual.dtype()),
                expected["dtype"],
                "{case}: dtype"
            );
            if actual.dtype().is_integer() {
                let storage = actual.to_storage().unwrap();
                let exact = (0..storage.len())
                    .map(|index| storage.get_i128(index).unwrap().to_string())
                    .collect::<Vec<_>>();
                assert_eq!(json!(exact), expected["integers"], "{case}: exact integers");
            }
            let bits = expected["bits"].as_array().unwrap();
            assert_eq!(
                actual.values().len(),
                bits.len(),
                "{case}: stored components"
            );
            for (&actual, bits) in actual.values().iter().zip(bits) {
                let expected = f64::from_bits(bits.as_u64().unwrap());
                if expected.is_nan() {
                    assert!(actual.is_nan(), "{case}: expected NaN");
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{case}: component value"
                    );
                }
            }
        }
    }
}

#[test]
fn masks_select_complete_complex_elements() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let array = complex(
            &[2, 2],
            &[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)],
            dtype,
        );
        let flat = array.compress(&[false, true, true, false], None).unwrap();
        assert_eq!(flat.shape(), &[2, 2]);
        assert_eq!(flat.values(), &[3.0, 4.0, 5.0, 6.0]);
        let column = array.compress(&[false, true], Some(-1)).unwrap();
        assert_eq!(column.shape(), &[2, 1, 2]);
        assert_eq!(column.values(), &[3.0, 4.0, 7.0, 8.0]);
        assert!(array.compress(&[true], Some(2)).is_err());
        let mask = UFuncArray::new(vec![2], vec![0.0, 1.0], DType::Bool).unwrap();
        let row = array.boolean_index(&mask).unwrap();
        assert_eq!(row.shape(), &[1, 2, 2]);
        assert_eq!(row.values(), &[5.0, 6.0, 7.0, 8.0]);
    }
}

#[test]
fn extract_condition_lengths_shapes_and_dtypes_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [
        DType::Complex64,
        DType::Complex128,
        DType::F64,
        DType::I64,
        DType::U64,
    ] {
        for shape in [&[][..], &[2, 3], &[0, 3]] {
            let array = sample(shape, dtype);
            for condition_dtype in [
                DType::Bool,
                DType::F16,
                DType::F32,
                DType::F64,
                DType::Complex64,
                DType::Complex128,
            ] {
                for condition_shape in [&[][..], &[0], &[1], &[4], &[2, 3], &[8]] {
                    let condition = sample(condition_shape, condition_dtype);
                    batch.push("extract", &array, &condition, None);
                }
            }
            for values in [
                vec![0.0; 9],
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ] {
                let mask = UFuncArray::new(vec![values.len()], values, DType::Bool).unwrap();
                batch.push("extract", &array, &mask, None);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn extract_uses_declared_condition_precision() {
    let array = sample(&[4], DType::Complex128);
    let mut batch = OracleBatch::default();
    for dtype in [DType::F16, DType::F32, DType::Complex64, DType::Complex128] {
        let condition = if dtype.is_complex() {
            complex(
                &[4],
                &[(0.0, 1.0e-50), (1.0e-50, 0.0), (0.0, f64::NAN), (-0.0, 0.0)],
                dtype,
            )
        } else {
            UFuncArray::new(vec![4], vec![1.0e-50, -1.0e-50, f64::NAN, -0.0], dtype).unwrap()
        };
        batch.push("extract", &array, &condition, None);
    }
    batch.assert_matches_numpy();
}

#[test]
fn boolean_index_prefix_scalar_empty_and_mismatched_masks_match_numpy() {
    let mut batch = OracleBatch::default();
    let shapes: &[&[usize]] = &[&[], &[3], &[2, 3], &[2, 3, 4], &[0, 3], &[2, 0, 3]];
    let mask_shapes: &[&[usize]] = &[
        &[],
        &[0],
        &[1],
        &[2],
        &[3],
        &[2, 3],
        &[3, 2],
        &[2, 0],
        &[0, 3],
        &[0, 2],
        &[3, 0],
        &[2, 3, 0],
        &[2, 3, 4],
    ];
    for dtype in [
        DType::Complex64,
        DType::Complex128,
        DType::F64,
        DType::I64,
        DType::U64,
    ] {
        for &shape in shapes {
            let array = sample(shape, dtype);
            for &mask_shape in mask_shapes {
                let count = if mask_shape.contains(&0) {
                    0
                } else {
                    mask_shape.iter().product()
                };
                for pattern in 0..3 {
                    let values = (0..count)
                        .map(|i| {
                            if pattern == 2 || (pattern == 1 && i % 2 == 0) {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let mask = UFuncArray::new(mask_shape.to_vec(), values, DType::Bool).unwrap();
                    batch.push("boolean_index", &array, &mask, None);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn compress_logical_axes_and_mask_lengths_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128, DType::I64, DType::U64] {
        for shape in [&[3][..], &[2, 3], &[0, 3], &[2, 0, 3]] {
            let array = sample(shape, dtype);
            for axis in [
                None,
                Some(-4),
                Some(-2),
                Some(-1),
                Some(0),
                Some(1),
                Some(2),
                Some(3),
            ] {
                for values in [
                    vec![],
                    vec![1.0],
                    vec![0.0, 1.0],
                    vec![1.0, 0.0, 1.0],
                    vec![0.0; 8],
                    vec![1.0; 8],
                ] {
                    let mask = UFuncArray::new(vec![values.len()], values, DType::Bool).unwrap();
                    batch.push("compress", &array, &mask, axis);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn boolean_index_rejects_nonboolean_masks_and_malformed_complex_layouts() {
    let array = sample(&[3], DType::Complex128);
    for dtype in [
        DType::F64,
        DType::I8,
        DType::I64,
        DType::Complex64,
        DType::Complex128,
    ] {
        let mask = sample(&[3], dtype);
        assert!(array.boolean_index(&mask).is_err());
    }
    let empty_mask = UFuncArray::new(vec![0], vec![], DType::Bool).unwrap();
    for shape in [vec![], vec![3], vec![2, 1], vec![0, 3]] {
        let count = shape.iter().product();
        let malformed = UFuncArray::new(shape, vec![0.0; count], DType::Complex128).unwrap();
        assert!(malformed.boolean_index(&empty_mask).is_err());
        assert!(malformed.compress(&[], None).is_err());
        assert!(UFuncArray::extract(&empty_mask, &malformed).is_err());
        assert!(UFuncArray::extract(&malformed, &array).is_err());
    }
}

#[test]
fn empty_prefix_selection_checks_shape_without_materializing_elements() {
    let array = complex(&[0, usize::MAX, usize::MAX], &[], DType::Complex128);
    let mask = UFuncArray::new(vec![0], vec![], DType::Bool).unwrap();
    let result = array.boolean_index(&mask).unwrap();
    assert_eq!(result.shape(), &[0, usize::MAX, usize::MAX, 2]);
    assert!(result.values().is_empty());
}
