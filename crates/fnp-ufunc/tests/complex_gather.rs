//! Integer gathers select logical complex elements, including their axes.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ndarray::ShapeError;
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
use std::process::{Command, Stdio};

fn complex(shape: &[usize], dtype: DType) -> UFuncArray {
    let pattern = [
        (0.0, -0.0),
        (-0.0, 0.0),
        (1.0 + 2.0_f64.powi(-24), -1.0 - 2.0_f64.powi(-24)),
        (1.0e-50, -1.0e-50),
        (f64::INFINITY, f64::NEG_INFINITY),
        (f64::NAN, -3.5),
        (2.5, f64::NAN),
        (1.0e300, -1.0e300),
    ];
    let count = if shape.contains(&0) {
        0
    } else {
        shape.iter().product()
    };
    let mut physical_shape = shape.to_vec();
    physical_shape.push(2);
    let values = (0..count)
        .flat_map(|index| {
            let (real, imaginary) = pattern[index % pattern.len()];
            [real, imaginary]
        })
        .collect();
    UFuncArray::new(physical_shape, values, dtype).unwrap()
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
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
        _ => panic!("unsupported oracle dtype"),
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
    fn take(&mut self, array: &UFuncArray, indices: &[i64], axis: Option<isize>) {
        self.cases.push(json!({
            "op": "take", "array": oracle_input(array), "indices": indices, "axis": axis,
        }));
        self.actual.push(array.take(indices, axis));
    }

    fn along(&mut self, array: &UFuncArray, indices: &UFuncArray, axis: isize) {
        self.cases.push(json!({
            "op": "along", "array": oracle_input(array),
            "indices": oracle_input(indices), "axis": axis,
        }));
        self.actual.push(array.take_along_axis(indices, axis));
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np

warnings.simplefilter('ignore')
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

rows = []
for case in json.load(sys.stdin):
    try:
        array = load(case['array'])
        if case['op'] == 'take':
            result = np.take(array, np.array(case['indices'], dtype=np.int64), axis=case['axis'])
        else:
            result = np.take_along_axis(array, load(case['indices']), axis=case['axis'])
        if result.dtype.kind == 'c':
            parts = np.empty(result.size * 2, dtype=np.float64)
            parts[0::2] = result.real.ravel()
            parts[1::2] = result.imag.ravel()
        else:
            parts = result.astype(np.float64).ravel()
        rows.append(dict(shape=list(result.shape), dtype=str(result.dtype),
                         bits=parts.view(np.uint64).tolist(),
                         integers=[str(int(value)) for value in result.flat]
                             if result.dtype.kind in 'biu' else None))
    except (IndexError, TypeError, ValueError) as error:
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
            .expect("launch live NumPy gather oracle");
        serde_json::to_writer(child.stdin.take().unwrap(), &self.cases).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy gather oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.actual.len());
        for ((case, actual), expected) in self.cases.iter().zip(self.actual).zip(expected) {
            if let Some(error) = expected.get("error") {
                let actual = actual.expect_err(&format!("{case}: expected {expected}"));
                if error == "AxisError" {
                    assert!(
                        matches!(actual, UFuncError::AxisOutOfBounds { .. }),
                        "{case}: expected axis validation first, got {actual}"
                    );
                } else if expected["message"]
                    .as_str()
                    .unwrap()
                    .contains("must be an integer array")
                {
                    assert!(
                        actual.to_string().contains("must be an integer array"),
                        "{case}: expected index dtype validation first, got {actual}"
                    );
                }
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
                let values = (0..storage.len())
                    .map(|index| storage.get_i128(index).unwrap().to_string())
                    .collect::<Vec<_>>();
                assert_eq!(
                    json!(values),
                    expected["integers"],
                    "{case}: exact integers"
                );
            }
            let bits = expected["bits"].as_array().unwrap();
            assert_eq!(actual.values().len(), bits.len(), "{case}: component count");
            for (index, (&actual, expected)) in actual.values().iter().zip(bits).enumerate() {
                let expected = f64::from_bits(expected.as_u64().unwrap());
                if expected.is_nan() {
                    assert!(actual.is_nan(), "{case}: component {index} expected NaN");
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{case}: component {index}, got {actual}, expected {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn take_complex_flattening_and_logical_axes_match_numpy() {
    let shapes: &[&[usize]] = &[&[], &[1], &[8], &[2, 4], &[2, 1, 4], &[2, 3, 2]];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &shape in shapes {
            let array = complex(shape, dtype);
            let rank = shape.len().max(1) as isize;
            for axis in std::iter::once(None).chain((-rank..rank).map(Some)) {
                let length = axis.map_or(array.values().len() / 2, |axis| {
                    if shape.is_empty() {
                        1
                    } else {
                        shape[axis.rem_euclid(rank) as usize]
                    }
                }) as i64;
                batch.take(&array, &[length - 1, 0, -1, -length, 0], axis);
                batch.take(&array, &[], axis);
                for invalid in [length, -length - 1, i64::MIN, i64::MAX] {
                    batch.take(&array, &[invalid], axis);
                }
            }
            batch.take(&array, &[], Some(rank));
            batch.take(&array, &[0], Some(-rank - 1));
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn take_empty_axis_bounds_match_numpy() {
    let shapes: &[&[usize]] = &[
        &[0],
        &[0, 3],
        &[2, 0],
        &[0, 0],
        &[2, 0, 3],
        &[2, 3, 0],
        &[0, 3, 4],
    ];
    let mut batch = OracleBatch::default();
    for dtype in [
        DType::F64,
        DType::I64,
        DType::U64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for &shape in shapes {
            let array = if dtype.is_complex() {
                complex(shape, dtype)
            } else {
                UFuncArray::zeros(shape.to_vec(), dtype).unwrap()
            };
            let rank = shape.len() as isize;
            for axis in std::iter::once(None).chain((-rank - 1..=rank).map(Some)) {
                for indices in [&[][..], &[0][..], &[-1][..], &[99][..]] {
                    batch.take(&array, indices, axis);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn take_along_axis_complex_broadcasts_logical_dimensions_and_source_precision() {
    let layouts: &[(&[usize], &[usize], usize)] = &[
        (&[2, 4], &[2, 4], 1),
        (&[2, 4], &[1, 3], 1),
        (&[1, 4], &[3, 2], 1),
        (&[2, 4], &[3, 1], 0),
        (&[2, 1, 4], &[1, 3, 2], 2),
        (&[1, 3, 4], &[2, 2, 1], 1),
        (&[2, 3, 4], &[2, 0, 4], 1),
        (&[0, 3], &[1, 2], 1),
        (&[1, 3], &[0, 2], 1),
        (&[2, 0, 3], &[1, 0, 1], 1),
        (&[2, 0], &[2, 1], 1),
        (&[0, 0], &[1, 2], 1),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &(shape, index_shape, axis) in layouts {
            let array = complex(shape, dtype);
            let index_count: usize = index_shape.iter().product();
            for index_dtype in [
                DType::I8,
                DType::I16,
                DType::I32,
                DType::I64,
                DType::U8,
                DType::U16,
                DType::U32,
                DType::U64,
            ] {
                let signed = matches!(
                    index_dtype,
                    DType::I8 | DType::I16 | DType::I32 | DType::I64
                );
                let indices = (0..index_count)
                    .map(|index| {
                        if shape[axis] == 0 || shape.contains(&0) {
                            99
                        } else {
                            let value = (index % shape[axis]) as i64;
                            if signed && index % 2 == 1 {
                                value - shape[axis] as i64
                            } else {
                                value
                            }
                        }
                    })
                    .collect();
                let indices = UFuncArray::from_storage(
                    index_shape.to_vec(),
                    ArrayStorage::I64(indices).cast_to(index_dtype).unwrap(),
                )
                .unwrap();
                batch.along(&array, &indices, axis as isize);
                batch.along(&array, &indices, axis as isize - shape.len() as isize);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn take_along_axis_complex_validation_matches_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let array = complex(&[2, 3], dtype);
        for index_shape in [&[][..], &[1][..], &[2, 2][..], &[2, 2, 1][..]] {
            for index_dtype in [
                DType::Bool,
                DType::F16,
                DType::F32,
                DType::F64,
                DType::Complex64,
                DType::Complex128,
            ] {
                let indices = if index_dtype.is_complex() {
                    complex(index_shape, index_dtype)
                } else {
                    UFuncArray::zeros(index_shape.to_vec(), index_dtype).unwrap()
                };
                batch.along(&array, &indices, 1);
                batch.along(&array, &indices, 2);
            }
            let indices = UFuncArray::zeros(index_shape.to_vec(), DType::I64).unwrap();
            batch.along(&array, &indices, 0);
            batch.along(&array, &indices, 2);
        }
        for index in [-4, 3, i64::MIN, i64::MAX, (1_i64 << 53) + 1] {
            let indices =
                UFuncArray::from_storage(vec![1, 1], ArrayStorage::I64(vec![index])).unwrap();
            batch.along(&array, &indices, 1);
        }
        let scalar = complex(&[], dtype);
        batch.along(&scalar, &UFuncArray::scalar(0.0, DType::F64), 0);
        batch.along(&scalar, &UFuncArray::scalar(0.0, DType::I64), -1);
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_gathers_use_exact_unsigned_indices_and_preserve_real_sidecars() {
    let indices = UFuncArray::from_storage(
        vec![1, 3],
        ArrayStorage::U64(vec![u64::MAX, u64::MAX - 1, 0]),
    )
    .unwrap();
    assert_eq!(indices.values()[0], indices.values()[1]);
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        batch.along(&complex(&[2, 4], dtype), &indices, -1);
    }
    for storage in [
        ArrayStorage::I64(vec![i64::MAX, i64::MIN, (1_i64 << 53) + 1, -1, 0, 1]),
        ArrayStorage::U64(vec![u64::MAX, u64::MAX - 1, (1_u64 << 63) + 1, 0, 1, 2]),
    ] {
        let array = UFuncArray::from_storage(vec![2, 3], storage).unwrap();
        batch.take(&array, &[5, 0, -2], None);
        batch.take(&array, &[2, 0, -1], Some(1));
        batch.along(&array, &indices, 1);
    }
    let scalar = UFuncArray::from_storage(vec![], ArrayStorage::I64(vec![i64::MAX])).unwrap();
    for axis in [None, Some(0), Some(-1), Some(1), Some(-2)] {
        batch.take(&scalar, &[0, -1], axis);
        batch.take(&scalar, &[], axis);
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_gathers_reject_malformed_storage_and_indices() {
    let indices = UFuncArray::from_storage(vec![1], ArrayStorage::I64(vec![0])).unwrap();
    for malformed in [
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
        UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::Complex64).unwrap(),
        UFuncArray::new(vec![0, 1], vec![], DType::Complex128).unwrap(),
    ] {
        for axis in [None, Some(0)] {
            let error = malformed.take(&[], axis).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("complex arrays must use interleaved")
            );
        }
        assert!(malformed.take_along_axis(&indices, 0).is_err());
    }
    let source = complex(&[2, 3], DType::Complex128);
    for value in [0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let malformed = UFuncArray::new(vec![1, 1], vec![value], DType::I64).unwrap();
        assert!(source.take_along_axis(&malformed, 1).is_err());
    }
}

#[test]
fn complex_gathers_check_output_capacity_and_skip_empty_stride_products() {
    let index = UFuncArray::from_storage(vec![1, 1], ArrayStorage::I64(vec![0])).unwrap();
    let max_complex_count = isize::MAX as usize / std::mem::size_of::<[f64; 2]>();
    for length in [max_complex_count + 1, usize::MAX] {
        let source = UFuncArray::new(vec![0, length, 2], vec![], DType::Complex128).unwrap();
        assert!(matches!(
            source.take(&[0], Some(0)),
            Err(UFuncError::Shape(ShapeError::Overflow))
        ));
        assert!(matches!(
            source.take_along_axis(&index, 0),
            Err(UFuncError::Shape(ShapeError::Overflow))
        ));
        let empty = source.take(&[], None).unwrap();
        assert_eq!(empty.shape(), &[0, 2]);
        let empty = source.take(&[99], Some(1)).unwrap();
        assert_eq!(empty.shape(), &[0, 1, 2]);
        let empty = source.take_along_axis(&index, 1).unwrap();
        assert_eq!(empty.shape(), &[0, 1, 2]);
        assert!(empty.values().is_empty());
    }
    for dtype in [DType::F64, DType::I64, DType::U64] {
        let max_real_count = isize::MAX as usize / std::mem::size_of::<f64>();
        let source = UFuncArray::new(vec![0, max_real_count + 1], vec![], dtype).unwrap();
        assert!(matches!(
            source.take(&[0], Some(0)),
            Err(UFuncError::Shape(ShapeError::Overflow))
        ));
        let empty = source.take(&[99], Some(1)).unwrap();
        assert_eq!(empty.shape(), &[0, 1]);
        let trailing_empty = UFuncArray::new(vec![usize::MAX, 0], vec![], dtype).unwrap();
        assert_eq!(trailing_empty.take(&[0], Some(0)).unwrap().shape(), &[1, 0]);
    }
    #[cfg(target_pointer_width = "64")]
    {
        // NumPy reports MemoryError before bounds for this 1 EiB result.
        // Its count fits usize/isize, but exceeds the virtual address space;
        // try_reserve must fail recoverably without touching output pages.
        let source = UFuncArray::new(vec![0, 1usize << 56, 2], vec![], DType::Complex128).unwrap();
        let error = source.take(&[0], Some(0)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unable to allocate output array"),
            "allocation must precede empty-axis bounds: {error}"
        );
    }
}
