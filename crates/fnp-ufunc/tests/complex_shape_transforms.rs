//! Shape transforms operate on complete complex elements and logical axes.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ndarray::ShapeError;
use fnp_ufunc::{MaskedArray, UFuncArray, UFuncError};
use serde_json::{Value, json};
use std::process::{Command, Stdio};

fn complex(shape: &[usize], dtype: DType) -> UFuncArray {
    let count: usize = shape.iter().product();
    let mut physical_shape = shape.to_vec();
    physical_shape.push(2);
    let values = (0..count)
        .flat_map(|index| match index % 11 {
            0 => [0.0, -0.0],
            1 => [-0.0, 0.0],
            2 => [1.0 + 2.0_f64.powi(-24), -1.0 - 2.0_f64.powi(-24)],
            3 => [1.0e-50, -1.0e-50],
            4 => [f64::INFINITY, f64::NEG_INFINITY],
            5 => [f64::NAN, -(index as f64)],
            6 => [index as f64, f64::NAN],
            7 => [1.0e300, -1.0e300],
            _ => [index as f64 + 0.25, -(index as f64) - 0.75],
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
    inputs: Vec<Value>,
    cases: Vec<Value>,
    actual: Vec<Result<UFuncArray, UFuncError>>,
}

impl OracleBatch {
    fn input(&mut self, array: &UFuncArray) -> usize {
        let index = self.inputs.len();
        self.inputs.push(oracle_input(array));
        index
    }

    fn case(
        &mut self,
        input: usize,
        op: &str,
        args: Value,
        actual: Result<UFuncArray, UFuncError>,
    ) {
        self.cases
            .push(json!({"input": input, "op": op, "args": args}));
        self.actual.push(actual);
    }

    fn assert_matches_numpy(self) {
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

data = json.load(sys.stdin)
inputs = [load(item) for item in data['inputs']]
rows = []
for case in data['cases']:
    try:
        array = inputs[case['input']]
        op, args = case['op'], case['args']
        if op == 'reshape':
            result = np.reshape(array, args['shape'], order=args.get('order', 'C'))
        elif op == 'transpose':
            result = np.transpose(array, axes=args['axes'])
        elif op == 't':
            result = array.T
        elif op == 'matrix_transpose':
            result = np.matrix_transpose(array)
        elif op == 'flatten':
            result = array.flatten()
        elif op == 'ravel':
            result = np.ravel(array)
        elif op in ('squeeze', 'expand_dims', 'flip'):
            axis = args['axis']
            if isinstance(axis, list):
                axis = tuple(axis)
            result = getattr(np, op)(array, axis=axis)
        elif op == 'swapaxes':
            result = np.swapaxes(array, args['axis1'], args['axis2'])
        elif op == 'moveaxis':
            result = np.moveaxis(array, args['source'], args['destination'])
        elif op == 'rollaxis':
            result = np.rollaxis(array, args['axis'], args['start'])
        elif op == 'broadcast_to':
            result = np.broadcast_to(array, args['shape'])
        elif op in ('fliplr', 'flipud'):
            result = getattr(np, op)(array)
        elif op == 'rot90':
            result = np.rot90(array, k=args['k'], axes=args['axes'])
        elif op in ('diag', 'diagflat', 'tril', 'triu'):
            result = getattr(np, op)(array, k=args['k'])
        elif op in ('diagonal', 'trace'):
            result = getattr(np, op)(array, offset=args['offset'],
                                      axis1=args['axis1'], axis2=args['axis2'])
        elif op == 'diag_indices_from':
            result = np.asarray(np.diag_indices_from(array))
        elif op in ('tril_indices_from', 'triu_indices_from'):
            result = np.asarray(getattr(np, op)(array, k=args['k']))
        else:
            raise AssertionError(op)
        result = np.asarray(result)
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
    except (IndexError, TypeError, ValueError, OverflowError) as error:
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
            .expect("launch live NumPy shape transform oracle");
        let data = json!({"inputs": self.inputs, "cases": self.cases});
        serde_json::to_writer(child.stdin.take().unwrap(), &data).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy shape transform oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.actual.len());
        for ((case, actual), expected) in data["cases"]
            .as_array()
            .unwrap()
            .iter()
            .zip(self.actual)
            .zip(expected)
        {
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
                let integers = (0..storage.len())
                    .map(|index| storage.get_i128(index).unwrap().to_string())
                    .collect::<Vec<_>>();
                assert_eq!(
                    json!(integers),
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
fn complex_diagonal_construction_and_triangles_match_numpy() {
    let mut batch = OracleBatch::default();
    let shapes: &[&[usize]] = &[
        &[],
        &[0],
        &[1],
        &[4],
        &[0, 3],
        &[3, 0],
        &[2, 3],
        &[3, 2],
        &[3, 3],
        &[2, 3, 4],
        &[2, 0, 3],
        &[1, 2, 3, 4],
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        for &shape in shapes {
            let array = complex(shape, dtype);
            let input = batch.input(&array);
            for k in [-4, -2, -1, 0, 1, 2, 4] {
                batch.case(input, "diag", json!({"k": k}), array.diag(k));
                batch.case(input, "diagflat", json!({"k": k}), Ok(array.diagflat(k)));
                batch.case(input, "tril", json!({"k": k}), array.tril(k));
                batch.case(input, "triu", json!({"k": k}), array.triu(k));
                batch.case(
                    input,
                    "trace",
                    json!({"offset": k, "axis1": 0, "axis2": 1}),
                    array.trace(k),
                );
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_diagonal_and_trace_axes_match_numpy() {
    let mut batch = OracleBatch::default();
    let shapes: &[&[usize]] = &[
        &[],
        &[3],
        &[2, 3],
        &[2, 3, 4],
        &[3, 1, 4, 2],
        &[2, 0, 3],
        &[0, 3, 4],
        &[3, 4, 0],
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        for &shape in shapes {
            let array = complex(shape, dtype);
            let input = batch.input(&array);
            let rank = shape.len() as isize;
            for (axis1, axis2) in [
                (0, 1),
                (1, 0),
                (0, -1),
                (-1, 0),
                (-2, -1),
                (-1, -2),
                (1, 1),
                (-rank - 1, 0),
                (rank, 0),
            ] {
                for offset in [-3, -1, 0, 1, 3] {
                    let args = json!({"offset": offset, "axis1": axis1, "axis2": axis2});
                    batch.case(
                        input,
                        "diagonal",
                        args.clone(),
                        array.diagonal(offset, axis1, axis2),
                    );
                    batch.case(input, "trace", args, array.trace_axis(offset, axis1, axis2));
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_diagonal_index_helpers_count_logical_axes() {
    let mut batch = OracleBatch::default();
    let shapes: &[&[usize]] = &[
        &[],
        &[3],
        &[0, 0],
        &[0, 3],
        &[3, 0],
        &[2, 3],
        &[3, 2],
        &[3, 3],
        &[2, 2, 2],
        &[2, 2, 3],
    ];
    for dtype in [DType::Complex64, DType::Complex128] {
        for &shape in shapes {
            let array = complex(shape, dtype);
            let input = batch.input(&array);
            let diagonal = UFuncArray::diag_indices_from(&array)
                .and_then(|(axes, _)| UFuncArray::stack(&axes.iter().collect::<Vec<_>>(), 0));
            batch.case(input, "diag_indices_from", json!({}), diagonal);
            for k in [-4, -1, 0, 1, 4] {
                let lower = UFuncArray::tril_indices_from(&array, k)
                    .and_then(|(rows, cols)| UFuncArray::stack(&[&rows, &cols], 0));
                let upper = UFuncArray::triu_indices_from(&array, k)
                    .and_then(|(rows, cols)| UFuncArray::stack(&[&rows, &cols], 0));
                batch.case(input, "tril_indices_from", json!({"k": k}), lower);
                batch.case(input, "triu_indices_from", json!({"k": k}), upper);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn diagonal_family_preserves_real_integer_payloads_and_trace_promotion() {
    let mut batch = OracleBatch::default();
    let mut storage = vec![
        ArrayStorage::I64(vec![i64::MAX, (1_i64 << 53) + 1, 3, -4, i64::MIN, 8]),
        ArrayStorage::U64(vec![u64::MAX, (1_u64 << 53) + 1, 3, 4, 1_u64 << 63, 8]),
    ];
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::U8,
        DType::U16,
        DType::U32,
    ] {
        storage.push(
            ArrayStorage::F64(vec![127.0, 3.0, 1.0, 4.0, 127.0, 2.0])
                .cast_to(dtype)
                .unwrap(),
        );
    }
    for cells in storage {
        for shape in [vec![6], vec![2, 3], vec![1, 2, 3]] {
            let array = UFuncArray::from_storage(shape, cells.clone()).unwrap();
            let input = batch.input(&array);
            for k in [-2, -1, 0, 1, 2] {
                batch.case(input, "diag", json!({"k": k}), array.diag(k));
                batch.case(input, "diagflat", json!({"k": k}), Ok(array.diagflat(k)));
                batch.case(input, "tril", json!({"k": k}), array.tril(k));
                batch.case(input, "triu", json!({"k": k}), array.triu(k));
                for (axis1, axis2) in [(0, 1), (1, 0), (-2, -1), (-1, -2)] {
                    let args = json!({"offset": k, "axis1": axis1, "axis2": axis2});
                    batch.case(
                        input,
                        "diagonal",
                        args.clone(),
                        array.diagonal(k, axis1, axis2),
                    );
                    batch.case(input, "trace", args, array.trace_axis(k, axis1, axis2));
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_trace_keeps_both_components_and_source_precision() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let array = UFuncArray::new(
            vec![2, 2, 2],
            vec![1.0, 2.0, f64::NAN, f64::INFINITY, 99.0, -99.0, 3.0, 4.0],
            dtype,
        )
        .unwrap();
        let trace = array.trace(0).unwrap();
        assert_eq!(trace.shape(), &[2]);
        assert_eq!(trace.dtype(), dtype);
        assert_eq!(trace.values(), &[4.0, 6.0]);
        assert_eq!(
            array
                .diag(0)
                .unwrap()
                .sum_extracted_diagonal_to_scalar()
                .values(),
            &[4.0, 6.0]
        );
        let rounded =
            UFuncArray::new(vec![1, 1, 2], vec![16_777_217.0, -16_777_219.0], dtype).unwrap();
        let expected = if dtype == DType::Complex64 {
            [16_777_216.0, -16_777_220.0]
        } else {
            [16_777_217.0, -16_777_219.0]
        };
        assert_eq!(rounded.trace(0).unwrap().values(), &expected);
    }
}

#[test]
fn diagonal_family_rejects_malformed_layouts_and_handles_extreme_empty_shapes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        for malformed in [
            UFuncArray::new(vec![], vec![1.0], dtype).unwrap(),
            UFuncArray::new(vec![2, 3], vec![1.0; 6], dtype).unwrap(),
            UFuncArray::new(vec![0, 1], vec![], dtype).unwrap(),
        ] {
            assert!(malformed.diag(0).is_err());
            assert!(malformed.diagonal(0, 0, 1).is_err());
            assert!(malformed.trace(0).is_err());
            assert!(malformed.trace_axis(0, 0, 1).is_err());
            assert!(malformed.tril(0).is_err());
            assert!(malformed.triu(0).is_err());
            assert!(UFuncArray::diag_indices_from(&malformed).is_err());
            assert!(UFuncArray::tril_indices_from(&malformed, 0).is_err());
            assert!(UFuncArray::triu_indices_from(&malformed, 0).is_err());
        }
        let matrix = complex(&[2, 3], dtype);
        let vector = complex(&[3], dtype);
        for offset in [i64::MIN, i64::MAX] {
            assert_eq!(matrix.diag(offset).unwrap().shape(), &[0, 2]);
            assert_eq!(matrix.diagonal(offset, 1, 0).unwrap().shape(), &[0, 2]);
            assert_eq!(
                matrix.trace_axis(offset, 1, 0).unwrap().values(),
                &[0.0, 0.0]
            );
            assert!(vector.diag(offset).is_err());
            assert_eq!(vector.diagflat(offset).shape(), &[0, 0, 2]);
        }
        assert!(
            matrix
                .tril(i64::MIN)
                .unwrap()
                .values()
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(
            matrix
                .triu(i64::MAX)
                .unwrap()
                .values()
                .iter()
                .all(|&value| value == 0.0)
        );
        let huge = UFuncArray::new(vec![usize::MAX, 0, usize::MAX, 2], vec![], dtype).unwrap();
        assert_eq!(huge.diagonal(0, 0, 2).unwrap().shape(), &[0, usize::MAX, 2]);
        assert!(huge.tril(0).unwrap().values().is_empty());
        assert!(huge.triu(0).unwrap().values().is_empty());
        assert!(matches!(
            huge.trace_axis(0, 1, 2),
            Err(UFuncError::Shape(ShapeError::Overflow))
        ));
        let over_capacity = isize::MAX as usize / std::mem::size_of::<[f64; 2]>() + 1;
        let empty_diagonals = UFuncArray::new(vec![over_capacity, 0, 1, 2], vec![], dtype).unwrap();
        assert!(matches!(
            empty_diagonals.trace_axis(0, 1, 2),
            Err(UFuncError::Shape(ShapeError::Overflow))
        ));
        let huge_matrix = UFuncArray::new(vec![usize::MAX, 0, 2], vec![], dtype).unwrap();
        assert!(
            UFuncArray::tril_indices_from(&huge_matrix, 0)
                .unwrap()
                .0
                .values()
                .is_empty()
        );
        assert!(
            UFuncArray::triu_indices_from(&huge_matrix, 0)
                .unwrap()
                .0
                .values()
                .is_empty()
        );
    }
}

#[test]
fn complex_reshape_order_and_flatten_match_numpy() {
    let shapes: &[&[usize]] = &[
        &[],
        &[1],
        &[6],
        &[2, 3],
        &[2, 1, 3],
        &[2, 3, 4],
        &[0],
        &[0, 3],
        &[2, 0, 1],
    ];
    let targets: &[&[isize]] = &[
        &[],
        &[-1],
        &[1, -1],
        &[-1, 1],
        &[2, -1],
        &[3, 2],
        &[-2],
        &[-3, 2],
        &[0],
        &[0, 3],
        &[0, -1],
        &[-1, -2],
        &[5],
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &shape in shapes {
            let array = complex(shape, dtype);
            let input = batch.input(&array);
            batch.case(input, "ravel", json!({}), Ok(array.ravel()));
            batch.case(input, "flatten", json!({}), Ok(array.flatten()));
            for &target in targets {
                batch.case(
                    input,
                    "reshape",
                    json!({"shape": target}),
                    array.reshape(target),
                );
                for order in ["C", "F", "A", "c", "f", "a", "K", "", "CF"] {
                    batch.case(
                        input,
                        "reshape",
                        json!({"shape": target, "order": order}),
                        array.reshape_order(target, order),
                    );
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_axis_permutations_match_numpy() {
    let shapes: &[&[usize]] = &[&[], &[8], &[2, 4], &[2, 3, 4], &[2, 0, 3]];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &shape in shapes {
            let array = complex(shape, dtype);
            let input = batch.input(&array);
            batch.case(input, "t", json!({}), array.t());
            batch.case(
                input,
                "transpose",
                json!({"axes": null}),
                array.transpose(None),
            );
            batch.case(
                input,
                "matrix_transpose",
                json!({}),
                array.matrix_transpose(),
            );
            let permutations: &[&[usize]] = match shape.len() {
                0 => &[&[]],
                1 => &[&[0]],
                2 => &[&[0, 1], &[1, 0]],
                _ => &[
                    &[0, 1, 2],
                    &[0, 2, 1],
                    &[1, 0, 2],
                    &[1, 2, 0],
                    &[2, 0, 1],
                    &[2, 1, 0],
                ],
            };
            for &axes in permutations {
                batch.case(
                    input,
                    "transpose",
                    json!({"axes": axes}),
                    array.transpose(Some(axes)),
                );
                batch.case(
                    input,
                    "transpose",
                    json!({"axes": axes}),
                    array.permute_dims(axes),
                );
            }
            for axes in [
                vec![shape.len()],
                vec![0; shape.len() + 1],
                vec![0; shape.len()],
            ] {
                batch.case(
                    input,
                    "transpose",
                    json!({"axes": axes}),
                    array.transpose(Some(&axes)),
                );
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_squeeze_and_expand_dims_validate_logical_axes() {
    let shapes: &[&[usize]] = &[&[], &[1], &[8], &[1, 2, 1], &[1, 1, 1], &[0, 1, 3]];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &shape in shapes {
            let array = complex(shape, dtype);
            let input = batch.input(&array);
            let rank = shape.len() as isize;
            batch.case(input, "squeeze", json!({"axis": null}), array.squeeze(None));
            for axis in (-rank - 2..=rank + 1).chain([isize::MIN, isize::MAX]) {
                batch.case(
                    input,
                    "squeeze",
                    json!({"axis": axis}),
                    array.squeeze(Some(axis)),
                );
                batch.case(
                    input,
                    "expand_dims",
                    json!({"axis": axis}),
                    array.expand_dims(axis),
                );
            }
            let tuples: &[&[isize]] = &[
                &[],
                &[0],
                &[-1],
                &[0, -1],
                &[0, 1],
                &[1, 0],
                &[0, 0],
                &[-1, -2],
                &[0, 2],
                &[isize::MIN],
                &[isize::MAX],
            ];
            for &axes in tuples {
                batch.case(
                    input,
                    "squeeze",
                    json!({"axis": axes}),
                    array.squeeze_axes(axes),
                );
                batch.case(
                    input,
                    "expand_dims",
                    json!({"axis": axes}),
                    array.expand_dims_axes(axes),
                );
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_swap_move_and_roll_axes_match_numpy() {
    let shapes: &[&[usize]] = &[&[], &[8], &[2, 4], &[2, 3, 4], &[2, 0, 3]];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &shape in shapes {
            let array = complex(shape, dtype);
            let input = batch.input(&array);
            let rank = shape.len() as isize;
            for source in -rank - 1..=rank {
                for destination in -rank - 1..=rank {
                    batch.case(
                        input,
                        "swapaxes",
                        json!({"axis1": source, "axis2": destination}),
                        array.swapaxes(source, destination),
                    );
                    batch.case(
                        input,
                        "moveaxis",
                        json!({"source": source, "destination": destination}),
                        array.moveaxis(source, destination),
                    );
                    batch.case(
                        input,
                        "rollaxis",
                        json!({"axis": source, "start": destination}),
                        array.rollaxis(source, destination),
                    );
                }
            }
            let moves: &[(&[isize], &[isize])] = &[
                (&[], &[]),
                (&[0], &[-1]),
                (&[0, 1], &[1, 0]),
                (&[0, -1], &[-1, 0]),
                (&[0, 2], &[2, 0]),
                (&[2, 0], &[0, 1]),
                (&[0, 0], &[0, 1]),
                (&[0, 1], &[0, 0]),
                (&[0], &[]),
                (&[isize::MIN], &[0]),
                (&[0], &[isize::MAX]),
            ];
            for &(source, destination) in moves {
                batch.case(
                    input,
                    "moveaxis",
                    json!({"source": source, "destination": destination}),
                    array.moveaxis_multi(source, destination),
                );
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_broadcast_to_uses_logical_target_shapes() {
    let shapes: &[&[usize]] = &[&[], &[1], &[3], &[2, 1], &[1, 3], &[2, 3], &[0, 1], &[1, 0]];
    let targets: &[&[usize]] = &[
        &[],
        &[1],
        &[0],
        &[3],
        &[2, 3],
        &[4, 2, 3],
        &[2, 0, 3],
        &[0, 3],
        &[3, 2],
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &shape in shapes {
            let array = complex(shape, dtype);
            let input = batch.input(&array);
            for &target in targets {
                batch.case(
                    input,
                    "broadcast_to",
                    json!({"shape": target}),
                    array.broadcast_to(target),
                );
            }
        }
        // Exercise the pair-preserving block kernels above their parallel thresholds.
        let array = UFuncArray::new(vec![2, 1, 2], vec![2.5, -1.25, -3.25, 5.5], dtype).unwrap();
        let input = batch.input(&array);
        batch.case(
            input,
            "broadcast_to",
            json!({"shape": [2, 17000]}),
            array.broadcast_to(&[2, 17000]),
        );
        let array = complex(&[2, 129, 131], dtype);
        let input = batch.input(&array);
        batch.case(
            input,
            "transpose",
            json!({"axes": [2, 0, 1]}),
            array.transpose(Some(&[2, 0, 1])),
        );
    }
    batch.assert_matches_numpy();
}

#[test]
fn shape_transforms_preserve_exact_integer_sidecars() {
    let mut batch = OracleBatch::default();
    for storage in [
        ArrayStorage::I64(vec![i64::MAX, i64::MIN, (1_i64 << 53) + 1, -1, 0, 1]),
        ArrayStorage::U64(vec![u64::MAX, u64::MAX - 1, (1_u64 << 63) + 1, 0, 1, 2]),
    ] {
        let array = UFuncArray::from_storage(vec![2, 3], storage).unwrap();
        let input = batch.input(&array);
        for target in [&[3, 2][..], &[-2][..], &[1, 3, 2][..]] {
            for order in ["C", "F", "A", "c", "f", "a"] {
                batch.case(
                    input,
                    "reshape",
                    json!({"shape": target, "order": order}),
                    array.reshape_order(target, order),
                );
            }
        }
        batch.case(
            input,
            "transpose",
            json!({"axes": [1, 0]}),
            array.transpose(Some(&[1, 0])),
        );
        batch.case(input, "ravel", json!({}), Ok(array.ravel()));
        batch.case(
            input,
            "expand_dims",
            json!({"axis": [0, 2]}),
            array.expand_dims_axes(&[0, 2]),
        );
        batch.case(
            input,
            "broadcast_to",
            json!({"shape": [4, 2, 3]}),
            array.broadcast_to(&[4, 2, 3]),
        );
        let expanded = array.expand_dims(1).unwrap();
        let input = batch.input(&expanded);
        batch.case(
            input,
            "squeeze",
            json!({"axis": [1]}),
            expanded.squeeze_axes(&[1]),
        );
        batch.case(
            input,
            "moveaxis",
            json!({"source": [0, 2], "destination": [2, 0]}),
            expanded.moveaxis_multi(&[0, 2], &[2, 0]),
        );
    }
    batch.assert_matches_numpy();
}

#[test]
fn shape_transform_composition_preserves_all_numeric_dtypes() {
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
        DType::Complex64,
        DType::Complex128,
    ] {
        let array = if dtype.is_complex() {
            complex(&[2, 1, 3], dtype)
        } else {
            let storage = ArrayStorage::F64(vec![
                0.0,
                -0.0,
                1.0 + 2.0_f64.powi(-24),
                1.0e-50,
                -3.5,
                127.0,
            ]);
            UFuncArray::from_storage(vec![2, 1, 3], storage.cast_to(dtype).unwrap()).unwrap()
        };
        let input = batch.input(&array);
        batch.case(
            input,
            "reshape",
            json!({"shape": [-3, 2]}),
            array.reshape(&[-3, 2]),
        );
        batch.case(
            input,
            "reshape",
            json!({"shape": [3, 2], "order": "F"}),
            array.reshape_order(&[3, 2], "F"),
        );
        batch.case(
            input,
            "transpose",
            json!({"axes": [2, 0, 1]}),
            array.transpose(Some(&[2, 0, 1])),
        );
        batch.case(input, "squeeze", json!({"axis": 1}), array.squeeze(Some(1)));
        batch.case(
            input,
            "expand_dims",
            json!({"axis": [0, 3]}),
            array.expand_dims_axes(&[0, 3]),
        );
        batch.case(
            input,
            "broadcast_to",
            json!({"shape": [4, 2, 1, 3]}),
            array.broadcast_to(&[4, 2, 1, 3]),
        );
        let transposed = array.transpose(Some(&[2, 0, 1])).unwrap();
        let input = batch.input(&transposed);
        batch.case(
            input,
            "reshape",
            json!({"shape": [2, 3], "order": "F"}),
            transposed.reshape_order(&[2, 3], "F"),
        );
        batch.case(input, "ravel", json!({}), Ok(transposed.ravel()));
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_shape_transforms_reject_malformed_storage_and_oversized_outputs() {
    for malformed in [
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
        UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::Complex64).unwrap(),
        UFuncArray::new(vec![0, 1], vec![], DType::Complex128).unwrap(),
    ] {
        for result in [
            malformed.reshape(&[-1]),
            malformed.reshape_order(&[-1], "F"),
            malformed.transpose(None),
            malformed.matrix_transpose(),
            malformed.squeeze(None),
            malformed.squeeze_axes(&[]),
            malformed.expand_dims(0),
            malformed.expand_dims_axes(&[]),
            malformed.swapaxes(0, 0),
            malformed.moveaxis_multi(&[], &[]),
            malformed.rollaxis(0, 0),
            malformed.broadcast_to(&[2, 3]),
            malformed.flip(None),
            malformed.flip(Some(0)),
            malformed.flip_axes(&[]),
            malformed.fliplr(),
            malformed.flipud(),
            malformed.rot90(0),
            malformed.rot90_axes(1, (0, 1)),
        ] {
            assert!(
                result.is_err(),
                "malformed complex storage must be rejected"
            );
        }
    }
    let scalar = complex(&[], DType::Complex128);
    let max_complex_count = isize::MAX as usize / std::mem::size_of::<[f64; 2]>();
    for length in [max_complex_count + 1, usize::MAX / 2 + 1, usize::MAX] {
        assert!(matches!(
            scalar.broadcast_to(&[length]),
            Err(UFuncError::Shape(ShapeError::Overflow))
        ));
    }
    // An empty result has no cells or stride products to allocate.
    let empty = scalar.broadcast_to(&[0, usize::MAX]).unwrap();
    assert_eq!(empty.shape(), &[0, usize::MAX, 2]);
    assert!(empty.values().is_empty());
}

#[test]
fn complex_flips_and_rotations_match_numpy() {
    let shapes: &[&[usize]] = &[&[], &[1], &[8], &[2, 3], &[1, 3], &[2, 3, 4], &[2, 0, 3]];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &shape in shapes {
            let array = complex(shape, dtype);
            let input = batch.input(&array);
            let rank = shape.len() as isize;
            batch.case(input, "flip", json!({"axis": null}), array.flip(None));
            batch.case(input, "fliplr", json!({}), array.fliplr());
            batch.case(input, "flipud", json!({}), array.flipud());
            for axis in (-rank - 1..=rank).chain([isize::MIN, isize::MAX]) {
                batch.case(input, "flip", json!({"axis": axis}), array.flip(Some(axis)));
            }
            let tuples: &[&[isize]] = &[
                &[],
                &[0],
                &[-1],
                &[0, 1],
                &[1, 0],
                &[0, -1],
                &[0, 2],
                &[0, 0],
                &[0, -rank],
                &[0, rank],
                &[isize::MIN],
                &[isize::MAX],
            ];
            for &axes in tuples {
                batch.case(input, "flip", json!({"axis": axes}), array.flip_axes(axes));
            }
            for axes in [
                (0, 1),
                (1, 0),
                (-1, -2),
                (-2, -1),
                (0, -1),
                (-1, 0),
                (0, 0),
                (0, rank),
                (isize::MIN, 0),
                (0, isize::MAX),
            ] {
                for k in [-5, -2, -1, 0, 1, 2, 3, 4, i32::MIN, i32::MAX] {
                    batch.case(
                        input,
                        "rot90",
                        json!({"k": k, "axes": [axes.0, axes.1]}),
                        array.rot90_axes(k, axes),
                    );
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn flips_and_rotations_preserve_exact_integer_sidecars() {
    let mut batch = OracleBatch::default();
    for storage in [
        ArrayStorage::I64(vec![i64::MAX, i64::MIN, (1_i64 << 53) + 1, -1, 0, 1]),
        ArrayStorage::U64(vec![u64::MAX, u64::MAX - 1, (1_u64 << 63) + 1, 0, 1, 2]),
    ] {
        let array = UFuncArray::from_storage(vec![2, 3], storage).unwrap();
        let input = batch.input(&array);
        batch.case(input, "flip", json!({"axis": null}), array.flip(None));
        batch.case(input, "fliplr", json!({}), array.fliplr());
        batch.case(input, "flipud", json!({}), array.flipud());
        for axes in [&[0, 1][..], &[1, 0][..], &[-1][..], &[0, 0][..]] {
            batch.case(input, "flip", json!({"axis": axes}), array.flip_axes(axes));
        }
        for k in [-1, 0, 1, 2, 3, 4] {
            batch.case(
                input,
                "rot90",
                json!({"k": k, "axes": [0, 1]}),
                array.rot90(k),
            );
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn masked_complex_shape_only_operations_keep_physical_masks_aligned() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let data = UFuncArray::new(vec![2, 3, 2], (0..12).map(f64::from).collect(), dtype).unwrap();
        let mask = UFuncArray::from_storage(
            vec![2, 3, 2],
            ArrayStorage::Bool(vec![
                true, false, false, false, false, true, true, true, false, true, true, false,
            ]),
        )
        .unwrap();
        for mask in [Some(mask), None] {
            let mut array = MaskedArray::new(data.clone(), mask.clone(), Some(-19.5)).unwrap();
            array.harden_mask();
            for (result, shape) in [
                (array.reshape(&[3, 2]).unwrap(), vec![3, 2, 2]),
                (array.ravel(), vec![6, 2]),
                (array.expand_dims(-1).unwrap(), vec![2, 3, 1, 2]),
                (array.expand_dims(0).unwrap(), vec![1, 2, 3, 2]),
                (
                    array.expand_dims(-1).unwrap().squeeze(Some(-1)).unwrap(),
                    vec![2, 3, 2],
                ),
                (
                    array.expand_dims(0).unwrap().squeeze(None).unwrap(),
                    vec![2, 3, 2],
                ),
            ] {
                assert_eq!(result.data().shape(), shape);
                assert_eq!(result.data().values(), data.values());
                assert_eq!(result.dtype(), dtype);
                assert_eq!(result.fill_value(), -19.5);
                match (result.mask(), mask.as_ref()) {
                    (Some(actual), Some(original)) => {
                        assert_eq!(actual.shape(), result.data().shape());
                        assert_eq!(actual.dtype(), original.dtype());
                        assert_eq!(actual.values(), original.values());
                    }
                    (None, None) => {}
                    _ => panic!("shape transforms must preserve mask presence"),
                }
            }
            assert!(array.reshape(&[7]).is_err());
            assert!(array.squeeze(Some(-1)).is_err());
            assert!(array.expand_dims(4).is_err());
        }

        for (shape, values, target) in [
            (vec![2], vec![3.0, -4.0], vec![]),
            (vec![2, 0, 3, 2], vec![], vec![0, 6]),
        ] {
            let mask =
                UFuncArray::new(shape.clone(), vec![1.0; values.len()], DType::Bool).unwrap();
            let array = MaskedArray::new(
                UFuncArray::new(shape, values, dtype).unwrap(),
                Some(mask),
                None,
            )
            .unwrap();
            for result in [
                array.reshape(&target).unwrap(),
                array.ravel(),
                array.expand_dims(-1).unwrap(),
                array.transpose(None).unwrap(),
            ] {
                assert_eq!(result.mask().unwrap().shape(), result.data().shape());
                assert_eq!(
                    result.mask().unwrap().values().len(),
                    result.data().values().len()
                );
            }
        }
    }
}

#[test]
fn masked_complex_transpose_relocates_distinct_component_masks() {
    let source_masks = vec![
        true, false, false, false, false, true, true, true, false, true, true, false,
    ];
    let expected_cells = [0usize, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11];
    for dtype in [DType::Complex64, DType::Complex128] {
        let array = MaskedArray::new(
            UFuncArray::new(vec![2, 3, 2], (0..12).map(f64::from).collect(), dtype).unwrap(),
            Some(
                UFuncArray::from_storage(vec![2, 3, 2], ArrayStorage::Bool(source_masks.clone()))
                    .unwrap(),
            ),
            Some(-19.5),
        )
        .unwrap();
        for axes in [None, Some(&[1usize, 0][..])] {
            let result = array.transpose(axes).unwrap();
            assert_eq!(result.data().shape(), &[3, 2, 2]);
            let mask = result.mask().unwrap();
            assert_eq!(mask.shape(), result.data().shape());
            assert_eq!(result.fill_value(), -19.5);
            for (output, &source) in expected_cells.iter().enumerate() {
                assert_eq!(result.data().values()[output], source as f64);
                assert_eq!(
                    mask.values()[output],
                    if source_masks[source] { 1.0 } else { 0.0 }
                );
            }
            assert_eq!(
                result.transpose(None).unwrap().mask().unwrap().values(),
                array.mask().unwrap().values()
            );
        }
        assert!(array.transpose(Some(&[0, 0])).is_err());
        assert!(array.transpose(Some(&[0, 1, 2])).is_err());
    }
}

#[test]
fn masked_shape_operations_preserve_accepted_mask_dtypes_and_exact_payloads() {
    let shape = vec![2, 3, 2];
    let masks = [
        UFuncArray::from_storage(
            shape.clone(),
            ArrayStorage::I64((0..12).map(|index| (1_i64 << 53) + index).collect()),
        )
        .unwrap(),
        UFuncArray::new(
            shape.clone(),
            (0..12)
                .map(|index| {
                    if index == 0 {
                        -0.0
                    } else {
                        f64::from(index) + 2.0_f64.powi(-30)
                    }
                })
                .collect(),
            DType::Complex64,
        )
        .unwrap(),
    ];
    // The existing constructor accepts masks of any dtype when physical shapes
    // match. Shape operations must retain that payload without rounding it.
    for data_dtype in [DType::F64, DType::Complex128] {
        for mask in &masks {
            let array = MaskedArray::new(
                UFuncArray::new(shape.clone(), (0..12).map(f64::from).collect(), data_dtype)
                    .unwrap(),
                Some(mask.clone()),
                None,
            )
            .unwrap();
            let target: &[isize] = if data_dtype.is_complex() {
                &[3, 2]
            } else {
                &[3, 2, 2]
            };
            let reshaped = array.reshape(target).unwrap();
            assert_eq!(reshaped.mask().unwrap().shape(), &[3, 2, 2]);
            assert_eq!(reshaped.mask().unwrap().dtype(), mask.dtype());
            for (actual, expected) in reshaped.mask().unwrap().values().iter().zip(mask.values()) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
            let result = array.transpose(None).unwrap();
            let expected_cells = if data_dtype.is_complex() {
                [0usize, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11]
            } else {
                [0usize, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11]
            };
            let actual_mask = result.mask().unwrap();
            assert_eq!(actual_mask.shape(), result.data().shape());
            assert_eq!(actual_mask.dtype(), mask.dtype());
            for (output, &source) in expected_cells.iter().enumerate() {
                assert_eq!(
                    actual_mask.values()[output].to_bits(),
                    mask.values()[source].to_bits()
                );
            }
            if mask.dtype().is_integer() {
                let source = mask.to_storage().unwrap();
                let reshaped = reshaped.mask().unwrap().to_storage().unwrap();
                let transposed = actual_mask.to_storage().unwrap();
                for (output, &input) in expected_cells.iter().enumerate() {
                    assert_eq!(reshaped.get_i128(output), source.get_i128(output));
                    assert_eq!(transposed.get_i128(output), source.get_i128(input));
                }
            }
        }
    }
}
