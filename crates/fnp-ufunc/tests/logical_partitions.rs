//! Splitting and joint broadcasting preserve logical elements and each input dtype.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Stdio};

fn array(shape: &[usize], dtype: DType) -> UFuncArray {
    let count: usize = shape.iter().product();
    if dtype.is_complex() {
        let mut physical_shape = shape.to_vec();
        physical_shape.push(2);
        let values = (0..count)
            .flat_map(|index| match index % 10 {
                0 => [0.0, -0.0],
                1 => [-0.0, 0.0],
                2 => [1.0 + 2.0_f64.powi(-24), -1.0 - 2.0_f64.powi(-24)],
                3 => [1e-50, -1e-50],
                4 => [f64::INFINITY, f64::NEG_INFINITY],
                5 => [f64::NAN, index as f64],
                6 => [index as f64, f64::NAN],
                _ => [index as f64 + 0.25, -(index as f64) - 0.75],
            })
            .collect();
        return UFuncArray::new(physical_shape, values, dtype).unwrap();
    }
    if dtype == DType::I64 {
        let values = [
            i64::MIN,
            -9_007_199_254_740_993,
            -1,
            0,
            9_007_199_254_740_993,
            i64::MAX,
        ];
        return UFuncArray::from_storage(
            shape.to_vec(),
            ArrayStorage::I64(
                (0..count)
                    .map(|index| values[index % values.len()])
                    .collect(),
            ),
        )
        .unwrap();
    }
    if dtype == DType::U64 {
        let values = [0, 1, 9_007_199_254_740_993, u64::MAX - 1, u64::MAX];
        return UFuncArray::from_storage(
            shape.to_vec(),
            ArrayStorage::U64(
                (0..count)
                    .map(|index| values[index % values.len()])
                    .collect(),
            ),
        )
        .unwrap();
    }
    let values = (0..count)
        .map(|index| {
            if dtype == DType::Bool {
                if index % 3 != 0 { 1.0 } else { 0.0 }
            } else {
                match index % 7 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => f64::NAN,
                    3 => f64::INFINITY,
                    4 => f64::NEG_INFINITY,
                    _ => index as f64 - 8.25,
                }
            }
        })
        .collect();
    UFuncArray::new(shape.to_vec(), values, dtype).unwrap()
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::I64 => "int64",
        DType::U64 => "uint64",
        DType::F16 => "float16",
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
        _ => panic!("unexpected partition oracle dtype"),
    }
}

fn encode(array: &UFuncArray) -> Value {
    let mut shape = array.shape().to_vec();
    let values = if array.dtype().is_complex() {
        assert_eq!(shape.pop(), Some(2));
        json!(
            array
                .values()
                .chunks_exact(2)
                .map(|pair| [pair[0].to_string(), pair[1].to_string()])
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
    actual: Vec<Result<Vec<UFuncArray>, UFuncError>>,
}

impl OracleBatch {
    fn input(&mut self, array: &UFuncArray) -> usize {
        let index = self.inputs.len();
        self.inputs.push(encode(array));
        index
    }

    fn case(
        &mut self,
        inputs: &[usize],
        op: &str,
        args: Value,
        actual: Result<Vec<UFuncArray>, UFuncError>,
    ) {
        self.cases
            .push(json!({"inputs": inputs, "op": op, "args": args}));
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
        result = np.empty(len(item['values']), dtype=dtype)
        result.real = [float(pair[0]) for pair in item['values']]
        result.imag = [float(pair[1]) for pair in item['values']]
    else:
        convert = int if dtype.kind in 'biu' else float
        result = np.array([convert(value) for value in item['values']], dtype=dtype)
    return result.reshape(item['shape'])

def encode(array):
    flat = array.reshape(-1)
    if array.dtype.kind == 'c':
        values = [[str(float(value.real)), str(float(value.imag))] for value in flat]
    elif array.dtype.kind in 'biu':
        values = [str(int(value)) for value in flat]
    else:
        values = [str(float(value)) for value in flat]
    return dict(shape=list(array.shape), dtype=str(array.dtype), values=values)

data = json.load(sys.stdin)
inputs = [load(item) for item in data['inputs']]
rows = []
for case in data['cases']:
    try:
        arrays = [inputs[index] for index in case['inputs']]
        op, args = case['op'], case['args']
        if op == 'broadcast_arrays':
            result = np.broadcast_arrays(*arrays)
        elif op == 'unstack':
            result = np.unstack(arrays[0], axis=args['axis'])
        elif op in ('split', 'array_split'):
            result = getattr(np, op)(arrays[0], args['sections'], axis=args['axis'])
        elif op in ('split_at', 'array_split_at_indices'):
            operation = np.split if op == 'split_at' else np.array_split
            result = operation(arrays[0], args['indices'], axis=args['axis'])
        else:
            result = getattr(np, op)(arrays[0], args['sections'])
        rows.append(dict(arrays=[encode(array) for array in result]))
    except Exception as error:
        rows.append(dict(error=type(error).__name__))
json.dump(rows, sys.stdout)
"#;
        let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(python)
            .args(["-c", script])
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("NumPy partition oracle must start");
        child
            .stdin
            .take()
            .unwrap()
            .write_all(
                &serde_json::to_vec(&json!({"inputs": self.inputs, "cases": self.cases})).unwrap(),
            )
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.actual.len());
        for (case_index, (actual, expected)) in self.actual.into_iter().zip(expected).enumerate() {
            let context = format!("case {case_index}: {}", self.cases[case_index]);
            if expected.get("error").is_some() {
                assert!(
                    actual.is_err(),
                    "{context}: expected {expected}, got {actual:?}"
                );
                continue;
            }
            let actual = actual.unwrap_or_else(|error| panic!("{context}: {error:?}"));
            let expected = expected["arrays"].as_array().unwrap();
            assert_eq!(actual.len(), expected.len(), "{context}: output count");
            for (part, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                let context = format!("{context}, output {part}");
                let mut shape: Vec<usize> =
                    serde_json::from_value(expected["shape"].clone()).unwrap();
                if actual.dtype().is_complex() {
                    shape.push(2);
                }
                assert_eq!(actual.shape(), shape, "{context}: shape");
                assert_eq!(
                    dtype_name(actual.dtype()),
                    expected["dtype"].as_str().unwrap(),
                    "{context}: dtype"
                );
                let expected_values = expected["values"].as_array().unwrap();
                if actual.dtype().is_integer() || actual.dtype() == DType::Bool {
                    let storage = actual.to_storage().unwrap();
                    assert_eq!(
                        storage.len(),
                        expected_values.len(),
                        "{context}: integer length"
                    );
                    for (index, value) in expected_values.iter().enumerate() {
                        let expected: i128 = value.as_str().unwrap().parse().unwrap();
                        assert_eq!(
                            storage.get_i128(index).unwrap(),
                            expected,
                            "{context}: integer {index}"
                        );
                    }
                } else {
                    let expected: Vec<f64> = if actual.dtype().is_complex() {
                        expected_values
                            .iter()
                            .flat_map(|pair| {
                                pair.as_array()
                                    .unwrap()
                                    .iter()
                                    .map(|value| value.as_str().unwrap().parse::<f64>().unwrap())
                            })
                            .collect()
                    } else {
                        expected_values
                            .iter()
                            .map(|value| value.as_str().unwrap().parse::<f64>().unwrap())
                            .collect()
                    };
                    assert_eq!(
                        actual.values().len(),
                        expected.len(),
                        "{context}: cell count"
                    );
                    for (index, (&actual, expected)) in
                        actual.values().iter().zip(expected).enumerate()
                    {
                        if expected.is_nan() {
                            assert!(
                                actual.is_nan(),
                                "{context}: cell {index}: expected NaN, got {actual}"
                            );
                        } else {
                            assert_eq!(
                                actual.to_bits(),
                                expected.to_bits(),
                                "{context}: cell {index}: {actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn logical_splitting_and_unstack_match_numpy() {
    let mut batch = OracleBatch::default();
    let shapes: &[&[usize]] = &[
        &[],
        &[0],
        &[1],
        &[4],
        &[5],
        &[2, 4],
        &[2, 3, 4],
        &[2, 0, 3],
        &[2, 3, 0],
    ];
    for dtype in [
        DType::Complex64,
        DType::Complex128,
        DType::F64,
        DType::I64,
        DType::U64,
        DType::Bool,
    ] {
        for &shape in shapes {
            let array = array(shape, dtype);
            let input = batch.input(&array);
            let rank = shape.len() as isize;
            let mut axes = vec![-rank - 1, -rank, -1, 0, rank - 1, rank];
            axes.sort_unstable();
            axes.dedup();
            for axis in axes {
                batch.case(
                    &[input],
                    "unstack",
                    json!({"axis": axis}),
                    array.unstack(axis),
                );
                for sections in [0, 1, 2, 3, 6] {
                    let args = json!({"sections": sections, "axis": axis});
                    batch.case(&[input], "split", args.clone(), array.split(sections, axis));
                    batch.case(
                        &[input],
                        "array_split",
                        args,
                        array.array_split(sections, axis),
                    );
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn explicit_partition_boundaries_and_rank_wrappers_match_numpy() {
    let mut batch = OracleBatch::default();
    let shapes: &[&[usize]] = &[&[], &[0], &[5], &[2, 4], &[2, 3, 4], &[2, 0, 3], &[0, 4]];
    let boundaries: &[&[usize]] = &[&[], &[0], &[1, 1], &[3, 1, 4], &[8, 2, 0], &[usize::MAX]];
    for dtype in [
        DType::Complex64,
        DType::Complex128,
        DType::F32,
        DType::I64,
        DType::U64,
    ] {
        for &shape in shapes {
            let array = array(shape, dtype);
            let input = batch.input(&array);
            for axis in -(shape.len() as isize)..=shape.len() as isize {
                for &indices in boundaries {
                    let args = json!({"indices": indices, "axis": axis});
                    batch.case(
                        &[input],
                        "split_at",
                        args.clone(),
                        array.split_at(indices, axis),
                    );
                    if axis >= 0 {
                        batch.case(
                            &[input],
                            "array_split_at_indices",
                            args,
                            array.array_split_at_indices(indices, axis as usize),
                        );
                    }
                }
            }
            for sections in [0, 1, 2, 3] {
                let args = json!({"sections": sections});
                batch.case(&[input], "vsplit", args.clone(), array.vsplit(sections));
                batch.case(&[input], "hsplit", args.clone(), array.hsplit(sections));
                batch.case(&[input], "dsplit", args, array.dsplit(sections));
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn joint_broadcasting_uses_logical_shapes_without_dtype_promotion() {
    let mut batch = OracleBatch::default();
    batch.case(
        &[],
        "broadcast_arrays",
        json!({}),
        UFuncArray::broadcast_arrays(&[]),
    );
    let shapes: &[&[&[usize]]] = &[
        &[&[]],
        &[&[3]],
        &[&[], &[]],
        &[&[], &[3]],
        &[&[3], &[2, 1]],
        &[&[1, 3], &[2, 1], &[1, 1]],
        &[&[2, 1, 4], &[1, 3, 1], &[4]],
        &[&[0], &[1]],
        &[&[2, 0, 1], &[1, 3]],
        &[&[2, 0, 3], &[1, 3]],
        &[&[2, 3], &[4]],
        &[&[0, 3], &[2, 1]],
        &[&[0, 1], &[1, 0]],
    ];
    for dtypes in [
        [DType::Complex64, DType::F64, DType::I64],
        [DType::F64, DType::Complex128, DType::U64],
        [DType::Complex64, DType::Complex128, DType::F32],
        [DType::I64, DType::Complex64, DType::U64],
        [DType::U64, DType::F32, DType::Complex128],
        [DType::Bool, DType::Complex128, DType::I64],
        [DType::F16, DType::F32, DType::F64],
    ] {
        for &shapes in shapes {
            let arrays: Vec<_> = shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| array(shape, dtypes[index]))
                .collect();
            let inputs: Vec<_> = arrays.iter().map(|array| batch.input(array)).collect();
            let refs: Vec<_> = arrays.iter().collect();
            batch.case(
                &inputs,
                "broadcast_arrays",
                json!({}),
                UFuncArray::broadcast_arrays(&refs),
            );
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn partitions_preserve_complex_payload_bits_and_empty_shapes() {
    let payloads = [
        f64::from_bits(0x7ff8_1234_5678_9abc),
        -0.0,
        0.0,
        f64::from_bits(0xfff8_9876_5432_1001),
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let array = UFuncArray::new(vec![3, 2], payloads.to_vec(), DType::Complex128).unwrap();
    let parts = array.unstack(0).unwrap();
    for (index, part) in parts.iter().enumerate() {
        assert_eq!(part.shape(), &[2]);
        assert_eq!(
            part.values()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            payloads[2 * index..2 * index + 2]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
    let parts = array.split_at(&[2, 1], 0).unwrap();
    for (part, cells) in parts
        .iter()
        .zip([&payloads[..4], &payloads[0..0], &payloads[2..]])
    {
        assert_eq!(
            part.values()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            cells
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
    let empty = UFuncArray::new(
        vec![0, usize::MAX, usize::MAX, 2],
        vec![],
        DType::Complex128,
    )
    .unwrap();
    assert!(empty.unstack(0).unwrap().is_empty());
    for part in empty.split(2, 0).unwrap() {
        assert_eq!(part.shape(), empty.shape());
        assert!(part.values().is_empty());
    }
    let parts = empty.array_split(3, 1).unwrap();
    assert_eq!(parts.len(), 3);
    for part in parts {
        assert_eq!(part.shape(), &[0, usize::MAX / 3, usize::MAX, 2]);
        assert!(part.values().is_empty());
    }
    let scalar = UFuncArray::scalar(3.0, DType::F64);
    let broadcast = UFuncArray::broadcast_arrays(&[&empty, &scalar]).unwrap();
    assert_eq!(broadcast[0].shape(), empty.shape());
    assert_eq!(broadcast[1].shape(), &[0, usize::MAX, usize::MAX]);
    assert!(broadcast.iter().all(|array| array.values().is_empty()));
}

#[test]
fn partitions_reject_invalid_layouts_axes_and_output_capacities() {
    for array in [
        UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::Complex64).unwrap(),
        UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::Complex128).unwrap(),
    ] {
        assert!(array.split(1, 0).is_err());
        assert!(array.array_split(1, 0).is_err());
        assert!(array.split_at(&[], 0).is_err());
        assert!(array.array_split_at_indices(&[], 0).is_err());
        assert!(array.unstack(0).is_err());
        assert!(array.hsplit(1).is_err());
        assert!(array.vsplit(1).is_err());
        assert!(array.dsplit(1).is_err());
        assert!(UFuncArray::broadcast_arrays(&[&array]).is_err());
    }
    let array = array(&[2, 3], DType::Complex128);
    assert!(array.array_split_at_indices(&[], 2).is_err());
    assert!(array.array_split_at_indices(&[], usize::MAX).is_err());
    let empty = UFuncArray::new(vec![0], vec![], DType::F64).unwrap();
    assert!(empty.split(usize::MAX, 0).is_err());
    assert!(empty.array_split(usize::MAX, 0).is_err());
    let empty = UFuncArray::new(vec![0, usize::MAX, 2], vec![], DType::Complex128).unwrap();
    assert!(empty.unstack(1).is_err());
}
