//! Typed, batched pseudoinverse contracts and live NumPy comparisons.
//!
//! Portable decompositions and matrix products need not choose the same
//! singular-vector phases or accumulation order as a vendor LAPACK/BLAS pair.
//! These tests compare pseudoinverse values and all four Moore-Penrose identities.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::UFuncArray;
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
        _ => panic!("unsupported oracle dtype {dtype:?}"),
    }
}

fn array(shape: &[usize], values: &[(f64, f64)], dtype: DType) -> UFuncArray {
    let mut shape = shape.to_vec();
    let values = if dtype.is_complex() {
        shape.push(2);
        values.iter().flat_map(|&(re, im)| [re, im]).collect()
    } else {
        values.iter().map(|&(re, _)| re).collect()
    };
    UFuncArray::new(shape, values, dtype).unwrap()
}

fn record(array: &UFuncArray) -> Value {
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
    cases: Vec<Value>,
}

impl OracleBatch {
    fn case(&mut self, input: &UFuncArray, cutoff: Option<f64>, hermitian: bool, penrose: bool) {
        let result = if let Some(cutoff) = cutoff {
            input.pinv_with_options(cutoff, hermitian)
        } else {
            assert!(!hermitian);
            input.pinv()
        };
        let actual = match result {
            Ok(output) => {
                if matches!(output.dtype(), DType::F32 | DType::Complex64) {
                    for &value in output.values() {
                        assert!(
                            value.is_nan() || value == f64::from(value as f32),
                            "single-precision output contains an unrounded cell {value}"
                        );
                    }
                }
                record(&output)
            }
            Err(error) => json!({"error": error.to_string()}),
        };
        self.cases.push(json!({
            "input": record(input), "actual": actual,
            "cutoff": cutoff.map(|value| value.to_string()),
            "hermitian": hermitian, "penrose": penrose,
        }));
    }

    fn finish(self) {
        let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
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

def adjoint(a):
    return a.swapaxes(-1, -2).conj()

cases = json.load(sys.stdin)
for index, case in enumerate(cases):
    source = load(case['input'])
    options = {'hermitian': case['hermitian']}
    if case['cutoff'] is not None:
        options['rcond'] = float(case['cutoff'])
    try:
        expected = np.linalg.pinv(source, **options)
    except Exception as error:
        assert 'error' in case['actual'], (index, options, source, type(error).__name__)
        continue
    assert 'error' not in case['actual'], (index, options, source, case['actual'])
    actual = load(case['actual'])
    assert actual.shape == expected.shape, (index, actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (index, actual.dtype, expected.dtype)
    tolerance = 3e-5 if actual.dtype in (np.dtype('float32'), np.dtype('complex64')) else 3e-10
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance,
                               equal_nan=True, err_msg=f'case {index}: {options}, {source}')
    if case['penrose'] and source.size:
        source = source.astype(actual.dtype)
        if case['hermitian']:
            lower = np.tril(source, -1)
            selected = lower + adjoint(lower)
            diagonal = np.arange(source.shape[-1])
            selected[..., diagonal, diagonal] = source[..., diagonal, diagonal].real
            source = selected
        left = source @ actual
        right = actual @ source
        for found, wanted in ((left @ source, source), (right @ actual, actual),
                              (left, adjoint(left)), (right, adjoint(right))):
            np.testing.assert_allclose(found, wanted, rtol=tolerance*10,
                                       atol=tolerance*10, err_msg=f'Moore-Penrose case {index}')
print(f'NumPy {np.__version__}: {len(cases)} pseudoinverse cases passed')
"#;
        let mut child = Command::new(python)
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("NumPy oracle interpreter must be available");
        child
            .stdin
            .take()
            .unwrap()
            .write_all(serde_json::to_string(&self.cases).unwrap().as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy pseudoinverse oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        eprintln!("{}", String::from_utf8_lossy(&output.stdout));
    }
}

#[test]
fn pseudoinverse_matches_numpy_for_typed_rectangles_and_multidimensional_stacks() {
    let mut batch = OracleBatch::default();
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I32,
        DType::I64,
        DType::U64,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for shape in [
            vec![2, 2],
            vec![4, 2],
            vec![2, 4],
            vec![2, 3, 2],
            vec![1, 2, 2, 3],
            vec![2, 1, 3, 3],
        ] {
            let (m, n) = (shape[shape.len() - 2], shape[shape.len() - 1]);
            let values: Vec<_> = (0..shape.iter().product::<usize>())
                .map(|index| {
                    let (row, column) = ((index / n) % m, index % n);
                    let diagonal = if row == column { 7.0 } else { 0.0 };
                    let real = if dtype == DType::Bool {
                        if (index + row).is_multiple_of(3) {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        ((index * 3 + row) % 5) as f64 + diagonal
                    };
                    let imag = ((index * 5 + column) % 7) as f64 / 4.0 - 0.5;
                    (real, imag)
                })
                .collect();
            let input = array(&shape, &values, dtype);
            batch.case(&input, None, false, true);
            batch.case(&input, Some(0.2), false, false);
            batch.case(&input, Some(2.0), false, false);
        }
    }
    batch.finish();
}

#[test]
fn pseudoinverse_handles_rank_deficiency_zero_matrices_and_repeated_spectra() {
    let mut batch = OracleBatch::default();
    let fixtures = [
        (
            vec![3, 2],
            vec![
                (1.0, 2.0),
                (1.0, 2.0),
                (2.0, -1.0),
                (2.0, -1.0),
                (3.0, 0.5),
                (3.0, 0.5),
            ],
        ),
        (vec![2, 3], vec![(0.0, 0.0); 6]),
        (
            vec![2, 2, 2],
            vec![
                (3.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (3.0, 0.0),
                (0.0, 2.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, -2.0),
            ],
        ),
    ];
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        for (shape, values) in &fixtures {
            let input = array(shape, values, dtype);
            batch.case(&input, None, false, true);
        }
    }
    batch.finish();
}

#[test]
fn hermitian_pseudoinverse_preserves_signed_eigenvalues_and_ignores_unused_entries() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        let input = array(
            &[2, 3, 3],
            &[
                (4.0, f64::NAN),
                (f64::INFINITY, f64::NAN),
                (f64::NAN, -999.0),
                (1.0, -1.0),
                (-2.0, f64::INFINITY),
                (-777.0, f64::NAN),
                (0.5, 0.0),
                (1.0, 0.5),
                (5.0, -33.0),
                (-3.0, 100.0),
                (f64::NAN, f64::INFINITY),
                (f64::NAN, f64::NAN),
                (0.0, 0.5),
                (2.0, -1.0),
                (f64::INFINITY, 17.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, f64::NAN),
            ],
            dtype,
        );
        batch.case(&input, Some(1.0e-15), true, true);
        batch.case(&input, Some(0.5), true, false);
        batch.case(&input, Some(f64::NAN), true, false);
        batch.case(&input, Some(f64::INFINITY), true, false);
    }
    batch.finish();
}

#[test]
fn pseudoinverse_cutoffs_are_literal_and_default_is_one_e_minus_fifteen() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        let diagonal = array(
            &[3, 3],
            &[
                (4.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (-1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ],
            dtype,
        );
        for hermitian in [false, true] {
            for cutoff in [
                -1.0,
                -0.0,
                0.25_f64.next_down(),
                0.25,
                0.25_f64.next_up(),
                1.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ] {
                batch.case(&diagonal, Some(cutoff), hermitian, false);
            }
        }
    }
    for n in [2, 8] {
        let mut entries = vec![(0.0, 0.0); n * n];
        entries[0] = (1.0, 0.0);
        entries[n + 1] = (7.0e-16, 0.0);
        for dtype in [DType::F64, DType::Complex128] {
            let input = array(&[n, n], &entries, dtype);
            let actual = input.pinv().unwrap();
            assert_eq!(actual.values()[0], 1.0);
            assert_eq!(
                actual.values()[(n + 1) * if dtype.is_complex() { 2 } else { 1 }],
                0.0
            );
            batch.case(&input, None, false, false);
        }
    }
    batch.finish();
}

#[test]
fn pseudoinverse_normalizes_declared_precision_and_exact_integer_sources() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::F32, DType::Complex64] {
        let input = array(
            &[2, 2],
            &[
                (16_777_217.0, 1.0e-50),
                (16_777_216.0, 0.0),
                (16_777_216.0, 0.0),
                (16_777_217.0, -1.0e-50),
            ],
            dtype,
        );
        let expected = 1.0 / 67_108_864.0;
        let actual = input.pinv().unwrap();
        for pair in actual
            .values()
            .chunks(if dtype.is_complex() { 2 } else { 1 })
        {
            assert!((pair[0] - expected).abs() <= expected * 2.0e-6);
        }
        batch.case(&input, None, false, true);
        batch.case(&input, Some(1.0e-15), true, true);
    }
    let wide = UFuncArray::from_storage(
        vec![2, 2],
        ArrayStorage::U64(vec![(1_u64 << 53) + 1, 0, 0, (1_u64 << 53) + 3]),
    )
    .unwrap();
    batch.case(&wide, Some(0.0), false, true);
    batch.case(&wide, Some(0.0), true, true);
    let booleans = UFuncArray::new(vec![2, 2], vec![2.0, 0.0, 0.0, -3.0], DType::Bool).unwrap();
    assert_eq!(booleans.pinv().unwrap().values(), &[1.0, 0.0, 0.0, 1.0]);
    batch.case(&booleans, None, false, true);
    batch.finish();
}

#[test]
fn pseudoinverse_distinguishes_empty_cores_from_empty_batches() {
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
        for shape in [
            vec![0, 3],
            vec![3, 0],
            vec![0, 0],
            vec![2, 0, 3],
            vec![2, 3, 0],
            vec![0, 2, 3],
            vec![1, 0, 2, 2],
        ] {
            let input = array(&shape, &[], dtype);
            for hermitian in [false, true] {
                batch.case(&input, Some(1.0e-15), hermitian, false);
            }
        }
    }
    batch.finish();

    for dtype in [DType::F32, DType::Complex128, DType::I64] {
        for shape in [vec![usize::MAX, 0, 3], vec![0, usize::MAX, usize::MAX]] {
            let input = array(&shape, &[], dtype);
            let output = input.pinv().unwrap();
            let mut expected = shape.clone();
            expected.swap(1, 2);
            if dtype.is_complex() {
                expected.push(2);
            }
            assert_eq!(output.shape(), expected);
            assert!(output.values().is_empty());
        }
    }
}

#[test]
fn pseudoinverse_rejects_invalid_shapes_dtypes_layouts_and_selected_nonfinite_values() {
    let mut batch = OracleBatch::default();
    for dtype in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for shape in [vec![], vec![2], vec![2, 3]] {
            let values = vec![(1.0, 0.0); shape.iter().product()];
            let input = array(&shape, &values, dtype);
            batch.case(&input, Some(1.0e-15), true, false);
            if shape.len() < 2 {
                batch.case(&input, None, false, false);
            }
        }
        let unsupported = array(&[2, 2], &[(1.0, 0.0); 4], dtype);
        if dtype == DType::F16 {
            batch.case(&unsupported, None, false, false);
        }
    }
    batch.finish();

    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [vec![2, 2], vec![2, 3], vec![2, 2, 3]] {
            let invalid =
                UFuncArray::new(shape.clone(), vec![1.0; shape.iter().product()], dtype).unwrap();
            assert!(invalid.pinv().is_err());
            assert!(invalid.pinv_with_options(0.0, true).is_err());
        }
    }
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let invalid = array(
                &[2, 2],
                &[(1.0, 0.0), (0.0, 0.0), (value, 0.0), (2.0, 0.0)],
                dtype,
            );
            assert!(invalid.pinv().is_err());
            assert!(invalid.pinv_with_options(f64::NAN, true).is_err());
        }
    }
}
