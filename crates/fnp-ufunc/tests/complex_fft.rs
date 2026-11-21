//! Typed complex full Fourier transforms over logical axes, with NumPy as oracle.

use fnp_dtype::DType;
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
use std::process::{Command, Stdio};

fn transform(
    operation: &str,
    input: &UFuncArray,
    n: Option<usize>,
) -> Result<UFuncArray, UFuncError> {
    match operation {
        "fft" => input.fft(n),
        "ifft" => input.ifft(),
        "fft2" => input.fft2(),
        "ifft2" => input.ifft2(),
        "fftn" => input.fftn(),
        "ifftn" => input.ifftn(),
        "fft_ifft" => input.fft(n).and_then(|output| output.ifft()),
        "fft2_ifft2" => input.fft2().and_then(|output| output.ifft2()),
        "fftn_ifftn" => input.fftn().and_then(|output| output.ifftn()),
        _ => panic!("unknown FFT test operation"),
    }
}

fn source(shape: &[usize], dtype: DType) -> UFuncArray {
    let count: usize = shape.iter().product();
    let mut physical_shape = shape.to_vec();
    physical_shape.push(2);
    let values = (0..count)
        .flat_map(|index| {
            [
                ((index * index * 7 + 5) % 43) as f64 / 7.0 - 2.0,
                ((index * 17 + 3) % 29) as f64 / 5.0 - 3.0,
            ]
        })
        .collect();
    UFuncArray::new(physical_shape, values, dtype).unwrap()
}

#[derive(Default)]
struct OracleBatch {
    cases: Vec<Value>,
    actual: Vec<Result<UFuncArray, UFuncError>>,
}

impl OracleBatch {
    fn case(&mut self, operation: &str, input: &UFuncArray, n: Option<usize>) {
        let dtype = match input.dtype() {
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
            _ => panic!("oracle expects typed complex input"),
        };
        let mut shape = input.shape().to_vec();
        assert_eq!(shape.pop(), Some(2));
        self.cases.push(json!({
            "operation": operation,
            "shape": shape,
            "values": input.values().iter().map(f64::to_string).collect::<Vec<_>>(),
            "dtype": dtype,
            "n": n,
        }));
        self.actual.push(transform(operation, input, n));
    }

    fn finish(self) {
        let script = r#"
import json, sys
import numpy as np
rows = []
for case in json.load(sys.stdin):
    try:
        array = np.empty(case['shape'], dtype=case['dtype'])
        parts = np.array([float(value) for value in case['values']]).reshape(-1, 2)
        array.real = parts[:, 0].reshape(case['shape'])
        array.imag = parts[:, 1].reshape(case['shape'])
        operation = case['operation']
        kwargs = {'n': case['n']} if operation.startswith('fft') and case['n'] is not None else {}
        if operation == 'fft_ifft':
            result = np.fft.ifft(np.fft.fft(array, **kwargs))
        elif operation == 'fft2_ifft2':
            result = np.fft.ifft2(np.fft.fft2(array))
        elif operation == 'fftn_ifftn':
            result = np.fft.ifftn(np.fft.fftn(array))
        else:
            result = getattr(np.fft, operation)(array, **kwargs)
        result = np.asarray(result)
        parts = np.empty(result.shape + (2,), dtype=np.float64)
        parts[..., 0] = result.real
        parts[..., 1] = result.imag
        rows.append(dict(shape=list(parts.shape), dtype=str(result.dtype),
                         bits=parts.ravel().view(np.uint64).tolist()))
    except (ValueError, TypeError, IndexError, OverflowError) as error:
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
            .expect("launch live NumPy complex FFT oracle");
        serde_json::to_writer(child.stdin.take().unwrap(), &self.cases).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy complex FFT oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.cases.len());
        for ((case, actual), expected) in self.cases.iter().zip(self.actual).zip(expected) {
            if expected.get("error").is_some() {
                assert!(
                    actual.is_err(),
                    "{case}: expected {expected}, got {actual:?}"
                );
                continue;
            }
            let actual = actual.unwrap_or_else(|error| panic!("{case}: {error}"));
            let dtype = match actual.dtype() {
                DType::Complex64 => "complex64",
                DType::Complex128 => "complex128",
                dtype => panic!("{case}: lost complex output dtype: {dtype:?}"),
            };
            assert_eq!(dtype, expected["dtype"].as_str().unwrap(), "{case}: dtype");
            assert_eq!(json!(actual.shape()), expected["shape"], "{case}: shape");
            let expected: Vec<f64> = expected["bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|bits| f64::from_bits(bits.as_u64().unwrap()))
                .collect();
            assert_eq!(actual.values().len(), expected.len(), "{case}: length");
            let scale = expected
                .iter()
                .fold(1.0_f64, |scale, value| scale.max(value.abs()));
            let tolerance = if dtype == "complex64" {
                3.0e-5
            } else {
                2.0e-11
            } * scale;
            for (index, (&actual, &expected)) in actual.values().iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{case}: component {index}: got {actual}, expected {expected}, tolerance={tolerance}"
                );
                if dtype == "complex64" {
                    assert_eq!(actual, f64::from(actual as f32), "{case}: output precision");
                }
            }
        }
    }
}

#[test]
fn typed_complex_fft_axes_shapes_and_dtypes_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [
            vec![1],
            vec![2],
            vec![3],
            vec![5],
            vec![8],
            vec![13],
            vec![3, 4],
            vec![2, 3, 5],
            vec![2, 1, 7],
        ] {
            let input = source(&shape, dtype);
            for operation in ["fft", "ifft", "fftn", "ifftn"] {
                batch.case(operation, &input, None);
            }
            if shape.len() >= 2 {
                for operation in ["fft2", "ifft2"] {
                    batch.case(operation, &input, None);
                }
            }
        }
    }
    batch.finish();
}

#[test]
fn typed_complex_fft_resizes_last_axis_and_remains_invertible() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [vec![3], vec![8], vec![2, 3], vec![3, 2, 5], vec![2, 0]] {
            let input = source(&shape, dtype);
            for n in [0, 1, 2, 4, 7, 16] {
                batch.case("fft", &input, Some(n));
                batch.case("fft_ifft", &input, Some(n));
            }
        }
    }
    batch.finish();
}

#[test]
fn typed_complex_multidimensional_roundtrips_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [vec![], vec![7], vec![3, 5], vec![2, 3, 4], vec![3, 1, 7]] {
            let input = source(&shape, dtype);
            batch.case("fftn_ifftn", &input, None);
            if !shape.is_empty() {
                batch.case("fft_ifft", &input, None);
            }
            if shape.len() >= 2 {
                batch.case("fft2_ifft2", &input, None);
            }
        }
    }
    batch.finish();
}

#[test]
fn typed_complex_fft_normalizes_source_precision_before_transforming() {
    let values = vec![16_777_217.0, 1.0e-50, -16_777_216.0, -1.0e-50];
    let narrow = UFuncArray::new(vec![2, 2], values.clone(), DType::Complex64).unwrap();
    let wide = UFuncArray::new(vec![2, 2], values, DType::Complex128).unwrap();
    assert_eq!(narrow.fft(None).unwrap().values()[0], 0.0);
    assert_eq!(wide.fft(None).unwrap().values()[0], 1.0);
    let mut batch = OracleBatch::default();
    for input in [&narrow, &wide] {
        batch.case("fft", input, None);
        batch.case("ifft", input, None);
        batch.case("fftn_ifftn", input, None);
    }
    batch.finish();
}

#[test]
fn typed_complex_fft_scalar_empty_batch_and_invalid_shape_contracts() {
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [
            vec![],
            vec![0],
            vec![0, 3],
            vec![2, 0],
            vec![0, 2, 3],
            vec![2, 0, 3],
        ] {
            let input = source(&shape, dtype);
            for operation in ["fft", "ifft", "fft2", "ifft2", "fftn", "ifftn"] {
                batch.case(operation, &input, None);
            }
        }
    }
    batch.finish();

    for dtype in [DType::Complex64, DType::Complex128] {
        let input = source(&[1], dtype);
        for n in [usize::MAX, usize::MAX / 2, isize::MAX as usize / 8] {
            assert!(input.fft(Some(n)).is_err());
        }
        for physical_shape in [vec![], vec![3], vec![4], vec![2, 3]] {
            let count: usize = physical_shape.iter().product();
            let malformed = UFuncArray::new(physical_shape, vec![1.0; count], dtype).unwrap();
            for operation in ["fft", "ifft", "fft2", "ifft2", "fftn", "ifftn"] {
                assert!(transform(operation, &malformed, None).is_err());
            }
        }
        let empty_batch = UFuncArray::new(vec![0, usize::MAX, 4, 2], vec![], dtype).unwrap();
        for operation in ["fft", "ifft"] {
            let output = transform(operation, &empty_batch, None).unwrap();
            assert_eq!(output.shape(), empty_batch.shape());
            assert_eq!(output.dtype(), dtype);
            assert!(output.values().is_empty());
        }
        let overflowing_resize = UFuncArray::new(vec![usize::MAX, 0, 2], vec![], dtype).unwrap();
        assert!(overflowing_resize.fft(Some(2)).is_err());
    }
}

#[test]
fn legacy_real_and_packed_f64_fft_contract_remains_unchanged() {
    let signal = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64).unwrap();
    let full = signal.fft(None).unwrap();
    assert_eq!(full.dtype(), DType::F64);
    assert_eq!(full.shape(), &[4, 2]);
    let recovered = full.ifft().unwrap();
    assert_eq!(recovered.dtype(), DType::F64);
    for (&expected, pair) in signal
        .values()
        .iter()
        .zip(recovered.values().as_chunks::<2>().0)
    {
        assert!((pair[0] - expected).abs() < 1.0e-12);
        assert!(pair[1].abs() < 1.0e-12);
    }
    let matrix = signal.reshape(&[2, 2]).unwrap();
    for operation in ["fft2", "fftn"] {
        let full = transform(operation, &matrix, None).unwrap();
        assert_eq!(full.dtype(), DType::F64);
        assert_eq!(full.shape(), &[2, 2, 2]);
        let inverse = if operation == "fft2" {
            full.ifft2()
        } else {
            full.ifftn()
        }
        .unwrap();
        assert_eq!(inverse.dtype(), DType::F64);
        assert_eq!(inverse.shape(), &[2, 2, 2]);
        for (&expected, pair) in matrix
            .values()
            .iter()
            .zip(inverse.values().as_chunks::<2>().0)
        {
            assert!((pair[0] - expected).abs() < 1.0e-12);
            assert!(pair[1].abs() < 1.0e-12);
        }
    }
}
