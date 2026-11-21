//! Native real/Hermitian Fourier transforms against the live NumPy oracle.
//! The Rust FFT API stores spectra as real/imaginary component pairs on a
//! trailing axis; these tests compare logical spectra and reconstructed data.

use fnp_dtype::DType;
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
use std::process::{Command, Stdio};

#[derive(Default)]
struct OracleBatch {
    cases: Vec<Value>,
    actual: Vec<Result<UFuncArray, UFuncError>>,
}

impl OracleBatch {
    fn case(&mut self, operation: &str, input: &UFuncArray, n: Option<usize>) {
        let result = match operation {
            "rfft" => input.rfft(n),
            "irfft" => input.irfft(n),
            "hfft" => input.hfft(n),
            "ihfft" => input.ihfft(n),
            "rfftn" => input.rfftn(),
            "irfftn" => input.irfftn(n),
            "rfft2" => input.rfft2(),
            "irfft2" => input.irfft2(n),
            _ => panic!("unknown oracle operation"),
        };
        let dtype = match input.dtype() {
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
            _ => panic!("unsupported test dtype"),
        };
        self.cases.push(json!({
            "operation": operation,
            "shape": input.shape(),
            "values": input.values().iter().map(f64::to_string).collect::<Vec<_>>(),
            "dtype": dtype,
            "n": n,
        }));
        self.actual.push(result);
    }

    fn finish(self) {
        let script = r#"
import json, sys, warnings
import numpy as np
warnings.simplefilter('ignore')
rows = []
for case in json.load(sys.stdin):
    try:
        op = case['operation']
        dtype = np.dtype(case['dtype'])
        values = np.array([float(x) for x in case['values']])
        packed = op in ('irfft', 'irfftn', 'irfft2', 'hfft')
        if packed or dtype.kind == 'c':
            parts = values.reshape(case['shape'])
            component_dtype = np.float32 if dtype in (np.float32, np.complex64) else np.float64
            parts = parts.astype(component_dtype)
            array = np.empty(case['shape'][:-1], dtype=(np.complex64 if component_dtype == np.float32 else np.complex128))
            array.real = parts[..., 0]
            array.imag = parts[..., 1]
        else:
            array = values.astype(dtype).reshape(case['shape'])
        kwargs = {}
        if case['n'] is not None:
            if op in ('irfftn', 'irfft2'):
                kwargs['s'] = (*array.shape[:-1], case['n'])
                kwargs['axes'] = tuple(range(array.ndim))
            else:
                kwargs['n'] = case['n']
        result = np.asarray(getattr(np.fft, op)(array, **kwargs))
        if np.iscomplexobj(result):
            parts = np.empty(result.shape + (2,), dtype=np.float64)
            parts[..., 0] = result.real
            parts[..., 1] = result.imag
        else:
            parts = result.astype(np.float64)
        rows.append(dict(shape=list(parts.shape), bits=parts.ravel().view(np.uint64).tolist()))
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
            .expect("launch live NumPy real FFT oracle");
        serde_json::to_writer(child.stdin.take().unwrap(), &self.cases).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy real FFT oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(self.cases.len(), expected.len());
        for ((case, actual), expected) in self.cases.iter().zip(self.actual).zip(expected) {
            if expected.get("error").is_some() {
                assert!(
                    actual.is_err(),
                    "{case}: expected {expected}, got {actual:?}"
                );
                continue;
            }
            let actual = actual.unwrap_or_else(|error| panic!("{case}: {error}"));
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
                .filter(|value| value.is_finite())
                .fold(1.0_f64, |scale, value| scale.max(value.abs()));
            let tolerance = if matches!(case["dtype"].as_str(), Some("float32" | "complex64")) {
                5.0e-6 * scale
            } else {
                2.0e-11 * scale
            };
            for (index, (&actual, &expected)) in actual.values().iter().zip(&expected).enumerate() {
                assert!(
                    (actual == expected)
                        || (actual.is_nan() && expected.is_nan())
                        || (actual - expected).abs() <= tolerance,
                    "{case}: component {index}: got {actual}, expected {expected}, tolerance={tolerance}"
                );
            }
        }
    }
}

fn real_input(shape: &[usize]) -> UFuncArray {
    let count: usize = shape.iter().product();
    UFuncArray::new(
        shape.to_vec(),
        (0..count)
            .map(|i| ((i * i * 17 + i * 23 + 5) % 101) as f64 / 7.0 - 6.0)
            .collect(),
        DType::F64,
    )
    .unwrap()
}

fn spectrum(shape: &[usize]) -> UFuncArray {
    let count: usize = shape.iter().product();
    let mut packed_shape = shape.to_vec();
    packed_shape.push(2);
    UFuncArray::new(
        packed_shape,
        (0..count)
            .flat_map(|i| {
                [
                    ((i * 17 + 3) % 47) as f64 / 5.0 - 4.0,
                    ((i * i * 13 + 11) % 41) as f64 / 3.0 - 5.0,
                ]
            })
            .collect(),
        DType::F64,
    )
    .unwrap()
}

#[test]
fn multidimensional_real_fft_roundtrips_general_signals() {
    // A size-2 outer axis is its own negative frequency, which masks the
    // former incorrect conjugation within the same outer frequency row.
    for shape in [
        vec![3, 3],
        vec![4, 7],
        vec![5, 6],
        vec![3, 4, 5],
        vec![5, 3, 8],
        vec![3, 1, 7],
    ] {
        let input = real_input(&shape);
        let transformed = input.rfftn().unwrap();
        let recovered = transformed.irfftn(shape.last().copied()).unwrap();
        assert_eq!(recovered.shape(), input.shape());
        for (index, (&actual, &expected)) in
            recovered.values().iter().zip(input.values()).enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-10,
                "shape={shape:?}, index={index}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn inverse_real_and_hermitian_transforms_match_numpy_for_resized_spectra() {
    let mut batch = OracleBatch::default();
    for m in [0, 1, 2, 3, 5, 8, 11] {
        let input = spectrum(&[m]);
        for n in [
            None,
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(5),
            Some(6),
            Some(13),
            Some(16),
        ] {
            batch.case("irfft", &input, n);
            batch.case("hfft", &input, n);
        }
    }
    batch.finish();
}

#[test]
fn inverse_multidimensional_real_spectra_match_numpy() {
    let mut batch = OracleBatch::default();
    for shape in [
        vec![3, 3],
        vec![4, 5],
        vec![5, 2],
        vec![3, 4, 5],
        vec![5, 3, 4],
        vec![3, 0],
        vec![2, 1],
    ] {
        let input = spectrum(&shape);
        for n in [None, Some(0), Some(1), Some(2), Some(3), Some(6), Some(9)] {
            batch.case("irfftn", &input, n);
            if shape.len() == 2 {
                batch.case("irfft2", &input, n);
            }
        }
    }
    batch.finish();
}

#[test]
fn inverse_real_fft_ignores_imaginary_dc_and_even_nyquist() {
    let mut batch = OracleBatch::default();
    for n in [1, 2, 3, 5, 6, 8, 11] {
        let m = n / 2 + 1;
        for ignored in [5.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut values = spectrum(&[m]).values().to_vec();
            values[1] = ignored;
            if n % 2 == 0 {
                values[2 * m - 1] = ignored;
            }
            let input = UFuncArray::new(vec![m, 2], values, DType::F64).unwrap();
            batch.case("irfft", &input, Some(n));
            batch.case("hfft", &input, Some(n));
        }
    }
    batch.finish();
}

#[test]
fn real_fft_forward_families_and_inverse_hermitian_match_numpy() {
    let mut batch = OracleBatch::default();
    for length in [0, 1, 2, 3, 5, 8, 13, 32] {
        let input = real_input(&[length]);
        for n in [None, Some(0), Some(1), Some(3), Some(6), Some(13)] {
            batch.case("rfft", &input, n);
            batch.case("ihfft", &input, n);
        }
    }
    for shape in [vec![3, 5], vec![4, 6], vec![3, 5, 7]] {
        let input = real_input(&shape);
        batch.case("rfftn", &input, None);
        if shape.len() == 2 {
            batch.case("rfft2", &input, None);
        }
    }
    batch.finish();
}

#[test]
fn real_fft_shapes_fail_closed_without_overflow_or_panics() {
    let mut batch = OracleBatch::default();
    for shape in [vec![], vec![0], vec![0, 3], vec![3, 0], vec![2, 0, 4]] {
        let input = real_input(&shape);
        batch.case("rfftn", &input, None);
        if shape.len() == 2 {
            batch.case("rfft2", &input, None);
        }
    }
    for shape in [vec![0], vec![1], vec![0, 3], vec![3, 0], vec![2, 0, 4]] {
        let input = spectrum(&shape);
        batch.case("irfftn", &input, None);
        batch.case("irfftn", &input, Some(4));
    }
    batch.finish();

    let signal = real_input(&[1]);
    let half = spectrum(&[1]);
    for n in [usize::MAX, usize::MAX / 2, isize::MAX as usize / 8] {
        assert!(signal.rfft(Some(n)).is_err());
        assert!(half.irfft(Some(n)).is_err());
        assert!(half.irfftn(Some(n)).is_err());
        assert!(half.hfft(Some(n)).is_err());
    }
    let huge_empty = UFuncArray::new(vec![usize::MAX, 0, 2], vec![], DType::F64).unwrap();
    assert!(huge_empty.irfftn(Some(2)).is_err());
}

#[test]
fn real_fft_honors_source_precision_and_rejects_complex_forward_input() {
    let mut batch = OracleBatch::default();
    let signal = UFuncArray::new(
        vec![4],
        vec![16_777_217.0, -16_777_216.0, 1.0e-50, -1.0e-50],
        DType::F32,
    )
    .unwrap();
    // These cancellations would leave 1.0 (or 0.25) if the unrounded f64
    // working values leaked into an FFT declared to have float32 precision.
    assert_eq!(signal.rfft(None).unwrap().values()[0], 0.0);
    batch.case("rfft", &signal, None);
    batch.case("rfftn", &signal, None);
    let typed = UFuncArray::new(
        vec![3, 2],
        vec![16_777_217.0, 2.0, -8_388_608.0, 1.0e-50, 0.0, -1.0e-50],
        DType::Complex64,
    )
    .unwrap();
    assert_eq!(typed.irfft(Some(4)).unwrap().values()[0], 0.0);
    batch.case("irfft", &typed, Some(3));
    batch.case("irfftn", &typed, Some(5));
    batch.case("rfft", &typed, None);
    batch.case("rfftn", &typed, None);
    batch.finish();
}
