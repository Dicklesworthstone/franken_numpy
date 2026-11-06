//! Logical complex-element accumulation, including NumPy's first-element rule.

use fnp_dtype::DType;
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
use std::process::{Command, Stdio};

fn complex_array(shape: &[usize], values: &[(f64, f64)], dtype: DType) -> UFuncArray {
    let mut physical_shape = shape.to_vec();
    physical_shape.push(2);
    UFuncArray::new(
        physical_shape,
        values.iter().flat_map(|&(re, im)| [re, im]).collect(),
        dtype,
    )
    .expect("valid complex array")
}

fn accumulate(
    array: &UFuncArray,
    operation: &str,
    axis: Option<isize>,
) -> Result<UFuncArray, UFuncError> {
    match operation {
        "cumsum" => array.cumsum(axis),
        "cumprod" => array.cumprod(axis),
        "nancumsum" => array.nancumsum(axis),
        "nancumprod" => array.nancumprod(axis),
        _ => unreachable!("test cumulative operation"),
    }
}

#[derive(Default)]
struct OracleBatch {
    inputs: Vec<Value>,
    cases: Vec<Value>,
    actual: Vec<Result<UFuncArray, UFuncError>>,
}

impl OracleBatch {
    fn add(&mut self, array: &UFuncArray, axes: &[Option<isize>]) {
        let input = self.inputs.len();
        let dtype = match array.dtype() {
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
            _ => panic!("test requires complex inputs"),
        };
        self.inputs.push(json!({
            "shape": &array.shape()[..array.shape().len() - 1],
            "dtype": dtype,
            "values": array.values().as_chunks::<2>().0.iter()
                .map(|parts| [parts[0].to_string(), parts[1].to_string()])
                .collect::<Vec<_>>(),
        }));
        for operation in ["cumsum", "cumprod", "nancumsum", "nancumprod"] {
            for &axis in axes {
                self.cases
                    .push(json!({"input": input, "op": operation, "axis": axis}));
                self.actual.push(accumulate(array, operation, axis));
            }
        }
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np

warnings.simplefilter('ignore')
payload = json.load(sys.stdin)
inputs = []
for item in payload['inputs']:
    a = np.empty(len(item['values']), dtype=item['dtype'])
    a.real = [float(pair[0]) for pair in item['values']]
    a.imag = [float(pair[1]) for pair in item['values']]
    inputs.append(a.reshape(item['shape']))
rows = []
for case in payload['cases']:
    try:
        result = getattr(np, case['op'])(inputs[case['input']], axis=case['axis'])
        parts = np.empty(result.size * 2, dtype=np.float64)
        parts[0::2] = result.real.ravel()
        parts[1::2] = result.imag.ravel()
        rows.append(dict(shape=list(result.shape), dtype=str(result.dtype),
                         bits=parts.view(np.uint64).tolist()))
    except (TypeError, ValueError, IndexError) as error:
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
            .expect("launch live NumPy cumulative oracle");
        let mut stdin = child.stdin.take().expect("oracle stdin");
        serde_json::to_writer(
            &mut stdin,
            &json!({"inputs": self.inputs, "cases": self.cases}),
        )
        .expect("write oracle inputs");
        drop(stdin);
        let output = child.wait_with_output().expect("collect oracle output");
        assert!(
            output.status.success(),
            "NumPy cumulative oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
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
            assert_eq!(shape.pop(), Some(2), "{case}: missing interleaved lane");
            assert_eq!(json!(shape), expected["shape"], "{case}: shape");
            let dtype = if actual.dtype() == DType::Complex64 {
                "complex64"
            } else {
                assert_eq!(actual.dtype(), DType::Complex128, "{case}: dtype");
                "complex128"
            };
            assert_eq!(dtype, expected["dtype"], "{case}: dtype");
            let bits = expected["bits"].as_array().expect("result bits");
            assert_eq!(actual.values().len(), bits.len(), "{case}: length");
            for (index, (&actual, bits)) in actual.values().iter().zip(bits).enumerate() {
                let expected = f64::from_bits(bits.as_u64().expect("float bits"));
                if actual.is_nan() && expected.is_nan() {
                    continue;
                }
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{case}: component {index}, actual={actual:?}, expected={expected:?}"
                );
            }
        }
    }
}

#[test]
fn complex_cumulative_arithmetic_preserves_logical_elements() {
    let a = complex_array(&[2], &[(1.0, 2.0), (3.0, 4.0)], DType::Complex128);
    let sum = a.cumsum(None).unwrap();
    assert_eq!(sum.shape(), &[2, 2]);
    assert_eq!(sum.values(), &[1.0, 2.0, 4.0, 6.0]);
    let product = a.cumprod(Some(-1)).unwrap();
    assert_eq!(product.shape(), &[2, 2]);
    assert_eq!(product.values(), &[1.0, 2.0, -5.0, 10.0]);
    assert_eq!(a.cumulative_sum(None).unwrap().values(), sum.values());
    assert_eq!(
        a.cumulative_prod(Some(-1)).unwrap().values(),
        product.values()
    );
    for operation in ["cumsum", "cumprod", "nancumsum", "nancumprod"] {
        assert!(accumulate(&a, operation, Some(1)).is_err());
        assert!(accumulate(&a, operation, Some(-2)).is_err());
        let scalar =
            UFuncArray::new(vec![2], vec![1.0, 2.0], DType::Complex128).expect("logical scalar");
        assert_eq!(
            accumulate(&scalar, operation, Some(0)).unwrap().shape(),
            &[1, 2]
        );
        let malformed = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::Complex128)
            .expect("array construction validates physical element count");
        assert!(accumulate(&malformed, operation, None).is_err());
    }
}

#[test]
fn complex_cumulative_shapes_axes_and_source_precision_match_numpy() {
    let mut oracle = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for shape in [
            vec![],
            vec![0],
            vec![7],
            vec![2, 3],
            vec![2, 3, 4],
            vec![2, 0, 3],
        ] {
            let count: usize = shape.iter().product();
            let values: Vec<_> = (0..count)
                .map(|index| {
                    let x = index as f64;
                    (0.75 + (x % 5.0) * 0.123456789, -0.25 + (x % 7.0) * 0.137)
                })
                .collect();
            let array = complex_array(&shape, &values, dtype);
            let mut axes = vec![None];
            axes.extend((-(shape.len() as isize) - 1..=shape.len() as isize).map(Some));
            oracle.add(&array, &axes);
        }
        let cancellation = complex_array(
            &[5],
            &[
                (16_777_217.0, -16_777_217.0),
                (1.0, -1.0),
                (-16_777_216.0, 16_777_216.0),
                (1.0e-20, -1.0e-20),
                (-0.0, 0.0),
            ],
            dtype,
        );
        oracle.add(&cancellation, &[None, Some(0), Some(-1)]);
    }
    oracle.assert_matches_numpy();
}

#[test]
fn complex_cumulative_products_match_numpy_at_two_element_dispatch_boundary() {
    let mut oracle = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for len in [1, 2, 3, 4, 7, 16] {
            // Length two can use the fused SIMD multiply; lengths above two
            // overlap accumulator/output and use non-fused scalar arithmetic.
            // Exercise both contiguous and strided axes, including outer lanes.
            for shape in [vec![len], vec![2, len], vec![len, 3], vec![2, len, 3]] {
                let count: usize = shape.iter().product();
                let pattern = [
                    (0.873456789, -0.113),
                    (1.243827156, 0.298),
                    (-0.1250000001, 0.37),
                    (0.333333333333, -0.2),
                    (0.75, 1.25),
                ];
                let values: Vec<_> = (0..count).map(|i| pattern[i % pattern.len()]).collect();
                let array = complex_array(&shape, &values, dtype);
                let mut axes = vec![None];
                axes.extend((0..shape.len() as isize).map(Some));
                oracle.add(&array, &axes);
            }
        }
    }
    oracle.assert_matches_numpy();
}

#[test]
fn complex_cumulative_nan_components_infinities_and_signed_zero_match_numpy() {
    let mut oracle = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for values in [
            vec![(-0.0, -0.0), (0.0, -0.0), (-0.0, 0.0)],
            vec![(f64::INFINITY, 3.0), (2.0, 1.0), (1.0, -1.0)],
            vec![(2.0, f64::INFINITY), (2.0, 1.0), (1.0, -1.0)],
            vec![(f64::NAN, 4.0), (2.0, 1.0), (3.0, f64::NAN)],
            vec![(3.0, f64::NAN), (2.0, 1.0), (f64::NAN, 4.0)],
            vec![(f64::NAN, f64::NAN), (f64::NAN, 2.0), (3.0, f64::NAN)],
            vec![(-0.0, -0.0), (0.0, -0.0)],
            vec![(f64::INFINITY, 3.0), (2.0, 1.0)],
            vec![(2.0, f64::INFINITY), (2.0, 1.0)],
            vec![(f64::NAN, 4.0), (2.0, 1.0)],
            vec![(2.0, 1.0), (3.0, f64::NAN)],
        ] {
            oracle.add(
                &complex_array(&[values.len()], &values, dtype),
                &[None, Some(0), Some(-1)],
            );
            // Put each value first in a distinct lane to exercise direct copies.
            oracle.add(
                &complex_array(&[1, values.len()], &values, dtype),
                &[Some(0)],
            );
            for &value in &values {
                oracle.add(
                    &complex_array(&[], &[value], dtype),
                    &[None, Some(0), Some(-1)],
                );
            }
        }
    }
    oracle.assert_matches_numpy();
}
