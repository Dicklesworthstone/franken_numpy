//! Logical complex Einstein contractions and contraction-order planning.
//!
//! Relocations are checked bit-for-bit. Arithmetic uses componentwise forward
//! error bounds because optimized NumPy may select a vendor BLAS reduction.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ndarray::ShapeError;
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
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
        _ => panic!("non-numeric oracle dtype"),
    }
}

fn source(shape: &[usize], dtype: DType, seed: usize) -> UFuncArray {
    let count: usize = shape.iter().product();
    if dtype.is_complex() {
        let mut physical_shape = shape.to_vec();
        physical_shape.push(2);
        let values = (0..count)
            .flat_map(|index| {
                let real = match (index + seed) % 13 {
                    0 => 1.0 + 2.0_f64.powi(-24),
                    1 => 1.0e-50,
                    _ => ((index * 3 + seed) % 17) as f64 / 4.0 - 2.0,
                };
                let imag = ((index * 5 + seed) % 19) as f64 / 4.0 - 2.25;
                [real, imag]
            })
            .collect();
        UFuncArray::new(physical_shape, values, dtype).unwrap()
    } else {
        let values = (0..count)
            .map(|index| ((index * 3 + seed) % 17) as f64 / 4.0 - 2.0)
            .collect();
        UFuncArray::from_storage(
            shape.to_vec(),
            ArrayStorage::F64(values).cast_to(dtype).unwrap(),
        )
        .unwrap()
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
    fn case(&mut self, sub: &str, operands: &[UFuncArray], optimize: Option<&str>, exact: bool) {
        self.case_with_limit(sub, operands, optimize, None, exact);
    }

    fn case_with_limit(
        &mut self,
        sub: &str,
        operands: &[UFuncArray],
        optimize: Option<&str>,
        memory_limit: Option<usize>,
        exact: bool,
    ) {
        let refs: Vec<_> = operands.iter().collect();
        self.actual.push(match optimize {
            Some(strategy) => {
                UFuncArray::einsum_optimized_with_memory_limit(sub, &refs, strategy, memory_limit)
            }
            None => UFuncArray::einsum(sub, &refs),
        });
        self.cases.push(json!({
            "sub": sub,
            "operands": operands.iter().map(oracle_input).collect::<Vec<_>>(),
            "optimize": optimize,
            "memory_limit": memory_limit,
            "exact": exact,
        }));
    }

    fn finish(self) {
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

rows = []
for case in json.load(sys.stdin):
    try:
        arrays = [load(item) for item in case['operands']]
        optimize = case['optimize'] or False
        if optimize == 'auto':
            optimize = 'greedy'
        if case['memory_limit'] is not None:
            optimize = (optimize, case['memory_limit'])
        result = np.asarray(np.einsum(case['sub'], *arrays, optimize=optimize))
        parts = np.empty(result.size * 2, dtype=np.float64)
        parts[0::2] = result.real.ravel()
        parts[1::2] = result.imag.ravel()
        if case['exact']:
            bounds = np.zeros(result.size)
            terms = np.ones(result.size)
        else:
            # An absolute sum of products bounds both output components. This
            # uses the declared source precision, after common-type conversion.
            promoted = [a.astype(np.result_type(*arrays)) for a in arrays]
            magnitudes = [np.abs(a.real.astype(np.float64)) +
                          np.abs(a.imag.astype(np.float64)) for a in promoted]
            bounds = np.asarray(np.einsum(case['sub'], *magnitudes,
                                         optimize=optimize)).ravel()
            ones = [np.ones(a.shape, dtype=np.float64) for a in arrays]
            terms = np.asarray(np.einsum(case['sub'], *ones,
                                        optimize=optimize)).ravel()
        rows.append(dict(shape=list(result.shape), dtype=str(result.dtype),
                         bits=parts.view(np.uint64).tolist(),
                         bounds=[str(value) for value in bounds],
                         terms=terms.tolist()))
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
            .expect("launch live NumPy einsum oracle");
        serde_json::to_writer(child.stdin.take().unwrap(), &self.cases).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy einsum oracle failed: {}",
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
            let mut shape = actual.shape().to_vec();
            assert_eq!(shape.pop(), Some(2), "{case}: component axis");
            assert_eq!(json!(shape), expected["shape"], "{case}: logical shape");
            assert_eq!(
                dtype_name(actual.dtype()),
                expected["dtype"],
                "{case}: dtype"
            );
            let bits = expected["bits"].as_array().unwrap();
            assert_eq!(actual.values().len(), bits.len(), "{case}: component count");
            let (epsilon, smallest) = if actual.dtype() == DType::Complex64 {
                (f64::from(f32::EPSILON), f64::from(f32::from_bits(1)))
            } else {
                (f64::EPSILON, f64::from_bits(1))
            };
            for (index, (&actual, bits)) in actual.values().iter().zip(bits).enumerate() {
                let expected_value = f64::from_bits(bits.as_u64().unwrap());
                if expected_value.is_nan() {
                    assert!(actual.is_nan(), "{case}: component {index} expected NaN");
                } else if case["exact"].as_bool().unwrap() || expected_value.is_infinite() {
                    assert_eq!(
                        actual.to_bits(),
                        expected_value.to_bits(),
                        "{case}: component {index}"
                    );
                } else {
                    let scale: f64 = expected["bounds"][index / 2]
                        .as_str()
                        .unwrap()
                        .parse()
                        .unwrap();
                    let terms = expected["terms"][index / 2].as_f64().unwrap();
                    let factors = case["operands"].as_array().unwrap().len() as f64;
                    let tolerance = 8.0 * factors * (terms + 1.0) * (epsilon * scale + smallest);
                    assert!(
                        actual.is_finite() && (actual - expected_value).abs() <= tolerance,
                        "{case}: component {index}: got {actual}, expected {expected_value}, bound {tolerance}"
                    );
                }
            }
        }
    }
}

#[test]
fn complex_contractions_and_scalar_products_match_numpy() {
    let cases: &[(&str, &[&[usize]])] = &[
        ("i,i->", &[&[5], &[5]]),
        ("ij,jk->ik", &[&[2, 3], &[3, 4]]),
        ("ij,kj->ik", &[&[2, 3], &[4, 3]]),
        ("i,j->ij", &[&[2], &[3]]),
        (",i->i", &[&[], &[3]]),
        ("i,->i", &[&[3], &[]]),
        (",->", &[&[], &[]]),
        ("ij,ij->ij", &[&[2, 3], &[2, 3]]),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &(sub, shapes) in cases {
            let operands: Vec<_> = shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| source(shape, dtype, index * 3))
                .collect();
            for strategy in [None, Some("greedy"), Some("optimal")] {
                batch.case(sub, &operands, strategy, false);
            }
        }
    }
    batch.finish();
}

#[test]
fn complex_diagonals_permutations_and_reductions_use_logical_axes() {
    let cases: &[(&str, &[usize], bool)] = &[
        ("ij->ji", &[2, 3], true),
        ("ii->i", &[3, 3], true),
        ("iji->ji", &[2, 3, 2], true),
        ("ij->", &[2, 3], false),
        ("ii->", &[3, 3], false),
        ("ijk->k", &[2, 3, 4], false),
        ("i->", &[5], false),
        ("->", &[], true),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &(sub, shape, exact) in cases {
            let operands = [source(shape, dtype, 3)];
            for strategy in [None, Some("greedy")] {
                batch.case(sub, &operands, strategy, exact);
            }
        }
    }
    batch.finish();
}

#[test]
fn complex_ellipsis_broadcasts_and_optimized_ellipsis_reductions_match_numpy() {
    let cases: &[(&str, &[&[usize]])] = &[
        ("...ij,...jk->...ik", &[&[2, 1, 2, 3], &[4, 3, 2]]),
        ("...i,...i->...", &[&[2, 3], &[1, 3]]),
        ("...i,i->...i", &[&[2, 3], &[3]]),
        ("i...,...i->...", &[&[3, 2, 1], &[4, 3]]),
        ("...i->", &[&[2, 3]]),
        ("...i->i", &[&[2, 3]]),
        ("...->...", &[&[]]),
        ("...i,j->...ij", &[&[2, 3], &[4]]),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &(sub, shapes) in cases {
            let operands: Vec<_> = shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| source(shape, dtype, index + 2))
                .collect();
            for strategy in [None, Some("greedy"), Some("optimal")] {
                batch.case(sub, &operands, strategy, false);
            }
        }
    }
    batch.finish();
}

#[test]
fn multiple_complex_operands_support_direct_greedy_and_optimal_execution() {
    let cases: &[(&str, &[&[usize]])] = &[
        ("ij,jk,kl->il", &[&[2, 3], &[3, 4], &[4, 2]]),
        ("ij,jk,kl,lm->im", &[&[2, 3], &[3, 4], &[4, 2], &[2, 3]]),
        ("ij,jk,ki->", &[&[2, 3], &[3, 4], &[4, 2]]),
        ("ii,ij,jk->k", &[&[3, 3], &[3, 2], &[2, 4]]),
        ("i,j,k->ijk", &[&[2], &[3], &[4]]),
        ("i,i,->", &[&[5], &[5], &[]]),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &(sub, shapes) in cases {
            let operands: Vec<_> = shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| source(shape, dtype, index + 1))
                .collect();
            for strategy in [None, Some("greedy"), Some("optimal"), Some("auto")] {
                batch.case(sub, &operands, strategy, false);
            }
        }
    }
    batch.finish();
}

#[test]
fn complex_einsum_promotes_all_numeric_sources_at_declared_precision() {
    let dtypes = [
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
    ];
    let mut batch = OracleBatch::default();
    for complex in [DType::Complex64, DType::Complex128] {
        for other in dtypes {
            for (left, right) in [(complex, other), (other, complex)] {
                let operands = [source(&[2, 3], left, 1), source(&[2, 3], right, 4)];
                for sub in ["ij,ij->ij", "ij,ij->i", "ij,ij->"] {
                    batch.case(sub, &operands, None, false);
                }
            }
        }
    }
    batch.finish();
}

#[test]
fn wide_integer_inputs_and_pairwise_promotion_match_numpy() {
    let signed = UFuncArray::from_storage(
        vec![3],
        ArrayStorage::I64(vec![(1_i64 << 53) + 1, i64::MIN, i64::MAX]),
    )
    .unwrap();
    let unsigned = UFuncArray::from_storage(
        vec![3],
        ArrayStorage::U64(vec![(1_u64 << 53) + 3, (1_u64 << 63) + 3, u64::MAX]),
    )
    .unwrap();
    let narrow = UFuncArray::from_storage(vec![2], ArrayStorage::I8(vec![100, 100])).unwrap();
    let one = UFuncArray::new(vec![2], vec![1.0, 0.0], DType::Complex64).unwrap();
    let floats = source(&[7], DType::F32, 3);
    let mut batch = OracleBatch::default();
    for integer in [signed, unsigned] {
        for dtype in [DType::Complex64, DType::Complex128] {
            let identity =
                UFuncArray::new(vec![3, 2], vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype).unwrap();
            batch.case("i,i->i", &[integer.clone(), identity], None, true);
            batch.case(
                "i,i->i",
                &[integer.clone(), source(&[3], dtype, 5)],
                None,
                false,
            );
            batch.case(
                "i,i->",
                &[integer.clone(), source(&[3], dtype, 5)],
                None,
                false,
            );
        }
    }
    for optimize in [None, Some("greedy"), Some("optimal")] {
        batch.case(
            "i,i,->",
            &[narrow.clone(), narrow.clone(), one.clone()],
            optimize,
            false,
        );
        batch.case(
            "i,i,->",
            &[floats.clone(), floats.clone(), one.clone()],
            optimize,
            false,
        );
        for (dtype, tiny, large) in [(DType::F32, 1.0e-50, 1.0e30), (DType::F16, 1.0e-10, 1000.0)] {
            let tiny = UFuncArray::new(vec![1], vec![tiny], dtype).unwrap();
            let large = UFuncArray::new(vec![1], vec![large], dtype).unwrap();
            let complex_one = UFuncArray::new(vec![2], vec![1.0, 0.0], DType::Complex128).unwrap();
            batch.case("i,i,->", &[tiny, large, complex_one], optimize, true);
        }
        for (dtype, maximum) in [(DType::F32, f64::from(f32::MAX)), (DType::F16, 65504.0)] {
            let large = UFuncArray::new(vec![2], vec![maximum, maximum], dtype).unwrap();
            let signs = UFuncArray::new(vec![2], vec![2.0, -2.0], dtype).unwrap();
            let complex_one = UFuncArray::new(vec![2], vec![1.0, 0.0], DType::Complex128).unwrap();
            batch.case("i,i,->", &[large, signs, complex_one], optimize, true);
        }
    }
    batch.finish();
}

#[test]
fn complex_einsum_without_reduction_promotes_all_operands_together() {
    let half = UFuncArray::new(vec![2, 2], vec![65504.0; 4], DType::F16).unwrap();
    let signs = UFuncArray::new(vec![2, 2], vec![2.0, -2.0, 2.0, -2.0], DType::F16).unwrap();
    let half_vector = UFuncArray::new(vec![2], vec![65504.0; 2], DType::F16).unwrap();
    let sign_vector = UFuncArray::new(vec![2], vec![2.0, -2.0], DType::F16).unwrap();
    let narrow = UFuncArray::from_storage(vec![2], ArrayStorage::I8(vec![100, 100])).unwrap();
    let one = UFuncArray::new(vec![2], vec![1.0, 0.0], DType::Complex128).unwrap();
    let complex_vector =
        UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 1.0, 0.0], DType::Complex64).unwrap();
    let cases = [
        ("ij,ij,->ij", vec![half.clone(), signs.clone(), one.clone()]),
        (
            "ii,i,->i",
            vec![half.clone(), sign_vector.clone(), one.clone()],
        ),
        (
            "...i,...i,->...i",
            vec![half, sign_vector.clone(), one.clone()],
        ),
        (
            "i,j,->ij",
            vec![half_vector.clone(), sign_vector, one.clone()],
        ),
        ("i,j,", vec![half_vector, narrow.clone(), one.clone()]),
        (
            "i,i,i->i",
            vec![narrow.clone(), narrow.clone(), complex_vector],
        ),
        (
            ",,->",
            vec![
                UFuncArray::scalar(100.0, DType::I8),
                UFuncArray::scalar(100.0, DType::I8),
                one.clone(),
            ],
        ),
        (
            "i,i,->i",
            vec![
                UFuncArray::new(vec![2], vec![f64::from(f32::MAX); 2], DType::F32).unwrap(),
                UFuncArray::new(vec![2], vec![2.0, -2.0], DType::F32).unwrap(),
                one.clone(),
            ],
        ),
        (
            "ij,ij,,->ij",
            vec![
                UFuncArray::new(vec![2, 2], vec![65504.0; 4], DType::F16).unwrap(),
                UFuncArray::new(vec![2, 2], vec![2.0; 4], DType::F16).unwrap(),
                one.clone(),
                one.clone(),
            ],
        ),
        (
            "ij,j,->ij",
            vec![
                UFuncArray::new(vec![0, 2], vec![], DType::F16).unwrap(),
                narrow.clone(),
                one,
            ],
        ),
    ];
    let mut batch = OracleBatch::default();
    for (sub, operands) in cases {
        let refs: Vec<_> = operands.iter().collect();
        let (path, _) = UFuncArray::einsum_path(sub, &refs).unwrap();
        // NumPy's indices == output_set shortcut is shared by both planners.
        assert_eq!(path, vec![(0..operands.len()).collect::<Vec<_>>()], "{sub}");
        for strategy in [None, Some("greedy"), Some("optimal"), Some("auto")] {
            batch.case(sub, &operands, strategy, true);
        }
    }
    batch.finish();

    // The all-operand fallback must bypass the integer evaluator's own planner
    // shortcut, so a real integer expression terminates with native wrapping.
    let result =
        UFuncArray::einsum_optimized("i,i,i->i", &[&narrow, &narrow, &narrow], "greedy").unwrap();
    assert_eq!(result.dtype(), DType::I8);
    assert_eq!(result.values(), &[64.0, 64.0]);
}

#[test]
fn complex_einsum_relocations_preserve_signed_zero_and_nonfinite_components() {
    let payload = f64::from_bits(0x7ff8_0123_4567_89ab);
    let original = UFuncArray::new(
        vec![2, 2, 2],
        vec![1.0, payload, 2.0, -0.0, 3.0, 4.0, 5.0, 6.0],
        DType::Complex128,
    )
    .unwrap();
    let transpose = UFuncArray::einsum("ij->ji", &[&original]).unwrap();
    let expected = [1.0, payload, 3.0, 4.0, 2.0, -0.0, 5.0, 6.0];
    assert_eq!(
        transpose
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let array = UFuncArray::new(
            vec![2, 2, 2],
            vec![
                -0.0,
                0.0,
                f64::INFINITY,
                2.0,
                1.0,
                f64::NEG_INFINITY,
                f64::NAN,
                -0.0,
            ],
            dtype,
        )
        .unwrap();
        for sub in ["ij->ji", "ii->i", "ij->ij"] {
            for optimize in [None, Some("greedy")] {
                batch.case(sub, std::slice::from_ref(&array), optimize, true);
            }
        }
    }
    batch.finish();
}

#[test]
fn empty_complex_contractions_and_invalid_expressions_match_numpy() {
    let cases: &[(&str, &[&[usize]])] = &[
        ("i,i->", &[&[0], &[0]]),
        ("ij,jk->ik", &[&[2, 0], &[0, 3]]),
        ("ij,jk->ik", &[&[0, 3], &[3, 4]]),
        ("ii->i", &[&[0, 0]]),
        ("...i,...i->...", &[&[0, 3], &[1, 3]]),
        ("ij,jk,kl->il", &[&[2, 0], &[0, 3], &[3, 4]]),
        ("ij,jk->ik", &[&[2, 3], &[4, 2]]),
        ("ii->i", &[&[2, 3]]),
        ("ij->ii", &[&[2, 3]]),
        ("ij->k", &[&[2, 3]]),
        ("i->i", &[&[2, 3]]),
        ("ij...->...ij", &[&[3]]),
        ("i..->i", &[&[3]]),
        ("...i...->...", &[&[3]]),
        ("ij,jk->ik->i", &[&[2, 3], &[3, 2]]),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &(sub, shapes) in cases {
            let operands: Vec<_> = shapes
                .iter()
                .enumerate()
                .map(|(i, s)| source(s, dtype, i))
                .collect();
            for optimize in [None, Some("greedy")] {
                batch.case(sub, &operands, optimize, false);
            }
        }
    }
    batch.finish();
}

#[test]
fn malformed_complex_layouts_and_impossible_outputs_fail_without_panicking() {
    let malformed = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::Complex128).unwrap();
    assert!(UFuncArray::einsum("i->i", &[&malformed]).is_err());
    assert!(UFuncArray::einsum_path("i->i", &[&malformed]).is_err());
    assert!(UFuncArray::einsum_optimized("i->i", &[&malformed], "greedy").is_err());

    let huge = UFuncArray::new(vec![0, usize::MAX, 2], vec![], DType::Complex128).unwrap();
    assert!(matches!(
        UFuncArray::einsum("ij->j", &[&huge]),
        Err(UFuncError::Shape(ShapeError::Overflow))
    ));
    let byte_limit = (isize::MAX as usize) / std::mem::size_of::<f64>();
    let huge = UFuncArray::new(vec![0, byte_limit, 2], vec![], DType::Complex128).unwrap();
    assert!(
        UFuncArray::einsum("ij->j", &[&huge])
            .unwrap_err()
            .to_string()
            .contains("allocate")
    );
    assert!(UFuncArray::einsum_optimized("ij->ij", &[&huge], "unknown").is_err());
    let scalar = source(&[], DType::Complex128, 0);
    assert!(UFuncArray::einsum("\u{E000}->", &[&scalar]).is_err());
    assert!(UFuncArray::einsum_optimized("\u{E000}->", &[&scalar], "greedy").is_err());
}

#[test]
fn complex_einsum_path_counts_logical_elements_and_matches_numpy_matrix_chain() {
    let operands = [
        source(&[2, 7], DType::Complex64, 0),
        source(&[7, 3], DType::Complex128, 1),
        source(&[3, 4], DType::Complex64, 2),
    ];
    let refs: Vec<_> = operands.iter().collect();
    let (path, description) = UFuncArray::einsum_path("ij,jk,kl->il", &refs).unwrap();
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = Command::new(python).args(["-c", r#"
import json, numpy as np
arrays = [np.ones((2,7), dtype='complex64'), np.ones((7,3), dtype='complex128'), np.ones((3,4), dtype='complex64')]
print(json.dumps(np.einsum_path('ij,jk,kl->il', *arrays, optimize='greedy')[0][1:]))
"#]).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<Vec<usize>> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(path, expected);
    assert!(description.contains("[14, 21, 12]"), "{description}");
}

#[test]
fn optimal_einsum_minimizes_total_cost_before_narrow_intermediate_rounding() {
    // The smaller intermediate (5x5) costs 1,350 FLOPs over the full path;
    // the 1x26 intermediate costs only 520. Choosing the former also overflows
    // float16 before the complex operand can promote the expression.
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        let operands = [
            UFuncArray::new(vec![5, 1], vec![65504.0; 5], DType::F16).unwrap(),
            UFuncArray::new(vec![1, 5], vec![2.0; 5], DType::F16).unwrap(),
            UFuncArray::new(vec![5, 26, 2], [1.0, 0.0].repeat(5 * 26), dtype).unwrap(),
        ];
        let refs: Vec<_> = operands.iter().collect();
        let (path, _) =
            UFuncArray::einsum_path_with_options("ab,bc,cd->ad", &refs, "optimal", None).unwrap();
        assert_eq!(path, vec![vec![1, 2], vec![0, 1]]);
        let actual = UFuncArray::einsum_optimized("ab,bc,cd->ad", &refs, "optimal").unwrap();
        assert_eq!(actual.dtype(), dtype);
        assert_eq!(actual.shape(), &[5, 26, 2]);
        assert_eq!(actual.values(), [655040.0, 0.0].repeat(5 * 26));
        for limit in [None, Some(0), Some(25), Some(26)] {
            batch.case_with_limit("ab,bc,cd->ad", &operands, Some("optimal"), limit, true);
        }
    }
    batch.finish();
}

#[test]
fn einsum_paths_match_numpy_across_strategies_memory_broadcasts_and_empty_axes() {
    let cases: &[(&str, &[&[usize]])] = &[
        ("ab,bc,cd->ad", &[&[5, 1], &[1, 5], &[5, 26]]),
        ("ab,bc,cd,de->ae", &[&[1, 1], &[1, 1], &[1, 1], &[1, 1]]),
        ("ab,bc,cd,de->ae", &[&[2, 1], &[1, 2], &[2, 3], &[3, 5]]),
        ("ab,bc,cd,de->ae", &[&[5, 2], &[2, 3], &[3, 4], &[4, 2]]),
        ("ab,ac,ad->bcd", &[&[3, 4], &[3, 5], &[3, 6]]),
        ("ii,ij,jk->k", &[&[3, 3], &[3, 2], &[2, 4]]),
        ("ab,bc,cd->ad", &[&[1, 3], &[3, 1], &[4, 5]]),
        (
            "...ij,...jk,...kl->...il",
            &[&[2, 1, 2, 3], &[4, 3, 2], &[1, 2, 3]],
        ),
        ("...i,...i,i->", &[&[2, 3], &[1, 3], &[3]]),
        ("ab,bc,cd->ad", &[&[0, 3], &[3, 2], &[2, 4]]),
        ("i,j,k->ijk", &[&[2], &[3], &[4]]),
        (",i,i->", &[&[], &[3], &[3]]),
    ];
    let mut requests = Vec::new();
    let mut actual = Vec::new();
    for &(sub, shapes) in cases {
        let operands: Vec<_> = shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| {
                source(
                    shape,
                    if i % 2 == 0 {
                        DType::Complex64
                    } else {
                        DType::F64
                    },
                    i,
                )
            })
            .collect();
        let refs: Vec<_> = operands.iter().collect();
        for strategy in ["optimal", "greedy", "auto"] {
            for limit in [
                None,
                Some(0),
                Some(1),
                Some(4),
                Some(8),
                Some(32),
                Some(256),
            ] {
                let (path, _) = UFuncArray::einsum_path_with_options(sub, &refs, strategy, limit)
                    .unwrap_or_else(|error| panic!("{sub} {strategy} {limit:?}: {error}"));
                actual.push(path);
                requests.push(
                    json!({"sub": sub, "shapes": shapes, "strategy": strategy, "limit": limit}),
                );
            }
        }
    }
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let script = r#"
import json, sys
import numpy as np
rows = []
for case in json.load(sys.stdin):
    arrays = [np.broadcast_to(np.array(0, dtype=np.complex64), shape) for shape in case['shapes']]
    strategy = 'greedy' if case['strategy'] == 'auto' else case['strategy']
    optimize = strategy if case['limit'] is None else (strategy, case['limit'])
    rows.append(np.einsum_path(case['sub'], *arrays, optimize=optimize)[0][1:])
print(json.dumps(rows))
"#;
    let mut child = Command::new(python)
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch live NumPy path oracle");
    serde_json::to_writer(child.stdin.take().unwrap(), &requests).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<Vec<Vec<usize>>> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(expected.len(), actual.len());
    for ((request, actual), expected) in requests.iter().zip(actual).zip(expected) {
        assert_eq!(actual, expected, "{request}");
    }
}

#[test]
fn memory_limited_einsum_executes_remaining_operands_as_one_contraction() {
    let cases: &[(&str, &[&[usize]])] = &[
        ("ab,bc,cd,de->ae", &[&[2, 1], &[1, 2], &[2, 3], &[3, 5]]),
        ("ab,ac,ad->bcd", &[&[3, 4], &[3, 5], &[3, 6]]),
        (
            "...ij,...jk,...kl->...il",
            &[&[2, 1, 2, 3], &[4, 3, 2], &[1, 2, 3]],
        ),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &(sub, shapes) in cases {
            let operands: Vec<_> = shapes
                .iter()
                .enumerate()
                .map(|(i, shape)| source(shape, dtype, i + 2))
                .collect();
            for strategy in ["greedy", "optimal"] {
                for limit in [None, Some(0), Some(1), Some(4)] {
                    batch.case_with_limit(sub, &operands, Some(strategy), limit, false);
                }
            }
        }
    }
    batch.finish();

    let operands = [
        source(&[2, 1], DType::Complex64, 0),
        source(&[1, 2], DType::Complex64, 1),
        source(&[2, 3], DType::Complex64, 2),
        source(&[3, 5], DType::Complex64, 3),
    ];
    let refs: Vec<_> = operands.iter().collect();
    let (path, _) =
        UFuncArray::einsum_path_with_options("ab,bc,cd,de->ae", &refs, "optimal", Some(4)).unwrap();
    assert_eq!(path, vec![vec![0, 1], vec![0, 1, 2]]);
}

#[test]
fn optimal_einsum_merges_equivalent_states_above_ten_operands() {
    let empty = source(&[0], DType::Complex128, 0);
    let operands = vec![&empty; 11];
    let sub = format!("{}->", vec!["i"; operands.len()].join(","));
    let (path, _) = UFuncArray::einsum_path_with_options(&sub, &operands, "optimal", None).unwrap();
    assert_eq!(path, vec![vec![0, 1]; operands.len() - 1]);
    let result = UFuncArray::einsum_optimized(&sub, &operands, "optimal").unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.values(), &[0.0, 0.0]);

    let scalar = source(&[], DType::Complex128, 0);
    assert!(UFuncArray::einsum_path_with_options("->", &[&scalar], "unknown", Some(0)).is_err());
    assert!(
        UFuncArray::einsum_optimized_with_memory_limit("->", &[&scalar], "unknown", Some(0))
            .is_err()
    );
    let malformed = UFuncArray::new(vec![3], vec![1.0; 3], DType::Complex128).unwrap();
    assert!(
        UFuncArray::einsum_path_with_options("i->i", &[&malformed], "optimal", Some(0)).is_err()
    );

    let huge = UFuncArray::new(vec![0, usize::MAX, 2], vec![], DType::Complex128).unwrap();
    assert!(
        UFuncArray::einsum_optimized_with_memory_limit(
            "ab,ab,->b",
            &[&huge, &huge, &scalar],
            "optimal",
            Some(0),
        )
        .is_err()
    );
}

#[test]
fn real_einsum_memory_fallback_and_narrow_pairs_preserve_result_precision() {
    let shapes: &[&[usize]] = &[&[2, 1], &[1, 2], &[2, 3], &[3, 2]];
    let mut requests = Vec::new();
    let mut actual = Vec::new();
    for dtype in [DType::F16, DType::F32, DType::F64] {
        for extreme in [false, true] {
            let maximum = match dtype {
                DType::F16 => 65504.0,
                DType::F32 => f64::from(f32::MAX),
                _ => f64::MAX,
            };
            let operands: Vec<_> = shapes
                .iter()
                .enumerate()
                .map(|(operand, shape)| {
                    let count: usize = shape.iter().product();
                    let values = (0..count)
                        .map(|index| {
                            if extreme {
                                if operand == 0 { maximum } else { 2.0 }
                            } else if index % 5 == 0 {
                                1.0 + 2.0_f64.powi(-24)
                            } else {
                                ((index * 3 + operand) % 7 + 1) as f64 / 4.0
                            }
                        })
                        .collect();
                    UFuncArray::new(shape.to_vec(), values, dtype).unwrap()
                })
                .collect();
            let refs: Vec<_> = operands.iter().collect();
            for strategy in ["greedy", "optimal"] {
                for limit in [None, Some(0), Some(4)] {
                    actual.push(
                        UFuncArray::einsum_optimized_with_memory_limit(
                            "ab,bc,cd,de->ae",
                            &refs,
                            strategy,
                            limit,
                        )
                        .unwrap(),
                    );
                    requests.push(json!({
                        "operands": operands.iter().map(oracle_input).collect::<Vec<_>>(),
                        "strategy": strategy, "limit": limit,
                    }));
                }
            }
        }
    }
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let script = r#"
import json, sys, warnings
import numpy as np
warnings.simplefilter('ignore')
np.seterr(all='ignore')
rows = []
for case in json.load(sys.stdin):
    arrays = [np.array([float(v) for v in a['values']], dtype=a['dtype']).reshape(a['shape'])
              for a in case['operands']]
    optimize = case['strategy'] if case['limit'] is None else (case['strategy'], case['limit'])
    value = np.asarray(np.einsum('ab,bc,cd,de->ae', *arrays, optimize=optimize))
    rows.append(dict(dtype=str(value.dtype), shape=list(value.shape),
                     bits=value.astype(np.float64).ravel().view(np.uint64).tolist()))
print(json.dumps(rows))
"#;
    let mut child = Command::new(python)
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch live NumPy real-einsum oracle");
    serde_json::to_writer(child.stdin.take().unwrap(), &requests).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(expected.len(), actual.len());
    for ((request, actual), expected) in requests.iter().zip(actual).zip(expected) {
        assert_eq!(dtype_name(actual.dtype()), expected["dtype"], "{request}");
        assert_eq!(json!(actual.shape()), expected["shape"], "{request}");
        let epsilon = match actual.dtype() {
            DType::F16 => 2.0_f64.powi(-10),
            DType::F32 => f64::from(f32::EPSILON),
            _ => f64::EPSILON,
        };
        let bits = expected["bits"].as_array().unwrap();
        assert_eq!(actual.values().len(), bits.len());
        for (&got, bits) in actual.values().iter().zip(bits) {
            let want = f64::from_bits(bits.as_u64().unwrap());
            if want.is_nan() {
                assert!(got.is_nan(), "{request}: expected NaN, got {got}");
            } else if want.is_infinite() {
                assert_eq!(got.to_bits(), want.to_bits(), "{request}");
            } else {
                assert!(
                    got.is_finite() && (got - want).abs() <= 8.0 * epsilon * want.abs().max(1.0),
                    "{request}: got {got}, expected {want}"
                );
            }
        }
    }
}
