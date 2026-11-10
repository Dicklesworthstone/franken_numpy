//! NumPy choose contracts for logical broadcasting, numeric promotion and exact values.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{UFuncArray, UFuncError};
use serde_json::{Value, json};
use std::process::{Command, Stdio};

const NUMERIC_DTYPES: [DType; 14] = [
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
        DType::DateTime64 => "datetime64[ns]",
        DType::TimeDelta64 => "timedelta64[ns]",
        _ => panic!("non-numeric choose dtype"),
    }
}

fn complex(shape: &[usize], parts: &[(f64, f64)], dtype: DType) -> UFuncArray {
    let mut physical_shape = shape.to_vec();
    physical_shape.push(2);
    UFuncArray::new(
        physical_shape,
        parts
            .iter()
            .flat_map(|&(real, imaginary)| [real, imaginary])
            .collect(),
        dtype,
    )
    .unwrap()
}

fn sample(shape: &[usize], dtype: DType, offset: usize) -> UFuncArray {
    let count = if shape.contains(&0) {
        0
    } else {
        shape.iter().product()
    };
    let pattern = [
        (0.0, -0.0),
        (-0.0, 3.25),
        (1.0 + 2.0_f64.powi(-24), -1.0 - 2.0_f64.powi(-24)),
        (1.0e-50, -1.0e-50),
        (f64::NAN, 2.5),
        (f64::INFINITY, f64::NEG_INFINITY),
        (-7.25, f64::NAN),
    ];
    if dtype.is_complex() {
        let parts = (0..count)
            .map(|index| pattern[(index + offset) % pattern.len()])
            .collect::<Vec<_>>();
        return complex(shape, &parts, dtype);
    }
    if dtype.is_integer() || dtype == DType::Bool {
        let values = (0..count)
            .map(|index| ((index + offset) % 7) as f64)
            .collect();
        return UFuncArray::from_storage(
            shape.to_vec(),
            ArrayStorage::F64(values).cast_to(dtype).unwrap(),
        )
        .unwrap();
    }
    let values = (0..count)
        .map(|index| pattern[(index + offset) % pattern.len()].0)
        .collect();
    UFuncArray::new(shape.to_vec(), values, dtype).unwrap()
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
    } else if array.dtype().is_integer()
        || array.dtype() == DType::Bool
        || matches!(array.dtype(), DType::DateTime64 | DType::TimeDelta64)
    {
        let storage = if matches!(array.dtype(), DType::DateTime64 | DType::TimeDelta64) {
            array.astype(DType::I64).to_storage().unwrap()
        } else {
            array.to_storage().unwrap()
        };
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
    fn push(&mut self, indices: &UFuncArray, choices: &[UFuncArray], mode: &str) {
        self.cases.push(json!({
            "indices": oracle_input(indices),
            "choices": choices.iter().map(oracle_input).collect::<Vec<_>>(),
            "mode": mode,
        }));
        self.actual.push(indices.choose_with_mode(choices, mode));
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np

warnings.simplefilter('ignore')
def load(item):
    dtype = np.dtype(item['dtype'])
    if dtype.kind == 'c':
        result = np.empty(len(item['values']), dtype=dtype)
        result.real = [float(parts[0]) for parts in item['values']]
        result.imag = [float(parts[1]) for parts in item['values']]
    elif dtype.kind in 'Mm':
        result = np.array([int(value) for value in item['values']], dtype=np.int64).view(dtype)
    else:
        convert = int if dtype.kind in 'biu' else float
        result = np.array([convert(value) for value in item['values']], dtype=dtype)
    return result.reshape(item['shape'])

rows = []
for case in json.load(sys.stdin):
    try:
        result = np.asarray(np.choose(load(case['indices']), [load(choice) for choice in case['choices']], mode=case['mode']))
        if result.dtype.kind == 'c':
            parts = np.empty(result.size * 2, dtype=np.float64)
            parts[0::2] = result.real.ravel()
            parts[1::2] = result.imag.ravel()
        elif result.dtype.kind in 'Mm':
            parts = result.view(np.int64).astype(np.float64).ravel()
        else:
            parts = result.astype(np.float64).ravel()
        rows.append(dict(shape=list(result.shape), dtype=str(result.dtype), bits=parts.view(np.uint64).tolist(),
                         integers=[str(int(value)) for value in result.view(np.int64).flat] if result.dtype.kind in 'Mm'
                             else [str(int(value)) for value in result.flat] if result.dtype.kind in 'biu' else None))
    except (TypeError, ValueError, RuntimeError) as error:
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
            .expect("launch live NumPy choose oracle");
        let mut stdin = child.stdin.take().unwrap();
        serde_json::to_writer(&mut stdin, &self.cases).unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "choose oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(expected.len(), self.actual.len());
        for ((case, actual), expected) in self.cases.iter().zip(self.actual).zip(expected) {
            if let Some(error) = expected.get("error") {
                assert!(
                    actual.is_err(),
                    "{case}: expected {expected}, got {actual:?}"
                );
                if error == "TypeError" {
                    assert!(
                        matches!(actual, Err(UFuncError::TypeResolutionInvalid { .. })),
                        "{case}: wrong error {actual:?}"
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
            if actual.dtype().is_integer()
                || actual.dtype() == DType::Bool
                || matches!(actual.dtype(), DType::DateTime64 | DType::TimeDelta64)
            {
                let storage = if matches!(actual.dtype(), DType::DateTime64 | DType::TimeDelta64) {
                    actual.astype(DType::I64).to_storage().unwrap()
                } else {
                    actual.to_storage().unwrap()
                };
                let integers = (0..storage.len())
                    .map(|index| storage.get_i128(index).unwrap().to_string())
                    .collect::<Vec<_>>();
                assert_eq!(
                    json!(integers),
                    expected["integers"],
                    "{case}: exact values"
                );
            }
            let bits = expected["bits"].as_array().unwrap();
            assert_eq!(
                actual.values().len(),
                bits.len(),
                "{case}: physical component count"
            );
            for (index, (&value, bits)) in actual.values().iter().zip(bits).enumerate() {
                let expected = f64::from_bits(bits.as_u64().unwrap());
                if expected.is_nan() {
                    assert!(value.is_nan(), "{case}: expected NaN at component {index}");
                } else {
                    assert_eq!(
                        value.to_bits(),
                        expected.to_bits(),
                        "{case}: component {index}"
                    );
                }
            }
        }
    }
}

#[test]
fn choose_broadcasts_indices_and_all_logical_choice_shapes() {
    let indices = UFuncArray::from_storage(vec![2, 1], ArrayStorage::I64(vec![0, 2])).unwrap();
    let mut batch = OracleBatch::default();
    for dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        let choices = [
            sample(&[1, 3], dtype, 0),
            sample(&[], DType::I16, 3),
            sample(&[2, 3], dtype, 1),
        ];
        for mode in ["raise", "wrap", "clip"] {
            batch.push(&indices, &choices, mode);
            batch.push(&UFuncArray::scalar(1.0, DType::I8), &choices, mode);
        }
        let indices =
            UFuncArray::from_storage(vec![2, 1, 1], ArrayStorage::I64(vec![0, 1])).unwrap();
        let choices = [sample(&[1, 3, 1], dtype, 1), sample(&[4], dtype, 3)];
        batch.push(&indices, &choices, "raise");
    }
    batch.assert_matches_numpy();
}

#[test]
fn choose_all_numeric_dtype_pairs_preserve_promoted_source_values() {
    let indices =
        UFuncArray::from_storage(vec![7], ArrayStorage::I8(vec![0, 1, 0, 1, 0, 1, 0])).unwrap();
    let mut batch = OracleBatch::default();
    for left in NUMERIC_DTYPES {
        for right in NUMERIC_DTYPES {
            let choices = [sample(&[7], left, 0), sample(&[7], right, 1)];
            batch.push(&indices, &choices, "raise");
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn choose_mixed_integer_float_common_dtype_is_order_independent() {
    let indices = UFuncArray::from_storage(vec![3], ArrayStorage::I8(vec![0, 1, 2])).unwrap();
    let mut batch = OracleBatch::default();
    for (signed, unsigned, floating) in [
        (DType::I8, DType::U8, DType::F16),
        (DType::I16, DType::U16, DType::F32),
        (DType::I16, DType::U16, DType::Complex64),
        (DType::I64, DType::U64, DType::F64),
        (DType::I64, DType::U64, DType::Complex64),
    ] {
        let inputs = [
            sample(&[], signed, 1),
            sample(&[], unsigned, 2),
            sample(&[], floating, 3),
        ];
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let choices = order.map(|index| inputs[index].clone());
            batch.push(&indices, &choices, "raise");
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn array_result_type_uses_the_common_numeric_promotion_contract() {
    assert_eq!(UFuncArray::result_type(&[]), DType::F64);
    for (dtypes, expected) in [
        ([DType::I8, DType::U8, DType::F16], DType::F16),
        ([DType::I16, DType::U16, DType::F32], DType::F32),
        ([DType::I16, DType::U16, DType::Complex64], DType::Complex64),
        ([DType::I64, DType::U64, DType::F64], DType::F64),
        (
            [DType::I64, DType::U64, DType::Complex64],
            DType::Complex128,
        ),
        ([DType::Bool, DType::U8, DType::U64], DType::U64),
    ] {
        let arrays = dtypes.map(|dtype| sample(&[], dtype, 1));
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            assert_eq!(
                UFuncArray::result_type(&order.map(|index| &arrays[index])),
                expected,
                "dtypes={dtypes:?}, order={order:?}"
            );
        }
    }
}

#[test]
fn choose_preserves_exact_wide_integer_choices_when_broadcasting() {
    let indices = UFuncArray::from_storage(vec![2, 1], ArrayStorage::I8(vec![0, 1])).unwrap();
    let signed = UFuncArray::from_storage(
        vec![1, 4],
        ArrayStorage::I64(vec![
            i64::MIN,
            -(1_i64 << 53) - 1,
            (1_i64 << 53) + 1,
            i64::MAX,
        ]),
    )
    .unwrap();
    let unsigned = UFuncArray::from_storage(
        vec![1, 4],
        ArrayStorage::U64(vec![u64::MAX, (1_u64 << 53) + 1, (1_u64 << 63) + 1, 0]),
    )
    .unwrap();
    let mut batch = OracleBatch::default();
    for (wide, narrow) in [(&signed, DType::I32), (&unsigned, DType::U32)] {
        let scalar = sample(&[], narrow, 3);
        batch.push(&indices, &[wide.clone(), scalar.clone()], "raise");
        batch.push(&indices, &[scalar, wide.clone()], "raise");
        for dtype in [DType::F32, DType::Complex64, DType::Complex128] {
            batch.push(&indices, &[wide.clone(), sample(&[4], dtype, 0)], "raise");
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn choose_preserves_homogeneous_temporal_ticks_and_metadata() {
    let indices = UFuncArray::from_storage(vec![2, 1], ArrayStorage::I8(vec![0, 1])).unwrap();
    let ticks = vec![i64::MIN, -(1_i64 << 53) - 1, (1_i64 << 53) + 1, i64::MAX];
    let mut batch = OracleBatch::default();
    for dtype in [DType::DateTime64, DType::TimeDelta64] {
        let wide = UFuncArray::from_storage_with_dtype(
            vec![1, 4],
            ArrayStorage::I64(ticks.clone()),
            dtype,
        )
        .unwrap();
        let scalar =
            UFuncArray::from_storage_with_dtype(vec![], ArrayStorage::I64(vec![17]), dtype)
                .unwrap();
        let empty =
            UFuncArray::from_storage_with_dtype(vec![2, 0, 3], ArrayStorage::I64(vec![]), dtype)
                .unwrap();
        for mode in ["raise", "wrap", "clip"] {
            batch.push(&indices, &[wide.clone(), scalar.clone()], mode);
            batch.push(&indices, &[scalar.clone(), wide.clone()], mode);
            batch.push(&UFuncArray::scalar(0.0, DType::I8), &[scalar.clone()], mode);
            batch.push(
                &UFuncArray::scalar(99.0, DType::I8),
                &[empty.clone(), scalar.clone()],
                mode,
            );
        }
        let result = indices.choose(&[wide, scalar]).unwrap();
        assert_eq!(result.dtype(), dtype);
        assert_eq!(result.shape(), &[2, 4]);
        assert!(result.has_integer_sidecar());
        let expected = ticks.iter().copied().chain([17; 4]).collect::<Vec<_>>();
        assert_eq!(
            result.astype(DType::I64).to_storage().unwrap(),
            ArrayStorage::I64(expected)
        );
    }
    batch.assert_matches_numpy();
}

#[test]
fn choose_retains_homogeneous_opaque_marker_payloads() {
    // These markers retain the existing raw-slot representation; UFuncArray's
    // f64 bridge does not encode actual string or structured-record payloads.
    let indices = UFuncArray::from_storage(vec![2, 1], ArrayStorage::I8(vec![0, 1])).unwrap();
    for dtype in [DType::Str, DType::Structured] {
        let choices = [
            UFuncArray::new(vec![1, 3], vec![11.0, 12.0, 13.0], dtype).unwrap(),
            UFuncArray::new(vec![], vec![29.0], dtype).unwrap(),
        ];
        let result = indices.choose(&choices).unwrap();
        assert_eq!(result.dtype(), dtype);
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.values(), &[11.0, 12.0, 13.0, 29.0, 29.0, 29.0]);
    }
}

#[test]
fn choose_requires_index_dtype_safely_castable_to_intp() {
    let choices = [
        sample(&[], DType::Complex64, 1),
        sample(&[], DType::Complex64, 2),
    ];
    let mut batch = OracleBatch::default();
    for dtype in NUMERIC_DTYPES {
        for shape in [&[][..], &[3][..], &[0][..]] {
            let indices = sample(shape, dtype, 0);
            for mode in ["raise", "wrap", "clip"] {
                batch.push(&indices, &choices, mode);
            }
        }
    }
    let unsigned = UFuncArray::from_storage(vec![2], ArrayStorage::U64(vec![0, u64::MAX])).unwrap();
    for mode in ["raise", "wrap", "clip"] {
        batch.push(&unsigned, &choices, mode);
    }
    batch.assert_matches_numpy();
}

#[test]
fn choose_negative_and_out_of_range_modes_match_numpy() {
    let indices = UFuncArray::from_storage(
        vec![9],
        ArrayStorage::I64(vec![-10_001, -5, -1, 0, 1, 2, 3, 10, 10_003]),
    )
    .unwrap();
    let choices = [
        sample(&[], DType::I16, 1),
        sample(&[], DType::F32, 2),
        sample(&[9], DType::F64, 3),
    ];
    let mut batch = OracleBatch::default();
    for mode in ["raise", "wrap", "clip"] {
        batch.push(&indices, &choices, mode);
    }
    let extremes = UFuncArray::from_storage(
        vec![4],
        ArrayStorage::I64(vec![
            i64::MIN,
            -(1_i64 << 53) - 1,
            (1_i64 << 53) + 1,
            i64::MAX,
        ]),
    )
    .unwrap();
    let choices = [
        sample(&[], DType::F64, 0),
        sample(&[], DType::F64, 1),
        sample(&[], DType::F64, 2),
    ];
    for mode in ["raise", "clip"] {
        batch.push(&extremes, &choices, mode);
    }
    batch.assert_matches_numpy();

    // NumPy's wrap loop repeatedly adds/subtracts the number of choices and
    // cannot finish practical oracle runs on these extremes. The documented
    // modulo contract gives exact indices [1, 0, 0, 1] without a float bridge.
    let choices = [11.0, 22.0, 33.0].map(|value| UFuncArray::scalar(value, DType::I64));
    let result = extremes.choose_with_mode(&choices, "wrap").unwrap();
    assert_eq!(
        result.to_storage().unwrap(),
        ArrayStorage::I64(vec![22, 11, 11, 22])
    );
}

#[test]
fn choose_normalizes_declared_boolean_and_narrow_source_precision() {
    let mut batch = OracleBatch::default();
    let boolean_indices =
        UFuncArray::new(vec![4], vec![f64::NAN, 2.0, -0.0, -3.0], DType::Bool).unwrap();
    let narrow_indices =
        UFuncArray::new(vec![4], vec![128.0, -129.0, 1.9, -1.9], DType::I8).unwrap();
    for (dtype, midpoint, tiny) in [
        (DType::F16, 2049.0, 2.0_f64.powi(-26)),
        (DType::F32, 16777217.0, 2.0_f64.powi(-151)),
    ] {
        let narrow =
            UFuncArray::new(vec![4], vec![midpoint, -midpoint, tiny, -tiny], dtype).unwrap();
        let choices = [narrow.clone(), sample(&[4], DType::Complex128, 0)];
        batch.push(&boolean_indices, &choices, "raise");
        batch.push(&narrow_indices, &choices, "wrap");
        batch.push(&narrow_indices, &choices, "clip");
        batch.push(
            &boolean_indices,
            &[narrow, sample(&[], DType::F64, 2)],
            "raise",
        );
    }
    batch.assert_matches_numpy();
}

#[test]
fn choose_scalar_and_empty_outputs_observe_all_choice_shapes() {
    let mut batch = OracleBatch::default();
    for dtype in [
        DType::Bool,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::Complex64,
        DType::Complex128,
    ] {
        let scalar = sample(&[], dtype, 2);
        for shape in [&[][..], &[0][..], &[2, 0, 3][..]] {
            let choices = [sample(shape, dtype, 1), scalar.clone()];
            for mode in ["raise", "wrap", "clip"] {
                batch.push(&UFuncArray::scalar(0.0, DType::I8), &choices, mode);
                batch.push(&sample(shape, DType::I8, 0), &choices, mode);
            }
        }
        // No index is visited for empty outputs, even with a scalar index that
        // would be out of bounds in a nonempty result.
        batch.push(
            &UFuncArray::scalar(19.0, DType::I8),
            &[sample(&[0], dtype, 0)],
            "raise",
        );
    }
    batch.assert_matches_numpy();

    let scalar_indices = UFuncArray::scalar(99.0, DType::I64);
    let empty = complex(&[0, usize::MAX], &[], DType::Complex128);
    let result = scalar_indices
        .choose(&[empty.clone(), sample(&[], DType::Complex64, 0)])
        .unwrap();
    assert_eq!(result.shape(), empty.shape());
    assert!(result.values().is_empty());
}

#[test]
fn choose_rejects_unselected_incompatible_choices_and_invalid_modes() {
    let indices = UFuncArray::scalar(0.0, DType::I64);
    let choices = [
        sample(&[2], DType::F64, 0),
        sample(&[3], DType::Complex64, 1),
    ];
    let mut batch = OracleBatch::default();
    for mode in ["raise", "wrap", "clip", "", "bad", "RAISE", "r"] {
        batch.push(&indices, &choices, mode);
        batch.push(&UFuncArray::scalar(0.0, DType::F64), &[], mode);
        batch.push(&indices, &[], mode);
    }
    let choices = (0..64)
        .map(|_| sample(&[], DType::F64, 0))
        .collect::<Vec<_>>();
    batch.push(&indices, &choices[..63], "raise");
    batch.push(&indices, &choices, "raise");
    batch.push(&UFuncArray::scalar(0.0, DType::F64), &choices, "raise");
    let rank33 = sample(&[1; 33], DType::I8, 0);
    batch.push(
        &sample(&[1; 32], DType::I8, 0),
        &[sample(&[], DType::F64, 0)],
        "raise",
    );
    batch.push(&rank33, &[sample(&[], DType::F64, 0)], "raise");
    batch.assert_matches_numpy();
}

#[test]
fn choose_validates_inputs_before_index_values() {
    let invalid_index = UFuncArray::scalar(9.0, DType::I64);
    let incompatible = [
        sample(&[2], DType::F64, 0),
        sample(&[3], DType::Complex64, 0),
    ];
    assert!(matches!(
        invalid_index.choose(&incompatible),
        Err(UFuncError::Shape(_))
    ));
    let invalid_dtype = UFuncArray::scalar(0.0, DType::F64);
    assert!(matches!(
        invalid_dtype.choose(&incompatible),
        Err(UFuncError::TypeResolutionInvalid { .. })
    ));
    assert!(
        matches!(invalid_dtype.choose_with_mode(&[], "bad"), Err(UFuncError::Msg(message)) if message.contains("unsupported mode"))
    );
    assert!(
        matches!(invalid_dtype.choose(&[]), Err(UFuncError::Msg(message)) if message.contains("at least 1 choice"))
    );
    let mut rank33_shape = vec![1; 33];
    rank33_shape[32] = 2;
    let rank33 = sample(&rank33_shape, DType::F64, 0);
    let incompatible_rank = [rank33, sample(&[3], DType::F64, 0)];
    assert!(
        matches!(invalid_index.choose(&incompatible_rank), Err(UFuncError::Msg(message)) if message.contains("32 logical dimensions"))
    );
    assert!(matches!(
        invalid_dtype.choose(&incompatible_rank),
        Err(UFuncError::TypeResolutionInvalid { .. })
    ));
    let valid = sample(&[], DType::Complex128, 0);
    for malformed in [
        UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::Complex64).unwrap(),
        UFuncArray::new(vec![], vec![1.0], DType::Complex128).unwrap(),
    ] {
        let result = UFuncArray::scalar(0.0, DType::I8).choose(&[valid.clone(), malformed]);
        assert!(
            matches!(result, Err(UFuncError::Msg(message)) if message.contains("interleaved trailing-dimension shape"))
        );
    }
}

#[test]
fn choose_checks_output_capacity_before_allocation() {
    let ndim = (usize::BITS as usize - 4) / 2;
    let indices = UFuncArray::scalar(0.0, DType::I8);
    for dtype in [DType::F64, DType::Complex128] {
        let choices = (0..ndim)
            .map(|axis| {
                let mut shape = vec![1; ndim];
                shape[axis] = 4;
                sample(&shape, dtype, 0)
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            indices.choose(&choices),
            Err(UFuncError::Shape(_))
        ));
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn choose_allocation_failure_precedes_out_of_range_index() {
    let indices = UFuncArray::from_storage(vec![], ArrayStorage::I64(vec![i64::MAX])).unwrap();
    // Four elements per input; 2^61 output bytes exceed virtual address space.
    let choices = (0..29)
        .map(|axis| {
            let mut shape = vec![1; 29];
            shape[axis] = 4;
            sample(&shape, DType::F32, 0)
        })
        .collect::<Vec<_>>();
    assert!(
        matches!(indices.choose(&choices), Err(UFuncError::Msg(message)) if message.contains("unable to allocate output array"))
    );
}
