//! copyto assigns logical complex elements and applies the selected cast loop.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{
    FloatErrorCallback, FloatErrorMode, UFuncArray, UFuncError, errstate, seterrcall,
    take_float_error_events,
};
use serde_json::{Value, json};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

const NUMERIC: [DType; 14] = [
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
        _ => panic!("non-numeric copyto test input"),
    }
}

fn count(shape: &[usize]) -> usize {
    if shape.contains(&0) {
        0
    } else {
        shape.iter().product()
    }
}

fn complex(shape: &[usize], values: &[(f64, f64)], dtype: DType) -> UFuncArray {
    let mut physical = shape.to_vec();
    physical.push(2);
    UFuncArray::new(
        physical,
        values
            .iter()
            .flat_map(|&(real, imaginary)| [real, imaginary])
            .collect(),
        dtype,
    )
    .unwrap()
}

fn sample(shape: &[usize], dtype: DType, seed: usize) -> UFuncArray {
    let pattern = [
        (0.0, -0.0),
        (-3.5, 2.25),
        (1.0 + 2.0_f64.powi(-24), -1.0 - 2.0_f64.powi(-24)),
        (1.0e-50, -1.0e-50),
        (f64::NAN, 4.0),
        (0.0, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY),
        (1.0e100, -1.0e100),
        (-0.0, 0.0),
    ];
    if dtype.is_complex() {
        return complex(
            shape,
            &(0..count(shape))
                .map(|index| pattern[(index + seed) % pattern.len()])
                .collect::<Vec<_>>(),
            dtype,
        );
    }
    let values = (0..count(shape))
        .map(|index| {
            if dtype.is_integer() || dtype == DType::Bool {
                ((index + seed) % 19) as f64 - 9.0
            } else {
                pattern[(index + seed) % pattern.len()].0
            }
        })
        .collect();
    UFuncArray::from_storage(
        shape.to_vec(),
        ArrayStorage::F64(values).cast_to(dtype).unwrap(),
    )
    .unwrap()
}

fn destination(shape: &[usize], dtype: DType) -> UFuncArray {
    if dtype.is_complex() {
        complex(shape, &vec![(17.0, -19.0); count(shape)], dtype)
    } else {
        UFuncArray::from_storage(
            shape.to_vec(),
            ArrayStorage::F64(vec![17.0; count(shape)])
                .cast_to(dtype)
                .unwrap(),
        )
        .unwrap()
    }
}

fn mask(shape: &[usize], pattern: &[bool]) -> UFuncArray {
    UFuncArray::new(
        shape.to_vec(),
        (0..count(shape))
            .map(|index| {
                if pattern[index % pattern.len()] {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
        DType::Bool,
    )
    .unwrap()
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

fn error_category(error: UFuncError) -> String {
    if let UFuncError::FloatingPoint { kind, .. } = error {
        return format!("float:{}", kind.as_str());
    }
    let message = error.to_string();
    if message.contains("invalid casting rule") {
        "casting-rule"
    } else if message.contains("where mask must have boolean") {
        "mask-dtype"
    } else if message.contains("cannot cast") {
        "cast-dtype"
    } else if message.contains("broadcast input array") {
        "source-shape"
    } else if message.contains("broadcast where mask") {
        "mask-shape"
    } else {
        panic!("unexpected copyto error: {message}");
    }
    .into()
}

struct CopyOutcome {
    array: UFuncArray,
    error: Option<String>,
    warnings: Vec<String>,
    calls: Vec<String>,
}

#[derive(Default)]
struct OracleBatch {
    cases: Vec<Value>,
    actual: Vec<CopyOutcome>,
}

impl OracleBatch {
    fn push(
        &mut self,
        dst: &UFuncArray,
        src: &UFuncArray,
        where_mask: Option<&UFuncArray>,
        casting: Option<&str>,
    ) {
        self.push_policy(dst, src, where_mask, casting, FloatErrorMode::Ignore);
    }

    fn push_policy(
        &mut self,
        dst: &UFuncArray,
        src: &UFuncArray,
        where_mask: Option<&UFuncArray>,
        casting: Option<&str>,
        mode: FloatErrorMode,
    ) {
        let mode_name = match mode {
            FloatErrorMode::Ignore => "ignore",
            FloatErrorMode::Warn => "warn",
            FloatErrorMode::Raise => "raise",
            FloatErrorMode::Call => "call",
            _ => panic!("unsupported test error mode"),
        };
        self.cases
            .push(json!({"dst": oracle_input(dst), "src": oracle_input(src),
            "mask": where_mask.map(oracle_input), "casting": casting, "mode": mode_name}));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let callback_calls = Arc::clone(&calls);
        let callback: FloatErrorCallback = Arc::new(move |event| {
            assert_eq!(event.op, "cast");
            callback_calls
                .lock()
                .unwrap()
                .push(event.kind.as_str().to_string());
        });
        let previous_callback = seterrcall(Some(callback));
        take_float_error_events();
        let mut actual = dst.clone();
        let failure = {
            let _errors = errstate(Some(mode), None, None, None, None);
            actual
                .copyto(src, where_mask, casting)
                .err()
                .map(error_category)
        };
        let warnings = take_float_error_events()
            .into_iter()
            .map(|event| {
                assert_eq!(event.op, "cast");
                event.kind.as_str().to_string()
            })
            .collect();
        let _ = seterrcall(previous_callback);
        let calls = calls.lock().unwrap().clone();
        self.actual.push(CopyOutcome {
            array: actual,
            error: failure,
            warnings,
            calls,
        });
    }

    fn assert_matches_numpy(self) {
        assert!(!self.cases.is_empty());
        let script = r#"
import json, sys, warnings
import numpy as np

def load(item):
    dtype = np.dtype(item['dtype'])
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        if dtype.kind == 'c':
            result = np.empty(len(item['values']), dtype=dtype)
            result.real = [float(pair[0]) for pair in item['values']]
            result.imag = [float(pair[1]) for pair in item['values']]
        else:
            convert = int if dtype.kind in 'biu' else float
            result = np.array([convert(value) for value in item['values']], dtype=dtype)
    return result.reshape(item['shape'])

def float_kind(message):
    return {'overflow': 'over', 'underflow': 'under', 'invalid': 'invalid', 'divide': 'divide'}[message.split()[0]]

def category(error, case):
    text = str(error)
    if isinstance(error, FloatingPointError): return 'float:' + float_kind(text)
    if 'casting must be one of' in text: return 'casting-rule'
    if case['mask'] is not None and case['mask']['dtype'] != 'bool' and "to dtype('bool')" in text: return 'mask-dtype'
    if 'Cannot cast array data' in text: return 'cast-dtype'
    if 'broadcast input array' in text: return 'source-shape'
    if 'broadcast where mask' in text: return 'mask-shape'
    raise AssertionError((type(error).__name__, text))

rows = []
for case in json.load(sys.stdin):
    dst, src = load(case['dst']), load(case['src'])
    kwargs = {} if case['casting'] is None else dict(casting=case['casting'])
    if case['mask'] is not None: kwargs['where'] = load(case['mask'])
    calls = []
    previous_callback = np.seterrcall(lambda message, flags: calls.append(float_kind(message)))
    failure = None
    with warnings.catch_warnings(record=True) as caught, np.errstate(all=case['mode']):
        warnings.simplefilter('always')
        try: np.copyto(dst, src, **kwargs)
        except (TypeError, ValueError, FloatingPointError) as error: failure = category(error, case)
    np.seterrcall(previous_callback)
    messages = [float_kind(str(warning.message)) for warning in caught
                if issubclass(warning.category, RuntimeWarning)
                and not issubclass(warning.category, np.exceptions.ComplexWarning)]
    if dst.dtype.kind == 'c':
        parts = np.empty(dst.size * 2, dtype=np.float64)
        parts[0::2], parts[1::2] = dst.real.ravel(), dst.imag.ravel()
    else:
        parts = dst.astype(np.float64).ravel()
    rows.append(dict(shape=list(dst.shape), dtype=str(dst.dtype), error=failure,
                     warnings=messages, calls=calls, bits=parts.view(np.uint64).tolist(),
                     integers=[str(int(value)) for value in dst.flat] if dst.dtype.kind in 'biu' else None))
print(json.dumps(rows))
"#;
        let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(python)
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("launch live NumPy copyto oracle");
        let mut stdin = child.stdin.take().unwrap();
        serde_json::to_writer(&mut stdin, &self.cases).unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy copyto oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            expected.len(),
            self.cases.len(),
            "oracle returned every case"
        );
        assert_eq!(expected.len(), self.actual.len());
        for ((case, outcome), expected) in self.cases.iter().zip(self.actual).zip(expected) {
            let CopyOutcome {
                array: actual,
                error,
                warnings,
                calls,
            } = outcome;
            assert_eq!(json!(error), expected["error"], "{case}: error precedence");
            assert_eq!(
                json!(warnings),
                expected["warnings"],
                "{case}: floating warnings"
            );
            assert_eq!(
                json!(calls),
                expected["calls"],
                "{case}: floating callbacks"
            );
            let mut shape = actual.shape().to_vec();
            if actual.dtype().is_complex() {
                assert_eq!(shape.pop(), Some(2));
            }
            assert_eq!(
                json!(shape),
                expected["shape"],
                "{case}: logical destination shape"
            );
            assert_eq!(
                dtype_name(actual.dtype()),
                expected["dtype"],
                "{case}: destination dtype"
            );
            if actual.dtype().is_integer() || actual.dtype() == DType::Bool {
                let storage = actual.to_storage().unwrap();
                let integers = (0..storage.len())
                    .map(|index| storage.get_i128(index).unwrap().to_string())
                    .collect::<Vec<_>>();
                assert_eq!(
                    json!(integers),
                    expected["integers"],
                    "{case}: exact integer destination"
                );
            }
            let bits = expected["bits"].as_array().unwrap();
            assert_eq!(
                actual.values().len(),
                bits.len(),
                "{case}: stored components"
            );
            for (index, (&value, bits)) in actual.values().iter().zip(bits).enumerate() {
                let expected = f64::from_bits(bits.as_u64().unwrap());
                if expected.is_nan() {
                    assert!(value.is_nan(), "{case}: component {index} expected NaN");
                } else {
                    assert_eq!(
                        value.to_bits(),
                        expected.to_bits(),
                        "{case}: component {index}, got {value}, expected {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn copyto_complex_broadcasts_whole_elements_and_logical_masks() {
    let shapes: &[(&[usize], &[usize])] = &[
        (&[], &[]),
        (&[], &[1]),
        (&[], &[1, 1]),
        (&[3], &[]),
        (&[3], &[1]),
        (&[3], &[3]),
        (&[3], &[1, 3]),
        (&[2, 3], &[]),
        (&[2, 3], &[3]),
        (&[2, 3], &[2, 1]),
        (&[2, 3], &[2, 3]),
        (&[2, 3], &[1, 2, 3]),
        (&[2, 1, 3], &[3]),
        (&[2, 1, 3], &[2, 1, 1]),
        (&[0, 3], &[]),
        (&[0, 3], &[3]),
        (&[0, 3], &[0, 3]),
        (&[2, 0], &[]),
        (&[2, 0], &[2, 1]),
        (&[2, 0], &[2, 0]),
    ];
    let mut batch = OracleBatch::default();
    for target in [DType::Complex64, DType::Complex128] {
        for source in NUMERIC {
            for &(dst_shape, src_shape) in shapes {
                let dst = destination(dst_shape, target);
                let src = sample(src_shape, source, 1);
                let masks = [
                    mask(&[], &[true]),
                    mask(&[], &[false]),
                    mask(dst_shape, &[true, false, false, true]),
                    mask(
                        if dst_shape.is_empty() {
                            &[]
                        } else {
                            &dst_shape[dst_shape.len() - 1..]
                        },
                        &[false, true],
                    ),
                ];
                batch.push(&dst, &src, None, None);
                for where_mask in &masks {
                    batch.push(&dst, &src, Some(where_mask), None);
                }
            }
        }
    }
    for rank in [33, 64] {
        let shape = vec![1; rank];
        batch.push(
            &destination(&shape, DType::Complex128),
            &sample(&shape, DType::Complex64, 2),
            Some(&mask(&shape, &[true])),
            None,
        );
    }
    assert_eq!(batch.cases.len(), 2_802);
    batch.assert_matches_numpy();
}

#[test]
fn copyto_numeric_casting_rules_and_values_match_numpy() {
    let mut batch = OracleBatch::default();
    for target in NUMERIC {
        for source in NUMERIC {
            for casting in ["no", "equiv", "safe", "same_kind", "unsafe"] {
                let dst = destination(&[9], target);
                let src = sample(&[9], source, 0);
                batch.push(&dst, &src, None, Some(casting));
                batch.push(
                    &dst,
                    &src,
                    Some(&mask(&[9], &[true, false, true])),
                    Some(casting),
                );
            }
        }
    }
    assert_eq!(batch.cases.len(), 1_960);
    batch.assert_matches_numpy();
}

#[test]
fn copyto_validation_order_and_source_singletons_match_numpy() {
    let mut batch = OracleBatch::default();
    let destinations: &[&[usize]] = &[&[], &[1], &[2, 3], &[0, 3], &[2, 0]];
    let sources: &[&[usize]] = &[
        &[],
        &[1],
        &[1, 1],
        &[3],
        &[1, 3],
        &[2, 1],
        &[2, 3],
        &[1, 2, 3],
        &[2, 2, 3],
        &[0, 3],
        &[2, 0],
    ];
    let masks: &[&[usize]] = &[
        &[],
        &[1],
        &[1, 1],
        &[3],
        &[1, 3],
        &[2, 1],
        &[2, 3],
        &[1, 2, 3],
        &[0, 3],
        &[2, 0],
    ];
    for dst_shape in destinations {
        for src_shape in sources {
            for mask_shape in masks {
                batch.push(
                    &destination(dst_shape, DType::Complex64),
                    &sample(src_shape, DType::Complex128, 0),
                    Some(&mask(mask_shape, &[false, true])),
                    Some("unsafe"),
                );
            }
        }
    }
    for mask_dtype in NUMERIC {
        for casting in ["invalid", "same_kind", "unsafe"] {
            batch.push(
                &destination(&[2, 3], DType::I64),
                &sample(&[4], DType::Complex128, 0),
                Some(&sample(&[5], mask_dtype, 0)),
                Some(casting),
            );
        }
    }
    assert_eq!(batch.cases.len(), 592);
    batch.assert_matches_numpy();
}

#[test]
fn copyto_uint32_casts_preserve_broadcast_spans_and_true_runs() {
    let mut batch = OracleBatch::default();
    let shapes: &[&[usize]] = &[&[], &[1], &[16], &[1, 16], &[2, 1], &[2, 16]];
    let masks = [
        mask(&[], &[true]),
        mask(&[], &[false]),
        mask(&[2, 16], &[true]),
        mask(&[16], &[true]),
        mask(&[2, 1], &[true]),
        mask(&[2, 16], &[true, false]),
        mask(
            &[2, 16],
            &[true, true, true, true, true, true, true, true, false],
        ),
        mask(
            &[2, 16],
            &[false, true, true, true, true, true, true, false],
        ),
    ];
    for source in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        for value in [
            f64::NAN,
            f64::NEG_INFINITY,
            f64::INFINITY,
            -2_147_483_649.0,
            4_294_967_297.0,
        ] {
            for shape in shapes {
                let src = if source.is_complex() {
                    complex(shape, &vec![(value, f64::NAN); count(shape)], source)
                } else {
                    UFuncArray::from_storage(
                        shape.to_vec(),
                        ArrayStorage::F64(vec![value; count(shape)])
                            .cast_to(source)
                            .unwrap(),
                    )
                    .unwrap()
                };
                let dst = destination(&[2, 16], DType::U32);
                batch.push(&dst, &src, None, Some("unsafe"));
                for where_mask in &masks {
                    batch.push(&dst, &src, Some(where_mask), Some("unsafe"));
                }
            }
        }
    }
    assert_eq!(batch.cases.len(), 1_350);
    batch.assert_matches_numpy();
}

#[test]
fn copyto_cast_status_and_partial_mutation_match_numpy() {
    let mut batch = OracleBatch::default();
    for source in [DType::F64, DType::Complex64, DType::Complex128] {
        for target in [
            DType::F16,
            DType::F32,
            DType::F64,
            DType::Complex64,
            DType::Complex128,
            DType::I8,
            DType::I64,
            DType::U32,
            DType::U64,
            DType::Bool,
        ] {
            for &(real, imaginary) in &[
                (1.0e100, 1.0e-50),
                (f64::NAN, 3.0),
                (3.0, f64::NAN),
                (0.0, 1.0e-50),
                (1.5, -2.5),
            ] {
                for src_shape in [&[][..], &[1][..], &[2][..]] {
                    let src = if source.is_complex() {
                        complex(
                            src_shape,
                            &vec![(real, imaginary); count(src_shape)],
                            source,
                        )
                    } else {
                        UFuncArray::new(src_shape.to_vec(), vec![real; count(src_shape)], source)
                            .unwrap()
                    };
                    let dst = destination(&[2], target);
                    let masks = [
                        mask(&[], &[false]),
                        mask(&[], &[true]),
                        mask(&[2], &[true, false]),
                        mask(&[3], &[false]),
                    ];
                    for mode in [
                        FloatErrorMode::Ignore,
                        FloatErrorMode::Warn,
                        FloatErrorMode::Raise,
                        FloatErrorMode::Call,
                    ] {
                        for where_mask in &masks {
                            batch.push_policy(&dst, &src, Some(where_mask), Some("unsafe"), mode);
                        }
                    }
                }
            }
        }
    }
    // Empty and one-element destinations never take NumPy's scalar pre-cast shortcut.
    for dst_shape in [&[][..], &[1][..], &[0][..], &[2, 0][..]] {
        for truth in [false, true] {
            for mode in [
                FloatErrorMode::Warn,
                FloatErrorMode::Raise,
                FloatErrorMode::Call,
            ] {
                batch.push_policy(
                    &destination(dst_shape, DType::Complex64),
                    &complex(&[], &[(1.0e100, 1.0e-50)], DType::Complex128),
                    Some(&mask(&[], &[truth])),
                    Some("unsafe"),
                    mode,
                );
            }
        }
    }
    assert_eq!(batch.cases.len(), 7_224);
    batch.assert_matches_numpy();
}

#[test]
fn copyto_preserves_exact_wide_integers_and_source_precision() {
    let mut batch = OracleBatch::default();
    let signed = UFuncArray::from_storage(
        vec![4],
        ArrayStorage::I64(vec![
            i64::MIN,
            i64::MAX,
            (1_i64 << 62) + (1_i64 << 38) + 1,
            (1_i64 << 53) + 3,
        ]),
    )
    .unwrap();
    let unsigned = UFuncArray::from_storage(
        vec![4],
        ArrayStorage::U64(vec![
            u64::MAX,
            (1_u64 << 63) + (1_u64 << 39) + 1,
            (1_u64 << 53) + 1,
            0,
        ]),
    )
    .unwrap();
    for src in [&signed, &unsigned] {
        for target in NUMERIC {
            batch.push(
                &destination(&[2, 4], target),
                src,
                Some(&mask(&[2, 1], &[true, false])),
                Some("unsafe"),
            );
        }
    }
    for dst in [&signed, &unsigned] {
        for source in [DType::Complex64, DType::Complex128, DType::F64] {
            batch.push(
                dst,
                &sample(&[4], source, 0),
                Some(&mask(&[4], &[false, true])),
                Some("unsafe"),
            );
        }
    }
    let raw_complex64 = complex(
        &[4],
        &[
            (1.0 + 2.0_f64.powi(-24), -1.0 - 2.0_f64.powi(-24)),
            (0.0, 1.0e-50),
            (1.0e100, 3.0),
            (-0.0, -0.0),
        ],
        DType::Complex64,
    );
    for target in NUMERIC {
        batch.push(
            &destination(&[4], target),
            &raw_complex64,
            None,
            Some("unsafe"),
        );
    }
    assert_eq!(batch.cases.len(), 48);
    batch.assert_matches_numpy();
}

#[test]
fn copyto_validates_complex_layout_before_empty_or_masked_writes() {
    let good = complex(&[], &[(1.0, 2.0)], DType::Complex128);
    let never = mask(&[], &[false]);
    for shape in [vec![], vec![3], vec![0, 3]] {
        let malformed =
            UFuncArray::new(shape.clone(), vec![0.0; count(&shape)], DType::Complex128).unwrap();
        let mut dst = good.clone();
        let before = dst.values().to_vec();
        assert!(
            dst.copyto(&malformed, Some(&never), None)
                .unwrap_err()
                .to_string()
                .contains("interleaved")
        );
        assert_eq!(dst.values(), before);
        let mut dst = malformed;
        assert!(
            dst.copyto(&good, Some(&never), None)
                .unwrap_err()
                .to_string()
                .contains("interleaved")
        );
    }
    let mut huge_empty = UFuncArray::new(
        vec![0, usize::MAX, usize::MAX, 2],
        Vec::new(),
        DType::Complex128,
    )
    .unwrap();
    huge_empty.copyto(&good, None, None).unwrap();
    huge_empty.copyto(&good, Some(&never), None).unwrap();
    assert_eq!(huge_empty.shape(), &[0, usize::MAX, usize::MAX, 2]);
    assert!(huge_empty.values().is_empty());
}
