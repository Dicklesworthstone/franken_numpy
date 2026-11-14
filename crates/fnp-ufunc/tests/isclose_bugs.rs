use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{FloatErrorMode, UFuncArray, UFuncError, errstate};
use serde_json::{Value, json};
use std::process::{Command, Stdio};

#[test]
fn test_isclose_symmetry_and_special_values() {
    let a = UFuncArray::scalar(1e10, DType::F64);
    let b = UFuncArray::scalar(1e10 + 1.0, DType::F64);

    // Asymmetric formula: |a - b| <= atol + rtol * |b|
    // With rtol=1e-9, atol=0:
    // |1e10 - (1e10 + 1)| = 1.0
    // atol + rtol * |1e10 + 1| = 0 + 1e-9 * 10000000001 = 10.000000001
    // 1.0 <= 10.000000001 -> TRUE

    // BUT if we swap them:
    // |(1e10 + 1) - 1e10| = 1.0
    // atol + rtol * |1e10| = 0 + 1e-9 * 1e10 = 10.0
    // 1.0 <= 10.0 -> TRUE (still true here, let's pick values where it flips)

    let rtol = 1e-11;
    // |1e10 - (1e10 + 1)| = 1.0
    // 1e-11 * 1e10 = 0.1
    // 1.0 <= 0.1 -> FALSE

    assert_eq!(
        a.isclose(&b, rtol, 0.0).unwrap().values()[0],
        0.0,
        "Should be false"
    );

    // NaN handling: NumPy isclose(NaN, NaN) is FALSE by default
    let nan = UFuncArray::scalar(f64::NAN, DType::F64);
    assert_eq!(
        nan.isclose(&nan, 1e-5, 1e-8).unwrap().values()[0],
        0.0,
        "isclose(NaN, NaN) should be false"
    );

    // Inf handling: NumPy isclose(Inf, Inf) is TRUE
    let inf = UFuncArray::scalar(f64::INFINITY, DType::F64);
    assert_eq!(
        inf.isclose(&inf, 1e-5, 1e-8).unwrap().values()[0],
        1.0,
        "isclose(Inf, Inf) should be true"
    );
}

#[test]
fn test_isclose_broadcasting() {
    let a = UFuncArray::new(vec![2], vec![1.0, 2.0], DType::F64).unwrap();
    let b = UFuncArray::scalar(1.0, DType::F64);
    // This should work via broadcasting but currently fails with "size mismatch"
    let res = a.isclose(&b, 1e-5, 1e-8);
    assert!(
        res.is_ok(),
        "isclose should support broadcasting: {:?}",
        res.err()
    );
}

fn complex(shape: &[usize], values: &[(f64, f64)], dtype: DType) -> UFuncArray {
    let mut physical_shape = shape.to_vec();
    physical_shape.push(2);
    UFuncArray::new(
        physical_shape,
        values.iter().flat_map(|&(re, im)| [re, im]).collect(),
        dtype,
    )
    .unwrap()
}

#[test]
fn complex_isclose_uses_joint_magnitude_and_logical_broadcasting() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let a = complex(&[2, 1], &[(0.0, 0.0), (3.0, 4.0)], dtype);
        let b = complex(&[3], &[(0.0, 0.0), (0.08, 0.08), (3.06, 4.06)], dtype);
        let close = a.isclose(&b, 0.0, 0.1).unwrap();
        assert_eq!(close.shape(), &[2, 3]);
        assert_eq!(close.dtype(), DType::Bool);
        assert_eq!(close.values(), &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(!a.allclose(&b, 0.0, 0.1).unwrap());

        // Each component is within atol, but the complex distance is not.
        let zero = complex(&[], &[(0.0, 0.0)], dtype);
        let diagonal = complex(&[], &[(0.08, 0.08)], dtype);
        assert!(!zero.allclose(&diagonal, 0.0, 0.1).unwrap());
        assert_eq!(zero.isclose(&diagonal, 0.0, 0.1).unwrap().shape(), &[]);

        let real = UFuncArray::new(vec![3, 1], vec![0.0, 1.0, 2.0], DType::F64).unwrap();
        let scalar = complex(&[], &[(1.0, 0.0)], dtype);
        let close = real.isclose(&scalar, 0.0, 0.0).unwrap();
        assert_eq!(close.shape(), &[3, 1]);
        assert_eq!(close.values(), &[0.0, 1.0, 0.0]);
    }
}

#[test]
fn isclose_equality_and_whole_complex_nan_semantics() {
    let a = UFuncArray::new(
        vec![4],
        vec![1.0, -0.0, f64::INFINITY, f64::NEG_INFINITY],
        DType::F64,
    )
    .unwrap();
    for (rtol, atol) in [(-1.0, 0.0), (0.0, -1.0), (f64::NAN, 0.0)] {
        assert_eq!(a.isclose(&a, rtol, atol).unwrap().values(), &[1.0; 4]);
        assert!(a.allclose(&a, rtol, atol).unwrap());
    }
    let a = complex(
        &[4],
        &[
            (f64::NAN, 0.0),
            (1.0, f64::NAN),
            (f64::INFINITY, 2.0),
            (1.0, 2.0),
        ],
        DType::Complex128,
    );
    let b = complex(
        &[4],
        &[
            (9.0, f64::NAN),
            (f64::NAN, 7.0),
            (f64::INFINITY, 3.0),
            (1.0, 2.0),
        ],
        DType::Complex128,
    );
    assert_eq!(
        a.isclose_equal_nan(&b, 0.0, 0.0, true).unwrap().values(),
        &[1.0, 1.0, 0.0, 1.0]
    );
    assert_eq!(
        a.isclose(&b, 0.0, 0.0).unwrap().values(),
        &[0.0, 0.0, 0.0, 1.0]
    );

    let infinite = UFuncArray::scalar(f64::INFINITY, DType::F64);
    let finite = UFuncArray::scalar(1.0, DType::F64);
    assert!(infinite.allclose(&finite, f64::INFINITY, 0.0).unwrap());
    assert!(!finite.allclose(&infinite, f64::INFINITY, 0.0).unwrap());
}

#[test]
fn isclose_validates_layout_broadcasting_and_empty_shapes() {
    let malformed = UFuncArray::new(vec![3], vec![1.0; 3], DType::Complex128).unwrap();
    let scalar = UFuncArray::scalar(1.0, DType::F64);
    assert!(malformed.isclose(&scalar, 1e-5, 1e-8).is_err());
    assert!(scalar.isclose(&malformed, 1e-5, 1e-8).is_err());
    let a = complex(&[2], &[(1.0, 0.0); 2], DType::Complex128);
    let b = UFuncArray::new(vec![3], vec![1.0; 3], DType::F64).unwrap();
    assert!(a.isclose(&b, 1e-5, 1e-8).is_err());
    assert!(a.allclose(&b, 1e-5, 1e-8).is_err());

    // An empty output must not compute overflowing strides or allocate its
    // (unreachable) nonempty trailing dimensions.
    let empty = complex(&[0, usize::MAX, 4], &[], DType::Complex64);
    let close = empty.isclose(&scalar, 0.0, 0.0).unwrap();
    assert_eq!(close.shape(), &[0, usize::MAX, 4]);
    assert!(close.values().is_empty());
    assert!(empty.allclose(&scalar, 0.0, 0.0).unwrap());
}

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
        _ => panic!("isclose fixtures require numeric dtypes"),
    }
}

fn sample(shape: &[usize], dtype: DType, offset: usize) -> UFuncArray {
    let count: usize = shape.iter().product();
    let values = [
        (0.0, -0.0),
        (1.0 + 2.0_f64.powi(-24), 1.0e-50),
        (0.1, 0.2),
        (1.0000100135803223, -0.5),
        (1.0e-50, -1.0e-50),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (f64::INFINITY, 2.0),
        (f64::NEG_INFINITY, f64::INFINITY),
    ];
    if dtype.is_complex() {
        let parts = (0..count)
            .map(|index| values[(index + offset) % values.len()])
            .collect::<Vec<_>>();
        return complex(shape, &parts, dtype);
    }
    if dtype.is_integer() || dtype == DType::Bool {
        let storage = if dtype == DType::I64 {
            ArrayStorage::I64(
                (0..count)
                    .map(|index| {
                        [0, 1, -1, (1_i64 << 53) + 1, i64::MIN, i64::MAX][(index + offset) % 6]
                    })
                    .collect(),
            )
        } else if dtype == DType::U64 {
            ArrayStorage::U64(
                (0..count)
                    .map(|index| {
                        [0, 1, 2, (1_u64 << 53) + 1, 1_u64 << 63, u64::MAX][(index + offset) % 6]
                    })
                    .collect(),
            )
        } else {
            ArrayStorage::I32(
                (0..count)
                    .map(|index| [0, 1, 2, 3, 127, 53][(index + offset) % 6])
                    .collect(),
            )
            .cast_to(dtype)
            .unwrap()
        };
        return UFuncArray::from_storage(shape.to_vec(), storage).unwrap();
    }
    UFuncArray::new(
        shape.to_vec(),
        (0..count)
            .map(|index| values[(index + offset) % values.len()].0)
            .collect(),
        dtype,
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
                .chunks_exact(2)
                .map(|pair| { [pair[0].to_string(), pair[1].to_string()] })
                .collect::<Vec<_>>()
        )
    } else if array.dtype().is_integer() || array.dtype() == DType::Bool {
        let storage = array.to_storage().unwrap();
        json!(
            (0..storage.len())
                .map(|index| { storage.get_i128(index).unwrap().to_string() })
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
    actual: Vec<(Result<UFuncArray, UFuncError>, Result<bool, UFuncError>)>,
}

impl OracleBatch {
    fn push(&mut self, a: &UFuncArray, b: &UFuncArray, rtol: f64, atol: f64, equal_nan: bool) {
        self.cases.push(json!({
            "a": oracle_input(a), "b": oracle_input(b), "rtol": rtol.to_string(),
            "atol": atol.to_string(), "equal_nan": equal_nan,
        }));
        let _guard = errstate(Some(FloatErrorMode::Ignore), None, None, None, None);
        self.actual.push((
            a.isclose_equal_nan(b, rtol, atol, equal_nan),
            a.allclose_equal_nan(b, rtol, atol, equal_nan),
        ));
    }

    fn assert_matches_numpy(self) {
        let script = r#"
import json, sys, warnings
import numpy as np

def load(item):
    dtype = np.dtype(item['dtype'])
    if dtype.kind == 'c':
        array = np.empty(len(item['values']), dtype=dtype)
        array.real = [float(value[0]) for value in item['values']]
        array.imag = [float(value[1]) for value in item['values']]
    else:
        convert = int if dtype.kind in 'biu' else float
        array = np.array([convert(value) for value in item['values']], dtype=dtype)
    return array.reshape(item['shape'])

rows = []
with warnings.catch_warnings(), np.errstate(all='ignore'):
    warnings.simplefilter('ignore')
    for case in json.load(sys.stdin):
        a, b = load(case['a']), load(case['b'])
        kwargs = dict(rtol=float(case['rtol']), atol=float(case['atol']), equal_nan=case['equal_nan'])
        try:
            result = np.asarray(np.isclose(a, b, **kwargs))
            rows.append(dict(shape=list(result.shape), dtype=str(result.dtype),
                             values=result.ravel().tolist(), allclose=bool(np.allclose(a, b, **kwargs))))
        except (TypeError, ValueError) as error:
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
            .expect("launch live NumPy isclose oracle");
        let mut stdin = child.stdin.take().unwrap();
        serde_json::to_writer(&mut stdin, &self.cases).unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "NumPy isclose oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(self.actual.len(), expected.len());
        for ((case, (close, allclose)), expected) in self.cases.iter().zip(self.actual).zip(expected)
        {
            if expected.get("error").is_some() {
                assert!(close.is_err(), "{case}: expected {expected}, got {close:?}");
                assert!(
                    allclose.is_err(),
                    "{case}: expected {expected}, got {allclose:?}"
                );
                continue;
            }
            let close = close.unwrap_or_else(|error| panic!("{case}: {error}"));
            assert_eq!(close.dtype(), DType::Bool, "{case}: dtype");
            assert_eq!(expected["dtype"], "bool", "{case}: oracle dtype");
            assert_eq!(
                json!(close.shape()),
                expected["shape"],
                "{case}: logical shape"
            );
            assert!(close.values().iter().all(|&value| value == 0.0 || value == 1.0));
            assert_eq!(
                json!(
                    close
                        .values()
                        .iter()
                        .map(|&value| value != 0.0)
                        .collect::<Vec<_>>()
                ),
                expected["values"],
                "{case}: isclose values"
            );
            assert_eq!(
                json!(allclose.unwrap()),
                expected["allclose"],
                "{case}: allclose result"
            );
        }
    }
}

#[test]
fn isclose_all_numeric_dtype_pairs_match_numpy() {
    let mut batch = OracleBatch::default();
    for a_dtype in NUMERIC_DTYPES {
        for b_dtype in NUMERIC_DTYPES {
            let a = sample(&[9, 1], a_dtype, 0);
            let b = sample(&[9], b_dtype, 1);
            for (rtol, atol) in [
                (0.0, 0.0),
                (1e-5, 1e-8),
                (0.0, 0.1),
                (-1.0, 0.0),
                (f64::NAN, 0.0),
                (f64::INFINITY, 0.0),
                (0.0, f64::INFINITY),
            ] {
                for equal_nan in [false, true] {
                    batch.push(&a, &b, rtol, atol, equal_nan);
                }
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn isclose_narrow_precision_boundaries_match_numpy() {
    let mut batch = OracleBatch::default();
    for dtype in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        let values = [
            0.0,
            1.0e-50,
            2.0_f64.powi(-25),
            0.1,
            1.0,
            1.0 + 2.0_f64.powi(-24),
            1.0000100135803223,
            1.0 + 2.0_f64.powi(-11),
            65504.0,
            65520.0,
        ];
        let a = if dtype.is_complex() {
            complex(
                &[values.len(), 1],
                &values
                    .iter()
                    .map(|&value| (value, value * 0.5))
                    .collect::<Vec<_>>(),
                dtype,
            )
        } else {
            UFuncArray::new(vec![values.len(), 1], values.to_vec(), dtype).unwrap()
        };
        let b = if dtype.is_complex() {
            complex(
                &[values.len()],
                &values
                    .iter()
                    .map(|&value| (value, -value * 0.5))
                    .collect::<Vec<_>>(),
                dtype,
            )
        } else {
            UFuncArray::new(vec![values.len()], values.to_vec(), dtype).unwrap()
        };
        for (rtol, atol) in [
            (1e-5, 1e-8),
            (1e-12, 0.0),
            (0.0, 2.0_f64.powi(-25)),
            (0.0, 2.0_f64.powi(-24)),
            (0.0, 2.0_f64.powi(-11)),
            (0.5, 0.0),
            (1.0, 0.0),
            (1e300, 0.0),
            (0.0, 1e300),
        ] {
            batch.push(&a, &b, rtol, atol, false);
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn isclose_scalar_empty_and_incompatible_shapes_match_numpy() {
    let mut batch = OracleBatch::default();
    for a_dtype in [DType::F32, DType::F64, DType::Complex64, DType::Complex128] {
        for b_dtype in [
            DType::Bool,
            DType::I64,
            DType::U64,
            DType::Complex64,
            DType::Complex128,
        ] {
            for (a_shape, b_shape) in [
                (vec![], vec![]),
                (vec![3, 1], vec![]),
                (vec![], vec![2, 3]),
                (vec![0, 3], vec![1, 3]),
                (vec![2, 0, 1], vec![3]),
                (vec![2], vec![3]),
            ] {
                let a = sample(&a_shape, a_dtype, 0);
                let b = sample(&b_shape, b_dtype, 1);
                batch.push(&a, &b, 1e-5, 1e-8, true);
            }
        }
    }
    batch.assert_matches_numpy();
}

#[test]
fn complex_isclose_magnitude_boundaries_match_numpy() {
    let parts = [
        (0.9619742, -1.1085297),
        (-1.2602191, 0.47244662),
        (-0.4479426, -0.52783936),
        (0.6395355960945575, -0.35133015596536643),
        (0.02779910453823846, -0.39498460919812356),
        (-1.0742705306194253, 2.384034339655526),
        (1e-38, 1e-38),
        (1e-300, 1e-300),
        (1e30, -1e30),
        (1e300, -1e300),
    ];
    let mut batch = OracleBatch::default();
    for dtype in [DType::Complex64, DType::Complex128] {
        for &(re, im) in &parts {
            let a = complex(&[], &[(re, im)], dtype);
            let zero = complex(&[], &[(0.0, 0.0)], dtype);
            let array = complex(&[1], &[(re, im)], dtype);
            let zero_array = complex(&[1], &[(0.0, 0.0)], dtype);
            // Choose a tight band around the independent libc norm. NumPy's
            // scaled SIMD norm can lie on either side, so the live oracle
            // determines each exact Boolean instead of allowing a ULP budget.
            let tolerances = if dtype == DType::Complex64 {
                let norm = (re as f32).hypot(im as f32);
                [
                    f64::from(norm.next_down()),
                    f64::from(norm),
                    f64::from(norm.next_up()),
                ]
            } else {
                let norm = re.hypot(im);
                [norm.next_down(), norm, norm.next_up()]
            };
            for atol in tolerances {
                batch.push(&a, &zero, 0.0, atol, false);
                batch.push(&zero, &a, 0.0, atol, false);
                batch.push(&array, &zero_array, 0.0, atol, false);
                batch.push(&zero_array, &array, 0.0, atol, false);
            }
        }
    }
    batch.assert_matches_numpy();
}
