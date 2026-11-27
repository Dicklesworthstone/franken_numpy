//! Differential coverage for accumulator conversion and Array API scan options.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{
    ComplexWarningMode, FloatErrorMode, UFuncArray, UFuncError, set_complex_warning_mode, seterr,
    seterr_state,
};
use serde_json::{Value, json};
use std::process::Command;

#[test]
fn cumulative_options_match_live_numpy() {
    let script = r#"
import json, warnings
import numpy as np

samples = [
    np.array([[127.75, 255.0], [2.9, 3.0]]),
    np.array([2**53 + 1, 2**64 - 1], dtype=np.uint64),
    np.array([2j, complex(3, np.nan), 1-2j]),
    np.empty((2, 0), dtype=np.float32),
    np.array(complex(-0.0, 2.0)),
    np.array([np.nan]),
    np.array([2048.0, 1.0, -2048.0]),
]
cases = [
    (0, 'cumsum', -2, 'int8', False, False),
    (0, 'cumprod', -1, 'uint8', False, False),
    (0, 'cumulative_sum', 0, 'int8', True, False),
    (0, 'cumulative_prod', 1, 'uint8', True, False),
    (0, 'cumulative_sum', None, None, False, False),
    (1, 'cumsum', None, 'uint64', False, False),
    (1, 'cumulative_sum', None, None, True, False),
    (2, 'nancumsum', None, 'int16', False, False),
    (2, 'nancumprod', 0, 'complex64', False, False),
    (2, 'nancumsum', -1, 'bool', False, True),
    (2, 'nancumprod', None, 'bool', False, True),
    (2, 'cumsum', None, 'float64', False, True),
    (2, 'cumsum', 1, 'float64', False, True),
    (3, 'cumulative_prod', 1, None, True, False),
    (3, 'cumulative_sum', 0, 'float16', True, False),
    (4, 'cumulative_prod', -1, 'complex64', True, False),
    (5, 'cumsum', None, 'int32', False, True),
    (5, 'nancumsum', None, 'int32', False, True),
    (6, 'cumsum', None, 'float16', False, False),
]
rows = []
for index, op, axis, dtype, initial, strict in cases:
    kwargs = dict(axis=axis, dtype=dtype)
    if op.startswith('cumulative_'):
        kwargs['include_initial'] = initial
    with warnings.catch_warnings(), np.errstate(all='raise' if strict else 'ignore'):
        warnings.simplefilter('error' if strict else 'ignore', np.exceptions.ComplexWarning)
        try:
            result = getattr(np, op)(samples[index], **kwargs)
            flat = result.ravel()
            if flat.dtype.kind in 'bui':
                values = [str(int(x)) for x in flat]
            else:
                if flat.dtype.kind == 'c':
                    flat = flat.astype(np.complex128).view(np.float64)
                values = flat.astype(np.float64).view(np.uint64).tolist()
            expected = dict(shape=list(result.shape), dtype=str(result.dtype), values=values)
        except (ValueError, IndexError, FloatingPointError, np.exceptions.ComplexWarning) as e:
            expected = dict(error=type(e).__name__)
    rows.append([index, op, axis, dtype, initial, strict, expected])
print(json.dumps(rows))
"#;
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = Command::new(python).args(["-c", script]).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    let samples = [
        UFuncArray::new(vec![2, 2], vec![127.75, 255.0, 2.9, 3.0], DType::F64).unwrap(),
        UFuncArray::from_storage(
            vec![2],
            ArrayStorage::U64(vec![(1u64 << 53) + 1, u64::MAX]),
        )
        .unwrap(),
        UFuncArray::new(
            vec![3, 2],
            vec![0.0, 2.0, 3.0, f64::NAN, 1.0, -2.0],
            DType::Complex128,
        )
        .unwrap(),
        UFuncArray::new(vec![2, 0], vec![], DType::F32).unwrap(),
        UFuncArray::new(vec![2], vec![-0.0, 2.0], DType::Complex128).unwrap(),
        UFuncArray::new(vec![1], vec![f64::NAN], DType::F64).unwrap(),
        UFuncArray::new(vec![3], vec![2048.0, 1.0, -2048.0], DType::F64).unwrap(),
    ];
    for row in rows {
        let input = &samples[row[0].as_u64().unwrap() as usize];
        let axis = row[2].as_i64().map(|axis| axis as isize);
        let dtype = row[3].as_str().map(|name| DType::parse(name).unwrap());
        let initial = row[4].as_bool().unwrap();
        let strict = row[5].as_bool().unwrap();
        let old_warning = set_complex_warning_mode(if strict {
            ComplexWarningMode::Raise
        } else {
            ComplexWarningMode::Ignore
        });
        let mode = if strict {
            FloatErrorMode::Raise
        } else {
            FloatErrorMode::Ignore
        };
        let old_errors = seterr(Some(mode), None, None, None, None);
        let actual = match row[1].as_str().unwrap() {
            "cumsum" => input.cumsum_with_dtype(axis, dtype.unwrap()),
            "cumprod" => input.cumprod_with_dtype(axis, dtype.unwrap()),
            "nancumsum" => input.nancumsum_with_dtype(axis, dtype.unwrap()),
            "nancumprod" => input.nancumprod_with_dtype(axis, dtype.unwrap()),
            "cumulative_sum" => input.cumulative_sum_with_options(axis, dtype, initial),
            "cumulative_prod" => input.cumulative_prod_with_options(axis, dtype, initial),
            op => panic!("unknown oracle operation {op}"),
        };
        let _ = seterr_state(old_errors);
        let _ = set_complex_warning_mode(old_warning);

        let expected = &row[6];
        if let Some(error) = expected["error"].as_str() {
            let got = match actual.unwrap_err() {
                UFuncError::AxisOutOfBounds { .. } => "AxisError",
                UFuncError::ComplexWarning => "ComplexWarning",
                UFuncError::FloatingPoint { .. } => "FloatingPointError",
                UFuncError::Msg(_) => "ValueError",
                e => panic!("{row}: unexpected error {e}"),
            };
            assert_eq!(got, error, "{row}");
            continue;
        }
        let result = actual.unwrap_or_else(|e| panic!("{row}: {e}"));
        let dtype = DType::parse(expected["dtype"].as_str().unwrap()).unwrap();
        assert_eq!(result.dtype(), dtype, "{row}");
        let mut shape = result.shape().to_vec();
        if dtype.is_complex() {
            assert_eq!(shape.pop(), Some(2), "{row}");
        }
        assert_eq!(json!(shape), expected["shape"], "{row}");
        let values = if dtype.is_integer() || dtype == DType::Bool {
            let storage = result.to_storage().unwrap();
            json!((0..storage.len())
                .map(|i| storage.get_i128(i).unwrap().to_string())
                .collect::<Vec<_>>())
        } else {
            json!(result.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>())
        };
        assert_eq!(values, expected["values"], "{row}");
    }
}
