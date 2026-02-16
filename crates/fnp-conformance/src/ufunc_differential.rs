#![forbid(unsafe_code)]

use fnp_dtype::DType;
use fnp_ufunc::{BinaryOp, UFuncArray};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const PY_CAPTURE_SCRIPT: &str = r#"
import json
import importlib
import sys
import math

input_path = sys.argv[1]
output_path = sys.argv[2]
legacy_root = sys.argv[3]

with open(input_path, 'r', encoding='utf-8') as fh:
    cases = json.load(fh)

oracle_source = 'legacy'
np = None

try:
    if 'numpy' in sys.modules:
        del sys.modules['numpy']
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)
    np = importlib.import_module('numpy')
    _ = np.arange(1)
    np_file = str(getattr(np, '__file__', ''))
    if 'legacy_numpy_code' not in np_file:
        oracle_source = 'system'
except Exception:
    try:
        oracle_source = 'system'
        if 'numpy' in sys.modules:
            del sys.modules['numpy']
        if legacy_root in sys.path:
            sys.path.remove(legacy_root)
        np = importlib.import_module('numpy')
        _ = np.arange(1)
    except Exception:
        oracle_source = 'pure_python_fallback'
        np = None

def py_broadcast_shape(lhs, rhs):
    nd = max(len(lhs), len(rhs))
    out = []
    for i in range(nd):
        l = lhs[-1 - i] if i < len(lhs) else 1
        r = rhs[-1 - i] if i < len(rhs) else 1
        if l == r:
            out.append(l)
        elif l == 1:
            out.append(r)
        elif r == 1:
            out.append(l)
        else:
            raise ValueError(f'cannot broadcast {lhs} with {rhs}')
    out.reverse()
    return out

def py_strides(shape):
    strides = [0] * len(shape)
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    return strides

def py_unravel(flat, shape, strides):
    if not shape:
        return []
    rem = flat
    out = [0] * len(shape)
    for i, (dim, stride) in enumerate(zip(shape, strides)):
        if dim == 0 or stride == 0:
            out[i] = 0
        else:
            out[i] = (rem // stride) % dim
            rem = rem % stride
    return out

def py_ravel(multi, strides):
    return sum(idx * stride for idx, stride in zip(multi, strides))

def py_src_index(out_multi, src_shape, src_strides, out_nd):
    if not src_shape:
        return 0
    offset = out_nd - len(src_shape)
    flat = 0
    for axis, (dim, stride) in enumerate(zip(src_shape, src_strides)):
        out_axis = axis + offset
        src_i = 0 if dim == 1 else out_multi[out_axis]
        flat += src_i * stride
    return flat

def py_binary(lhs_vals, lhs_shape, rhs_vals, rhs_shape, op):
    out_shape = py_broadcast_shape(lhs_shape, rhs_shape)
    out_count = math.prod(out_shape) if out_shape else 1
    out_strides = py_strides(out_shape)
    lhs_strides = py_strides(lhs_shape)
    rhs_strides = py_strides(rhs_shape)
    out_vals = []
    for flat in range(out_count):
        multi = py_unravel(flat, out_shape, out_strides)
        li = py_src_index(multi, lhs_shape, lhs_strides, len(out_shape))
        ri = py_src_index(multi, rhs_shape, rhs_strides, len(out_shape))
        l = lhs_vals[li]
        r = rhs_vals[ri]
        if op == 'add':
            out_vals.append(l + r)
        elif op == 'sub':
            out_vals.append(l - r)
        elif op == 'mul':
            out_vals.append(l * r)
        elif op == 'div':
            if r == 0.0:
                if l == 0.0:
                    out_vals.append(float('nan'))
                elif l > 0:
                    out_vals.append(float('inf'))
                else:
                    out_vals.append(float('-inf'))
            else:
                out_vals.append(l / r)
        else:
            raise ValueError(f'unsupported op {op}')
    return out_shape, out_vals

def py_reduced_shape(shape, axis, keepdims):
    if keepdims:
        return [1 if i == axis else dim for i, dim in enumerate(shape)]
    return [dim for i, dim in enumerate(shape) if i != axis]

def py_sum(vals, shape, axis, keepdims):
    if axis is None:
        out_shape = [1] * len(shape) if keepdims else []
        return out_shape, [float(sum(vals))]

    if axis < 0 or axis >= len(shape):
        raise ValueError(f'axis {axis} out of bounds for shape {shape}')

    in_strides = py_strides(shape)
    out_shape = py_reduced_shape(shape, axis, keepdims)
    out_strides = py_strides(out_shape)
    out_count = math.prod(out_shape) if out_shape else 1
    out = [0.0] * out_count

    for flat in range(len(vals)):
        multi = py_unravel(flat, shape, in_strides)
        out_multi = []
        for i, idx in enumerate(multi):
            if i == axis:
                if keepdims:
                    out_multi.append(0)
            else:
                out_multi.append(idx)
        out_flat = py_ravel(out_multi, out_strides) if out_multi else 0
        out[out_flat] += vals[flat]
    return out_shape, out

def normalize_dtype_name(name):
    aliases = {
        'f64': 'float64',
        'f32': 'float32',
        'i64': 'int64',
        'i32': 'int32',
        'u64': 'uint64',
        'u32': 'uint32',
        'bool': 'bool_',
    }
    if name is None:
        return 'float64'
    key = str(name).strip().lower()
    return aliases.get(key, name)

results = []

for case in cases:
    case_id = case['id']
    op = case['op']
    try:
        if np is not None:
            lhs_dtype = normalize_dtype_name(case.get('lhs_dtype', 'float64'))
            lhs = np.array(case['lhs_values'], dtype=lhs_dtype).reshape(tuple(case['lhs_shape']))

            if op in ('add', 'sub', 'mul', 'div'):
                rhs_dtype = normalize_dtype_name(case.get('rhs_dtype', 'float64'))
                rhs = np.array(case['rhs_values'], dtype=rhs_dtype).reshape(tuple(case['rhs_shape']))
                if op == 'add':
                    out = lhs + rhs
                elif op == 'sub':
                    out = lhs - rhs
                elif op == 'mul':
                    out = lhs * rhs
                else:
                    out = lhs / rhs
            elif op == 'sum':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                out = lhs.sum(axis=axis, keepdims=keepdims)
            else:
                raise ValueError(f'unsupported op: {op}')

            arr = np.asarray(out)
            values = arr.astype(np.float64, copy=False).reshape(-1).tolist()
            shape = list(arr.shape)
            dtype = str(arr.dtype)
        else:
            lhs_shape = case['lhs_shape']
            lhs_vals = [float(v) for v in case['lhs_values']]
            if op in ('add', 'sub', 'mul', 'div'):
                rhs_shape = case['rhs_shape']
                rhs_vals = [float(v) for v in case['rhs_values']]
                shape, values = py_binary(lhs_vals, lhs_shape, rhs_vals, rhs_shape, op)
                dtype = 'f64'
            elif op == 'sum':
                axis = case.get('axis')
                keepdims = bool(case.get('keepdims', False))
                shape, values = py_sum(lhs_vals, lhs_shape, axis, keepdims)
                dtype = 'f64'
            else:
                raise ValueError(f'unsupported op: {op}')

        results.append({
            'id': case_id,
            'status': 'ok',
            'error': None,
            'shape': shape,
            'values': values,
            'dtype': dtype,
        })
    except Exception as exc:
        results.append({
            'id': case_id,
            'status': 'error',
            'error': str(exc),
            'shape': [],
            'values': [],
            'dtype': 'unknown',
        })

payload = {
    'schema_version': 1,
    'oracle_source': oracle_source,
    'generated_at_unix_ms': 0,
    'cases': results,
}

with open(output_path, 'w', encoding='utf-8') as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UFuncOperation {
    Add,
    Sub,
    Mul,
    Div,
    Sum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UFuncInputCase {
    pub id: String,
    pub op: UFuncOperation,
    pub lhs_shape: Vec<usize>,
    pub lhs_values: Vec<f64>,
    #[serde(default = "default_f64_dtype")]
    pub lhs_dtype: String,
    pub rhs_shape: Option<Vec<usize>>,
    pub rhs_values: Option<Vec<f64>>,
    pub rhs_dtype: Option<String>,
    pub axis: Option<usize>,
    pub keepdims: Option<bool>,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub env_fingerprint: String,
    #[serde(default)]
    pub artifact_refs: Vec<String>,
    #[serde(default)]
    pub reason_code: String,
    #[serde(default)]
    pub expected_reason_code: String,
    #[serde(default)]
    pub expected_error_contains: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UFuncOracleCase {
    pub id: String,
    pub status: String,
    pub error: Option<String>,
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UFuncOracleCapture {
    pub schema_version: u8,
    pub oracle_source: String,
    pub generated_at_unix_ms: u128,
    pub cases: Vec<UFuncOracleCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UFuncDifferentialCaseResult {
    pub id: String,
    pub seed: u64,
    pub mode: String,
    pub env_fingerprint: String,
    pub artifact_refs: Vec<String>,
    pub pass: bool,
    pub max_abs_error: f64,
    pub expected_shape: Vec<usize>,
    pub actual_shape: Vec<usize>,
    pub expected_dtype: String,
    pub actual_dtype: String,
    pub expected_reason_code: String,
    pub actual_reason_code: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UFuncDifferentialReport {
    pub schema_version: u8,
    pub oracle_source: String,
    pub generated_at_unix_ms: u128,
    pub abs_tol: f64,
    pub rel_tol: f64,
    pub total_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub failures: Vec<UFuncDifferentialCaseResult>,
}

fn default_f64_dtype() -> String {
    "f64".to_string()
}

fn normalize_mode(raw: &str) -> String {
    if raw.trim().is_empty() {
        "strict".to_string()
    } else {
        raw.trim().to_string()
    }
}

fn normalize_env_fingerprint(raw: &str) -> String {
    if raw.trim().is_empty() {
        "unknown_env".to_string()
    } else {
        raw.trim().to_string()
    }
}

fn normalize_artifact_refs(mut refs: Vec<String>) -> Vec<String> {
    refs.retain(|entry| !entry.trim().is_empty());
    if refs.is_empty() {
        refs.push("crates/fnp-conformance/fixtures/ufunc_input_cases.json".to_string());
    }
    refs
}

fn classify_reason_code(op: UFuncOperation, detail: &str) -> String {
    let lowered = detail.to_lowercase();

    if lowered.contains("oracle case missing")
        || lowered.contains("unsupported oracle status")
        || lowered.contains("unknown metadata")
    {
        "ufunc_policy_unknown_metadata".to_string()
    } else if lowered.contains("signature")
        || lowered.contains("rhs_shape")
        || lowered.contains("rhs_values")
    {
        "ufunc_signature_parse_failed".to_string()
    } else if lowered.contains("dtype mismatch")
        || lowered.contains("unsupported dtype")
        || lowered.contains("type resolution")
    {
        "ufunc_type_resolution_invalid".to_string()
    } else if matches!(op, UFuncOperation::Sum)
        && (lowered.contains("axis")
            || lowered.contains("keepdims")
            || lowered.contains("reduce sum"))
    {
        "ufunc_reduction_contract_violation".to_string()
    } else if lowered.contains("override") {
        "ufunc_override_precedence_violation".to_string()
    } else {
        "ufunc_dispatch_resolution_failed".to_string()
    }
}

fn resolve_expected_reason_code(input: &UFuncInputCase, fallback: &str) -> String {
    if !input.expected_reason_code.trim().is_empty() {
        input.expected_reason_code.trim().to_string()
    } else if !input.reason_code.trim().is_empty() {
        input.reason_code.trim().to_string()
    } else {
        fallback.to_string()
    }
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn resolve_oracle_python() -> String {
    std::env::var("FNP_ORACLE_PYTHON")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "python3".to_string())
}

pub fn load_input_cases(path: &Path) -> Result<Vec<UFuncInputCase>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("invalid input json {}: {err}", path.display()))
}

pub fn load_oracle_capture(path: &Path) -> Result<UFuncOracleCapture, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("invalid oracle json {}: {err}", path.display()))
}

pub fn write_oracle_capture(path: &Path, capture: &UFuncOracleCapture) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }

    let raw = serde_json::to_string_pretty(capture)
        .map_err(|err| format!("failed to serialize oracle capture: {err}"))?;
    fs::write(path, raw).map_err(|err| format!("failed writing {}: {err}", path.display()))
}

pub fn write_differential_report(
    path: &Path,
    report: &UFuncDifferentialReport,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }

    let raw = serde_json::to_string_pretty(report)
        .map_err(|err| format!("failed to serialize differential report: {err}"))?;
    fs::write(path, raw).map_err(|err| format!("failed writing {}: {err}", path.display()))
}

pub fn capture_numpy_oracle(
    input_path: &Path,
    output_path: &Path,
    legacy_oracle_root: &Path,
) -> Result<UFuncOracleCapture, String> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }

    let python = resolve_oracle_python();
    let output = Command::new(&python)
        .arg("-c")
        .arg(PY_CAPTURE_SCRIPT)
        .arg(input_path)
        .arg(output_path)
        .arg(legacy_oracle_root)
        .output()
        .map_err(|err| format!("failed to invoke oracle python '{}': {err}", python))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "python capture failed interpreter={} status={} stdout={} stderr={}",
            python,
            output.status,
            stdout.trim(),
            stderr.trim()
        ));
    }

    let mut capture = load_oracle_capture(output_path)?;
    capture.generated_at_unix_ms = now_unix_ms();
    write_oracle_capture(output_path, &capture)?;
    Ok(capture)
}

pub fn compare_against_oracle(
    input_path: &Path,
    oracle_path: &Path,
    abs_tol: f64,
    rel_tol: f64,
) -> Result<UFuncDifferentialReport, String> {
    let inputs = load_input_cases(input_path)?;
    let oracle = load_oracle_capture(oracle_path)?;

    let mut oracle_map: BTreeMap<&str, &UFuncOracleCase> = BTreeMap::new();
    for case in &oracle.cases {
        oracle_map.insert(case.id.as_str(), case);
    }

    let mut failures = Vec::new();
    let mut passed = 0usize;

    for input in &inputs {
        let mode = normalize_mode(&input.mode);
        let env_fingerprint = normalize_env_fingerprint(&input.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(input.artifact_refs.clone());

        let oracle_case = match oracle_map.get(input.id.as_str()) {
            Some(case) => *case,
            None => {
                let actual_reason_code = classify_reason_code(input.op, "oracle case missing");
                let expected_reason_code = resolve_expected_reason_code(input, &actual_reason_code);
                failures.push(UFuncDifferentialCaseResult {
                    id: input.id.clone(),
                    seed: input.seed,
                    mode: mode.clone(),
                    env_fingerprint: env_fingerprint.clone(),
                    artifact_refs: artifact_refs.clone(),
                    pass: false,
                    max_abs_error: f64::INFINITY,
                    expected_shape: Vec::new(),
                    actual_shape: Vec::new(),
                    expected_dtype: "missing".to_string(),
                    actual_dtype: "unknown".to_string(),
                    expected_reason_code,
                    actual_reason_code,
                    reason: Some("oracle case missing".to_string()),
                });
                continue;
            }
        };

        let outcome = execute_input_case(input);
        match (oracle_case.status.as_str(), outcome) {
            ("ok", Ok((actual_shape, actual_values, actual_dtype))) => {
                let expected_dtype_canonical = canonical_dtype_name(&oracle_case.dtype);
                let actual_dtype_canonical = canonical_dtype_name(&actual_dtype);
                if expected_dtype_canonical != actual_dtype_canonical {
                    let reason = format!(
                        "dtype mismatch expected={} actual={}",
                        oracle_case.dtype, actual_dtype
                    );
                    let actual_reason_code = classify_reason_code(input.op, &reason);
                    let expected_reason_code =
                        resolve_expected_reason_code(input, "ufunc_type_resolution_invalid");
                    failures.push(UFuncDifferentialCaseResult {
                        id: input.id.clone(),
                        seed: input.seed,
                        mode: mode.clone(),
                        env_fingerprint: env_fingerprint.clone(),
                        artifact_refs: artifact_refs.clone(),
                        pass: false,
                        max_abs_error: f64::INFINITY,
                        expected_shape: oracle_case.shape.clone(),
                        actual_shape,
                        expected_dtype: oracle_case.dtype.clone(),
                        actual_dtype,
                        expected_reason_code,
                        actual_reason_code,
                        reason: Some(reason),
                    });
                    continue;
                }

                let (pass, max_abs_error, reason) = compare_arrays(
                    &oracle_case.shape,
                    &oracle_case.values,
                    &actual_shape,
                    &actual_values,
                    abs_tol,
                    rel_tol,
                );

                if pass {
                    passed += 1;
                } else {
                    let reason_text = reason
                        .clone()
                        .unwrap_or_else(|| "unknown differential mismatch".to_string());
                    let actual_reason_code = classify_reason_code(input.op, &reason_text);
                    let expected_reason_code =
                        resolve_expected_reason_code(input, &actual_reason_code);
                    failures.push(UFuncDifferentialCaseResult {
                        id: input.id.clone(),
                        seed: input.seed,
                        mode: mode.clone(),
                        env_fingerprint: env_fingerprint.clone(),
                        artifact_refs: artifact_refs.clone(),
                        pass,
                        max_abs_error,
                        expected_shape: oracle_case.shape.clone(),
                        actual_shape,
                        expected_dtype: oracle_case.dtype.clone(),
                        actual_dtype,
                        expected_reason_code,
                        actual_reason_code,
                        reason,
                    });
                }
            }
            ("ok", Err(err)) => {
                let detail = format!("execution failed: {err}");
                let actual_reason_code = classify_reason_code(input.op, &detail);
                let expected_reason_code = resolve_expected_reason_code(input, &actual_reason_code);
                failures.push(UFuncDifferentialCaseResult {
                    id: input.id.clone(),
                    seed: input.seed,
                    mode: mode.clone(),
                    env_fingerprint: env_fingerprint.clone(),
                    artifact_refs: artifact_refs.clone(),
                    pass: false,
                    max_abs_error: f64::INFINITY,
                    expected_shape: oracle_case.shape.clone(),
                    actual_shape: Vec::new(),
                    expected_dtype: oracle_case.dtype.clone(),
                    actual_dtype: "error".to_string(),
                    expected_reason_code,
                    actual_reason_code,
                    reason: Some(detail),
                });
            }
            ("error", Ok((actual_shape, _, actual_dtype))) => {
                let oracle_error = oracle_case
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown error".to_string());
                let expected_reason_code = resolve_expected_reason_code(
                    input,
                    &classify_reason_code(input.op, &oracle_error),
                );
                failures.push(UFuncDifferentialCaseResult {
                    id: input.id.clone(),
                    seed: input.seed,
                    mode: mode.clone(),
                    env_fingerprint: env_fingerprint.clone(),
                    artifact_refs: artifact_refs.clone(),
                    pass: false,
                    max_abs_error: f64::INFINITY,
                    expected_shape: oracle_case.shape.clone(),
                    actual_shape,
                    expected_dtype: oracle_case.dtype.clone(),
                    actual_dtype,
                    expected_reason_code,
                    actual_reason_code: "ufunc_dispatch_resolution_failed".to_string(),
                    reason: Some(format!(
                        "oracle expected error '{}' but execution succeeded",
                        oracle_error
                    )),
                });
            }
            ("error", Err(err)) => {
                let oracle_error = oracle_case
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown error".to_string());
                let actual_reason_code = classify_reason_code(input.op, &err);
                let expected_reason_code = resolve_expected_reason_code(
                    input,
                    &classify_reason_code(input.op, &oracle_error),
                );

                let expected_error_contains = input.expected_error_contains.trim().to_lowercase();
                let actual_lower = err.to_lowercase();
                let error_match = if expected_error_contains.is_empty() {
                    true
                } else {
                    actual_lower.contains(&expected_error_contains)
                };
                let reason_match = expected_reason_code == actual_reason_code;

                if error_match && reason_match {
                    passed += 1;
                } else {
                    let detail = if expected_error_contains.is_empty() {
                        format!(
                            "oracle/local error disagreement: oracle='{}' actual='{}' expected_reason_code='{}' actual_reason_code='{}'",
                            oracle_error, err, expected_reason_code, actual_reason_code
                        )
                    } else {
                        format!(
                            "oracle/local error mismatch expected_error_contains='{}' oracle='{}' actual='{}' expected_reason_code='{}' actual_reason_code='{}'",
                            expected_error_contains,
                            oracle_error,
                            err,
                            expected_reason_code,
                            actual_reason_code
                        )
                    };
                    failures.push(UFuncDifferentialCaseResult {
                        id: input.id.clone(),
                        seed: input.seed,
                        mode: mode.clone(),
                        env_fingerprint: env_fingerprint.clone(),
                        artifact_refs: artifact_refs.clone(),
                        pass: false,
                        max_abs_error: f64::INFINITY,
                        expected_shape: oracle_case.shape.clone(),
                        actual_shape: Vec::new(),
                        expected_dtype: oracle_case.dtype.clone(),
                        actual_dtype: "error".to_string(),
                        expected_reason_code,
                        actual_reason_code,
                        reason: Some(detail),
                    });
                }
            }
            (status, _) => {
                let detail = format!("unsupported oracle status: {status}");
                let actual_reason_code = classify_reason_code(input.op, &detail);
                let expected_reason_code = resolve_expected_reason_code(input, &actual_reason_code);
                failures.push(UFuncDifferentialCaseResult {
                    id: input.id.clone(),
                    seed: input.seed,
                    mode,
                    env_fingerprint,
                    artifact_refs,
                    pass: false,
                    max_abs_error: f64::INFINITY,
                    expected_shape: oracle_case.shape.clone(),
                    actual_shape: Vec::new(),
                    expected_dtype: oracle_case.dtype.clone(),
                    actual_dtype: "unknown".to_string(),
                    expected_reason_code,
                    actual_reason_code,
                    reason: Some(detail),
                });
            }
        }
    }

    Ok(UFuncDifferentialReport {
        schema_version: 1,
        oracle_source: oracle.oracle_source,
        // Keep differential artifacts deterministic across repeated local/CI runs
        // when input/oracle fixtures are unchanged.
        generated_at_unix_ms: oracle.generated_at_unix_ms,
        abs_tol,
        rel_tol,
        total_cases: inputs.len(),
        passed_cases: passed,
        failed_cases: failures.len(),
        failures,
    })
}

pub fn execute_input_case(case: &UFuncInputCase) -> Result<(Vec<usize>, Vec<f64>, String), String> {
    let lhs_dtype = parse_dtype(&case.lhs_dtype)?;
    let lhs = UFuncArray::new(case.lhs_shape.clone(), case.lhs_values.clone(), lhs_dtype)
        .map_err(|err| format!("lhs array error: {err}"))?;

    let out = match case.op {
        UFuncOperation::Add | UFuncOperation::Sub | UFuncOperation::Mul | UFuncOperation::Div => {
            let rhs_shape = case
                .rhs_shape
                .clone()
                .ok_or_else(|| "binary op requires rhs_shape".to_string())?;
            let rhs_values = case
                .rhs_values
                .clone()
                .ok_or_else(|| "binary op requires rhs_values".to_string())?;
            let rhs_dtype_name = case.rhs_dtype.as_deref().unwrap_or("f64");
            let rhs_dtype = parse_dtype(rhs_dtype_name)?;
            let rhs = UFuncArray::new(rhs_shape, rhs_values, rhs_dtype)
                .map_err(|err| format!("rhs array error: {err}"))?;

            let op = match case.op {
                UFuncOperation::Add => BinaryOp::Add,
                UFuncOperation::Sub => BinaryOp::Sub,
                UFuncOperation::Mul => BinaryOp::Mul,
                UFuncOperation::Div => BinaryOp::Div,
                UFuncOperation::Sum => unreachable!("handled above"),
            };

            lhs.elementwise_binary(&rhs, op)
                .map_err(|err| format!("binary op error: {err}"))?
        }
        UFuncOperation::Sum => {
            let keepdims = case.keepdims.unwrap_or(false);
            lhs.reduce_sum(case.axis, keepdims)
                .map_err(|err| format!("reduce sum error: {err}"))?
        }
    };

    Ok((
        out.shape().to_vec(),
        out.values().to_vec(),
        out.dtype().name().to_string(),
    ))
}

fn parse_dtype(name: &str) -> Result<DType, String> {
    DType::parse(name).ok_or_else(|| format!("unsupported dtype: {name}"))
}

fn canonical_dtype_name(name: &str) -> String {
    match name.trim().to_ascii_lowercase().as_str() {
        "f64" | "float64" => "float64".to_string(),
        "f32" | "float32" => "float32".to_string(),
        "i64" | "int64" => "int64".to_string(),
        "i32" | "int32" => "int32".to_string(),
        "u64" | "uint64" => "uint64".to_string(),
        "u32" | "uint32" => "uint32".to_string(),
        "bool" | "bool_" => "bool_".to_string(),
        other => other.to_string(),
    }
}

fn compare_arrays(
    expected_shape: &[usize],
    expected_values: &[f64],
    actual_shape: &[usize],
    actual_values: &[f64],
    abs_tol: f64,
    rel_tol: f64,
) -> (bool, f64, Option<String>) {
    if expected_shape != actual_shape {
        return (
            false,
            f64::INFINITY,
            Some(format!(
                "shape mismatch expected={expected_shape:?} actual={actual_shape:?}"
            )),
        );
    }

    if expected_values.len() != actual_values.len() {
        return (
            false,
            f64::INFINITY,
            Some(format!(
                "value length mismatch expected={} actual={}",
                expected_values.len(),
                actual_values.len()
            )),
        );
    }

    let mut max_abs_error = 0.0_f64;

    for (idx, (&expected, &actual)) in expected_values.iter().zip(actual_values).enumerate() {
        let abs_err = (expected - actual).abs();
        if abs_err > max_abs_error {
            max_abs_error = abs_err;
        }

        let threshold = abs_tol + rel_tol * expected.abs();
        if abs_err > threshold {
            return (
                false,
                max_abs_error,
                Some(format!(
                    "value mismatch at index {idx}: expected={expected} actual={actual} abs_err={abs_err} threshold={threshold}"
                )),
            );
        }
    }

    (true, max_abs_error, None)
}

#[cfg(test)]
mod tests {
    use super::{
        UFuncOracleCapture, UFuncOracleCase, compare_against_oracle, load_oracle_capture,
        write_differential_report,
    };
    use std::fs;
    use std::path::PathBuf;

    fn temp_file(name: &str) -> PathBuf {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!("fnp_{name}_{ts}.json"))
    }

    #[test]
    fn write_and_load_oracle_capture_roundtrip() {
        let path = temp_file("oracle_roundtrip");
        let capture = UFuncOracleCapture {
            schema_version: 1,
            oracle_source: "system".to_string(),
            generated_at_unix_ms: 1,
            cases: vec![UFuncOracleCase {
                id: "case1".to_string(),
                status: "ok".to_string(),
                error: None,
                shape: vec![1],
                values: vec![1.0],
                dtype: "float64".to_string(),
            }],
        };

        super::write_oracle_capture(&path, &capture).expect("write capture");
        let loaded = load_oracle_capture(&path).expect("load capture");
        assert_eq!(loaded.cases.len(), 1);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn compare_reports_missing_oracle_case() {
        let input_path = temp_file("input_missing");
        let oracle_path = temp_file("oracle_missing");
        let report_path = temp_file("report_missing");

        fs::write(
            &input_path,
            r#"[{"id":"x","op":"sum","lhs_shape":[1],"lhs_values":[1.0],"lhs_dtype":"f64","rhs_shape":null,"rhs_values":null,"rhs_dtype":null,"axis":null,"keepdims":false}]"#,
        )
        .expect("write input");

        fs::write(
            &oracle_path,
            r#"{"schema_version":1,"oracle_source":"system","generated_at_unix_ms":0,"cases":[]}"#,
        )
        .expect("write oracle");

        let report = compare_against_oracle(&input_path, &oracle_path, 1e-9, 1e-9)
            .expect("comparison should return report");
        assert_eq!(report.failed_cases, 1);

        write_differential_report(&report_path, &report).expect("write report");

        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(oracle_path);
        let _ = fs::remove_file(report_path);
    }
}
