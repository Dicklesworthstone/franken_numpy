//! Conformance tests for numpy percentile, quantile, median, ptp against NumPy oracle.
//!
//! Tests percentile, quantile, median, ptp.

use std::process::Command;

fn numpy_oracle(script: &str) -> Result<String, String> {
    let output = Command::new("python3")
        .args(["-c", script])
        .output()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NumPy oracle failed: {stderr}\nScript: {script}"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn fnp_script(body: String) -> String {
    let library_name = format!(
        "{}fnp_python{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let module_path = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join(&library_name)))
        .unwrap_or_else(|| library_name.into());
    let module_literal = format!("{module_path:?}");
    format!(
        "import importlib.util\n\
         import numpy as np\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}

fn indent_python(body: &str) -> String {
    body.lines().map(|line| format!("    {line}\n")).collect()
}

fn outcome_body(body: &str) -> String {
    let indented = indent_python(body);
    r#"import json

def normalize(value):
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [normalize(item) for item in value]}
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if np.isscalar(value):
        scalar_type = type(value).__name__
        scalar_dtype = str(value.dtype) if hasattr(value, "dtype") else None
        scalar_value = value.item() if hasattr(value, "item") else value
        return {
            "kind": "scalar",
            "type": scalar_type,
            "dtype": scalar_dtype,
            "value": scalar_value,
        }
    return {"kind": "object", "type": type(value).__name__, "repr": repr(value)}

try:
__BODY__    payload = {"status": "ok", "result": normalize(result)}
    if "out" in locals():
        payload["out"] = normalize(out)
        payload["result_is_out"] = result is out
    print(json.dumps(payload, sort_keys=True, default=str))
except Exception as exc:
    message = str(exc).splitlines()[0] if str(exc) else ""
    print(json.dumps(
        {"status": "err", "type": type(exc).__name__, "message": message},
        sort_keys=True,
        default=str,
    ))
"#
    .replace("__BODY__", &indented)
}

fn numpy_outcome_script(body: &str) -> String {
    format!(
        "import numpy as np\n\
         MODULE = np\n\
         {}",
        outcome_body(body)
    )
}

fn fnp_outcome_script(body: &str) -> String {
    fnp_script(format!("MODULE = fnp\n{}", outcome_body(body)))
}

#[test]
fn percentile_quantile_median_keyword_outcomes_match_numpy() -> Result<(), String> {
    let cases = [
        (
            "percentile list scalar q",
            "result = MODULE.percentile([1, 2, 3, 4], 50)",
        ),
        (
            "percentile q sequence method keepdims",
            "result = MODULE.percentile(
    np.array([[1, 2, 3], [4, 5, 6]]),
    [25, 75],
    axis=1,
    method='nearest',
    keepdims=True,
)",
        ),
        (
            "percentile out forwarding",
            "out = np.empty((2,), dtype=np.float64)
result = MODULE.percentile(np.array([[1.0, 2.0], [3.0, 4.0]]), 50, axis=0, out=out)",
        ),
        (
            "quantile tuple q sequence axis",
            "result = MODULE.quantile(((1, 2, 3), (4, 5, 6)), [0.25, 0.75], axis=0)",
        ),
        (
            "quantile method fallback",
            "result = MODULE.quantile(np.array([1, 2, 3, 4]), 0.5, method='lower')",
        ),
        (
            "median tuple axis keepdims",
            "result = MODULE.median(((1, 3, 2), (6, 4, 5)), axis=1, keepdims=True)",
        ),
        (
            "median out forwarding",
            "out = np.empty((2,), dtype=np.float64)
result = MODULE.median(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0, out=out)",
        ),
        (
            "median axis error type",
            "result = MODULE.median([1, 2, 3], axis=2)",
        ),
    ];

    for (name, body) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script(body))?;
        let fnp_result = numpy_oracle(&fnp_outcome_script(body))?;

        assert_eq!(
            fnp_result, numpy_result,
            "percentile/quantile/median outcome mismatch for {name}\n\
             numpy: {numpy_result}\nfnp:   {fnp_result}"
        );
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// percentile
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn percentile_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = fnp.percentile(a, 50)
expected = np.percentile(a, 50)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "percentile basic should match numpy");
    Ok(())
}

#[test]
fn percentile_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = fnp.percentile(a, [25, 50, 75])
expected = np.percentile(a, [25, 50, 75])
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "percentile multiple should match numpy"
    );
    Ok(())
}

#[test]
fn percentile_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.percentile(a, 50, axis=0)
expected = np.percentile(a, 50, axis=0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "percentile 2d axis should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// quantile
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn quantile_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = fnp.quantile(a, 0.5)
expected = np.quantile(a, 0.5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "quantile basic should match numpy");
    Ok(())
}

#[test]
fn quantile_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = fnp.quantile(a, [0.25, 0.5, 0.75])
expected = np.quantile(a, [0.25, 0.5, 0.75])
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "quantile multiple should match numpy"
    );
    Ok(())
}

#[test]
fn quantile_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.quantile(a, 0.5, axis=1)
expected = np.quantile(a, 0.5, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "quantile 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// median
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn median_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 3, 2, 5, 4])
result = fnp.median(a)
expected = np.median(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "median basic should match numpy");
    Ok(())
}

#[test]
fn median_even_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
result = fnp.median(a)
expected = np.median(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "median even count should match numpy"
    );
    Ok(())
}

#[test]
fn median_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.median(a, axis=1)
expected = np.median(a, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "median 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ptp (peak-to-peak)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ptp_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
result = fnp.ptp(a)
expected = np.ptp(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp basic should match numpy");
    Ok(())
}

#[test]
fn ptp_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 5, 3], [2, 8, 1]])
result = fnp.ptp(a, axis=1)
expected = np.ptp(a, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp 2d axis should match numpy");
    Ok(())
}

#[test]
fn ptp_2d_all() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 5, 3], [2, 8, 1]])
result = fnp.ptp(a)
expected = np.ptp(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp 2d all should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn percentile_50_equals_median() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 3, 2, 5, 4])
p50 = fnp.percentile(a, 50)
med = fnp.median(a)
print(np.allclose(p50, med))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "percentile 50 should equal median");
    Ok(())
}

#[test]
fn quantile_05_equals_percentile_50() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
q = fnp.quantile(a, 0.5)
p = fnp.percentile(a, 50)
print(np.allclose(q, p))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "quantile 0.5 should equal percentile 50"
    );
    Ok(())
}

#[test]
fn ptp_equals_max_minus_min() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
ptp_val = fnp.ptp(a)
manual = np.max(a) - np.min(a)
print(np.allclose(ptp_val, manual))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp should equal max - min");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn percentile_boundary_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
p0 = fnp.percentile(a, 0)
p100 = fnp.percentile(a, 100)
np_p0 = np.percentile(a, 0)
np_p100 = np.percentile(a, 100)
print(np.allclose(p0, np_p0) and np.allclose(p100, np_p100))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "percentile 0/100 should match numpy");
    Ok(())
}

#[test]
fn quantile_boundary_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
q0 = fnp.quantile(a, 0.0)
q1 = fnp.quantile(a, 1.0)
np_q0 = np.quantile(a, 0.0)
np_q1 = np.quantile(a, 1.0)
print(np.allclose(q0, np_q0) and np.allclose(q1, np_q1))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "quantile 0/1 should match numpy");
    Ok(())
}

#[test]
fn median_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([42.0])
result = fnp.median(a)
expected = np.median(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "median single element should match numpy"
    );
    Ok(())
}

#[test]
fn percentile_nan_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
result = fnp.percentile(a, 50)
expected = np.percentile(a, 50)
# Both should return nan
print(np.isnan(result) == np.isnan(expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "percentile nan handling should match numpy"
    );
    Ok(())
}

#[test]
fn median_nan_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
result = fnp.median(a)
expected = np.median(a)
# Both should return nan
print(np.isnan(result) == np.isnan(expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "median nan handling should match numpy"
    );
    Ok(())
}

#[test]
fn ptp_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([42.0])
result = fnp.ptp(a)
expected = np.ptp(a)
# ptp of single element should be 0
print(np.allclose(result, expected) and result == 0.0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp single element should be 0");
    Ok(())
}

#[test]
fn ptp_nan_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
result = fnp.ptp(a)
expected = np.ptp(a)
# Both should return nan
print(np.isnan(result) == np.isnan(expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ptp nan propagation should match numpy"
    );
    Ok(())
}

#[test]
fn percentile_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
fnp_result = fnp.percentile(a, 50)
np_result = np.percentile(a, 50)
print(type(fnp_result).__name__ == type(np_result).__name__)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim() == "True",
        "percentile scalar return type should match numpy: {result}"
    );
    Ok(())
}
