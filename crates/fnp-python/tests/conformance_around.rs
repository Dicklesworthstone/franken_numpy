//! Conformance tests for numpy around/round against NumPy oracle.

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
fn around_round_keyword_outcomes_match_numpy() -> Result<(), String> {
    let cases = [
        (
            "list decimals",
            "result = MODULE.around([1.25, 2.75, -3.125], decimals=1)",
        ),
        (
            "round scalar decimals",
            "result = MODULE.round(np.float64(1.567), decimals=2)",
        ),
        (
            "around bool fallback dtype",
            "result = MODULE.around(np.array([True, False]))",
        ),
        (
            "around complex fallback",
            "result = MODULE.around(np.array([1.25 + 2.75j, -3.125 - 4.5j]), decimals=1)",
        ),
        (
            "around out forwarding",
            "out = np.empty((3,), dtype=np.float32)
result = MODULE.around(np.array([1.25, 2.75, -3.125]), decimals=1, out=out)",
        ),
        (
            "round out shape error",
            "out = np.empty((2,), dtype=np.float64)
result = MODULE.round(np.array([1.25, 2.75, -3.125]), out=out)",
        ),
    ];

    for (name, body) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script(body))?;
        let fnp_result = numpy_oracle(&fnp_outcome_script(body))?;

        assert_eq!(
            fnp_result, numpy_result,
            "around/round outcome mismatch for {name}\nnumpy: {numpy_result}\nfnp:   {fnp_result}"
        );
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// around basic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn around_default_decimals() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.4, 1.5, 1.6, 2.5, -1.5])
result = fnp.around(a)
expected = np.around(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around default should match numpy");
    Ok(())
}

#[test]
fn around_positive_decimals() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.234, 2.567, 3.891])
result = fnp.around(a, decimals=2)
expected = np.around(a, decimals=2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "around decimals=2 should match numpy"
    );
    Ok(())
}

#[test]
fn around_negative_decimals() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1234.5, 2567.5, 3891.5])
result = fnp.around(a, decimals=-2)
expected = np.around(a, decimals=-2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "around decimals=-2 should match numpy"
    );
    Ok(())
}

#[test]
fn around_integer_input() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4], dtype='int32')
result = fnp.around(a, decimals=2)
expected = np.around(a, decimals=2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "around integer input should match numpy"
    );
    Ok(())
}

#[test]
fn around_2d_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.234, 2.567], [3.891, 4.123]])
result = fnp.around(a, decimals=1)
expected = np.around(a, decimals=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around 2d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// round (alias for around)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn round_matches_around() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.234, 2.567, 3.891])
around_result = fnp.around(a, decimals=2)
round_result = fnp.round(a, decimals=2)
print(np.array_equal(around_result, round_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "round should match around");
    Ok(())
}

#[test]
fn round_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.4, 1.5, 2.5, 3.5])
result = fnp.round(a)
expected = np.round(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "round default should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn around_nan_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.nan, np.inf, -np.inf, 1.5])
result = fnp.around(a)
expected = np.around(a)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around nan/inf should match numpy");
    Ok(())
}

#[test]
fn around_preserves_dtype_float32() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.234, 2.567], dtype='float32')
result = fnp.around(a, decimals=1)
expected = np.around(a, decimals=1)
print(result.dtype == expected.dtype and np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "around should preserve float32 dtype"
    );
    Ok(())
}

#[test]
fn around_scalar_input() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.around(np.array(1.567), decimals=2)
expected = np.around(1.567, decimals=2)
print(np.isclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around scalar should match numpy");
    Ok(())
}

#[test]
fn around_bankers_rounding() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Test banker's rounding (round half to even)
a = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
result = fnp.around(a)
expected = np.around(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around should use banker's rounding");
    Ok(())
}

#[test]
fn around_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(5.5)
fnp_result = fnp.around(x)
np_result = np.around(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "around scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn around_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5+1.5j, 2.4-2.4j], dtype=np.complex128)
fnp_result = fnp.around(a)
np_result = np.around(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around complex should match numpy");
    Ok(())
}

/// Locks the zero-copy `numpy.around` fast path for decimals != 0
/// (`try_zerocopy_f64_around`, computing (v*10^d).round_ties_even()/10^d) to
/// bit-exact parity with numpy. Compares the sha256 of raw output bytes across
/// positive and negative decimals, multi-D inputs, and exact-half / extreme
/// values.
#[test]
fn around_decimals_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for n in [1000, 100003]:
    x = rng.standard_normal(n) * 100
    for dec in [1, 2, 3, -1, -2]:
        chunks.append(np.asarray(mod.around(x, dec)).tobytes())
chunks.append(np.asarray(mod.around(rng.standard_normal((30, 40)) * 10, 2)).tobytes())
chunks.append(np.asarray(mod.around(np.array([1.23456789, -9.87, 0.125, 2.5, -0.5, 1.5, -0.0, np.inf, -np.inf, np.nan], dtype=np.float64), 2)).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp").to_string()))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy around(decimals) must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}

#[test]
fn around_f32_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = (1 << 21) + 17
base = (np.arange(n, dtype=np.float32) - np.float32(n // 2)) * np.float32(0.12345)
special = np.array([0.0, -0.0, 0.5, 1.5, 2.5, -0.5, -1.5, np.inf, -np.inf, np.nan], dtype=np.float32)
a = np.concatenate([base, special])
ok = True
for decimals in [3, 0, -1]:
    actual = fnp.around(a, decimals=decimals)
    expected = np.around(a, decimals=decimals)
    ok = (
        ok
        and actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and actual.flags["C_CONTIGUOUS"]
        and actual.flags["WRITEABLE"]
        and actual.tobytes() == expected.tobytes()
    )
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large f32 around parallel path should be byte-exact"
    );
    Ok(())
}
