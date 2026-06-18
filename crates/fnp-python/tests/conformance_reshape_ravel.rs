//! Conformance tests for numpy.reshape and numpy.ravel against NumPy oracle.
//!
//! Tests the native Rust implementations against NumPy.

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

// ─────────────────────────────────────────────────────────────────────────────
// reshape
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn reshape_ravel_python_container_and_keyword_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def clean(value):
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value

def normalize(value):
    array = np.asarray(value)
    return (str(array.dtype), tuple(array.shape), clean(array.tolist()))

def outcome(call_fn, *args, **kwargs):
    try:
        return ("ok", normalize(call_fn(*args, **kwargs)))
    except Exception as exc:
        return ("err", type(exc).__name__)

cases = [
    ("reshape Python list tuple shape", "reshape", lambda: (([1, 2, 3, 4], (2, 2)), {})),
    ("reshape scalar empty shape", "reshape", lambda: ((7, ()), {})),
    (
        "reshape list shape order F",
        "reshape",
        lambda: ((np.arange(6).reshape(2, 3), [3, 2]), {"order": "F"}),
    ),
    (
        "reshape newshape keyword",
        "reshape",
        lambda: ((np.arange(6),), {"newshape": (3, 2)}),
    ),
    (
        "reshape copy false view-compatible",
        "reshape",
        lambda: ((np.arange(6), (2, 3)), {"copy": False}),
    ),
    ("ravel Python list", "ravel", lambda: (([[1, 2], [3, 4]],), {})),
    ("ravel scalar", "ravel", lambda: ((7,), {})),
    (
        "ravel Fortran order array",
        "ravel",
        lambda: ((np.asfortranarray(np.arange(6).reshape(2, 3)),), {"order": "F"}),
    ),
    ("reshape missing shape error", "reshape", lambda: ((np.arange(3),), {})),
    (
        "reshape invalid order error",
        "reshape",
        lambda: ((np.arange(4), (2, 2)), {"order": "K"}),
    ),
    ("ravel invalid order error", "ravel", lambda: ((np.arange(4),), {"order": "Z"})),
]

ok = True
for label, name, factory in cases:
    args, kwargs = factory()
    actual = outcome(getattr(fnp, name), *args, **kwargs)
    args, kwargs = factory()
    expected = outcome(getattr(np, name), *args, **kwargs)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "reshape/ravel Python-container and keyword surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn reshape_1d_to_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6])
result = fnp.reshape(a, (2, 3))
expected = np.reshape(a, (2, 3))
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape 1d to 2d should match numpy");
    Ok(())
}

#[test]
fn reshape_2d_to_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.reshape(a, (6,))
expected = np.reshape(a, (6,))
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape 2d to 1d should match numpy");
    Ok(())
}

#[test]
fn reshape_with_minus1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6])
result = fnp.reshape(a, (2, -1))
expected = np.reshape(a, (2, -1))
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape with -1 should match numpy");
    Ok(())
}

#[test]
fn reshape_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24)
result = fnp.reshape(a, (2, 3, 4))
expected = np.reshape(a, (2, 3, 4))
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape 3d should match numpy");
    Ok(())
}

#[test]
fn reshape_order_c() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.reshape(a, (3, 2), order='C')
expected = np.reshape(a, (3, 2), order='C')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape order=C should match numpy");
    Ok(())
}

#[test]
fn reshape_order_f() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.reshape(a, (3, 2), order='F')
expected = np.reshape(a, (3, 2), order='F')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape order=F should match numpy");
    Ok(())
}

#[test]
fn reshape_float_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5, 4.5])
result = fnp.reshape(a, (2, 2))
expected = np.reshape(a, (2, 2))
print(np.allclose(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "reshape float array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ravel
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ravel_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.ravel(a)
expected = np.ravel(a)
print(np.array_equal(result, expected) and result.ndim == 1)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel 2d should match numpy");
    Ok(())
}

#[test]
fn ravel_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.ravel(a)
expected = np.ravel(a)
print(np.array_equal(result, expected) and result.ndim == 1)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel 3d should match numpy");
    Ok(())
}

#[test]
fn ravel_already_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.ravel(a)
expected = np.ravel(a)
print(np.array_equal(result, expected) and np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel already 1d should match numpy");
    Ok(())
}

#[test]
fn ravel_order_c() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.ravel(a, order='C')
expected = np.ravel(a, order='C')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel order=C should match numpy");
    Ok(())
}

#[test]
fn ravel_order_f() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.ravel(a, order='F')
expected = np.ravel(a, order='F')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel order=F should match numpy");
    Ok(())
}

#[test]
fn ravel_float_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.5, 2.5], [3.5, 4.5]])
result = fnp.ravel(a)
expected = np.ravel(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ravel float array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ravel_reshape_minus1_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
ravel_result = fnp.ravel(a)
reshape_result = fnp.reshape(a, (-1,))
print(np.array_equal(ravel_result, reshape_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel should equal reshape(-1)");
    Ok(())
}

#[test]
fn reshape_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.reshape(a, (2, 2))
np_result = np.reshape(a, (2, 2))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape complex should match numpy");
    Ok(())
}

#[test]
fn ravel_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2-1j], [3+2j, 4-2j]], dtype=np.complex128)
fnp_result = fnp.ravel(a)
np_result = np.ravel(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel complex should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Error behavior tests
// ─────────────────────────────────────────────────────────────────────────────

fn classify_error(script: &str) -> String {
    let output = std::process::Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("python3 should be available");
    if output.status.success() {
        "ok".to_string()
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("ValueError") {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn reshape_incompatible_size_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(12)
fnp.reshape(a, (5, 5))
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(12)
np.reshape(a, (5, 5))
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "reshape with incompatible size should raise same error as numpy"
    );
}

#[test]
fn reshape_multiple_unknowns_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(12)
fnp.reshape(a, (-1, -1))
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(12)
np.reshape(a, (-1, -1))
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "reshape with multiple -1 should raise same error as numpy"
    );
}
