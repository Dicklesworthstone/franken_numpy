//! Conformance tests for numpy moveaxis and pad functions against NumPy oracle.
//!
//! Tests moveaxis, pad.

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

#[test]
fn moveaxis_pad_python_container_and_keyword_surfaces_match_numpy() -> Result<(), String> {
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
    (
        "moveaxis Python list source",
        "moveaxis",
        lambda: (([[[1, 2], [3, 4]]], 0, -1), {}),
    ),
    (
        "moveaxis tuple/list axes",
        "moveaxis",
        lambda: ((np.arange(24).reshape(2, 3, 4), (0, 2), [2, 0]), {}),
    ),
    ("pad Python list scalar width", "pad", lambda: (([1, 2, 3], 2), {})),
    (
        "pad nested width constant values",
        "pad",
        lambda: ((np.array([[1, 2], [3, 4]], dtype=np.int16), ((1, 0), (2, 1))), {"mode": "constant", "constant_values": ((9, 8), (7, 6))}),
    ),
    (
        "pad linear ramp end values",
        "pad",
        lambda: ((np.array([1.0, 2.0, 3.0]), (2, 1)), {"mode": "linear_ramp", "end_values": (0.5, 9.5)}),
    ),
    (
        "pad maximum stat length",
        "pad",
        lambda: ((np.arange(6).reshape(2, 3), ((1, 1), (2, 0))), {"mode": "maximum", "stat_length": ((1, 1), (2, 1))}),
    ),
    (
        "pad reflect odd",
        "pad",
        lambda: ((np.array([1, 2, 4]), 2), {"mode": "reflect", "reflect_type": "odd"}),
    ),
    (
        "moveaxis repeated source error",
        "moveaxis",
        lambda: ((np.arange(24).reshape(2, 3, 4), (0, 0), (1, 2)), {}),
    ),
    (
        "moveaxis length mismatch error",
        "moveaxis",
        lambda: ((np.arange(24).reshape(2, 3, 4), (0, 1), (2,)), {}),
    ),
    ("pad negative width error", "pad", lambda: ((np.arange(3), (-1, 1)), {})),
    ("pad invalid mode error", "pad", lambda: ((np.arange(3), 1), {"mode": "not-a-mode"})),
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
        "moveaxis/pad Python-container and keyword surfaces should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// moveaxis
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn moveaxis_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.moveaxis(a, 0, -1)
expected = np.moveaxis(a, 0, -1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "moveaxis basic should match numpy");
    Ok(())
}

#[test]
fn moveaxis_single_to_single() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.moveaxis(a, 2, 0)
expected = np.moveaxis(a, 2, 0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "moveaxis single to single should match numpy"
    );
    Ok(())
}

#[test]
fn moveaxis_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.moveaxis(a, [0, 1], [-1, -2])
expected = np.moveaxis(a, [0, 1], [-1, -2])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "moveaxis multiple should match numpy"
    );
    Ok(())
}

#[test]
fn moveaxis_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.moveaxis(a, -1, 0)
expected = np.moveaxis(a, -1, 0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "moveaxis negative should match numpy"
    );
    Ok(())
}

#[test]
fn moveaxis_4d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(120).reshape(2, 3, 4, 5)
result = fnp.moveaxis(a, 1, 3)
expected = np.moveaxis(a, 1, 3)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "moveaxis 4d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// pad
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn pad_constant_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.pad(a, 2, mode='constant')
expected = np.pad(a, 2, mode='constant')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad constant 1d should match numpy");
    Ok(())
}

#[test]
fn pad_constant_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.pad(a, 2, mode='constant', constant_values=99)
expected = np.pad(a, 2, mode='constant', constant_values=99)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "pad constant value should match numpy"
    );
    Ok(())
}

#[test]
fn pad_edge() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.pad(a, 2, mode='edge')
expected = np.pad(a, 2, mode='edge')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad edge should match numpy");
    Ok(())
}

#[test]
fn pad_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.pad(a, 1, mode='constant')
expected = np.pad(a, 1, mode='constant')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad 2d should match numpy");
    Ok(())
}

#[test]
fn pad_asymmetric() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.pad(a, (1, 3), mode='constant')
expected = np.pad(a, (1, 3), mode='constant')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad asymmetric should match numpy");
    Ok(())
}

#[test]
fn pad_reflect() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.pad(a, 2, mode='reflect')
expected = np.pad(a, 2, mode='reflect')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad reflect should match numpy");
    Ok(())
}

#[test]
fn pad_wrap() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.pad(a, 2, mode='wrap')
expected = np.pad(a, 2, mode='wrap')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad wrap should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn moveaxis_vs_transpose() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
moveaxis_result = fnp.moveaxis(a, 0, 1)
transpose_result = fnp.transpose(a)
print(np.array_equal(moveaxis_result, transpose_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "moveaxis should equal transpose for 2d swap"
    );
    Ok(())
}

#[test]
fn pad_preserves_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='float32')
result = fnp.pad(a, 1, mode='constant')
print(result.dtype == a.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad should preserve dtype");
    Ok(())
}

#[test]
fn moveaxis_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.zeros((2, 3, 4), dtype=np.complex128)
fnp_result = fnp.moveaxis(a, 0, -1)
np_result = np.moveaxis(a, 0, -1)
print(fnp_result.shape == np_result.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "moveaxis complex should match numpy");
    Ok(())
}

#[test]
fn pad_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j], dtype=np.complex128)
fnp_result = fnp.pad(a, 1, mode='constant')
np_result = np.pad(a, 1, mode='constant')
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad complex should match numpy");
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
        if stderr.contains("AxisError") || stderr.contains("axis") {
            "AxisError".to_string()
        } else if stderr.contains("ValueError") {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn moveaxis_invalid_source_raises_axiserror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(24).reshape(2, 3, 4)
fnp.moveaxis(a, 5, 0)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(24).reshape(2, 3, 4)
np.moveaxis(a, 5, 0)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "moveaxis with invalid source axis should raise same error as numpy"
    );
}

#[test]
fn pad_negative_width_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(5)
fnp.pad(a, -1)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(5)
np.pad(a, -1)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "pad with negative width should raise same error as numpy"
    );
}
