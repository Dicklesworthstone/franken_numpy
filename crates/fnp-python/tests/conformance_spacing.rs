//! Conformance tests for numpy.spacing against NumPy oracle.
//!
//! Tests spacing (distance to nearest representable value).

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
fn spacing_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 1e10, 1e-10])
result = fnp.spacing(x)
expected = np.spacing(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "spacing basic should match numpy");
    Ok(())
}

#[test]
fn spacing_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.0, -2.0, -1e10])
result = fnp.spacing(x)
expected = np.spacing(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "spacing with negative values should match numpy"
    );
    Ok(())
}

#[test]
fn spacing_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, np.inf, np.nan])
result = fnp.spacing(x)
expected = np.spacing(x)
# Check that non-NaN values match and NaN positions align
match = True
for r, e in zip(result.flat, expected.flat):
    if np.isnan(e):
        if not np.isnan(r):
            match = False
    elif r != e:
        match = False
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "spacing with special values should match numpy"
    );
    Ok(())
}

#[test]
fn spacing_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(1.0)
fnp_result = fnp.spacing(x)
np_result = np.spacing(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "spacing scalar return type should match numpy: {result}"
    );
    Ok(())
}
