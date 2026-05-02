//! Conformance tests for numpy range functions against NumPy oracle.
//!
//! Tests arange, linspace, logspace, geomspace.

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
// arange
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn arange_stop_only() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.arange(10)
expected = np.arange(10)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arange stop only should match numpy");
    Ok(())
}

#[test]
fn arange_start_stop() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.arange(2, 10)
expected = np.arange(2, 10)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arange start stop should match numpy");
    Ok(())
}

#[test]
fn arange_with_step() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.arange(0, 10, 2)
expected = np.arange(0, 10, 2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arange with step should match numpy");
    Ok(())
}

#[test]
fn arange_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.arange(0.5, 5.5, 0.5)
expected = np.arange(0.5, 5.5, 0.5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arange float should match numpy");
    Ok(())
}

#[test]
fn arange_negative_step() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.arange(10, 0, -1)
expected = np.arange(10, 0, -1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arange negative step should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// linspace
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn linspace_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.linspace(0, 10, 5)
expected = np.linspace(0, 10, 5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "linspace basic should match numpy");
    Ok(())
}

#[test]
fn linspace_endpoint_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.linspace(0, 10, 5, endpoint=False)
expected = np.linspace(0, 10, 5, endpoint=False)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "linspace endpoint=False should match numpy");
    Ok(())
}

#[test]
fn linspace_single_point() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.linspace(5, 5, 1)
expected = np.linspace(5, 5, 1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "linspace single point should match numpy");
    Ok(())
}

#[test]
fn linspace_many_points() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.linspace(0, 1, 100)
expected = np.linspace(0, 1, 100)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "linspace many points should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// logspace
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn logspace_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.logspace(0, 2, 5)  # 10^0 to 10^2
expected = np.logspace(0, 2, 5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logspace basic should match numpy");
    Ok(())
}

#[test]
fn logspace_custom_base() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.logspace(0, 3, 4, base=2)  # 2^0 to 2^3
expected = np.logspace(0, 3, 4, base=2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logspace custom base should match numpy");
    Ok(())
}

#[test]
fn logspace_endpoint_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.logspace(0, 2, 5, endpoint=False)
expected = np.logspace(0, 2, 5, endpoint=False)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logspace endpoint=False should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// geomspace
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn geomspace_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.geomspace(1, 1000, 4)
expected = np.geomspace(1, 1000, 4)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "geomspace basic should match numpy");
    Ok(())
}

#[test]
fn geomspace_endpoint_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.geomspace(1, 1000, 4, endpoint=False)
expected = np.geomspace(1, 1000, 4, endpoint=False)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "geomspace endpoint=False should match numpy");
    Ok(())
}

#[test]
fn geomspace_small_range() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.geomspace(0.1, 10, 5)
expected = np.geomspace(0.1, 10, 5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "geomspace small range should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn linspace_arange_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
# linspace(0, 9, 10) should equal arange(10) for integers
linspace_result = fnp.linspace(0, 9, 10)
arange_result = fnp.arange(10).astype(float)
print(np.allclose(linspace_result, arange_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "linspace should match arange for equivalent params");
    Ok(())
}

#[test]
fn logspace_geomspace_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
# logspace(a, b, n) should equal geomspace(10^a, 10^b, n)
logspace_result = fnp.logspace(1, 3, 5)
geomspace_result = fnp.geomspace(10, 1000, 5)
print(np.allclose(logspace_result, geomspace_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logspace should relate to geomspace via powers");
    Ok(())
}
