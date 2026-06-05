//! Conformance tests for numpy.gradient against NumPy oracle.
//!
//! Tests gradient (gradient of an N-dimensional array).

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
fn gradient_1d_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([1, 2, 4, 7, 11])
result = fnp.gradient(f)
expected = np.gradient(f)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient 1D basic should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_with_spacing() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([1, 2, 4, 7, 11])
result = fnp.gradient(f, 2.0)
expected = np.gradient(f, 2.0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient with spacing should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([[1, 2, 6], [3, 4, 5], [7, 8, 9]])
result_y, result_x = fnp.gradient(f)
expected_y, expected_x = np.gradient(f)
print(np.allclose(result_y, expected_y) and np.allclose(result_x, expected_x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "gradient 2D should match numpy");
    Ok(())
}

#[test]
fn gradient_single_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([[1, 2, 6], [3, 4, 5], [7, 8, 9]])
result = fnp.gradient(f, axis=0)
expected = np.gradient(f, axis=0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient single axis should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 3+4j, 6+9j], dtype=np.complex128)
fnp_result = fnp.gradient(z)
np_result = np.gradient(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "gradient complex should match numpy");
    Ok(())
}

#[test]
fn gradient_edge_order_1() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([1, 2, 4, 7, 11])
fnp_result = fnp.gradient(f, edge_order=1)
np_result = np.gradient(f, edge_order=1)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient edge_order=1 should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_edge_order_2() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([1, 2, 4, 7, 11])
fnp_result = fnp.gradient(f, edge_order=2)
np_result = np.gradient(f, edge_order=2)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient edge_order=2 should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([1.0, np.inf, 3.0, np.nan, 5.0])
fnp_result = fnp.gradient(f)
np_result = np.gradient(f)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient special values should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_constant_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([5.0, 5.0, 5.0, 5.0])
fnp_result = fnp.gradient(f)
np_result = np.gradient(f)
# Gradient of constant should be zero
print(np.allclose(fnp_result, np_result) and np.allclose(fnp_result, 0.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient constant array should match numpy"
    );
    Ok(())
}

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
fn gradient_single_element_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
f = np.array([5.0])
fnp.gradient(f)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
f = np.array([5.0])
np.gradient(f)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "gradient single element should raise same error as numpy"
    );
}

#[test]
fn gradient_two_elements() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([1.0, 5.0])
fnp_result = fnp.gradient(f)
np_result = np.gradient(f)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient two elements should match numpy"
    );
    Ok(())
}
