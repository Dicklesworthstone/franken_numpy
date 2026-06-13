//! Conformance tests for numpy.diag against NumPy oracle.
//!
//! Tests diag (extract diagonal or construct diagonal matrix).

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
fn diag_extract_main_diagonal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diag(a)
expected = np.diag(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diag extract main diagonal should match numpy"
    );
    Ok(())
}

#[test]
fn diag_extract_offset_diagonal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diag(a, k=1)
expected = np.diag(a, k=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diag extract offset diagonal should match numpy"
    );
    Ok(())
}

#[test]
fn diag_construct_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
v = np.array([1, 2, 3])
result = fnp.diag(v)
expected = np.diag(v)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diag construct matrix from vector should match numpy"
    );
    Ok(())
}

#[test]
fn diag_construct_offset_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
v = np.array([1, 2])
result = fnp.diag(v, k=1)
expected = np.diag(v, k=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diag construct offset matrix should match numpy"
    );
    Ok(())
}

#[test]
fn diag_negative_offset() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diag(a, k=-1)
expected = np.diag(a, k=-1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diag negative offset should match numpy"
    );
    Ok(())
}

#[test]
fn diag_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j]], dtype=np.complex128)
fnp_result = fnp.diag(a)
np_result = np.diag(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diag complex should match numpy");
    Ok(())
}

#[test]
fn diagonal_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j]], dtype=np.complex128)
fnp_result = fnp.diagonal(a)
np_result = np.diagonal(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diagonal complex should match numpy");
    Ok(())
}

#[test]
fn diag_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[np.inf, np.nan], [-np.inf, 0.0]])
fnp_result = fnp.diag(a)
np_result = np.diag(a)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diag special values should match numpy"
    );
    Ok(())
}

#[test]
fn diag_empty_vector() -> Result<(), String> {
    let script = fnp_script(
        r#"
v = np.array([], dtype=np.float64)
fnp_result = fnp.diag(v)
np_result = np.diag(v)
print(fnp_result.shape == np_result.shape == (0, 0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diag empty vector should match numpy"
    );
    Ok(())
}

#[test]
fn diag_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
v = np.array([5.0])
fnp_result = fnp.diag(v)
np_result = np.diag(v)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diag single element should match numpy"
    );
    Ok(())
}

#[test]
fn diagonal_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
fnp_result = fnp.diagonal(a)
np_result = np.diagonal(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diagonal 3d should match numpy");
    Ok(())
}

/// Locks the zero-copy 1-D diag construction path (try_zerocopy_f64_diagflat via
/// diag): a deterministic f64 vector with NaN/inf/-0.0, built into a diagonal
/// matrix across several k offsets, must be BYTE-IDENTICAL to numpy.diag (numpy
/// is the oracle), plus a sha256 golden over the fnp output bytes for drift.
#[test]
fn diag_1d_construct_zerocopy_matches_numpy_bytes_and_golden() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib
s = 0x9E3779B97F4A7C15
n = 257
v = np.empty(n, dtype=np.float64)
for i in range(n):
    s = (s * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
    v[i] = ((s >> 33) / 4294967295.0) - 0.5
v[3] = np.nan; v[10] = np.inf; v[20] = -np.inf; v[30] = -0.0; v[31] = 0.0
h = hashlib.sha256()
allmatch = True
for k in (0, 1, -1, 7, -7, n, -n):
    r = np.asarray(fnp.diag(v, k))
    e = np.diag(v, k)
    if r.shape != e.shape or r.tobytes() != e.tobytes():
        allmatch = False
    h.update(r.tobytes())
print(allmatch)
print(h.hexdigest())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    assert_eq!(
        lines.next().unwrap_or("").trim(),
        "True",
        "1-D diag must be byte-identical to numpy.diag across k offsets"
    );
    assert_eq!(
        lines.next().unwrap_or("").trim(),
        "3ef89b038d261e7cf219620121a849a4ad1aa9833a23ac572d21b492c17dec67",
        "diag 1-D zero-copy golden sha256 drifted"
    );
    Ok(())
}
