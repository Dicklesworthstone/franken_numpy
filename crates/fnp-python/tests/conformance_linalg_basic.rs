//! Conformance tests for numpy basic linear algebra operations against NumPy oracle.
//!
//! Tests dot, matmul, inner, outer, cross, tensordot.

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
// dot
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dot_1d_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.dot(a, b)
expected = np.dot(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot 1d-1d should match numpy");
    Ok(())
}

#[test]
fn dot_2d_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([1, 2])
result = fnp.dot(a, b)
expected = np.dot(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot 2d-1d should match numpy");
    Ok(())
}

#[test]
fn dot_2d_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.dot(a, b)
expected = np.dot(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot 2d-2d should match numpy");
    Ok(())
}

#[test]
fn dot_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5])
b = np.array([0.5, 1.5, 2.5])
result = fnp.dot(a, b)
expected = np.dot(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// matmul
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn matmul_2d_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul 2d-2d should match numpy");
    Ok(())
}

#[test]
fn matmul_1d_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2])
b = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul 1d-2d should match numpy");
    Ok(())
}

#[test]
fn matmul_2d_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3])
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul 2d-1d should match numpy");
    Ok(())
}

#[test]
fn matmul_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(12).reshape(2, 3, 2)
b = np.arange(8).reshape(2, 2, 2)
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul 3d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// inner
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn inner_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.inner(a, b)
expected = np.inner(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inner 1d should match numpy");
    Ok(())
}

#[test]
fn inner_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.inner(a, b)
expected = np.inner(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inner 2d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// outer
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn outer_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5])
result = fnp.outer(a, b)
expected = np.outer(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "outer 1d should match numpy");
    Ok(())
}

#[test]
fn outer_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])
result = fnp.outer(a, b)
expected = np.outer(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "outer 2d should match numpy (flattens input)"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cross
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cross_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross 3d should match numpy");
    Ok(())
}

#[test]
fn cross_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2])
b = np.array([3, 4])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross 2d should match numpy");
    Ok(())
}

#[test]
fn cross_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross multiple should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// tensordot
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tensordot_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Default axes=2 contracts last 2 axes of a with first 2 axes of b
a = np.arange(12).reshape(2, 3, 2)
b = np.arange(12).reshape(3, 2, 2)
result = fnp.tensordot(a, b)
expected = np.tensordot(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot default should match numpy"
    );
    Ok(())
}

#[test]
fn tensordot_axes_1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
result = fnp.tensordot(a, b, axes=1)
expected = np.tensordot(a, b, axes=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensordot axes=1 should match numpy");
    Ok(())
}

#[test]
fn tensordot_axes_0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(4).reshape(2, 2)
b = np.arange(4).reshape(2, 2)
result = fnp.tensordot(a, b, axes=0)
expected = np.tensordot(a, b, axes=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensordot axes=0 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dot_matmul_equivalence_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
dot_result = fnp.dot(a, b)
matmul_result = fnp.matmul(a, b)
print(np.array_equal(dot_result, matmul_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "dot and matmul should be equivalent for 2d arrays"
    );
    Ok(())
}

#[test]
fn inner_dot_equivalence_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
inner_result = fnp.inner(a, b)
dot_result = fnp.dot(a, b)
print(inner_result == dot_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "inner and dot should be equivalent for 1d arrays"
    );
    Ok(())
}

#[test]
fn cross_antisymmetric() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
cross_ab = fnp.cross(a, b)
cross_ba = fnp.cross(b, a)
print(np.array_equal(cross_ab, -cross_ba))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cross product should be antisymmetric"
    );
    Ok(())
}

#[test]
fn linalg_det_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
fnp_result = fnp.linalg.det(a)
np_result = np.linalg.det(a)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "linalg.det scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn linalg_norm_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
fnp_result = fnp.linalg.norm(a)
np_result = np.linalg.norm(a)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "linalg.norm scalar return type should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Complex number tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dot_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
b = np.array([4-1j, 5-2j, 6-3j], dtype=np.complex128)
result = fnp.dot(a, b)
expected = np.dot(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot complex should match numpy");
    Ok(())
}

#[test]
fn matmul_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j]], dtype=np.complex128)
b = np.array([[5+2j, 6], [7, 8-2j]], dtype=np.complex128)
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul complex should match numpy");
    Ok(())
}

#[test]
fn inner_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2+2j], dtype=np.complex128)
b = np.array([3-1j, 4-2j], dtype=np.complex128)
result = fnp.inner(a, b)
expected = np.inner(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inner complex should match numpy");
    Ok(())
}

#[test]
fn outer_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
b = np.array([3+2j, 4-2j, 5], dtype=np.complex128)
result = fnp.outer(a, b)
expected = np.outer(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "outer complex should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// norm tests with different orders
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn norm_vector_l1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, -2, 3, -4], dtype=np.float64)
fnp_result = fnp.linalg.norm(a, ord=1)
np_result = np.linalg.norm(a, ord=1)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "norm L1 should match numpy");
    Ok(())
}

#[test]
fn norm_vector_l2() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 4], dtype=np.float64)
fnp_result = fnp.linalg.norm(a, ord=2)
np_result = np.linalg.norm(a, ord=2)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "norm L2 should match numpy");
    Ok(())
}

#[test]
fn norm_vector_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, -5, 3], dtype=np.float64)
fnp_result = fnp.linalg.norm(a, ord=np.inf)
np_result = np.linalg.norm(a, ord=np.inf)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "norm inf should match numpy");
    Ok(())
}

#[test]
fn norm_matrix_fro() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]], dtype=np.float64)
fnp_result = fnp.linalg.norm(a, ord='fro')
np_result = np.linalg.norm(a, ord='fro')
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "norm Frobenius should match numpy");
    Ok(())
}

#[test]
fn norm_matrix_nuc() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]], dtype=np.float64)
fnp_result = fnp.linalg.norm(a, ord='nuc')
np_result = np.linalg.norm(a, ord='nuc')
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "norm nuclear should match numpy");
    Ok(())
}

#[test]
fn norm_zero_vector() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.zeros(5)
fnp_result = fnp.linalg.norm(a)
np_result = np.linalg.norm(a)
print(fnp_result == np_result == 0.0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "norm of zero vector should be 0");
    Ok(())
}

#[test]
fn norm_empty_vector() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([])
fnp_result = fnp.linalg.norm(a)
np_result = np.linalg.norm(a)
print(fnp_result == np_result == 0.0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "norm of empty vector should be 0");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// det tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn det_2x2() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]], dtype=np.float64)
fnp_result = fnp.linalg.det(a)
np_result = np.linalg.det(a)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "det 2x2 should match numpy");
    Ok(())
}

#[test]
fn det_3x3() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=np.float64)
fnp_result = fnp.linalg.det(a)
np_result = np.linalg.det(a)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "det 3x3 should match numpy");
    Ok(())
}

#[test]
fn det_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(4)
fnp_result = fnp.linalg.det(a)
np_result = np.linalg.det(a)
print(np.allclose(fnp_result, 1.0) and np.allclose(np_result, 1.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "det of identity should be 1");
    Ok(())
}

#[test]
fn det_singular() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Singular matrix (rows are linearly dependent)
a = np.array([[1, 2], [2, 4]], dtype=np.float64)
fnp_result = fnp.linalg.det(a)
np_result = np.linalg.det(a)
print(np.allclose(fnp_result, 0.0) and np.allclose(np_result, 0.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "det of singular matrix should be 0");
    Ok(())
}

#[test]
fn det_1x1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[5.0]], dtype=np.float64)
fnp_result = fnp.linalg.det(a)
np_result = np.linalg.det(a)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "det of 1x1 should match numpy");
    Ok(())
}

#[test]
fn det_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j]], dtype=np.complex128)
fnp_result = fnp.linalg.det(a)
np_result = np.linalg.det(a)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "det complex should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Error behavior tests - verify fnp raises same errors as numpy
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn inv_singular_raises_linalgerror() -> Result<(), String> {
    let script = fnp_script(
        r#"
def classify(call):
    try:
        call()
        return ("ok", "")
    except np.linalg.LinAlgError as e:
        return ("LinAlgError", str(e).lower())
    except Exception as e:
        return (type(e).__name__, str(e).lower())

a = np.array([[1, 2], [2, 4]], dtype=np.float64)
fnp_result = classify(lambda: fnp.linalg.inv(a))
np_result = classify(lambda: np.linalg.inv(a))
# Both should raise LinAlgError with "singular" in the message
print(fnp_result[0] == np_result[0] == "LinAlgError" and "singular" in fnp_result[1])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "inv of singular matrix should raise LinAlgError"
    );
    Ok(())
}

#[test]
fn det_nonsquare_raises_linalgerror() -> Result<(), String> {
    let script = fnp_script(
        r#"
def classify(call):
    try:
        call()
        return ("ok", "")
    except np.linalg.LinAlgError as e:
        return ("LinAlgError", str(e).lower())
    except Exception as e:
        return (type(e).__name__, str(e).lower())

a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
fnp_result = classify(lambda: fnp.linalg.det(a))
np_result = classify(lambda: np.linalg.det(a))
# Both should raise LinAlgError with "square" in message
print(fnp_result[0] == np_result[0] == "LinAlgError")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "det of non-square matrix should raise LinAlgError"
    );
    Ok(())
}

#[test]
fn solve_singular_raises_linalgerror() -> Result<(), String> {
    let script = fnp_script(
        r#"
def classify(call):
    try:
        call()
        return ("ok", "")
    except np.linalg.LinAlgError as e:
        return ("LinAlgError", str(e).lower())
    except Exception as e:
        return (type(e).__name__, str(e).lower())

a = np.array([[1, 2], [2, 4]], dtype=np.float64)
b = np.array([1, 2], dtype=np.float64)
fnp_result = classify(lambda: fnp.linalg.solve(a, b))
np_result = classify(lambda: np.linalg.solve(a, b))
# Both should raise LinAlgError with "singular" in message
print(fnp_result[0] == np_result[0] == "LinAlgError" and "singular" in fnp_result[1])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "solve with singular matrix should raise LinAlgError"
    );
    Ok(())
}

#[test]
fn cholesky_non_posdef_raises_linalgerror() -> Result<(), String> {
    let script = fnp_script(
        r#"
def classify(call):
    try:
        call()
        return ("ok", "")
    except np.linalg.LinAlgError as e:
        return ("LinAlgError", str(e).lower())
    except Exception as e:
        return (type(e).__name__, str(e).lower())

# Not positive definite (negative eigenvalue)
a = np.array([[1, 2], [2, 1]], dtype=np.float64)
fnp_result = classify(lambda: fnp.linalg.cholesky(a))
np_result = classify(lambda: np.linalg.cholesky(a))
# Both should raise LinAlgError
print(fnp_result[0] == np_result[0] == "LinAlgError")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cholesky of non-positive-definite matrix should raise LinAlgError"
    );
    Ok(())
}

#[test]
fn norm_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# norm of signed-zero vector: norm([±0, ±0]) = +0 (always positive magnitude)
tests = [
    np.array([0.0, 0.0]),
    np.array([-0.0, -0.0]),
    np.array([0.0, -0.0]),
]
all_pass = True
for a in tests:
    fnp_result = fnp.linalg.norm(a)
    np_result = np.linalg.norm(a)
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: norm({a.tolist()}) signbit fnp={fnp_sign} np={np_sign}")
        all_pass = False
    if fnp_result != np_result:
        print(f"FAIL: norm({a.tolist()}) value mismatch")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "norm signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn det_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# det of matrix with signed zeros
# det([[0, 0], [0, 1]]) = 0, det([[-0, 0], [0, 1]]) = -0
tests = [
    np.array([[0.0, 0.0], [0.0, 1.0]]),
    np.array([[-0.0, 0.0], [0.0, 1.0]]),
    np.array([[0.0, -0.0], [0.0, 1.0]]),
]
all_pass = True
for a in tests:
    fnp_result = fnp.linalg.det(a)
    np_result = np.linalg.det(a)
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: det of matrix with zeros signbit fnp={fnp_sign} np={np_sign}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "det signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// NaN propagation tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn norm_nan_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, np.nan, 3.0])
fnp_result = fnp.linalg.norm(x)
np_result = np.linalg.norm(x)
print(np.isnan(fnp_result) and np.isnan(np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "norm should propagate NaN");
    Ok(())
}

#[test]
fn det_nan_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, np.nan], [3.0, 4.0]])
fnp_result = fnp.linalg.det(a)
np_result = np.linalg.det(a)
print(np.isnan(fnp_result) and np.isnan(np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "det should propagate NaN");
    Ok(())
}

#[test]
fn inv_nan_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, np.nan], [3.0, 4.0]])
fnp_result = fnp.linalg.inv(a)
np_result = np.linalg.inv(a)
print(np.all(np.isnan(fnp_result) == np.isnan(np_result)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inv should propagate NaN similarly");
    Ok(())
}

#[test]
fn solve_nan_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([1.0, np.nan])
fnp_result = fnp.linalg.solve(a, b)
np_result = np.linalg.solve(a, b)
print(np.all(np.isnan(fnp_result) == np.isnan(np_result)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "solve should propagate NaN similarly"
    );
    Ok(())
}

#[test]
fn svd_nan_raises_error() -> Result<(), String> {
    let script = fnp_script(
        r#"
# NumPy SVD raises LinAlgError for NaN input ("SVD did not converge")
a = np.array([[1.0, np.nan], [3.0, 4.0]])
fnp_raised = False
np_raised = False
try:
    fnp.linalg.svd(a)
except Exception:
    fnp_raised = True
try:
    np.linalg.svd(a)
except Exception:
    np_raised = True
print(fnp_raised == np_raised)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "svd should raise error for NaN input similarly to numpy"
    );
    Ok(())
}

#[test]
fn eig_nan_raises_error() -> Result<(), String> {
    let script = fnp_script(
        r#"
# NumPy eig raises LinAlgError for NaN input ("Array must not contain infs or NaNs")
a = np.array([[1.0, np.nan], [3.0, 4.0]])
fnp_raised = False
np_raised = False
try:
    fnp.linalg.eig(a)
except Exception:
    fnp_raised = True
try:
    np.linalg.eig(a)
except Exception:
    np_raised = True
print(fnp_raised == np_raised)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "eig should raise error for NaN input similarly to numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Inf handling tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn norm_inf_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, np.inf, 3.0])
fnp_result = fnp.linalg.norm(x)
np_result = np.linalg.norm(x)
print(np.isinf(fnp_result) and np.isinf(np_result) and fnp_result > 0 and np_result > 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "norm should return +inf for input with inf"
    );
    Ok(())
}

#[test]
fn norm_neginf_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, -np.inf, 3.0])
fnp_result = fnp.linalg.norm(x)
np_result = np.linalg.norm(x)
print(np.isinf(fnp_result) and np.isinf(np_result) and fnp_result > 0 and np_result > 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "norm should return +inf for input with -inf"
    );
    Ok(())
}

#[test]
fn det_inf_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[np.inf, 1.0], [1.0, 1.0]])
fnp_result = fnp.linalg.det(a)
np_result = np.linalg.det(a)
print(np.isinf(fnp_result) == np.isinf(np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "det should propagate inf similarly to numpy"
    );
    Ok(())
}

#[test]
fn eig_inf_raises_error() -> Result<(), String> {
    let script = fnp_script(
        r#"
# NumPy eig raises LinAlgError for Inf input
a = np.array([[np.inf, 1.0], [1.0, 1.0]])
fnp_raised = False
np_raised = False
try:
    fnp.linalg.eig(a)
except Exception:
    fnp_raised = True
try:
    np.linalg.eig(a)
except Exception:
    np_raised = True
print(fnp_raised == np_raised)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "eig should raise error for Inf input similarly to numpy"
    );
    Ok(())
}
