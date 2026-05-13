//! Conformance tests for numpy 2.x Array API top-level functions:
//! unstack, permute_dims, vecdot.

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
// unstack
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unstack_2d_default_axis_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
ours = fnp.unstack(a)
theirs = np.unstack(a)
print(len(ours) == len(theirs) and
      all(np.array_equal(x, y) for x, y in zip(ours, theirs)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unstack default axis should match numpy"
    );
    Ok(())
}

#[test]
fn unstack_axis_minus_one_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
ours = fnp.unstack(a, axis=-1)
theirs = np.unstack(a, axis=-1)
print(len(ours) == len(theirs) and
      all(np.array_equal(x, y) for x, y in zip(ours, theirs)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unstack axis=-1 should match numpy");
    Ok(())
}

#[test]
fn unstack_3d_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
for ax in (0, 1, 2, -1):
    ours = fnp.unstack(a, axis=ax)
    theirs = np.unstack(a, axis=ax)
    if not (len(ours) == len(theirs) and all(np.array_equal(x, y) for x, y in zip(ours, theirs))):
        print(False)
        break
else:
    print(True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unstack on 3D arrays across all axes should match numpy"
    );
    Ok(())
}

#[test]
fn unstack_1d_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([10, 20, 30, 40])
ours = fnp.unstack(a)
theirs = np.unstack(a)
print(len(ours) == len(theirs) and
      all(np.array_equal(x, y) for x, y in zip(ours, theirs)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unstack on 1D array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// permute_dims
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn permute_dims_2d_swap_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
ours = fnp.permute_dims(a, (1, 0))
theirs = np.permute_dims(a, (1, 0))
print(np.array_equal(ours, theirs) and ours.shape == theirs.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "permute_dims 2D swap should match numpy"
    );
    Ok(())
}

#[test]
fn permute_dims_3d_cycle_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
ours = fnp.permute_dims(a, (2, 0, 1))
theirs = np.permute_dims(a, (2, 0, 1))
print(np.array_equal(ours, theirs) and ours.shape == theirs.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "permute_dims 3D cycle should match numpy"
    );
    Ok(())
}

#[test]
fn permute_dims_identity_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
ours = fnp.permute_dims(a, (0, 1, 2))
theirs = np.permute_dims(a, (0, 1, 2))
print(np.array_equal(ours, theirs) and np.array_equal(ours, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "permute_dims identity should match numpy and be value-equal to input"
    );
    Ok(())
}

#[test]
fn permute_dims_reverse_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(60).reshape(3, 4, 5)
ours = fnp.permute_dims(a, (2, 1, 0))
theirs = np.permute_dims(a, (2, 1, 0))
print(np.array_equal(ours, theirs) and ours.shape == (5, 4, 3))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "permute_dims reverse should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// vecdot
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn vecdot_1d_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
ours = fnp.vecdot(a, b)
theirs = np.vecdot(a, b)
print(ours == theirs)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vecdot 1D should match numpy");
    Ok(())
}

#[test]
fn vecdot_2d_default_axis_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 1, 1], [2, 2, 2]])
ours = fnp.vecdot(a, b)
theirs = np.vecdot(a, b)
print(np.array_equal(ours, theirs))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "vecdot 2D default axis should match numpy"
    );
    Ok(())
}

#[test]
fn vecdot_axis_zero_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 1, 1], [2, 2, 2]])
ours = fnp.vecdot(a, b, axis=0)
theirs = np.vecdot(a, b, axis=0)
print(np.array_equal(ours, theirs))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vecdot axis=0 should match numpy");
    Ok(())
}

#[test]
fn vecdot_dtype_promotion_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
ours = fnp.vecdot(a, b)
theirs = np.vecdot(a, b)
print(ours == theirs and ours.dtype == theirs.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "vecdot dtype promotion should match numpy"
    );
    Ok(())
}

#[test]
fn vecdot_complex_matches_numpy() -> Result<(), String> {
    // numpy.vecdot on complex inputs uses the conjugate of x1 (or x2 — verify
    // current numpy convention via the oracle rather than hard-coding).
    let script = fnp_script(
        r#"
a = np.array([1+2j, 3+4j])
b = np.array([5+6j, 7+8j])
ours = fnp.vecdot(a, b)
theirs = np.vecdot(a, b)
print(ours == theirs)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "vecdot complex inputs should match numpy"
    );
    Ok(())
}
