//! Conformance tests for fnp_python.mgrid and fnp_python.ogrid — the
//! bracket-indexable grid-class instances re-exported from numpy.

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
fn mgrid_and_ogrid_identity_equal_to_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.mgrid is np.mgrid and fnp.ogrid is np.ogrid)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.mgrid / fnp.ogrid must be the same objects as numpy's"
    );
    Ok(())
}

#[test]
fn mgrid_1d_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
ours = fnp.mgrid[0:5]
theirs = np.mgrid[0:5]
print(np.array_equal(ours, theirs))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.mgrid[0:5] must match numpy"
    );
    Ok(())
}

#[test]
fn mgrid_2d_dense_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
ours = fnp.mgrid[0:3, 0:4]
theirs = np.mgrid[0:3, 0:4]
print(np.array_equal(ours, theirs) and ours.shape == theirs.shape == (2, 3, 4))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.mgrid[0:3, 0:4] must produce dense 2D grid matching numpy"
    );
    Ok(())
}

#[test]
fn mgrid_step_slice_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
ours = fnp.mgrid[0:1:0.25]
theirs = np.mgrid[0:1:0.25]
print(np.array_equal(ours, theirs))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.mgrid step slice must match numpy"
    );
    Ok(())
}

#[test]
fn mgrid_complex_step_matches_numpy() -> Result<(), String> {
    // mgrid supports an imaginary step interpreted as a count of points.
    let script = fnp_script(
        r#"
ours = fnp.mgrid[0:1:5j]
theirs = np.mgrid[0:1:5j]
print(np.array_equal(ours, theirs))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.mgrid complex-step (count) syntax must match numpy"
    );
    Ok(())
}

#[test]
fn ogrid_2d_sparse_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
ours = fnp.ogrid[0:3, 0:4]
theirs = np.ogrid[0:3, 0:4]
ok = (len(ours) == len(theirs) and
      all(np.array_equal(a, b) for a, b in zip(ours, theirs)) and
      ours[0].shape == (3, 1) and ours[1].shape == (1, 4))
print(ok)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.ogrid[0:3, 0:4] must produce sparse grid matching numpy"
    );
    Ok(())
}

#[test]
fn ogrid_complex_step_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
ours = fnp.ogrid[0:1:5j, 0:1:3j]
theirs = np.ogrid[0:1:5j, 0:1:3j]
ok = (len(ours) == len(theirs) and
      all(np.array_equal(a, b) for a, b in zip(ours, theirs)))
print(ok)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.ogrid complex-step (count) syntax must match numpy"
    );
    Ok(())
}
