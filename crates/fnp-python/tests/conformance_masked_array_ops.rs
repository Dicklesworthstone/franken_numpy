//! Conformance tests for numpy.ma masked array operations against NumPy oracle.
//!
//! Tests: compress_rows, compress_cols, clump_masked, clump_unmasked,
//! flatnotmasked_edges, flatnotmasked_contiguous

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
         import numpy.ma as ma\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// compress_rows
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compress_rows_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2], [3, 4], [5, 6]], mask=[[0, 0], [1, 0], [0, 0]])
fnp_result = fnp.compress_rows(x)
np_result = ma.compress_rows(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_rows basic mismatch");
    Ok(())
}

#[test]
fn compress_rows_no_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2], [3, 4], [5, 6]])
fnp_result = fnp.compress_rows(x)
np_result = ma.compress_rows(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_rows no masked mismatch");
    Ok(())
}

#[test]
fn compress_rows_all_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2], [3, 4]], mask=[[1, 1], [1, 1]])
fnp_result = fnp.compress_rows(x)
np_result = ma.compress_rows(x)
print(fnp_result.shape == np_result.shape)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_rows all masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// compress_cols
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compress_cols_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 1, 0], [0, 0, 0]])
fnp_result = fnp.compress_cols(x)
np_result = ma.compress_cols(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_cols basic mismatch");
    Ok(())
}

#[test]
fn compress_cols_no_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2, 3], [4, 5, 6]])
fnp_result = fnp.compress_cols(x)
np_result = ma.compress_cols(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_cols no masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// clump_masked
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn clump_masked_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5], mask=[0, 1, 1, 0, 1])
fnp_result = fnp.clump_masked(a)
np_result = ma.clump_masked(a)
match = len(fnp_result) == len(np_result)
if match:
    for f, n in zip(fnp_result, np_result):
        if f != n:
            match = False
            break
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_masked basic mismatch");
    Ok(())
}

#[test]
fn clump_masked_no_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5])
fnp_result = fnp.clump_masked(a)
np_result = ma.clump_masked(a)
print(len(fnp_result) == len(np_result) == 0)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_masked no masked mismatch");
    Ok(())
}

#[test]
fn clump_masked_all_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3], mask=[1, 1, 1])
fnp_result = fnp.clump_masked(a)
np_result = ma.clump_masked(a)
match = len(fnp_result) == len(np_result) == 1
if match:
    match = fnp_result[0] == np_result[0]
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_masked all masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// clump_unmasked
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn clump_unmasked_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5], mask=[1, 0, 0, 1, 0])
fnp_result = fnp.clump_unmasked(a)
np_result = ma.clump_unmasked(a)
match = len(fnp_result) == len(np_result)
if match:
    for f, n in zip(fnp_result, np_result):
        if f != n:
            match = False
            break
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_unmasked basic mismatch");
    Ok(())
}

#[test]
fn clump_unmasked_all_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3], mask=[1, 1, 1])
fnp_result = fnp.clump_unmasked(a)
np_result = ma.clump_unmasked(a)
print(len(fnp_result) == len(np_result) == 0)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_unmasked all masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// flatnotmasked_edges
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn flatnotmasked_edges_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5], mask=[1, 0, 0, 0, 1])
fnp_result = fnp.flatnotmasked_edges(a)
np_result = ma.flatnotmasked_edges(a)
# flatnotmasked_edges returns a tuple or None
if np_result is None:
    print(fnp_result is None)
else:
    print(fnp_result[0] == np_result[0] and fnp_result[1] == np_result[1])
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "flatnotmasked_edges basic mismatch");
    Ok(())
}

#[test]
fn flatnotmasked_edges_all_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3], mask=[1, 1, 1])
fnp_result = fnp.flatnotmasked_edges(a)
np_result = ma.flatnotmasked_edges(a)
print(fnp_result is None and np_result is None)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "flatnotmasked_edges all masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// flatnotmasked_contiguous
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn flatnotmasked_contiguous_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5], mask=[0, 1, 1, 0, 0])
fnp_result = fnp.flatnotmasked_contiguous(a)
np_result = ma.flatnotmasked_contiguous(a)
match = len(fnp_result) == len(np_result)
if match:
    for f, n in zip(fnp_result, np_result):
        if f != n:
            match = False
            break
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "flatnotmasked_contiguous basic mismatch");
    Ok(())
}

#[test]
fn flatnotmasked_contiguous_no_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5])
fnp_result = fnp.flatnotmasked_contiguous(a)
np_result = ma.flatnotmasked_contiguous(a)
match = len(fnp_result) == len(np_result) == 1
if match:
    match = fnp_result[0] == np_result[0]
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "flatnotmasked_contiguous no masked mismatch");
    Ok(())
}
