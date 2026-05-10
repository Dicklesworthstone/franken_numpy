//! Conformance tests for numpy.savetxt and ndarray.tofile against NumPy oracle.
//!
//! Validates that fnp_python.savetxt and fnp_python.tofile produce byte/text output
//! identical to numpy's reference implementations across the documented kwarg surface.

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
         import io\n\
         import numpy as np\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// savetxt
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn savetxt_2d_default_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float64)
buf_a = io.StringIO()
buf_b = io.StringIO()
fnp.savetxt(buf_a, arr)
np.savetxt(buf_b, arr)
print(buf_a.getvalue() == buf_b.getvalue())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "savetxt 2D default must match numpy");
    Ok(())
}

#[test]
fn savetxt_custom_fmt_and_delimiter_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array([[0.1234567, 9.8765432], [1.0, 2.0]], dtype=np.float64)
buf_a = io.StringIO()
buf_b = io.StringIO()
fnp.savetxt(buf_a, arr, fmt='%.5e', delimiter=',')
np.savetxt(buf_b, arr, fmt='%.5e', delimiter=',')
print(buf_a.getvalue() == buf_b.getvalue())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "savetxt fmt='%.5e' + delimiter=',' must match numpy"
    );
    Ok(())
}

#[test]
fn savetxt_header_footer_comments_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
buf_a = io.StringIO()
buf_b = io.StringIO()
fnp.savetxt(buf_a, arr, header='col_a col_b', footer='end', comments='## ')
np.savetxt(buf_b, arr, header='col_a col_b', footer='end', comments='## ')
print(buf_a.getvalue() == buf_b.getvalue())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "savetxt header/footer/comments must match numpy"
    );
    Ok(())
}

#[test]
fn savetxt_int_fmt_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
buf_a = io.StringIO()
buf_b = io.StringIO()
fnp.savetxt(buf_a, arr, fmt='%d')
np.savetxt(buf_b, arr, fmt='%d')
print(buf_a.getvalue() == buf_b.getvalue())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "savetxt fmt='%d' must match numpy");
    Ok(())
}

#[test]
fn savetxt_1d_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
buf_a = io.StringIO()
buf_b = io.StringIO()
fnp.savetxt(buf_a, arr)
np.savetxt(buf_b, arr)
print(buf_a.getvalue() == buf_b.getvalue())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "savetxt 1D array must match numpy");
    Ok(())
}

#[test]
fn savetxt_per_column_fmt_list_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array([[1.5, 2, 3.0], [4.5, 5, 6.0]], dtype=np.float64)
buf_a = io.StringIO()
buf_b = io.StringIO()
fnp.savetxt(buf_a, arr, fmt=['%.2f', '%d', '%.1e'])
np.savetxt(buf_b, arr, fmt=['%.2f', '%d', '%.1e'])
print(buf_a.getvalue() == buf_b.getvalue())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "savetxt per-column fmt list must match numpy"
    );
    Ok(())
}

#[test]
fn savetxt_loadtxt_roundtrip_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array([[1.25, 2.75], [3.5, 4.0]], dtype=np.float64)
buf = io.StringIO()
fnp.savetxt(buf, arr, fmt='%.4f')
buf.seek(0)
restored = np.loadtxt(buf)
print(np.allclose(arr, restored))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "savetxt → loadtxt roundtrip must restore values"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// tofile
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tofile_binary_float64_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import tempfile, os
arr = np.array([1.0, 2.0, 3.0, -4.5, 5.25], dtype=np.float64)
with tempfile.TemporaryDirectory() as td:
    path_a = os.path.join(td, 'a.bin')
    path_b = os.path.join(td, 'b.bin')
    fnp.tofile(arr, path_a)
    arr.tofile(path_b)
    with open(path_a, 'rb') as f: a = f.read()
    with open(path_b, 'rb') as f: b = f.read()
print(a == b)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tofile binary float64 must match numpy"
    );
    Ok(())
}

#[test]
fn tofile_binary_int32_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import tempfile, os
arr = np.array([1, 2, 3, -4, 5, -6, 7], dtype=np.int32)
with tempfile.TemporaryDirectory() as td:
    path_a = os.path.join(td, 'a.bin')
    path_b = os.path.join(td, 'b.bin')
    fnp.tofile(arr, path_a)
    arr.tofile(path_b)
    with open(path_a, 'rb') as f: a = f.read()
    with open(path_b, 'rb') as f: b = f.read()
print(a == b)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tofile binary int32 must match numpy"
    );
    Ok(())
}

#[test]
fn tofile_text_sep_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import tempfile, os
arr = np.array([1.0, 2.5, 3.75, 4.125], dtype=np.float64)
with tempfile.TemporaryDirectory() as td:
    path_a = os.path.join(td, 'a.txt')
    path_b = os.path.join(td, 'b.txt')
    fnp.tofile(arr, path_a, sep=',')
    arr.tofile(path_b, sep=',')
    with open(path_a) as f: a = f.read()
    with open(path_b) as f: b = f.read()
print(a == b)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tofile sep=',' must match numpy");
    Ok(())
}

#[test]
fn tofile_text_format_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import tempfile, os
arr = np.array([1.234567, 2.345678, 3.456789], dtype=np.float64)
with tempfile.TemporaryDirectory() as td:
    path_a = os.path.join(td, 'a.txt')
    path_b = os.path.join(td, 'b.txt')
    fnp.tofile(arr, path_a, sep=' ', format='%.3f')
    arr.tofile(path_b, sep=' ', format='%.3f')
    with open(path_a) as f: a = f.read()
    with open(path_b) as f: b = f.read()
print(a == b)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tofile sep=' ' + format='%.3f' must match numpy"
    );
    Ok(())
}

#[test]
fn tofile_fromfile_binary_roundtrip_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import tempfile, os
arr = np.array([3.14, -1.41, 2.71, -0.57], dtype=np.float64)
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, 'a.bin')
    fnp.tofile(arr, path)
    restored = np.fromfile(path, dtype=np.float64)
print(np.array_equal(arr, restored))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tofile → fromfile roundtrip must restore values"
    );
    Ok(())
}
