//! Conformance tests for dtype promotion in reduction operations.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

static NUMPY_SITE_DIR: OnceLock<Result<PathBuf, String>> = OnceLock::new();

fn ensure_numpy_site_dir() -> Result<PathBuf, String> {
    NUMPY_SITE_DIR
        .get_or_init(|| {
            let site_dir =
                std::env::temp_dir().join(format!("fnp_numpy_oracle_site_{}", std::process::id()));
            let pip_install_output = Command::new("python3")
                .args(["-m", "pip", "install", "--target"])
                .arg(&site_dir)
                .arg("numpy")
                .output();
            match pip_install_output {
                Ok(output) if output.status.success() => return Ok(site_dir),
                Ok(_) | Err(_) => {}
            }

            let install_output = Command::new("uv")
                .args(["pip", "install", "--python", "python3", "--target"])
                .arg(&site_dir)
                .arg("numpy")
                .output()
                .map_err(|error| format!("uv pip install --target numpy should run: {error}"))?;
            if install_output.status.success() {
                Ok(site_dir)
            } else {
                Err(format!(
                    "uv pip install --target numpy failed: stdout={:?} stderr={:?}",
                    String::from_utf8_lossy(&install_output.stdout),
                    String::from_utf8_lossy(&install_output.stderr)
                ))
            }
        })
        .clone()
}

fn pythonpath_with_site_dir(site_dir: &Path) -> OsString {
    let mut path = OsString::from(site_dir);
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        path.push(":");
        path.push(existing);
    }
    path
}

fn command_python(
    script: &str,
    python: impl AsRef<std::ffi::OsStr>,
    pythonpath: Option<&OsString>,
) -> Result<Output, String> {
    let mut command = Command::new(python);
    command
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(path) = pythonpath {
        command.env("PYTHONPATH", path);
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("Python oracle should start: {error}\nScript: {script}"))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| format!("Python oracle stdin should be available\nScript: {script}"))?;
    stdin
        .write_all(script.as_bytes())
        .map_err(|error| format!("Python oracle stdin write should succeed: {error}"))?;
    drop(stdin);
    child
        .wait_with_output()
        .map_err(|error| format!("Python oracle should finish: {error}\nScript: {script}"))
}

fn run_numpy_oracle(script: &str) -> Result<Output, String> {
    if let Ok(python) = std::env::var("FNP_ORACLE_PYTHON") {
        return command_python(script, python, None);
    }

    let first_output = command_python(script, "python3", None)?;
    if first_output.status.success() {
        return Ok(first_output);
    }

    let stderr = String::from_utf8_lossy(&first_output.stderr);
    if !stderr.contains("No module named 'numpy'") {
        return Ok(first_output);
    }

    let site_dir = ensure_numpy_site_dir()?;
    let pythonpath = pythonpath_with_site_dir(&site_dir);
    command_python(script, "python3", Some(&pythonpath))
}

fn numpy_oracle(script: &str) -> Result<String, String> {
    let output = run_numpy_oracle(script)?;
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
// sum dtype promotion
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sum_int32_promotes_to_int64() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
fnp_result = fnp.sum(a)
np_result = np.sum(a)
print(fnp_result.dtype == np_result.dtype and fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum of int32 should promote to int64"
    );
    Ok(())
}

#[test]
fn sum_int16_promotes_to_int64() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int16')
fnp_result = fnp.sum(a)
np_result = np.sum(a)
print(fnp_result.dtype == np_result.dtype and fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum of int16 should promote to int64"
    );
    Ok(())
}

#[test]
fn sum_uint8_promotes_to_uint64() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='uint8')
fnp_result = fnp.sum(a)
np_result = np.sum(a)
print(fnp_result.dtype == np_result.dtype and fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum of uint8 should promote to uint64"
    );
    Ok(())
}

#[test]
fn sum_float32_stays_float32() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0], dtype='float32')
fnp_result = fnp.sum(a)
np_result = np.sum(a)
print(fnp_result.dtype == np_result.dtype and np.isclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sum of float32 should stay float32");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// prod dtype promotion
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn prod_int32_promotes_to_int64() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
fnp_result = fnp.prod(a)
np_result = np.prod(a)
print(fnp_result.dtype == np_result.dtype and fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "prod of int32 should promote to int64"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cumsum dtype promotion
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cumsum_int32_promotes_to_int64() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
fnp_result = fnp.cumsum(a)
np_result = np.cumsum(a)
print(fnp_result.dtype == np_result.dtype and np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumsum of int32 should promote to int64"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cumprod dtype promotion
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cumprod_int32_promotes_to_int64() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
fnp_result = fnp.cumprod(a)
np_result = np.cumprod(a)
print(fnp_result.dtype == np_result.dtype and np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumprod of int32 should promote to int64"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// diff dtype preservation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn diff_int32_stays_int32() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 3, 6, 10], dtype='int32')
fnp_result = fnp.diff(a)
np_result = np.diff(a)
print(fnp_result.dtype == np_result.dtype and np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff of int32 should stay int32");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ediff1d dtype preservation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ediff1d_int32_stays_int32() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 3, 6, 10], dtype='int32')
fnp_result = fnp.ediff1d(a)
np_result = np.ediff1d(a)
print(fnp_result.dtype == np_result.dtype and np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d of int32 should stay int32");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// mean/std/var always return float64
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mean_int32_returns_float64() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
fnp_result = fnp.mean(a)
np_result = np.mean(a)
print(fnp_result.dtype == np_result.dtype and np.isclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "mean of int32 should return float64");
    Ok(())
}

#[test]
fn std_int32_returns_float64() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
fnp_result = fnp.std(a)
np_result = np.std(a)
print(fnp_result.dtype == np_result.dtype and np.isclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "std of int32 should return float64");
    Ok(())
}

#[test]
fn var_int32_returns_float64() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
fnp_result = fnp.var(a)
np_result = np.var(a)
print(fnp_result.dtype == np_result.dtype and np.isclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "var of int32 should return float64");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// explicit dtype= keyword parity
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn reduction_dtype_keyword_controls_result_dtype_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def dtype_and_value_match(ours, expected):
    return (
        getattr(ours, 'dtype', None) == getattr(expected, 'dtype', None)
        and np.allclose(ours, expected, rtol=1e-6, atol=1e-6)
    )

a = np.array([[1, 2, 3], [4, 5, 6]], dtype='int16')
cases = [
    (
        'sum-axis0-float32',
        fnp.sum(a, axis=0, dtype=np.float32),
        np.sum(a, axis=0, dtype=np.float32),
    ),
    (
        'prod-axis1-int64-keepdims',
        fnp.prod(a, axis=1, dtype=np.int64, keepdims=True),
        np.prod(a, axis=1, dtype=np.int64, keepdims=True),
    ),
    (
        'mean-axis0-float32',
        fnp.mean(a, axis=0, dtype='float32'),
        np.mean(a, axis=0, dtype='float32'),
    ),
    (
        'std-axis1-float32-ddof1',
        fnp.std(a, axis=1, dtype='float32', ddof=1),
        np.std(a, axis=1, dtype='float32', ddof=1),
    ),
    (
        'var-axis1-float32-ddof1-keepdims',
        fnp.var(a, axis=1, dtype='float32', ddof=1, keepdims=True),
        np.var(a, axis=1, dtype='float32', ddof=1, keepdims=True),
    ),
]
failed = [name for name, ours, expected in cases if not dtype_and_value_match(ours, expected)]
print('True' if not failed else failed)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "reduction dtype= keyword should match NumPy result dtype and value"
    );
    Ok(())
}

#[test]
fn reduction_dtype_keyword_preserves_numpy_out_where_initial_semantics() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]], dtype='int16')
mask = np.array([[True, False, True], [False, True, True]])

ours_sum_out = np.full((2,), -99.0, dtype=np.float64)
expected_sum_out = np.full((2,), -99.0, dtype=np.float64)
ours_sum = fnp.sum(
    a,
    axis=1,
    dtype=np.float64,
    out=ours_sum_out,
    where=mask,
    initial=1,
)
expected_sum = np.sum(
    a,
    axis=1,
    dtype=np.float64,
    out=expected_sum_out,
    where=mask,
    initial=1,
)

ours_prod_out = np.full((2,), -9, dtype=np.int64)
expected_prod_out = np.full((2,), -9, dtype=np.int64)
ours_prod = fnp.prod(
    a,
    axis=1,
    dtype=np.int64,
    out=ours_prod_out,
    where=mask,
    initial=2,
)
expected_prod = np.prod(
    a,
    axis=1,
    dtype=np.int64,
    out=expected_prod_out,
    where=mask,
    initial=2,
)

ok = (
    ours_sum is ours_sum_out
    and expected_sum is expected_sum_out
    and ours_prod is ours_prod_out
    and expected_prod is expected_prod_out
    and ours_sum_out.dtype == expected_sum_out.dtype
    and ours_prod_out.dtype == expected_prod_out.dtype
    and np.array_equal(ours_sum_out, expected_sum_out)
    and np.array_equal(ours_prod_out, expected_prod_out)
)
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "dtype= fallback should preserve NumPy out/where/initial semantics"
    );
    Ok(())
}
