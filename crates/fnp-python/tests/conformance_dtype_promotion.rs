//! Conformance tests for dtype promotion in reduction operations.

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
    assert_eq!(result.trim(), "True", "sum of int32 should promote to int64");
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
    assert_eq!(result.trim(), "True", "sum of int16 should promote to int64");
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
    assert_eq!(result.trim(), "True", "sum of uint8 should promote to uint64");
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
    assert_eq!(result.trim(), "True", "prod of int32 should promote to int64");
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
    assert_eq!(result.trim(), "True", "cumsum of int32 should promote to int64");
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
    assert_eq!(result.trim(), "True", "cumprod of int32 should promote to int64");
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
