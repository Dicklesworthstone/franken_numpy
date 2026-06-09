//! Conformance tests for numpy cumulative operations against NumPy oracle.
//!
//! Tests cumsum, cumprod, diff (diff is in here for completeness with incremental ops).

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
// cumsum
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cumsum_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.cumsum(a)
expected = np.cumsum(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum 1d should match numpy");
    Ok(())
}

#[test]
fn cumsum_2d_no_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumsum(a)
expected = np.cumsum(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumsum 2d no axis should flatten and match numpy"
    );
    Ok(())
}

#[test]
fn cumsum_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumsum(a, axis=0)
expected = np.cumsum(a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn cumsum_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumsum(a, axis=1)
expected = np.cumsum(a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum 2d axis=1 should match numpy");
    Ok(())
}

#[test]
fn cumsum_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0.1, 0.2, 0.3, 0.4])
result = fnp.cumsum(a)
expected = np.cumsum(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cumprod
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cumprod_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.cumprod(a)
expected = np.cumprod(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumprod 1d should match numpy");
    Ok(())
}

#[test]
fn cumprod_2d_no_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumprod(a)
expected = np.cumprod(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumprod 2d no axis should flatten and match numpy"
    );
    Ok(())
}

#[test]
fn cumprod_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumprod(a, axis=0)
expected = np.cumprod(a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumprod 2d axis=0 should match numpy"
    );
    Ok(())
}

#[test]
fn cumprod_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumprod(a, axis=1)
expected = np.cumprod(a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumprod 2d axis=1 should match numpy"
    );
    Ok(())
}

#[test]
fn cumprod_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.1, 1.2, 1.3, 1.4])
result = fnp.cumprod(a)
expected = np.cumprod(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumprod float should match numpy");
    Ok(())
}

#[test]
fn cumprod_with_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 0, 4, 5])
result = fnp.cumprod(a)
expected = np.cumprod(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumprod with zero should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cumsum_last_equals_sum() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
cumsum_result = fnp.cumsum(a)
sum_result = fnp.sum(a)
print(cumsum_result[-1] == sum_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "last cumsum element should equal sum"
    );
    Ok(())
}

#[test]
fn cumprod_last_equals_prod() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
cumprod_result = fnp.cumprod(a)
prod_result = fnp.prod(a)
print(cumprod_result[-1] == prod_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "last cumprod element should equal prod"
    );
    Ok(())
}

#[test]
fn cumsum_diff_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
# For cumulative sum: diff(cumsum(a)) gives a[1:] (the original array without first element)
a = np.array([1, 2, 3, 4, 5])
cumsum_a = fnp.cumsum(a)
diff_cumsum = fnp.diff(cumsum_a)
# diff of cumsum gives the original elements (except first)
print(np.array_equal(diff_cumsum, a[1:]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff(cumsum(a)) should equal a[1:]");
    Ok(())
}

#[test]
fn cumsum_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
fnp_result = fnp.cumsum(z)
np_result = np.cumsum(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum complex should match numpy");
    Ok(())
}

#[test]
fn cumprod_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 2+0j, 0+1j], dtype=np.complex128)
fnp_result = fnp.cumprod(z)
np_result = np.cumprod(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumprod complex should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cumulative_sum / cumulative_prod (NumPy 2.0 Array-API names, native-wired)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cumulative_sum_prod_match_numpy_across_dtype_axis_and_include_initial() -> Result<(), String> {
    let script = fnp_script(
        r#"
ok = True
rng = np.random.default_rng(7)
for op in ["cumulative_sum", "cumulative_prod"]:
    ffn = getattr(fnp, op); nfn = getattr(np, op)
    for dt in [np.float64, np.float32, np.int8, np.int32, np.uint8, np.int64, np.bool_]:
        for shape in [(20,), (6, 5), (4, 3, 2)]:
            if dt == np.bool_:
                a = rng.integers(0, 2, shape).astype(dt)
            elif np.issubdtype(dt, np.integer):
                a = rng.integers(0, 4, shape).astype(dt)
            else:
                a = rng.standard_normal(shape).astype(dt)
            axes = [None] if len(shape) == 1 else list(range(len(shape)))
            for ax in axes:
                for inc in (False, True):
                    kw = {"include_initial": inc}
                    if ax is not None:
                        kw["axis"] = ax
                    f = np.asarray(ffn(a, **kw)); n = np.asarray(nfn(a, **kw))
                    if f.dtype != n.dtype or f.shape != n.shape or not np.allclose(f, n, rtol=1e-6, atol=1e-6, equal_nan=True):
                        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumulative_sum/prod must match numpy across dtype/axis/include_initial"
    );
    Ok(())
}

#[test]
fn cumulative_sum_axis_none_on_nd_raises_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
ok = True
for op in ["cumulative_sum", "cumulative_prod"]:
    try:
        getattr(fnp, op)(np.arange(12).reshape(3, 4))
        ok = False  # numpy raises ValueError here
    except ValueError:
        pass
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumulative_sum/prod with axis=None on ndim>1 must raise ValueError like numpy"
    );
    Ok(())
}
