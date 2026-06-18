//! Conformance tests for numpy nan-ignoring functions against NumPy oracle.
//!
//! Tests nansum, nanmean, nanstd, nanvar, nanmin, nanmax, nanargmin, nanargmax,
//! nanprod, nancumsum, nancumprod, nanmedian, nanpercentile, nanquantile.

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

fn outcome_body(setup: &str, call_expr: &str) -> String {
    format!(
        "{setup}\n\
         def outcome(op):\n\
             try:\n\
                 value = {call_expr}\n\
                 arr = np.asarray(value)\n\
                 print('ok')\n\
                 print(type(value).__name__)\n\
                 print(str(arr.dtype))\n\
                 print(tuple(arr.shape))\n\
                 print(repr(arr.tolist()))\n\
             except Exception as exc:\n\
                 print('err')\n\
                 print(type(exc).__name__)\n\
         outcome(op)"
    )
}

fn numpy_outcome_script(function_expr: &str, setup: &str, call_expr: &str) -> String {
    format!(
        "import numpy as np\nop = {function_expr}\n{}",
        outcome_body(setup, call_expr)
    )
}

fn fnp_outcome_script(function_name: &str, setup: &str, call_expr: &str) -> String {
    fnp_script(format!(
        "op = fnp.{function_name}\n{}",
        outcome_body(setup, call_expr)
    ))
}

#[test]
fn nan_function_keyword_outcomes_match_numpy() -> Result<(), String> {
    let cases = [
        (
            "nansum",
            "np.nansum",
            "nansum",
            "",
            "op([[1.0, np.nan], [3.0, 4.0]], axis=1, keepdims=True)",
        ),
        (
            "nanmean",
            "np.nanmean",
            "nanmean",
            "",
            "op(((1.0, np.nan), (3.0, 5.0)), axis=0, dtype=np.float64, keepdims=True)",
        ),
        (
            "nanstd where ddof",
            "np.nanstd",
            "nanstd",
            "x = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])\nmask = np.array([[True, False, True], [True, True, False]])",
            "op(x, axis=1, ddof=1, where=mask, keepdims=True)",
        ),
        (
            "nanmin keepdims",
            "np.nanmin",
            "nanmin",
            "",
            "op([[np.nan, 2.0], [3.0, 4.0]], axis=0, keepdims=True)",
        ),
        (
            "nanargmin keepdims",
            "np.nanargmin",
            "nanargmin",
            "",
            "op([[np.nan, 2.0], [3.0, 4.0]], axis=0, keepdims=True)",
        ),
        (
            "nanargmin all nan error type",
            "np.nanargmin",
            "nanargmin",
            "",
            "op([np.nan, np.nan])",
        ),
    ];

    for (label, numpy_name, fnp_name, setup, call_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script(numpy_name, setup, call_expr))?;
        let rust_result = numpy_oracle(&fnp_outcome_script(fnp_name, setup, call_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "nan function keyword outcome mismatch for {label}"
        );
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nansum
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nansum_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4])
result = fnp.nansum(a)
expected = np.nansum(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nansum basic should match numpy");
    Ok(())
}

#[test]
fn nansum_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, np.nan], [3, 4]])
result = fnp.nansum(a, axis=0)
expected = np.nansum(a, axis=0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nansum 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanmean
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanmean_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4])
result = fnp.nanmean(a)
expected = np.nanmean(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmean basic should match numpy");
    Ok(())
}

#[test]
fn nanmean_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, np.nan, 3], [4, 5, np.nan]])
result = fnp.nanmean(a, axis=1)
expected = np.nanmean(a, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmean 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanstd / nanvar
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanstd_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4, 5])
result = fnp.nanstd(a)
expected = np.nanstd(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanstd basic should match numpy");
    Ok(())
}

#[test]
fn nanvar_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4, 5])
result = fnp.nanvar(a)
expected = np.nanvar(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanvar basic should match numpy");
    Ok(())
}

#[test]
fn nanstd_ddof() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4, 5])
result = fnp.nanstd(a, ddof=1)
expected = np.nanstd(a, ddof=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanstd ddof should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanmin / nanmax
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanmin_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, np.nan, 5])
result = fnp.nanmin(a)
expected = np.nanmin(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmin basic should match numpy");
    Ok(())
}

#[test]
fn nanmax_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, np.nan, 5])
result = fnp.nanmax(a)
expected = np.nanmax(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmax basic should match numpy");
    Ok(())
}

#[test]
fn nanmin_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, np.nan, 3], [np.nan, 5, 6]])
result = fnp.nanmin(a, axis=1)
expected = np.nanmin(a, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmin 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanargmin / nanargmax
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanargmin_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.nan, 2, 1, np.nan, 5])
result = fnp.nanargmin(a)
expected = np.nanargmin(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanargmin basic should match numpy");
    Ok(())
}

#[test]
fn nanargmax_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.nan, 2, 5, np.nan, 1])
result = fnp.nanargmax(a)
expected = np.nanargmax(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanargmax basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanprod
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanprod_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4])
result = fnp.nanprod(a)
expected = np.nanprod(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanprod basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nancumsum / nancumprod
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nancumsum_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, 4])
result = fnp.nancumsum(a)
expected = np.nancumsum(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nancumsum basic should match numpy");
    Ok(())
}

#[test]
fn nancumprod_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, 4])
result = fnp.nancumprod(a)
expected = np.nancumprod(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nancumprod basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanmedian
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanmedian_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, 4, np.nan])
result = fnp.nanmedian(a)
expected = np.nanmedian(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmedian basic should match numpy");
    Ok(())
}

#[test]
fn nanmedian_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, np.nan, 3], [4, 5, np.nan]])
result = fnp.nanmedian(a, axis=1)
expected = np.nanmedian(a, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nanmedian 2d axis should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanpercentile / nanquantile
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanpercentile_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, 4, 5])
result = fnp.nanpercentile(a, 50)
expected = np.nanpercentile(a, 50)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nanpercentile basic should match numpy"
    );
    Ok(())
}

#[test]
fn nanquantile_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, 4, 5])
result = fnp.nanquantile(a, 0.5)
expected = np.nanquantile(a, 0.5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nanquantile basic should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanmean_no_nan_equals_mean() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])  # no NaN
nanmean_result = fnp.nanmean(a)
mean_result = np.mean(a)
print(np.allclose(nanmean_result, mean_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nanmean without NaN should equal mean"
    );
    Ok(())
}

#[test]
fn nanstd_squared_equals_nanvar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4, 5])
std = fnp.nanstd(a)
var = fnp.nanvar(a)
print(np.allclose(std**2, var))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanstd squared should equal nanvar");
    Ok(())
}

#[test]
fn nanpercentile_50_equals_nanmedian() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, 4, 5])
percentile = fnp.nanpercentile(a, 50)
median = fnp.nanmedian(a)
print(np.allclose(percentile, median))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nanpercentile 50 should equal nanmedian"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// All-NaN array edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nansum_all_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    a = np.array([np.nan, np.nan, np.nan])
    result = fnp.nansum(a)
    expected = np.nansum(a)
    print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nansum all-nan should match numpy");
    Ok(())
}

#[test]
fn nanmean_all_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    a = np.array([np.nan, np.nan, np.nan])
    result = fnp.nanmean(a)
    expected = np.nanmean(a)
    print(np.isnan(result) and np.isnan(expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmean all-nan should return nan");
    Ok(())
}

#[test]
fn nanprod_all_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    a = np.array([np.nan, np.nan, np.nan])
    result = fnp.nanprod(a)
    expected = np.nanprod(a)
    print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanprod all-nan should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Inf handling in nan-ignoring functions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nansum_with_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, np.inf, 4])
result = fnp.nansum(a)
expected = np.nansum(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nansum with inf should match numpy");
    Ok(())
}

#[test]
fn nanmean_with_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, np.inf, 4])
result = fnp.nanmean(a)
expected = np.nanmean(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmean with inf should match numpy");
    Ok(())
}

#[test]
fn nanmax_with_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, np.inf, 4])
result = fnp.nanmax(a)
expected = np.nanmax(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmax with inf should match numpy");
    Ok(())
}

#[test]
fn nanmin_with_neg_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, -np.inf, 4])
result = fnp.nanmin(a)
expected = np.nanmin(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmin with -inf should match numpy");
    Ok(())
}

#[test]
fn nansum_signed_zero_parity() -> Result<(), String> {
    // Test signed-zero behavior for nansum (NaN-ignoring sum)
    let script = fnp_script(
        r#"
# Signed-zero nansum semantics
tests = [
    ([0.0, 0.0, np.nan], False),      # nansum([0.0, 0.0, nan]) = 0.0 (positive)
    ([-0.0, -0.0, np.nan], True),     # nansum([-0.0, -0.0, nan]) = -0.0 (negative)
    ([0.0, -0.0, np.nan], False),     # nansum([0.0, -0.0, nan]) = 0.0 (IEEE 754)
    ([np.nan, np.nan], False),        # nansum([nan, nan]) = 0.0 (default initial)
]
all_pass = True
for values, expected_signbit in tests:
    arr = np.array(values)
    fnp_result = fnp.nansum(arr)
    np_result = np.nansum(arr)
    if np.signbit(fnp_result) != np.signbit(np_result):
        print(f"FAIL: nansum({values}) fnp signbit={np.signbit(fnp_result)} np signbit={np.signbit(np_result)}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nansum signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn nanprod_signed_zero_parity() -> Result<(), String> {
    // Test signed-zero behavior for nanprod (NaN-ignoring product)
    let script = fnp_script(
        r#"
# Signed-zero nanprod semantics (XOR sign rule)
tests = [
    ([0.0, 1.0, np.nan], False),      # nanprod([0.0, 1.0, nan]) = 0.0 (positive)
    ([-0.0, 1.0, np.nan], True),      # nanprod([-0.0, 1.0, nan]) = -0.0 (negative)
    ([0.0, -0.0, np.nan], True),      # nanprod([0.0, -0.0, nan]) = -0.0 (XOR)
    ([-0.0, -0.0, np.nan], False),    # nanprod([-0.0, -0.0, nan]) = 0.0 (XOR)
]
all_pass = True
for values, expected_signbit in tests:
    arr = np.array(values)
    fnp_result = fnp.nanprod(arr)
    np_result = np.nanprod(arr)
    if np.signbit(fnp_result) != np.signbit(np_result):
        print(f"FAIL: nanprod({values}) fnp signbit={np.signbit(fnp_result)} np signbit={np.signbit(np_result)}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nanprod signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn nancumsum_nancumprod_match_numpy_across_dtype_axis_nan_and_edges() -> Result<(), String> {
    // Locks the native zero-copy nan-aware scan wiring: NaN treated as the additive (0)
    // / multiplicative (1) identity, flatten on axis=None (any ndim), integer == cum*,
    // float32 / per-axis multi-dim defer to numpy. Covers leading-NaN, all-NaN, -0.0,
    // and empty inputs.
    let script = fnp_script(
        r#"
ok = True
rng = np.random.default_rng(0)
for op in ["nancumsum", "nancumprod"]:
    ffn = getattr(fnp, op); nfn = getattr(np, op)
    for dt in [np.float64, np.float32, np.int8, np.int32, np.uint8, np.int64, np.bool_]:
        for shape in [(20,), (6, 5)]:
            if dt == np.bool_:
                a = rng.integers(0, 2, shape).astype(dt)
            elif np.issubdtype(dt, np.integer):
                a = rng.integers(0, 4, shape).astype(dt)
            else:
                a = rng.standard_normal(shape).astype(dt)
                a.flat[0] = np.nan; a.flat[3] = np.nan
            axes = [None, 0] if len(shape) == 1 else [None, 0, 1]
            for ax in axes:
                kw = {} if ax is None else {"axis": ax}
                f = np.asarray(ffn(a, **kw)); n = np.asarray(nfn(a, **kw))
                if f.dtype != n.dtype or f.shape != n.shape or not np.allclose(f, n, rtol=1e-9, atol=1e-9, equal_nan=True):
                    ok = False
    for arr in [np.array([np.nan, np.nan]), np.array([-0.0, 1.0]), np.array([np.nan, 5.0, np.nan, 2.0]), np.array([])]:
        f = np.asarray(ffn(arr)); n = np.asarray(nfn(arr))
        if f.dtype != n.dtype or f.shape != n.shape or not np.array_equal(f, n, equal_nan=True):
            ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nancumsum/nancumprod must match numpy across dtype/axis/NaN patterns and edges"
    );
    Ok(())
}

/// Locks the zero-copy NON-LAST-AXIS nanmax/nanmin reduction (the branchless
/// f64::max/min + saw-OR strided path). A deterministic 2-D and 3-D f64 array with
/// scattered NaN / +-inf / an all-NaN slice, reduced over axis 0 (and a middle
/// axis), must be byte-identical to numpy nanmax/nanmin plus a sha256 golden.
#[test]
fn nanextreme_nonlast_axis_matches_numpy_bytes_and_golden() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib, warnings
warnings.filterwarnings("ignore")
s = 0x9E3779B97F4A7C15
def nxt():
    global s
    s = (s * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
    return s
A = np.empty((130, 71), dtype=np.float64)
for i in range(130):
    for j in range(71):
        A[i, j] = ((nxt() >> 11) / (1 << 53)) * 10.0 - 5.0
A[::13, 4] = np.nan
A[7, ::9] = np.inf
A[9, ::11] = -np.inf
A[:, 20] = np.nan          # all-NaN column -> NaN (axis 0)
B = np.empty((11, 13, 9), dtype=np.float64)
for x in np.ndindex(11, 13, 9):
    B[x] = ((nxt() >> 11) / (1 << 53)) * 6.0 - 3.0
B[:, 4, :] = np.nan        # all-NaN slab over axis 1
h = hashlib.sha256()
allmatch = True
for (arr, ax) in ((A, 0), (B, 0), (B, 1)):
    for fn, nf in ((fnp.nanmax, np.nanmax), (fnp.nanmin, np.nanmin)):
        r = np.asarray(fn(arr, axis=ax))
        e = np.asarray(nf(arr, axis=ax))
        if r.shape != e.shape or r.dtype != e.dtype or r.tobytes() != e.tobytes():
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
        "non-last-axis nanmax/nanmin must be byte-identical to numpy"
    );
    assert_eq!(
        lines.next().unwrap_or("").trim(),
        "6f715e77a90ef5083737c8ef7e03aa02a8cda84c6354328b16b66ccefd5fe908",
        "nanextreme non-last-axis golden sha256 drifted"
    );
    Ok(())
}
