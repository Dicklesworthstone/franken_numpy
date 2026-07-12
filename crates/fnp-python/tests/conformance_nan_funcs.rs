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
         {I4}try:\n\
         {I8}value = {call_expr}\n\
         {I8}arr = np.asarray(value)\n\
         {I8}print('ok')\n\
         {I8}print(type(value).__name__)\n\
         {I8}print(str(arr.dtype))\n\
         {I8}print(tuple(arr.shape))\n\
         {I8}print(repr(arr.tolist()))\n\
         {I4}except Exception as exc:\n\
         {I8}print('err')\n\
         {I8}print(type(exc).__name__)\n\
         outcome(op)",
        I4 = "    ",
        I8 = "        ",
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
        (
            "nanpercentile q sequence method keepdims",
            "np.nanpercentile",
            "nanpercentile",
            "",
            "op(np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]]), [25, 75], axis=1, method='nearest', keepdims=True)",
        ),
        (
            "nanpercentile out forwarding",
            "np.nanpercentile",
            "nanpercentile",
            "x = np.array([[1.0, np.nan], [3.0, 5.0]])\nout = np.empty((2,), dtype=np.float64)",
            "op(x, 50, axis=0, out=out)",
        ),
        (
            "nanquantile q sequence method",
            "np.nanquantile",
            "nanquantile",
            "",
            "op(((1.0, np.nan, 3.0), (4.0, 5.0, np.nan)), [0.25, 0.75], axis=0, method='lower')",
        ),
        (
            "nanquantile out forwarding",
            "np.nanquantile",
            "nanquantile",
            "x = np.array([[1.0, np.nan], [3.0, 5.0]])\nout = np.empty((2,), dtype=np.float64)",
            "op(x, 0.5, axis=0, out=out)",
        ),
        (
            "nanmedian keepdims",
            "np.nanmedian",
            "nanmedian",
            "",
            "op(((1.0, np.nan, 3.0), (6.0, 4.0, np.nan)), axis=1, keepdims=True)",
        ),
        (
            "nanmedian axis error type",
            "np.nanmedian",
            "nanmedian",
            "",
            "op([1.0, np.nan, 3.0], axis=2)",
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

#[test]
fn nanvar_nanstd_multiaxis_trailing_matches_numpy() -> Result<(), String> {
    // Exercises the native multi-axis trailing nanvar/nanstd fold (axis a tuple
    // resolving to the contiguous trailing axes) against numpy bit-exactly
    // (atol=0, equal_nan=True) incl dtype/shape: nanvar and nanstd, ddof 0/1,
    // keepdims, reversed axis order (symmetric), a 3-axis reduce, plain 2-D
    // axis=(0,1), a non-trailing axis fallthrough, and blocks containing NaN
    // (and an all-NaN block, which must defer + match numpy's NaN + warning).
    let script = fnp_script(
        r#"
import warnings
def same(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.allclose(a, b, rtol=0, atol=0, equal_nan=True)

rng = np.random.default_rng(11)
s3 = rng.standard_normal((4, 5, 6)); s3[s3 < -0.8] = np.nan
s4 = rng.standard_normal((2, 3, 4, 5)); s4[s4 > 1.0] = np.nan
m2 = rng.standard_normal((7, 8)); m2[0, 0] = np.nan
allnan = np.full((3, 4, 4), np.nan, dtype=np.float64)
allnan[1] = rng.standard_normal((4, 4))
cases = [
    (s3, (-2, -1), 0, False, False),
    (s3, (-2, -1), 1, True, False),
    (s3, (-1, -2), 0, False, False),
    (s4, (-3, -2, -1), 0, False, False),
    (s4, (-2, -1), 0, False, True),
    (m2, (0, 1), 0, False, False),
    (s3, (0, 1), 0, False, False),
    (allnan, (-2, -1), 0, False, False),
]
ok = True
for arr, axis, ddof, keepdims, use_std in cases:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_std:
            f = fnp.nanstd(arr, axis=axis, ddof=ddof, keepdims=keepdims)
            n = np.nanstd(arr, axis=axis, ddof=ddof, keepdims=keepdims)
        else:
            f = fnp.nanvar(arr, axis=axis, ddof=ddof, keepdims=keepdims)
            n = np.nanvar(arr, axis=axis, ddof=ddof, keepdims=keepdims)
    if not same(f, n):
        print("FAIL", axis, ddof, keepdims, use_std, np.asarray(f), np.asarray(n)); ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "multi-axis trailing nanvar/nanstd parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn nanvar_nanstd_axis0_first_axis_matches_numpy() -> Result<(), String> {
    // Exercises the native first-axis (axis=0) streaming nanvar/nanstd fold against
    // numpy bit-exactly (atol=0, equal_nan=True) incl dtype/shape: nanvar and nanstd,
    // ddof 0/1, keepdims, negative axis, 3-D axis=0, NaN-containing columns, an Inf
    // column, and an all-NaN column (which defers + matches numpy NaN + warning).
    let script = fnp_script(
        r#"
import warnings
def same(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.allclose(a, b, rtol=0, atol=0, equal_nan=True)

rng = np.random.default_rng(19)
m2 = rng.standard_normal((1000, 257)); m2[rng.random((1000, 257)) < 0.1] = np.nan
tall = rng.standard_normal((50000, 16)); tall[rng.random((50000, 16)) < 0.1] = np.nan
s3 = rng.standard_normal((64, 9, 7)); s3[rng.random((64, 9, 7)) < 0.1] = np.nan
infm = rng.standard_normal((40, 8)); infm[3, 2] = np.inf
allnan = rng.standard_normal((20, 5)); allnan[:, 2] = np.nan  # column 2 all NaN -> defer
ok = True
cases = [
    (m2, 0, 0, False, False),
    (m2, 0, 1, True, False),
    (m2, -2, 0, False, False),
    (tall, 0, 0, False, True),
    (s3, 0, 0, False, False),
    (infm, 0, 0, False, False),
    (allnan, 0, 0, False, False),
]
for arr, axis, ddof, keepdims, use_std in cases:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_std:
            f = fnp.nanstd(arr, axis=axis, ddof=ddof, keepdims=keepdims)
            n = np.nanstd(arr, axis=axis, ddof=ddof, keepdims=keepdims)
        else:
            f = fnp.nanvar(arr, axis=axis, ddof=ddof, keepdims=keepdims)
            n = np.nanvar(arr, axis=axis, ddof=ddof, keepdims=keepdims)
    if not same(f, n):
        print("FAIL", axis, ddof, keepdims, use_std, np.asarray(f), np.asarray(n)); ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "axis=0 nanvar/nanstd parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn nanmean_axis0_first_axis_matches_numpy() -> Result<(), String> {
    // Exercises the native first-axis (axis=0) streaming nanmean fold against numpy
    // bit-exactly (atol=0, equal_nan=True) incl dtype/shape: keepdims, negative axis,
    // 3-D axis=0, NaN columns, an Inf column, and an all-NaN column (-> NaN + "Mean of
    // empty slice" warning, computed directly as 0/0, not deferred).
    let script = fnp_script(
        r#"
import warnings
def same(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.allclose(a, b, rtol=0, atol=0, equal_nan=True)

rng = np.random.default_rng(23)
m2 = rng.standard_normal((1000, 257)); m2[rng.random((1000, 257)) < 0.1] = np.nan
tall = rng.standard_normal((50000, 16)); tall[rng.random((50000, 16)) < 0.1] = np.nan
s3 = rng.standard_normal((64, 9, 7)); s3[rng.random((64, 9, 7)) < 0.1] = np.nan
infm = rng.standard_normal((40, 8)); infm[3, 2] = np.inf
allnan = rng.standard_normal((20, 5)); allnan[:, 2] = np.nan  # column 2 all NaN
ok = True
cases = [
    (m2, 0, False),
    (m2, 0, True),
    (m2, -2, False),
    (tall, 0, True),
    (s3, 0, False),
    (infm, 0, False),
    (allnan, 0, False),
]
for arr, axis, keepdims in cases:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = fnp.nanmean(arr, axis=axis, keepdims=keepdims)
        n = np.nanmean(arr, axis=axis, keepdims=keepdims)
    if not same(f, n):
        print("FAIL", axis, keepdims, np.asarray(f), np.asarray(n)); ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "axis=0 nanmean parity should match numpy: {result}"
    );
    Ok(())
}

// The native parallel-across-lanes complex nanprod (np.nanprod) along the LAST contiguous axis must be
// byte-identical to numpy: numpy replaces NaN-complex with 1+0j and runs the slow multiply.reduce chain;
// this kernel replaces NaN inline during the identical per-lane sequential product. Exercises the engaged
// path (large arrays past the 1<<18 gate, c128 + c64, keepdims) with sprinkled NaNs, the NaN-edge lanes
// (all-NaN -> 1+0j, (nan,x), (x,nan), (inf,nan), overflow, zero), and every defer path (below gate,
// axis=0, flatten, non-contiguous) — all of which must equal numpy bit-for-bit.
#[test]
fn nanprod_complex_lastaxis_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib, warnings
rng = np.random.default_rng(20260701)
ok = True
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for cdt in (np.complex128, np.complex64):
        chunks_f, chunks_n = [], []
        def add(arr, **kw):
            chunks_f.append(np.ascontiguousarray(fnp.nanprod(arr, **kw)).tobytes())
            chunks_n.append(np.ascontiguousarray(np.nanprod(arr, **kw)).tobytes())
        # engaged last-axis path (rows*cols >= 1<<18) with sprinkled NaNs
        for shp in [(2000, 2000), (512, 2049), (2, 131072), (262144, 2), (100, 50, 80)]:
            x = (rng.standard_normal(shp) + 1j * rng.standard_normal(shp)).astype(cdt)
            x.ravel()[::11] = np.nan
            add(x, axis=-1)
            add(x, axis=-1, keepdims=True)
        # NaN-edge lanes
        xe = (rng.standard_normal((2000, 2000)) + 1j * rng.standard_normal((2000, 2000))).astype(cdt)
        xe[0, :] = np.nan
        xe[1, 5] = complex(np.nan, 1.0)
        xe[2, 7] = complex(1.0, np.nan)
        xe[3, 9] = complex(np.inf, np.nan)
        xe[4, :] = complex(1e30, 1e30) if cdt == np.complex128 else complex(1e18, 1e18)
        xe[5, 11] = complex(0.0, 0.0)
        add(xe, axis=1)
        # defer paths (must still equal numpy)
        sm = (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))).astype(cdt)
        add(sm, axis=1)
        big = (rng.standard_normal((2000, 2000)) + 1j * rng.standard_normal((2000, 2000))).astype(cdt)
        big.ravel()[::13] = np.nan
        add(big, axis=0)
        chunks_f.append(np.asarray(fnp.nanprod(big)).tobytes())
        chunks_n.append(np.asarray(np.nanprod(big)).tobytes())
        add(np.asfortranarray(big), axis=1)
        add(big[:, ::2], axis=1)
        if hashlib.sha256(b"".join(chunks_f)).hexdigest() != hashlib.sha256(b"".join(chunks_n)).hexdigest():
            ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native complex last-axis nanprod must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn flat_multi_quantile_and_weighted_average_track_numpy() -> Result<(), String> {
    // Convergence-sweep probe (2026-07-12): the last two ambiguous wide-rank rows.
    // quantile(array-q, flat) and percentile(list-q, flat) ride the shipped native
    // order-statistics path - assert byte parity and record the coarse A/B.
    // average(weights=) rides the extract path whose serial sum is NOT numpy's
    // pairwise order - assert allclose (documented mean-family tolerance) and
    // record the A/B so the extract-tax gap is measured, not assumed.
    let script = fnp_script(
        r#"
import time
import warnings
verdicts = []
rng = np.random.default_rng(20260712)
a = rng.standard_normal(8_000_000)
w = np.abs(rng.standard_normal(8_000_000)) + 0.01
qs = np.linspace(0.1, 0.9, 9)
# Byte-exact since the numpy_quantile_lerp fix (bead deadlock-audit-19jv4): the
# linear method now runs numpy's two-sided _lerp, so multi-q flat is tobytes-equal.
r, e = fnp.quantile(a, qs), np.quantile(a, qs)
if r.dtype != e.dtype or r.tobytes() != e.tobytes():
    verdicts.append("FAIL quantile9 bytes")
r, e = fnp.percentile(a, [25, 50, 75]), np.percentile(a, [25, 50, 75])
if r.dtype != e.dtype or r.tobytes() != e.tobytes():
    verdicts.append("FAIL percentile-trio bytes")
ra, ea = fnp.average(a, weights=w), np.average(a, weights=w)
if not np.allclose(ra, ea, rtol=1e-12):
    verdicts.append("FAIL average allclose")
# multi-q LAST-axis native path (fractions_last_axis): byte parity + q-first layout
m = rng.standard_normal((2896, 2896))
r, e = fnp.percentile(m, [25, 50, 75], axis=1), np.percentile(m, [25, 50, 75], axis=1)
if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL percentile3-ax1 bytes")
r, e = fnp.quantile(m, qs, axis=1), np.quantile(m, qs, axis=1)
if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL quantile9-ax1 bytes")
r, e = fnp.quantile(m, [0.5], axis=-1), np.quantile(m, [0.5], axis=-1)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL single-q-list ax-1 bytes")
mn = m.copy(); mn[7, 123] = np.nan
r, e = fnp.percentile(mn, [25, 75], axis=1), np.percentile(mn, [25, 75], axis=1)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL nan-lane bytes")
r, e = fnp.percentile(m, [25, 75], axis=0), np.percentile(m, [25, 75], axis=0)
if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL percentile-ax0 bytes")
r, e = fnp.quantile(m, qs, axis=0), np.quantile(m, qs, axis=0)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL quantile9-ax0 bytes")
mc = m.copy(); mc[123, 7] = np.nan
r, e = fnp.percentile(mc, [25, 75], axis=0), np.percentile(mc, [25, 75], axis=0)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL nan-column-ax0 bytes")
# N-D non-last-axis multi-q via the generalized strided-lane kernel
t3 = rng.standard_normal((64, 512, 256))
for tag, ax in (("3d-ax1", 1), ("3d-ax0", 0)):
    r, e = fnp.percentile(t3, [25, 50, 75], axis=ax), np.percentile(t3, [25, 50, 75], axis=ax)
    if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL {tag} bytes")
r, e = fnp.quantile(t3, qs, axis=1, keepdims=True), np.quantile(t3, qs, axis=1, keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL 3d-ax1-keepdims bytes")
t3n = t3.copy(); t3n[3, 100, 7] = np.nan
r, e = fnp.percentile(t3n, [25, 75], axis=1), np.percentile(t3n, [25, 75], axis=1)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL 3d-nan-lane bytes")
# nan N-D non-last-axis multi-q: compaction composed into the strided kernel
t3nn = t3.copy()
t3nn.ravel()[rng.integers(0, t3.size, 20000)] = np.nan
for tag, ax in (("3d-nanpct-ax1", 1), ("3d-nanpct-ax0", 0)):
    r, e = fnp.nanpercentile(t3nn, [25, 50, 75], axis=ax), np.nanpercentile(t3nn, [25, 50, 75], axis=ax)
    if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL {tag} bytes")
r, e = fnp.nanquantile(t3nn, qs, axis=1, keepdims=True), np.nanquantile(t3nn, qs, axis=1, keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL 3d-nan-keepdims bytes")
t3all = t3nn.copy(); t3all[5, :, 9] = np.nan
with warnings.catch_warnings(record=True) as wf:
    warnings.simplefilter("always")
    r = fnp.nanpercentile(t3all, [25, 75], axis=1)
with warnings.catch_warnings(record=True) as wn:
    warnings.simplefilter("always")
    e = np.nanpercentile(t3all, [25, 75], axis=1)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL 3d-all-nan-lane bytes")
if [str(w.message) for w in wf] != [str(w.message) for w in wn]:
    verdicts.append("FAIL 3d-all-nan-lane warnings")
r, e = fnp.quantile(m, qs, axis=1, keepdims=True), np.quantile(m, qs, axis=1, keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL keepdims-ax1 bytes")
r, e = fnp.percentile(m, [25, 75], axis=0, keepdims=True), np.percentile(m, [25, 75], axis=0, keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL keepdims-ax0 bytes")
# nan multi-q native path: per-lane NaN compaction + shared plan/lerp
mn = m.copy()
mn[rng.integers(0, 2896, 20000), rng.integers(0, 2896, 20000)] = np.nan
r, e = fnp.nanpercentile(mn, [25, 50, 75], axis=1), np.nanpercentile(mn, [25, 50, 75], axis=1)
if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL nanpercentile3-ax1 bytes")
r, e = fnp.nanquantile(mn, qs, axis=1), np.nanquantile(mn, qs, axis=1)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL nanquantile9-ax1 bytes")
r, e = fnp.nanquantile(mn.ravel(), [0.1, 0.5, 0.9]), np.nanquantile(mn.ravel(), [0.1, 0.5, 0.9])
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL nan-flat multi-q bytes")
allnan = mn.copy(); allnan[5, :] = np.nan
with warnings.catch_warnings(record=True) as wf:
    warnings.simplefilter("always")
    r = fnp.nanpercentile(allnan, [25, 75], axis=1)
with warnings.catch_warnings(record=True) as wn:
    warnings.simplefilter("always")
    e = np.nanpercentile(allnan, [25, 75], axis=1)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL all-nan-lane bytes")
if [str(w.message) for w in wf] != [str(w.message) for w in wn]:
    verdicts.append("FAIL all-nan-lane warnings")
# nan multi-q axis 0: block gather + compaction composition
r, e = fnp.nanpercentile(mn, [25, 50, 75], axis=0), np.nanpercentile(mn, [25, 50, 75], axis=0)
if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL nanpercentile3-ax0 bytes")
allnan0 = mn.copy(); allnan0[:, 5] = np.nan
with warnings.catch_warnings(record=True) as wf:
    warnings.simplefilter("always")
    r = fnp.nanquantile(allnan0, [0.25, 0.75], axis=0)
with warnings.catch_warnings(record=True) as wn:
    warnings.simplefilter("always")
    e = np.nanquantile(allnan0, [0.25, 0.75], axis=0)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL all-nan-column-ax0 bytes")
if [str(w.message) for w in wf] != [str(w.message) for w in wn]:
    verdicts.append("FAIL all-nan-column-ax0 warnings")
r, e = fnp.nanpercentile(mn, [25, 75], axis=1, keepdims=True), np.nanpercentile(mn, [25, 75], axis=1, keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL nan-keepdims-ax1 bytes")
r, e = fnp.nanquantile(mn, [0.1, 0.9], keepdims=True), np.nanquantile(mn, [0.1, 0.9], keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL nan-keepdims-flat bytes")
r, e = fnp.quantile(m, qs, keepdims=True), np.quantile(m, qs, keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL plain-keepdims-flat bytes")
r, e = fnp.percentile(a, [25, 50, 75], keepdims=True), np.percentile(a, [25, 50, 75], keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL plain-keepdims-flat-1d bytes")
# method='midpoint' native unlock: numpy _lerp(a,b,0.5), not (a+b)/2
for tag, call_f, call_n in (
    ("mid-flat", lambda: fnp.percentile(a, 37.3, method="midpoint"), lambda: np.percentile(a, 37.3, method="midpoint")),
    ("mid-ax1", lambda: fnp.percentile(m, 50, axis=1, method="midpoint"), lambda: np.percentile(m, 50, axis=1, method="midpoint")),
    ("mid-ax0", lambda: fnp.quantile(m, 0.66, axis=0, method="midpoint"), lambda: np.quantile(m, 0.66, axis=0, method="midpoint")),
    ("mid-exact-idx", lambda: fnp.percentile(a[:100001], 50, method="midpoint"), lambda: np.percentile(a[:100001], 50, method="midpoint")),
):
    rf, rn = call_f(), call_n()
    rf, rn = np.asarray(rf), np.asarray(rn)
    if rf.shape != rn.shape or rf.tobytes() != rn.tobytes():
        verdicts.append(f"FAIL {tag} bytes")
def best(fn, reps=5):
    fn(); best_s = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best_s = min(best_s, time.perf_counter() - t0)
    return best_s * 1000
for name, nf, ff in (
    ("quantile9", lambda: np.quantile(a, qs), lambda: fnp.quantile(a, qs)),
    ("percentile3_ax1", lambda: np.percentile(m, [25, 50, 75], axis=1), lambda: fnp.percentile(m, [25, 50, 75], axis=1)),
    ("quantile9_ax1", lambda: np.quantile(m, qs, axis=1), lambda: fnp.quantile(m, qs, axis=1)),
    ("nanpct3_ax1", lambda: np.nanpercentile(mn, [25, 50, 75], axis=1), lambda: fnp.nanpercentile(mn, [25, 50, 75], axis=1)),
    ("percentile3_ax0", lambda: np.percentile(m, [25, 50, 75], axis=0), lambda: fnp.percentile(m, [25, 50, 75], axis=0)),
    ("nanpct3_ax0", lambda: np.nanpercentile(mn, [25, 50, 75], axis=0), lambda: fnp.nanpercentile(mn, [25, 50, 75], axis=0)),
    ("quantile9_ax1_kd", lambda: np.quantile(m, qs, axis=1, keepdims=True), lambda: fnp.quantile(m, qs, axis=1, keepdims=True)),
    ("quantile9_flat_kd", lambda: np.quantile(m, qs, keepdims=True), lambda: fnp.quantile(m, qs, keepdims=True)),
    ("pct50_ax1_midpoint", lambda: np.percentile(m, 50, axis=1, method="midpoint"), lambda: fnp.percentile(m, 50, axis=1, method="midpoint")),
    ("pct3_3d_ax1", lambda: np.percentile(t3, [25, 50, 75], axis=1), lambda: fnp.percentile(t3, [25, 50, 75], axis=1)),
    ("nanpct3_3d_ax1", lambda: np.nanpercentile(t3nn, [25, 50, 75], axis=1), lambda: fnp.nanpercentile(t3nn, [25, 50, 75], axis=1)),
    ("percentile3", lambda: np.percentile(a, [25, 50, 75]), lambda: fnp.percentile(a, [25, 50, 75])),
    ("avg_weights", lambda: np.average(a, weights=w), lambda: fnp.average(a, weights=w)),
):
    tn, tf = best(nf), best(ff)
    print(f"SURFACE_PROBE_AB row={name} numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces SURFACE_PROBE_AB rows under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "flat multi-quantile/percentile must stay byte-exact and weighted average allclose: {result}"
    );
    Ok(())
}
