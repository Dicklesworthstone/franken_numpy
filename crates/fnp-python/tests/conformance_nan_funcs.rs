//! Conformance tests for numpy nan-ignoring functions against NumPy oracle.
//!
//! Tests nansum, nanmean, nanstd, nanvar, nanmin, nanmax, nanargmin, nanargmax,
//! nanprod, nancumsum, nancumprod, nanmedian, nanpercentile, nanquantile.
//!
//! `flat_multi_quantile_and_weighted_average_track_numpy` — the single test that made this
//! shard exceed a 120s budget — now lives in the companion shard
//! `conformance_nan_funcs_wide.rs`. Its array sizes are LOAD-BEARING (two 8M-element
//! arrays, a 2896x2896 matrix, a 64x512x256 tensor) because they cross the native parallel
//! dispatch gates, so it could not be shrunk without moving onto the serial path; it was
//! separated rather than weakened. Both shards run by default and coverage is unchanged.
//! Split under bead `deadlock-audit-syi8e`.
//!
//! RUNTIME IS HOST-SCOPED — DO NOT QUOTE A NUMBER WITHOUT ITS HOST. These same 41 tests
//! measured **9.03s** on 2026-08-09 (CreamGlen's box), **183.46s** on rch worker vmi1153651
//! and **617.62s** on rch worker vmi1152480, both on 2026-08-15, from one `cargo test` with
//! `--report-time` under default test parallelism. That is a 68x spread for identical
//! source, and the ranking moves too: on vmi1152480 five tests each exceeded 60s
//! (nanprod_complex_lastaxis 601.7s, int_nanargmax_nanargmin 345.9s,
//! narrow_int_argextreme_nonlast_axis 191.5s, nan_function_keyword_outcomes 186.7s,
//! nanvar_nanstd_axis0_first_axis 69.2s) while the same tests on vmi1153651 topped out at
//! 173.1s. "One slow test" is a property of the host that observed it, not of this file.
//! Each test shells out to `python3`, so a contended worker inflates every one of them.
//!
//! The split itself was measured the only way such a comparison is valid — parent and
//! companion in ONE `cargo test` invocation on ONE worker (vmi1153651): parent 183.46s,
//! companion 676.29s, so the companion carries 78.7% of the family's wall time. Never
//! compare a time from one worker against a time from another; see the fleet directive on
//! bead `deadlock-audit-alold`.
//!
//! Keep this shard's own work small: a new probe over multi-million-element arrays belongs
//! in the `_wide` companion. And whenever you run these under a time cap, CHECK THAT THE
//! BINARY REPORTED — a cap kills a shard mid-execution and the run then prints no
//! `test result:` line for it at all, which reads exactly like a pass if you are only
//! grepping for failures.

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
fn int_nanargmax_nanargmin_route_to_argextreme_bit_exact_matches_numpy() -> Result<(), String> {
    // Integer/bool nanargmax/nanargmin route to fnp's argmax/argmin (numpy's
    // _replace_nan returns non-inexact arrays untouched, so the nan* forms
    // equal the plain reductions byte-exactly); fnp's native int/bool axis
    // kernels cover the strided forms numpy runs serially. Ties (first-max),
    // keepdims, flat, and float inputs (nan machinery intact) all pinned.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(193)
verdicts = []
M = rng.integers(-2**40, 2**40, (2048, 1024))
for fname, nfname in (("nanargmax", "nanargmax"), ("nanargmin", "nanargmin")):
    ff = getattr(fnp, fname); nf = getattr(np, nfname)
    for kw in [dict(), dict(axis=0), dict(axis=1), dict(axis=-1), dict(axis=0, keepdims=True)]:
        r = ff(M, **kw); e = nf(M, **kw)
        ra = np.asarray(r); ea = np.asarray(e)
        if ra.dtype != ea.dtype or ra.shape != ea.shape or ra.tobytes() != ea.tobytes():
            verdicts.append(f"FAIL {fname} {kw}")
# ties resolve to FIRST occurrence
T = np.zeros((512, 512), dtype=np.int64); T[7] = 5; T[300] = 5
if np.asarray(fnp.nanargmax(T, axis=0)).tobytes() != np.asarray(np.nanargmax(T, axis=0)).tobytes():
    verdicts.append("FAIL tie first-max")
# bool + narrow widths
B = rng.random((2048, 1024)) > 0.9
if np.asarray(fnp.nanargmax(B, axis=0)).tobytes() != np.asarray(np.nanargmax(B, axis=0)).tobytes():
    verdicts.append("FAIL bool")
M8 = rng.integers(-100, 100, (2048, 1024)).astype(np.int8)
if np.asarray(fnp.nanargmin(M8, axis=1)).tobytes() != np.asarray(np.nanargmin(M8, axis=1)).tobytes():
    verdicts.append("FAIL int8")
# float inputs keep the nan machinery (regression)
F = rng.standard_normal((1024, 512)); F[rng.random((1024, 512)) < 0.01] = np.nan
if np.asarray(fnp.nanargmax(F, axis=1)).tobytes() != np.asarray(np.nanargmax(F, axis=1)).tobytes():
    verdicts.append("FAIL float nan regression")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.integers(-2**40, 2**40, (8192, 4096))
tn = best(lambda: np.nanargmax(W, axis=0)); tf = best(lambda: fnp.nanargmax(W, axis=0))
print(f"NANARGMAX_INT_AX0_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces NANARGMAX_INT_AX0_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "int nanargmax/nanargmin routing must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn narrow_int_argextreme_nonlast_axis_bit_exact_matches_numpy() -> Result<(), String> {
    // The narrow-int (1/2-byte) argmax/argmin axis delegate is now LAST-axis
    // only; non-last axes engage the native narrow arms (numpy's strided walk
    // probed 249.1ms int16 axis=0 at (8192,4096) vs 4.4ms last-axis). First-
    // occurrence ties and the last-axis delegate stay pinned. Also reaches
    // through nanargmax/nanargmin via the int routing.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(199)
verdicts = []
for dt in [np.int16, np.int8, np.uint16, np.uint8]:
    info = np.iinfo(dt)
    M = rng.integers(info.min, int(info.max) + 1, (2048, 1024)).astype(dt)
    for fname in ("argmax", "argmin"):
        ff = getattr(fnp, fname); nf = getattr(np, fname)
        for ax in (0, -2):
            r = ff(M, axis=ax); e = nf(M, axis=ax)
            ra = np.asarray(r); ea = np.asarray(e)
            if ra.dtype != ea.dtype or ra.tobytes() != ea.tobytes():
                verdicts.append(f"FAIL {fname} {dt.__name__} ax={ax}")
        # last axis stays a byte-identical delegate
        if np.asarray(ff(M, axis=1)).tobytes() != np.asarray(nf(M, axis=1)).tobytes():
            verdicts.append(f"FAIL {fname} {dt.__name__} last-axis delegate")
# 3-D middle axis + dense ties (first occurrence)
M3 = rng.integers(-100, 100, (64, 256, 256)).astype(np.int16)
if np.asarray(fnp.argmax(M3, axis=1)).tobytes() != np.asarray(np.argmax(M3, axis=1)).tobytes():
    verdicts.append("FAIL 3-D mid-axis")
T = np.zeros((512, 512), dtype=np.int8); T[9] = 5; T[400] = 5
if np.asarray(fnp.argmax(T, axis=0)).tobytes() != np.asarray(np.argmax(T, axis=0)).tobytes():
    verdicts.append("FAIL tie first-max")
if np.asarray(fnp.nanargmax(T, axis=0)).tobytes() != np.asarray(np.nanargmax(T, axis=0)).tobytes():
    verdicts.append("FAIL nanargmax narrow ax0")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.integers(-30000, 30000, (8192, 4096)).astype(np.int16)
tn = best(lambda: np.argmax(W, axis=0)); tf = best(lambda: fnp.argmax(W, axis=0))
print(f"ARGMAX_INT16_AX0_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces ARGMAX_INT16_AX0_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "narrow-int non-last-axis argextreme must be bit-identical to numpy: {result}"
    );
    Ok(())
}

// numpy's nan reductions take initial= and where= (nanmean takes where= only).
// fnp did not declare them, so the documented call raised TypeError here while
// numpy answered - and the native kernels do not implement either, so a
// non-None value has to delegate the same way dtype=/out= already do. This walks
// both the values and the surfaces where numpy is deliberately awkward: an
// all-False where= (nansum gives 0.0, nanmean gives nan + a RuntimeWarning),
// and nanmax(where=) WITHOUT initial=, which numpy rejects because fmax has no
// identity - fnp must reproduce that ValueError rather than inventing an answer.
#[test]
fn nan_reduction_initial_and_where_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform
import warnings

x = np.array([1.0, 2.0, np.nan, 4.0])
mask = np.array([True, False, True, True])
all_false = np.zeros(4, dtype=bool)
finite = np.array([1.0, 2.0, 3.0, 4.0])
with_zero = np.array([1.0, 0.0, 3.0])
y = np.array([[1.0, np.nan], [3.0, 4.0]])
mask_2d = np.array([[True, True], [False, True]])

cases = [
    ("nansum where", "nansum", (x,), {"where": mask}),
    ("nansum initial", "nansum", (x,), {"initial": 10.0}),
    ("nansum initial+where", "nansum", (x,), {"where": mask, "initial": 10.0}),
    ("nansum where all-false", "nansum", (x,), {"where": all_false}),
    ("nansum where keepdims", "nansum", (x,), {"where": mask, "keepdims": True}),
    ("nansum 2-D axis where", "nansum", (y,), {"axis": 1, "where": mask_2d}),
    ("nanmax initial", "nanmax", (x,), {"initial": 99.0}),
    ("nanmax initial+where", "nanmax", (x,), {"where": mask, "initial": -np.inf}),
    ("nanmax where without initial", "nanmax", (x,), {"where": mask}),
    ("nanmin initial", "nanmin", (x,), {"initial": -99.0}),
    ("nanmin initial+where", "nanmin", (x,), {"where": mask, "initial": np.inf}),
    ("nanmin where without initial", "nanmin", (x,), {"where": mask}),
    ("nanmean where", "nanmean", (x,), {"where": mask}),
    ("nanmean where all-false", "nanmean", (x,), {"where": all_false}),
    ("nanmean 2-D axis where", "nanmean", (y,), {"axis": 0, "where": mask_2d}),
    ("nanmean where keepdims", "nanmean", (x,), {"where": mask, "keepdims": True}),
    # initial=None IS "not passed at all" for numpy (nanmax(initial=None) ==
    # nanmax(x)), so those cases sit alongside the where=None ones below.
    ("nanmax initial=None", "nanmax", (x,), {"initial": None}),
    ("nanmin initial=None", "nanmin", (x,), {"initial": None}),
    ("nansum initial=None", "nansum", (x,), {"initial": None}),
    # where=None is NOT "not passed": numpy reads it as a mask that selects
    # nothing, and the answer differs per function - nansum 0.0, nanprod 1.0,
    # nanmean/nanstd/nanvar nan, nanmax/nanmin a ValueError because fmax/fmin
    # have no identity. Telling it from an omitted argument is what
    # deadlock-audit-where-none-vs-absent-sentinel-bbk62 added the three-state
    # WhereArg for; these cases are that bead's closing probe and would all have
    # returned the UNMASKED answer before it.
    ("nansum where=None", "nansum", (x,), {"where": None}),
    ("nanmean where=None", "nanmean", (x,), {"where": None}),
    ("nanprod where=None", "nanprod", (x,), {"where": None}),
    ("nanstd where=None", "nanstd", (x,), {"where": None}),
    ("nanvar where=None", "nanvar", (x,), {"where": None}),
    ("nanmax where=None", "nanmax", (x,), {"where": None}),
    ("nanmin where=None", "nanmin", (x,), {"where": None}),
    ("nansum where=None keepdims", "nansum", (x,), {"where": None, "keepdims": True}),
    ("nanstd where=None ddof", "nanstd", (x,), {"where": None, "ddof": 1}),
    # And the omitted spelling for the same functions, so the fix is proven to
    # have changed only the explicit-None case.
    ("nansum omitted", "nansum", (x,), {}),
    ("nanmean omitted", "nanmean", (x,), {}),
    ("nanprod omitted", "nanprod", (x,), {}),
    ("nanstd omitted", "nanstd", (x,), {}),
    ("nanvar omitted", "nanvar", (x,), {}),
    # The NON-nan reductions have the same numpy semantic - where=None reduces
    # over nothing, so sum gives 0.0, prod 1.0, mean/std/var raise, max/min
    # raise, any gives False and all gives the identity True. These wrappers
    # take **kwargs and gate every native path on the kwargs dict being empty,
    # so a supplied where= (None included) already delegates - but nothing
    # asserted that, and the assumption is exactly what
    # deadlock-audit-nonnan-reductions-where-none-nh23c was filed over. These
    # cases hold it.
    #
    # `all` uses an input CONTAINING A ZERO on purpose: all(where=None) returns
    # the identity True regardless of input, which coincides with the unmasked
    # answer whenever that answer is already True. An all-nonzero input would
    # make this case vacuous.
    ("sum where=None", "sum", (finite,), {"where": None}),
    ("sum omitted", "sum", (finite,), {}),
    ("prod where=None", "prod", (finite,), {"where": None}),
    ("prod omitted", "prod", (finite,), {}),
    ("mean where=None", "mean", (finite,), {"where": None}),
    ("mean omitted", "mean", (finite,), {}),
    ("std where=None", "std", (finite,), {"where": None}),
    ("var where=None", "var", (finite,), {"where": None}),
    ("max where=None", "max", (finite,), {"where": None}),
    ("min where=None", "min", (finite,), {"where": None}),
    ("any where=None", "any", (finite,), {"where": None}),
    ("any omitted", "any", (finite,), {}),
    ("all where=None with a zero", "all", (with_zero,), {"where": None}),
    ("all omitted with a zero", "all", (with_zero,), {}),
    ("sum where=mask", "sum", (finite,), {"where": mask}),
    ("all where=mask with a zero", "all", (with_zero,), {"where": np.array([True, False, True])}),
]

def outcome(module, name, args, kwargs):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = np.asarray(getattr(module, name)(*args, **kwargs))
        return ("ok", str(result.dtype), tuple(result.shape), result.tobytes().hex())
    except Exception as exc:
        return ("err", type(exc).__name__)

ok = True
for label, name, args, kwargs in cases:
    actual = outcome(fnp, name, args, kwargs)
    expected = outcome(np, name, args, kwargs)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False
print(ok)
print("oracle", platform.node(), np.__version__)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "nan reduction initial=/where= surfaces should match numpy ({provenance}): {result}"
    );
    Ok(())
}
