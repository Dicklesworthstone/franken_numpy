//! Conformance tests for numpy apply/vectorize functions against NumPy oracle.
//!
//! Tests apply_along_axis, apply_over_axes, vectorize, select.

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
// apply_along_axis
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn apply_along_axis_python_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def apply_along_axis_outcome(fn, func1d, axis, arr):
    try:
        result = fn(func1d, axis, arr)
        out = np.asarray(result)
        return ("ok", type(result).__name__, str(out.dtype), tuple(out.shape), out.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))

cases = [
    ("list axis zero sum", lambda: (np.sum, 0, [[1, 2, 3], [4, 5, 6]])),
    ("tuple axis one mean", lambda: (np.mean, 1, ((1, 2, 3), (4, 5, 6)))),
    ("list negative axis max", lambda: (np.max, -1, [[1, 9, 3], [4, 2, 6]])),
    ("axis out of bounds", lambda: (np.sum, 3, [[1, 2, 3], [4, 5, 6]])),
]

ok = True
for label, factory in cases:
    func1d, axis, arr = factory()
    actual = apply_along_axis_outcome(fnp.apply_along_axis, func1d, axis, arr)
    func1d, axis, arr = factory()
    expected = apply_along_axis_outcome(np.apply_along_axis, func1d, axis, arr)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "apply_along_axis Python-container surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn apply_along_axis_sum() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.apply_along_axis(np.sum, 0, a)
expected = np.apply_along_axis(np.sum, 0, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "apply_along_axis sum should match numpy"
    );
    Ok(())
}

#[test]
fn apply_along_axis_mean() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.apply_along_axis(np.mean, 1, a)
expected = np.apply_along_axis(np.mean, 1, a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "apply_along_axis mean should match numpy"
    );
    Ok(())
}

#[test]
fn apply_along_axis_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.apply_along_axis(np.sum, 2, a)
expected = np.apply_along_axis(np.sum, 2, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "apply_along_axis 3d should match numpy"
    );
    Ok(())
}

#[test]
fn apply_along_axis_negative_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.apply_along_axis(np.sum, -1, a)
expected = np.apply_along_axis(np.sum, -1, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "apply_along_axis negative axis should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_over_axes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn apply_over_axes_python_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def apply_over_axes_outcome(fn, func, arr, axes):
    try:
        result = fn(func, arr, axes)
        out = np.asarray(result)
        return ("ok", type(result).__name__, str(out.dtype), tuple(out.shape), out.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))

cases = [
    ("list single axis", lambda: (np.sum, [[1, 2, 3], [4, 5, 6]], 0)),
    ("list multiple axes", lambda: (np.sum, np.arange(24).reshape(2, 3, 4).tolist(), [0, 2])),
    ("tuple mean axis", lambda: (np.mean, (((1, 2), (3, 4)), ((5, 6), (7, 8))), [1])),
    ("axis out of bounds", lambda: (np.sum, [[1, 2, 3], [4, 5, 6]], [3])),
]

ok = True
for label, factory in cases:
    func, arr, axes = factory()
    actual = apply_over_axes_outcome(fnp.apply_over_axes, func, arr, axes)
    func, arr, axes = factory()
    expected = apply_over_axes_outcome(np.apply_over_axes, func, arr, axes)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "apply_over_axes Python-container surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn apply_over_axes_sum_single() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.apply_over_axes(np.sum, a, 0)
expected = np.apply_over_axes(np.sum, a, 0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "apply_over_axes sum single axis should match numpy"
    );
    Ok(())
}

#[test]
fn apply_over_axes_sum_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.apply_over_axes(np.sum, a, [0, 2])
expected = np.apply_over_axes(np.sum, a, [0, 2])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "apply_over_axes sum multiple axes should match numpy"
    );
    Ok(())
}

#[test]
fn apply_over_axes_mean() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4).astype(float)
result = fnp.apply_over_axes(np.mean, a, [1, 2])
expected = np.apply_over_axes(np.mean, a, [1, 2])
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "apply_over_axes mean should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// vectorize
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn vectorize_exists() -> Result<(), String> {
    let script = fnp_script(
        r#"
# fnp.vectorize has a different API than np.vectorize
# Just verify it exists and is callable
print(callable(fnp.vectorize))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vectorize should be callable");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// select
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn select_python_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def select_outcome(fn, condlist, choicelist, **kwargs):
    try:
        result = fn(condlist, choicelist, **kwargs)
        arr = np.asarray(result)
        return ("ok", type(result).__name__, str(arr.dtype), tuple(arr.shape), arr.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))

cases = [
    ("list conditions list choices", lambda: ([[True, False, True], [False, True, False]], [[1, 2, 3], [10, 20, 30]], {"default": 0})),
    (
        "ndarray conditions choices",
        lambda: ([np.array([False, True, True]), np.array([True, False, False])], [np.array([1, 2, 3], dtype=np.int16), np.array([10, 20, 30], dtype=np.int16)], {"default": -5}),
    ),
    ("tuple conditions tuple choices", lambda: (((True, False, False), (False, True, True)), ((1.5, 2.5, 3.5), (10.5, 20.5, 30.5)), {})),
    ("scalar choices default", lambda: ([[True, False, True], [False, True, False]], [1, 2], {"default": -1})),
    ("scalar condition broadcasts", lambda: ([True, False], [np.array([1, 2, 3]), np.array([4, 5, 6])], {"default": 0})),
    ("mixed numeric promotion", lambda: ([[False, True, False], [True, False, False]], [1.5, [1, 2, 3]], {"default": -1})),
    ("nested list choices", lambda: ([[[True, False], [False, True]]], [[["a", "b"], ["c", "d"]]], {"default": "fallback"})),
    ("string choices default", lambda: ([[True, False, True]], [["alpha", "beta", "gamma"]], {"default": "fallback"})),
    ("non-bool condition error", lambda: ([[1, 0, 1]], [[1, 2, 3]], {})),
    ("length mismatch error", lambda: ([[True, False, True], [False, True, False]], [[1, 2, 3]], {})),
]

ok = True
for label, factory in cases:
    condlist, choicelist, kwargs = factory()
    actual = select_outcome(fnp.select, condlist, choicelist, **kwargs)
    condlist, choicelist, kwargs = factory()
    expected = select_outcome(np.select, condlist, choicelist, **kwargs)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "select Python-container surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn select_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.arange(10)
condlist = [x < 3, x > 5]
choicelist = [x, x**2]
result = fnp.select(condlist, choicelist)
expected = np.select(condlist, choicelist)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "select basic should match numpy");
    Ok(())
}

#[test]
fn select_with_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.arange(10)
condlist = [x < 3, x > 5]
choicelist = [x, x**2]
result = fnp.select(condlist, choicelist, default=-1)
expected = np.select(condlist, choicelist, default=-1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "select with default should match numpy"
    );
    Ok(())
}

#[test]
fn select_three_conditions() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.arange(10)
condlist = [x < 2, (x >= 2) & (x < 6), x >= 6]
choicelist = [0, 1, 2]
result = fnp.select(condlist, choicelist)
expected = np.select(condlist, choicelist)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "select three conditions should match numpy"
    );
    Ok(())
}

#[test]
fn select_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.arange(12).reshape(3, 4)
condlist = [x < 3, x > 8]
choicelist = [x * 10, x * 100]
result = fnp.select(condlist, choicelist)
expected = np.select(condlist, choicelist)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "select 2d should match numpy");
    Ok(())
}

#[test]
fn select_string_choices_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
condlist = [np.array([True, False, True])]
choicelist = [np.array(["alpha", "beta", "gamma"])]
result = fnp.select(condlist, choicelist, default="fallback")
expected = np.select(condlist, choicelist, default="fallback")
print(np.array_equal(result, expected) and np.array_equal([result.dtype.str], [expected.dtype.str]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "select should preserve NumPy string choice behavior"
    );
    Ok(())
}

#[test]
fn select_string_default_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
condlist = [np.array([True, False, True])]
choicelist = [np.array([10, 20, 30])]
result = fnp.select(condlist, choicelist, default="fallback")
expected = np.select(condlist, choicelist, default="fallback")
print(np.array_equal(result, expected) and np.array_equal([result.dtype.str], [expected.dtype.str]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "select should preserve NumPy string default promotion"
    );
    Ok(())
}

#[test]
fn select_rejects_non_bool_numeric_condition_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def classify(func):
    try:
        func()
    except Exception as exc:
        return type(exc).__name__
    return "ok"

condlist = [np.array([0, 1, 0])]
choicelist = [np.array([10, 20, 30])]
result = classify(lambda: fnp.select(condlist, choicelist, default=-1))
expected = classify(lambda: np.select(condlist, choicelist, default=-1))
print(result in (expected,))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "select should reject non-bool numeric conditions like NumPy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn apply_along_axis_vs_direct_sum() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
apply_result = fnp.apply_along_axis(np.sum, 0, a)
direct_result = fnp.sum(a, axis=0)
print(np.array_equal(apply_result, direct_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "apply_along_axis sum should equal direct sum"
    );
    Ok(())
}

#[test]
fn select_vs_where() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.arange(10)
condlist = [x < 5]
choicelist = [x * 2]
select_result = fnp.select(condlist, choicelist, default=0)
where_result = fnp.where(x < 5, x * 2, 0)
print(np.array_equal(select_result, where_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "select should equal where for single condition"
    );
    Ok(())
}

#[test]
fn select_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
condlist = [np.array([True, False]), np.array([False, True])]
choicelist = [np.array([1+1j, 2+2j], dtype=np.complex128), np.array([3+3j, 4+4j], dtype=np.complex128)]
fnp_result = fnp.select(condlist, choicelist)
np_result = np.select(condlist, choicelist)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "select complex should match numpy");
    Ok(())
}
