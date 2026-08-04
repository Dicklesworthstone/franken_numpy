//! Conformance tests for numpy split functions against NumPy oracle.
//!
//! Tests split, array_split, hsplit, vsplit, dsplit.

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
// split
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn split_helpers_python_container_and_index_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def clean(value):
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value

def normalize_parts(parts):
    normalized = []
    for part in parts:
        array = np.asarray(part)
        normalized.append((str(array.dtype), tuple(array.shape), clean(array.tolist())))
    return normalized

def outcome(split_fn, *args, **kwargs):
    try:
        return ("ok", normalize_parts(split_fn(*args, **kwargs)))
    except Exception as exc:
        return ("err", type(exc).__name__)

cases = [
    ("split Python list equal sections", "split", lambda: (([1, 2, 3, 4], 2), {})),
    (
        "split tuple indices with empty partitions",
        "split",
        lambda: ((np.arange(6, dtype=np.int16), (0, 2, 6)), {}),
    ),
    (
        "split negative axis list indices",
        "split",
        lambda: ((np.arange(12).reshape(2, 3, 2), [1, 2]), {"axis": -2}),
    ),
    (
        "array_split more sections than elements",
        "array_split",
        lambda: (([10, 20, 30], 5), {}),
    ),
    (
        "array_split tuple indices",
        "array_split",
        lambda: ((np.arange(5, dtype=np.uint16), (0, 2, 5)), {}),
    ),
    (
        "hsplit Python list matrix",
        "hsplit",
        lambda: (([[1, 2, 3, 4], [5, 6, 7, 8]], 2), {}),
    ),
    (
        "vsplit Python list matrix",
        "vsplit",
        lambda: (([[1, 2], [3, 4], [5, 6], [7, 8]], 2), {}),
    ),
    (
        "dsplit tuple indices",
        "dsplit",
        lambda: ((np.arange(24, dtype=np.int32).reshape(2, 3, 4), (0, 2, 4)), {}),
    ),
    ("split uneven sections error", "split", lambda: ((np.arange(5), 2), {})),
    (
        "split invalid axis error",
        "split",
        lambda: ((np.arange(4), 2), {"axis": 2}),
    ),
    ("hsplit scalar error", "hsplit", lambda: ((np.array(1), 2), {})),
    ("vsplit one dimensional error", "vsplit", lambda: ((np.arange(4), 2), {})),
    ("dsplit two dimensional error", "dsplit", lambda: ((np.ones((2, 2)), 2), {})),
]

ok = True
for label, name, factory in cases:
    args, kwargs = factory()
    actual = outcome(getattr(fnp, name), *args, **kwargs)
    args, kwargs = factory()
    expected = outcome(getattr(np, name), *args, **kwargs)
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
        "split helper Python-container and index surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn split_equal_parts() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6])
result = fnp.split(a, 3)
expected = np.split(a, 3)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "split equal parts should match numpy"
    );
    Ok(())
}

#[test]
fn split_with_indices() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6])
result = fnp.split(a, [2, 4])
expected = np.split(a, [2, 4])
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "split with indices should match numpy"
    );
    Ok(())
}

#[test]
fn split_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
result = fnp.split(a, 2, axis=0)
expected = np.split(a, 2, axis=0)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "split 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn split_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
result = fnp.split(a, 2, axis=1)
expected = np.split(a, 2, axis=1)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "split 2d axis=1 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// array_split
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_split_unequal_parts() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.array_split(a, 3)
expected = np.array_split(a, 3)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "array_split unequal parts should match numpy"
    );
    Ok(())
}

#[test]
fn array_split_with_indices() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7])
result = fnp.array_split(a, [2, 5])
expected = np.array_split(a, [2, 5])
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "array_split with indices should match numpy"
    );
    Ok(())
}

#[test]
fn array_split_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.array_split(a, 2, axis=0)
expected = np.array_split(a, 2, axis=0)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "array_split 2d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// hsplit
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hsplit_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6])
result = fnp.hsplit(a, 3)
expected = np.hsplit(a, 3)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hsplit 1d should match numpy");
    Ok(())
}

#[test]
fn hsplit_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
result = fnp.hsplit(a, 2)
expected = np.hsplit(a, 2)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hsplit 2d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// vsplit
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn vsplit_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
result = fnp.vsplit(a, 2)
expected = np.vsplit(a, 2)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vsplit basic should match numpy");
    Ok(())
}

#[test]
fn vsplit_unequal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.vsplit(a, [1, 2])
expected = np.vsplit(a, [1, 2])
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vsplit unequal should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// dsplit
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dsplit_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(16).reshape(2, 2, 4)
result = fnp.dsplit(a, 2)
expected = np.dsplit(a, 2)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dsplit basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn split_concatenate_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6])
splits = fnp.split(a, 3)
result = fnp.concatenate(splits)
print(np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "split then concatenate should be identity"
    );
    Ok(())
}

#[test]
fn hsplit_hstack_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
splits = fnp.hsplit(a, 2)
result = fnp.hstack(splits)
print(np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hsplit then hstack should be identity"
    );
    Ok(())
}

#[test]
fn vsplit_vstack_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
splits = fnp.vsplit(a, 2)
result = fnp.vstack(splits)
print(np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "vsplit then vstack should be identity"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// SHOULD-level edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_split_more_sections_than_elements() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([10, 20, 30], dtype=np.int16)
result = fnp.array_split(a, 5)
expected = np.array_split(a, 5)
match = (
    len(result) == len(expected)
    and all(r.dtype == e.dtype for r, e in zip(result, expected))
    and all(r.shape == e.shape for r, e in zip(result, expected))
    and all(np.array_equal(r, e) for r, e in zip(result, expected))
)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "array_split should preserve empty partitions and dtype"
    );
    Ok(())
}

#[test]
fn split_uneven_integer_sections_error_surface() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(5)
def capture(fn):
    try:
        fn(a, 2)
    except Exception as exc:
        return type(exc).__name__, str(exc)
    return "OK", ""

ours = capture(fnp.split)
expected = capture(np.split)
print(ours[0] == expected[0] and ours[1] == expected[1])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "split uneven sections error should match numpy"
    );
    Ok(())
}

#[test]
fn dsplit_with_explicit_indices_and_empty_tail() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
result = fnp.dsplit(a, [0, 2, 4])
expected = np.dsplit(a, [0, 2, 4])
match = (
    len(result) == len(expected)
    and all(r.dtype == e.dtype for r, e in zip(result, expected))
    and all(r.shape == e.shape for r, e in zip(result, expected))
    and all(np.array_equal(r, e) for r, e in zip(result, expected))
)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "dsplit explicit indices should preserve empty slices"
    );
    Ok(())
}

#[test]
fn split_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.split(a, 2)
np_result = np.split(a, 2)
match = (
    len(fnp_result) == len(np_result)
    and all(np.array_equal(f, n) for f, n in zip(fnp_result, np_result))
)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "split complex should match numpy");
    Ok(())
}
