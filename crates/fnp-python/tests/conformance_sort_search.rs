//! Conformance tests for numpy sorting and searching functions against NumPy oracle.
//!
//! Tests sort, argsort, unique, searchsorted, nonzero, count_nonzero.

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
// sort
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sort_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
result = fnp.sort(a)
expected = np.sort(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sort 1d should match numpy");
    Ok(())
}

#[test]
fn sort_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 1, 4], [1, 5, 9]])
result = fnp.sort(a, axis=0)
expected = np.sort(a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sort 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn sort_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 1, 4], [1, 5, 9]])
result = fnp.sort(a, axis=1)
expected = np.sort(a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sort 2d axis=1 should match numpy");
    Ok(())
}

#[test]
fn sort_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3.14, 1.41, 2.71, 1.61])
result = fnp.sort(a)
expected = np.sort(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sort float should match numpy");
    Ok(())
}

#[test]
fn sort_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([-3, 1, -4, 1, -5, 9, -2, 6])
result = fnp.sort(a)
expected = np.sort(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sort with negatives should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// argsort
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn argsort_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
result = fnp.argsort(a)
expected = np.argsort(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "argsort 1d should match numpy");
    Ok(())
}

#[test]
fn argsort_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 1, 4], [1, 5, 9]])
result = fnp.argsort(a, axis=1)
expected = np.argsort(a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "argsort 2d should match numpy");
    Ok(())
}

#[test]
fn argsort_reconstructs_sort() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9])
indices = fnp.argsort(a)
sorted_via_indices = a[indices]
sorted_direct = fnp.sort(a)
print(np.array_equal(sorted_via_indices, sorted_direct))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "a[argsort(a)] should equal sort(a)");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// unique
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unique_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4])
result = fnp.unique(a)
expected = np.unique(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique basic should match numpy");
    Ok(())
}

#[test]
fn unique_unsorted() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5, 2, 3, 2, 5, 1, 3, 1])
result = fnp.unique(a)
expected = np.unique(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique unsorted should match numpy");
    Ok(())
}

#[test]
fn unique_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 1.5, 3.5, 2.5])
result = fnp.unique(a)
expected = np.unique(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// searchsorted
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn searchsorted_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
v = np.array([2, 3, 4])
result = fnp.searchsorted(a, v)
expected = np.searchsorted(a, v)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "searchsorted basic should match numpy"
    );
    Ok(())
}

#[test]
fn searchsorted_left() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4])
v = np.array([2, 3])
result = fnp.searchsorted(a, v, side='left')
expected = np.searchsorted(a, v, side='left')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "searchsorted left should match numpy"
    );
    Ok(())
}

#[test]
fn searchsorted_right() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4])
v = np.array([2, 3])
result = fnp.searchsorted(a, v, side='right')
expected = np.searchsorted(a, v, side='right')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "searchsorted right should match numpy"
    );
    Ok(())
}

#[test]
fn searchsorted_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.searchsorted(a, 3.5)
expected = np.searchsorted(a, 3.5)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "searchsorted scalar should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nonzero
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nonzero_python_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def nonzero_outcome(fn, value):
    try:
        result = fn(value)
        arrays = []
        for item in result:
            arr = np.asarray(item)
            arrays.append((str(arr.dtype), tuple(arr.shape), arr.tolist()))
        return ("ok", type(result).__name__, len(result), arrays)
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))

cases = [
    ("list fallback", lambda: [0, 2, 0, 3]),
    ("tuple fallback", lambda: (0, 0, 5, 0)),
    ("bool list", lambda: [False, True, False, True]),
    ("nested list", lambda: [[0, 1], [2, 0]]),
    ("scalar nonzero", lambda: 7),
    ("scalar zero", lambda: 0),
    ("object truthiness", lambda: np.array(["", "x", "0"], dtype=object)),
    ("ragged list error", lambda: [[1], [0, 2]]),
]

ok = True
for label, factory in cases:
    actual = nonzero_outcome(fnp.nonzero, factory())
    expected = nonzero_outcome(np.nonzero, factory())
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
        "nonzero Python-container surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn nonzero_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 0, 2, 0, 3])
result = fnp.nonzero(a)
expected = np.nonzero(a)
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nonzero 1d should match numpy");
    Ok(())
}

#[test]
fn nonzero_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[0, 1, 0], [2, 0, 3]])
result = fnp.nonzero(a)
expected = np.nonzero(a)
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nonzero 2d should match numpy");
    Ok(())
}

#[test]
fn nonzero_all_zeros() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 0, 0])
result = fnp.nonzero(a)
expected = np.nonzero(a)
print(len(result) == len(expected) and result[0].size == 0 and expected[0].size == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nonzero all zeros should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// count_nonzero
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn count_nonzero_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 0, 2, 0, 3])
result = fnp.count_nonzero(a)
expected = np.count_nonzero(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "count_nonzero 1d should match numpy");
    Ok(())
}

#[test]
fn count_nonzero_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[0, 1, 0], [2, 0, 3]])
result = fnp.count_nonzero(a)
expected = np.count_nonzero(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "count_nonzero 2d should match numpy");
    Ok(())
}

#[test]
fn count_nonzero_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[0, 1, 0], [2, 0, 3]])
result = fnp.count_nonzero(a, axis=0)
expected = np.count_nonzero(a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "count_nonzero axis should match numpy"
    );
    Ok(())
}

#[test]
fn count_nonzero_all_zeros() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 0, 0, 0])
result = fnp.count_nonzero(a)
expected = np.count_nonzero(a)
print(result == expected == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "count_nonzero all zeros should be 0");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sort_unique_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5])
# unique returns sorted unique values
unique_vals = fnp.unique(a)
# all unique values should be in sorted order
is_sorted = np.array_equal(unique_vals, fnp.sort(unique_vals))
print(is_sorted)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique should return sorted values");
    Ok(())
}

#[test]
fn nonzero_count_nonzero_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 0, 2, 0, 3, 0])
nz_indices = fnp.nonzero(a)[0]
count = fnp.count_nonzero(a)
print(len(nz_indices) == count)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "len(nonzero) should equal count_nonzero"
    );
    Ok(())
}

#[test]
fn searchsorted_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
v = np.float64(2.5)
fnp_result = fnp.searchsorted(a, v)
np_result = np.searchsorted(a, v)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "searchsorted scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sort_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3+1j, 1-1j, 2+2j], dtype=np.complex128)
fnp_result = fnp.sort(a)
np_result = np.sort(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sort complex should match numpy");
    Ok(())
}

#[test]
fn argsort_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3+1j, 1-1j, 2+2j], dtype=np.complex128)
fnp_result = fnp.argsort(a)
np_result = np.argsort(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "argsort complex should match numpy");
    Ok(())
}
