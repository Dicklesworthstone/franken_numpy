//! Conformance tests for numpy.lexsort against NumPy oracle.

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
// Basic lexsort
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lexsort_two_keys_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
secondary = np.array([1, 2, 1, 2, 1])
primary = np.array([3, 1, 4, 1, 5])
fnp_result = fnp.lexsort((secondary, primary))
np_result = np.lexsort((secondary, primary))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort two keys basic mismatch");
    Ok(())
}

#[test]
fn lexsort_single_key() -> Result<(), String> {
    let script = fnp_script(
        r#"
key = np.array([3, 1, 4, 1, 5, 9, 2, 6])
fnp_result = fnp.lexsort((key,))
np_result = np.lexsort((key,))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort single key mismatch");
    Ok(())
}

#[test]
fn lexsort_three_keys() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 1, 1, 2, 2, 2])
b = np.array([1, 2, 1, 1, 2, 1])
c = np.array([3, 2, 1, 6, 5, 4])
fnp_result = fnp.lexsort((a, b, c))
np_result = np.lexsort((a, b, c))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort three keys mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Ties and ordering
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lexsort_with_ties() -> Result<(), String> {
    let script = fnp_script(
        r#"
primary = np.array([1, 2, 2, 1, 3])
secondary = np.array([5, 4, 3, 6, 0])
fnp_result = fnp.lexsort((secondary, primary))
np_result = np.lexsort((secondary, primary))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort with ties mismatch");
    Ok(())
}

#[test]
fn lexsort_all_equal_primary() -> Result<(), String> {
    let script = fnp_script(
        r#"
primary = np.array([5, 5, 5, 5, 5])
secondary = np.array([3, 1, 4, 1, 5])
fnp_result = fnp.lexsort((secondary, primary))
np_result = np.lexsort((secondary, primary))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort all equal primary mismatch");
    Ok(())
}

#[test]
fn lexsort_already_sorted() -> Result<(), String> {
    let script = fnp_script(
        r#"
primary = np.array([1, 2, 3, 4, 5])
secondary = np.array([1, 2, 3, 4, 5])
fnp_result = fnp.lexsort((secondary, primary))
np_result = np.lexsort((secondary, primary))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort already sorted mismatch");
    Ok(())
}

#[test]
fn lexsort_reverse_sorted() -> Result<(), String> {
    let script = fnp_script(
        r#"
primary = np.array([5, 4, 3, 2, 1])
secondary = np.array([5, 4, 3, 2, 1])
fnp_result = fnp.lexsort((secondary, primary))
np_result = np.lexsort((secondary, primary))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort reverse sorted mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 2D keys array
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lexsort_2d_keys() -> Result<(), String> {
    let script = fnp_script(
        r#"
keys = np.array([[5, 4, 3, 2, 1], [1, 1, 1, 1, 1]])
fnp_result = fnp.lexsort(keys)
np_result = np.lexsort(keys)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort 2d keys mismatch");
    Ok(())
}

#[test]
fn lexsort_2d_keys_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
keys = np.array([[3, 1, 2], [5, 5, 5]])
fnp_result = fnp.lexsort(keys, axis=0)
np_result = np.lexsort(keys, axis=0)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort 2d keys axis=0 mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Float keys
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lexsort_float_keys() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 1.5, 2.5])
b = np.array([3.14, 2.71, 1.41, 0.58])
fnp_result = fnp.lexsort((a, b))
np_result = np.lexsort((a, b))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort float keys mismatch");
    Ok(())
}

#[test]
fn lexsort_negative_floats() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([-1.0, -2.0, 0.0, 1.0, 2.0])
b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
fnp_result = fnp.lexsort((a, b))
np_result = np.lexsort((a, b))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort negative floats mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lexsort_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([42])
b = np.array([7])
fnp_result = fnp.lexsort((a, b))
np_result = np.lexsort((a, b))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort single element mismatch");
    Ok(())
}

#[test]
fn lexsort_empty_keys() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([])
b = np.array([])
fnp_result = fnp.lexsort((a, b))
np_result = np.lexsort((a, b))
print(np.array_equal(fnp_result, np_result) and len(fnp_result) == 0)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort empty keys mismatch");
    Ok(())
}

#[test]
fn lexsort_result_is_permutation() -> Result<(), String> {
    let script = fnp_script(
        r#"
primary = np.array([3, 1, 4, 1, 5, 9, 2, 6])
secondary = np.array([0, 1, 2, 3, 4, 5, 6, 7])
fnp_result = fnp.lexsort((secondary, primary))
# Check that result is a valid permutation
is_permutation = set(fnp_result.tolist()) == set(range(len(primary)))
# Check that applying permutation sorts the primary key
sorted_primary = primary[fnp_result]
is_sorted = all(sorted_primary[i] <= sorted_primary[i+1] for i in range(len(sorted_primary)-1))
print(is_permutation and is_sorted)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(
        output, "True",
        "lexsort result should be valid sorting permutation"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// String keys
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lexsort_string_keys() -> Result<(), String> {
    let script = fnp_script(
        r#"
secondary = np.array(['b', 'a', 'c', 'b', 'a'])
primary = np.array(['one', 'one', 'two', 'two', 'three'])
fnp_result = fnp.lexsort((secondary, primary))
np_result = np.lexsort((secondary, primary))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "lexsort string keys mismatch");
    Ok(())
}
