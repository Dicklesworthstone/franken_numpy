//! Conformance tests for fnp_python.strings — the numpy.strings submodule.
//!
//! fnp_python.strings is a re-export of numpy.strings (45+ string element-wise
//! ops). These tests verify the submodule is reachable, that every documented
//! function exists, and that representative call paths produce numpy-equal
//! output across the standard function families.

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

#[test]
fn strings_namespace_attached_and_is_numpy_strings() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.strings is np.strings)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.strings must be the numpy.strings submodule"
    );
    Ok(())
}

#[test]
fn strings_full_function_set_reachable() -> Result<(), String> {
    let script = fnp_script(
        r#"
funcs = [
    'add', 'multiply', 'equal', 'not_equal', 'greater', 'less',
    'greater_equal', 'less_equal',
    'isupper', 'islower', 'isdigit', 'isalpha', 'isalnum',
    'isnumeric', 'isdecimal', 'istitle', 'isspace',
    'lower', 'upper', 'capitalize', 'title', 'swapcase',
    'replace', 'strip', 'lstrip', 'rstrip',
    'find', 'rfind', 'index', 'rindex', 'count',
    'startswith', 'endswith',
    'partition', 'rpartition',
    'center', 'ljust', 'rjust', 'zfill',
    'translate', 'encode', 'decode',
    'expandtabs', 'mod', 'str_len',
]
missing = [f for f in funcs if not hasattr(fnp.strings, f)]
print(missing == [])
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "all expected numpy.strings functions must be reachable through fnp.strings"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Case-conversion family
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn strings_case_conversion_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array(['Hello', 'World', 'fNp Python'])
ok = (np.array_equal(fnp.strings.lower(arr), np.strings.lower(arr)) and
      np.array_equal(fnp.strings.upper(arr), np.strings.upper(arr)) and
      np.array_equal(fnp.strings.capitalize(arr), np.strings.capitalize(arr)) and
      np.array_equal(fnp.strings.title(arr), np.strings.title(arr)) and
      np.array_equal(fnp.strings.swapcase(arr), np.strings.swapcase(arr)))
print(ok)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "case-conversion family must match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Predicate family
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn strings_predicate_family_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array(['Hello', 'WORLD', 'mixed1', '12345', '   ', 'Title Case'])
ok = (np.array_equal(fnp.strings.isupper(arr), np.strings.isupper(arr)) and
      np.array_equal(fnp.strings.islower(arr), np.strings.islower(arr)) and
      np.array_equal(fnp.strings.isalpha(arr), np.strings.isalpha(arr)) and
      np.array_equal(fnp.strings.isdigit(arr), np.strings.isdigit(arr)) and
      np.array_equal(fnp.strings.isalnum(arr), np.strings.isalnum(arr)) and
      np.array_equal(fnp.strings.isspace(arr), np.strings.isspace(arr)) and
      np.array_equal(fnp.strings.istitle(arr), np.strings.istitle(arr)))
print(ok)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "predicate family must match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Search / count family
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn strings_search_count_family_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array(['abcabc', 'def', 'gabchg'])
needle = 'abc'
ok = (np.array_equal(fnp.strings.find(arr, needle), np.strings.find(arr, needle)) and
      np.array_equal(fnp.strings.rfind(arr, needle), np.strings.rfind(arr, needle)) and
      np.array_equal(fnp.strings.count(arr, needle), np.strings.count(arr, needle)) and
      np.array_equal(fnp.strings.startswith(arr, 'a'), np.strings.startswith(arr, 'a')) and
      np.array_equal(fnp.strings.endswith(arr, 'c'), np.strings.endswith(arr, 'c')))
print(ok)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "search/count family must match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Strip / pad family
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn strings_strip_and_pad_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array(['  hi  ', 'xxhix', '..yy'])
ok = (np.array_equal(fnp.strings.strip(arr), np.strings.strip(arr)) and
      np.array_equal(fnp.strings.lstrip(arr, 'x'), np.strings.lstrip(arr, 'x')) and
      np.array_equal(fnp.strings.rstrip(arr, 'x'), np.strings.rstrip(arr, 'x')) and
      np.array_equal(fnp.strings.center(arr, 10, '-'), np.strings.center(arr, 10, '-')) and
      np.array_equal(fnp.strings.ljust(arr, 8, '.'), np.strings.ljust(arr, 8, '.')) and
      np.array_equal(fnp.strings.rjust(arr, 8, '.'), np.strings.rjust(arr, 8, '.')) and
      np.array_equal(fnp.strings.zfill(arr, 8), np.strings.zfill(arr, 8)))
print(ok)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "strip/pad family must match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Comparison / equality family
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn strings_comparison_family_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array(['apple', 'banana', 'cherry'])
b = np.array(['apple', 'banner', 'cherry'])
ok = (np.array_equal(fnp.strings.equal(a, b), np.strings.equal(a, b)) and
      np.array_equal(fnp.strings.not_equal(a, b), np.strings.not_equal(a, b)) and
      np.array_equal(fnp.strings.greater(a, b), np.strings.greater(a, b)) and
      np.array_equal(fnp.strings.less(a, b), np.strings.less(a, b)) and
      np.array_equal(fnp.strings.greater_equal(a, b), np.strings.greater_equal(a, b)) and
      np.array_equal(fnp.strings.less_equal(a, b), np.strings.less_equal(a, b)))
print(ok)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "comparison family must match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic / replace family
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn strings_add_multiply_replace_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array(['foo', 'bar', 'baz'])
b = np.array(['1', '2', '3'])
ok = (np.array_equal(fnp.strings.add(a, b), np.strings.add(a, b)) and
      np.array_equal(fnp.strings.multiply(a, 3), np.strings.multiply(a, 3)) and
      np.array_equal(fnp.strings.replace(a, 'a', 'X'),
                     np.strings.replace(a, 'a', 'X')))
print(ok)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "add/multiply/replace must match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Length family
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn strings_str_len_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array(['', 'a', 'bb', 'cccc'])
print(np.array_equal(fnp.strings.str_len(arr), np.strings.str_len(arr)))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "str_len must match numpy"
    );
    Ok(())
}
