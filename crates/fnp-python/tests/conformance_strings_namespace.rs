//! Conformance tests for fnp_python.strings — the numpy.strings submodule.
//!
//! fnp_python.strings is a shallow native overlay over numpy.strings. These
//! tests verify the submodule is reachable, that every documented function
//! exists, and that representative call paths produce numpy-equal output across
//! the standard function families.

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

mod support;
use support::fnp_script;

#[test]
fn strings_namespace_attached_and_preserves_numpy_surface() -> Result<(), String> {
    // The old form asserted `fnp.strings.add is np.strings.add`, which is False
    // on a healthy build: strings_add_native is registered as `add`, so `add` is
    // one of the OVERRIDDEN names, not a copied one. `startswith` is genuinely
    // untouched by the overlay and is the correct probe for "copied verbatim".
    // The overridden side is asserted in the opposite direction so a silent
    // replacement of the native path by numpy's own function fails here.
    let script = fnp_script(
        r#"
checks = {
    'overlay_present': hasattr(fnp.strings, 'upper'),
    'copied_surface_is_numpys': fnp.strings.startswith is np.strings.startswith,
    'overridden_add_is_native': fnp.strings.add is not np.strings.add,
    'overridden_add_still_matches_numpy': np.array_equal(
        fnp.strings.add(np.array(['a']), np.array(['b'])),
        np.strings.add(np.array(['a']), np.array(['b']))),
}
print({k: v for k, v in checks.items() if not v} or 'ALL_OK')
print(all(checks.values()))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "fnp.strings must expose the native overlay plus copied numpy.strings surface; output: {result}"
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

/// Every numpy.strings / numpy.char function on a 0-d unicode array, a 0-d bytes array, a numpy
/// str_ scalar and a 1-D control, with the first extra-argument tuple numpy accepts: fnp must end
/// the same way (ok or the same exception type) with the same result type, dtype and repr. Found
/// by the panic audit (bead rc0923 .20): the native string routes `.view()` their operand as
/// uint32/uint8, which numpy refuses for a 0-d array ("Changing the dtype of a 0d array ..."), so
/// 79 of 369 cells raised ValueError where numpy answers (`np.strings.strip(np.array(' ab '))` is
/// `np.str_('ab')`). The 1-D operand is the control: it passed before the fix too.
#[test]
fn strings_and_char_functions_match_numpy_on_zero_dim_and_scalar_operands() -> Result<(), String> {
    let script = fnp_script(
        r#"
import inspect, warnings
warnings.simplefilter("ignore")
def o(f):
    try:
        r = f()
        return ("ok", type(r).__name__, str(getattr(r, "dtype", "")), repr(r))
    except Exception as e:
        return (type(e).__name__,)
OPS = {"U0d": np.array("  aB c "), "S0d": np.array(b"  aB c "), "U1d": np.array(["  aB c ", "x"]),
       "Uscalar": np.str_("  aB c ")}
EXTRA = [(), (6,), ("a",), (b"a",), ("aB", "Z"), (b"aB", b"Z"), (2,)]
bad, cells, zero_d = [], 0, 0
for modname in ("strings", "char"):
    nm, fm = getattr(np, modname), getattr(fnp, modname)
    for name in sorted(getattr(nm, "__all__", dir(nm))):
        f1, f2 = getattr(nm, name, None), getattr(fm, name, None)
        if not callable(f1) or f2 is None or inspect.isclass(f1):
            continue
        for key, a in OPS.items():
            for extra in EXTRA:
                s = o(lambda: f1(a, *extra))
                if s[0] != "ok":
                    continue
                cells += 1
                zero_d += key.endswith("0d")
                r = o(lambda: f2(a, *extra))
                if r != s:
                    bad.append(f"{modname}.{name}{extra} {key}: fnp={r[:3]} numpy={s[:3]}")
                break
print(cells, zero_d, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(3, ' ');
    let cells: usize = fields.next().unwrap_or("").parse().unwrap_or(0);
    let zero_d: usize = fields.next().unwrap_or("").parse().unwrap_or(0);
    assert!(cells >= 300, "cell table drifted: {result}");
    assert!(zero_d >= 150, "too few 0-d cells to test anything: {result}");
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "strings/char must match numpy on 0-d operands: {result}"
    );
    Ok(())
}
