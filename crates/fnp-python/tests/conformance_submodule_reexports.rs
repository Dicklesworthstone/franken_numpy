//! Conformance tests for re-exported numpy submodules:
//! np.char, np.rec, np.emath, np.matrixlib (plus np.strings sanity).
//!
//! fnp_python re-exports these submodules verbatim from numpy. The tests
//! verify each submodule is reachable, is identity-equal to its numpy peer,
//! and that a representative call from each submodule produces numpy-equal
//! output.

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
fn submodules_identity_equal_to_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
checks = {
    'strings':   fnp.strings   is np.strings,
    'char':      fnp.char      is np.char,
    'rec':       fnp.rec       is np.rec,
    'emath':     fnp.emath     is np.emath,
    'matrixlib': fnp.matrixlib is np.matrixlib,
}
print(checks)
print(all(checks.values()))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "all re-exported submodules must be identity-equal to their numpy peers; got: {result}"
    );
    Ok(())
}

#[test]
fn char_lower_upper_round_trip_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array(['Hello', 'WORLD'])
ok = (np.array_equal(fnp.char.lower(arr), np.char.lower(arr)) and
      np.array_equal(fnp.char.upper(arr), np.char.upper(arr)))
print(ok)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.char.lower/upper must match numpy"
    );
    Ok(())
}

#[test]
fn char_add_multiply_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array(['foo', 'bar'])
b = np.array(['1', '2'])
ok = (np.array_equal(fnp.char.add(a, b), np.char.add(a, b)) and
      np.array_equal(fnp.char.multiply(a, 3), np.char.multiply(a, 3)))
print(ok)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.char.add/multiply must match numpy"
    );
    Ok(())
}

#[test]
fn rec_fromarrays_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array(['x', 'y', 'z'])
ours = fnp.rec.fromarrays([a, b], names='ids,labels')
theirs = np.rec.fromarrays([a, b], names='ids,labels')
print(ours.dtype == theirs.dtype and np.array_equal(ours, theirs))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.rec.fromarrays must match numpy"
    );
    Ok(())
}

#[test]
fn rec_fromrecords_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
records = [(1, 'a'), (2, 'b')]
dtype = [('n', np.int32), ('s', 'U1')]
ours = fnp.rec.fromrecords(records, dtype=dtype)
theirs = np.rec.fromrecords(records, dtype=dtype)
print(ours.dtype == theirs.dtype and np.array_equal(ours, theirs))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.rec.fromrecords must match numpy"
    );
    Ok(())
}

#[test]
fn emath_sqrt_negative_returns_complex_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
# emath.sqrt of a negative real returns a complex number rather than NaN
ours = fnp.emath.sqrt(-4.0)
theirs = np.emath.sqrt(-4.0)
print(ours == theirs and type(ours).__name__ == type(theirs).__name__)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.emath.sqrt(-x) must return complex like numpy"
    );
    Ok(())
}

#[test]
fn emath_log_zero_returns_neginf_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
# emath.log family also dispatches into complex for negative inputs
ours_log_neg = fnp.emath.log(-1.0)
theirs_log_neg = np.emath.log(-1.0)
ours_arccos = fnp.emath.arccos(2.0)
theirs_arccos = np.emath.arccos(2.0)
print(ours_log_neg == theirs_log_neg and ours_arccos == theirs_arccos)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.emath.log(neg) / arccos(>1) must match numpy"
    );
    Ok(())
}

#[test]
fn matrixlib_asmatrix_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [[1, 2], [3, 4]]
ours = fnp.matrixlib.asmatrix(a)
theirs = np.matrixlib.asmatrix(a)
print(type(ours).__name__ == type(theirs).__name__ and np.array_equal(ours, theirs))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.matrixlib.asmatrix must match numpy"
    );
    Ok(())
}
