//! Conformance tests for re-exported numpy submodules:
//! np.char, np.rec, np.emath, np.matrixlib (plus np.strings sanity).
//!
//! fnp_python re-exports most of these submodules verbatim from numpy. `char`
//! and `strings` are shallow native overlays for ASCII upper/lower, so the tests
//! verify reachability, copied non-overridden attributes, and numpy-equal output.

use std::{
    io::Write,
    process::{Command, Stdio},
};

fn numpy_oracle(script: &str) -> Result<String, String> {
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| format!("python3 stdin pipe was unavailable\nScript: {script}"))?;
        stdin
            .write_all(script.as_bytes())
            .map_err(|error| format!("failed to write python script: {error}\nScript: {script}"))?;
    }
    let output = child
        .wait_with_output()
        .map_err(|error| format!("failed to wait for python3: {error}\nScript: {script}"))?;
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
         import sys\n\
         import numpy as np\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         sys.modules[spec.name] = fnp\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}

fn expect_equal(actual: &str, expected: &str, context: &str) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{context}; expected {expected:?}, got {actual:?}"))
    }
}

#[test]
fn submodules_identity_or_overlay_match_numpy() -> Result<(), String> {
    // `add` is NOT a valid "non-overridden" probe: char_add_native and
    // strings_add_native are both registered as `add`, so `fnp.char.add is
    // np.char.add` is False on a perfectly healthy build. The old check asserted
    // it anyway and was red wherever the submodule resolved at all.
    //
    // `startswith` and `encode` are the real probes for "copied verbatim": both
    // exist on numpy.strings AND numpy.char and neither is in the native overlay
    // set. (`split` is not usable — numpy.char has it but numpy.strings does not.)
    // The overridden names get the opposite assertion, which is what actually
    // protects the native fast paths from a silent verbatim re-export.
    let script = fnp_script(
        r#"
checks = {
    # The submodules resolved at all. Empty dict is the healthy state; a
    # non-empty one names the submodule and the reason it could not be imported.
    'no_submodule_import_errors': fnp.__submodule_import_errors__ == {},
    'strings_reachable': hasattr(fnp.strings, 'upper'),
    'char_reachable': hasattr(fnp.char, 'upper'),
    # Attributes the overlay does NOT touch must be numpy's own objects.
    'strings_untouched_identity': fnp.strings.startswith is np.strings.startswith,
    'strings_untouched_identity_2': fnp.strings.encode is np.strings.encode,
    'char_untouched_identity': fnp.char.startswith is np.char.startswith,
    'char_untouched_identity_2': fnp.char.encode is np.char.encode,
    # Attributes the overlay DOES override must NOT be numpy's — otherwise the
    # native ASCII fast path has been silently replaced by a verbatim re-export.
    'strings_upper_is_native': fnp.strings.upper is not np.strings.upper,
    'char_upper_is_native': fnp.char.upper is not np.char.upper,
    'char_add_is_native': fnp.char.add is not np.char.add,
    # ...and overriding must not have changed the answer.
    'char_upper_matches_numpy': np.array_equal(
        fnp.char.upper(np.array(['ab', 'Cd'])), np.char.upper(np.array(['ab', 'Cd']))),
    'char_add_matches_numpy': np.array_equal(
        fnp.char.add(np.array(['a']), np.array(['b'])), np.char.add(np.array(['a']), np.array(['b']))),
    'rec':       fnp.rec       is np.rec,
    'emath':     fnp.emath     is np.emath,
    'matrixlib': fnp.matrixlib is np.matrixlib,
}
print({k: v for k, v in checks.items() if not v} or 'ALL_OK')
print(fnp.__submodule_import_errors__)
print(all(checks.values()))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    expect_equal(
        last,
        "True",
        &format!(
            "re-exported submodules must preserve numpy identity or overlay contract; output: {result}"
        ),
    )
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
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.char.lower/upper must match numpy",
    )
}

#[test]
fn char_strings_ascii_case_golden_sha256() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib
cases = [
    np.array(['azByCxD0123_', 'HELLO_world', '', 'MiXeD'], dtype='<U20').reshape(2, 2),
    np.array(['éclair', 'MÜNCHEN', 'ASCII'], dtype='<U20'),
]
chunks = []
for namespace in ['char', 'strings']:
    for method in ['upper', 'lower']:
        fnp_func = getattr(getattr(fnp, namespace), method)
        np_func = getattr(getattr(np, namespace), method)
        for arr in cases:
            got = fnp_func(arr)
            want = np_func(arr)
            if not np.array_equal(got, want):
                raise AssertionError((namespace, method, got, want))
            chunks.extend([
                namespace.encode(),
                method.encode(),
                str(got.shape).encode(),
                str(got.dtype).encode(),
                got.tobytes(),
            ])
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "46c7c95b14749dc8f5ea6951e9134cec0f1a461c530984898b5df4860678f1b7",
        "char/strings ASCII case golden SHA-256 must match numpy",
    )
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
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.char.add/multiply must match numpy",
    )
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
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.rec.fromarrays must match numpy",
    )
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
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.rec.fromrecords must match numpy",
    )
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
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.emath.sqrt(-x) must return complex like numpy",
    )
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
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.emath.log(neg) / arccos(>1) must match numpy",
    )
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
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.matrixlib.asmatrix must match numpy",
    )
}
