//! Conformance tests for fnp_python's remaining numpy top-level re-exports:
//! iterable, ndim, size, packbits, unpackbits, fromfunction, pow (ufunc),
//! typecodes, typename, sctypeDict, ScalarType, __array_namespace_info__,
//! typing (submodule), ctypeslib (submodule), test (PytestTester),
//! getbufsize, nested_iters, from_dlpack.

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
fn remaining_top_level_attrs_identity_equal_to_numpy() -> Result<(), String> {
    // The bulk safety check: every attribute we set is `is`-equal to its
    // numpy counterpart. Identity equality is enough to prove the re-export
    // is genuine; semantic equivalence is then numpy's own contract.
    let script = fnp_script(
        r#"
names = [
    'iterable', 'ndim', 'size', 'packbits', 'unpackbits', 'fromfunction',
    'pow', 'typecodes', 'typename', 'sctypeDict', 'ScalarType',
    '__array_namespace_info__', 'typing', 'ctypeslib', 'test',
    'getbufsize', 'nested_iters', 'from_dlpack',
]
mismatches = []
for n in names:
    if not hasattr(fnp, n):
        mismatches.append((n, 'missing'))
        continue
    if getattr(fnp, n) is not getattr(np, n):
        mismatches.append((n, 'not-identity-equal'))
print(mismatches)
print(mismatches == [])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "all remaining top-level attributes must be identity-equal to numpy; got: {result}"
    );
    Ok(())
}

#[test]
fn pow_ufunc_acts_like_numpy_power_alias() -> Result<(), String> {
    // np.pow is a numpy 2.x alias for np.power. Both must produce identical
    // output when called as a ufunc.
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4], dtype=np.float64)
b = np.array([2, 2, 2, 2], dtype=np.float64)
ours = fnp.pow(a, b)
theirs = np.pow(a, b)
print(np.array_equal(ours, theirs) and np.array_equal(ours, np.power(a, b)))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.pow must match np.pow and np.power"
    );
    Ok(())
}

#[test]
fn packbits_unpackbits_round_trip_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=np.uint8)
packed_ours = fnp.packbits(a)
packed_theirs = np.packbits(a)
unpacked_ours = fnp.unpackbits(packed_ours)[:len(a)]
unpacked_theirs = np.unpackbits(packed_theirs)[:len(a)]
print(np.array_equal(packed_ours, packed_theirs) and
      np.array_equal(unpacked_ours, unpacked_theirs) and
      np.array_equal(unpacked_ours, a))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.packbits/unpackbits must match numpy and round-trip"
    );
    Ok(())
}

#[test]
fn fromfunction_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
ours = fnp.fromfunction(lambda i, j: i + j, (3, 4), dtype=int)
theirs = np.fromfunction(lambda i, j: i + j, (3, 4), dtype=int)
print(np.array_equal(ours, theirs))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.fromfunction must match numpy"
    );
    Ok(())
}

#[test]
fn ndim_size_iterable_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.arange(12).reshape(3, 4)
ok = (fnp.ndim(arr) == np.ndim(arr) == 2 and
      fnp.size(arr) == np.size(arr) == 12 and
      fnp.size(arr, axis=0) == np.size(arr, axis=0) == 3 and
      fnp.iterable([1, 2, 3]) == np.iterable([1, 2, 3]) == True and
      fnp.iterable(5) == np.iterable(5) == False)
print(ok)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.ndim/size/iterable must match numpy"
    );
    Ok(())
}

#[test]
#[allow(non_snake_case)]
fn typecodes_and_sctypeDict_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.typecodes == np.typecodes and
      sorted(fnp.sctypeDict.keys()) == sorted(np.sctypeDict.keys()))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.typecodes and fnp.sctypeDict must match numpy"
    );
    Ok(())
}

#[test]
fn getbufsize_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.getbufsize() == np.getbufsize())
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.getbufsize must match numpy"
    );
    Ok(())
}

#[test]
#[allow(non_snake_case)]
fn ScalarType_contains_basic_scalars() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Both fnp.ScalarType and np.ScalarType are tuples of Python scalar types.
ok = (fnp.ScalarType == np.ScalarType and int in fnp.ScalarType and
      float in fnp.ScalarType and complex in fnp.ScalarType)
print(ok)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.ScalarType must equal np.ScalarType"
    );
    Ok(())
}

#[test]
fn array_namespace_info_callable_via_fnp() -> Result<(), String> {
    let script = fnp_script(
        r#"
ours = fnp.__array_namespace_info__()
theirs = np.__array_namespace_info__()
ok = (type(ours).__name__ == type(theirs).__name__ and
      ours.devices() == theirs.devices())
print(ok)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.__array_namespace_info__() must mirror np"
    );
    Ok(())
}
