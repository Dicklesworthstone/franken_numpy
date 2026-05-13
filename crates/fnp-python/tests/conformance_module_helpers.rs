//! Conformance tests for numpy module-level helpers:
//! fromregex, min_scalar_type, get_printoptions, mintypecode.

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
         import io\n\
         import numpy as np\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// min_scalar_type
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn min_scalar_type_uint8_small_int_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.min_scalar_type(255) == np.min_scalar_type(255))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "min_scalar_type(255) must match numpy"
    );
    Ok(())
}

#[test]
fn min_scalar_type_float_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.min_scalar_type(0.5) == np.min_scalar_type(0.5))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "min_scalar_type(0.5) must match numpy"
    );
    Ok(())
}

#[test]
fn min_scalar_type_negative_int_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.min_scalar_type(-32768) == np.min_scalar_type(-32768))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "min_scalar_type(-32768) must match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// get_printoptions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn get_printoptions_keys_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
ours = fnp.get_printoptions()
theirs = np.get_printoptions()
print(sorted(ours.keys()) == sorted(theirs.keys()))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "get_printoptions keys must match numpy"
    );
    Ok(())
}

#[test]
fn get_printoptions_values_match_numpy_for_unchanged_state() -> Result<(), String> {
    // get_printoptions reads global state, so unless something has perturbed
    // numpy's print state, ours and theirs should be value-equal too.
    let script = fnp_script(
        r#"
ours = fnp.get_printoptions()
theirs = np.get_printoptions()
# Compare key-by-key for shallow equality on the documented fields.
match = all(ours.get(k) == theirs.get(k) for k in ('precision', 'threshold', 'edgeitems', 'linewidth', 'suppress', 'sign', 'floatmode', 'nanstr', 'infstr'))
print(match)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "get_printoptions documented values must match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// mintypecode
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mintypecode_default_single_float_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.mintypecode(['f']) == np.mintypecode(['f']))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "mintypecode(['f']) must match numpy"
    );
    Ok(())
}

#[test]
fn mintypecode_mixed_types_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.mintypecode(['f', 'd', 'F']) == np.mintypecode(['f', 'd', 'F']))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "mintypecode(['f','d','F']) must match numpy"
    );
    Ok(())
}

#[test]
fn mintypecode_custom_typeset_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.mintypecode(['l'], typeset='ld', default='d') ==
      np.mintypecode(['l'], typeset='ld', default='d'))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "mintypecode custom typeset must match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// fromregex
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fromregex_simple_int_pattern_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
buf_a = io.BytesIO(b"1 2\n3 4\n5 6\n")
buf_b = io.BytesIO(b"1 2\n3 4\n5 6\n")
dtype = [('x', np.int32), ('y', np.int32)]
ours = fnp.fromregex(buf_a, rb"(\d+)\s+(\d+)", dtype)
theirs = np.fromregex(buf_b, rb"(\d+)\s+(\d+)", dtype)
print(ours.dtype == theirs.dtype and np.array_equal(ours, theirs))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fromregex with int pattern must match numpy"
    );
    Ok(())
}

#[test]
fn fromregex_single_group_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
text = b"a=10\nb=20\nc=30\n"
buf_a = io.BytesIO(text)
buf_b = io.BytesIO(text)
dtype = [('val', np.int32)]
ours = fnp.fromregex(buf_a, rb"=(\d+)", dtype)
theirs = np.fromregex(buf_b, rb"=(\d+)", dtype)
print(ours.dtype == theirs.dtype and np.array_equal(ours, theirs))
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fromregex single-group must match numpy"
    );
    Ok(())
}
