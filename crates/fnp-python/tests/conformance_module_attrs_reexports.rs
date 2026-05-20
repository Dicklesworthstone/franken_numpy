//! Conformance tests for fnp_python module attribute re-exports:
//! index-trick helpers (s_, index_exp, newaxis),
//! bool singletons (False_, True_),
//! error-handling functions (errstate, seterr, geterr, seterrcall, geterrcall),
//! diagnostic functions (show_config, show_runtime, info).

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
fn index_trick_helpers_identity_equal_to_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
checks = {
    's_':         fnp.s_         is np.s_,
    'index_exp':  fnp.index_exp  is np.index_exp,
    'newaxis':    fnp.newaxis    is np.newaxis,
}
print(checks); print(all(checks.values()))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    expect_equal(
        last,
        "True",
        &format!("index-trick helpers must be identity-equal to numpy; output: {result}"),
    )
}

#[test]
fn bool_singletons_identity_equal_to_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.False_ is np.False_ and fnp.True_ is np.True_)
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.False_ / fnp.True_ must be the numpy singletons",
    )
}

#[test]
fn errstate_round_trip_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
# errstate is a context manager; using it under fnp must affect numpy's
# global err state identically.
import warnings
with fnp.errstate(divide='raise'):
    threw = False
    try:
        np.array([1.0]) / np.array([0.0])
    except FloatingPointError:
        threw = True
print(threw)
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.errstate must propagate to numpy's global state",
    )
}

#[test]
fn seterr_geterr_round_trip_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Snapshot state, mutate via fnp.seterr, observe via np.geterr (and vice
# versa), then restore. The two paths must touch the same dict.
saved = np.geterr()
try:
    fnp.seterr(over='warn', under='ignore')
    ours = fnp.geterr()
    theirs = np.geterr()
    same = ours == theirs and ours['over'] == 'warn' and ours['under'] == 'ignore'
finally:
    np.seterr(**saved)
print(same)
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.seterr / fnp.geterr must share state with numpy",
    )
}

#[test]
fn seterrcall_geterrcall_round_trip_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
saved = np.geterrcall()
try:
    def cb(error_type, flag):
        return None
    fnp.seterrcall(cb)
    same = fnp.geterrcall() is cb and np.geterrcall() is cb
finally:
    np.seterrcall(saved)
print(same)
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.seterrcall / fnp.geterrcall must share state with numpy",
    )
}

#[test]
fn show_config_and_show_runtime_callable() -> Result<(), String> {
    // These are diagnostic functions; we don't pin the output, just that
    // fnp's reference is the same object as numpy's.
    let script = fnp_script(
        r#"
print(fnp.show_config is np.show_config and fnp.show_runtime is np.show_runtime)
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.show_config / fnp.show_runtime must be the numpy functions",
    )
}

#[test]
fn info_is_callable_via_fnp_python() -> Result<(), String> {
    let script = fnp_script(
        r#"
# np.info prints help text — we don't pin its content, just that it runs
# without raising via fnp.
import io, contextlib
out = io.StringIO()
with contextlib.redirect_stdout(out):
    fnp.info(np.sum)
print(len(out.getvalue()) > 0)
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.info must run and produce output via numpy's implementation",
    )
}
