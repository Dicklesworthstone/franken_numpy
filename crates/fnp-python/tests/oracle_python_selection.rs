//! Regression tests for oracle Python interpreter selection and ABI mismatch diagnostics.

mod support;

#[test]
fn resolve_oracle_python_precedence() {
    assert_eq!(
        support::resolve_oracle_python_from(Some("/opt/oracle/python3.12"), Some("/opt/pyo3/py")),
        "/opt/oracle/python3.12"
    );
    assert_eq!(
        support::resolve_oracle_python_from(Some("   "), Some("/opt/pyo3/python3.12")),
        "/opt/pyo3/python3.12"
    );
    assert_eq!(
        support::resolve_oracle_python_from(None, Some("/opt/pyo3/python3.12")),
        "/opt/pyo3/python3.12"
    );
    assert_eq!(
        support::resolve_oracle_python_from(None, Some("")),
        "python3"
    );
    assert_eq!(support::resolve_oracle_python_from(None, None), "python3");
}

#[test]
fn abi_check_accepts_matching_and_none_built_version() {
    let python = support::resolve_oracle_python();
    let actual_xy = support::python_minor_version(&python)
        .expect("resolved oracle python should report major.minor version");

    support::check_python_abi_match(&python, None)
        .expect("missing build-time Python version must be allowed");
    support::check_python_abi_match(&python, Some(&actual_xy))
        .expect("matching build-time Python minor version must pass");
}

#[test]
fn abi_check_rejects_mismatched_python_minor_with_clear_diagnostic() {
    let python = support::resolve_oracle_python();
    let err = support::check_python_abi_match(&python, Some("99.99"))
        .expect_err("mismatched Python minor version must return an explicit ABI error");
    assert!(
        err.contains("fnp_python ABI mismatch"),
        "expected ABI mismatch headline, got: {err}"
    );
    assert!(
        err.contains("Python 99.99"),
        "expected built version in error, got: {err}"
    );
    assert!(
        err.contains("FNP_ORACLE_PYTHON") && err.contains("PYO3_PYTHON"),
        "expected env-var remediation hint in error, got: {err}"
    );
}

#[test]
fn python_command_executes_resolved_interpreter() {
    let output = support::python_command()
        .args(["-c", "import sys; print(sys.version_info[0])"])
        .output()
        .expect("support::python_command should run resolved oracle interpreter");
    assert!(
        output.status.success(),
        "oracle python exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "3");
}
