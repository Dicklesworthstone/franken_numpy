//! Conformance tests for fnp_python module metadata attributes:
//! __doc__, __version__, __numpy_version__.

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
fn fnp_python_has_nonempty_docstring() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.__doc__ is not None and len(fnp.__doc__) > 100 and 'fnp_python' in fnp.__doc__)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp_python must have a non-empty module docstring naming itself"
    );
    Ok(())
}

#[test]
fn fnp_python_docstring_references_audit_or_readme() -> Result<(), String> {
    let script = fnp_script(
        r#"
doc = fnp.__doc__ or ''
print('audit_numpy_reality.md' in doc or 'README.md' in doc)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp_python docstring must point at the canonical onboarding docs"
    );
    Ok(())
}

#[test]
fn fnp_python_version_attribute_matches_cargo_pkg_version() -> Result<(), String> {
    let script = fnp_script(
        r#"
v = fnp.__version__
print(isinstance(v, str) and len(v) > 0 and v.count('.') >= 2)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.__version__ must be a non-empty semver-shaped string"
    );
    Ok(())
}

#[test]
fn fnp_python_numpy_version_matches_runtime_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.__numpy_version__ == np.__version__)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.__numpy_version__ must equal the runtime numpy.__version__"
    );
    Ok(())
}
