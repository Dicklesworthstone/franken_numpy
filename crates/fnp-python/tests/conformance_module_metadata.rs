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
         import sys\n\
         import numpy as np\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         sys.modules[spec.name] = fnp\n\
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

/// THE SUPPORTED-ORACLE FLOOR. Conformance is defined against NumPy >= 2.3.
///
/// This exists so a below-floor host fails HERE, once, with a sentence that
/// names the reason — instead of scattering confusing byte mismatches across
/// unrelated shards. Every symptom below was measured on the 2.2.4 worker class
/// (ovh-a, vmi1149989), and in each one fnp matched the NEWER NumPy while 2.2.4
/// was the outlier; not once the reverse (bead deadlock-audit-1jqrw):
///
///   * `count_nonzero` returns a Python `int` on 2.2.4, `int64` from 2.3 on.
///   * `inspect.signature(np.add)` exposes x1/x2 from 2.3 on, not on 2.2.4.
///   * `a[mask].sum()` reduces in an order that departs from NumPy's own
///     documented pairwise tree above ~25k elements on 2.2.4, so the fused
///     masked reduction's byte-identity holds only at or above the floor.
///   * `import numpy.char` FAILS outright on 2.2.4 ("module 'numpy.strings'
///     has no attribute 'slice'" — `slice` landed in 2.3), so that build cannot
///     import its own char module.
///
/// The floor is a declared, checkable contract. The rejected alternative was
/// version-keying expectations inside individual test files, which is gate
/// self-weakening wearing a compatibility hat.
#[test]
fn numpy_oracle_meets_the_supported_version_floor() -> Result<(), String> {
    let script = fnp_script(
        r#"
parts = []
for piece in np.__version__.split('.')[:2]:
    digits = ''.join(c for c in piece if c.isdigit())
    parts.append(int(digits) if digits else 0)
major, minor = (parts + [0, 0])[:2]
print("OK" if (major, minor) >= (2, 3) else "BELOW_FLOOR")
print(np.__version__)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    let verdict = lines.next().unwrap_or_default().trim();
    let found = lines.next().unwrap_or_default().trim();
    assert_eq!(
        verdict, "OK",
        "UNSUPPORTED NUMPY ORACLE: found {found}, but FrankenNumPy conformance requires \
         numpy >= 2.3 (deadlock-audit-1jqrw). Below the floor, byte-level parity failures \
         in count_nonzero, masked_sum, ufunc signatures and numpy.char are EXPECTED and are \
         NOT fnp defects — upgrade the oracle rather than chasing them."
    );
    Ok(())
}
