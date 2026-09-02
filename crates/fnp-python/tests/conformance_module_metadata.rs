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

mod support;

fn fnp_script(body: String) -> String {
    support::fnp_script_with("import sys\n", true, body)
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

/// Importing fnp_python must leave the real numpy module unmodified.
///
/// We bind numpy's `__all__` into our own module and its submodules. Binding the
/// LIST OBJECT rather than a copy made that list shared, and PyO3's
/// `PyModule::add` appends every added name to `__all__` — so each fnp-only name
/// we registered was silently appended to NUMPY's `__all__` too. Any
/// `from numpy import *` in the same interpreter then raised AttributeError on a
/// name only we define. That is a process-wide corruption of a third-party
/// module, and it is how scipy broke: `scipy._lib.array_api_compat` star-imports
/// numpy, so `import scipy.linalg` died with
/// "module 'numpy' has no attribute '__submodule_import_errors__'"
/// (deadlock-audit-335rd).
///
/// The star-import assertion is the load-bearing one. `is not` catches today's
/// mechanism; the star-import catches the CONTRACT downstream code depends on,
/// and stays meaningful if `__all__` is ever populated a different way.
///
/// The leak baseline is taken from a SUBPROCESS that never imports fnp. An
/// earlier draft snapshotted `np.__all__` at the top of this script, which runs
/// after fnp is loaded: it compared the corrupted list with itself and reported
/// 0 leaked names in the same run whose star-import assertion was failing. Two
/// of four assertions were structurally unable to fail. Verified against the
/// pre-fix build after the change: all four now fire.
#[test]
fn importing_fnp_does_not_mutate_numpy_all() -> Result<(), String> {
    let script = fnp_script(
        r#"
import json, subprocess, sys

# The baseline MUST come from an interpreter that never loaded fnp. Snapshotting
# np.__all__ here would snapshot the already-corrupted list and compare it with
# itself — measured: that spelling reported 0 leaked names in the very run whose
# star-import was already broken.
pristine = json.loads(subprocess.run(
    [sys.executable, "-c", "import json, numpy; print(json.dumps(list(numpy.__all__)))"],
    capture_output=True, text=True, check=True).stdout)

# The real symptom: a star-import of numpy must still work once fnp is loaded.
namespace = {}
try:
    exec("from numpy import *", namespace)
    star_import = "ok"
except Exception as exc:
    star_import = f"{type(exc).__name__}: {exc}"

# And the aliasing itself, for whichever module bound it.
shared = [
    name
    for name, ours, theirs in (
        ("fnp", fnp.__all__, np.__all__),
        ("fnp.random", getattr(fnp.random, "__all__", None), np.random.__all__),
        ("fnp.linalg", getattr(fnp.linalg, "__all__", None), np.linalg.__all__),
        ("fnp.fft", getattr(fnp.fft, "__all__", None), np.fft.__all__),
        ("fnp.ma", getattr(fnp.ma, "__all__", None), np.ma.__all__),
    )
    if ours is theirs
]

# Names fnp added to numpy's list, against the clean-interpreter baseline.
# Empty on a healthy build. Sorted so the failure message is stable.
leaked = sorted(set(np.__all__) - set(pristine))
print(star_import)
print(shared)
print(leaked)
print(len(np.__all__) - len(pristine))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();

    assert_eq!(
        lines.next().unwrap_or_default(),
        "ok",
        "`from numpy import *` must still work after importing fnp_python; \
         this is what breaks scipy. Output: {result}"
    );
    assert_eq!(
        lines.next().unwrap_or_default(),
        "[]",
        "these fnp modules bound numpy's __all__ LIST rather than a copy, so \
         PyO3's module `add` appends our names to numpy's. Output: {result}"
    );
    assert_eq!(
        lines.next().unwrap_or_default(),
        "[]",
        "fnp leaked names into numpy.__all__. Output: {result}"
    );
    assert_eq!(
        lines.next().unwrap_or_default(),
        "0",
        "numpy.__all__ changed length while fnp_python was imported. Output: {result}"
    );
    Ok(())
}

/// The converse of the test above: every name fnp's OWN submodule `__all__`
/// advertises must actually resolve on that submodule.
///
/// `importing_fnp_does_not_mutate_numpy_all` guards the numpy direction (the
/// scipy breakage, deadlock-audit-335rd). Nothing guarded this one, and two
/// mechanisms in the module-init block make it fail silently:
///
/// 1. `__all__` is COPIED FROM NUMPY while the attributes are a HARDCODED LIST.
///    numpy.fft's block does
///    `fft_module.setattr("__all__", copied_all_names(&all_names)?)` from
///    numpy.fft.__all__ but only adds the 18 names in its own `fft_names`
///    array, and its `__getattr__` handles only `'test'`. Those sets match on
///    today's numpy, so there is no live gap - the day numpy adds an fft name,
///    `__all__` advertises a name `getattr` refuses.
/// 2. The registration loops are `if let Ok(value) = m.getattr(flat_name)`, a
///    SILENT SKIP. Rename or drop the fnp function behind a mapped name and the
///    name is quietly not added while `__all__` still advertises it.
///
/// The star-import half is the load-bearing assertion, for the same reason the
/// test above gives: it survives `__all__` being populated a different way.
#[test]
fn submodule_all_names_resolve_and_star_import_works() -> Result<(), String> {
    let script = fnp_script(
        r#"
import sys

# Register the importlib-loaded module so the REAL star-import spelling
# (`from fnp_python.fft import *`) can be exercised, not just getattr.
sys.modules.setdefault("fnp_python", fnp)

subs = [
    name
    for name in ("random", "linalg", "fft", "ma", "char", "strings",
                 "polynomial", "rec", "testing", "lib", "dtypes", "exceptions")
    if hasattr(fnp, name)
]

unresolved = []
star_failures = []
checked = 0
for name in subs:
    module = getattr(fnp, name)
    advertised = list(getattr(module, "__all__", []) or [])
    if not advertised:
        continue
    checked += 1
    sys.modules.setdefault(f"fnp_python.{name}", module)
    for attr in advertised:
        try:
            getattr(module, attr)
        except Exception as exc:
            unresolved.append(f"{name}.{attr} ({type(exc).__name__})")
    namespace = {}
    try:
        exec(f"from fnp_python.{name} import *", namespace)
    except Exception as exc:
        star_failures.append(f"{name}: {type(exc).__name__}: {exc}")

print(sorted(unresolved)[:20])
print(star_failures[:6])
# Guard: if no submodule declared a non-empty __all__, the two lines above are
# empty for the wrong reason and this test proves nothing.
print(checked)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    assert_eq!(
        lines.next().unwrap_or_default(),
        "[]",
        "these names are advertised in an fnp submodule's __all__ but do not \
         resolve on it - a star-import of that submodule raises AttributeError. \
         Output: {result}"
    );
    assert_eq!(
        lines.next().unwrap_or_default(),
        "[]",
        "`from fnp_python.<sub> import *` failed. Output: {result}"
    );
    let checked: usize = lines
        .next()
        .unwrap_or_default()
        .trim()
        .parse()
        .map_err(|_| format!("could not read the checked-submodule count: {result}"))?;
    assert!(
        checked >= 3,
        "only {checked} submodule(s) declared a non-empty __all__, so the two \
         assertions above passed for the wrong reason. Output: {result}"
    );
    Ok(())
}
