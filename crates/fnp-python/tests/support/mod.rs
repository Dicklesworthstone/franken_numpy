//! Locating the built extension for the conformance suites that drive it through a REAL
//! interpreter (`python3 -c ...`) rather than through `common`'s in-process harness.
//!
//! WHY THIS SEARCHES INSTEAD OF ASSUMING (`deadlock-audit-ovdn9`). Every such suite used to
//! compute the path as a SIBLING of its own test binary:
//!
//!     std::env::current_exe()?.parent()?.join("libfnp_python.so")
//!
//! which holds in a `target/debug/deps` layout and nowhere else. Measured on worker hz4, the
//! two artifacts land under DIFFERENT hashes:
//!
//!     .rch-target/debug/build/fnp-python/0efa4e6151b9ad04/out/conformance_sqrt-0efa4e...
//!     .rch-target/debug/build/fnp-python/dfbfae95a8d36650/out/libfnp_python.so
//!
//! so the sibling did not exist and all 15 cases of conformance_sqrt failed with
//! `ImportError: ... cannot open shared object file`, on every worker - and rch is the only
//! sanctioned build route here. A MUST clause panics, which aborts the whole test binary and
//! hides every target listed after it, so this was never a cosmetic failure.
//!
//! WHY IT IS NOT IN `common`: 142 suites need this and none of them need `common`'s pyo3
//! harness, which every one of them would otherwise recompile. `cargo test -p fnp-python`
//! already has ~2300 targets and sits near rch's 1800s ceiling, so this module deliberately
//! depends on `std` alone.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

/// Newest `<child>/out/<name>` among a directory's immediate children.
///
/// Cargo writes one `<hash>/out/` per build unit and gives the cdylib and the test binary
/// DIFFERENT hashes, so the cdylib is a COUSIN of the test binary, not a sibling. Several
/// stale hashes can coexist in a pooled target directory, so the newest wins.
fn newest_in_cousin_out_dirs(parent: &Path, name: &str) -> Option<PathBuf> {
    let mut best: Option<(SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(parent).ok()?.flatten() {
        let candidate = entry.path().join("out").join(name);
        let Ok(metadata) = std::fs::metadata(&candidate) else {
            continue;
        };
        let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        if best
            .as_ref()
            .is_none_or(|(best_time, _)| modified > *best_time)
        {
            best = Some((modified, candidate));
        }
    }
    best.map(|(_, path)| path)
}

/// Absolute path to the built `fnp_python` extension module.
///
/// Falls back to the bare file name, which reproduces the original error text when the
/// extension genuinely has not been built. `FNP_PYTHON_MODULE` overrides everything, for a
/// binary copied to a measurement host.
pub fn extension_module_path() -> PathBuf {
    let name = format!(
        "{}fnp_python{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    if let Some(explicit) = std::env::var_os("FNP_PYTHON_MODULE") {
        return PathBuf::from(explicit);
    }
    let Ok(exe) = std::env::current_exe() else {
        return PathBuf::from(name);
    };

    // 1. The `target/debug/deps` layout: the cdylib sits next to the test binary.
    if let Some(parent) = exe.parent() {
        let sibling = parent.join(&name);
        if sibling.exists() {
            return sibling;
        }
    }

    for ancestor in exe.ancestors() {
        // 2. Any ancestor profile directory: `target/debug/` and its `deps/`.
        for candidate in [ancestor.join(&name), ancestor.join("deps").join(&name)] {
            if candidate.exists() {
                return candidate;
            }
        }
        // 3. The pooled layout: when an ancestor is the package's own `build/fnp-python`
        //    directory, the cdylib is under a SIBLING hash.
        if ancestor
            .file_name()
            .is_some_and(|name| name == "fnp-python")
            && let Some(found) = newest_in_cousin_out_dirs(ancestor, &name)
        {
            return found;
        }
    }

    PathBuf::from(name)
}

/// The parity predicate every suite should use instead of a hand-rolled `tobytes()`.
///
/// THE RULE, and its one dtype exception (`deadlock-audit-6cukd`): compare RAW BYTES, because
/// a value comparison cannot see a NaN's SIGN BIT or its payload and this campaign has shipped
/// fixes for exactly that class - EXCEPT for
///
///   * `longdouble` / `clongdouble` (dtype char 'g' / 'G'). x86 stores an 80-bit value in a
///     16-byte slot; the remaining 6 bytes are NOT initialised and differ between two
///     independently allocated results. A raw-byte probe over 23 unary ops x 15 dtypes
///     reported 34 phantom divergences on longdouble alone, every one of them padding.
///   * `object` and any dtype with `hasobject`, which have no byte representation at all -
///     the bytes are pointers, and two correct results hold different ones.
///
/// Both fall back to a VALUE comparison. Relaxing the byte comparison everywhere would be the
/// wrong repair: it would blind the suites to the NaN sign-bit class they exist to catch.
const PARITY_PREDICATE: &str = "\
def parity_equal(ours, theirs):\n\
\x20   import numpy as _np\n\
\x20   if type(ours) is not type(theirs):\n\
\x20       return False\n\
\x20   if not isinstance(ours, _np.ndarray):\n\
\x20       return ours == theirs or (ours != ours and theirs != theirs)\n\
\x20   if ours.dtype != theirs.dtype or ours.shape != theirs.shape:\n\
\x20       return False\n\
\x20   if ours.dtype.hasobject or ours.dtype.char in 'gG':\n\
\x20       return bool(_np.array_equal(ours, theirs, equal_nan=ours.dtype.kind == 'f'))\n\
\x20   return ours.tobytes() == theirs.tobytes()\n";

/// Wraps a Python snippet so it runs against the built extension with `fnp` and `np` bound,
/// and `parity_equal` available.
///
/// This is the shared form of the `fnp_script` each suite used to carry its own copy of.
pub fn fnp_script(body: String) -> String {
    fnp_script_with("", false, body)
}

/// `fnp_script` with EXTRA IMPORTS and optional `sys.modules` registration.
///
/// Nine suites need slightly more than `np` in scope - `numpy.ma`, `io`, `BytesIO`/`StringIO` -
/// and the re-export suites need the module registered in `sys.modules` BEFORE `exec_module`
/// runs, because that is the behaviour they assert on. Those were the only differences between
/// their local `fnp_script` copies and this one, so they keep a one-line wrapper and their call
/// sites are untouched, rather than each carrying a duplicated resolver
/// (`deadlock-audit-propagate-extension-resolver-to-135-suites`).
pub fn fnp_script_with(extra_imports: &str, register_in_sys_modules: bool, body: String) -> String {
    let module_literal = format!("{:?}", extension_module_path());
    let registration = if register_in_sys_modules {
        "sys.modules[spec.name] = fnp\n"
    } else {
        ""
    };
    format!(
        "import importlib.machinery, importlib.util\n\
         import numpy as np\n\
         {extra_imports}\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal}, \
         loader=importlib.machinery.ExtensionFileLoader('fnp_python', {module_literal}))\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         {registration}\
         spec.loader.exec_module(fnp)\n\
         {PARITY_PREDICATE}\
         {body}"
    )
}

/// Resolves the NumPy oracle Python interpreter from explicit env-var values.
///
/// Precedence: `FNP_ORACLE_PYTHON` -> `PYO3_PYTHON` -> `"python3"`.
/// Blank or whitespace-only values are ignored so callers can unset via `=""`.
pub fn resolve_oracle_python_from(
    fnp_oracle_python: Option<&str>,
    pyo3_python: Option<&str>,
) -> String {
    fnp_oracle_python
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or_else(|| pyo3_python.map(str::trim).filter(|s| !s.is_empty()))
        .unwrap_or("python3")
        .to_string()
}

/// Resolves the configured NumPy oracle Python interpreter from the process environment.
pub fn resolve_oracle_python() -> String {
    let fnp = std::env::var("FNP_ORACLE_PYTHON").ok();
    let pyo3 = std::env::var("PYO3_PYTHON").ok();
    resolve_oracle_python_from(fnp.as_deref(), pyo3.as_deref())
}

/// Queries `<python>` for its `major.minor` CPython version string (e.g. `"3.12"`).
pub fn python_minor_version(python: &str) -> Result<String, String> {
    let output = Command::new(python)
        .args([
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
        ])
        .output()
        .map_err(|error| format!("failed to invoke oracle interpreter {python:?}: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "oracle interpreter {python:?} failed version probe: {stderr}"
        ));
    }
    let xy = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if xy.split('.').count() != 2 {
        return Err(format!(
            "oracle interpreter {python:?} returned unexpected version {xy:?}"
        ));
    }
    Ok(xy)
}

/// Verifies that the oracle interpreter's `major.minor` matches the CPython ABI
/// the `fnp_python` extension was compiled against (`FNP_BUILT_PYTHON_XY`).
///
/// A mismatch would otherwise fail deep inside `pyo3-ffi` with a misaligned-pointer
/// panic or segfault when the subprocess loads the compiled cdylib.
pub fn check_python_abi_match(python: &str, built_xy: Option<&str>) -> Result<(), String> {
    let Some(expected_xy) = built_xy.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(());
    };
    let Ok(actual_xy) = python_minor_version(python) else {
        // Leave missing/unrunnable executable errors to the caller's own `.output()` error path.
        return Ok(());
    };
    if actual_xy != expected_xy {
        return Err(format!(
            "fnp_python ABI mismatch: extension was built against Python {expected_xy}, \
             but oracle interpreter {python:?} is Python {actual_xy}. \
             Set FNP_ORACLE_PYTHON and PYO3_PYTHON to matching Python minor versions."
        ));
    }
    Ok(())
}

fn verify_oracle_abi_cached(python: &str) {
    static CACHE: OnceLock<Mutex<HashMap<String, Result<(), String>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let verdict = {
        let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .entry(python.to_string())
            .or_insert_with(|| check_python_abi_match(python, option_env!("FNP_BUILT_PYTHON_XY")))
            .clone()
    };
    if let Err(message) = verdict {
        panic!("{message}");
    }
}

/// Returns a `std::process::Command` targeting the configured NumPy oracle interpreter.
///
/// Precedence: `FNP_ORACLE_PYTHON` -> `PYO3_PYTHON` -> `"python3"`.
/// Also verifies once per interpreter that its `major.minor` matches the build-time
/// CPython ABI (`FNP_BUILT_PYTHON_XY`) so mismatches report a clear error instead of
/// panicking inside `pyo3-ffi`.
pub fn python_command() -> Command {
    let python = resolve_oracle_python();
    verify_oracle_abi_cached(&python);
    Command::new(python)
}
