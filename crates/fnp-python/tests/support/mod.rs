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

use std::path::{Path, PathBuf};
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

/// Wraps a Python snippet so it runs against the built extension with `fnp` and `np` bound.
///
/// This is the shared form of the `fnp_script` each suite used to carry its own copy of.
pub fn fnp_script(body: String) -> String {
    let module_literal = format!("{:?}", extension_module_path());
    format!(
        "import importlib.util\n\
         import numpy as np\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}
