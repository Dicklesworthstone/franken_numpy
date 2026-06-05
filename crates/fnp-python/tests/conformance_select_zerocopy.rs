//! Bit-exact conformance lock for the zero-copy `numpy.select` fast path
//! (`try_zerocopy_f64_select`).
//!
//! The native single-pass select copies the first matching choice verbatim, so
//! parity must hold at the IEEE-754 bit level (signed zero, nan payloads, inf).
//! This compares the sha256 of the raw output bytes against the NumPy oracle
//! across first-match-wins overlaps, an explicit scalar default, multi-D shapes,
//! and signed-zero/inf/nan extremes — all of which take the zero-copy path.

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

fn fnp_script(body: &str) -> String {
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

/// The body builds a sha256 of concatenated raw output bytes for a fixed battery
/// of cases. `{module}` is either `fnp` or `np`, so the identical computation
/// runs against both and the digests must match exactly.
fn select_golden_body(module: &str) -> String {
    format!(
        r#"
import hashlib
rng = np.random.default_rng(20260605)
chunks = []
for shp in [(1000,), (30, 40), (7,)]:
    x = rng.standard_normal(shp)
    y = rng.standard_normal(shp)
    z = rng.standard_normal(shp)
    c1 = x > 0.3
    c2 = x > -0.3
    chunks.append(np.asarray({module}.select([c1, c2], [y, z], default=1.5)).tobytes())
xe = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e308, -1e308], dtype=np.float64)
ye = np.array([-0.0, 0.0, np.nan, np.inf, -np.inf, -1e308, 1e308], dtype=np.float64)
ce = np.array([True, False, True, False, True, False, True])
chunks.append(np.asarray({module}.select([ce], [xe], default=-0.0)).tobytes())
chunks.append(np.asarray({module}.select([ce, ~ce], [xe, ye])).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#
    )
}

#[test]
fn select_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let fnp_hash = numpy_oracle(&fnp_script(&select_golden_body("fnp")))?;

    let numpy_script = format!("import numpy as np\n{}", select_golden_body("np"));
    let numpy_hash = numpy_oracle(&numpy_script)?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy select must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}
