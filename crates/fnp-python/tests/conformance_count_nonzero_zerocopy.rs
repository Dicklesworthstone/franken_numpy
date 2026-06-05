//! Bit-exact conformance lock for the zero-copy `numpy.count_nonzero` fast path
//! (`try_zerocopy_count_nonzero`).
//!
//! Counting is order-independent, so the zero-copy buffer count is trivially
//! bit-identical to numpy. This pins both the per-axis int64 output bytes and the
//! axis=None numpy.int64 scalar surface across bool and float64 inputs (for
//! float64, x != 0.0 excludes +-0.0 and includes NaN, matching numpy).

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
fn count_nonzero_zerocopy_bool_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for shp in [(100, 50), (5, 5, 5)]:
    m = rng.standard_normal(shp) > 0
    f = rng.standard_normal(shp)
    for axis in range(len(shp)):
        chunks.append(np.asarray(mod.count_nonzero(m, axis=axis)).tobytes())
        chunks.append(np.asarray(mod.count_nonzero(f, axis=axis)).tobytes())
# axis=None scalar surface: type tag (numpy.int64) + value
total = mod.count_nonzero(rng.standard_normal((200, 200)) > 0)
chunks.append(bytes([1 if isinstance(total, np.int64) else 0]))
chunks.append(str(int(total)).encode())
# float special values: -0.0 excluded, nan included
fe = np.array([0.0, -0.0, 1.0, np.nan, 0.0, -2.0, np.inf], dtype=np.float64).reshape(7, 1)
chunks.append(np.asarray(mod.count_nonzero(fe, axis=0)).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy count_nonzero must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}
