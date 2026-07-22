//! Bit-exact conformance lock for the zero-copy `numpy.cumsum` fast path
//! (`try_zerocopy_f64_cumsum`).
//!
//! The native cumsum accumulates strictly left-to-right (out[i] = out[i-1] +
//! in[i]) with the first element copied verbatim, so parity must hold at the
//! IEEE-754 bit level — non-associative f64 adds mean any reordering would
//! diverge, and a leading -0.0 must keep its sign. This compares the sha256 of
//! the raw output bytes against the NumPy oracle across 1-D and multi-D
//! (axis=None flatten) inputs, a leading -0.0, and inf/nan extremes — all of
//! which take the zero-copy path.

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
fn cumsum_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for n in [1000, 100003]:
    chunks.append(np.asarray(mod.cumsum(rng.standard_normal(n))).tobytes())
chunks.append(np.asarray(mod.cumsum(rng.standard_normal((30, 40)))).tobytes())
# A leading -0.0 plus finite extremes and a trailing nan that propagates. (We
# avoid inf + -inf here: that *generates* an invalid-op NaN whose payload bits
# are not stable across opt levels, while the shipped release build still
# matches numpy bit-for-bit.)
chunks.append(np.asarray(mod.cumsum(np.array([-0.0, 1.5, -2.5, 3.0, 1e300, -1e300, np.nan]))).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy cumsum must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}

/// Locks the zero-copy per-axis cumsum fast path
/// (`try_zerocopy_f64_cumulative_axis`, sum variant) to bit-exact parity. The
/// per-axis running sum accumulates strictly along the axis (same order as
/// numpy), so parity must hold at the IEEE-754 bit level. Compares the sha256 of
/// raw output bytes across every axis of 2-D and 3-D inputs.
#[test]
fn cumsum_axis_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for shp in [(30, 40), (5, 5, 5), (100, 200)]:
    x = rng.standard_normal(shp)
    for axis in range(len(shp)):
        chunks.append(np.asarray(mod.cumsum(x, axis=axis)).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy per-axis cumsum must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}

#[test]
fn flat_int_cumsum_cumprod_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // Above the 1<<21 gate the flat 1-D integer cumsum/cumprod runs the two-pass
    // parallel prefix scan. Integer wrapping add/mul are associative, so the block
    // scan must be byte-identical to numpy's serial wrapping accumulate — incl. the
    // overflow-WRAP case (numpy promotes to int64/uint64 and wraps in 2's complement).
    let script = fnp_script(
        r#"
n = (1 << 21) + 65
ok = True

# int64 cumsum that OVERFLOWS -> wraps (tests wrapping bit-exactness)
x = np.full(n, 9_000_000_000_000_000_000, dtype=np.int64)
a = fnp.cumsum(x); e = np.cumsum(x)
ok = ok and a.dtype == e.dtype and a.shape == e.shape and a.tobytes() == e.tobytes()

# int32 cumsum -> int64 accumulator (widening path)
x32 = (np.arange(n, dtype=np.int32) % 1000) - 500
a = fnp.cumsum(x32); e = np.cumsum(x32)
ok = ok and a.dtype == e.dtype and a.tobytes() == e.tobytes()

# uint64 cumsum wrap
xu = np.full(n, 18_000_000_000_000_000_000, dtype=np.uint64)
a = fnp.cumsum(xu); e = np.cumsum(xu)
ok = ok and a.dtype == e.dtype and a.tobytes() == e.tobytes()

# int64 cumprod that WRAPS (alternating sign + magnitude grows then wraps)
xp = np.where(np.arange(n) % 7 == 0, np.int64(-3), np.int64(2))
a = fnp.cumprod(xp); e = np.cumprod(xp)
ok = ok and a.dtype == e.dtype and a.tobytes() == e.tobytes()

print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large parallel flat int cumsum/cumprod must be bit-identical to numpy"
    );
    Ok(())
}
