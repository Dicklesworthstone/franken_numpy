//! Bit-exact conformance lock for the zero-copy `numpy.cumprod` fast path
//! (`try_zerocopy_f64_cumprod`).
//!
//! The native cumprod accumulates strictly left-to-right (out[i] = out[i-1] *
//! in[i]) with the first element copied verbatim, so parity must hold at the
//! IEEE-754 bit level — non-associative f64 multiplies mean any reordering would
//! diverge, and a leading -0.0 must keep its sign. This compares the sha256 of
//! the raw output bytes against the NumPy oracle across 1-D and multi-D
//! (axis=None flatten) inputs plus signed zeros. (Inputs are kept near 1.0 so
//! the running product neither overflows to inf nor underflows to 0 — products
//! that reach inf*0 would generate an invalid-op NaN whose payload bits are not
//! stable across opt levels, though the shipped release build still matches
//! numpy bit-for-bit.)

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
use support::fnp_script;

#[test]
fn cumprod_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for n in [1000, 100003]:
    chunks.append(np.asarray(mod.cumprod(rng.standard_normal(n) * 1.0005)).tobytes())
chunks.append(np.asarray(mod.cumprod(rng.standard_normal((30, 40)) * 1.0005)).tobytes())
chunks.append(np.asarray(mod.cumprod(np.array([-0.0, 2.0, -0.5, 3.0, 1.5, -1.0, 2.0]))).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy cumprod must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}

/// Locks the zero-copy per-axis cumprod fast path
/// (`try_zerocopy_f64_cumulative_axis`, product variant) to bit-exact parity.
/// Inputs are kept near 1.0 so the running product stays finite (see the
/// module note on opt-level-unstable invalid-op NaN payloads). Compares the
/// sha256 of raw output bytes across every axis of 2-D and 3-D inputs.
#[test]
fn cumprod_axis_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for shp in [(30, 40), (5, 5, 5), (100, 200)]:
    x = 1.0 + rng.standard_normal(shp) * 0.0005
    for axis in range(len(shp)):
        chunks.append(np.asarray(mod.cumprod(x, axis=axis)).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy per-axis cumprod must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}

/// A float64 cumprod's floating-point reports - warnings and FloatingPointErrors - are numpy's
/// under EVERY errstate mode, on the data where the native kernel skips work (bead
/// `deadlock-audit-1uf80`): when every lane ends finite only `under` is possible, and the
/// replay that finds it now runs only when numpy's errstate does not ignore underflow. That
/// skip took `cumprod` of 262,144 values in [0.5, 1.5) - which underflows - from 1.79x numpy's
/// time to 0.38x; a skip that also fired under `under='warn'`/`'raise'` would lose numpy's
/// "underflow encountered in accumulate", which this pins, alongside overflow, inf * 0 and a
/// clean product, over axis None/0/-1.
#[test]
fn cumprod_float_errors_match_numpy_under_every_errstate_mode() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(9)
data = {
    "underflow-to-0": rng.random(65536) + 0.5,
    "subnormal-end": np.array([1e-300, 1e-10]),
    "overflow": np.array([1e300, 1e300]),
    "inf*0": np.array([np.inf, 0.0]),
    "finite": np.array([1.5, 2.0, 3.0]),
    "2d": rng.random((4000, 3)) + 0.5,
}
bad, cells, reported = [], 0, 0
for mode in ("default", "warn", "raise", "ignore"):
    for label, x in data.items():
        for axis in (None, 0, -1):
            out = {}
            for m in (fnp, np):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    ctx = np.errstate() if mode == "default" else np.errstate(under=mode, over=mode, invalid=mode)
                    with ctx:
                        try:
                            got = ("ok", np.asarray(m.cumprod(x, axis=axis)).tobytes())
                        except Exception as ex:
                            got = (type(ex).__name__, str(ex))
                out[m is fnp] = (got, sorted(str(c.message) for c in w))
            cells += 1
            reported += bool(out[False][1]) or out[False][0][0] != "ok"
            if out[True] != out[False]:
                bad.append(f"{mode} {label} axis={axis}: fnp={out[True][1]} {out[True][0][0]} numpy={out[False][1]} {out[False][0][0]}")
print(cells, reported, bad)
"#
        .to_string(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("");
    let mut fields = last.splitn(3, ' ');
    let cells: usize = fields.next().and_then(|n| n.parse().ok()).unwrap_or(0);
    let reported: usize = fields.next().and_then(|n| n.parse().ok()).unwrap_or(0);
    assert_eq!(cells, 72, "{result}");
    assert!(
        reported >= 20,
        "numpy must report on a good share of these cells, or the sweep proves nothing: {result}"
    );
    assert!(
        last.ends_with(" []"),
        "cumprod's float-error reports must be numpy's: {result}"
    );
    Ok(())
}
