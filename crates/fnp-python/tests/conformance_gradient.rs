//! Conformance tests for numpy.gradient against NumPy oracle.
//!
//! Tests gradient (gradient of an N-dimensional array).

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
fn gradient_1d_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([1, 2, 4, 7, 11])
result = fnp.gradient(f)
expected = np.gradient(f)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient 1D basic should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_with_spacing() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([1, 2, 4, 7, 11])
result = fnp.gradient(f, 2.0)
expected = np.gradient(f, 2.0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient with spacing should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([[1, 2, 6], [3, 4, 5], [7, 8, 9]])
result_y, result_x = fnp.gradient(f)
expected_y, expected_x = np.gradient(f)
print(np.allclose(result_y, expected_y) and np.allclose(result_x, expected_x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "gradient 2D should match numpy");
    Ok(())
}

#[test]
fn gradient_single_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([[1, 2, 6], [3, 4, 5], [7, 8, 9]])
result = fnp.gradient(f, axis=0)
expected = np.gradient(f, axis=0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient single axis should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 3+4j, 6+9j], dtype=np.complex128)
fnp_result = fnp.gradient(z)
np_result = np.gradient(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "gradient complex should match numpy");
    Ok(())
}

#[test]
fn gradient_edge_order_1() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([1, 2, 4, 7, 11])
fnp_result = fnp.gradient(f, edge_order=1)
np_result = np.gradient(f, edge_order=1)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient edge_order=1 should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_edge_order_2() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([1, 2, 4, 7, 11])
fnp_result = fnp.gradient(f, edge_order=2)
np_result = np.gradient(f, edge_order=2)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient edge_order=2 should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([1.0, np.inf, 3.0, np.nan, 5.0])
fnp_result = fnp.gradient(f)
np_result = np.gradient(f)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient special values should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_constant_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([5.0, 5.0, 5.0, 5.0])
fnp_result = fnp.gradient(f)
np_result = np.gradient(f)
# Gradient of constant should be zero
print(np.allclose(fnp_result, np_result) and np.allclose(fnp_result, 0.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient constant array should match numpy"
    );
    Ok(())
}

fn classify_error(script: &str) -> String {
    let output = std::process::Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("python3 should be available");
    if output.status.success() {
        "ok".to_string()
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("ValueError") {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn gradient_single_element_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
f = np.array([5.0])
fnp.gradient(f)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
f = np.array([5.0])
np.gradient(f)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "gradient single element should raise same error as numpy"
    );
}

#[test]
fn gradient_two_elements() -> Result<(), String> {
    let script = fnp_script(
        r#"
f = np.array([1.0, 5.0])
fnp_result = fnp.gradient(f)
np_result = np.gradient(f)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient two elements should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_strided_nonlast_axis_matches_numpy() -> Result<(), String> {
    // Exercises the native non-last (strided) single-axis gradient against numpy
    // bit-exactly (atol=0, equal_nan=True) incl dtype/shape: 2-D axis=0, 3-D middle
    // and first axes, negative axis index, a non-uniform-spacing fallthrough (scalar
    // dx), edge_order=2 fallthrough, a NaN-containing array, and the last-axis case
    // (which must route through the existing contiguous path and still match).
    let script = fnp_script(
        r#"
def same(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.allclose(a, b, rtol=0, atol=0, equal_nan=True)

rng = np.random.default_rng(13)
m2 = rng.standard_normal((64, 40))
m3 = rng.standard_normal((12, 9, 7))
nanm = rng.standard_normal((20, 8)); nanm[3, 4] = np.nan; nanm[0, 1] = np.inf
ok = True
cases = [
    (m2, 0, 1.0, 1),
    (m2, -2, 1.0, 1),
    (m3, 1, 1.0, 1),
    (m3, 0, 1.0, 1),
    (m2, 0, 0.5, 1),    # scalar non-unit spacing
    (m2, 0, 1.0, 2),    # edge_order=2 -> fallthrough to numpy
    (nanm, 0, 1.0, 1),  # NaN/Inf propagation
    (m2, 1, 1.0, 1),    # last axis (existing path)
]
for arr, axis, dx, eo in cases:
    f = fnp.gradient(arr, dx, axis=axis, edge_order=eo)
    n = np.gradient(arr, dx, axis=axis, edge_order=eo)
    if not same(f, n):
        print("FAIL", axis, dx, eo, np.asarray(f), np.asarray(n)); ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "non-last-axis gradient parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn gradient_full_no_axis_tuple_matches_numpy() -> Result<(), String> {
    // Exercises the native no-axis full gradient (returns a TUPLE of per-axis gradients)
    // against numpy bit-exactly (atol=0, equal_nan=True): 2-D and 3-D arrays, default and
    // scalar spacing, plus 1-D (single-array return) and an edge_order=2 fallthrough.
    let script = fnp_script(
        r#"
def same_seq(fs, ns):
    if type(fs).__name__ != type(ns).__name__:
        # 1-D returns a single ndarray, N-D returns a tuple; require the same kind
        if isinstance(fs, np.ndarray) and isinstance(ns, np.ndarray):
            pass
        else:
            return False
    fs = fs if isinstance(fs, tuple) else (fs,)
    ns = ns if isinstance(ns, tuple) else (ns,)
    if len(fs) != len(ns):
        return False
    for a, b in zip(fs, ns):
        a = np.asarray(a); b = np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype or not np.allclose(a, b, rtol=0, atol=0, equal_nan=True):
            return False
    return True

rng = np.random.default_rng(37)
f2 = rng.standard_normal((128, 96))
f3 = rng.standard_normal((24, 18, 11))
f1 = rng.standard_normal(5000)
ok = True
# no-axis full gradient
for f in (f2, f3, f1):
    if not same_seq(fnp.gradient(f), np.gradient(f)):
        print("FAIL no-axis", f.shape); ok = False
# scalar spacing applied to all axes
if not same_seq(fnp.gradient(f2, 0.5), np.gradient(f2, 0.5)):
    print("FAIL scalar-dx"); ok = False
# edge_order=2 must defer + still match
if not same_seq(fnp.gradient(f2, edge_order=2), np.gradient(f2, edge_order=2)):
    print("FAIL edge_order=2"); ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "no-axis full gradient parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_gradient_via_f64_conversion_bit_exact_matches_numpy() -> Result<(), String> {
    // Integer inputs within +-2^51 convert once to f64 and ride every proven
    // f64 kernel; numpy's int-subtract-then-divide chain is byte-equivalent
    // there (exact integer differences either way). Out-of-range i64/u64 and
    // small arrays keep the delegate. Rows cover 1-D/2-D axes, full-tuple,
    // edge_order=2, scalar dx, coordinate arrays, widths, negatives.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(131)
verdicts = []
g1 = rng.integers(-100_000, 100_000, 2_000_000)
g2 = rng.integers(-100_000, 100_000, (2048, 1024))
for dt in [np.int64, np.int32, np.int16, np.uint8]:
    a1 = g1.astype(dt); a2 = g2.astype(dt)
    r = fnp.gradient(a1); e = np.gradient(a1)
    if r.dtype != e.dtype or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL 1-D {dt.__name__}")
    for ax in (0, 1, -1):
        r = fnp.gradient(a2, axis=ax); e = np.gradient(a2, axis=ax)
        if r.tobytes() != e.tobytes():
            verdicts.append(f"FAIL 2-D ax={ax} {dt.__name__}")
# full tuple, scalar dx, edge_order=2, coordinate arrays
a2 = g2.copy()
rt = fnp.gradient(a2); et = np.gradient(a2)
if len(rt) != 2 or any(rt[i].tobytes() != et[i].tobytes() for i in range(2)):
    verdicts.append("FAIL full tuple")
if fnp.gradient(g1, 0.5).tobytes() != np.gradient(g1, 0.5).tobytes():
    verdicts.append("FAIL scalar dx")
if fnp.gradient(g1, edge_order=2).tobytes() != np.gradient(g1, edge_order=2).tobytes():
    verdicts.append("FAIL edge_order=2")
cx = np.cumsum(rng.random(2_000_000)) + 1.0
if fnp.gradient(g1, cx).tobytes() != np.gradient(g1, cx).tobytes():
    verdicts.append("FAIL coordinate array")
# out-of-range i64 / u64 keep the numpy delegate (wrap contract)
big = rng.integers(2**60, 2**62, 200_000)
if fnp.gradient(big).tobytes() != np.gradient(big).tobytes():
    verdicts.append("FAIL huge-i64 delegate")
ub = (rng.integers(0, 2**62, 200_000)).astype(np.uint64)
if fnp.gradient(ub).tobytes() != np.gradient(ub).tobytes():
    verdicts.append("FAIL huge-u64 delegate")
sm = rng.integers(-50, 50, 500)
if fnp.gradient(sm).tobytes() != np.gradient(sm).tobytes():
    verdicts.append("FAIL small delegate")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.integers(-100_000, 100_000, (4096, 4096))
tn = best(lambda: np.gradient(W, axis=0)); tf = best(lambda: fnp.gradient(W, axis=0))
print(f"GRADIENT_INT_AX0_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
W1 = rng.integers(-100_000, 100_000, 16_000_000)
tn = best(lambda: np.gradient(W1)); tf = best(lambda: fnp.gradient(W1))
print(f"GRADIENT_INT_1D_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces GRADIENT_INT_*_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "int gradient via f64 conversion must be bit-identical to numpy: {result}"
    );
    Ok(())
}
