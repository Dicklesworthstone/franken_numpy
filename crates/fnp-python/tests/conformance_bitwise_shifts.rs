//! Conformance tests for numpy bitwise shift functions against NumPy oracle.
//!
//! Tests bitwise_left_shift, bitwise_right_shift, bitwise_count.

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

// ─────────────────────────────────────────────────────────────────────────────
// bitwise_left_shift
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bitwise_left_shift_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 8], dtype='int64')
result = fnp.bitwise_left_shift(a, 1)
expected = np.left_shift(a, 1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bitwise_left_shift basic should match numpy"
    );
    Ok(())
}

#[test]
fn bitwise_left_shift_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 8], dtype='int64')
b = np.array([1, 2, 3, 4], dtype='int64')
result = fnp.bitwise_left_shift(a, b)
expected = np.left_shift(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bitwise_left_shift array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// bitwise_right_shift
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bitwise_right_shift_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([16, 32, 64, 128], dtype='int64')
result = fnp.bitwise_right_shift(a, 1)
expected = np.right_shift(a, 1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bitwise_right_shift basic should match numpy"
    );
    Ok(())
}

#[test]
fn bitwise_right_shift_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([16, 32, 64, 128], dtype='int64')
b = np.array([1, 2, 3, 4], dtype='int64')
result = fnp.bitwise_right_shift(a, b)
expected = np.right_shift(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bitwise_right_shift array should match numpy"
    );
    Ok(())
}

#[test]
fn int64_shift_zerocopy_bit_exact_golden_sha256() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE

def call(module, name, a, b):
    if module is np and name == "bitwise_left_shift":
        return np.left_shift(a, b)
    if module is np and name == "bitwise_right_shift":
        return np.right_shift(a, b)
    return getattr(module, name)(a, b)

ops = ["left_shift", "right_shift", "bitwise_left_shift", "bitwise_right_shift"]
base = np.array([-(2**62), -257, -8, -1, 0, 1, 7, 255, 2**62 - 1], dtype=np.int64)
shifts = np.array([-2, -1, 0, 1, 7, 63, 64, 65, 3], dtype=np.int64)
mat = np.arange(-60, 60, dtype=np.int64).reshape(10, 12)
mat_shifts = (np.arange(120, dtype=np.int64).reshape(10, 12) % 70) - 3
chunks = []
for name in ops:
    for a, b in [
        (base, np.int64(-1)),
        (base, np.int64(0)),
        (base, np.int64(1)),
        (base, np.int64(63)),
        (base, np.int64(64)),
        (base, np.int64(65)),
        (base, 3),
        (base, shifts),
        (mat, mat_shifts),
    ]:
        got = np.asarray(call(mod, name, a, b))
        expected = np.asarray(call(np, name, a, b))
        assert got.dtype == expected.dtype, (name, got.dtype, expected.dtype)
        assert got.shape == expected.shape, (name, got.shape, expected.shape)
        assert got.tobytes() == expected.tobytes(), (name, a, b, got, expected)
        chunks.append(str(got.dtype).encode())
        chunks.append(str(got.shape).encode())
        chunks.append(got.tobytes())
print(hashlib.sha256(b"".join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy int64 shifts must be bit-identical to numpy"
    );
    assert_eq!(
        fnp_hash, "9a91043b1d91535deadb96ba5072446f43ceec53d7d8226be845a2a5ac51cf5d",
        "golden sha256 of int64 shift dtype/shape/raw-output bytes"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// bitwise_count
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bitwise_count_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 7, 15, 255], dtype='uint8')
result = fnp.bitwise_count(a)
expected = np.bitwise_count(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bitwise_count basic should match numpy"
    );
    Ok(())
}

#[test]
fn bitwise_count_int64() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 3, 7, 15], dtype='int64')
result = fnp.bitwise_count(a)
expected = np.bitwise_count(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bitwise_count int64 should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn left_right_shift_inverse() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 8], dtype='int64')
shifted = fnp.bitwise_left_shift(a, 2)
back = fnp.bitwise_right_shift(shifted, 2)
print(np.array_equal(a, back))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "left then right shift should restore original"
    );
    Ok(())
}

#[test]
fn bitwise_count_power_of_2() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Powers of 2 have exactly one bit set
powers = np.array([1, 2, 4, 8, 16, 32, 64], dtype='int64')
counts = fnp.bitwise_count(powers)
print(np.all(counts == 1))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "powers of 2 should have count 1");
    Ok(())
}

#[test]
fn bitwise_shifts_scalar_return_type_matches_numpy() -> Result<(), String> {
    for func in &["bitwise_left_shift", "bitwise_right_shift"] {
        let script = format!(
            "import numpy as np; x = np.int64(8); y = np.int64(2); r = np.{func}(x, y); print(type(r).__name__, r)"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "x = np.int64(8); y = np.int64(2); r = fnp.{func}(x, y); print(type(r).__name__, r)"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "{func} scalar return type mismatch\nnumpy: {numpy_result}\nfnp: {rust_result}"
        );
    }

    Ok(())
}

#[test]
fn bitwise_count_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script =
        "import numpy as np; x = np.int64(15); r = np.bitwise_count(x); print(type(r).__name__, r)";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("x = np.int64(15); r = fnp.bitwise_count(x); print(type(r).__name__, r)".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "bitwise_count scalar return type mismatch\nnumpy: {numpy_result}\nfnp: {rust_result}"
    );

    Ok(())
}
