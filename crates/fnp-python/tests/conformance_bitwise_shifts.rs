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

mod support;
use support::fnp_script;

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

/// Every integer width at full range - numpy counts the bits of |v|, so int8 -128 is 1 and -1 is
/// 1, where a count of the raw two's-complement bits gives 8 and 64 - on both sides of the pool
/// floors: 2-, 4- and 8-byte operands are native at every size (numpy's loop for them is ~10x
/// the native count), 1-byte ones are numpy's below the streaming floor. Plus 2-D, F-order and
/// strided operands, bool, 0-d. Values, dtype and shape.
#[test]
fn bitwise_count_every_width_matches_numpy_serial_and_pooled() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(31)
bad = []
cells = 0
for dt in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"):
    info = np.iinfo(dt)
    for n in (1, 7, 4096, 70_001, (1 << 20) + 5):
        a = rng.integers(info.min, info.max, n, dtype=dt, endpoint=True)
        if info.min < 0:
            a[:1] = info.min
            a[1:2] = -1
        views = [a, a.reshape(1, -1)]
        if n == 4096:
            views += [a.reshape(64, 64, order="F"), a[::3]]
        for v in views:
            cells += 1
            r, s = fnp.bitwise_count(v), np.bitwise_count(v)
            r, s = np.asarray(r), np.asarray(s)
            if r.dtype != s.dtype or r.shape != s.shape or not np.array_equal(r, s):
                bad.append(f"{dt} {n} {v.shape}")
for v in (np.array([True, False, True]), np.array(-5, dtype=np.int16), np.int32(-7)):
    cells += 1
    r, s = fnp.bitwise_count(v), np.bitwise_count(v)
    if type(r) is not type(s) or np.asarray(r).dtype != np.asarray(s).dtype or not np.array_equal(r, s):
        bad.append(f"{type(v).__name__} {v!r}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "99", "cell table drifted: {result}");
    assert_eq!(bad, "[]", "bitwise_count must match numpy: {result}");
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
