//! Conformance tests for numpy isin, frombuffer, fromiter against NumPy oracle.
//!
//! Tests isin, frombuffer, fromiter.

mod common;

use common::with_fnp_and_numpy;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use std::io::Write;
use std::process::{Command, Stdio};

fn numpy_oracle(script: &str) -> Result<String, String> {
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| format!("Python oracle stdin should be available\nScript: {script}"))?;
    stdin
        .write_all(script.as_bytes())
        .map_err(|error| format!("Python oracle stdin write should succeed: {error}"))?;
    drop(stdin);
    let output = child
        .wait_with_output()
        .map_err(|error| format!("Python oracle should finish: {error}\nScript: {script}"))?;
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
// isin
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isin_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
test_elements = np.array([2, 4])
result = fnp.isin(a, test_elements)
expected = np.isin(a, test_elements)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isin basic should match numpy");
    Ok(())
}

#[test]
fn isin_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
test_elements = [1, 3]
result = fnp.isin(a, test_elements)
expected = np.isin(a, test_elements)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isin 2d should match numpy");
    Ok(())
}

#[test]
fn isin_invert() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
test_elements = np.array([2, 4])
result = fnp.isin(a, test_elements, invert=True)
expected = np.isin(a, test_elements, invert=True)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isin invert should match numpy");
    Ok(())
}

#[test]
fn isin_empty_test() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
test_elements = np.array([])
result = fnp.isin(a, test_elements)
expected = np.isin(a, test_elements)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isin empty test should match numpy");
    Ok(())
}

/// Locks the dense-table integer isin path (and the hash fallback for wide ranges):
/// deterministic int64 arrays — one small-range (table) and one huge-range (hash) —
/// across invert, with the bool result BYTE-IDENTICAL to numpy.isin plus a sha256
/// golden over the fnp bytes. Membership is exact either way, so the table path is
/// bit-identical to both numpy and the prior hash path.
#[test]
fn isin_int_table_and_hash_paths_match_numpy_bytes_and_golden() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib
s = 0x2545F4914F6CDD1D
def nxt():
    global s
    s = (s * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
    return s
n = 4000
small = np.array([nxt() % 1500 for _ in range(n)], dtype=np.int64)
small_t = np.array([nxt() % 1500 for _ in range(700)], dtype=np.int64)
wide = np.array([(nxt() % (1 << 60)) - (1 << 59) for _ in range(n)], dtype=np.int64)
wide_t = np.array([(nxt() % (1 << 60)) - (1 << 59) for _ in range(700)], dtype=np.int64)
h = hashlib.sha256()
allmatch = True
for (a, t) in ((small, small_t), (wide, wide_t)):
    for inv in (False, True):
        r = np.asarray(fnp.isin(a, t, invert=inv))
        e = np.isin(a, t, invert=inv)
        if r.shape != e.shape or r.dtype != e.dtype or r.tobytes() != e.tobytes():
            allmatch = False
        h.update(r.tobytes())
print(allmatch)
print(h.hexdigest())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    assert_eq!(
        lines.next().unwrap_or("").trim(),
        "True",
        "isin int table/hash paths must be byte-identical to numpy.isin"
    );
    assert_eq!(
        lines.next().unwrap_or("").trim(),
        "eba11167d3d5a8caeacb2d36cfd75cd01615590af5ffc635dba3b586e23bc9a1",
        "isin int table/hash golden sha256 drifted"
    );
    Ok(())
}

// ────────────────────────────────────────���────────────────────────────────────
// frombuffer
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn frombuffer_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
buf = b'\x01\x02\x03\x04'
result = fnp.frombuffer(buf, dtype='uint8')
expected = np.frombuffer(buf, dtype='uint8')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "frombuffer basic should match numpy");
    Ok(())
}

#[test]
fn frombuffer_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
import struct
buf = struct.pack('4f', 1.0, 2.0, 3.0, 4.0)
result = fnp.frombuffer(buf, dtype='float32')
expected = np.frombuffer(buf, dtype='float32')
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "frombuffer float should match numpy");
    Ok(())
}

#[test]
fn frombuffer_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
buf = b'\x01\x02\x03\x04\x05\x06'
result = fnp.frombuffer(buf, dtype='uint8', count=3)
expected = np.frombuffer(buf, dtype='uint8', count=3)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "frombuffer count should match numpy");
    Ok(())
}

#[test]
fn frombuffer_offset() -> Result<(), String> {
    let script = fnp_script(
        r#"
buf = b'\x01\x02\x03\x04\x05\x06'
result = fnp.frombuffer(buf, dtype='uint8', offset=2)
expected = np.frombuffer(buf, dtype='uint8', offset=2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "frombuffer offset should match numpy"
    );
    Ok(())
}

fn assert_transfer_array_matches_numpy(
    numpy: &Bound<'_, PyModule>,
    ours: &Bound<'_, PyAny>,
    expected: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<()> {
    let values_match: bool = numpy
        .getattr("array_equal")?
        .call1((ours, expected))?
        .extract()?;
    assert!(values_match, "{label} values should match NumPy");

    let ours_dtype = ours.getattr("dtype")?.getattr("str")?.extract::<String>()?;
    let expected_dtype = expected
        .getattr("dtype")?
        .getattr("str")?
        .extract::<String>()?;
    assert_eq!(
        ours_dtype, expected_dtype,
        "{label} dtype descriptor should match NumPy"
    );

    let ours_bytes = ours.call_method0("tobytes")?.extract::<Vec<u8>>()?;
    let expected_bytes = expected.call_method0("tobytes")?.extract::<Vec<u8>>()?;
    assert_eq!(
        ours_bytes, expected_bytes,
        "{label} transfer bytes should match NumPy"
    );
    Ok(())
}

#[test]
fn frombuffer_fixed_width_string_and_unicode_transfer_edges_match_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let frombuffer = module.getattr("frombuffer")?;
        let numpy_frombuffer = numpy.getattr("frombuffer")?;
        let bytes = py.import("builtins")?.getattr("bytes")?;

        let byte_string_buffer = bytes.call1((vec![b'a', 0, b'z', b'b', b'c', 0, b'X', b'Y'],))?;
        let byte_string_kwargs = PyDict::new(py);
        byte_string_kwargs.set_item("dtype", "S3")?;
        byte_string_kwargs.set_item("count", 2_i64)?;
        let ours_s = frombuffer.call((byte_string_buffer.clone(),), Some(&byte_string_kwargs))?;
        let expected_s =
            numpy_frombuffer.call((byte_string_buffer.clone(),), Some(&byte_string_kwargs))?;
        assert_transfer_array_matches_numpy(&numpy, &ours_s, &expected_s, "S3 count-limited")?;

        let offset_buffer = bytes.call1((vec![b'_', b'_', b'a', b'b', b'c', b'd'],))?;
        let offset_kwargs = PyDict::new(py);
        offset_kwargs.set_item("dtype", "S2")?;
        offset_kwargs.set_item("offset", 2_i64)?;
        let ours_offset = frombuffer.call((offset_buffer.clone(),), Some(&offset_kwargs))?;
        let expected_offset =
            numpy_frombuffer.call((offset_buffer.clone(),), Some(&offset_kwargs))?;
        assert_transfer_array_matches_numpy(&numpy, &ours_offset, &expected_offset, "S2 offset")?;

        let unicode_values = PyList::new(py, ["A\u{00df}", "\u{732b}\u{72ac}"])?;
        let unicode_source_kwargs = PyDict::new(py);
        unicode_source_kwargs.set_item("dtype", "<U2")?;
        let unicode_source = numpy
            .getattr("array")?
            .call((unicode_values,), Some(&unicode_source_kwargs))?;
        let unicode_buffer = unicode_source.call_method0("tobytes")?;
        let unicode_kwargs = PyDict::new(py);
        unicode_kwargs.set_item("dtype", "<U2")?;
        unicode_kwargs.set_item("count", 2_i64)?;
        let ours_u = frombuffer.call((unicode_buffer.clone(),), Some(&unicode_kwargs))?;
        let expected_u = numpy_frombuffer.call((unicode_buffer.clone(),), Some(&unicode_kwargs))?;
        assert_transfer_array_matches_numpy(&numpy, &ours_u, &expected_u, "<U2 transfer")?;

        let bad_unicode_buffer = bytes.call1((vec![0_u8; 7],))?;
        let ours_bad = frombuffer.call((bad_unicode_buffer.clone(),), Some(&unicode_kwargs));
        let expected_bad = numpy_frombuffer.call((bad_unicode_buffer,), Some(&unicode_kwargs));
        assert!(
            ours_bad.is_err() && expected_bad.is_err(),
            "misaligned Unicode buffer should raise like NumPy"
        );

        Ok(())
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// fromiter
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fromiter_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
iterable = (x*x for x in range(5))
result = fnp.fromiter(iterable, dtype='int64', count=5)
expected = np.fromiter((x*x for x in range(5)), dtype='int64', count=5)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fromiter basic should match numpy");
    Ok(())
}

#[test]
fn fromiter_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
iterable = (x/2 for x in range(5))
result = fnp.fromiter(iterable, dtype='float64', count=5)
expected = np.fromiter((x/2 for x in range(5)), dtype='float64', count=5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fromiter float should match numpy");
    Ok(())
}

#[test]
fn fromiter_list() -> Result<(), String> {
    let script = fnp_script(
        r#"
iterable = [1, 2, 3, 4, 5]
result = fnp.fromiter(iterable, dtype='int64')
expected = np.fromiter(iterable, dtype='int64')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fromiter list should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isin_matches_membership() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
test = [2, 4]
result = fnp.isin(a, test)
# Manual check: positions 1 and 3 should be True
print(result[1] == True and result[3] == True and result[0] == False)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isin should correctly identify membership"
    );
    Ok(())
}

#[test]
fn frombuffer_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
original = np.array([1, 2, 3, 4], dtype='int32')
buf = original.tobytes()
result = fnp.frombuffer(buf, dtype='int32')
print(np.array_equal(result, original))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "frombuffer roundtrip should preserve values"
    );
    Ok(())
}
