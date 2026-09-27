//! Conformance tests for numpy ediff1d against NumPy oracle.

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
// ediff1d basic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ediff1d_1d_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 0])
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d 1d basic should match numpy");
    Ok(())
}

#[test]
fn ediff1d_2d_flattens() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d 2d should flatten then diff");
    Ok(())
}

#[test]
fn ediff1d_3d_flattens() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d 3d should flatten then diff");
    Ok(())
}

#[test]
fn ediff1d_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 4.0, 7.5])
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ediff1d with to_begin / to_end
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ediff1d_with_to_begin_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 0])
result = fnp.ediff1d(a, to_begin=-99)
expected = np.ediff1d(a, to_begin=-99)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d with to_begin scalar should match numpy"
    );
    Ok(())
}

#[test]
fn ediff1d_with_to_end_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 0])
result = fnp.ediff1d(a, to_end=88)
expected = np.ediff1d(a, to_end=88)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d with to_end scalar should match numpy"
    );
    Ok(())
}

#[test]
fn ediff1d_with_to_begin_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 0])
result = fnp.ediff1d(a, to_begin=np.array([-99, -88]))
expected = np.ediff1d(a, to_begin=np.array([-99, -88]))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d with to_begin array should match numpy"
    );
    Ok(())
}

#[test]
fn ediff1d_with_to_end_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 0])
result = fnp.ediff1d(a, to_end=np.array([88, 99]))
expected = np.ediff1d(a, to_end=np.array([88, 99]))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d with to_end array should match numpy"
    );
    Ok(())
}

#[test]
fn ediff1d_with_both() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 0])
result = fnp.ediff1d(a, to_begin=-99, to_end=np.array([88, 99]))
expected = np.ediff1d(a, to_begin=-99, to_end=np.array([88, 99]))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d with both to_begin and to_end should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ediff1d edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ediff1d_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5])
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(len(result) == 0 and len(expected) == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d single element returns empty array"
    );
    Ok(())
}

#[test]
fn ediff1d_empty() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([])
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(len(result) == 0 and len(expected) == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d empty array returns empty array"
    );
    Ok(())
}

#[test]
fn ediff1d_preserves_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7], dtype='int32')
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d should preserve dtype");
    Ok(())
}

#[test]
fn ediff1d_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+2j, 3+4j, 6+1j])
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d complex should match numpy");
    Ok(())
}

#[test]
fn ediff1d_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.inf, 3.0, np.nan, 5.0])
fnp_result = fnp.ediff1d(a)
np_result = np.ediff1d(a)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d special values should match numpy"
    );
    Ok(())
}

#[test]
fn ediff1d_constant_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5.0, 5.0, 5.0, 5.0])
fnp_result = fnp.ediff1d(a)
np_result = np.ediff1d(a)
# Diff of constant should be zero
print(np.allclose(fnp_result, np_result) and np.allclose(fnp_result, 0.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d constant array should match numpy"
    );
    Ok(())
}

/// numpy's test_ediff1d through the drop-in harness (bead rc0923 .8): an EMPTY float64 input
/// with to_begin / to_end raised PanicException ("range start index 1 out of range for slice
/// of length 0"). a7a21e76 had fused the hazard check into the kernel without the
/// empty-input guard its typed sibling has. Every dtype x {empty, 1, 2, 3 elements} x
/// {none, to_begin, to_end, both empty}; values, dtype and raise type must be numpy's. A
/// PanicException is a BaseException, so it is caught as one.
#[test]
fn ediff1d_empty_and_short_inputs_with_to_begin_to_end_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def outcome(f):
    try:
        r = np.asarray(f())
        return ("ok", r.dtype.str, r.tobytes())
    except BaseException as ex:
        return (type(ex).__name__, str(ex)[:80])
cells = 0
bad = []
for dt in (np.float64, np.float32, np.int64, np.int32, np.int8, np.uint8, np.float16, np.complex128, bool):
    for data in ([], [1], [1, 2], [3, 1, 4]):
        for kw in ({}, {"to_begin": [0]}, {"to_end": [9]}, {"to_begin": [], "to_end": []}):
            cells += 1
            a = np.array(data, dtype=dt)
            ours, theirs = outcome(lambda: fnp.ediff1d(a, **kw)), outcome(lambda: np.ediff1d(a, **kw))
            if ours != theirs:
                bad.append(f"{np.dtype(dt).name} {data} {kw}: fnp={ours} numpy={theirs}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(2, ' ');
    assert_eq!(
        fields.next().unwrap_or("0"),
        "144",
        "cell table drifted: {result}"
    );
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "ediff1d must match numpy on empty and short inputs: {result}"
    );
    Ok(())
}

/// The f64 route copies `to_begin` / `to_end` ONCE, straight from a `copy=False` float64 view
/// into the output (it had copied a large `to_end` three times through a Vec, 2.6x slower than
/// numpy). A view is only a view of the right values when the operand is already contiguous
/// native float64, so the grid holds every operand shape that must be cast or copied first:
/// a byte-swapped `>f8` array (raw bytes read as native would be garbage), a strided view and
/// an F-ordered 2-D array (read through the base buffer they would come out in the wrong
/// order), narrow / unsigned / bool / float16 arrays, a 0-d array, Python scalars and lists,
/// and the input array itself. Values, dtype and raise type must be numpy's.
#[test]
fn ediff1d_f64_to_begin_to_end_operand_grid_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def outcome(f):
    try:
        r = np.asarray(f())
        return ("ok", r.dtype.str, r.shape, r.tobytes())
    except BaseException as ex:
        return (type(ex).__name__, str(ex)[:80])
rng = np.random.default_rng(7)
cells = 0
bad = []
for n in (5, 3000):
    a = rng.standard_normal(n)
    big = rng.standard_normal(4096)
    operands = {
        "int": 3, "float": -2.5, "bools": [True, False], "list": [1, 2.5, -3],
        "i8": np.array([-128, 127], dtype=np.int8),
        "u64": np.array([0, 2**64 - 1], dtype=np.uint64),
        "f16": np.array([1.5, -0.0], dtype=np.float16),
        "f32": rng.standard_normal(7).astype(np.float32),
        "f64 big": big,
        "f64 >f8": big[:9].astype(">f8"),
        "f64 strided": big[::3],
        "f64 2-D F": np.asfortranarray(big[:12].reshape(3, 4)),
        "0-d": np.array(4.25),
        "empty": np.array([]),
        "self": a,
    }
    for name, value in operands.items():
        for kw in ({"to_end": value}, {"to_begin": value}, {"to_begin": value, "to_end": big[:3]}):
            cells += 1
            ours = outcome(lambda: fnp.ediff1d(a, **kw))
            theirs = outcome(lambda: np.ediff1d(a, **kw))
            if ours != theirs:
                bad.append(f"n={n} {name} {sorted(kw)}: fnp={ours[:3]} numpy={theirs[:3]}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(2, ' ');
    assert_eq!(
        fields.next().unwrap_or("0"),
        "90",
        "cell table drifted: {result}"
    );
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "ediff1d to_begin/to_end operands must match numpy: {result}"
    );
    Ok(())
}
