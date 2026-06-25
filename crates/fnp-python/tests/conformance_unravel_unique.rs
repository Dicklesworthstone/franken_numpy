//! Conformance tests for numpy index conversion and unique_* functions against NumPy oracle.
//!
//! Tests unravel_index, ravel_multi_index, unique_all, unique_counts, unique_inverse, unique_values.

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
fn ravel_unravel_python_container_and_keyword_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def clean(value):
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value

def normalize(value):
    if isinstance(value, tuple):
        return ("tuple", tuple(normalize(item) for item in value))
    array = np.asarray(value)
    return (
        "array",
        str(array.dtype),
        tuple(array.shape),
        clean(array.tolist()),
    )

def outcome(call_fn, *args, **kwargs):
    try:
        return ("ok", normalize(call_fn(*args, **kwargs)))
    except Exception as exc:
        return ("err", type(exc).__name__)

cases = [
    (
        "unravel list indices list dims C",
        "unravel_index",
        lambda: (([3, 4, 5], [2, 3]), {}),
    ),
    (
        "unravel ndarray indices Fortran",
        "unravel_index",
        lambda: ((np.array([[0, 5], [6, 11]], dtype=np.int64), (3, 4)), {"order": "F"}),
    ),
    ("unravel scalar ndarray", "unravel_index", lambda: ((np.array(5), (2, 3)), {})),
    ("unravel invalid order error", "unravel_index", lambda: ((5, (2, 3)), {"order": "A"})),
    ("unravel out of bounds error", "unravel_index", lambda: ((6, (2, 3)), {})),
    (
        "ravel tuple coords list dims",
        "ravel_multi_index",
        lambda: ((([0, 1, 2], [1, 2, 3]), [3, 4]), {}),
    ),
    (
        "ravel ndarray coords mode sequence",
        "ravel_multi_index",
        lambda: ((np.array([[-1, 2, 3], [0, 5, -1]], dtype=np.int64), (3, 4)), {"mode": ("wrap", "clip")}),
    ),
    (
        "ravel list coords Fortran order",
        "ravel_multi_index",
        lambda: ((([0, 1, 2], [1, 2, 3]), (3, 4)), {"order": "F"}),
    ),
    ("ravel invalid mode error", "ravel_multi_index", lambda: (((0, 1), (3, 4)), {"mode": "middle"})),
    ("ravel coord out of bounds error", "ravel_multi_index", lambda: (((3, 1), (3, 4)), {})),
]

ok = True
for label, name, factory in cases:
    args, kwargs = factory()
    actual = outcome(getattr(fnp, name), *args, **kwargs)
    args, kwargs = factory()
    expected = outcome(getattr(np, name), *args, **kwargs)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ravel/unravel Python-container and keyword surfaces should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// unravel_index
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unravel_index_scalar_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.unravel_index(22, (7, 6))
expected = np.unravel_index(22, (7, 6))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unravel_index scalar 2d should match numpy"
    );
    Ok(())
}

#[test]
fn unravel_index_scalar_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.unravel_index(41, (3, 4, 5))
expected = np.unravel_index(41, (3, 4, 5))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unravel_index scalar 3d should match numpy"
    );
    Ok(())
}

#[test]
fn unravel_index_array_indices() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.unravel_index([22, 33, 41], (7, 6))
expected = np.unravel_index([22, 33, 41], (7, 6))
match = all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unravel_index array indices should match numpy"
    );
    Ok(())
}

#[test]
fn unravel_index_fortran_order() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.unravel_index(22, (7, 6), order='F')
expected = np.unravel_index(22, (7, 6), order='F')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unravel_index fortran order should match numpy"
    );
    Ok(())
}

#[test]
fn unravel_index_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.unravel_index(0, (3, 4, 5))
expected = np.unravel_index(0, (3, 4, 5))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unravel_index zero should match numpy"
    );
    Ok(())
}

#[test]
fn unravel_index_last_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
shape = (3, 4, 5)
last_idx = 3 * 4 * 5 - 1
result = fnp.unravel_index(last_idx, shape)
expected = np.unravel_index(last_idx, shape)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unravel_index last element should match numpy"
    );
    Ok(())
}

#[test]
fn unravel_index_1d_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.unravel_index(5, (10,))
expected = np.unravel_index(5, (10,))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unravel_index 1d shape should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ravel_multi_index
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ravel_multi_index_basic_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.ravel_multi_index((3, 5), (7, 6))
expected = np.ravel_multi_index((3, 5), (7, 6))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ravel_multi_index basic 2d should match numpy"
    );
    Ok(())
}

#[test]
fn ravel_multi_index_basic_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.ravel_multi_index((2, 1, 3), (3, 4, 5))
expected = np.ravel_multi_index((2, 1, 3), (3, 4, 5))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ravel_multi_index basic 3d should match numpy"
    );
    Ok(())
}

#[test]
fn ravel_multi_index_array_coords() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.ravel_multi_index(([3, 5, 1], [2, 4, 0]), (7, 6))
expected = np.ravel_multi_index(([3, 5, 1], [2, 4, 0]), (7, 6))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ravel_multi_index array coords should match numpy"
    );
    Ok(())
}

#[test]
fn ravel_multi_index_fortran_order() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.ravel_multi_index((3, 5), (7, 6), order='F')
expected = np.ravel_multi_index((3, 5), (7, 6), order='F')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ravel_multi_index fortran order should match numpy"
    );
    Ok(())
}

#[test]
fn ravel_multi_index_zero_coords() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.ravel_multi_index((0, 0, 0), (3, 4, 5))
expected = np.ravel_multi_index((0, 0, 0), (3, 4, 5))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ravel_multi_index zero coords should match numpy"
    );
    Ok(())
}

#[test]
fn ravel_multi_index_last_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
shape = (3, 4, 5)
coords = (2, 3, 4)
result = fnp.ravel_multi_index(coords, shape)
expected = np.ravel_multi_index(coords, shape)
print(result == expected == 3 * 4 * 5 - 1)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ravel_multi_index last element should match numpy"
    );
    Ok(())
}

#[test]
fn ravel_multi_index_mode_clip() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.ravel_multi_index((10, 2), (7, 6), mode='clip')
expected = np.ravel_multi_index((10, 2), (7, 6), mode='clip')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ravel_multi_index mode clip should match numpy"
    );
    Ok(())
}

#[test]
fn ravel_multi_index_mode_wrap() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.ravel_multi_index((8, 2), (7, 6), mode='wrap')
expected = np.ravel_multi_index((8, 2), (7, 6), mode='wrap')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ravel_multi_index mode wrap should match numpy"
    );
    Ok(())
}

#[test]
fn ravel_unravel_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
shape = (5, 6, 7)
coords = (2, 3, 4)
flat = fnp.ravel_multi_index(coords, shape)
back = fnp.unravel_index(flat, shape)
print(coords == back)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel and unravel should roundtrip");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// unique_all
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unique_all_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 1, 4, 3, 2])
result = fnp.unique_all(a)
expected = np.unique_all(a)
match = (np.array_equal(result.values, expected.values) and
         np.array_equal(result.indices, expected.indices) and
         np.array_equal(result.inverse_indices, expected.inverse_indices) and
         np.array_equal(result.counts, expected.counts))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique_all basic should match numpy");
    Ok(())
}

#[test]
fn unique_all_floats() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 1.5, 3.5, 2.5])
result = fnp.unique_all(a)
expected = np.unique_all(a)
match = (np.array_equal(result.values, expected.values) and
         np.array_equal(result.indices, expected.indices) and
         np.array_equal(result.inverse_indices, expected.inverse_indices) and
         np.array_equal(result.counts, expected.counts))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unique_all floats should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// unique_counts
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unique_counts_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 1, 4, 3, 2])
result = fnp.unique_counts(a)
expected = np.unique_counts(a)
match = (np.array_equal(result.values, expected.values) and
         np.array_equal(result.counts, expected.counts))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unique_counts basic should match numpy"
    );
    Ok(())
}

#[test]
fn unique_counts_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5, 5, 5, 5])
result = fnp.unique_counts(a)
expected = np.unique_counts(a)
match = (np.array_equal(result.values, expected.values) and
         np.array_equal(result.counts, expected.counts))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unique_counts single element should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// unique_inverse
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unique_inverse_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 1, 4, 3, 2])
result = fnp.unique_inverse(a)
expected = np.unique_inverse(a)
match = (np.array_equal(result.values, expected.values) and
         np.array_equal(result.inverse_indices, expected.inverse_indices))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unique_inverse basic should match numpy"
    );
    Ok(())
}

#[test]
fn unique_inverse_reconstruct() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
result = fnp.unique_inverse(a)
expected = np.unique_inverse(a)
reconstructed_fnp = result.values[result.inverse_indices]
reconstructed_np = expected.values[expected.inverse_indices]
print(np.array_equal(reconstructed_fnp, a) and np.array_equal(reconstructed_np, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unique_inverse should allow reconstruction"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// unique_values
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unique_values_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 1, 4, 3, 2])
result = fnp.unique_values(a)
expected = np.unique_values(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unique_values basic should match numpy"
    );
    Ok(())
}

#[test]
fn unique_values_matches_numpy_set_semantics() -> Result<(), String> {
    // numpy 2.x's unique_values follows the Array API standard, which only
    // requires set equality — not sorted output. fnp.unique_values must
    // match numpy's actual return order (whatever it is) and produce the
    // same multiset as numpy when sorted.
    let script = fnp_script(
        r#"
a = np.array([5, 3, 1, 4, 2])
result = fnp.unique_values(a)
expected = np.unique_values(a)
print(np.array_equal(result, expected) and
      np.array_equal(np.sort(result), np.sort(expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unique_values must match numpy's return order and set"
    );
    Ok(())
}

#[test]
fn unique_values_empty() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([], dtype=int)
result = fnp.unique_values(a)
expected = np.unique_values(a)
print(np.array_equal(result, expected) and len(result) == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unique_values empty should match numpy"
    );
    Ok(())
}

#[test]
fn unique_values_strings() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array(['b', 'a', 'c', 'a', 'b'])
result = fnp.unique_values(a)
expected = np.unique_values(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unique_values strings should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-function consistency
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unique_functions_consistent() -> Result<(), String> {
    // numpy 2.x's unique_all/unique_counts/unique_inverse return sorted
    // values, but unique_values follows the Array API spec and is allowed
    // to be unsorted. So the cross-function invariant is set equality on
    // values plus exact equality on counts/inverse_indices (which use the
    // canonical sorted ordering).
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 1, 4, 3, 2])
all_result = fnp.unique_all(a)
counts_result = fnp.unique_counts(a)
inverse_result = fnp.unique_inverse(a)
values_result = fnp.unique_values(a)
match = (np.array_equal(np.sort(all_result.values), np.sort(values_result)) and
         np.array_equal(all_result.values, counts_result.values) and
         np.array_equal(all_result.counts, counts_result.counts) and
         np.array_equal(all_result.values, inverse_result.values) and
         np.array_equal(all_result.inverse_indices, inverse_result.inverse_indices))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unique functions should be consistent"
    );
    Ok(())
}

#[test]
fn unique_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 1+1j, 3+2j], dtype=np.complex128)
fnp_result = fnp.unique(a)
np_result = np.unique(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique complex should match numpy");
    Ok(())
}

/// Locks the fast plain-`numpy.unique` path (the float64 no-index/inverse/counts
/// branch that reuses the parallel total_cmp sort + adjacent dedup) to bit-exact
/// parity with numpy. unique returns sorted distinct values verbatim, so parity
/// must hold at the IEEE-754 bit level. The inputs deliberately avoid arrays that
/// mix +0.0 and -0.0: which signed zero numpy keeps there is governed by its
/// unstable introsort (impl-defined, like the fmin/fmax position-dependence), and
/// our path preserves the prior fnp behavior rather than that unstable choice.
#[test]
fn unique_plain_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
# standard_normal never yields an exact +-0.0, so these are deterministic.
for n in [1000, 100003, 500000, 1048576]:
    chunks.append(np.asarray(mod.unique(rng.standard_normal(n))).tobytes())
    chunks.append(np.asarray(mod.unique(np.round(rng.standard_normal(n) * 4) + 0.5)).tobytes())
repeated = (np.arange(1048576, dtype=np.float64) * 37 % 65536) / 16.0
chunks.append(np.asarray(mod.unique(repeated)).tobytes())
chunks.append(np.asarray(mod.unique(np.array([np.inf, -np.inf, 1.0, -1.0, 1.0, np.nan, 2.0, np.nan], dtype=np.float64))).tobytes())
chunks.append(np.asarray(mod.unique(np.array([-0.0, 1.0, 2.0, -0.0, 3.0], dtype=np.float64))).tobytes())
chunks.append(np.asarray(mod.unique(np.full(5000, 2.5, dtype=np.float64))).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp").to_string()))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "fast plain unique must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}
