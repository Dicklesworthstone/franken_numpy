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

#[test]
fn f16_unique_presence_table_bit_exact() -> Result<(), String> {
    // f16 unique via the 65536-slot presence table: bit-bijective for inputs
    // with a single zero pattern and no NaNs; both-zeros/NaN inputs DEFER
    // (numpy's kept pattern for the equal-class is an introsort partition
    // artifact - flood-majority at scale, first-occurrence-looking on small
    // inputs - pinned across three numpy versions, ledger 2026-07-12).
    let script = fnp_script(
        r#"
verdicts = []
rng = np.random.default_rng(20260716)
for n in (200000, 2_000_003):
    a = (rng.standard_normal(n) * 2).astype(np.float16)
    r = fnp.unique(a); e = np.unique(a)
    if r.dtype != e.dtype or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL n={n}")
# single zero pattern (either sign alone), +-inf, denormals, full-range bits
a = (rng.standard_normal(300000)).astype(np.float16)
a[a == 0] = np.float16(1.0)
a[0] = np.float16(-0.0); a[1] = np.float16(np.inf); a[2] = np.float16(-np.inf)
a[3] = np.uint16(0x0001).view(np.float16); a[4] = np.uint16(0x8001).view(np.float16)
r = fnp.unique(a); e = np.unique(a)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL neg-zero-only/specials")
# degenerate full-bit-range coverage via a uint16 ramp viewed as f16 (nan-free slice)
ramp = np.arange(0, 0x7C00, dtype=np.uint16).view(np.float16)
big = np.tile(ramp, 8)
r = fnp.unique(big); e = np.unique(big)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL positive ramp")
# ambiguity defers: both zeros / NaNs present -> byte-equal via numpy path
a2 = (rng.standard_normal(300000)).astype(np.float16)
a2[10] = np.float16(0.0); a2[11] = np.float16(-0.0)
if fnp.unique(a2).tobytes() != np.unique(a2).tobytes():
    verdicts.append("FAIL both-zeros defer")
a3 = (rng.standard_normal(300000)).astype(np.float16)
a3[10] = np.float16(np.nan)
if fnp.unique(a3).tobytes() != np.unique(a3).tobytes():
    verdicts.append("FAIL nan defer")
# below-gate + kwargs variants stay on numpy
sm = (rng.standard_normal(1000)).astype(np.float16)
if fnp.unique(sm).tobytes() != np.unique(sm).tobytes():
    verdicts.append("FAIL below-gate")
ri_f, ri_n = fnp.unique(sm, return_counts=True), np.unique(sm, return_counts=True)
if ri_f[0].tobytes() != ri_n[0].tobytes() or ri_f[1].tobytes() != ri_n[1].tobytes():
    verdicts.append("FAIL return_counts defer")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "f16 unique presence table must be bit-identical (with ambiguity defers): {result}"
    );
    Ok(())
}

#[test]
fn f16_isin_bitmap_bit_exact() -> Result<(), String> {
    // f16 isin via the 65536-slot presence bitmap (the unique-table lever
    // applied to membership). Pinned semantics: NaN NEVER matches (even the
    // identical bit pattern); +-0 match each other; invert flips.
    let script = fnp_script(
        r#"
verdicts = []
rng = np.random.default_rng(20260717)
for n, m in ((100000, 100), (2_000_003, 5000)):
    a = (rng.standard_normal(n) * 2).astype(np.float16)
    t = (rng.standard_normal(m) * 2).astype(np.float16)
    for inv in (False, True):
        r = fnp.isin(a, t, invert=inv); e = np.isin(a, t, invert=inv)
        if r.dtype != e.dtype or r.tobytes() != e.tobytes():
            verdicts.append(f"FAIL n={n} inv={inv}")
# semantics battery: nan (both patterns), +-0 cross, inf
a = np.array([np.nan, 0.0, -0.0, np.inf, 1.5, -1.5], dtype=np.float16)
a = np.tile(a, 20000)  # over the gate
t = np.array([np.nan, -0.0, np.inf], dtype=np.float16)
r = fnp.isin(a, t); e = np.isin(a, t)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL semantics battery")
t2 = np.array([0xfe00], dtype=np.uint16).view(np.float16)
r = fnp.isin(a, t2); e = np.isin(a, t2)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL neg-nan test set")
# 2-D element shape preserved; below-gate defers
m2 = (rng.standard_normal((300, 400))).astype(np.float16)
t3 = m2.ravel()[:50]
r = fnp.isin(m2, t3); e = np.isin(m2, t3)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL 2-D shape")
sm = (rng.standard_normal(100)).astype(np.float16)
if fnp.isin(sm, sm[:5]).tobytes() != np.isin(sm, sm[:5]).tobytes():
    verdicts.append("FAIL below-gate")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "f16 isin bitmap must be bit-identical incl nan/signed-zero semantics: {result}"
    );
    Ok(())
}

#[test]
fn wide_int_setops_bit_exact_match_numpy() -> Result<(), String> {
    // Wide (4/8-byte) int set ops via par_sort+dedup+merge: sorted unique
    // integer outputs are value-deterministic -> byte-exact across the FULL
    // dtype range (incl. > 2^53, where the old extract-precise route raised
    // and fell back to numpy's multi-second sort path).
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(257)
verdicts = []
for dt in [np.int64, np.uint64, np.int32, np.uint32]:
    info = np.iinfo(dt)
    a = rng.integers(info.min, info.max, 400_000, dtype=dt, endpoint=True)
    b = rng.integers(info.min, info.max, 400_000, dtype=dt, endpoint=True)
    b[:5000] = a[:5000]  # guaranteed overlap
    for fname in ("intersect1d", "union1d", "setdiff1d", "setxor1d"):
        ff = getattr(fnp, fname); nf = getattr(np, fname)
        r = ff(a, b); e = nf(a, b)
        if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
            verdicts.append(f"FAIL {fname} {dt.__name__}")
# duplicates-heavy, disjoint, empty, N-D ravel
d1 = rng.integers(0, 100, 300_000)
d2 = rng.integers(50, 150, 300_000)
for fname in ("intersect1d", "union1d", "setdiff1d", "setxor1d"):
    ff = getattr(fnp, fname); nf = getattr(np, fname)
    if ff(d1, d2).tobytes() != nf(d1, d2).tobytes():
        verdicts.append(f"FAIL dup-heavy {fname}")
lo = rng.integers(-2**62, -2**61, 200_000)
hi = rng.integers(2**61, 2**62, 200_000)
if fnp.intersect1d(lo, hi).tobytes() != np.intersect1d(lo, hi).tobytes():
    verdicts.append("FAIL disjoint")
e_ = np.array([], dtype=np.int64)
big = rng.integers(-2**62, 2**62, 200_000)
if fnp.union1d(e_, big).tobytes() != np.union1d(e_, big).tobytes():
    verdicts.append("FAIL empty operand")
M2 = rng.integers(-2**62, 2**62, (600, 500))
if fnp.intersect1d(M2, big).tobytes() != np.intersect1d(M2, big).tobytes():
    verdicts.append("FAIL N-D ravel")
# assume_unique keeps the delegate
au1 = np.unique(rng.integers(-2**62, 2**62, 200_000))
au2 = np.unique(rng.integers(-2**62, 2**62, 200_000))
if fnp.intersect1d(au1, au2, assume_unique=True).tobytes() != np.intersect1d(au1, au2, assume_unique=True).tobytes():
    verdicts.append("FAIL assume_unique delegate")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W1 = rng.integers(-2**62, 2**62, 8_000_000)
W2 = rng.integers(-2**62, 2**62, 8_000_000)
W2[:100_000] = W1[:100_000]
tn = best(lambda: np.intersect1d(W1, W2)); tf = best(lambda: fnp.intersect1d(W1, W2))
print(f"INTERSECT_INT64_WIDE_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn = best(lambda: np.setdiff1d(W1, W2)); tf = best(lambda: fnp.setdiff1d(W1, W2))
print(f"SETDIFF_INT64_WIDE_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces INTERSECT/UNION_INT64_WIDE_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "wide int setops must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn wide_int_unique_bit_exact_matches_numpy() -> Result<(), String> {
    // Wide-range 4/8-byte int unique PARITY coverage. A native par_sort+dedup
    // arm was gate-REJECTED at 1.056x: numpy 2.4's hash-table integer unique
    // runs 8M wide int64 in ~96ms (numpy 2.3.5 on hz2 took 4723ms via its
    // sort path - a 49x VERSION divergence). The rows pin byte parity of the
    // existing delegate/native routing across the full dtype range.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(269)
verdicts = []
for dt in [np.int64, np.uint64, np.int32, np.uint32]:
    info = np.iinfo(dt)
    a = rng.integers(info.min, info.max, 500_000, dtype=dt, endpoint=True)
    r = fnp.unique(a); e = np.unique(a)
    if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL {dt.__name__}")
# dup-heavy wide values, all-same, sorted input, N-D ravel
d = rng.integers(-2**62, 2**62, 50).repeat(20_000)
if fnp.unique(d).tobytes() != np.unique(d).tobytes():
    verdicts.append("FAIL dup-heavy")
s = np.full(400_000, 2**60, dtype=np.int64)
if fnp.unique(s).tobytes() != np.unique(s).tobytes():
    verdicts.append("FAIL all-same")
srt = np.sort(rng.integers(-2**62, 2**62, 400_000))
if fnp.unique(srt).tobytes() != np.unique(srt).tobytes():
    verdicts.append("FAIL pre-sorted")
M = rng.integers(-2**62, 2**62, (800, 600))
if fnp.unique(M).tobytes() != np.unique(M).tobytes():
    verdicts.append("FAIL N-D ravel")
# kwargs forms keep prior behavior (byte parity via delegate/native)
a = rng.integers(-2**62, 2**62, 400_000)
rv, rc = fnp.unique(a, return_counts=True)
ev, ec = np.unique(a, return_counts=True)
if rv.tobytes() != ev.tobytes() or np.asarray(rc).tobytes() != np.asarray(ec).tobytes():
    verdicts.append("FAIL return_counts parity")
# small-range int64 keeps the counting arm (byte parity)
sm = rng.integers(0, 1000, 2_000_000)
if fnp.unique(sm).tobytes() != np.unique(sm).tobytes():
    verdicts.append("FAIL small-range parity")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.integers(-2**62, 2**62, 8_000_000)
if fnp.unique(W).tobytes() != np.unique(W).tobytes():
    verdicts.append("FAIL 8M wide parity")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces UNIQUE_INT64_WIDE_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "wide int unique must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
#[ignore = "numpy-baseline probe for lever sizing; run explicitly"]
fn probe_gate_worker_sort_class_bases() -> Result<(), String> {
    let script = fnp_script(
        r#"
import time
def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)
rng = np.random.default_rng(271)
a = rng.integers(-2**62, 2**62, 8_000_000)
print(f"BASE argsort_stable_i64_8m {best(lambda: np.argsort(a, kind='stable')):.1f}")
print(f"BASE argsort_default_i64_8m {best(lambda: np.argsort(a)):.1f}")
k1 = rng.integers(-2**62, 2**62, 4_000_000)
k2 = rng.integers(-2**62, 2**62, 4_000_000)
print(f"BASE lexsort_2xi64_4m {best(lambda: np.lexsort((k1, k2))):.1f}")
f = rng.standard_normal(8_000_000)
print(f"BASE argsort_stable_f64_8m {best(lambda: np.argsort(f, kind='stable')):.1f}")
# ufunc.at bases (add.at lever pricing, 2026-07-13): scatter-add with duplicate
# indices, histogram-style small target and large-target forms, int + f64.
idx_small = rng.integers(0, 1024, 8_000_000)
idx_large = rng.integers(0, 8_000_000, 8_000_000)
vi = rng.integers(-1000, 1000, 8_000_000)
vf = rng.standard_normal(8_000_000)
ti = np.zeros(1024, dtype=np.int64)
tl = np.zeros(8_000_000, dtype=np.int64)
tf = np.zeros(1024)
print(f"BASE add_at_i64_hist1k_8m {best(lambda: np.add.at(ti, idx_small, vi)):.1f}")
print(f"BASE add_at_i64_large_8m {best(lambda: np.add.at(tl, idx_large, vi)):.1f}")
print(f"BASE add_at_f64_hist1k_8m {best(lambda: np.add.at(tf, idx_small, vf)):.1f}")
print(f"BASE bincount_weights_ref {best(lambda: np.bincount(idx_small, weights=vf, minlength=1024)):.1f}")
print("numpy", np.__version__)
print(True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}");
    Ok(())
}

#[test]
fn wide2_lexsort_bit_exact_matches_numpy() -> Result<(), String> {
    // Two wide-range int keys (composite span overflow) sort as
    // (primary, secondary, index) i128 tuples - tuple order IS numpy's
    // stable lexsort contract, deterministic incl. ties -> byte-exact.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(277)
verdicts = []
k1 = rng.integers(-2**62, 2**62, 500_000)
k2 = rng.integers(-2**62, 2**62, 500_000)
if fnp.lexsort((k1, k2)).tobytes() != np.lexsort((k1, k2)).tobytes():
    verdicts.append("FAIL wide i64 pair")
# ties in the primary key exercise the secondary + stable index
kp = rng.integers(0, 50, 500_000) * 2**58
ks = rng.integers(-2**62, 2**62, 500_000)
if fnp.lexsort((ks, kp)).tobytes() != np.lexsort((ks, kp)).tobytes():
    verdicts.append("FAIL primary ties")
# ties in BOTH keys -> original-index stability
kd1 = rng.integers(0, 10, 500_000) * 2**58
kd2 = rng.integers(0, 10, 500_000) * 2**58
if fnp.lexsort((kd1, kd2)).tobytes() != np.lexsort((kd1, kd2)).tobytes():
    verdicts.append("FAIL double ties stability")
# mixed widths/signedness
m1 = rng.integers(0, 2**63, 400_000, dtype=np.uint64, endpoint=True)
m2 = rng.integers(-2**31, 2**31 - 1, 400_000).astype(np.int32)
if fnp.lexsort((m2, m1)).tobytes() != np.lexsort((m2, m1)).tobytes():
    verdicts.append("FAIL mixed u64/i32")
# 3 keys + small n keep prior behavior (byte parity)
k3 = rng.integers(-2**62, 2**62, 300_000)
if fnp.lexsort((k1[:300_000], k2[:300_000], k3)).tobytes() != np.lexsort((k1[:300_000], k2[:300_000], k3)).tobytes():
    verdicts.append("FAIL 3-key parity")
s1 = rng.integers(-2**62, 2**62, 1000)
s2 = rng.integers(-2**62, 2**62, 1000)
if fnp.lexsort((s1, s2)).tobytes() != np.lexsort((s1, s2)).tobytes():
    verdicts.append("FAIL small parity")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W1 = rng.integers(-2**62, 2**62, 4_000_000)
W2 = rng.integers(-2**62, 2**62, 4_000_000)
tn = best(lambda: np.lexsort((W1, W2))); tf = best(lambda: fnp.lexsort((W1, W2)))
print(f"LEXSORT_WIDE2_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces LEXSORT_WIDE2_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "wide 2-key lexsort must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn float_stable_argsort_nan_native_matches_numpy() -> Result<(), String> {
    // kind='stable' float argsort with NaNs present used to delegate wholesale
    // (radix AND typed fallback both deferred NaN). numpy's float lt is
    // `a < b || (b != b && a == a)`: every NaN — either sign bit, any payload —
    // orders after +inf, and NaN-vs-NaN is a tie broken by original index under
    // stable kind. Mapping ALL NaNs to one maximal radix key reproduces that
    // byte-exactly in the stable LSD radix; rows pin sign/payload mixes, dense
    // NaN ties, specials, f32, and the widened f16 route.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(283)
verdicts = []
n = 2_000_000
def ab(name, a, kind="stable"):
    if fnp.argsort(a, kind=kind).tobytes() != np.argsort(a, kind=kind).tobytes():
        verdicts.append(f"FAIL {name}")
# sparse NaNs of BOTH signs sprinkled into distinct data
f = rng.standard_normal(n)
pos = rng.choice(n, 200, replace=False)
f[pos[:100]] = np.nan
f[pos[100:]] = -np.nan
ab("sparse +-nan f64", f)
# differing NaN payloads must share one key (numpy treats all NaNs as one tie class)
fp = f.copy()
fp[pos[:50]] = np.array([0x7FF8_0000_0000_0001] * 50, dtype=np.uint64).view(np.float64)
fp[pos[150:]] = np.array([0xFFF0_0000_0000_0007] * 50, dtype=np.uint64).view(np.float64)
ab("nan payload mix f64", fp)
# dense ties + 10% NaN: stable index order among values AND among NaNs
d = rng.integers(0, 40, n).astype(np.float64)
d[rng.random(n) < 0.10] = np.nan
ab("dense ties 10pct nan f64", d)
# specials: +-inf, +-0.0, subnormals, +-nan
s = rng.standard_normal(n)
s[: n // 8] = np.inf
s[n // 8 : n // 4] = -np.inf
s[n // 4 : n // 2 : 2] = 0.0
s[n // 4 + 1 : n // 2 : 2] = -0.0
s[n // 2 : n // 2 + 1000] = 5e-324
s[-2000:-1000] = np.nan
s[-1000:] = -np.nan
ab("specials f64", s)
# f32 twin
g = rng.standard_normal(n).astype(np.float32)
g[pos[:100]] = np.float32(np.nan)
g[pos[100:]] = -np.float32(np.nan)
ab("sparse +-nan f32", g)
# f16 widened route
h = rng.standard_normal(500_000 * 3).astype(np.float16)
h[rng.random(1_500_000) < 0.05] = np.float16(np.nan)
ab("5pct nan f16", h)
# small n keeps prior behavior (below the native gate)
sm = rng.standard_normal(1000)
sm[7] = np.nan
ab("small parity", sm)
# default kind with NaN still defers -> byte parity
ab("default-kind nan parity", f, "quicksort")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.standard_normal(8_000_000)
W[rng.choice(8_000_000, 800, replace=False)] = np.nan
tn = best(lambda: np.argsort(W, kind="stable")); tf = best(lambda: fnp.argsort(W, kind="stable"))
print(f"ARGSORT_STABLE_F64_NAN_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces ARGSORT_STABLE_F64_NAN_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "NaN-bearing stable float argsort must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn wide34_lexsort_bit_exact_matches_numpy() -> Result<(), String> {
    // 3- and 4-key wide-range int lexsort (composite span-product overflow)
    // sort ([u64; K], index) tuples with each key mapped to an
    // order-preserving u64 - tuple order IS numpy's stable lexsort contract,
    // deterministic incl. ties -> byte-exact. 5+ keys keep prior routing.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(281)
verdicts = []
def w(n):
    return rng.integers(-2**62, 2**62, n)
k1, k2, k3 = w(500_000), w(500_000), w(500_000)
if fnp.lexsort((k1, k2, k3)).tobytes() != np.lexsort((k1, k2, k3)).tobytes():
    verdicts.append("FAIL wide i64 triple")
# dense primary + dense secondary ties exercise tertiary ordering + key priority
kp = rng.integers(0, 12, 500_000) * 2**58
ks = rng.integers(0, 12, 500_000) * 2**58
if fnp.lexsort((k1, ks, kp)).tobytes() != np.lexsort((k1, ks, kp)).tobytes():
    verdicts.append("FAIL dense primary+secondary")
# ties in ALL THREE keys -> original-index stability
d1 = rng.integers(0, 6, 500_000) * 2**58
d2 = rng.integers(0, 6, 500_000) * 2**58
d3 = rng.integers(0, 6, 500_000) * 2**58
if fnp.lexsort((d1, d2, d3)).tobytes() != np.lexsort((d1, d2, d3)).tobytes():
    verdicts.append("FAIL triple ties stability")
# mixed widths/signedness incl. u64 above 2**63 as PRIMARY
m1 = rng.integers(0, 2**63, 400_000, dtype=np.uint64, endpoint=True) * 2
m2 = rng.integers(-2**31, 2**31 - 1, 400_000).astype(np.int32)
m3 = w(400_000)
if fnp.lexsort((m2, m3, m1)).tobytes() != np.lexsort((m2, m3, m1)).tobytes():
    verdicts.append("FAIL mixed u64/i64/i32")
# 4 keys wide + a tied layer
q4 = rng.integers(0, 8, 400_000) * 2**58
if fnp.lexsort((k1[:400_000], q4, k2[:400_000], k3[:400_000])).tobytes() != np.lexsort((k1[:400_000], q4, k2[:400_000], k3[:400_000])).tobytes():
    verdicts.append("FAIL 4-key")
# 5 keys + small n keep prior behavior (byte parity)
f5 = [w(300_000) for _ in range(5)]
if fnp.lexsort(tuple(f5)).tobytes() != np.lexsort(tuple(f5)).tobytes():
    verdicts.append("FAIL 5-key parity")
s = [w(1000) for _ in range(3)]
if fnp.lexsort(tuple(s)).tobytes() != np.lexsort(tuple(s)).tobytes():
    verdicts.append("FAIL small parity")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W1, W2, W3 = w(4_000_000), w(4_000_000), w(4_000_000)
tn = best(lambda: np.lexsort((W1, W2, W3))); tf = best(lambda: fnp.lexsort((W1, W2, W3)))
print(f"LEXSORT_WIDE3_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces LEXSORT_WIDE3_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "wide 3/4-key lexsort must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn wide_rows_lexsort_bit_exact_matches_numpy() -> Result<(), String> {
    // 2-D ROW-KEYS lexsort form: one (K, n) int array, rows are the keys,
    // last row primary. Same ([u64; K], index) tuple contract as the
    // sequence form -> byte-exact; F-contig/strided/5-row/axis=0/bool/small
    // keep prior routing (delegate parity).
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(293)
verdicts = []
def w(*shape):
    return rng.integers(-2**62, 2**62, shape)
def ab(name, m, **kw):
    if fnp.lexsort(m, **kw).tobytes() != np.lexsort(m, **kw).tobytes():
        verdicts.append(f"FAIL {name}")
ab("2-row wide i64", w(2, 600_000))
ab("3-row wide i64", w(3, 600_000))
# dense PRIMARY (last row) ties exercise lower-row ordering
mp = w(3, 600_000)
mp[2] = rng.integers(0, 10, 600_000) * 2**58
ab("dense primary row", mp)
# ALL rows dense -> original-index stability (small-span data through the arm)
ab("all dense rows", rng.integers(0, 7, (3, 600_000)) * 2**58)
# u64 above 2**63 + i32 dtypes
ab("u64 rows", rng.integers(0, 2**64, (2, 500_000), dtype=np.uint64))
ab("i32 rows", rng.integers(-2**31, 2**31 - 1, (3, 500_000)).astype(np.int32))
# layout/scope defers keep byte parity: F-contig (.T), strided view, 5 rows,
# axis=0, bool rows, small n
ab("F-contig transposed", w(400_000, 2).T)
ab("strided rows", w(2, 800_000)[:, ::2])
ab("5 rows", w(5, 300_000))
ab("axis=0", w(2, 1000), axis=0)
ab("bool rows", rng.integers(0, 2, (2, 400_000)).astype(bool))
ab("small", w(3, 1000))

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = w(3, 4_000_000)
tn = best(lambda: np.lexsort(W)); tf = best(lambda: fnp.lexsort(W))
print(f"LEXSORT_WIDEROWS3_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces LEXSORT_WIDEROWS3_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "wide 2-D row-keys lexsort must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn f64_sort_uniform_nan_native_matches_numpy() -> Result<(), String> {
    // NaN-bearing f64 np.sort parity coverage + gate-worker basis probe.
    // A native uniform-NaN arm (canonical np.nan -> f64_sortable_key total
    // order in all four value-sort kernels) was built and gate-REJECTED
    // 2026-07-13: numpy 2.4.6's AVX-512 f64 qsort saturates this basis -
    // flat 0.967x (92.9ms @8M), lastaxis 1.128x, axis0 1.011x - and the gate
    // also caught AVX-512 DIVERGING byte-wise on uniform NEGATIVE-sign NaN
    // (its NaN byte behavior is ISA/version-fragile; parity-by-defer is
    // robust by construction). All NaN-bearing rows here go through the
    // numpy delegate; the AB rows pin the measured bases for future re-eval.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(307)
verdicts = []
def ab(name, a, **kw):
    if fnp.sort(a, **kw).tobytes() != np.sort(a, **kw).tobytes():
        verdicts.append(f"FAIL {name}")
n = 2_000_000
# sparse canonical NaN in distinct data (flat)
f = rng.standard_normal(n)
f[rng.choice(n, 500, replace=False)] = np.nan
ab("flat sparse nan", f)
# dense ties + 10% NaN
d = rng.integers(0, 40, n).astype(np.float64)
d[rng.random(n) < 0.10] = np.nan
ab("flat dense ties nan", d)
# specials: +-inf, subnormals, single-sign zeros, canonical NaN
s = rng.standard_normal(n)
s[:1000] = np.inf; s[1000:2000] = -np.inf
s[2000:3000] = 5e-324; s[3000:4000] = 0.0
s[-1500:] = np.nan
ab("flat specials nan", s)
# uniform NEGATIVE-sign NaN defers (gate-measured AVX-512 divergence) -> parity
g = rng.standard_normal(n)
g[rng.choice(n, 300, replace=False)] = -np.nan
ab("flat uniform -nan parity", g)
ab("lastaxis -nan parity", g[: 1_999_000].reshape(1000, 1999), axis=-1)
# uniform NON-CANONICAL positive payload defers -> parity
q = rng.standard_normal(n)
q[rng.choice(n, 300, replace=False)] = np.array([0x7FF8_0000_0000_0001], dtype=np.uint64).view(np.float64)[0]
ab("lastaxis odd-payload parity", q[: 1_999_000].reshape(1000, 1999), axis=-1)
# MIXED NaN payloads -> defer, parity via numpy
p = f.copy()
p[7] = np.array([0x7FF8_0000_0000_0001], dtype=np.uint64).view(np.float64)[0]
ab("flat mixed payload parity", p)
# mixed-sign zeros WITH uniform NaN -> still defer, parity
z = f.copy(); z[11] = -0.0; z[13] = 0.0
ab("flat mixed zeros parity", z)
# lastaxis / axis0 / midaxis forms with NaN
m2 = rng.standard_normal((1500, 1500))
m2[rng.random((1500, 1500)) < 0.01] = np.nan
ab("lastaxis nan", m2, axis=-1)
ab("axis0 nan", m2, axis=0)
m3 = rng.standard_normal((160, 120, 130))
m3[rng.random((160, 120, 130)) < 0.01] = np.nan
ab("midaxis nan", m3, axis=1)
# stable kind with NaN keeps the defer -> parity
ab("stable kind parity", f, kind="stable")
# small n parity
sm = rng.standard_normal(1000); sm[3] = np.nan
ab("small parity", sm)

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

M = rng.standard_normal((4000, 2000))
M[rng.random((4000, 2000)) < 0.005] = np.nan
tn = best(lambda: np.sort(M, axis=-1)); tf = best(lambda: fnp.sort(M, axis=-1))
print(f"SORT_F64_NAN_LASTAXIS_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn0 = best(lambda: np.sort(M, axis=0)); tf0 = best(lambda: fnp.sort(M, axis=0))
print(f"SORT_F64_NAN_AXIS0_AB numpy_ms={tn0:.3f} fnp_ms={tf0:.3f} ratio={tn0 / tf0:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces SORT_F64_NAN_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "uniform-NaN f64 sort must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn axis_none_sort_argsort_flat_view_matches_numpy() -> Result<(), String> {
    // np.sort/np.argsort with EXPLICIT axis=None operate on the FLATTENED
    // array; for C-contiguous N-D input the ravel is a zero-copy reshape
    // view, so the dispatch now normalizes to 1-D and every flat fast path
    // (int radix, stable counting, NaN-aware stable radix, par value sort)
    // serves the form that previously delegated wholesale. Byte-exact by
    // construction: numpy's axis=None result IS its result on the ravel.
    // F-contig / non-contig / bool / small keep the delegate (parity).
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(311)
verdicts = []
def ab(fn_name, name, a):
    ours = getattr(fnp, fn_name)(a, axis=None)
    theirs = getattr(np, fn_name)(a, axis=None)
    if ours.tobytes() != theirs.tobytes() or ours.shape != theirs.shape:
        verdicts.append(f"FAIL {fn_name} {name}")
W2 = rng.integers(-2**62, 2**62, (2000, 2000))
ab("argsort", "2-D wide int", W2)
ab("sort", "2-D wide int", W2)
# 3-D f64 + canonical NaN, stable kind route via the flat stable radix
F3 = rng.standard_normal((160, 120, 110))
F3[rng.random((160, 120, 110)) < 0.01] = np.nan
if fnp.argsort(F3, axis=None, kind="stable").tobytes() != np.argsort(F3, axis=None, kind="stable").tobytes():
    verdicts.append("FAIL argsort 3-D stable nan")
ab("sort", "3-D f64 nan (delegate)", F3)
# dense ties -> the flat default-kind paths defer -> delegate parity through the view
D2 = rng.integers(0, 50, (1600, 1600))
ab("argsort", "dense ties", D2)
ab("sort", "dense ties", D2)
# layout/scope defers keep parity: F-contig, strided slice, bool, small, 1-D
ab("argsort", "F-contig", np.asfortranarray(W2[:800, :800]))
ab("sort", "F-contig", np.asfortranarray(W2[:800, :800]))
ab("argsort", "strided", W2[::2, ::2])
ab("sort", "strided", W2[::2, ::2])
ab("argsort", "bool", rng.integers(0, 2, (1500, 1500)).astype(bool))
ab("argsort", "small", rng.integers(-2**62, 2**62, (10, 10)))
ab("argsort", "1-D unchanged", rng.integers(-2**62, 2**62, 2_000_000))

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.integers(-2**62, 2**62, (2828, 2828))
tn = best(lambda: np.argsort(W, axis=None)); tf = best(lambda: fnp.argsort(W, axis=None))
print(f"ARGSORT_AXISNONE_INT_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tns = best(lambda: np.sort(W, axis=None)); tfs = best(lambda: fnp.sort(W, axis=None))
print(f"SORT_AXISNONE_INT_AB numpy_ms={tns:.3f} fnp_ms={tfs:.3f} ratio={tns / tfs:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces ARGSORT_AXISNONE_INT_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "axis=None flat-view sort/argsort must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn unique_nd_flat_view_matches_numpy() -> Result<(), String> {
    // np.unique's default axis=None operates on the FLATTENED array; the
    // dispatch now normalizes C-contiguous N-D input to a zero-copy
    // reshape(-1) view so the ndim==1-gated flat kernels (string, c128,
    // c64, datetime, struct) serve N-D input that previously delegated
    // wholesale. int/f64 kernels read the flat buffer either way
    // (regression rows). Byte-exact: np.unique(a) IS np.unique(a.ravel());
    // F-contig / defer cases keep delegate parity.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(313)
verdicts = []
def ab(name, a):
    ours = fnp.unique(a)
    theirs = np.unique(a)
    if ours.tobytes() != theirs.tobytes() or str(ours.dtype) != str(theirs.dtype):
        verdicts.append(f"FAIL {name}")
# unlocked arms: 2-D datetime64, 2-D complex128, 2-D fixed-width unicode
D = rng.integers(0, 5_000_000, (1414, 1414)).astype("datetime64[s]")
ab("2-D datetime64", D)
C = (rng.standard_normal((1414, 1414)) + 1j * rng.standard_normal((1414, 1414))).astype(np.complex128)
ab("2-D complex128", C)
S = np.array([f"k{v:07d}" for v in rng.integers(0, 400_000, 300_000)], dtype="U8").reshape(600, 500)
ab("2-D unicode", S)
# already-covered arms keep working through the view (regression)
ab("3-D small-range int", rng.integers(0, 300, (128, 128, 128)))
ab("2-D f64", np.round(rng.standard_normal((1500, 1400)), 3))
# mixed-sign-zero f64 defers (signed-zero-tie parity fix: which zero survives
# dedup is the sort's algorithm-specific tie choice) - 1-D row pins the fix
# for the pre-existing flat path, 2-D covered by the rounded row above
z1 = rng.standard_normal(2_000_000)
z1[::3] = 0.0
z1[1::3] = -0.0
ab("1-D mixed-zero f64", z1)
# defer/delegate parity: F-contig, NaN complex (kernel defers), small, 1-D
ab("F-contig datetime", np.asfortranarray(D[:500, :500]))
Cn = C[:800, :800].copy(); Cn[3, 5] = complex(np.nan, 1.0)
ab("2-D c128 nan parity", Cn)
ab("small 2-D", rng.integers(0, 10, (8, 9)))
ab("1-D unchanged", rng.integers(0, 5_000_000, 2_000_000).astype("datetime64[s]"))

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

tn = best(lambda: np.unique(C)); tf = best(lambda: fnp.unique(C))
print(f"UNIQUE_ND_C128_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tnd = best(lambda: np.unique(D)); tfd = best(lambda: fnp.unique(D))
print(f"UNIQUE_ND_DT64_AB numpy_ms={tnd:.3f} fnp_ms={tfd:.3f} ratio={tnd / tfd:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces UNIQUE_ND_C128_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "N-D flat-view unique must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn sort_complex_native_complex_matches_numpy() -> Result<(), String> {
    // np.sort_complex on a complexfloating input is copy + in-place LAST-AXIS
    // sort + unchanged dtype; the dispatch now routes exact c128/c64 ndarrays
    // to the shipped flat (1-D) and last-axis (N-D) complex value-sort
    // kernels instead of the old wholesale numpy defer (stale-routing class).
    // NaN / -0.0 inputs defer inside the kernels -> numpy fallback parity.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(317)
verdicts = []
def ab(name, a):
    ours = fnp.sort_complex(a)
    theirs = np.sort_complex(a)
    if ours.tobytes() != theirs.tobytes() or str(ours.dtype) != str(theirs.dtype) or ours.shape != theirs.shape:
        verdicts.append(f"FAIL {name}")
def cplx(shape, dt=np.complex128):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dt)
C1 = cplx(4_000_000)
ab("1-D c128", C1)
ab("2-D c128 lastaxis", cplx((1500, 1400)))
ab("1-D c64", cplx(4_000_000, np.complex64))
ab("2-D c64 lastaxis", cplx((1500, 1400), np.complex64))
# ties in both components exercise byte-identical tie handling
T = cplx(2_000_000)
T[::2] = T[0]
ab("dense ties c128", T)
# NaN / -0.0 defer -> numpy fallback parity
N1 = cplx(1_600_000); N1[7] = complex(np.nan, 1.0)
ab("c128 nan parity", N1)
Z1 = cplx(1_600_000); Z1[9] = complex(-0.0, 1.0)
ab("c128 -0.0 parity", Z1)
# real/int inputs keep the existing native/fallback routes
ab("1-D f64 real", rng.standard_normal(1_500_000))
ab("1-D int", rng.integers(-1000, 1000, 1_500_000))
ab("2-D real fallback", rng.standard_normal((800, 700)))
ab("small c128", cplx(64))

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

tn = best(lambda: np.sort_complex(C1)); tf = best(lambda: fnp.sort_complex(C1))
print(f"SORT_COMPLEX_C128_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
M = cplx((2000, 2000))
tn2 = best(lambda: np.sort_complex(M)); tf2 = best(lambda: fnp.sort_complex(M))
print(f"SORT_COMPLEX_2D_AB numpy_ms={tn2:.3f} fnp_ms={tf2:.3f} ratio={tn2 / tf2:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces SORT_COMPLEX_C128_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "complex sort_complex must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn histogram_nd_flat_view_matches_numpy() -> Result<(), String> {
    // np.histogram operates on the FLATTENED input; the dispatch now
    // normalizes C-contiguous N-D input to a zero-copy reshape(-1) view so
    // the parallel typed kernels (gated shape.len() == 1) serve the N-D form
    // (2-D image data is the common case) that previously delegated
    // wholesale. Byte-exact for counts AND edges; kwarg forms (bins array,
    // range, weights, density) and F-contig keep the delegate (parity).
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(331)
verdicts = []
def ab(name, a, **kw):
    oh, oe = fnp.histogram(a, **kw)
    th, te = np.histogram(a, **kw)
    if oh.tobytes() != th.tobytes() or oe.tobytes() != te.tobytes() \
       or str(oh.dtype) != str(th.dtype) or str(oe.dtype) != str(te.dtype):
        verdicts.append(f"FAIL {name}")
F2 = rng.standard_normal((2828, 2828))
ab("2-D f64 default", F2)
ab("2-D f64 bins=64", F2, bins=64)
ab("3-D int32", rng.integers(-500, 500, (200, 200, 200)).astype(np.int32))
ab("2-D u8 image", rng.integers(0, 256, (3000, 2500)).astype(np.uint8))
ab("2-D f32", rng.standard_normal((2000, 2000)).astype(np.float32))
# kwarg/layout forms keep prior routing -> parity
ab("range kwarg", F2[:800, :800], bins=32, range=(-2.0, 2.0))
ab("weights kwarg", F2[:500, :500], weights=np.abs(F2[:500, :500]))
ab("density kwarg", F2[:500, :500], density=True)
ab("bins array", F2[:500, :500], bins=np.linspace(-4, 4, 33))
ab("F-contig", np.asfortranarray(F2[:800, :800]))
ab("small 2-D", rng.standard_normal((8, 9)))
ab("1-D unchanged", rng.standard_normal(4_000_000))

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

tn = best(lambda: np.histogram(F2)); tf = best(lambda: fnp.histogram(F2))
print(f"HISTOGRAM_ND_F64_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
U = rng.integers(0, 256, (3000, 2500)).astype(np.uint8)
tnu = best(lambda: np.histogram(U, bins=256)); tfu = best(lambda: fnp.histogram(U, bins=256))
print(f"HISTOGRAM_ND_U8_AB numpy_ms={tnu:.3f} fnp_ms={tfu:.3f} ratio={tnu / tfu:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces HISTOGRAM_ND_F64_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "N-D flat-view histogram must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn isin_nd_string_complex_flat_view_matches_numpy() -> Result<(), String> {
    // The string/c128/c64/struct isin arms gate BOTH operands to 1-D; numpy's
    // isin is elementwise over any element shape and flattens test_elements,
    // so the dispatch now retries those arms on zero-copy ravel views and
    // reshapes the bool result back. N-D forms previously delegated wholesale
    // (numpy string isin sorts |elem|+|test|). Defers keep delegate parity.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(337)
verdicts = []
def ab(name, el, te, **kw):
    ours = fnp.isin(el, te, **kw)
    theirs = np.isin(el, te, **kw)
    if ours.tobytes() != theirs.tobytes() or ours.shape != theirs.shape or str(ours.dtype) != str(theirs.dtype):
        verdicts.append(f"FAIL {name}")
S2 = np.array([f"id{v:06d}" for v in rng.integers(0, 500_000, 2_100_000)], dtype="U8").reshape(1500, 1400)
ST = np.array([f"id{v:06d}" for v in rng.integers(0, 500_000, 200_000)], dtype="U8")
ab("2-D string element", S2, ST)
ab("2-D string invert", S2, ST, invert=True)
ab("string 1-D elem 2-D test", ST, S2[:400, :400])
C2 = (rng.standard_normal((1200, 1200)) + 1j * rng.standard_normal((1200, 1200))).astype(np.complex128)
CT = C2.ravel()[rng.choice(1_440_000, 100_000, replace=False)].copy()
ab("2-D c128 element", C2, CT)
ab("2-D c64 element", C2.astype(np.complex64), CT.astype(np.complex64))
# c128 NaN defer -> delegate parity through the views
Cn = C2[:400, :400].copy(); Cn[3, 5] = complex(np.nan, 1.0)
ab("2-D c128 nan parity", Cn, CT[:1000])
# layout/scope defers keep parity
ab("F-contig string", np.asfortranarray(S2[:500, :500]), ST)
ab("small 2-D string", S2[:4, :5], ST[:100])
ab("1-D string unchanged", S2.ravel()[:2_000_000], ST)
# int/f64 N-D were already covered - regression rows
ab("2-D int regression", rng.integers(0, 500_000, (1500, 1400)), rng.integers(0, 500_000, 200_000))
ab("2-D f64 regression", np.round(rng.standard_normal((1200, 1200)), 4), np.round(rng.standard_normal(150_000), 4))

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

tn = best(lambda: np.isin(S2, ST)); tf = best(lambda: fnp.isin(S2, ST))
print(f"ISIN_ND_STRING_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tnc = best(lambda: np.isin(C2, CT)); tfc = best(lambda: fnp.isin(C2, CT))
print(f"ISIN_ND_C128_AB numpy_ms={tnc:.3f} fnp_ms={tfc:.3f} ratio={tnc / tfc:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces ISIN_ND_STRING_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "N-D flat-view string/complex isin must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn setops_nd_flat_view_matches_numpy() -> Result<(), String> {
    // intersect1d/union1d/setdiff1d/setxor1d FLATTEN their inputs by
    // contract; the entry points now normalize C-contiguous N-D input to
    // zero-copy reshape(-1) views so the whole 1-D-gated arm chain
    // (narrow-int, wide-int, string, c128, datetime, struct) serves N-D
    // forms that previously delegated wholesale. Defers keep delegate
    // parity with ORIGINAL args.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(347)
verdicts = []
def ab(op, name, a, b, **kw):
    ours = getattr(fnp, op)(a, b, **kw)
    theirs = getattr(np, op)(a, b, **kw)
    if ours.tobytes() != theirs.tobytes() or str(ours.dtype) != str(theirs.dtype):
        verdicts.append(f"FAIL {op} {name}")
def strs(n, hi):
    return np.array([f"id{v:06d}" for v in rng.integers(0, hi, n)], dtype="U8")
S1 = strs(1_200_000, 400_000).reshape(1200, 1000)
S2 = strs(1_200_000, 400_000).reshape(1000, 1200)
ab("intersect1d", "2-D string", S1, S2)
ab("union1d", "2-D string", S1, S2)
ab("setdiff1d", "2-D string", S1, S2)
ab("setxor1d", "2-D string", S1[:600, :500], S2[:500, :600])
W1 = rng.integers(-2**62, 2**62, (1500, 1400))
W2 = rng.integers(-2**62, 2**62, (1400, 1500))
ab("intersect1d", "2-D wide int", W1, W2)
ab("setdiff1d", "2-D wide int", W1, W2)
ab("union1d", "2-D narrow int", rng.integers(0, 200, (1200, 1100)).astype(np.int16), rng.integers(0, 200, (1100, 1200)).astype(np.int16))
D1 = rng.integers(0, 3_000_000, (1200, 1100)).astype("datetime64[s]")
D2 = rng.integers(0, 3_000_000, (1100, 1200)).astype("datetime64[s]")
ab("setdiff1d", "2-D datetime", D1, D2)
# mixed 2-D vs 1-D, delegate-parity forms
ab("intersect1d", "2-D vs 1-D string", S1, S2.ravel()[:300_000])
ab("intersect1d", "F-contig parity", np.asfortranarray(W1[:500, :500]), W2[:400, :400])
ab("setdiff1d", "assume_unique parity", np.unique(W1)[:200_000].reshape(400, 500), np.unique(W2)[:200_000], assume_unique=True)
ab("union1d", "small 2-D", rng.integers(0, 50, (8, 9)), rng.integers(0, 50, (7, 6)))
ab("intersect1d", "1-D unchanged", W1.ravel()[:2_000_000], W2.ravel()[:2_000_000])

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

tn = best(lambda: np.union1d(S1, S2)); tf = best(lambda: fnp.union1d(S1, S2))
print(f"UNION1D_ND_STRING_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tni = best(lambda: np.intersect1d(W1, W2)); tfi = best(lambda: fnp.intersect1d(W1, W2))
print(f"INTERSECT1D_ND_WIDEINT_AB numpy_ms={tni:.3f} fnp_ms={tfi:.3f} ratio={tni / tfi:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces UNION1D_ND_STRING_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "N-D flat-view setops must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn f64_flat_sort_avx512_regate_matches_numpy() -> Result<(), String> {
    // Stale-basis regate (ledger 2026-07-13): on avx512f hosts numpy's
    // x86-simd-sort f64 qsort owns the flat basis (92.9ms at 8M vs 96.1ms
    // parallel merge sort, same-run measurement), so the flat f64 value-sort
    // arm now defers there; the original 1.6-1.85x margin predates that
    // numpy kernel and survives only on non-AVX-512 hosts, where the arm
    // stays enabled. AXIS arms are NOT gated (their NaN twins measured
    // 1.01-1.13x - wins). On the (avx512) gate worker the flat AB should
    // read ~1.0x (delegate) and the lastaxis control must stay a win.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(349)
verdicts = []
def ab(name, a, **kw):
    if fnp.sort(a, **kw).tobytes() != np.sort(a, **kw).tobytes():
        verdicts.append(f"FAIL {name}")
F = rng.standard_normal(8_000_000)
ab("flat clean f64", F)
ab("flat stable kind", F, kind="stable")
ab("flat small", F[:1000])
M = rng.standard_normal((4000, 2000))
ab("lastaxis clean", M, axis=-1)
ab("axis0 clean", M, axis=0)

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

tn = best(lambda: np.sort(F)); tf = best(lambda: fnp.sort(F))
print(f"SORT_F64_FLAT_REGATE_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn2 = best(lambda: np.sort(M, axis=-1)); tf2 = best(lambda: fnp.sort(M, axis=-1))
print(f"SORT_F64_LASTAXIS_CONTROL_AB numpy_ms={tn2:.3f} fnp_ms={tf2:.3f} ratio={tn2 / tf2:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces SORT_F64_FLAT_REGATE_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "regated flat f64 sort must stay bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_sort_class_stale_basis_probe_and_parity() -> Result<(), String> {
    // Stale-basis follow-through (the flat-f64-sort regate rule: numpy 2.x
    // x86-simd-sort ships int qsort/argsort on avx2+ hosts too — re-run
    // sort-class ABs after any numpy upgrade). Parity rows pin byte-exactness
    // of whatever routing is live; the AB rows measure the int flat sort /
    // default argsort arms against the current worker's numpy for the regate
    // decision.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(353)
verdicts = []
def ab(fn, name, a, **kw):
    if getattr(fnp, fn)(a, **kw).tobytes() != getattr(np, fn)(a, **kw).tobytes():
        verdicts.append(f"FAIL {fn} {name}")
W = rng.integers(-2**62, 2**62, 8_000_000)
I32 = rng.integers(-2**31, 2**31 - 1, 8_000_000).astype(np.int32)
D = rng.integers(0, 5_000_000_000, 8_000_000).astype("datetime64[ns]")
ab("sort", "i64 flat", W)
ab("sort", "i32 flat", I32)
ab("sort", "dt64 flat", D)
ab("argsort", "i64 default distinct", np.random.default_rng(354).permutation(8_000_000).astype(np.int64) * 1099511627776 + np.arange(8_000_000))
ab("sort", "i64 small", W[:1000])

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

tn = best(lambda: np.sort(W)); tf = best(lambda: fnp.sort(W))
print(f"SORT_I64_FLAT_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn2 = best(lambda: np.sort(I32)); tf2 = best(lambda: fnp.sort(I32))
print(f"SORT_I32_FLAT_AB numpy_ms={tn2:.3f} fnp_ms={tf2:.3f} ratio={tn2 / tf2:.3f}")
tn3 = best(lambda: np.sort(D)); tf3 = best(lambda: fnp.sort(D))
print(f"SORT_DT64_FLAT_AB numpy_ms={tn3:.3f} fnp_ms={tf3:.3f} ratio={tn3 / tf3:.3f}")
AD = np.random.default_rng(355).permutation(8_000_000).astype(np.int64) * 1099511627776 + np.arange(8_000_000)
tn4 = best(lambda: np.argsort(AD)); tf4 = best(lambda: fnp.argsort(AD))
print(f"ARGSORT_I64_DEFAULT_AB numpy_ms={tn4:.3f} fnp_ms={tf4:.3f} ratio={tn4 / tf4:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces SORT_I64_FLAT_AB etc. under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "int sort-class arms must stay bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn packbits_native_parallel_matches_numpy() -> Result<(), String> {
    // packbits parity coverage + basis pin. A native parallel packbits arm
    // (mirror of the shipped unpackbits kernel) was built and gate-REJECTED
    // 2026-07-13 at 1.040x: numpy's packbits is a SIMD movemask kernel
    // (64M bools in 3.755ms = ~17 GB/s, memory-bound) - the 'single-threaded
    // compute-bound' premise transferred from unpackbits was wrong. The
    // DIRECTION ASYMMETRY: unpack (mask-expand, 8x larger DRAM writes)
    // shipped at 3.5x; pack (movemask) is saturated. Rows pin passthrough
    // parity incl. tail bytes, N-D flatten, kwargs, and the round-trip
    // through the still-native unpackbits.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(359)
verdicts = []
def ab(name, a, **kw):
    ours = fnp.packbits(a, **kw)
    theirs = np.packbits(a, **kw)
    if ours.tobytes() != theirs.tobytes() or str(ours.dtype) != str(theirs.dtype) or ours.shape != theirs.shape:
        verdicts.append(f"FAIL {name}")
B = rng.random(64_000_000) < 0.5
ab("1-D bool 64M", B)
ab("1-D bool tail n%8=3", B[: 32_000_003])
U = rng.integers(0, 256, 40_000_000).astype(np.uint8)
ab("1-D uint8 nonzero->1", U)
ab("2-D bool flatten", B[: 48_000_000].reshape(6000, 8000))
ab("3-D bool flatten", B[: 24_000_000].reshape(200, 400, 300))
# kwarg / scope forms keep the delegate -> parity
ab("axis kwarg", B[: 1_600_000].reshape(400, 4000), axis=1)
ab("bitorder little", B[:2_000_000], bitorder="little")
ab("int32 input", rng.integers(0, 2, 2_000_000).astype(np.int32))
ab("F-contig", np.asfortranarray(B[: 1_000_000].reshape(1000, 1000)))
ab("small", B[:1000])
# round-trip through the native unpackbits sibling
P = fnp.packbits(B[: 32_000_000])
if not np.array_equal(fnp.unpackbits(P), np.unpackbits(np.packbits(B[: 32_000_000]))):
    verdicts.append("FAIL roundtrip")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

tn = best(lambda: np.packbits(B)); tf = best(lambda: fnp.packbits(B))
print(f"PACKBITS_BOOL_64M_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces PACKBITS_BOOL_64M_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "native packbits must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn add_at_i64_large_target_parallel_matches_numpy() -> Result<(), String> {
    // np.add.at(i64, idx, vals) large-target regime: numpy's ufunc.at is a
    // DRAM-latency-bound serial scatter there (probe: 136ms for 8M into 8M);
    // the parallel atomic fetch_add arm is byte-exact because wrapping i64
    // addition commutes (duplicate-index order unobservable) and fetch_add
    // wraps exactly like numpy's i64 overflow. Histogram-style small targets
    // (numpy fast path, 7.5ms), floats, scalar vals, other dtypes, 2-D, and
    // OOB indices keep the delegate (parity / numpy's exact errors).
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(367)
verdicts = []
def ab(name, n, idx, vals, dtype=np.int64):
    a1 = np.zeros(n, dtype=dtype); a2 = np.zeros(n, dtype=dtype)
    fnp.add.at(a1, idx, vals)
    np.add.at(a2, idx, vals)
    if a1.tobytes() != a2.tobytes():
        verdicts.append(f"FAIL {name}")
n = 4_000_000
idx = rng.integers(0, n, 4_000_000)
vals = rng.integers(-10**9, 10**9, 4_000_000)
ab("large target dup-heavy", n, idx, vals)
# negative indices wrap once
idxn = idx.copy(); idxn[::3] -= n
ab("negative indices", n, idxn, vals)
# wrapping overflow parity
big = rng.integers(2**62, 2**63 - 1, 4_000_000)
ab("i64 wrap overflow", n, idx, big)
# delegate-parity forms: histogram-regime small target, f64, scalar vals, i32, 2-D target
ab("small target delegate", 1024, rng.integers(0, 1024, 4_000_000), vals)
af1 = np.zeros(n); af2 = np.zeros(n)
fv = rng.standard_normal(4_000_000)
fnp.add.at(af1, idx, fv); np.add.at(af2, idx, fv)
if af1.tobytes() != af2.tobytes():
    verdicts.append("FAIL f64 delegate")
s1 = np.zeros(n, dtype=np.int64); s2 = np.zeros(n, dtype=np.int64)
fnp.add.at(s1, idx, 7); np.add.at(s2, idx, 7)
if s1.tobytes() != s2.tobytes():
    verdicts.append("FAIL scalar delegate")
ab("i32 delegate", n, idx.astype(np.int32), vals.astype(np.int32), dtype=np.int32)
t1 = np.zeros((2000, 2000), dtype=np.int64); t2 = np.zeros((2000, 2000), dtype=np.int64)
r = rng.integers(0, 2000, 3_000_000)
fnp.add.at(t1, r, 1); np.add.at(t2, r, 1)
if t1.tobytes() != t2.tobytes():
    verdicts.append("FAIL 2-D delegate")
# OOB raises identically through the delegate
try:
    fnp.add.at(np.zeros(n, dtype=np.int64), np.array([0, n], dtype=np.int64), np.array([1, 1], dtype=np.int64))
    verdicts.append("FAIL oob no-raise")
except IndexError:
    pass

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

A = np.zeros(8_000_000, dtype=np.int64)
IDX = rng.integers(0, 8_000_000, 8_000_000)
V = rng.integers(-1000, 1000, 8_000_000)
tn = best(lambda: np.add.at(A, IDX, V)); tf = best(lambda: fnp.add.at(A, IDX, V))
print(f"ADD_AT_I64_LARGE_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces ADD_AT_I64_LARGE_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "parallel i64 add.at must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn add_at_int_dtype_arms_parallel_match_numpy() -> Result<(), String> {
    // Generalizes the shipped i64 add.at arm (7d7e8faf) to u64/i32/u32 via
    // one macro: wrapping addition commutes for every fixed-width int, so
    // the parallel atomic scatter is byte-exact per dtype, values must share
    // the target's exact dtype, and the latency-bound large-target physics
    // is dtype-independent. Mixed-dtype values and i16/u16/i8/u8 targets
    // keep the delegate (parity).
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(373)
verdicts = []
def ab(name, n, idx, vals, dtype):
    a1 = np.zeros(n, dtype=dtype); a2 = np.zeros(n, dtype=dtype)
    fnp.add.at(a1, idx, vals)
    np.add.at(a2, idx, vals)
    if a1.tobytes() != a2.tobytes():
        verdicts.append(f"FAIL {name}")
n = 4_000_000
idx = rng.integers(0, n, 4_000_000)
idx[::5] -= n  # negative-index wrap mixed in
ab("u64", n, idx, rng.integers(0, 2**64, 4_000_000, dtype=np.uint64), np.uint64)
ab("u64 wrap", n, idx, rng.integers(2**63, 2**64, 4_000_000, dtype=np.uint64), np.uint64)
ab("i32", n, idx, rng.integers(-2**30, 2**30, 4_000_000).astype(np.int32), np.int32)
ab("i32 wrap", n, idx, rng.integers(2**30, 2**31 - 1, 4_000_000).astype(np.int32), np.int32)
ab("u32", n, idx, rng.integers(0, 2**32, 4_000_000, dtype=np.uint32), np.uint32)
ab("i64 regression", n, idx, rng.integers(-10**9, 10**9, 4_000_000), np.int64)
# delegate-parity: mixed dtype vals, narrow targets
m1 = np.zeros(n, dtype=np.uint64); m2 = np.zeros(n, dtype=np.uint64)
iv = rng.integers(0, 1000, 4_000_000).astype(np.int32)
fnp.add.at(m1, np.abs(idx), iv); np.add.at(m2, np.abs(idx), iv)
if m1.tobytes() != m2.tobytes():
    verdicts.append("FAIL mixed dtype delegate")
ab("i16 delegate", n, np.abs(idx), rng.integers(0, 100, 4_000_000).astype(np.int16), np.int16)

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

AU = np.zeros(8_000_000, dtype=np.uint64)
I8 = rng.integers(0, 8_000_000, 8_000_000)
VU = rng.integers(0, 2**63, 8_000_000, dtype=np.uint64)
tn = best(lambda: np.add.at(AU, I8, VU)); tf = best(lambda: fnp.add.at(AU, I8, VU))
print(f"ADD_AT_U64_LARGE_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
A3 = np.zeros(8_000_000, dtype=np.int32)
V3 = rng.integers(-1000, 1000, 8_000_000).astype(np.int32)
tn2 = best(lambda: np.add.at(A3, I8, V3)); tf2 = best(lambda: fnp.add.at(A3, I8, V3))
print(f"ADD_AT_I32_LARGE_AB numpy_ms={tn2:.3f} fnp_ms={tf2:.3f} ratio={tn2 / tf2:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces ADD_AT_U64_LARGE_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "int-dtype add.at arms must be bit-identical to numpy: {result}"
    );
    Ok(())
}
