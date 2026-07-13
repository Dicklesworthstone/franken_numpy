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
