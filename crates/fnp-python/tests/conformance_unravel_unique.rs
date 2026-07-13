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
