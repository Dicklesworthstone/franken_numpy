//! Conformance tests for numpy sorting and searching functions against NumPy oracle.
//!
//! Tests sort, argsort, unique, searchsorted, nonzero, count_nonzero.

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
// sort
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sort_argsort_python_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def sort_like_outcome(fn, args, kwargs):
    try:
        result = fn(*args, **kwargs)
        arr = np.asarray(result)
        return ("ok", type(result).__name__, str(arr.dtype), tuple(arr.shape), arr.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))

cases = [
    ("sort list ints", fnp.sort, np.sort, lambda: (([3, 1, 2, 1],), {})),
    ("sort tuple floats", fnp.sort, np.sort, lambda: (((3.5, 1.5, 2.5),), {})),
    ("sort bool list", fnp.sort, np.sort, lambda: (([True, False, True],), {})),
    ("sort nested axis none", fnp.sort, np.sort, lambda: (([[3, 1], [2, 4]],), {"axis": None})),
    ("sort nested axis minus one", fnp.sort, np.sort, lambda: (([[3, 1], [2, 4]],), {"axis": -1})),
    ("sort invalid axis error", fnp.sort, np.sort, lambda: (([[3, 1], [2, 4]],), {"axis": 2})),
    ("sort string list", fnp.sort, np.sort, lambda: ((["b", "a", "c"],), {})),
    ("argsort list ints", fnp.argsort, np.argsort, lambda: (([3, 1, 2, 1],), {})),
    ("argsort tuple floats", fnp.argsort, np.argsort, lambda: (((3.5, 1.5, 2.5),), {})),
    ("argsort bool list", fnp.argsort, np.argsort, lambda: (([True, False, True],), {})),
    ("argsort nested axis none", fnp.argsort, np.argsort, lambda: (([[3, 1], [2, 4]],), {"axis": None})),
    ("argsort nested axis zero", fnp.argsort, np.argsort, lambda: (([[3, 1], [2, 4]],), {"axis": 0})),
    ("argsort invalid axis error", fnp.argsort, np.argsort, lambda: (([[3, 1], [2, 4]],), {"axis": 2})),
    ("argsort stable ties", fnp.argsort, np.argsort, lambda: (([2, 1, 2, 1],), {"stable": True})),
]

ok = True
for label, actual_fn, expected_fn, factory in cases:
    args, kwargs = factory()
    actual = sort_like_outcome(actual_fn, args, kwargs)
    args, kwargs = factory()
    expected = sort_like_outcome(expected_fn, args, kwargs)
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
        "sort/argsort Python-container surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sort_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
result = fnp.sort(a)
expected = np.sort(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sort 1d should match numpy");
    Ok(())
}

#[test]
fn sort_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 1, 4], [1, 5, 9]])
result = fnp.sort(a, axis=0)
expected = np.sort(a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sort 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn sort_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 1, 4], [1, 5, 9]])
result = fnp.sort(a, axis=1)
expected = np.sort(a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sort 2d axis=1 should match numpy");
    Ok(())
}

#[test]
fn sort_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3.14, 1.41, 2.71, 1.61])
result = fnp.sort(a)
expected = np.sort(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sort float should match numpy");
    Ok(())
}

#[test]
fn sort_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([-3, 1, -4, 1, -5, 9, -2, 6])
result = fnp.sort(a)
expected = np.sort(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sort with negatives should match numpy"
    );
    Ok(())
}

#[test]
fn sort_f16_no_ship_passthrough_preserves_numpy_bytes() -> Result<(), String> {
    let script = fnp_script(
        r#"
original_sort = np.sort
observed_dtypes = []

def sort_spy(value, *args, **kwargs):
    observed_dtypes.append(np.asarray(value).dtype.str)
    return original_sort(value, *args, **kwargs)

np.sort = sort_spy

def same_bytes_and_route(value, inner_dtype, **kwargs):
    expected = original_sort(value, **kwargs)
    observed_dtypes.clear()
    actual = fnp.sort(value, **kwargs)
    return (
        actual.dtype == expected.dtype
        and actual.dtype.str == expected.dtype.str
        and actual.dtype.metadata == expected.dtype.metadata
        and actual.shape == expected.shape
        and actual.tobytes() == expected.tobytes()
        and observed_dtypes == [np.dtype(inner_dtype).str]
    )

# Exercise every finite f16 bit pattern plus both infinities. The only excluded
# values are NaNs and -0.0, whose payload/tie ordering is deliberately guarded.
bits = np.arange(1 << 16, dtype=np.uint16)
all_values = bits.view(np.float16)
route_values = all_values[(~np.isnan(all_values)) & (bits != 0x8000)]

finite_ties = np.tile(
    np.array([0x0000, 0x3c00, 0xbc00, 0x7c00, 0xfc00], dtype=np.uint16),
    1 << 13,
).view(np.float16)
nan_guard = np.tile(
    np.array([0x0000, 0x3c00, 0x7e01, 0xfe55], dtype=np.uint16),
    1 << 13,
).view(np.float16)
negative_zero_guard = np.tile(
    np.array([0x0000, 0x8000, 0x3c00, 0xbc00], dtype=np.uint16),
    1 << 13,
).view(np.float16)
noncontiguous = route_values[::2]
matrix = route_values[:32768].reshape(128, 256)
non_native_dtype = np.dtype('>f2' if np.little_endian else '<f2')
non_native_values = route_values.astype(non_native_dtype)
metadata_dtype = np.dtype(np.float16, metadata={'tag': 'fnp-f16-sort'})
metadata_values = np.array(route_values, dtype=metadata_dtype, copy=True)

checks = []
for kind in [None, 'quicksort', 'stable', 'mergesort', 'heapsort']:
    kwargs = {} if kind is None else {'kind': kind}
    checks.append(same_bytes_and_route(route_values, np.float16, **kwargs))
    checks.append(same_bytes_and_route(finite_ties, np.float16, **kwargs))
    checks.append(same_bytes_and_route(nan_guard, np.float16, **kwargs))
    checks.append(same_bytes_and_route(negative_zero_guard, np.float16, **kwargs))

checks.extend([
    same_bytes_and_route(route_values, np.float16, axis=None),
    same_bytes_and_route(route_values, np.float16, axis=0),
    same_bytes_and_route(route_values, np.float16, axis=-1),
    same_bytes_and_route(route_values[:1024], np.float16),
    same_bytes_and_route(noncontiguous, np.float16),
    same_bytes_and_route(matrix, np.float16, axis=-1),
    same_bytes_and_route(non_native_values, non_native_dtype),
    same_bytes_and_route(metadata_values, metadata_dtype),
])
print(all(checks))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "f16 no-ship passthrough should preserve every NumPy byte and dtype edge: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// argsort
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn argsort_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
result = fnp.argsort(a)
expected = np.argsort(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "argsort 1d should match numpy");
    Ok(())
}

#[test]
fn argsort_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 1, 4], [1, 5, 9]])
result = fnp.argsort(a, axis=1)
expected = np.argsort(a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "argsort 2d should match numpy");
    Ok(())
}

#[test]
fn argsort_reconstructs_sort() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9])
indices = fnp.argsort(a)
sorted_via_indices = a[indices]
sorted_direct = fnp.sort(a)
print(np.array_equal(sorted_via_indices, sorted_direct))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "a[argsort(a)] should equal sort(a)");
    Ok(())
}

#[test]
fn argsort_struct_mixed_float_matches_numpy_distinct_and_tied() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(0)
n = 70_000
dt = [('id', '<i8'), ('val', '<f8')]

distinct = np.zeros(n, dtype=dt)
distinct['id'] = rng.permutation(n)
distinct['val'] = rng.standard_normal(n)
distinct_actual = fnp.argsort(distinct)
distinct_expected = np.argsort(distinct)

tied = np.zeros(n, dtype=dt)
tied['id'] = np.repeat(np.arange(n // 4), 4)
tied['val'] = 1.0
tied_actual = fnp.argsort(tied)
tied_expected = np.argsort(tied)

print(np.array_equal(distinct_actual, distinct_expected) and np.array_equal(tied_actual, tied_expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "structured mixed float argsort should match numpy for distinct records and tie fallback"
    );
    Ok(())
}

#[test]
fn argsort_temporal_complex_stable_dense_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(43)
n = (1 << 20) + 129
ticks = rng.integers(-1000, 1000, n, dtype=np.int64)
c_re = rng.integers(-100, 100, n)
c_im = rng.integers(-100, 100, n)
cases = [
    ("datetime64", ticks.astype("datetime64[s]")),
    ("timedelta64", ticks.astype("timedelta64[ns]")),
    ("complex128", (c_re + 1j * c_im).astype(np.complex128)),
    ("complex64", (c_re + 1j * c_im).astype(np.complex64)),
]
ok = True
for label, arr in cases:
    for kind in ("stable", "mergesort"):
        got = fnp.argsort(arr, kind=kind)
        exp = np.argsort(arr, kind=kind)
        if not np.array_equal(got, exp):
            print(("dense", label, kind))
            ok = False

special_cases = [
    ("datetime64_NaT", np.array(["1970-01-03", "NaT", "1970-01-01", "NaT"], dtype="datetime64[D]")),
    ("timedelta64_NaT", np.array([3, "NaT", 1, "NaT"], dtype="timedelta64[D]")),
    ("complex128_NaN", np.array([1 + 2j, np.nan + 0j, 1 + 1j, np.nan + 3j], dtype=np.complex128)),
]
for label, arr in special_cases:
    got = fnp.argsort(arr, kind="stable")
    exp = np.argsort(arr, kind="stable")
    if not np.array_equal(got, exp):
        print(("special", label))
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "stable temporal/complex argsort should match numpy on dense ties: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// unique
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unique_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4])
result = fnp.unique(a)
expected = np.unique(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique basic should match numpy");
    Ok(())
}

#[test]
fn unique_unsorted() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5, 2, 3, 2, 5, 1, 3, 1])
result = fnp.unique(a)
expected = np.unique(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique unsorted should match numpy");
    Ok(())
}

#[test]
fn unique_axis1_large_small_range_int_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(73)
a = rng.integers(-8, 12, (4, (1 << 17) + 17), dtype=np.int64)
a[:, :1024] = a[:, 1024:2048]
a[0, 0] = -8
a[1, 1] = -8
a[2, 2] = 11
a[3, 3] = 0
ok = True
for axis in (1, -1):
    result = fnp.unique(a, axis=axis)
    expected = np.unique(a, axis=axis)
    if result.dtype != expected.dtype:
        print(("dtype", axis, str(result.dtype), str(expected.dtype)))
        ok = False
    if result.shape != expected.shape:
        print(("shape", axis, result.shape, expected.shape))
        ok = False
    if not np.array_equal(result, expected):
        print(("values", axis))
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unique axis=1 small-range int should match numpy"
    );
    Ok(())
}

#[test]
fn unique_axis0_narrow_int_rows_preserves_dtype_and_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(91)
ok = True
for dtype in (np.int32, np.uint16):
    base = rng.integers(0, 60000, ((1 << 16) + 9, 4), dtype=dtype)
    if np.issubdtype(dtype, np.signedinteger):
        base = (base.astype(np.int64) - 30000).astype(dtype)
    a = np.concatenate([base, base[:2048], base[1024:3072]])
    result = fnp.unique(a, axis=0)
    expected = np.unique(a, axis=0)
    if result.dtype != expected.dtype:
        print(("dtype", str(dtype), str(result.dtype), str(expected.dtype)))
        ok = False
    if result.shape != expected.shape:
        print(("shape", str(dtype), result.shape, expected.shape))
        ok = False
    if not np.array_equal(result, expected):
        print(("values", str(dtype)))
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unique axis=0 narrow-int rows should preserve dtype and match numpy"
    );
    Ok(())
}

#[test]
fn unique_axis0_narrow_int_rows_return_flags_preserve_dtype_and_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(92)
ok = True
cases = []
base_i32 = rng.integers(-(1 << 30), 1 << 30, ((1 << 16) + 17, 4), dtype=np.int32)
cases.append(base_i32)
base_u32 = rng.integers(0, 1 << 31, ((1 << 16) + 17, 4), dtype=np.uint32)
cases.append(base_u32)
for base in cases:
    a = np.concatenate([base, base[:4096], base[2048:6144]])
    result = fnp.unique(a, axis=0, return_index=True, return_inverse=True, return_counts=True)
    expected = np.unique(a, axis=0, return_index=True, return_inverse=True, return_counts=True)
    if result[0].dtype != expected[0].dtype:
        print(("dtype", str(a.dtype), str(result[0].dtype), str(expected[0].dtype)))
        ok = False
    if result[0].shape != expected[0].shape:
        print(("shape", str(a.dtype), result[0].shape, expected[0].shape))
        ok = False
    for i, (got, exp) in enumerate(zip(result, expected)):
        if not np.array_equal(got, exp):
            print(("values", str(a.dtype), i))
            ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unique axis=0 narrow-int rows return flags should preserve dtype and match numpy"
    );
    Ok(())
}

#[test]
fn unique_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 1.5, 3.5, 2.5])
result = fnp.unique(a)
expected = np.unique(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique float should match numpy");
    Ok(())
}

#[test]
fn unique_return_flag_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def clean(value):
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value

def normalize_unique_result(value):
    if isinstance(value, tuple):
        return ("tuple", [normalize_unique_result(item) for item in value])
    arr = np.asarray(value)
    return ("array", type(value).__name__, str(arr.dtype), tuple(arr.shape), clean(arr.tolist()))

def unique_outcome(fn, value, **kwargs):
    try:
        return ("ok", normalize_unique_result(fn(value, **kwargs)))
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))

cases = [
    ("empty list all return flags", lambda: ([], {
        "return_index": True,
        "return_inverse": True,
        "return_counts": True,
    })),
    ("list return_index", lambda: ([3, 1, 2, 1, 3], {"return_index": True})),
    ("bool list return_counts", lambda: ([False, True, False, True], {"return_counts": True})),
    ("string list delegate counts", lambda: (["b", "a", "b"], {
        "return_index": True,
        "return_counts": True,
    })),
    ("tuple all return flags", lambda: ((3, 1, 2, 1, 3), {
        "return_index": True,
        "return_inverse": True,
        "return_counts": True,
    })),
    ("int16 counting full return", lambda: (np.array([2, 1, 2, 0, 1, 2], dtype=np.int16), {
        "return_index": True,
        "return_inverse": True,
        "return_counts": True,
    })),
    ("equal_nan false delegate", lambda: (np.array([np.nan, 1.0, np.nan]), {"equal_nan": False})),
    ("axis rows counts", lambda: ([[1, 2], [1, 2], [3, 4]], {
        "axis": 0,
        "return_counts": True,
    })),
    ("axis columns inverse", lambda: ([[1, 1, 2], [3, 3, 4]], {
        "axis": 1,
        "return_inverse": True,
    })),
]

ok = True
for label, factory in cases:
    value, kwargs = factory()
    actual = unique_outcome(fnp.unique, value, **kwargs)
    value, kwargs = factory()
    expected = unique_outcome(np.unique, value, **kwargs)
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
        "unique return-flag container surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn unique_string_return_flags_large_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(725)
n = (1 << 17) + 333
u = rng.integers(97, 103, (n, 4), dtype=np.uint32).reshape(-1).view("U4")
s = rng.integers(97, 103, (n, 8), dtype=np.uint8).view("S8").reshape(-1)

ok = True
for label, arr in [("U4", u), ("S8", s)]:
    got = fnp.unique(arr, return_index=True, return_inverse=True, return_counts=True)
    exp = np.unique(arr, return_index=True, return_inverse=True, return_counts=True)
    if not isinstance(got, tuple) or len(got) != 4:
        print(("surface", label, type(got).__name__, len(got) if isinstance(got, tuple) else None))
        ok = False
        continue
    for i, (g, e) in enumerate(zip(got, exp)):
        if np.asarray(g).dtype != np.asarray(e).dtype:
            print(("dtype", label, i, str(np.asarray(g).dtype), str(np.asarray(e).dtype)))
            ok = False
        if np.asarray(g).shape != np.asarray(e).shape:
            print(("shape", label, i, np.asarray(g).shape, np.asarray(e).shape))
            ok = False
        if not np.array_equal(g, e):
            mismatch = np.flatnonzero(np.asarray(g).ravel() != np.asarray(e).ravel())[:10].tolist()
            print(("values", label, i, mismatch))
            ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large string unique return flags should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// searchsorted
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn searchsorted_python_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def searchsorted_outcome(fn, a, v, **kwargs):
    try:
        result = fn(a, v, **kwargs)
        arr = np.asarray(result)
        return ("ok", type(result).__name__, str(arr.dtype), tuple(arr.shape), arr.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))

cases = [
    ("list query stays array", lambda: ([1, 3, 5], [4], {})),
    ("tuple query right", lambda: ((1, 2, 2, 3), (2, 3), {"side": "right"})),
    ("python scalar query", lambda: ([1, 3, 5], 4, {})),
    ("zero-dimensional query", lambda: ([1, 3, 5], np.array(3), {})),
    ("matrix query shape", lambda: ([1, 3, 5, 7], [[0, 4], [8, 1]], {})),
    ("empty haystack", lambda: ([], [0, 1], {})),
    ("empty query", lambda: ([1, 2, 3], [], {})),
    ("nan-aware f64 ordering", lambda: (np.array([0.0, 1.0, np.nan]), [np.nan, 0.5], {})),
    ("sorter as list", lambda: ([30, 10, 20], [15, 30], {"sorter": [1, 2, 0]})),
    ("invalid side error", lambda: ([1, 3, 5], [3], {"side": "middle"})),
    ("string list delegate", lambda: (["a", "c", "e"], ["b", "e"], {})),
]

ok = True
for label, factory in cases:
    a, v, kwargs = factory()
    actual = searchsorted_outcome(fnp.searchsorted, a, v, **kwargs)
    a, v, kwargs = factory()
    expected = searchsorted_outcome(np.searchsorted, a, v, **kwargs)
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
        "searchsorted Python-container surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn searchsorted_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
v = np.array([2, 3, 4])
result = fnp.searchsorted(a, v)
expected = np.searchsorted(a, v)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "searchsorted basic should match numpy"
    );
    Ok(())
}

#[test]
fn searchsorted_structured_uint64_records_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = 70000
dt = [('a','<u8'),('b','<u8')]
base = np.arange(n, dtype=np.uint64)
h = np.zeros(n, dtype=dt)
h['a'] = base // np.uint64(100)
h['b'] = base % np.uint64(100)
q = h[(np.arange(n, dtype=np.int64) * 37) % n]
left = np.array_equal(fnp.searchsorted(h, q, side='left'), np.searchsorted(h, q, side='left'))
right = np.array_equal(fnp.searchsorted(h, q, side='right'), np.searchsorted(h, q, side='right'))
print(left and right)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "structured uint64 searchsorted should match numpy"
    );
    Ok(())
}

#[test]
fn searchsorted_structured_int64_prefix_records_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = 70000
dt = [('a','<i8'),('b','<i8')]
base = np.arange(n, dtype=np.int64)
h = np.zeros(n, dtype=dt)
h['a'] = (base // 5) * 3 - 50000
h['b'] = (base % 5) - 2
probe = (base * 37) % (n + 2000)
q = np.zeros(n, dtype=dt)
q['a'] = (probe // 5) * 3 - 50003
q['b'] = (probe % 11) - 5
left = np.array_equal(fnp.searchsorted(h, q, side='left'), np.searchsorted(h, q, side='left'))
right = np.array_equal(fnp.searchsorted(h, q, side='right'), np.searchsorted(h, q, side='right'))
print(left and right)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "structured int64 prefix searchsorted should match numpy"
    );
    Ok(())
}

#[test]
fn searchsorted_packed_latin1_string_records_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = 70000
base = np.arange(n, dtype=np.uint32)

u_cells = np.empty((n, 8), dtype=np.uint32)
u_cells[:, 0] = 97 + (base // 2048) % 5
u_cells[:, 1] = 97 + (base // 256) % 7
u_cells[:, 2] = 0
u_cells[:, 3] = 97 + (base // 32) % 11
u_cells[:, 4] = 97 + (base // 8) % 13
u_cells[:, 5] = 97 + base % 17
u_cells[:, 6] = 0
u_cells[:, 7] = 97 + (base * 3) % 19
u = np.sort(u_cells.reshape(-1).view('U8'))
uq_cells = u_cells[(base * 37 + 11) % n].copy()
uq_cells[0, :] = 0
uq_cells[1, :] = 255
uq = uq_cells.reshape(-1).view('U8')

s_cells = np.empty((n, 8), dtype=np.uint8)
s_cells[:, 0] = (97 + (base // 2048) % 5).astype(np.uint8)
s_cells[:, 1] = (97 + (base // 256) % 7).astype(np.uint8)
s_cells[:, 2] = 0
s_cells[:, 3] = (97 + (base // 32) % 11).astype(np.uint8)
s_cells[:, 4] = (97 + (base // 8) % 13).astype(np.uint8)
s_cells[:, 5] = (97 + base % 17).astype(np.uint8)
s_cells[:, 6] = 0
s_cells[:, 7] = (97 + (base * 3) % 19).astype(np.uint8)
s = np.sort(s_cells.view('S8').reshape(-1))
sq_cells = s_cells[(base * 41 + 7) % n].copy()
sq_cells[0, :] = 0
sq_cells[1, :] = 255
sq = sq_cells.view('S8').reshape(-1)

same_u = np.array(['aaaa', 'aaaa', 'bbbb'], dtype='U4')
same_s = np.array([b'aaaa', b'aaaa', b'bbbb'], dtype='S4')

ok = True
for label, hay, query in [
    ('U8', u, uq),
    ('S8', s, sq),
    ('U4-small-identical', same_u, np.array(['aaaa', 'aaaz', 'bbbb'], dtype='U4')),
    ('S4-small-identical', same_s, np.array([b'aaaa', b'aaaz', b'bbbb'], dtype='S4')),
]:
    for side in ('left', 'right'):
        got = fnp.searchsorted(hay, query, side=side)
        exp = np.searchsorted(hay, query, side=side)
        if not np.array_equal(got, exp):
            print((label, side, got[:10].tolist() if hasattr(got, '__len__') else got, exp[:10].tolist() if hasattr(exp, '__len__') else exp))
            ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "packed Latin-1 string searchsorted should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn searchsorted_left() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4])
v = np.array([2, 3])
result = fnp.searchsorted(a, v, side='left')
expected = np.searchsorted(a, v, side='left')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "searchsorted left should match numpy"
    );
    Ok(())
}

#[test]
fn searchsorted_right() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4])
v = np.array([2, 3])
result = fnp.searchsorted(a, v, side='right')
expected = np.searchsorted(a, v, side='right')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "searchsorted right should match numpy"
    );
    Ok(())
}

#[test]
fn searchsorted_large_f64_array_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(123)
a = np.concatenate([np.linspace(-4.0, 4.0, 50), np.array([np.nan])])
v = rng.standard_normal((2049, 1025)).astype(np.float64)
flat = v.ravel()
flat[::499983] = np.nan
flat[1::500003] = np.inf
flat[2::500009] = -np.inf

ok = True
for side in ("left", "right"):
    result = fnp.searchsorted(a, v, side=side)
    expected = np.searchsorted(a, v, side=side)
    if result.dtype != expected.dtype:
        print(("dtype", side, str(result.dtype), str(expected.dtype)))
        ok = False
    if result.shape != expected.shape:
        print(("shape", side, result.shape, expected.shape))
        ok = False
    if not np.array_equal(result, expected):
        mismatch = np.flatnonzero(result.ravel() != expected.ravel())[:10].tolist()
        print(("values", side, mismatch, result.ravel()[mismatch].tolist(), expected.ravel()[mismatch].tolist()))
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large f64 searchsorted parallel path should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn searchsorted_large_i64_array_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(124)
a = np.sort(rng.integers(-1_000_000, 1_000_000, (1 << 19) + 4099, dtype=np.int64))
v = rng.integers(-1_250_000, 1_250_000, 1025 * 513, dtype=np.int64).reshape(1025, 513)
flat = v.ravel()
flat[:2048] = a[rng.integers(0, a.size, 2048)]  # exact-match ties
flat[1] = np.iinfo(np.int64).min
flat[2] = np.iinfo(np.int64).max

ok = True
for side in ("left", "right"):
    result = fnp.searchsorted(a, v, side=side)
    expected = np.searchsorted(a, v, side=side)
    if result.dtype != expected.dtype:
        print(("dtype", side, str(result.dtype), str(expected.dtype)))
        ok = False
    if result.shape != expected.shape:
        print(("shape", side, result.shape, expected.shape))
        ok = False
    if not np.array_equal(result, expected):
        mismatch = np.flatnonzero(result.ravel() != expected.ravel())[:10].tolist()
        print(("values", side, mismatch, result.ravel()[mismatch].tolist(), expected.ravel()[mismatch].tolist()))
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large i64 searchsorted merge path should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn searchsorted_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.searchsorted(a, 3.5)
expected = np.searchsorted(a, 3.5)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "searchsorted scalar should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nonzero
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nonzero_python_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def nonzero_outcome(fn, value):
    try:
        result = fn(value)
        arrays = []
        for item in result:
            arr = np.asarray(item)
            arrays.append((str(arr.dtype), tuple(arr.shape), arr.tolist()))
        return ("ok", type(result).__name__, len(result), arrays)
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))

cases = [
    ("list fallback", lambda: [0, 2, 0, 3]),
    ("tuple fallback", lambda: (0, 0, 5, 0)),
    ("bool list", lambda: [False, True, False, True]),
    ("nested list", lambda: [[0, 1], [2, 0]]),
    ("scalar nonzero", lambda: 7),
    ("scalar zero", lambda: 0),
    ("zero-d ndarray nonzero", lambda: np.array(7)),
    ("zero-d ndarray zero", lambda: np.array(0)),
    ("bool ndarray", lambda: np.array([False, True, False, True], dtype=np.bool_)),
    ("uint16 ndarray", lambda: np.array([0, 4, 0, 5], dtype=np.uint16)),
    ("signed-zero nan float", lambda: np.array([-0.0, np.nan, 2.5, 0.0])),
    ("empty two-dimensional ndarray", lambda: np.zeros((0, 2), dtype=np.int64)),
    ("object truthiness", lambda: np.array(["", "x", "0"], dtype=object)),
    ("ragged list error", lambda: [[1], [0, 2]]),
]

ok = True
for label, factory in cases:
    actual = nonzero_outcome(fnp.nonzero, factory())
    expected = nonzero_outcome(np.nonzero, factory())
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
        "nonzero Python-container surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn nonzero_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 0, 2, 0, 3])
result = fnp.nonzero(a)
expected = np.nonzero(a)
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nonzero 1d should match numpy");
    Ok(())
}

#[test]
fn nonzero_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[0, 1, 0], [2, 0, 3]])
result = fnp.nonzero(a)
expected = np.nonzero(a)
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nonzero 2d should match numpy");
    Ok(())
}

#[test]
fn nonzero_all_zeros() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 0, 0])
result = fnp.nonzero(a)
expected = np.nonzero(a)
print(len(result) == len(expected) and result[0].size == 0 and expected[0].size == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nonzero all zeros should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// count_nonzero
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn count_nonzero_python_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def count_nonzero_outcome(fn, value, **kwargs):
    try:
        result = fn(value, **kwargs)
        arr = np.asarray(result)
        return ("ok", type(result).__name__, str(arr.dtype), tuple(arr.shape), arr.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))

cases = [
    ("list fallback", lambda: ([0, 2, 0, 3], {})),
    ("tuple fallback", lambda: ((0, 0, 5, 0), {})),
    ("bool list", lambda: ([False, True, False, True], {})),
    ("nested list", lambda: ([[0, 1], [2, 0]], {})),
    ("nested keepdims", lambda: ([[0, 1], [2, 0]], {"axis": 1, "keepdims": True})),
    ("scalar nonzero", lambda: (7, {})),
    ("scalar zero", lambda: (0, {})),
    ("zero-d ndarray nonzero", lambda: (np.array(7), {})),
    ("zero-d ndarray zero", lambda: (np.array(0), {})),
    ("bool ndarray", lambda: (np.array([False, True, False, True], dtype=np.bool_), {})),
    ("uint16 ndarray", lambda: (np.array([0, 4, 0, 5], dtype=np.uint16), {})),
    ("signed-zero nan float", lambda: (np.array([-0.0, np.nan, 2.5, 0.0]), {})),
    (
        "empty two-dimensional axis zero",
        lambda: (np.zeros((0, 2), dtype=np.int64), {"axis": 0}),
    ),
    ("object truthiness", lambda: (np.array(["", "x", "0"], dtype=object), {})),
    ("ragged list error", lambda: ([[1], [0, 2]], {})),
]

ok = True
for label, factory in cases:
    value, kwargs = factory()
    actual = count_nonzero_outcome(fnp.count_nonzero, value, **kwargs)
    value, kwargs = factory()
    expected = count_nonzero_outcome(np.count_nonzero, value, **kwargs)
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
        "count_nonzero Python-container surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn count_nonzero_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 0, 2, 0, 3])
result = fnp.count_nonzero(a)
expected = np.count_nonzero(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "count_nonzero 1d should match numpy");
    Ok(())
}

#[test]
fn count_nonzero_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[0, 1, 0], [2, 0, 3]])
result = fnp.count_nonzero(a)
expected = np.count_nonzero(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "count_nonzero 2d should match numpy");
    Ok(())
}

#[test]
fn count_nonzero_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[0, 1, 0], [2, 0, 3]])
result = fnp.count_nonzero(a, axis=0)
expected = np.count_nonzero(a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "count_nonzero axis should match numpy"
    );
    Ok(())
}

#[test]
fn count_nonzero_all_zeros() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 0, 0, 0])
result = fnp.count_nonzero(a)
expected = np.count_nonzero(a)
print(result == expected == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "count_nonzero all zeros should be 0");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sort_unique_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5])
# unique returns sorted unique values
unique_vals = fnp.unique(a)
# all unique values should be in sorted order
is_sorted = np.array_equal(unique_vals, fnp.sort(unique_vals))
print(is_sorted)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique should return sorted values");
    Ok(())
}

#[test]
fn nonzero_count_nonzero_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 0, 2, 0, 3, 0])
nz_indices = fnp.nonzero(a)[0]
count = fnp.count_nonzero(a)
print(len(nz_indices) == count)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "len(nonzero) should equal count_nonzero"
    );
    Ok(())
}

#[test]
fn searchsorted_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
v = np.float64(2.5)
fnp_result = fnp.searchsorted(a, v)
np_result = np.searchsorted(a, v)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "searchsorted scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sort_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3+1j, 1-1j, 2+2j], dtype=np.complex128)
fnp_result = fnp.sort_complex(a)
np_result = np.sort_complex(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sort complex should match numpy");
    Ok(())
}

#[test]
fn sort_complex_python_container_outcomes_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def sort_complex_outcome(call):
    try:
        result = call()
        arr = np.asarray(result)
        return ("ok", type(result).__name__, str(arr.dtype), tuple(arr.shape), repr(arr.tolist()))
    except Exception as exc:
        return ("err", type(exc).__name__)

cases = [
    ("real list", lambda module: module.sort_complex([3, 1, 2])),
    ("complex list", lambda module: module.sort_complex([3 + 1j, 1 - 1j, 2 + 2j])),
    ("2d list fallback", lambda module: module.sort_complex([[3, 1], [2, 4]])),
    ("scalar error", lambda module: module.sort_complex(3)),
    ("string error", lambda module: module.sort_complex(["b", "a"])),
]

ok = True
for label, factory in cases:
    actual = sort_complex_outcome(lambda: factory(fnp))
    expected = sort_complex_outcome(lambda: factory(np))
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
        "sort_complex Python-container outcomes should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn argsort_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3+1j, 1-1j, 2+2j], dtype=np.complex128)
fnp_result = fnp.argsort(a)
np_result = np.argsort(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "argsort complex should match numpy");
    Ok(())
}

#[test]
fn searchsorted_sorter_gather_route_matches_numpy() -> Result<(), String> {
    // sorter= route: gather a[sorter] once, then the shipped sorted-haystack fast
    // paths; indices identical by construction (integer outputs, no fp). Batteries:
    // f64 + int64 haystacks x side left/right, dense ties, NaN in haystack,
    // out-of-range sorter error parity, non-intp sorter delegate, below-gate.
    // Prints a coarse interleaved best-of-7 A/B for the ship record.
    let script = fnp_script(
        r#"
import time
verdicts = []
rng = np.random.default_rng(20260713)
n = 1_000_000
a = rng.standard_normal(n)
srt = np.argsort(a, kind="stable")
v = rng.standard_normal(n)
for side in ("left", "right"):
    r = fnp.searchsorted(a, v, side=side, sorter=srt)
    e = np.searchsorted(a, v, side=side, sorter=srt)
    if r.dtype != e.dtype or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL f64 side={side} bytes")
ai = rng.integers(-1000, 1000, n)
si = np.argsort(ai, kind="stable")
vi = rng.integers(-1200, 1200, n)
r, e = fnp.searchsorted(ai, vi, sorter=si), np.searchsorted(ai, vi, sorter=si)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL int64-ties bytes")
an = a.copy(); an[12345] = np.nan
sn = np.argsort(an, kind="stable")
r, e = fnp.searchsorted(an, v[:1000], sorter=sn), np.searchsorted(an, v[:1000], sorter=sn)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL nan-haystack bytes")
bad = srt.copy(); bad[7] = n + 5
fe = ne = None
try:
    fnp.searchsorted(a, v[:10], sorter=bad)
except Exception as ex:
    fe = type(ex).__name__
try:
    np.searchsorted(a, v[:10], sorter=bad)
except Exception as ex:
    ne = type(ex).__name__
if fe != ne:
    verdicts.append(f"FAIL oob-sorter error parity fnp={fe} np={ne}")
s32 = srt.astype(np.int32)
r, e = fnp.searchsorted(a, v[:1000], sorter=s32), np.searchsorted(a, v[:1000], sorter=s32)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL int32-sorter delegate bytes")
small = a[:512]
ss = np.argsort(small, kind="stable")
r, e = fnp.searchsorted(small, v[:100], sorter=ss), np.searchsorted(small, v[:100], sorter=ss)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL below-gate bytes")
def best(fn, reps=7):
    fn(); best_s = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best_s = min(best_s, time.perf_counter() - t0)
    return best_s * 1000
tn = best(lambda: np.searchsorted(a, v, sorter=srt))
tf = best(lambda: fnp.searchsorted(a, v, sorter=srt))
print(f"SEARCHSORTED_SORTER_COARSE_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces SEARCHSORTED_SORTER_COARSE_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "sorter= gather route must return numpy-identical indices incl error parity: {result}"
    );
    Ok(())
}

#[test]
fn f64_sort_signed_zero_mix_defers_byte_exact() -> Result<(), String> {
    // Inputs mixing -0.0 with +0.0 now DEFER in all four f64 value-sort
    // kernels: the zeros compare equal but differ in bytes, and numpy's
    // unstable tie arrangement is algorithm-specific (its sorted zero block
    // is NOT totally ordered by sign), so the native parallel sort must not
    // produce its own arrangement. Single-sign zeros still engage (ties are
    // byte-identical). Every row pins bytes against numpy.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(157)
verdicts = []
def mixed(n):
    a = rng.standard_normal(n)
    a[rng.random(n) < 0.01] = 0.0
    a[rng.random(n) < 0.01] = -0.0
    return a
a1 = mixed(4_000_000)
if fnp.sort(a1).tobytes() != np.sort(a1).tobytes():
    verdicts.append("FAIL flat mixed-zero")
a2 = mixed(4_194_304).reshape(2048, 2048)
if fnp.sort(a2, axis=1).tobytes() != np.sort(a2, axis=1).tobytes():
    verdicts.append("FAIL lastaxis mixed-zero")
if fnp.sort(a2, axis=0).tobytes() != np.sort(a2, axis=0).tobytes():
    verdicts.append("FAIL axis0 mixed-zero")
a3 = mixed(4_194_304).reshape(64, 256, 256)
if fnp.sort(a3, axis=1).tobytes() != np.sort(a3, axis=1).tobytes():
    verdicts.append("FAIL midaxis mixed-zero")
# single-sign zeros still engage natively and stay byte-exact
b = rng.standard_normal(4_000_000)
b[rng.random(4_000_000) < 0.02] = -0.0
if fnp.sort(b).tobytes() != np.sort(b).tobytes():
    verdicts.append("FAIL neg-zero-only flat")
c = rng.standard_normal(4_000_000)
c[rng.random(4_000_000) < 0.02] = 0.0
if fnp.sort(c).tobytes() != np.sort(c).tobytes():
    verdicts.append("FAIL pos-zero-only flat")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}");
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "f64 sort signed-zero handling must be bit-identical to numpy: {result}"
    );
    Ok(())
}
