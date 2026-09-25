//! Conformance tests for numpy I/O functions against NumPy oracle.
//!
//! Tests save/load, loadtxt, genfromtxt.

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

fn fnp_script(body: String) -> String {
    support::fnp_script_with("from io import BytesIO, StringIO\n", false, body)
}

// ─────────────────────────────────────────────────────────────────────────────
// save / load
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn save_load_bytesio_roundtrip_matches_numpy_float64() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array([[1.25, -2.5], [3.75, 4.5]], dtype=np.float64)
buf = BytesIO()
fnp.save(buf, arr)
payload = buf.getvalue()
loaded = fnp.load(BytesIO(payload))
expected_buf = BytesIO()
np.save(expected_buf, arr)
expected = np.load(BytesIO(expected_buf.getvalue()))
print(
    payload.startswith(b"\x93NUMPY")
    and np.array_equal(loaded, expected)
    and loaded.shape == expected.shape
    and loaded.dtype == expected.dtype
)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "save/load BytesIO float64 roundtrip should match numpy"
    );
    Ok(())
}

/// Bead rc0923 .22. For every dtype x {C, F} x {0-d, 1-d, empty, 3-d}:
/// (a) fnp.save bytes == numpy.save bytes;
/// (b) fnp.load(numpy's bytes) has numpy's values, dtype, shape and contiguity;
/// (c) savez members are byte-identical to numpy.savez members (the zip container carries
///     timestamps, so members are compared, not archives).
/// Before this, fnp.load returned PERMUTED values for any Fortran-order float file with ndim >= 2,
/// and fnp.save wrote F-ordered and big-endian float arrays with a different header.
#[test]
fn save_load_savez_are_byte_identical_to_numpy_across_dtypes_orders_and_shapes()
-> Result<(), String> {
    let script = fnp_script(
        r#"
import zipfile
dtypes = ["?", "i1", "u1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f2", "<f4", "<f8",
          "<c8", "<c16", ">i4", ">u8", ">f2", ">f4", ">f8", ">c16", "S5", "<U3", "<M8[s]", "<m8[ms]"]
rng = np.random.default_rng(22)
def make(dt, shape, order):
    d = np.dtype(dt)
    size = int(np.prod(shape)) if shape else 1
    if d.kind in "iu":
        vals = rng.integers(0, 100, size).astype(d)
    elif d.kind in "fc":
        vals = (rng.standard_normal(size) * 10).astype(d)
    elif d.kind == "b":
        vals = rng.integers(0, 2, size).astype(d)
    elif d.kind == "S":
        vals = np.array([b"ab", b"xyz12", b""] * size, dtype=d)[:size]
    elif d.kind == "U":
        vals = np.array(["ab", "éx", ""] * size, dtype=d)[:size]
    else:
        vals = rng.integers(0, 10**6, size).astype(d)
    arr = vals.reshape(shape) if shape else vals.reshape(())
    return np.asfortranarray(arr) if order == "F" else np.ascontiguousarray(arr)
def members(payload):
    with zipfile.ZipFile(BytesIO(payload)) as z:
        return {name: z.read(name) for name in z.namelist()}
bad, cells, fortran_files = [], 0, 0
for dt in dtypes:
    for shape in [(), (5,), (0,), (2, 3, 4)]:
        for order in "CF":
            arr = make(dt, shape, order)
            cells += 1
            theirs = BytesIO(); np.save(theirs, arr); theirs = theirs.getvalue()
            ours = BytesIO(); fnp.save(ours, arr); ours = ours.getvalue()
            fortran_files += b"'fortran_order': True" in theirs[:128]
            if ours != theirs:
                off = next((i for i, (x, y) in enumerate(zip(ours, theirs)) if x != y), min(len(ours), len(theirs)))
                bad.append(f"save {np.dtype(dt)} {shape} {order}: first differing byte {off}")
            got, want = fnp.load(BytesIO(theirs)), np.load(BytesIO(theirs))
            if (got.dtype != want.dtype or got.shape != want.shape or got.tobytes() != want.tobytes()
                    or got.flags.f_contiguous != want.flags.f_contiguous
                    or got.flags.c_contiguous != want.flags.c_contiguous):
                bad.append(f"load {np.dtype(dt)} {shape} {order}: values/dtype/layout differ")
            za, zb = BytesIO(), BytesIO()
            fnp.savez(za, a=arr, b=arr); np.savez(zb, a=arr, b=arr)
            if members(za.getvalue()) != members(zb.getvalue()):
                bad.append(f"savez {np.dtype(dt)} {shape} {order}: members differ")
print(cells, fortran_files, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(3, ' ');
    let cells: usize = fields.next().unwrap_or("").parse().unwrap_or(0);
    let fortran_files: usize = fields.next().unwrap_or("").parse().unwrap_or(0);
    assert_eq!(cells, 24 * 4 * 2, "cell table drifted: {result}");
    // Negative control: the grid must contain Fortran-order files, where a loader that ignores
    // `fortran_order` returns permuted values.
    assert!(fortran_files >= 20, "too few Fortran-order files: {result}");
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "npy IO differs from numpy: {result}"
    );
    Ok(())
}

/// Bead rc0923 .22, the half the dtype grid above cannot reach: every NPY header VERSION numpy
/// writes (1.0; 2.0 for a header past 65535 bytes - a 3000-field record; 3.0 for UTF-8 field
/// names), structured and nested dtypes, savez_compressed members (names, order, decompressed
/// bytes; the deflate container itself may differ), the save -> load -> save fixed point, numpy's
/// max_header_size refusal, files forced to each version by np.lib.format.write_array, and the
/// object-array pickle policy in both directions. The byte comparison is the whole contract: a
/// scratch fnp.save with ONE header-padding byte changed failed all 192 cells of the grid above.
#[test]
fn npy_header_versions_structured_dtypes_and_compressed_npz_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import zipfile, warnings
warnings.simplefilter("ignore", UserWarning)  # "Stored array in format 3.0", from both writers
def members(payload):
    with zipfile.ZipFile(BytesIO(payload)) as z:
        return [(name, z.read(name)) for name in z.namelist()]
def outcome(fn):
    try:
        return "ok", fn()
    except Exception as ex:
        return type(ex).__name__, None
def loaded(arr):
    return str(arr.dtype), arr.shape, arr.tobytes()
def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
long_names = [(f"field_{i:05d}", "<f8") for i in range(3000)]
utf8_names = [("温度", "<f4"), ("ß", "<i2")]
cases = {
    "f8 3-d C": np.arange(24.0).reshape(2, 3, 4),
    "i2 2-d F": np.asfortranarray(np.arange(12, dtype="<i2").reshape(3, 4)),
    "U4 1-d": np.array(["ab", "éxyz", ""], dtype="<U4"),
    "struct C": np.array([(1, 2.5), (3, -1.0)], dtype=[("a", "<i4"), ("b", "<f8")]),
    "struct nested": np.zeros(3, dtype=[("p", [("x", "<f4"), ("y", "<f4")]), ("id", "<u2", (2,))]),
    "struct long header (v2.0)": np.zeros(2, dtype=long_names),
    "struct utf8 names (v3.0)": np.zeros(2, dtype=utf8_names),
    "f8 0-d": np.array(3.5),
    "c16 empty 2-d": np.empty((0, 3), dtype="<c16"),
}
bad, cells, versions = [], 0, set()
for label, arr in cases.items():
    cells += 1
    theirs = BytesIO(); np.save(theirs, arr); theirs = theirs.getvalue()
    versions.add(theirs[6:8])
    ours = BytesIO(); fnp.save(ours, arr); ours = ours.getvalue()
    if ours != theirs:
        bad.append(f"save {label}: first differing byte {first_diff(ours, theirs)} (v{theirs[6]}.{theirs[7]})")
    # numpy refuses a header over max_header_size (10000) unless told otherwise: the same
    # refusal, then the same values once allowed.
    for kw in ({}, {"max_header_size": 200000}):
        s = outcome(lambda: loaded(np.load(BytesIO(theirs), **kw)))
        r = outcome(lambda: loaded(fnp.load(BytesIO(theirs), **kw)))
        if s != r:
            bad.append(f"load {label} {kw}: fnp={r[0]} numpy={s[0]}")
    again = BytesIO(); fnp.save(again, fnp.load(BytesIO(ours), max_header_size=200000)); again = again.getvalue()
    if again != ours:
        bad.append(f"save-load-save {label}: first differing byte {first_diff(again, ours)}")
    zc_ours, zc_theirs = BytesIO(), BytesIO()
    fnp.savez_compressed(zc_ours, first=arr, second=arr); np.savez_compressed(zc_theirs, first=arr, second=arr)
    if members(zc_ours.getvalue()) != members(zc_theirs.getvalue()):
        bad.append(f"savez_compressed {label}: member names/order/bytes differ")
for version in [(1, 0), (2, 0), (3, 0)]:
    for label in ["f8 3-d C", "i2 2-d F", "U4 1-d", "struct C"]:
        cells += 1
        buf = BytesIO(); np.lib.format.write_array(buf, cases[label], version=version)
        s = outcome(lambda: loaded(np.load(BytesIO(buf.getvalue()))))
        r = outcome(lambda: loaded(fnp.load(BytesIO(buf.getvalue()))))
        if s != r:
            bad.append(f"load v{version} {label}: fnp={r[0]} numpy={s[0]}")
obj = np.array([{"k": 1}, [2, 3], "s"], dtype=object)
for allow in (False, True):
    cells += 1
    theirs = BytesIO(); np.save(theirs, obj, allow_pickle=True); theirs = theirs.getvalue()
    s = outcome(lambda: np.load(BytesIO(theirs), allow_pickle=allow).tolist())
    r = outcome(lambda: fnp.load(BytesIO(theirs), allow_pickle=allow).tolist())
    if s != r:
        bad.append(f"load object allow_pickle={allow}: fnp={r[0]} numpy={s[0]}")
    s = outcome(lambda: (np.save(BytesIO(), obj, allow_pickle=allow), "saved")[1])
    r = outcome(lambda: (fnp.save(BytesIO(), obj, allow_pickle=allow), "saved")[1])
    if s != r:
        bad.append(f"save object allow_pickle={allow}: fnp={r[0]} numpy={s[0]}")
print(cells, "|".join(sorted(v.hex() for v in versions)), bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(3, ' ');
    assert_eq!(fields.next(), Some("23"), "cell table drifted: {result}");
    // Negative control on the grid itself: numpy must really have written all three versions,
    // or the version cells compare nothing.
    assert_eq!(
        fields.next(),
        Some("0100|0200|0300"),
        "numpy no longer writes versions 1.0, 2.0 and 3.0 here: {result}"
    );
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "npy/npz IO differs from numpy: {result}"
    );
    Ok(())
}

#[test]
fn load_numpy_saved_bytesio_float32_preserves_shape_dtype_and_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)
buf = BytesIO()
np.save(buf, arr)
loaded = fnp.load(BytesIO(buf.getvalue()))
print(
    np.array_equal(loaded, arr)
    and loaded.shape == arr.shape
    and loaded.dtype == arr.dtype
)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "load should preserve numpy-saved float32 NPY payloads"
    );
    Ok(())
}

#[test]
fn savez_bytesio_positional_and_keyword_arrays_match_numpy_npz() -> Result<(), String> {
    let script = fnp_script(
        r#"
import zipfile
first = np.array([[1.25, -2.5], [3.75, 4.5]], dtype=np.float64)
named = np.array([5.5, 6.5, 7.5], dtype=np.float32)
buf = BytesIO()
result = fnp.savez(buf, first, named=named)
payload = buf.getvalue()
with zipfile.ZipFile(BytesIO(payload)) as archive:
    methods = {info.filename: info.compress_type for info in archive.infolist()}
loaded = np.load(BytesIO(payload))
print(
    result is None
    and loaded.files == ["arr_0", "named"]
    and methods == {"arr_0.npy": zipfile.ZIP_STORED, "named.npy": zipfile.ZIP_STORED}
    and np.array_equal(loaded["arr_0"], first)
    and np.array_equal(loaded["named"], named)
    and loaded["arr_0"].dtype == first.dtype
    and loaded["named"].dtype == named.dtype
)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "savez BytesIO archive should match NumPy NPZ names, dtypes, and values"
    );
    Ok(())
}

#[test]
fn savez_compressed_bytesio_writes_deflated_numpy_loadable_npz() -> Result<(), String> {
    let script = fnp_script(
        r#"
import zipfile
first = np.arange(12, dtype=np.float64).reshape(3, 4)
second = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
buf = BytesIO()
result = fnp.savez_compressed(buf, first, second=second)
payload = buf.getvalue()
with zipfile.ZipFile(BytesIO(payload)) as archive:
    methods = {info.filename: info.compress_type for info in archive.infolist()}
loaded = np.load(BytesIO(payload))
print(
    result is None
    and loaded.files == ["arr_0", "second"]
    and methods == {"arr_0.npy": zipfile.ZIP_DEFLATED, "second.npy": zipfile.ZIP_DEFLATED}
    and np.array_equal(loaded["arr_0"], first)
    and np.array_equal(loaded["second"], second)
    and loaded["arr_0"].dtype == first.dtype
    and loaded["second"].dtype == second.dtype
)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "savez_compressed BytesIO archive should be deflated and NumPy-loadable"
    );
    Ok(())
}

#[test]
fn savez_path_appends_npz_suffix_and_roundtrips_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "archive"
    expected_path = path.with_suffix(".npz")
    data = np.array([1.0, 2.5, 4.0], dtype=np.float64)
    result = fnp.savez(path, data=data)
    loaded = np.load(expected_path)
    print(
        result is None
        and expected_path.exists()
        and loaded.files == ["data"]
        and np.array_equal(loaded["data"], data)
        and loaded["data"].dtype == data.dtype
    )
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "savez path wrapper should append .npz and roundtrip through NumPy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// loadtxt
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn loadtxt_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1 2 3\n4 5 6\n7 8 9"
result = fnp.loadtxt(StringIO(data))
expected = np.loadtxt(StringIO(data))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "loadtxt basic should match numpy");
    Ok(())
}

#[test]
fn loadtxt_with_delimiter() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1,2,3\n4,5,6\n7,8,9"
result = fnp.loadtxt(StringIO(data), delimiter=',')
expected = np.loadtxt(StringIO(data), delimiter=',')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "loadtxt with delimiter should match numpy"
    );
    Ok(())
}

#[test]
fn loadtxt_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1 2 3\n4 5 6"
result = fnp.loadtxt(StringIO(data), dtype='int32')
expected = np.loadtxt(StringIO(data), dtype='int32')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "loadtxt with dtype should match numpy"
    );
    Ok(())
}

#[test]
fn loadtxt_with_skiprows() -> Result<(), String> {
    let script = fnp_script(
        r##"
data = "# header\n1 2 3\n4 5 6"
result = fnp.loadtxt(StringIO(data), skiprows=1)
expected = np.loadtxt(StringIO(data), skiprows=1)
print(np.array_equal(result, expected))
"##
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "loadtxt with skiprows should match numpy"
    );
    Ok(())
}

#[test]
fn loadtxt_with_usecols() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1 2 3 4\n5 6 7 8"
result = fnp.loadtxt(StringIO(data), usecols=(0, 2))
expected = np.loadtxt(StringIO(data), usecols=(0, 2))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "loadtxt with usecols should match numpy"
    );
    Ok(())
}

#[test]
fn loadtxt_with_comments() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "% comment\n1 2 3\n4 5 6"
result = fnp.loadtxt(StringIO(data), comments='%')
expected = np.loadtxt(StringIO(data), comments='%')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "loadtxt with comments should match numpy"
    );
    Ok(())
}

#[test]
fn loadtxt_unpack() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1 2 3\n4 5 6"
result = fnp.loadtxt(StringIO(data), unpack=True)
expected = np.loadtxt(StringIO(data), unpack=True)
match = all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "loadtxt unpack should match numpy");
    Ok(())
}

#[test]
fn loadtxt_max_rows() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1 2 3\n4 5 6\n7 8 9"
result = fnp.loadtxt(StringIO(data), max_rows=2)
expected = np.loadtxt(StringIO(data), max_rows=2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "loadtxt max_rows should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// genfromtxt
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn genfromtxt_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1 2 3\n4 5 6\n7 8 9"
result = fnp.genfromtxt(StringIO(data))
expected = np.genfromtxt(StringIO(data))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "genfromtxt basic should match numpy");
    Ok(())
}

#[test]
fn genfromtxt_with_delimiter() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1,2,3\n4,5,6"
result = fnp.genfromtxt(StringIO(data), delimiter=',')
expected = np.genfromtxt(StringIO(data), delimiter=',')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "genfromtxt with delimiter should match numpy"
    );
    Ok(())
}

#[test]
fn genfromtxt_with_missing() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1,2,3\n4,,6"
result = fnp.genfromtxt(StringIO(data), delimiter=',', filling_values=0)
expected = np.genfromtxt(StringIO(data), delimiter=',', filling_values=0)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "genfromtxt with missing should match numpy"
    );
    Ok(())
}

#[test]
fn genfromtxt_with_skip_header() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "col1,col2,col3\n1,2,3\n4,5,6"
result = fnp.genfromtxt(StringIO(data), delimiter=',', skip_header=1)
expected = np.genfromtxt(StringIO(data), delimiter=',', skip_header=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "genfromtxt with skip_header should match numpy"
    );
    Ok(())
}

#[test]
fn genfromtxt_negative_usecols_last_column() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1,2,3,4\n5,6,7,8"
result = fnp.genfromtxt(StringIO(data), delimiter=',', usecols=-1)
expected = np.genfromtxt(StringIO(data), delimiter=',', usecols=-1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "genfromtxt with usecols=-1 should select last column"
    );
    Ok(())
}

#[test]
fn genfromtxt_negative_usecols_mixed_order() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1,2,3,4\n5,6,7,8"
result = fnp.genfromtxt(StringIO(data), delimiter=',', usecols=(-1, 0))
expected = np.genfromtxt(StringIO(data), delimiter=',', usecols=(-1, 0))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "genfromtxt with usecols=(-1, 0) should select last then first"
    );
    Ok(())
}

#[test]
fn genfromtxt_negative_usecols_third_from_last() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1,2,3,4\n5,6,7,8"
result = fnp.genfromtxt(StringIO(data), delimiter=',', usecols=-3)
expected = np.genfromtxt(StringIO(data), delimiter=',', usecols=-3)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "genfromtxt with usecols=-3 should select third from last"
    );
    Ok(())
}

// Every other save test in this file writes to a BytesIO, which has .write and
// so takes fnp's native writer. A STRING PATH does not, and takes the numpy
// fallback - which used to forward fix_imports unconditionally and therefore
// raised TypeError on numpy >= 2.4, where save dropped that parameter. This
// covers the path-string form for its own sake, and asserts fix_imports=
// produces whatever the INSTALLED numpy produces (accepted on <= 2.3, TypeError
// on 2.4+) rather than pinning one build's verdict. np.load is deliberately
// exercised with fix_imports= too, because numpy did NOT remove it there and a
// blanket "strip fix_imports" fix would have broken load.
#[test]
fn save_path_string_roundtrips_and_tracks_numpys_fix_imports_contract() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform
import os
import tempfile

values = np.array([1.5, -2.0, 3.25, 4.0], dtype=np.float64)

def save_path_string(module, path):
    module.save(path, values)
    return np.load(path).tolist()

def save_path_string_allow_pickle(module, path):
    module.save(path, values, allow_pickle=True)
    return np.load(path).tolist()

def save_path_no_suffix(module, path):
    # numpy appends .npy when the name lacks it; the fallback must keep that.
    stem = path[:-4]
    module.save(stem, values)
    return (os.path.exists(stem + ".npy"), np.load(stem + ".npy").tolist())

def save_file_object(module, path):
    with open(path, "wb") as handle:
        module.save(handle, values)
    return np.load(path).tolist()

def save_int_array(module, path):
    module.save(path, np.array([[1, 2], [3, 4]], dtype=np.int32))
    restored = np.load(path)
    return (str(restored.dtype), restored.tolist())

def save_fix_imports_true(module, path):
    module.save(path, values, fix_imports=True)
    return np.load(path).tolist()

def save_fix_imports_false(module, path):
    module.save(path, values, fix_imports=False)
    return np.load(path).tolist()

def load_fix_imports(module, path):
    np.save(path, values)
    return module.load(path, fix_imports=True).tolist()

cases = [
    ("save(path_string)", save_path_string),
    ("save(path_string, allow_pickle=True)", save_path_string_allow_pickle),
    ("save(name without .npy)", save_path_no_suffix),
    ("save(file object)", save_file_object),
    ("save int32 via path", save_int_array),
    ("save fix_imports=True", save_fix_imports_true),
    ("save fix_imports=False", save_fix_imports_false),
    ("load fix_imports=True", load_fix_imports),
]

def outcome(module, call):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "data.npy")
        try:
            return ("ok", call(module, path))
        except Exception as exc:
            return ("err", type(exc).__name__)

ok = True
for label, call in cases:
    actual = outcome(fnp, call)
    expected = outcome(np, call)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False

# The regression itself, stated independently of numpy: a plain path save must
# simply work.
try:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "plain.npy")
        fnp.save(path, values)
        if not np.array_equal(np.load(path), values):
            print("fnp.save(path) round-trip lost values")
            ok = False
except Exception as exc:
    print(f"fnp.save(path) raised {type(exc).__name__}: {exc}")
    ok = False

print(ok)
print("oracle", platform.node(), np.__version__)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "save path-string and fix_imports contract should match numpy ({provenance}): {result}"
    );
    Ok(())
}

#[test]
fn loadtxt_bool_usecols_prefix_bounded_tokenise_matches_numpy() -> Result<(), String> {
    // The bool + positive-usecols path tokenises only as far as the farthest
    // SELECTED column instead of splitting every field of every row. The whole
    // risk of that change is that the truncated token buffer alters an
    // observable outcome, so this sweeps the cases where it could, against live
    // numpy, and every case is one a naive `.take(n)` would get wrong:
    //
    //  - garbage AFTER the last selected column must stay ignored (that is the
    //    point of the optimisation), and garbage BEFORE it must stay ignored
    //    too (it is tokenised but never parsed);
    //  - a selection that INCLUDES the last column must be unchanged, since it
    //    is the no-op case that proves the budget is computed from max(cols)
    //    and not from cols.len();
    //  - out-of-order and DUPLICATED selections must keep numpy's output order,
    //    which is the selection order, not sorted column order;
    //  - a row too SHORT to reach a requested column must still raise exactly
    //    as numpy does. This is the case a budget-off-by-one silently breaks:
    //    with take(max_col) instead of take(max_col+1) the last selected column
    //    disappears from the buffer and every well-formed row starts raising.
    let script = fnp_script(
        r#"
import platform

ROWS = [
    "1,0,1,0,1,1,0,not_a_bool,1,0,1,1,0,1,0,1",
    "0,1,1,1,0,0,1,not_a_bool,0,1,0,0,1,0,1,0",
    "1,1,0,0,1,0,1,not_a_bool,1,1,1,0,0,1,1,0",
]
WIDE = "\n".join(ROWS) + "\n"
SHORT = "1,0,1\n0,1,1\n"          # only 3 columns
LEADING_GARBAGE = "\n".join(
    r.replace("1,0,1,0", "1,0,x,0", 1) if i == 0 else r for i, r in enumerate(ROWS)
) + "\n"

CASES = [
    ("selected_before_garbage",      WIDE,            [0, 1, 3, 4]),
    ("selection_includes_last_col",  WIDE,            [0, 15]),
    ("only_last_col",                WIDE,            [15]),
    ("only_first_col",               WIDE,            [0]),
    ("out_of_order",                 WIDE,            [4, 1, 0, 3]),
    ("duplicated",                   WIDE,            [2, 2, 0]),
    ("spans_the_garbage_column",     WIDE,            [0, 8]),
    ("garbage_is_selected",          WIDE,            [7]),
    ("row_too_short",                SHORT,           [0, 5]),
    ("short_exact_last",             SHORT,           [2]),
    ("unselected_garbage_before",    LEADING_GARBAGE, [0, 1, 4]),
]

def outcome(fn, text, cols):
    try:
        out = fn(StringIO(text), delimiter=",", dtype=np.bool_, usecols=cols)
        arr = np.asarray(out)
        return "ok:%s:%s:%s" % (arr.shape, arr.dtype, arr.tobytes().hex())
    except Exception as exc:
        return "raise:" + type(exc).__name__

failures = []
for label, text, cols in CASES:
    got = outcome(fnp.loadtxt, text, cols)
    want = outcome(np.loadtxt, text, cols)
    if got != want:
        failures.append("%s usecols=%r: fnp %s vs numpy %s" % (label, cols, got, want))

# Guard against the grid quietly losing its negative cases: at least one case
# must raise on BOTH sides, otherwise the short-row arm is untested and this
# test would pass while only exercising the happy path.
raised = sum(1 for label, text, cols in CASES
             if outcome(np.loadtxt, text, cols).startswith("raise:"))
if raised == 0:
    failures.append("no case made numpy raise; the short-row arm is untested")

if failures:
    print("FAILURES\n" + "\n".join(failures))
else:
    print("True")
print("oracle", platform.node(), np.__version__)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "prefix-bounded bool usecols tokenise must be observationally identical to \
         numpy ({provenance}):\n{result}"
    );
    Ok(())
}

/// numpy's text readers take `comments` as str, bytes, a sequence of str, or None (no
/// comment character), and genfromtxt's `delimiter` as an int or a tuple of field widths.
/// fnp typed them `&str`, so every other form was a PyO3 TypeError before the numpy delegate
/// could see it, and a non-integral `ndmin` raised TypeError where numpy raises ValueError
/// (numpy's own TestLoadTxt / TestFromTxt). The plain-str fast path must still match.
#[test]
fn text_readers_accept_numpys_comments_delimiter_and_ndmin_forms() -> Result<(), String> {
    // `r##` because the Python below contains `"#` (a string opening with a comment char).
    let script = fnp_script(
        r##"
import io
def outcome(fn):
    try:
        r = fn()
        return ("ok", str(r.dtype), r.shape, r.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__)
S = io.StringIO
cases = [
    lambda m: m.loadtxt(S("1 2\n3 4\n"), comments=None),
    lambda m: m.loadtxt(S("# c\n1,2,3,5\n"), dtype=int, delimiter=",", comments=b"#"),
    lambda m: m.loadtxt(S("# c\n@ d\n// e\n1,2\n"), delimiter=",", comments=["#", "@", "//"]),
    lambda m: m.loadtxt(S("1 2\n3 4\n"), ndmin=1.5),
    lambda m: m.loadtxt(S("1 2\n3 4\n"), ndmin=2),
    lambda m: m.loadtxt(S("# c\n1,2\n3,4\n"), delimiter=","),
    lambda m: m.genfromtxt(S("  1  2  3\n  4  5 67\n890123  4"), delimiter=3),
    lambda m: m.genfromtxt(S("  1  2  3\n  4  5 67\n890123  4"), delimiter=(3, 3, 4)),
    lambda m: m.genfromtxt(S("1 2\n3 4\n"), comments=None),
    lambda m: m.genfromtxt(S("# c\n1,2\n3,4\n"), delimiter=",", dtype=float),
]
bad = [i for i, c in enumerate(cases) if outcome(lambda: c(fnp)) != outcome(lambda: c(np))]
print(bad if bad else True)
"##
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "loadtxt/genfromtxt parameter forms must match numpy: {result}"
    );
    Ok(())
}

/// Binary/text readers must accept every dtype numpy reads and never answer with a clamped
/// value: `fromfile` raised "buffer parsing: unsupported dtype complex128" (and an object-dtype
/// error) for files numpy reads, and `fromstring("18446744073709551615", dtype=uint64, sep=" ")`
/// SATURATED to 9223372036854775807 (numpy's own TestIO under the drop-in harness). The text
/// decoder also accepted what numpy's integer scan rejects - a float-looking token ("2.0",
/// "1e3") and any sign on an unsigned dtype ("-1" wrapped to u32::MAX) - on both
/// `fromstring` and `fromfile(sep=...)`.
#[test]
fn fromfile_and_fromstring_take_numpys_dtypes_without_clamping() -> Result<(), String> {
    let script = fnp_script(
        r#"
import os, tempfile
def outcome(fn):
    try:
        r = fn()
        return ("ok", str(r.dtype), np.shape(r), r.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__)
d = tempfile.mkdtemp()
path = os.path.join(d, "c.bin")
np.array([1 + 2j, 3 - 4j]).tofile(path)
text_path = os.path.join(d, "t.txt")
with open(text_path, "w") as fh:
    fh.write("1 2 3")
float_text_path = os.path.join(d, "f.txt")
with open(float_text_path, "w") as fh:
    fh.write("1 2.0 3")
cases = [
    lambda m: m.fromfile(path, dtype=np.complex128),
    lambda m: m.fromfile(path, dtype=np.complex64),
    lambda m: m.fromfile(path, dtype=np.float64),
    lambda m: m.fromfile(text_path, dtype=object, sep=" "),
    lambda m: m.fromfile(text_path, dtype=np.int64, sep=" "),
    lambda m: m.fromstring("18446744073709551615 1", dtype=np.uint64, sep=" "),
    lambda m: m.fromstring("9223372036854775807 -3", dtype=np.int64, sep=" "),
    lambda m: m.fromstring("1 2.0 3", dtype=np.int32, sep=" "),
    lambda m: m.fromstring("7 1e3", dtype=np.int64, sep=" "),
    lambda m: m.fromstring("-1", dtype=np.uint32, sep=" "),
    lambda m: m.fromstring("+5", dtype=np.uint8, sep=" "),
    lambda m: m.fromstring("-9223372036854775809", dtype=np.int64, sep=" "),
    lambda m: m.fromstring("18446744073709551616", dtype=np.uint64, sep=" "),
    lambda m: m.fromstring("300 -1", dtype=np.int8, sep=" "),
    lambda m: m.fromstring("+5 -0 00012", dtype=np.int64, sep=" "),
    lambda m: m.fromstring("1, 0.0, 2.5", dtype=bool, sep=","),
    lambda m: m.fromfile(float_text_path, dtype=np.int64, sep=" "),
    lambda m: m.fromfile(float_text_path, dtype=np.float32, sep=" "),
]
bad = [i for i, c in enumerate(cases) if outcome(lambda: c(fnp)) != outcome(lambda: c(np))]
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "fromfile/fromstring dtypes must match numpy: {result}"
    );
    Ok(())
}

/// numpy's genfromtxt defaults to `dtype=float`, and an EXPLICIT `dtype=None` detects each
/// column's type. fnp's defaulted `Option` merged the two and forwarded nothing for None, so
/// `genfromtxt(f, dtype=None)` returned float64 with NaN in every text column (numpy's own
/// TestFromTxt: 20 tests, found through the drop-in harness, bead rc0923 .8). Controls: an omitted
/// dtype (float) and an explicit float keep matching.
#[test]
fn genfromtxt_explicit_dtype_none_detects_column_types_like_numpy() -> Result<(), String> {
    // `r##` because the script itself contains `"#` (a commented header line).
    let script = fnp_script(
        r##"
import warnings
warnings.simplefilter("ignore")
def o(f):
    try:
        r = f()
        return ("ok", str(r.dtype), r.shape, r.tobytes() if r.dtype.kind != "O" else repr(r))
    except Exception as e:
        return (type(e).__name__,)
header = "gender age weight\nM 64.0 75.0\nF 25.0 60.0"
cases = {
    "names header": lambda m: m.genfromtxt(StringIO(header), dtype=None, names=True, encoding=None),
    "auto dtype": lambda m: m.genfromtxt(StringIO("A 64 75.0 3+4j True\nBCD 25 60.0 5+6j False"), dtype=None, encoding=None),
    "commented header": lambda m: m.genfromtxt(StringIO("# gender age weight\nM 21 72.1\nF 35 58.33"), dtype=None, names=True, encoding=None),
    "usecols names": lambda m: m.genfromtxt(StringIO("1 2 3\n4 5 6"), usecols=(0, 2), names="a, b", dtype=None),
    "autostrip": lambda m: m.genfromtxt(StringIO("01/01/2003  , 1.3,   abcde"), delimiter=",", dtype=None, autostrip=True, encoding=None),
    "numeric ints": lambda m: m.genfromtxt(StringIO("1 2 3\n4 5 6"), dtype=None),
    "mixed": lambda m: m.genfromtxt(StringIO("1 a 3.5\n4 b 6.5"), dtype=None, encoding=None),
    "omitted dtype (control)": lambda m: m.genfromtxt(StringIO("1 2 3\n4 5 6")),
    "explicit float (control)": lambda m: m.genfromtxt(StringIO("1 2 3\n4 5 6"), dtype=float),
}
bad = [k for k, f in cases.items() if o(lambda: f(np)) != o(lambda: f(fnp))]
print(len(cases), bad)
"##
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "9 []",
        "genfromtxt dtype=None must match numpy: {result}"
    );
    Ok(())
}

/// `loadtxt` across 28 inputs x {StringIO, path} x {None, ' ', '\t', ','} delimiters plus
/// BytesIO, a list of lines, int dtype, usecols, unpack and skiprows (392 cells), compared by
/// dtype, shape and bytes or exception type and message. Three native defects:
/// - numpy SQUEEZES every size-1 axis (`_ensure_ndmin_ndarray`), so a one-row file is 1-D and
///   a single value 0-d; the native arms squeezed only a single column (`loadtxt("1 2")` was
///   (1, 2), `loadtxt("5")` was (1,));
/// - an EXPLICIT whitespace delimiter is literal in numpy (`delimiter='\t'` does not split on
///   spaces, doubled spaces under `delimiter=' '` are empty fields numpy rejects); the native
///   splitters treated both as any-whitespace;
/// - numpy opens a PATH in universal-newline mode (`\r` ends a line) and rejects a `\r` inside
///   a line read from a file-like; the native reader did neither.
///
/// 106 of the 392 cells failed before the fix (numpy 2.4.3); 0 fail after, on numpy 2.4.3 and
/// 2.3.5.
#[test]
fn loadtxt_shapes_delimiters_and_newlines_match_numpy() -> Result<(), String> {
    // `r##` because the body contains `"#`.
    let script = fnp_script(
        r##"
import io, os, tempfile, warnings

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            a = np.asarray(r)
            got = ("ok", a.dtype.str, a.shape, a.tobytes())
        except Exception as ex:
            got = (type(ex).__name__, str(ex)[:120])
    return got + (sorted({w.category.__name__ for w in caught}),)

texts = {
    "plain": "1 2\n3 4\n",
    "one row": "1 2\n",
    "one row no nl": "1 2",
    "one col": "1\n2\n3\n",
    "single value": "5\n",
    "CR": "1 21\r3 42\r",
    "CRLF": "1 21\r\n3 42\r\n",
    "mid CR": "1 2\r 3\n4 5 6\n",
    "tabs": "1\t2\n3\t4\n",
    "mixed ws": "1 \t 2\n3\t\t4\n",
    "double space": "1  2\n3  4\n",
    "leading ws": "  1 2\n  3 4\n",
    "trailing ws": "1 2  \n3 4\t\n",
    "commas": "1,2\n3,4\n",
    "commas spaces": "1, 2\n3 ,4\n",
    "empty field comma": "1,,2\n3,4,5\n",
    "comment": "# head\n1 2 # tail\n3 4\n",
    "blank lines": "\n1 2\n\n3 4\n\n",
    "vt ff": "1\x0b2\n3\x0c4\n",
    "x1c sep": "1\x1c2\n3\x1c4\n",
    "nbsp": "1 2\n3 4\n",
    "ragged": "1 2\n3\n",
    "empty": "",
    "only comments": "# a\n# b\n",
    "sci": "1e3 -2.5E-1\ninf nan\n",
    "hex": "0x10 1\n",
    "plus": "+1 -0\n",
    "underscore": "1_000 2\n",
}

def sweep(tmpdir):
    cases = {}
    for tname, text in texts.items():
        p = os.path.join(tmpdir, tname.replace(" ", "_") + ".txt")
        with open(p, "w", newline="") as f:
            f.write(text)
        for delim in (None, " ", "\t", ","):
            cases[f"{tname} StringIO delim={delim!r}"] = lambda m, t=text, d=delim: m.loadtxt(io.StringIO(t), delimiter=d)
            cases[f"{tname} path delim={delim!r}"] = lambda m, p=p, d=delim: m.loadtxt(p, delimiter=d)
        cases[f"{tname} BytesIO"] = lambda m, t=text: m.loadtxt(io.BytesIO(t.encode("utf-8")))
        cases[f"{tname} list"] = lambda m, t=text: m.loadtxt(t.splitlines(keepends=True))
        cases[f"{tname} int dtype"] = lambda m, t=text: m.loadtxt(io.StringIO(t), dtype=int)
        cases[f"{tname} usecols"] = lambda m, t=text: m.loadtxt(io.StringIO(t), usecols=0)
        cases[f"{tname} unpack"] = lambda m, t=text: m.loadtxt(io.StringIO(t), unpack=True)
        cases[f"{tname} skiprows"] = lambda m, t=text: m.loadtxt(io.StringIO(t), skiprows=1)
    bad = []
    for name, case in cases.items():
        ours, theirs = outcome(lambda: case(fnp)), outcome(lambda: case(np))
        if ours != theirs:
            bad.append(f"{name}: fnp={str(ours)[:110]} numpy={str(theirs)[:110]}")
    return len(cases), bad

with tempfile.TemporaryDirectory() as tmpdir:
    cells, bad = sweep(tmpdir)
print(cells, bad)
"##
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "392 []",
        "loadtxt shapes, delimiters and newlines must be numpy's: {result}"
    );
    Ok(())
}

/// `load` reads ONE array from a file-like at its current position and opens a PATH itself:
/// - reading the whole stream here made `load(f); load(f)` raise EOFError on the second
///   array (numpy's own test_load_multiple_arrays_until_eof);
/// - handing numpy a BytesIO copy of an .npz lost the NpzFile's filename (its repr) and its
///   ownership of the file it closes (numpy's own TestSavezLoad::test_repr_lists_keys /
///   test_closing_zipfile_after_load);
/// - `bytes` is a path to numpy (os.fspath) and was read as file CONTENTS;
/// - a missing path raised a generic OSError, not numpy's FileNotFoundError.
///
/// 10 of the 17 cells failed before the fix (numpy 2.4.3); 0 fail after, on numpy 2.4.3 and
/// 2.3.5.
#[test]
fn load_reads_one_array_per_call_and_opens_paths_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import io, os, pathlib, tempfile, warnings

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            if isinstance(r, (np.ndarray, np.generic)):
                a = np.asarray(r)
                got = ("ok", type(r).__name__, a.dtype.str, a.shape, a.tobytes())
            else:
                got = ("ok", type(r).__name__, repr(r))
        except Exception as ex:
            got = (type(ex).__name__, str(ex))
    return got + (sorted({w.category.__name__ for w in caught}),)

def sweep(tmpdir):
    def path(name):
        return os.path.join(tmpdir, name)

    a = np.array([[1, 2], [3, 4]], float)
    np.savez(path("one.npz"), a)
    np.savez(path("five.npz"), *[a] * 5)
    np.savez(path("six.npz"), *[a] * 6)
    np.savez(path("lab.npz"), lab="place holder")
    np.save(path("x.npy"), a)
    with open(path("cr.txt"), "w") as f:
        f.write("1 21\r3 42\r")
    with open(path("crlf.txt"), "w", newline="") as f:
        f.write("1 21\r\n3 42\r\n")

    def npz_repr(m, name):
        data = m.load(path(name))
        try:
            return repr(data).replace(path(name), "<P>")
        finally:
            data.close()

    def npz_close_closes_fp(m):
        data = m.load(path("lab.npz"))
        fp = data.zip.fp
        data.close()
        return fp.closed

    def multi_load(m):
        f = io.BytesIO()
        np.save(f, 1)
        np.save(f, 2)
        f.seek(0)
        out = [m.load(f).tolist(), m.load(f).tolist()]
        try:
            m.load(f)
        except Exception as ex:
            out.append(type(ex).__name__)
        return out

    def npz_type(m):
        data = m.load(path("one.npz"))
        try:
            return (type(data).__name__, sorted(data.keys()), data["arr_0"].tolist(), "arr_0" in data, len(data))
        finally:
            data.close()

    def npz_context(m):
        with m.load(path("five.npz")) as data:
            return sorted(data.files)

    raw = open(path("x.npy"), "rb").read()
    cases = {
        "npz repr 1": lambda m: npz_repr(m, "one.npz"),
        "npz repr 5": lambda m: npz_repr(m, "five.npz"),
        "npz repr 6": lambda m: npz_repr(m, "six.npz"),
        "npz close closes fp": npz_close_closes_fp,
        "npz type/keys": npz_type,
        "npz context": npz_context,
        "load multiple until EOF": multi_load,
        "loadtxt CR newline": lambda m: m.loadtxt(path("cr.txt")),
        "loadtxt CRLF newline": lambda m: m.loadtxt(path("crlf.txt")),
        "loadtxt StringIO CR": lambda m: m.loadtxt(io.StringIO("1 21\r3 42\r")),
        "genfromtxt CR": lambda m: m.genfromtxt(path("cr.txt")),
        "load empty BytesIO": lambda m: m.load(io.BytesIO(b"")),
        "load npy path": lambda m: m.load(path("x.npy")),
        "load bytes path": lambda m: m.load(path("x.npy").encode()),
        "load raw npy bytes": lambda m: m.load(raw),
        "load missing path": lambda m: m.load(path("missing.npy")).tolist(),
        "load pathlib": lambda m: m.load(pathlib.Path(path("x.npy"))),
    }
    bad = []
    for name, case in cases.items():
        ours, theirs = outcome(lambda: case(fnp)), outcome(lambda: case(np))
        if ours != theirs:
            bad.append(f"{name}: fnp={str(ours)[:170]} numpy={str(theirs)[:170]}")
    return len(cases), bad

with tempfile.TemporaryDirectory() as tmpdir:
    cells, bad = sweep(tmpdir)
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "17 []",
        "load must read one array per call and open paths as numpy does: {result}"
    );
    Ok(())
}
