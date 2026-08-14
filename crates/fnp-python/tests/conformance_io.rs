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
         from io import BytesIO, StringIO\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
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
