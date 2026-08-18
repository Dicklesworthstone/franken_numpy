//! Conformance for the hoisted dtype sniff's routing decisions (`deadlock-audit-ei9jz`).
//!
//! `PyUFunc::__call__` sniffs `x1.dtype` once and derives every native-route guard from it
//! (`x1_is_maybe_f16`, `x1_is_maybe_complex`, `x1_is_maybe_temporal`, and the f64 block's
//! `== Some('d')`). Those guards decide WHICH CODE computes the answer, so a sniff that
//! mis-classifies a dtype silently reroutes the operation.
//!
//! WHY THESE CASES AND NOT MORE ARITHMETIC: the ordinary f64 path is already covered
//! extensively elsewhere. What is NOT covered anywhere else is the set of dtypes that must
//! NOT be classified as f16/complex/temporal, and the operands that carry no `dtype` at all
//! and must therefore leave every guard at its conservative default. These are the inputs a
//! faster sniff is most likely to get wrong, and the ones no existing test would catch.
//!
//! WHY `tobytes()` AND NOT `np.array_equal`: `array_equal` treats `0.0 == -0.0` as equal, so
//! it is BLIND to exactly the signed-zero divergence a mis-routed multiply produces, and it
//! reports NaN != NaN as a mismatch so it false-alarms on any NaN payload. Comparing raw
//! bytes catches sign bits and NaN payloads and needs no `equal_nan` flag. This was not
//! theoretical: an earlier version of this suite using `array_equal` reported a false
//! mismatch on the signed-zero case while being unable to detect a real one.

use std::process::Command;

fn numpy_oracle(script: &str) -> Result<String, String> {
    let output = Command::new("python3")
        .args(["-c", script])
        .output()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("oracle failed: {stderr}\nScript: {script}"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn fnp_script(body: &str) -> String {
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
         bad = []\n\
         def check(label, g, w):\n\
        \x20   try:\n\
        \x20       got, gerr = g(), None\n\
        \x20   except Exception as e:\n\
        \x20       got, gerr = None, type(e).__name__\n\
        \x20   try:\n\
        \x20       want, werr = w(), None\n\
        \x20   except Exception as e:\n\
        \x20       want, werr = None, type(e).__name__\n\
        \x20   if gerr or werr:\n\
        \x20       ok = gerr == werr\n\
        \x20   else:\n\
        \x20       ok = (np.asarray(got).tobytes() == np.asarray(want).tobytes()\n\
        \x20             and np.asarray(got).dtype == np.asarray(want).dtype)\n\
        \x20   if not ok:\n\
        \x20       bad.append(label)\n\
         {body}\n\
         print('OK' if not bad else 'MISMATCH:' + ','.join(bad))\n"
    )
}

/// Dtypes that MUST be recognised: getting these wrong silently disables a native route.
#[test]
fn sniff_classifies_interned_builtin_dtypes() -> Result<(), String> {
    let body = "\
a = np.arange(8.0)\n\
c128 = np.arange(4, dtype=np.complex128) + 1j\n\
c64 = np.arange(4, dtype=np.complex64) + np.complex64(1j)\n\
h = np.arange(4, dtype=np.float16)\n\
check('f64', lambda: fnp.multiply(a, a + 1), lambda: np.multiply(a, a + 1))\n\
check('c128', lambda: fnp.multiply(c128, c128), lambda: np.multiply(c128, c128))\n\
check('c64', lambda: fnp.multiply(c64, c64), lambda: np.multiply(c64, c64))\n\
check('f16', lambda: fnp.add(h, h), lambda: np.add(h, h))";
    assert_eq!(numpy_oracle(&fnp_script(body))?, "OK");
    Ok(())
}

/// PARAMETERISED dtypes. `np.dtype('M8[ns]')` is NOT the interned `M8` object, so any
/// identity-based recognition MUST miss here and fall back rather than mis-classify - and
/// `'M'`/`'m'` are in the temporal guard set, so a mis-classification reroutes real work.
#[test]
fn sniff_falls_back_for_parameterised_datetime_dtypes() -> Result<(), String> {
    let body = "\
td = np.array([10, 20, 30], dtype='timedelta64[ns]')\n\
td_s = np.array([1, 2], dtype='timedelta64[s]')\n\
dt = np.array(['2026-01-01', '2026-06-01'], dtype='datetime64[D]')\n\
check('td_rem', lambda: fnp.remainder(td, np.timedelta64(7, 'ns')), lambda: np.remainder(td, np.timedelta64(7, 'ns')))\n\
check('td_add', lambda: fnp.add(td, td), lambda: np.add(td, td))\n\
check('td_s_mul', lambda: fnp.multiply(td_s, 3), lambda: np.multiply(td_s, 3))\n\
check('dt_sub', lambda: fnp.subtract(dt, dt), lambda: np.subtract(dt, dt))\n\
check('dt_mul_raises', lambda: fnp.multiply(dt, dt), lambda: np.multiply(dt, dt))";
    assert_eq!(numpy_oracle(&fnp_script(body))?, "OK");
    Ok(())
}

/// Dtypes that are neither interned builtins nor parameterised temporals. Several of these
/// make NumPy raise, and the error TYPE must match - "both failed" is not parity.
#[test]
fn sniff_falls_back_for_other_dtypes_including_errors() -> Result<(), String> {
    let body = "\
st = np.array([(1, 2.0), (3, 4.0)], dtype=[('i', 'i4'), ('x', 'f8')])\n\
f32 = np.arange(4, dtype=np.float32)\n\
i64 = np.arange(4, dtype=np.int64) + 1\n\
b = np.array([True, False])\n\
S = np.array(['ab', 'cd'])\n\
check('struct_raises', lambda: fnp.multiply(st, st), lambda: np.multiply(st, st))\n\
check('f32', lambda: fnp.multiply(f32, f32), lambda: np.multiply(f32, f32))\n\
check('i64_mul', lambda: fnp.multiply(i64, i64), lambda: np.multiply(i64, i64))\n\
check('i64_rem', lambda: fnp.remainder(i64, i64), lambda: np.remainder(i64, i64))\n\
check('bool', lambda: fnp.multiply(b, b), lambda: np.multiply(b, b))\n\
check('unicode_raises', lambda: fnp.multiply(S, S), lambda: np.multiply(S, S))";
    assert_eq!(numpy_oracle(&fnp_script(body))?, "OK");
    Ok(())
}

/// Operands with NO `dtype` attribute. Every guard reads `None => true`, i.e. "maybe" - the
/// CONSERVATIVE direction. A sniff that turned an absent dtype into a decline would silently
/// skip native routes for lists and scalars.
#[test]
fn sniff_leaves_guards_conservative_when_there_is_no_dtype() -> Result<(), String> {
    let body = "\
a = np.arange(8.0)\n\
check('lists', lambda: fnp.multiply([1, 2, 3], [4, 5, 6]), lambda: np.multiply([1, 2, 3], [4, 5, 6]))\n\
check('scalars', lambda: fnp.multiply(3.5, 2.0), lambda: np.multiply(3.5, 2.0))\n\
check('list_x_ndarray', lambda: fnp.multiply([1.0, 2.0, 3.0, 4.0], a[:4]), lambda: np.multiply([1.0, 2.0, 3.0, 4.0], a[:4]))\n\
check('none_raises', lambda: fnp.multiply(None, 1.0), lambda: np.multiply(None, 1.0))";
    assert_eq!(numpy_oracle(&fnp_script(body))?, "OK");
    Ok(())
}

/// Edge VALUES on the recognised f64 path. Signed zeros are the reason this suite compares
/// bytes: `np.array_equal` would pass a route that returned `+0.0` where NumPy returns
/// `-0.0`, and sign is exactly what a mis-routed multiply gets wrong.
#[test]
fn sniff_preserves_signed_zeros_nan_and_strides_on_the_f64_path() -> Result<(), String> {
    let body = "\
sp = np.array([0.0, -0.0, np.inf, -np.inf, np.nan])\n\
sz = np.array([0.0, -0.0])\n\
neg = np.array([-1.0, -1.0])\n\
a = np.arange(8.0)\n\
check('specials', lambda: fnp.multiply(sp, sp), lambda: np.multiply(sp, sp))\n\
check('signed_zero_sign', lambda: fnp.multiply(sz, neg), lambda: np.multiply(sz, neg))\n\
check('strided', lambda: fnp.multiply(a[::2], a[::2]), lambda: np.multiply(a[::2], a[::2]))\n\
check('broadcast', lambda: fnp.multiply(a, 2.0), lambda: np.multiply(a, 2.0))";
    assert_eq!(numpy_oracle(&fnp_script(body))?, "OK");
    Ok(())
}
