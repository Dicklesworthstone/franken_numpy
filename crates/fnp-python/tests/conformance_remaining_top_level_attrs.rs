//! Conformance tests for fnp_python's remaining numpy top-level re-exports:
//! iterable, ndim, size, packbits, unpackbits, fromfunction, pow (ufunc),
//! typecodes, typename, sctypeDict, ScalarType, __array_namespace_info__,
//! typing (submodule), ctypeslib (submodule), test (PytestTester),
//! getbufsize, nested_iters, from_dlpack.

use std::{
    io::Write,
    process::{Command, Stdio},
};

fn numpy_oracle(script: &str) -> Result<String, String> {
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| format!("python3 stdin pipe was unavailable\nScript: {script}"))?;
        stdin
            .write_all(script.as_bytes())
            .map_err(|error| format!("failed to write python script: {error}\nScript: {script}"))?;
    }
    let output = child
        .wait_with_output()
        .map_err(|error| format!("failed to wait for python3: {error}\nScript: {script}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NumPy oracle failed: {stderr}\nScript: {script}"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

mod support;

fn fnp_script(body: String) -> String {
    support::fnp_script_with("import sys\n", true, body)
}

fn expect_equal(actual: &str, expected: &str, context: &str) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{context}; expected {expected:?}, got {actual:?}"))
    }
}

#[test]
fn remaining_top_level_attrs_identity_equal_to_numpy() -> Result<(), String> {
    // The bulk safety check: every attribute we RE-EXPORT is `is`-equal to its
    // numpy counterpart. Identity equality is enough to prove the re-export
    // is genuine; semantic equivalence is then numpy's own contract.
    //
    // `unpackbits` is deliberately NOT in this list. It is a native parallel
    // pyfunction (lib.rs `fn unpackbits`, registered via wrap_pyfunction!), and
    // the verbatim re-export list omits it on purpose because `m.add` overwrites
    // — re-exporting it would silently replace the native path with numpy's.
    // Identity is therefore the WRONG contract for it; the right one is asserted
    // in unpackbits_is_native_not_a_numpy_reexport below, and its behaviour is
    // pinned by packbits_unpackbits_round_trip_matches_numpy.
    let script = fnp_script(
        r#"
names = [
    'iterable', 'ndim', 'size', 'packbits', 'fromfunction',
    'pow', 'typecodes', 'typename', 'sctypeDict', 'ScalarType',
    '__array_namespace_info__', 'typing', 'ctypeslib', 'test',
    'getbufsize', 'nested_iters', 'from_dlpack',
]
mismatches = []
for n in names:
    if not hasattr(fnp, n):
        mismatches.append((n, 'missing'))
        continue
    if getattr(fnp, n) is not getattr(np, n):
        mismatches.append((n, 'not-identity-equal'))
print(mismatches)
print(mismatches == [])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    expect_equal(
        last,
        "True",
        &format!(
            "all remaining top-level attributes must be identity-equal to numpy; output: {result}"
        ),
    )
}

#[test]
fn unpackbits_is_native_not_a_numpy_reexport() -> Result<(), String> {
    // The regression this exists to catch: adding "unpackbits" back to the
    // verbatim re-export list in lib.rs would silently overwrite the native
    // parallel pyfunction with numpy's (`m.add` overwrites), killing the fast
    // path with no other test noticing — every behavioural test would still
    // pass, because numpy's answer is the one we compare against.
    //
    // So identity is asserted in the NEGATIVE direction, and "not identity" is
    // not allowed to pass vacuously: it would also be satisfied by the
    // attribute being missing or broken, so presence, callability and
    // agreement with numpy are all required in the same script. `packbits` is
    // checked alongside as the contrast case — it IS a genuine re-export, so
    // the deliberate asymmetry between the two is what gets pinned here.
    let script = fnp_script(
        r#"
checks = []
checks.append(('unpackbits present', hasattr(fnp, 'unpackbits')))
checks.append(('unpackbits is native, not numpy', fnp.unpackbits is not np.unpackbits))
checks.append(('packbits is a genuine re-export', fnp.packbits is np.packbits))
a = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
packed = np.packbits(a)
checks.append(('native unpackbits matches numpy', np.array_equal(fnp.unpackbits(packed), np.unpackbits(packed))))
checks.append(('native unpackbits round-trips', np.array_equal(fnp.unpackbits(packed)[:len(a)], a)))
failed = [name for name, ok in checks if not ok]
print(failed)
print(failed == [])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    expect_equal(
        last,
        "True",
        &format!("unpackbits must stay native and numpy-equivalent; output: {result}"),
    )
}

#[test]
fn pow_ufunc_acts_like_numpy_power_alias() -> Result<(), String> {
    // np.pow is a numpy 2.x alias for np.power. Both must produce identical
    // output when called as a ufunc.
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4], dtype=np.float64)
b = np.array([2, 2, 2, 2], dtype=np.float64)
ours = fnp.pow(a, b)
theirs = np.pow(a, b)
print(np.array_equal(ours, theirs) and np.array_equal(ours, np.power(a, b)))
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.pow must match np.pow and np.power",
    )
}

#[test]
fn packbits_unpackbits_round_trip_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=np.uint8)
packed_ours = fnp.packbits(a)
packed_theirs = np.packbits(a)
unpacked_ours = fnp.unpackbits(packed_ours)[:len(a)]
unpacked_theirs = np.unpackbits(packed_theirs)[:len(a)]
print(np.array_equal(packed_ours, packed_theirs) and
      np.array_equal(unpacked_ours, unpacked_theirs) and
      np.array_equal(unpacked_ours, a))
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.packbits/unpackbits must match numpy and round-trip",
    )
}

#[test]
fn fromfunction_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
ours = fnp.fromfunction(lambda i, j: i + j, (3, 4), dtype=int)
theirs = np.fromfunction(lambda i, j: i + j, (3, 4), dtype=int)
print(np.array_equal(ours, theirs))
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.fromfunction must match numpy",
    )
}

#[test]
fn ndim_size_iterable_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.arange(12).reshape(3, 4)
ok = (fnp.ndim(arr) == np.ndim(arr) == 2 and
      fnp.size(arr) == np.size(arr) == 12 and
      fnp.size(arr, axis=0) == np.size(arr, axis=0) == 3 and
      fnp.iterable([1, 2, 3]) == np.iterable([1, 2, 3]) == True and
      fnp.iterable(5) == np.iterable(5) == False)
print(ok)
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.ndim/size/iterable must match numpy",
    )
}

#[test]
#[allow(non_snake_case)]
fn typecodes_and_sctypeDict_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.typecodes == np.typecodes and
      sorted(fnp.sctypeDict.keys()) == sorted(np.sctypeDict.keys()))
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.typecodes and fnp.sctypeDict must match numpy",
    )
}

#[test]
fn getbufsize_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.getbufsize() == np.getbufsize())
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.getbufsize must match numpy",
    )
}

#[test]
#[allow(non_snake_case)]
fn ScalarType_contains_basic_scalars() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Both fnp.ScalarType and np.ScalarType are tuples of Python scalar types.
ok = (fnp.ScalarType == np.ScalarType and int in fnp.ScalarType and
      float in fnp.ScalarType and complex in fnp.ScalarType)
print(ok)
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.ScalarType must equal np.ScalarType",
    )
}

#[test]
fn fnp_python_covers_full_numpy_all() -> Result<(), String> {
    // End-of-parity-wave gate: after this commit, every name in
    // numpy.__all__ must be reachable via fnp_python. This is the
    // structural lock that catches any future regression that
    // accidentally drops a re-export.
    let script = fnp_script(
        r#"
missing = [n for n in np.__all__ if not hasattr(fnp, n)]
print(len(missing), missing[:10])
print(missing == [])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    expect_equal(
        last,
        "True",
        &format!("every name in numpy.__all__ must be reachable via fnp_python; output: {result}"),
    )
}

/// Every callable in `numpy.__all__` has numpy's `inspect.signature` on fnp too (the live
/// numpy's; a numpy builtin that carries none on that version is skipped). 21 fnp functions
/// reported a bare `(*args, **kwargs)`, and numpy's own TestCreationFuncs::test_signatures
/// failed for empty/zeros/ones. Four failing checks were behaviour, not introspection: an
/// explicit `indices(dtype=None)` returned int64 where numpy returns float64, `row_stack`
/// rejected numpy's keyword-only `dtype`/`casting` (two cases), and `loadtxt` took `quotechar`
/// positionally. 35 checks failed before the fix (31 signatures + 4 behaviours, numpy 2.4.3).
///
/// eye/tri/loadtxt/genfromtxt/indices default `dtype` to a CLASS, which a builtin's text
/// signature cannot carry (inspect evaluates constant defaults only); they compare on names,
/// kinds and every other default, and only while numpy's default really is a class.
#[test]
fn every_numpy_all_callable_has_numpys_signature() -> Result<(), String> {
    // `r##` because the body contains `"#"`.
    let script = fnp_script(
        r##"
import inspect, io, warnings

# numpy's default is a CLASS in these (`dtype=float` / `dtype=int`). A builtin's text signature
# carries only constant defaults, so they compare on names, kinds and every other default.
CLASS_DEFAULT = {"eye", "tri", "loadtxt", "genfromtxt", "indices"}

def signature(f):
    try:
        return inspect.signature(f)
    except (ValueError, TypeError):
        return None

def same_but_class_defaults(got, want):
    g, w = list(got.parameters.values()), list(want.parameters.values())
    return len(g) == len(w) and any(isinstance(p.default, type) for p in w) and all(
        a.name == b.name and a.kind == b.kind and (isinstance(b.default, type) or a.default == b.default)
        for a, b in zip(g, w))

checked, bad = 0, []
for name in sorted(np.__all__):
    nf, ff = getattr(np, name, None), getattr(fnp, name, None)
    if not callable(nf) or isinstance(nf, (type, np.ufunc)):
        continue
    want = signature(nf)
    if want is None:  # numpy's own builtin carries no signature on this numpy version
        continue
    checked += 1
    got = signature(ff)
    if got is not None and (got == want or (name in CLASS_DEFAULT and same_but_class_defaults(got, want))):
        continue
    bad.append(f"{name}: numpy{want} fnp{got}")

# The behaviour behind three of those signatures.
def outcome(call, type_only=False):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            a = np.asarray(r)
            got = ("ok", type(r).__name__, a.dtype.str, a.shape, a.tobytes())
        except Exception as ex:
            got = (type(ex).__name__,) if type_only else (type(ex).__name__, str(ex))
    return got + (sorted({w.category.__name__ for w in caught}),)

cases = {
    # An explicit dtype=None is numpy's `empty(..., dtype=None)`: float64, not the int default.
    "indices dtype=None": (lambda m: m.indices((2, 3), dtype=None), False),
    "indices default": (lambda m: m.indices((2, 3)), False),
    # quotechar is keyword-only: a 12th positional argument is a TypeError.
    "loadtxt positional quotechar": (lambda m: m.loadtxt(io.StringIO("1 2"), float, "#", None, None, 0, None, False, 0, None, None, '"'), True),
    "loadtxt quotechar kw": (lambda m: m.loadtxt(io.StringIO('"1" 2'), quotechar='"'), False),
    # func is positional-only.
    "frompyfunc func=": (lambda m: m.frompyfunc(func=abs, nin=1, nout=1), True),
}
if hasattr(np, "row_stack"):
    # row_stack is vstack: keyword-only dtype / casting.
    cases["row_stack dtype"] = (lambda m: m.row_stack(([1, 2], [3, 4]), dtype=np.float32), False)
    cases["row_stack casting"] = (lambda m: m.row_stack(([1.5, 2], [3, 4]), dtype=np.int64, casting="unsafe"), False)
for name, (case, type_only) in cases.items():
    ours, theirs = outcome(lambda: case(fnp), type_only), outcome(lambda: case(np), type_only)
    if ours != theirs:
        bad.append(f"{name}: fnp={str(ours)[:160]} numpy={str(theirs)[:160]}")
print(checked, len(cases), bad)
print(checked >= 230 and not bad)
"##
        .into(),
    );
    let result = numpy_oracle(&script)?;
    expect_equal(
        result.lines().last().unwrap_or("").trim(),
        "True",
        &format!("numpy.__all__ callables must carry numpy's signature; output: {result}"),
    )
}

#[test]
fn fnp_python_top_level_all_matches_numpy_verbatim() -> Result<(), String> {
    // Emit four signals so a failure points at the actual divergence rather
    // than just "False":
    //   1. presence (`hasattr`)
    //   2. verbatim list equality (order-sensitive, the canonical contract)
    //   3. sorted set-equality (catches order-only drift vs. content drift)
    //   4. symmetric-diff names (only printed when sets differ, so the
    //      diagnostic shows up exactly when it's useful)
    let script = fnp_script(
        r#"
print(hasattr(fnp, '__all__'))
print(fnp.__all__ == np.__all__)
print(sorted(fnp.__all__) == sorted(np.__all__))
missing = sorted(set(np.__all__) - set(fnp.__all__))
extra = sorted(set(fnp.__all__) - set(np.__all__))
print(f"missing={missing} extra={extra}")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    expect_equal(
        lines.next().unwrap_or("").trim(),
        "True",
        &format!("fnp_python must expose top-level __all__; output: {result}"),
    )?;
    let verbatim = lines.next().unwrap_or("").trim();
    let sorted_eq = lines.next().unwrap_or("").trim();
    let diff_line = lines.next().unwrap_or("").trim();
    expect_equal(
        verbatim,
        "True",
        &format!(
            "fnp_python.__all__ must match numpy.__all__ verbatim (order-sensitive); \
             sorted-equal={sorted_eq} {diff_line}; full output: {result}"
        ),
    )
}

#[test]
fn core_and_f2py_identity_equal_to_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
print(fnp.core is np.core and fnp.f2py is np.f2py)
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.core / fnp.f2py must be identity-equal to numpy's",
    )
}

#[test]
fn array_namespace_info_callable_via_fnp() -> Result<(), String> {
    let script = fnp_script(
        r#"
ours = fnp.__array_namespace_info__()
theirs = np.__array_namespace_info__()
ok = (type(ours).__name__ == type(theirs).__name__ and
      ours.devices() == theirs.devices())
print(ok)
"#
        .into(),
    );
    expect_equal(
        numpy_oracle(&script)?.trim(),
        "True",
        "fnp.__array_namespace_info__() must mirror np",
    )
}

/// Submodule callables whose typed PyO3 parameters diverged from numpy's behaviour (bead
/// deadlock-audit-qxy9u), 68 cells compared by outcome and warnings:
/// - `{char,strings}.{find,rfind,count,index,rindex}` re-packed `start`/`end` POSITIONALLY and
///   dropped a missing `start`, so `find(a, sub, end=2)` searched from 2 - a WRONG ANSWER - and
///   an explicit `start=None` (numpy's cast error) searched from 0;
/// - explicit `None` for `fillchar`/`count`/`tabsize`/`ma.count(keepdims=)` answered where
///   numpy raises;
/// - `ma.average` refused `keepdims`; `ma.mask_rows(axis=None)` lost numpy's
///   DeprecationWarning;
/// - `testing.assert_allclose` refused `strict`, `assert_array_almost_equal` refused numpy's
///   `actual=`/`desired=`, `assert_array_equal` took keyword-only `strict` positionally, and the
///   native assertion fast paths PASSED shape-mismatched operands numpy fails (only a 0-d side
///   broadcasts) and raised one-line summaries instead of numpy's report.
/// 26 of the 68 cells failed before the fix (numpy 2.4.3); 0 fail after, on numpy 2.4.3 and
/// 2.3.5.
#[test]
fn submodule_calls_match_numpys_parameters_and_outcomes() -> Result<(), String> {
    let script = fnp_script(
        r##"
import warnings

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            if isinstance(r, np.ma.MaskedArray):
                got = ("masked", r.dtype.str, r.shape, np.ma.getdata(r).tobytes(), np.ma.getmaskarray(r).tobytes())
            elif isinstance(r, (np.ndarray, np.generic)):
                a = np.asarray(r)
                got = ("ok", type(r).__name__, a.dtype.str, a.shape, repr(a.tolist()) if a.dtype == object else a.tobytes())
            else:
                got = ("ok", type(r).__name__, repr(r))
        except Exception as ex:
            got = (type(ex).__name__, str(ex).splitlines()[0][:160] if str(ex) else "")
    return got + (sorted({w.category.__name__ for w in caught}),)

x = np.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[0, 1, 0], [0, 0, 1]])
s = np.array(["abc", "aXbXc", "xx"])
b = np.array([b"abc", b"aXbXc"])
cases = {
    # numpy.ma
    "ma.argmax keepdims=True": lambda m: m.ma.argmax(x, axis=1, keepdims=True),
    "ma.argmax keepdims=False": lambda m: m.ma.argmax(x, axis=1, keepdims=False),
    "ma.argmin keepdims=True": lambda m: m.ma.argmin(x, axis=0, keepdims=True),
    "ma.count keepdims=True": lambda m: m.ma.count(x, axis=1, keepdims=True),
    "ma.count keepdims=None": lambda m: m.ma.count(x, axis=1, keepdims=None),
    "ma.average keepdims=True": lambda m: m.ma.average(x, axis=1, keepdims=True),
    "ma.average keepdims=False": lambda m: m.ma.average(x, axis=0, keepdims=False),
    "ma.average returned": lambda m: m.ma.average(x, axis=1, returned=True)[1],
    "ma.fix_invalid mask=None": lambda m: m.ma.fix_invalid(np.array([1.0, np.nan]), mask=None),
    "ma.fix_invalid mask=False_": lambda m: m.ma.fix_invalid(np.array([1.0, np.nan]), mask=np.False_),
    "ma.make_mask dtype=None": lambda m: m.ma.make_mask([0, 1], dtype=None),
    "ma.make_mask default": lambda m: m.ma.make_mask([0, 1]),
    "ma.mask_rows axis=None": lambda m: m.ma.mask_rows(x, axis=None),
    "ma.mask_rows": lambda m: m.ma.mask_rows(x),
    "ma.mask_cols axis=0": lambda m: m.ma.mask_cols(x, axis=0),
    "ma.masked_all dtype=None": lambda m: m.ma.masked_all((2,), dtype=None),
    "ma.masked_all": lambda m: m.ma.masked_all((2,)),
    "ma.common_fill_value kw": lambda m: m.ma.common_fill_value(a=x, b=x),
    "ma.default_fill_value kw": lambda m: m.ma.default_fill_value(obj=x),
    "ma.set_fill_value kw": lambda m: m.ma.set_fill_value(a=x.copy(), fill_value=0),
    "ma.masked_object kw": lambda m: m.ma.masked_object(x=np.array([1, 2], object), value=2),
    "ma.flatten_mask kw": lambda m: m.ma.flatten_mask(mask=[True, False]),
    # numpy.char / numpy.strings
    **{f"{mod}.find start=0": (lambda m, mod=mod: getattr(m, mod).find(s, "X", start=0)) for mod in ("char", "strings")},
    **{f"{mod}.find start=None": (lambda m, mod=mod: getattr(m, mod).find(s, "X", start=None)) for mod in ("char", "strings")},
    **{f"{mod}.count end=None": (lambda m, mod=mod: getattr(m, mod).count(s, "X", 0, None)) for mod in ("char", "strings")},
    **{f"{mod}.replace count=-1": (lambda m, mod=mod: getattr(m, mod).replace(s, "X", "-", count=-1)) for mod in ("char", "strings")},
    **{f"{mod}.replace count=None": (lambda m, mod=mod: getattr(m, mod).replace(s, "X", "-", count=None)) for mod in ("char", "strings")},
    **{f"{mod}.center fillchar=' '": (lambda m, mod=mod: getattr(m, mod).center(s, 7, fillchar=" ")) for mod in ("char", "strings")},
    **{f"{mod}.center fillchar=None": (lambda m, mod=mod: getattr(m, mod).center(s, 7, fillchar=None)) for mod in ("char", "strings")},
    **{f"{mod}.ljust bytes": (lambda m, mod=mod: getattr(m, mod).ljust(b, 7)) for mod in ("char", "strings")},
    **{f"{mod}.rjust fillchar kw": (lambda m, mod=mod: getattr(m, mod).rjust(s, 5, "*")) for mod in ("char", "strings")},
    **{f"{mod}.expandtabs tabsize=8": (lambda m, mod=mod: getattr(m, mod).expandtabs(np.array(["a\tb"]), tabsize=8)) for mod in ("char", "strings")},
    **{f"{mod}.expandtabs tabsize=None": (lambda m, mod=mod: getattr(m, mod).expandtabs(np.array(["a\tb"]), tabsize=None)) for mod in ("char", "strings")},
    **{f"{mod}.index start=0": (lambda m, mod=mod: getattr(m, mod).index(s[:2], "b", start=0)) for mod in ("char", "strings")},
    **{f"{mod}.rindex start=None": (lambda m, mod=mod: getattr(m, mod).rindex(s[:2], "b", start=None)) for mod in ("char", "strings")},
    "strings.slice": lambda m: m.strings.slice(s, 1, 3),
    "strings.slice kw": lambda m: m.strings.slice(a=s, start=1),
    # numpy.testing
    "testing.assert_allclose strict=True": lambda m: m.testing.assert_allclose(np.array([1.0]), 1.0, strict=True),
    "testing.assert_allclose strict=False": lambda m: m.testing.assert_allclose(np.array([1.0]), 1.0, strict=False),
    "testing.assert_array_almost_equal kw": lambda m: m.testing.assert_array_almost_equal(actual=[1.0], desired=[1.0]),
    "testing.assert_array_equal positional strict": lambda m: m.testing.assert_array_equal([1], [1], "", True, True),
    "testing.assert_equal err_msg=None fail": lambda m: m.testing.assert_equal(1, 2, err_msg=None),
    "testing.assert_array_equal err_msg='' fail": lambda m: m.testing.assert_array_equal([1], [2], err_msg=""),
    "testing.assert_almost_equal kw": lambda m: m.testing.assert_almost_equal(actual=1.0, desired=1.0),
    "testing.assert_array_less kw": lambda m: m.testing.assert_array_less(x=[1], y=[2]),
    # A shape mismatch FAILS numpy's assertions (only a 0-d side broadcasts).
    "testing.assert_allclose shape mismatch": lambda m: m.testing.assert_allclose([1.0, 1.0], [[1.0, 1.0], [1.0, 1.0]]),
    "testing.assert_array_equal shape mismatch": lambda m: m.testing.assert_array_equal([1, 1], [[1, 1], [1, 1]]),
    "testing.assert_allclose scalar side": lambda m: m.testing.assert_allclose([1.0, 1.0], 1.0),
    "testing.assert_allclose float32": lambda m: m.testing.assert_allclose(np.float32([1.0000001]), np.float32([1.0]), rtol=0),
    # `end=` alone is END, not start.
    **{f"{mod}.{fn} end= only": (lambda m, mod=mod, fn=fn: getattr(getattr(m, mod), fn)(np.array(["aXbXc", "XXXX"]), "X", end=2)) for mod in ("char", "strings") for fn in ("find", "rfind", "count")},
}
bad = []
for name, case in cases.items():
    ours, theirs = outcome(lambda: case(fnp)), outcome(lambda: case(np))
    if ours != theirs:
        bad.append(f"{name}: fnp={str(ours)[:150]} numpy={str(theirs)[:150]}")
print(len(cases), bad)
"##
        .into(),
    );
    let result = numpy_oracle(&script)?;
    expect_equal(
        result.lines().last().unwrap_or("").trim(),
        "68 []",
        &format!("submodule calls must match numpy's parameters and outcomes; output: {result}"),
    )
}

/// `inspect.signature` parity for the submodules whose every callable now carries numpy's:
/// fft, linalg, char, strings (bar `slice`, whose `<no value>` default a builtin's text
/// signature cannot carry), testing, emath and rec (the live numpy's; a numpy builtin without
/// a signature is skipped). ma and random are not all there yet (bead deadlock-audit-qxy9u).
/// 27 of the 105 checked callables differed before the fix (numpy 2.4.3); 0 after, on numpy
/// 2.4.3 and 2.3.5.
#[test]
fn submodule_callables_have_numpys_signature() -> Result<(), String> {
    let script = fnp_script(
        r#"
import inspect

def signature(f):
    try:
        return inspect.signature(f)
    except (ValueError, TypeError):
        return None

checked, bad = 0, []
for sub in ("fft", "linalg", "char", "strings", "testing", "emath", "rec"):
    nm, fm = getattr(np, sub), getattr(fnp, sub)
    for name in getattr(nm, "__all__", [n for n in dir(nm) if not n.startswith("_")]):
        if (sub, name) == ("strings", "slice"):
            continue
        nf, ff = getattr(nm, name, None), getattr(fm, name, None)
        if not callable(nf) or isinstance(nf, (type, np.ufunc)) or ff is None or ff is nf:
            continue
        want = signature(nf)
        if want is None:
            continue
        checked += 1
        got = signature(ff)
        if got != want:
            bad.append(f"{sub}.{name}: numpy{want} fnp{got}")
print(checked, bad)
print(checked >= 100 and not bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    expect_equal(
        result.lines().last().unwrap_or("").trim(),
        "True",
        &format!("submodule callables must carry numpy's signature; output: {result}"),
    )
}
