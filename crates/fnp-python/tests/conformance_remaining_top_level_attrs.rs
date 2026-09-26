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

/// Names numpy resolves dynamically (its module `__getattr__`) and the `version` / `matlib`
/// submodules. fnp's top module had no `__getattr__`: `np.float` was a bare AttributeError
/// without numpy's "was a deprecated alias for the builtin" guidance, `np.str` gave no
/// FutureWarning, `np.chararray` and `np.lib.math` were missing, and `np.version` / `np.matlib`
/// did not exist (numpy's test_deprecations, test_numpy_version, test_matlib: 20 tests). On
/// a8d9a337 this test died at `fnp.version`. A plain miss must stay fnp's AttributeError.
#[test]
fn dynamic_attributes_version_and_matlib_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import numpy.matlib as np_matlib
def outcome(f):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            got = ("ok", f())
        except AttributeError as ex:
            text = str(ex)
            got = ("AttributeError", text[text.find("\n"):] if "\n" in text else "`" in text)
        except Exception as ex:
            got = (type(ex).__name__, str(ex)[:60])
    return got + (sorted({w.category.__name__ for w in caught}),)
def matlib_of(m):
    return m.matlib if m is fnp else np_matlib
cases = {}
for name in ("float", "complex", "int", "object", "str", "bytes", "float_", "unicode_"):
    cases[f"np.{name}"] = lambda m, name=name: repr(getattr(m, name))
cases["np.chararray"] = lambda m: m.chararray.__name__
cases["np.lib.math"] = lambda m: m.lib.math.__name__
cases["plain miss"] = lambda m: getattr(m, "definitely_not_an_attribute")
cases["matlib empty"] = lambda m: (lambda x: (type(x).__name__, x.shape))(matlib_of(m).empty((2,)))
cases["matlib ones"] = lambda m: repr(matlib_of(m).ones((2, 3)))
cases["matlib zeros"] = lambda m: repr(matlib_of(m).zeros(2, dtype=int))
cases["matlib identity"] = lambda m: repr(matlib_of(m).identity(3))
cases["matlib eye"] = lambda m: repr(matlib_of(m).eye(2, 3, k=1))
cases["matlib rand"] = lambda m: (lambda x: (type(x).__name__, x.shape))(matlib_of(m).rand((2, 3)))
cases["matlib randn"] = lambda m: (lambda x: (type(x).__name__, x.shape))(matlib_of(m).randn(3))
cases["matlib repmat"] = lambda m: repr(matlib_of(m).repmat(np.arange(3), 2, 2))
cases["matlib repmat 0-d"] = lambda m: repr(matlib_of(m).repmat(5, 2, 3))
cases["matlib all"] = lambda m: sorted(matlib_of(m).__all__) == sorted(["rand", "randn", "repmat"] + list(np.__all__))
bad = []
for label, f in cases.items():
    ours, theirs = outcome(lambda: f(fnp)), outcome(lambda: f(np))
    if ours != theirs:
        bad.append(f"{label}: fnp={ours} numpy={theirs}")
import numpy.version as np_version
names = lambda mod: sorted(k for k in dir(mod) if not k.startswith("_"))
if names(fnp.version) != names(np_version):
    bad.append(f"version names {names(fnp.version)} != {names(np_version)}")
if fnp.version.short_version != fnp.__version__ or fnp.version.version != fnp.__version__:
    bad.append("version strings must be fnp's __version__")
miss = outcome(lambda: getattr(fnp, "definitely_not_an_attribute"))
try:
    getattr(fnp, "definitely_not_an_attribute")
except AttributeError as ex:
    if fnp.__name__ not in str(ex):
        bad.append(f"a plain miss must name fnp's module: {ex}")
print(len(cases) + 2, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    expect_equal(
        last,
        "23 []",
        &format!("numpy's dynamic attributes, version and matlib; output: {result}"),
    )
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
    # fnp's wrappers for these had become a straight call to numpy's (deadlock-audit-1uf80)
    'reshape', 'transpose', 'swapaxes', 'moveaxis', 'rollaxis', 'squeeze', 'expand_dims',
    'empty_like',
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
/// deadlock-audit-qxy9u), 81 cells compared by outcome and warnings:
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
///
/// 26 of the 68 cells failed before the fix (numpy 2.4.3); 0 fail after, on numpy 2.4.3 and
/// 2.3.5. Thirteen later cells cover `np._NoValue` passed to an ma flag (numpy's "not
/// passed"), np.random called by numpy's parameter names, and pickling submodule functions: 5
/// failed before (`ma.argmax(keepdims=np._NoValue)` read the sentinel as truthy and kept the
/// axis; `random.sample` was named `random_sample`; `strings.upper`, `strings.slice` and
/// `char.find` could not be pickled), 0 after, on numpy 2.4.3 and 2.3.5.
#[test]
fn submodule_calls_match_numpys_parameters_and_outcomes() -> Result<(), String> {
    let script = fnp_script(
        r##"
import pickle, warnings

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
    # `np._NoValue` is numpy's "not passed", not a truthy flag.
    "ma.argmax keepdims=_NoValue": lambda m: m.ma.argmax(x, axis=1, keepdims=np._NoValue),
    "ma.count keepdims=_NoValue": lambda m: m.ma.count(x, axis=1, keepdims=np._NoValue),
    "ma.average keepdims=_NoValue": lambda m: m.ma.average(x, axis=1, keepdims=np._NoValue),
    "ma.mask_cols axis=_NoValue": lambda m: m.ma.mask_cols(x, axis=np._NoValue),
    # numpy.random, called by numpy's parameter names; ranf/sample are functions of their own.
    "random.binomial kw": lambda m: (m.random.seed(3), m.random.binomial(n=10, p=0.5, size=4))[1],
    "random.choice kw": lambda m: (m.random.seed(3), m.random.choice(a=5, size=3, replace=False))[1],
    "random.normal kw": lambda m: (m.random.seed(3), m.random.normal(loc=1.0, scale=2.0, size=3))[1],
    "random.ranf kw": lambda m: (m.random.seed(3), m.random.ranf(size=3))[1],
    "random.sample name": lambda m: m.random.sample.__name__,
    # Functions pickle by reference (multiprocessing, joblib): fnp.strings/char claimed numpy's
    # module name, so pickle resolved every native to numpy's object and refused it.
    **{f"{mod}.{fn} pickles": (lambda m, mod=mod, fn=fn: pickle.loads(pickle.dumps(getattr(getattr(m, mod), fn))) is getattr(getattr(m, mod), fn))
       for mod, fn in (("strings", "upper"), ("strings", "slice"), ("char", "find"))},
    # numpy <= 2.3's `ma.count` is a `_frommethod` instance and pickles by value, so identity is
    # version-dependent; the round-tripped function must still work.
    "ma.count pickles": lambda m: pickle.loads(pickle.dumps(m.ma.count))(x, axis=1),
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
        "81 []",
        &format!("submodule calls must match numpy's parameters and outcomes; output: {result}"),
    )
}

/// `inspect.signature` parity for every callable of fft, linalg, char, strings, testing, emath,
/// rec, ma and random (the live numpy's; a numpy builtin without a signature is skipped). As in
/// the top-level lock, a CLASS-valued numpy default (`randint(dtype=int)`, `ma.make_mask(dtype=
/// np.bool)`, `ma.masked_all(dtype=float)`) compares on names and kinds only: a builtin's text
/// signature carries only constants. np.random's functions are `RandomState` methods, which
/// rendered as `(*args, **kwargs)` or with `loc=Ellipsis` for their sentinel defaults; seven
/// ma functions and `strings.slice` have `np._NoValue`/`np.False_` defaults and are now thin
/// wrappers carrying numpy's signature. 27 of the 105 callables of the first seven submodules
/// differed before their fix, and 36 of the 210 callables of all nine before this one (numpy
/// 2.4.3; functions fnp re-exports from numpy are skipped); 0 after, on numpy 2.4.3 and 2.3.5.
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

def same_but_class_defaults(got, want):
    g, w = list(got.parameters.values()), list(want.parameters.values())
    return len(g) == len(w) and any(isinstance(p.default, type) for p in w) and all(
        a.name == b.name and a.kind == b.kind and (isinstance(b.default, type) or a.default == b.default)
        for a, b in zip(g, w))

checked, bad = 0, []
for sub in ("fft", "linalg", "char", "strings", "testing", "emath", "rec", "ma", "random"):
    nm, fm = getattr(np, sub), getattr(fnp, sub)
    for name in getattr(nm, "__all__", [n for n in dir(nm) if not n.startswith("_")]):
        nf, ff = getattr(nm, name, None), getattr(fm, name, None)
        if not callable(nf) or isinstance(nf, (type, np.ufunc)) or ff is None or ff is nf:
            continue
        want = signature(nf)
        if want is None:
            continue
        checked += 1
        got = signature(ff)
        if got is not None and (got == want or same_but_class_defaults(got, want)):
            continue
        bad.append(f"{sub}.{name}: numpy{want} fnp{got}")
print(checked, bad)
print(checked >= 200 and not bad)
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

/// How each `numpy.__all__` callable BINDS a call numpy refuses, compared by outcome: an
/// unknown keyword, and its first keyword-only parameter passed positionally. The signature
/// lock cannot see this - a NEP 18 dispatcher reports numpy's signature whatever its native
/// function binds - and fnp answered both: `nanstd(a, 0, None, None, 0, False, where)`, `isin(..,
/// kind)` and `count_nonzero(a, None, True)` took keyword-only parameters positionally,
/// `cumulative_sum(x, bogus=1)` ignored the keyword, and `convolve(a, v, bogus=1)` read it as an
/// omitted `mode`. 11 of the 262 cells failed before the fix (numpy 2.4.3; on 2.3.5 also
/// `corrcoef` and `in1d`); 0 after, on numpy 2.4.3 (262) and 2.3.5 (233).
#[test]
fn numpy_all_calls_numpy_refuses_are_refused() -> Result<(), String> {
    let script = fnp_script(
        r##"
import inspect, warnings

A2 = np.array([[3.0, 1.0, 2.0], [0.5, 4.0, 1.5]])
SAMPLE = {
    "a": A2, "x": np.array([1.0, 2.5, 4.0]), "y": np.array([2.0, 0.5, 1.0]), "arr": A2, "ary": A2,
    "x1": np.array([1.0, 2.0, 3.0]), "x2": np.array([2.0, 2.0, 2.0]), "b": np.array([1.0, 0.0, 2.0]),
    "v": np.array([1.0, 2.0]), "m": A2, "array": A2, "ar": np.array([3, 1, 2, 3]),
    "ar1": np.array([1, 2, 3]), "ar2": np.array([2, 3, 4]), "element": np.array([1, 5]),
    "test_elements": np.array([1, 2]), "p": np.array([1.0, -2.0, 1.0]), "c": np.array([1.0, 2.0]),
    "q": 0.5, "n": 3, "N": 3, "shape": (2, 3), "dtype": np.float64, "tup": (np.ones(2), np.zeros(2)),
    "arrays": (np.ones(2), np.zeros(2)), "condition": np.array([True, False, True]), "indices": np.array([0, 1]),
    "fill_value": 7.0, "start": 1.0, "stop": 10.0, "num": 5, "obj": 1, "values": np.array([9.0]),
    "axis": 0, "source": 0, "destination": 1, "axes": (1, 0), "newshape": (3, 2), "repeats": 2,
    "reps": 2, "pad_width": 1, "decimals": 1, "k": 1, "bins": 3, "weights": None, "val": 1.0,
    "func": np.sum, "func1d": np.sum, "subscripts": "ij->ji", "operands": A2, "fname": None,
    "object": [1, 2, 3], "prototype": A2, "a_min": 1.0, "a_max": 2.0, "sorter": None, "side": "left",
    "kth": 1, "choicelist": [np.array([1, 2, 3])], "condlist": [np.array([True, False, True])],
    "mask": np.array([True, False]), "vals": np.array([0.0]), "ind": np.array([0]), "xp": [0.0, 1.0],
    "fp": [0.0, 10.0], "dims": (2, 3), "multi_index": (np.array([1]), np.array([2])),
    "f": lambda i, j: i + j, "old_behavior": False, "seq": [1, 2], "precision": 3,
}
# I/O, printing, global state, and the uninitialised-memory constructors.
SKIP = {"fromfile", "fromregex", "genfromtxt", "load", "loadtxt", "save", "savez", "savez_compressed",
        "savetxt", "memmap", "set_printoptions", "printoptions", "seterr", "setbufsize", "seterrcall",
        "get_include", "show_config", "show_runtime", "info", "test", "vectorize", "frompyfunc",
        "nditer", "nested_iters", "errstate", "from_dlpack", "empty", "empty_like"}

def outcome(call):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            call()
            return "ok"
        except Exception as ex:
            return type(ex).__name__

cases = {}
for fn in sorted(np.__all__):
    nf = getattr(np, fn, None)
    if fn in SKIP or not callable(nf) or isinstance(nf, (type, np.ufunc)):
        continue
    try:
        params = list(inspect.signature(nf).parameters.values())
    except (TypeError, ValueError):
        continue
    if any(p.kind is p.VAR_KEYWORD for p in params):
        continue
    required = [p for p in params if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if any(p.name not in SAMPLE for p in required):
        continue
    cases[f"{fn} bogus keyword"] = (fn, [SAMPLE[p.name] for p in required], {"bogus_kw": 1})
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    kwonly = [p for p in params if p.kind is p.KEYWORD_ONLY]
    if kwonly and not any(p.kind is p.VAR_POSITIONAL for p in params):
        # Every positional slot filled (its default, or the sample when required), then one more.
        full = [SAMPLE[p.name] if p.default is inspect.Parameter.empty else p.default for p in positional]
        default = kwonly[0].default
        extra = SAMPLE.get(kwonly[0].name, None if default in (inspect.Parameter.empty, np._NoValue) else default)
        cases[f"{fn} {kwonly[0].name} positional"] = (fn, full + [extra], {})

bad = []
for name, (fn, args, kwargs) in cases.items():
    ours = outcome(lambda: getattr(fnp, fn)(*args, **kwargs))
    theirs = outcome(lambda: getattr(np, fn)(*args, **kwargs))
    if ours != theirs:
        bad.append(f"{name}: fnp={ours} numpy={theirs}")
print(len(cases), bad)
"##
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    expect_equal(
        last.split_once(' ').map_or("", |(_, bad)| bad),
        "[]",
        &format!("calls numpy refuses must be refused; output: {result}"),
    )
}

/// Every `numpy.__all__` callable that takes `axis`, called with numpy's whole axis surface on a
/// 3-D operand - negative, out of range (AxisError), tuples incl. a repeated axis, `np.int64`,
/// `True`, `1.0`, None - compared by outcome, dtype, bytes and warnings. numpy's axis converter
/// refuses a bool; fnp's native routes read `axis` by integer extraction and answered
/// `np.sum(a, axis=True)` as axis 1 (32 functions). A tuple axis was also mishandled:
/// `nanargmax(a, axis=(0,))` answered where numpy raises, and `cumulative_sum`/`unstack`
/// raised TypeError where numpy accepts `(0,)` (and raises ValueError for longer tuples). 45 of
/// the 910 cells failed before the fix (numpy 2.4.3); 0 after, on numpy 2.4.3 (910) and 2.3.5
/// (868).
#[test]
fn numpy_all_axis_arguments_mean_what_numpy_means() -> Result<(), String> {
    let script = fnp_script(
        r##"
import inspect, warnings

A3 = np.arange(24.0).reshape(2, 3, 4) % 7 - 2.5
SAMPLE = {
    "a": A3, "x": A3, "arr": A3, "ary": A3, "m": A3, "array": A3, "y": A3 + 1, "x1": A3, "x2": A3 + 1,
    "q": 0.5, "n": 3, "N": 3, "indices": np.array([0, 1]), "repeats": 2, "shift": 1,
    "condition": np.array([True, False]), "prepend": None, "values": np.array([9.0]), "obj": 1,
    "ind": np.array([0]), "v": np.array([1.0, 2.0]), "b": A3 + 1, "arrays": [A3, A3], "tup": (A3, A3),
    "kth": 1, "p": np.array([1.0, 2.0]), "decimals": 1, "k": 1, "bins": 3,
}
# I/O, callables-as-arguments, and the uninitialised-memory / sequence constructors.
SKIP = {"einsum", "vectorize", "memmap", "fromfile", "loadtxt", "genfromtxt", "savetxt", "save", "savez",
        "savez_compressed", "apply_along_axis", "apply_over_axes", "piecewise", "frombuffer", "fromiter",
        "fromstring", "fromfunction", "empty", "empty_like", "linspace", "logspace", "geomspace"}
AXES = [0, -1, 2, -3, 3, -4, (0, 2), (0,), (2, 0), np.int64(1), True, 1.0, None, (0, 0)]

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            if isinstance(r, (tuple, list)) and r and all(isinstance(x, np.ndarray) for x in r):
                got = ("seq", type(r).__name__) + tuple((x.dtype.str, x.shape, x.tobytes()) for x in r)
            elif isinstance(r, (np.ndarray, np.generic)) or np.isscalar(r):
                a = np.asarray(r)
                got = ("ok", type(r).__name__, a.dtype.str, a.shape, a.tobytes() if a.dtype != object else repr(a.tolist()))
            else:
                got = ("ok", type(r).__name__)
        except Exception as ex:
            got = (type(ex).__name__,)
    return got + (sorted({w.category.__name__ for w in caught}),)

cases = {}
for fn in sorted(np.__all__):
    nf = getattr(np, fn, None)
    if fn in SKIP or not callable(nf) or isinstance(nf, (type, np.ufunc)):
        continue
    try:
        params = inspect.signature(nf).parameters
    except (TypeError, ValueError):
        continue
    if "axis" not in params:
        continue
    required = [p for p in params.values() if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != "axis"]
    if any(p.name not in SAMPLE for p in required):
        continue
    args = [SAMPLE[p.name] for p in required]
    for axis in AXES:
        cases[f"{fn} axis={axis!r}"] = (fn, args, axis)

bad = []
for name, (fn, args, axis) in cases.items():
    ours = outcome(lambda: getattr(fnp, fn)(*args, axis=axis))
    theirs = outcome(lambda: getattr(np, fn)(*args, axis=axis))
    if ours != theirs:
        bad.append(f"{name}: fnp={str(ours)[:110]} numpy={str(theirs)[:110]}")
print(len(cases), bad)
"##
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    expect_equal(
        last.split_once(' ').map_or("", |(_, bad)| bad),
        "[]",
        &format!("axis arguments must mean what numpy means; output: {result}"),
    )
}

/// Every `numpy.__all__` callable that takes `keepdims`, called with it spelled True, False, 1,
/// 0, None, np.True_, np.False_, 2 and 'yes', crossed with axis None / 0 / -1 / (0, 2) on a 3-D
/// operand. The bool-flag and explicit-default sweeps skip `keepdims`, whose numpy default is
/// `np._NoValue`. numpy's `count_nonzero` reads it twice - `if axis is None and not keepdims`
/// (truthiness) returns early, anything else goes to `.sum(keepdims=...)`, which takes an
/// integer only - and fnp read it by truthiness everywhere, answering `keepdims=None`,
/// `np.True_` and 'yes' where numpy raises TypeError. 14 of the 1,116 cells failed before the
/// fix (numpy 2.4.3 and 2.3.5); 0 after on both.
#[test]
fn numpy_all_keepdims_spellings_mean_what_numpy_means() -> Result<(), String> {
    let script = fnp_script(
        r##"
import inspect, warnings

A3 = np.arange(24.0).reshape(2, 3, 4) % 7 - 2.5
SAMPLE = {
    "a": A3, "x": A3, "arr": A3, "ary": A3, "m": A3, "array": A3, "y": A3 + 1, "x1": A3, "x2": A3 + 1,
    "q": 0.5, "n": 3, "N": 3, "indices": np.array([0, 1]), "repeats": 2, "shift": 1,
    "condition": np.array([True, False]), "prepend": None, "values": np.array([9.0]), "obj": 1,
    "ind": np.array([0]), "v": np.array([1.0, 2.0]), "b": A3 + 1, "arrays": [A3, A3], "tup": (A3, A3),
    "kth": 1, "p": np.array([1.0, 2.0]), "decimals": 1, "k": 1, "bins": 3,
}
KEEPS = [True, False, 1, 0, None, np.True_, np.False_, 2, "yes"]
AXES = [None, 0, -1, (0, 2)]

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            if isinstance(r, (tuple, list)) and r and all(isinstance(x, np.ndarray) for x in r):
                got = ("seq", type(r).__name__) + tuple((x.dtype.str, x.shape, x.tobytes()) for x in r)
            elif isinstance(r, (np.ndarray, np.generic)) or np.isscalar(r):
                a = np.asarray(r)
                got = ("ok", type(r).__name__, a.dtype.str, a.shape, a.tobytes() if a.dtype != object else repr(a.tolist()))
            else:
                got = ("ok", type(r).__name__)
        except Exception as ex:
            got = (type(ex).__name__,)
    return got + (sorted({w.category.__name__ for w in caught}),)

cases = {}
for fn in sorted(np.__all__):
    nf = getattr(np, fn, None)
    if not callable(nf) or isinstance(nf, (type, np.ufunc)):
        continue
    try:
        params = inspect.signature(nf).parameters
    except (TypeError, ValueError):
        continue
    if "keepdims" not in params:
        continue
    required = [p for p in params.values() if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name not in ("axis", "keepdims")]
    if any(p.name not in SAMPLE for p in required):
        continue
    args = [SAMPLE[p.name] for p in required]
    for keep in KEEPS:
        for axis in (AXES if "axis" in params else [None]):
            kw = {"keepdims": keep, "axis": axis} if "axis" in params else {"keepdims": keep}
            cases[f"{fn} keepdims={keep!r} axis={axis!r}"] = (fn, args, kw)

bad = []
for name, (fn, args, kw) in cases.items():
    ours = outcome(lambda: getattr(fnp, fn)(*args, **kw))
    theirs = outcome(lambda: getattr(np, fn)(*args, **kw))
    if ours != theirs:
        bad.append(f"{name}: fnp={str(ours)[:110]} numpy={str(theirs)[:110]}")
print(len(cases), bad)
"##
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    expect_equal(
        last.split_once(' ').map_or("", |(_, bad)| bad),
        "[]",
        &format!("keepdims spellings must mean what numpy means; output: {result}"),
    )
}

/// The submodule twin of `numpy_all_explicit_defaults_and_none_mean_what_numpy_means`: every
/// defaulted parameter of linalg / fft / ma / char / strings / testing / emath callables passed
/// explicitly as its default and as None, compared by outcome, warnings and exception type.
/// Float results are rounded to 12 decimals: the cells are about argument handling, and a
/// native 2x2 `pinv`/`lstsq` differs from LAPACK in the last bit. `ma`'s `copy=` goes on to
/// `np.array(copy=...)`, where None is a value, and its `shrink`/`fill_value` flags are read
/// by truthiness; typed `bool`s refused `copy=None`/`shrink=None` in 14 functions. `eigh`/
/// `eigvalsh(UPLO=None)` and `qr(mode=None)` raised TypeError where numpy raises
/// AttributeError/ValueError. 19 of the 296 cells failed before the fix (numpy 2.4.3); 0 after,
/// on numpy 2.4.3 (296) and 2.3.5 (289).
#[test]
fn submodule_explicit_defaults_and_none_mean_what_numpy_means() -> Result<(), String> {
    let script = fnp_script(
        r##"
import inspect, warnings

A2 = np.array([[3.0, 1.0, 2.0], [0.5, 4.0, 1.5]])
SQ = np.array([[4.0, 1.0], [1.0, 3.0]])
MA = np.ma.array(A2, mask=[[0, 1, 0], [0, 0, 1]])
S = np.array(["abc", "aXbXc", "  x "])
BASE = {
    "x": np.array([1.0, 2.5, 4.0]), "y": np.array([2.0, 0.5, 1.0]), "q": 0.5, "n": 3, "N": 3,
    "shape": (2, 3), "dtype": np.float64, "axis": 0, "fill_value": 7.0, "sub": "X", "old": "X", "new": "-",
    "width": 7, "i": 3, "chars": None, "value": 2.0, "condition": np.array([True, False, True]),
    "b": np.array([1.0, 0.0]), "x1": np.array([1.0, 2.0, 3.0]), "x2": np.array([2.0, 2.0, 2.0]),
    "actual": np.array([1.0, 2.0]), "desired": np.array([1.0, 2.0]), "v1": 1.0, "v2": 1.0,
    "sep": ",", "size": 3, "values": np.array([9.0]), "v": np.array([1.0, 2.0]),
    "arr": A2, "ary": A2, "obj": MA, "mask": [True, False], "m": [0, 1],
    "seq": [A2, A2], "arrays": [A2, A2], "tup": (A2, A2), "ind": [0], "indices": [0, 1],
    "c": np.array([1.0, 2.0]), "pol": np.array([1.0, 2.0]), "p": np.array([1.0, 2.0]),
    "M": SQ, "A": SQ, "B": SQ, "decimals": 1, "shift": 1, "s": None, "k": 1,
}
PER_MODULE = {
    "linalg": {"a": SQ, "b": np.array([1.0, 2.0]), "x": SQ, "x1": SQ, "x2": SQ, "M": SQ},
    "fft": {"a": np.array([1.0, 2.0, 3.0, 4.0]), "x": np.array([1.0, 2.0, 3.0, 4.0]), "d": 1.0},
    "ma": {"a": MA, "x": MA, "arr": MA, "b": MA, "x1": MA, "x2": MA},
    "char": {"a": S, "x1": S, "x2": S, "b": S},
    "strings": {"a": S, "x1": S, "x2": S, "b": S},
    "testing": {"a": np.array([1.0, 2.0]), "b": np.array([1.0, 2.0]), "x": np.array([1.0]), "y": np.array([1.0])},
    "emath": {"x": np.array([4.0, -1.0]), "n": 2},
}
# Context managers, decorators, runners and helpers that take callables or mutate an argument.
SKIP = {("ma", "set_fill_value"), ("ma", "apply_along_axis"), ("ma", "apply_over_axes"), ("ma", "vander"),
        *(("testing", n) for n in ("assert_warns", "assert_raises", "assert_raises_regex", "assert_no_warnings",
                                   "rundocs", "run_threaded", "clear_and_catch_warnings", "tempdir", "temppath",
                                   "suppress_warnings", "measure", "decorate_methods", "break_cycles", "memusage",
                                   "print_assert_equal", "assert_no_gc_cycles", "assert_"))}

def canon(value):
    a = np.asarray(value)
    return np.round(a, 12) if a.dtype.kind in "fc" else a

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            if isinstance(r, np.ma.MaskedArray):
                got = ("masked", r.dtype.str, r.shape, canon(np.ma.getdata(r)).tobytes(), np.ma.getmaskarray(r).tobytes())
            elif isinstance(r, tuple):
                got = ("tuple",) + tuple((canon(x).dtype.str, canon(x).shape, canon(x).tobytes()) for x in r)
            elif isinstance(r, (np.ndarray, np.generic)) or np.isscalar(r):
                a = canon(r)
                got = ("ok", type(r).__name__, a.dtype.str, a.shape, repr(a.tolist()) if a.dtype == object else a.tobytes())
            else:
                got = ("ok", type(r).__name__)
        except Exception as ex:
            got = (type(ex).__name__,)
    return got + (sorted({w.category.__name__ for w in caught}),)

cases = {}
for sub in ("linalg", "fft", "ma", "char", "strings", "testing", "emath"):
    nm, fm = getattr(np, sub), getattr(fnp, sub)
    sample = {**BASE, **PER_MODULE.get(sub, {})}
    for name in getattr(nm, "__all__", [n for n in dir(nm) if not n.startswith("_")]):
        nf, ff = getattr(nm, name, None), getattr(fm, name, None)
        if (sub, name) in SKIP or not callable(nf) or isinstance(nf, (type, np.ufunc)) or ff is None or ff is nf:
            continue
        try:
            params = list(inspect.signature(nf).parameters.values())
        except (TypeError, ValueError):
            continue
        required = [p for p in params if p.default is inspect.Parameter.empty
                    and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if any(p.name not in sample for p in required):
            continue
        args = [sample[p.name] for p in required]
        for p in params:
            if p.default is inspect.Parameter.empty or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            if p.default is np._NoValue or isinstance(p.default, type) or p.kind is p.POSITIONAL_ONLY:
                continue
            for val in [p.default] + ([None] if p.default is not None else []):
                cases[f"{sub}.{name} {p.name}={val!r}"] = (
                    lambda m, sub=sub, name=name, args=args, kw={p.name: val}: getattr(getattr(m, sub), name)(*args, **kw))

bad = []
for label, call in cases.items():
    ours, theirs = outcome(lambda: call(fnp)), outcome(lambda: call(np))
    if ours != theirs:
        bad.append(f"{label}: fnp={str(ours)[:110]} numpy={str(theirs)[:110]}")
print(len(cases), bad)
print(len(cases) >= 250 and not bad)
"##
        .into(),
    );
    let result = numpy_oracle(&script)?;
    expect_equal(
        result.lines().last().unwrap_or("").trim(),
        "True",
        &format!(
            "submodule explicit defaults and None must mean what numpy means; output: {result}"
        ),
    )
}

/// Every bool flag of a `numpy.__all__` callable, passed as numpy's default spelled 0/1,
/// `np.bool_` and its negation (156 cells), compared by outcome and warnings. numpy's
/// Python-level functions read flags with `if flag:` (truthiness), and the `*_like` family's C
/// layer converts `subok` as an integer (1 works, `np.bool_` raises). PyO3's typed `bool`
/// accepted only `bool`/`np.bool_`, so `np.linspace(0, 1, 5, endpoint=0)`,
/// `np.percentile(a, 50, keepdims=1)`, `np.unique(a, return_counts=1)`,
/// `np.meshgrid(x, y, sparse=1)` and `np.median(a, overwrite_input=1)` raised TypeError where
/// numpy answers, and `np.zeros_like(a, subok=np.True_)` answered where numpy raises. 56 of the
/// 156 cells failed before the fix (numpy 2.4.3); 0 after, on numpy 2.4.3 and 2.3.5.
/// `digitize` is skipped: the shared sample `bins` is an int, valid for histogram and not for
/// digitize.
#[test]
fn numpy_all_bool_flags_take_numpys_truthy_spellings() -> Result<(), String> {
    let script = fnp_script(
        r##"
import inspect, warnings

A2 = np.array([[3.0, 1.0, 2.0], [0.5, 4.0, 1.5]])
SAMPLE = {
    "a": A2, "x": np.array([1.0, 2.5, 4.0]), "y": np.array([2.0, 0.5, 1.0]), "arr": A2, "ary": A2,
    "x1": np.array([1.0, 2.0, 3.0]), "x2": np.array([2.0, 2.0, 2.0]), "b": np.array([1.0, 0.0, 2.0]),
    "v": np.array([1.0, 2.0]), "m": A2, "array": A2, "ar": np.array([3, 1, 2, 3]),
    "ar1": np.array([1, 2, 3]), "ar2": np.array([2, 3, 4]), "element": np.array([1, 5]),
    "test_elements": np.array([1, 2]), "p": np.array([1.0, -2.0, 1.0]), "c": np.array([1.0, 2.0]),
    "q": 0.5, "n": 3, "N": 3, "shape": (2, 3), "dtype": np.float64, "tup": (np.ones(2), np.zeros(2)),
    "arrays": (np.ones(2), np.zeros(2)), "condition": np.array([True, False, True]), "indices": np.array([0, 1]),
    "fill_value": 7.0, "start": 0.0, "stop": 1.0, "num": 5, "obj": 1, "values": np.array([9.0]),
    "axis": 0, "source": 0, "destination": 1, "axes": (1, 0), "newshape": (3, 2), "repeats": 2,
    "reps": 2, "pad_width": 1, "decimals": 1, "k": 1, "bins": 3, "weights": None, "val": 1.0,
    "func": np.sum, "func1d": np.sum, "subscripts": "ij->ji", "operands": A2, "fname": None,
    "object": [1, 2, 3], "prototype": A2, "a_min": 1.0, "a_max": 2.0, "sorter": None, "side": "left",
    "kth": 1, "choicelist": [np.array([1, 2, 3])], "condlist": [np.array([True, False, True])],
    "mask": np.array([True, False]), "vals": np.array([0.0]), "ind": np.array([0]), "xp": [0.0, 1.0],
    "fp": [0.0, 10.0], "dims": (2, 3), "multi_index": (np.array([1]), np.array([2])),
    "f": lambda i, j: i + j, "old_behavior": False, "seq": [1, 2], "precision": 3,
}
SKIP = {"digitize", "fromfile", "fromregex", "genfromtxt", "load", "loadtxt", "save", "savez", "savez_compressed",
        "savetxt", "memmap", "set_printoptions", "printoptions", "seterr", "setbufsize", "seterrcall",
        "get_include", "show_config", "show_runtime", "info", "test", "vectorize", "frompyfunc",
        "fromfunction", "apply_along_axis", "apply_over_axes", "piecewise", "nditer", "nested_iters",
        "einsum", "einsum_path", "errstate", "from_dlpack", "fromstring", "frombuffer", "fromiter",
        "empty", "empty_like", "require", "busday_count", "busday_offset", "is_busday", "datetime_as_string"}

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            if isinstance(r, tuple):
                got = ("tuple",) + tuple((np.asarray(x).dtype.str, np.asarray(x).shape, np.asarray(x).tobytes()) for x in r)
            elif isinstance(r, (np.ndarray, np.generic)) or np.isscalar(r):
                a = np.asarray(r)
                data = repr(a.tolist()) if a.dtype == object else a.tobytes()
                got = ("ok", type(r).__name__, a.dtype.str, a.shape, data)
            else:
                got = ("ok", type(r).__name__)
        except Exception as ex:
            got = (type(ex).__name__, str(ex)[:100])
    return got + (sorted({w.category.__name__ for w in caught}),)

cases = {}
for fn in sorted(np.__all__):
    nf = getattr(np, fn, None)
    if fn in SKIP or not callable(nf) or isinstance(nf, (type, np.ufunc)):
        continue
    try:
        params = list(inspect.signature(nf).parameters.values())
    except (TypeError, ValueError):
        continue
    required = [p for p in params if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if any(p.name not in SAMPLE for p in required):
        continue
    args = [SAMPLE[p.name] for p in required]
    for p in params:
        if p.default is inspect.Parameter.empty or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        if p.default is np._NoValue or isinstance(p.default, type):
            continue
        if p.kind is p.POSITIONAL_ONLY:
            continue
        if isinstance(p.default, bool):
            for alt in (int(p.default), np.bool_(p.default), int(not p.default)):
                cases[f"{fn} {p.name}={alt!r}"] = (fn, args, {p.name: alt})

bad = []
for name, (fn, args, kwargs) in cases.items():
    ours = outcome(lambda: getattr(fnp, fn)(*args, **kwargs))
    theirs = outcome(lambda: getattr(np, fn)(*args, **kwargs))
    if ours != theirs:
        bad.append(f"{name}: fnp={str(ours)[:120]} numpy={str(theirs)[:120]}")
print(len(cases), bad)
"##
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    expect_equal(
        last.split_once(' ').map_or("", |(_, bad)| bad),
        "[]",
        &format!("bool flags must take numpy's spellings; output: {result}"),
    )
}

/// Every defaulted parameter of a `numpy.__all__` callable passed EXPLICITLY - as its own
/// default, and as None - compared by outcome, warnings and exception TYPE (PyO3 prefixes its
/// own conversion errors with the argument name, so messages are not compared). A defaulted
/// PyO3 `Option` cannot tell an explicit None from an omitted argument, and a typed `&str`/`i64`
/// refuses None outright, so fnp answered where numpy raises (`percentile(method=None)` ran
/// 'linear', `histogram(bins=None)` used 10 bins, `stack(axis=None)` stacked on axis 0,
/// `tensordot(axes=None)` contracted two axes) and raised where numpy answers (`copy`/`ravel`/
/// `reshape`/`full`/the `*_like` family with `order=None`, `take`/`put` with `mode=None`,
/// `shares_memory(max_work=None)`). Plus the named cases the shared samples cannot reach, and
/// numpy <= 2.3's `interpolation=` (the sweep iterates the INSTALLED numpy's signatures, so on
/// 2.3.5 it also found `insert` warning where numpy does not: `extract::<f64>` converted a
/// 1-element array). 37 of the 590 cells failed before the fix (numpy 2.4.3); 0 after, on
/// numpy 2.4.3 (590 cells) and 2.3.5 (542).
#[test]
fn numpy_all_explicit_defaults_and_none_mean_what_numpy_means() -> Result<(), String> {
    let script = fnp_script(
        r##"
import inspect, warnings

A2 = np.array([[3.0, 1.0, 2.0], [0.5, 4.0, 1.5]])
SAMPLE = {
    "a": A2, "x": np.array([1.0, 2.5, 4.0]), "y": np.array([2.0, 0.5, 1.0]), "arr": A2, "ary": A2,
    "x1": np.array([1.0, 2.0, 3.0]), "x2": np.array([2.0, 2.0, 2.0]), "b": np.array([1.0, 0.0, 2.0]),
    "v": np.array([1.0, 2.0]), "m": A2, "array": A2, "ar": np.array([3, 1, 2, 3]),
    "ar1": np.array([1, 2, 3]), "ar2": np.array([2, 3, 4]), "element": np.array([1, 5]),
    "test_elements": np.array([1, 2]), "p": np.array([1.0, -2.0, 1.0]), "c": np.array([1.0, 2.0]),
    "q": 0.5, "n": 3, "N": 3, "shape": (2, 3), "dtype": np.float64, "tup": (np.ones(2), np.zeros(2)),
    "arrays": (np.ones(2), np.zeros(2)), "condition": np.array([True, False, True]), "indices": np.array([0, 1]),
    "fill_value": 7.0, "start": 1.0, "stop": 10.0, "num": 5, "obj": 1, "values": np.array([9.0]),
    "axis": 0, "source": 0, "destination": 1, "axes": (1, 0), "newshape": (3, 2), "repeats": 2,
    "reps": 2, "pad_width": 1, "decimals": 1, "k": 1, "bins": 3, "weights": None, "val": 1.0,
    "func": np.sum, "func1d": np.sum, "subscripts": "ij->ji", "operands": A2, "fname": None,
    "object": [1, 2, 3], "prototype": A2, "a_min": 1.0, "a_max": 2.0, "sorter": None, "side": "left",
    "kth": 1, "choicelist": [np.array([1, 2, 3])], "condlist": [np.array([True, False, True])],
    "mask": np.array([True, False]), "vals": np.array([0.0]), "ind": np.array([0]), "xp": [0.0, 1.0],
    "fp": [0.0, 10.0], "dims": (2, 3), "multi_index": (np.array([1]), np.array([2])),
    "f": lambda i, j: i + j, "old_behavior": False, "seq": [1, 2], "precision": 3,
}
# I/O, printing and global-state functions, and those whose required arguments the samples
# cannot satisfy; `empty`/`empty_like` return uninitialised memory.
SKIP = {"fromfile", "fromregex", "genfromtxt", "load", "loadtxt", "save", "savez", "savez_compressed",
        "savetxt", "memmap", "set_printoptions", "printoptions", "seterr", "setbufsize", "seterrcall",
        "get_include", "show_config", "show_runtime", "info", "test", "vectorize", "frompyfunc",
        "fromfunction", "apply_along_axis", "apply_over_axes", "piecewise", "nditer", "nested_iters",
        "einsum", "einsum_path", "errstate", "from_dlpack", "fromstring", "frombuffer", "fromiter",
        "empty", "empty_like", "require", "busday_count", "busday_offset", "is_busday", "datetime_as_string"}

def outcome(call, type_only=True):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            if isinstance(r, tuple):
                got = ("tuple",) + tuple((np.asarray(x).dtype.str, np.asarray(x).shape, np.asarray(x).tobytes()) for x in r)
            elif isinstance(r, (np.ndarray, np.generic)) or np.isscalar(r):
                a = np.asarray(r)
                data = repr(a.tolist()) if a.dtype == object else a.tobytes()
                got = ("ok", type(r).__name__, a.dtype.str, a.shape, data)
            else:
                got = ("ok", type(r).__name__)
        except Exception as ex:
            got = (type(ex).__name__,) if type_only else (type(ex).__name__, str(ex))
    return got + (sorted({w.category.__name__ for w in caught}),)

cases = {}
for fn in sorted(np.__all__):
    nf = getattr(np, fn, None)
    if fn in SKIP or not callable(nf) or isinstance(nf, (type, np.ufunc)):
        continue
    try:
        params = list(inspect.signature(nf).parameters.values())
    except (TypeError, ValueError):
        continue
    required = [p for p in params if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if any(p.name not in SAMPLE for p in required):
        continue
    args = [SAMPLE[p.name] for p in required]
    for p in params:
        if p.default is inspect.Parameter.empty or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        if p.default is np._NoValue or isinstance(p.default, type) or p.kind is p.POSITIONAL_ONLY:
            continue
        cases[f"{fn} {p.name}={p.default!r}"] = (lambda m, fn=fn, args=args, kw={p.name: p.default}: getattr(m, fn)(*args, **kw), True)
        if p.default is not None:
            cases[f"{fn} {p.name}=None"] = (lambda m, fn=fn, args=args, kw={p.name: None}: getattr(m, fn)(*args, **kw), True)

NAN2 = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.5]])
named = {
    # `mean=None` raises only once a NaN is present (numpy's no-NaN route hands it to `var`).
    "nanvar mean=None": lambda m: m.nanvar(NAN2, mean=None),
    "nanstd mean=None": lambda m: m.nanstd(NAN2, mean=None),
    "nanvar correction=None": lambda m: m.nanvar(NAN2, correction=None),
    "nanstd ddof=None int": lambda m: m.nanstd(np.arange(6), ddof=None),
    "nanquantile method=None": lambda m: m.nanquantile(NAN2, 0.5, method=None),
    "quantile method=linear": lambda m: m.quantile(A2, 0.3, method="linear"),
    # numpy <= 2.3's `interpolation=` (None is its default there, a TypeError on 2.4+), and the
    # keyword-only `weights`.
    "nanpercentile interpolation=None": lambda m: m.nanpercentile(NAN2, 50, interpolation=None),
    "quantile interpolation=nearest": lambda m: m.quantile(A2, 0.3, interpolation="nearest"),
    "percentile weights positional": lambda m: m.percentile(A2, 50, None, None, False, "inverted_cdf", False, np.ones_like(A2)),
    "stack unknown keyword": lambda m: m.stack([np.ones(3), np.zeros(3)], bogus=1),
    "stack axis=0 explicit": lambda m: m.stack([np.ones(3), np.zeros(3)], axis=0),
    "tensordot axes=None": lambda m: m.tensordot(np.ones((2, 3)), np.ones((2, 3)), axes=None),
    "tensordot axes=1": lambda m: m.tensordot(np.ones((2, 3)), np.ones((3, 2)), axes=1),
    "histogram2d bins=None positional": lambda m: m.histogram2d(A2[0], A2[1], None),
    "histogram2d range, bins=None": lambda m: m.histogram2d(A2[0], A2[1], bins=None, range=[[0, 5], [0, 5]]),
    "unwrap period=None": lambda m: m.unwrap(np.array([0.0, 4.0, 8.0]), period=None),
    "put mode=None": lambda m: m.put(np.zeros(3), [5], [1.0], mode=None),
    "take mode=None out of range": lambda m: m.take(np.arange(3.0), [5], mode=None),
    "unravel_index order=None": lambda m: m.unravel_index(np.array([4, 1]), (2, 3), order=None),
    "full_like order=None F": lambda m: m.full_like(np.asfortranarray(A2), 2.0, order=None),
}
# Messages that are numpy's own compare in full: a delegate must call numpy's function by the
# name the caller used (`np.amin is not np.min`).
exact = {
    "amin dtype=": lambda m: m.amin(A2, dtype=np.float64),
    "amax ddof=": lambda m: m.amax(A2, ddof=1),
    "meshgrid indexing=None": lambda m: m.meshgrid(A2[0], A2[1], indexing=None),
    "meshgrid indexing=1": lambda m: m.meshgrid(A2[0], A2[1], indexing=1),
    "meshgrid indexing=np.str_": lambda m: m.meshgrid(A2[0], A2[1], indexing=np.str_("ij")),
    "percentile method=None": lambda m: m.percentile(A2, 50, method=None),
}
cases.update({k: (v, True) for k, v in named.items()})
cases.update({k: (v, False) for k, v in exact.items()})

bad = []
for name, (call, type_only) in cases.items():
    ours, theirs = outcome(lambda: call(fnp), type_only), outcome(lambda: call(np), type_only)
    if ours != theirs:
        bad.append(f"{name}: fnp={str(ours)[:120]} numpy={str(theirs)[:120]}")
print(len(cases), bad)
"##
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    expect_equal(
        last.split_once(' ').map_or("", |(_, bad)| bad),
        "[]",
        &format!("explicit defaults and None must mean what numpy means; output: {result}"),
    )
}
