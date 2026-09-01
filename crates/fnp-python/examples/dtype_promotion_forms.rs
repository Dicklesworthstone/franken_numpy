//! Dtype-only audit of the FORMS the existing promotion audit does not cover
//! (`deadlock-audit-oxxzf`): `out=`, `dtype=` overrides, `axis=`, 0-d and EMPTY operands, and
//! the `*_like` / `asarray` / `astype` family.
//!
//! WHY DTYPE-ONLY. `sort_complex` once returned complex128 where numpy returns complex64 with
//! the VALUES IDENTICAL - a class no value-comparing test in this repo could see. The audit
//! that found it asserts output dtype and python `type()` and nothing else, and is clean at
//! 0 of 1800 on the forms it covers. These are the forms it does not.
//!
//! The script it was meant to extend (scratchpad/dtype_promotion_audit.py) no longer exists -
//! it lived in another agent's scratchpad - so this lands the same idea where it survives: an
//! in-process example that needs no `.so` on disk and runs under rch.
//!
//!     rch exec --job -- cargo run --release -j4 -p fnp-python --example dtype_promotion_forms

use pyo3::prelude::*;
use std::ffi::CString;

use fnp_python::fnp_python;

const HARNESS: &str = r#"
import faulthandler, hashlib, os, sys

faulthandler.enable()

def out(*items):
    print(*items, file=sys.stderr, flush=True)

import numpy as np
import fnp_python as fnp

out("python", sys.version.split()[0], "| numpy", np.__version__)
out("host", os.uname().nodename)
out("in-process ELF sha256", hashlib.sha256(open(EXE_PATH, "rb").read()).hexdigest())

DTYPES = ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
          "float16", "float32", "float64", "complex64", "complex128")

rows, checked, skipped = [], 0, 0

def describe(value):
    # The two things this audit asserts, and nothing else.
    dtype = getattr(value, "dtype", None)
    return (type(value).__name__, None if dtype is None else str(dtype))

def check(label, call):
    # `call` runs against a module; both arms get their OWN operands, since `out=` mutates.
    global checked, skipped
    try:
        theirs = call(np)
    except Exception:
        skipped += 1
        return
    try:
        ours = call(fnp)
    except Exception as exception:
        rows.append((label, "%s: %s" % (type(exception).__name__, str(exception)[:60]),
                     "%s %s" % describe(theirs)))
        checked += 1
        return
    checked += 1
    theirs_shape, ours_shape = describe(theirs), describe(ours)
    if theirs_shape != ours_shape:
        rows.append((label, "%s %s" % ours_shape, "%s %s" % theirs_shape))

def data(dtype, n=4):
    return np.arange(1, n + 1).astype(dtype)

# ---------------------------------------------------------------------------
# A. `out=` GOVERNS THE RESULT DTYPE, and the result IS `out`.
# ---------------------------------------------------------------------------
for in_dtype in DTYPES:
    for out_dtype in DTYPES:
        for op in ("add", "multiply", "maximum"):
            def call(module, o=op, i=in_dtype, d=out_dtype):
                buffer = np.zeros(4, dtype=d)
                return getattr(module, o)(data(i), data(i), buffer)
            check("out= %s(%s) -> %s" % (op, in_dtype, out_dtype), call)
        for op in ("sqrt", "negative", "absolute"):
            def call(module, o=op, i=in_dtype, d=out_dtype):
                buffer = np.zeros(4, dtype=d)
                return getattr(module, o)(data(i), buffer)
            check("out= %s(%s) -> %s" % (op, in_dtype, out_dtype), call)

# ---------------------------------------------------------------------------
# B. `dtype=` OVERRIDES ON THE REDUCTIONS, including the nan* twins.
# ---------------------------------------------------------------------------
REDUCTIONS = ("sum", "prod", "mean", "std", "var", "cumsum", "cumprod",
              "nansum", "nanprod", "nanmean", "nanstd", "nanvar", "nancumsum", "nancumprod")
for reduction in REDUCTIONS:
    for in_dtype in DTYPES:
        for override in ("float32", "float64", "int32", "int64", "complex64", None):
            def call(module, r=reduction, i=in_dtype, d=override):
                values = data(i)
                if d is None:
                    return getattr(module, r)(values)
                return getattr(module, r)(values, dtype=np.dtype(d))
            check("dtype= %s(%s, dtype=%s)" % (reduction, in_dtype, override), call)

# ---------------------------------------------------------------------------
# C. `axis=` FORMS. A reduced axis can change the promotion for some entries.
# ---------------------------------------------------------------------------
for reduction in REDUCTIONS + ("min", "max", "argmin", "argmax", "any", "all", "ptp"):
    for in_dtype in DTYPES:
        for axis in (0, 1, -1, None):
            def call(module, r=reduction, i=in_dtype, a=axis):
                values = data(i, 6).reshape(2, 3)
                if a is None:
                    return getattr(module, r)(values)
                return getattr(module, r)(values, axis=a)
            check("axis= %s(%s, axis=%s)" % (reduction, in_dtype, axis), call)

# ---------------------------------------------------------------------------
# D. 0-d AND EMPTY OPERANDS, where numpy's promotion rules have special cases.
# ---------------------------------------------------------------------------
UNARY = ("sqrt", "negative", "absolute", "sign", "isnan", "isfinite", "rint", "conjugate")
BINARY = ("add", "subtract", "multiply", "divide", "maximum", "minimum", "power")
for in_dtype in DTYPES:
    for shape_name, make in (("0d", lambda d: np.array(2).astype(d)),
                             ("empty", lambda d: np.array([], dtype=d))):
        for op in UNARY:
            def call(module, o=op, i=in_dtype, m=make):
                return getattr(module, o)(m(i))
            check("%s %s(%s)" % (shape_name, op, in_dtype), call)
        for op in BINARY:
            def call(module, o=op, i=in_dtype, m=make):
                return getattr(module, o)(m(i), m(i))
            check("%s %s(%s)" % (shape_name, op, in_dtype), call)
        for reduction in ("sum", "prod", "mean", "cumsum"):
            def call(module, r=reduction, i=in_dtype, m=make):
                return getattr(module, r)(m(i))
            check("%s %s(%s)" % (shape_name, reduction, in_dtype), call)

# ---------------------------------------------------------------------------
# E. THE *_like FAMILY AND THE asarray / astype WRAPPERS.
# ---------------------------------------------------------------------------
for in_dtype in DTYPES:
    for like in ("empty_like", "zeros_like", "ones_like"):
        def call(module, l=like, i=in_dtype):
            # empty_like's CONTENT is uninitialised; only its dtype and type are asserted.
            return getattr(module, l)(data(i))
        check("%s(%s)" % (like, in_dtype), call)
        for override in ("float32", "int32", "complex64"):
            def call(module, l=like, i=in_dtype, d=override):
                return getattr(module, l)(data(i), dtype=np.dtype(d))
            check("%s(%s, dtype=%s)" % (like, in_dtype, override), call)
    def call(module, i=in_dtype):
        return module.full_like(data(i), 3)
    check("full_like(%s, 3)" % in_dtype, call)
    for target in DTYPES:
        def call(module, i=in_dtype, d=target):
            return module.asarray(data(i), dtype=np.dtype(d))
        check("asarray(%s, dtype=%s)" % (in_dtype, target), call)

out("")
out("dtype-only audit: %d cells compared, %d inadmissible (numpy refused the form)"
    % (checked, skipped))
out("DIVERGENCES: %d" % len(rows))
for label, ours, theirs in rows[:200]:
    out("  %-42s ours=%-28s theirs=%s" % (label, ours, theirs))
if len(rows) > 200:
    out("  ... %d more" % (len(rows) - 200))
"#;

fn main() -> PyResult<()> {
    pyo3::append_to_inittab!(fnp_python);
    Python::initialize();
    let exe = std::env::current_exe()
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyOSError, _>(error.to_string()))?;
    Python::attach(|py| {
        let globals = pyo3::types::PyDict::new(py);
        globals.set_item("EXE_PATH", exe.to_string_lossy().as_ref())?;
        py.run(&CString::new(HARNESS).unwrap(), Some(&globals), None)
    })
}
