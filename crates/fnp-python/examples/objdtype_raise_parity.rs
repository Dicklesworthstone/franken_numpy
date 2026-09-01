//! Runtime raise-parity sweep for the object-dtype DECLINE-BY-RAISING class
//! (`deadlock-audit-7zait`, successor to `deadlock-audit-nq438`).
//!
//! A static census of `.view(...)` sites is an UPPER BOUND, not a defect count: most view an
//! output buffer we allocated ourselves, or an operand an upstream `kind` gate already
//! restricted. Only a runtime call can separate the three cases, so this sweep calls every
//! exported entry point with an object-carrying operand in each argument position and records
//! the 2x2 against the live incumbent:
//!
//!     ours raised / theirs raised  -> fine (exception type diffs reported separately)
//!     ours ok     / theirs ok      -> fine (values compared)
//!     ours RAISED / theirs OK      -> DEFECT, this class
//!     ours OK     / theirs raised  -> a different defect, also reported
//!
//! Run it on a worker (the harness prints its own nodename, so a local escape is visible):
//!     rch exec --job -- cargo run --release -j 4 -p fnp-python --example objdtype_raise_parity
//!
//! `append_to_inittab` embeds the extension in this executable, so no `libfnp_python.so` has
//! to exist on disk - the `current_exe()`-sibling idiom that breaks the conformance suites
//! under rch (`deadlock-audit-ovdn9`) is not used here.

use pyo3::prelude::*;
use std::ffi::CString;

use fnp_python::fnp_python;

const HARNESS: &str = r#"
import faulthandler, hashlib, itertools, os, resource, sys

# A sweep that dies mid-way exits 139 with an EMPTY report and no indication of where it was.
# One numpy-vs-numpy null run of this probe did exactly that on a loaded box; the next did not.
faulthandler.enable()

def out(*items):
    print(*items, file=sys.stderr, flush=True)

import numpy as np
import fnp_python as fnp

out("python", sys.version.split()[0], "| numpy", np.__version__)
out("host", os.uname().nodename, "| loadavg", [round(value, 2) for value in os.getloadavg()])
out("in-process ELF sha256", hashlib.sha256(open(EXE_PATH, "rb").read()).hexdigest())

# Every operand below is <= 3 elements, but a wrong-arity call that numpy ACCEPTS can still
# ask for a large allocation (a shape argument read from the wrong position). Cap address
# space so such a call raises MemoryError - which the 2x2 records honestly - instead of
# taking the worker down.
resource.setrlimit(resource.RLIMIT_AS, (16 << 30, 16 << 30))

# Names whose effect is a file, the process, or numpy's global state rather than a dtype
# route. Excluded because calling them proves nothing about this class and can mutate the
# environment the rest of the sweep runs in.
DENY = {
    "save", "savez", "savez_compressed", "savetxt", "load", "loadtxt", "genfromtxt",
    "fromfile", "tofile", "memmap", "fromregex", "info", "source", "show_config",
    "show_runtime", "test", "seterr", "seterrcall", "geterr", "geterrcall",
    "set_printoptions", "get_printoptions", "printoptions", "setbufsize", "getbufsize",
    "deprecate", "deprecate_with_doc", "safe_eval", "disp", "who", "errstate",
}

# The operand under test. `hasobject` is the property that matters, not `kind == 'O'`:
# a structured dtype with an embedded object field reports kind 'V' and is equally
# unviewable, and that is the sibling nq438's fix also had to cover. The 0-d form is
# carried because a 0-d PyBuffer yields no slice, so 0-d operands take different gates.
PROBES = (
    ("object_1d", lambda: np.array([1, 2, 3], dtype=object)),
    ("object_0d", lambda: np.array(7, dtype=object)),
    ("object_field_V", lambda: np.zeros(3, dtype=[("a", object), ("b", "i4")])),
)

# Filler operands for the positions NOT under test. The sweep tries them until numpy accepts
# the call: numpy accepting is the precondition for the only cell that can be a defect.
FILLERS = (
    ("f64", lambda: np.array([1.0, 2.0, 3.0])),
    ("i64", lambda: np.array([1, 2, 3], dtype=np.int64)),
    ("int0", lambda: 0),
    ("int1", lambda: 1),
)

TRACE = os.environ.get("FNP_PROBE_TRACE") == "1"

def call(function, arguments):
    # BaseException, not Exception: a callee that raises SystemExit would otherwise unwind the
    # whole sweep and the interpreter would exit 0 having reported NOTHING - which is exactly
    # how the first null run of this probe died. KeyboardInterrupt stays fatal on purpose.
    try:
        return "ok", function(*arguments)
    except KeyboardInterrupt:
        raise
    except BaseException as exception:
        return "raise", exception

def describe(exception):
    text = str(exception).replace("\n", " ")
    return "%s: %s" % (type(exception).__name__, text[:120])

def values_agree(left, right):
    # NaN EQUALS NaN here. Position 1 of a unary ufunc is `out`, so the sweep legitimately
    # produces NaN-filled results (arccos(2.0), 1/0), and a bare `array_equal` called them
    # divergences in the numpy-vs-numpy null - 10 phantom rows from the probe, not the routes.
    if left.dtype.hasobject:
        for first, second in zip(left.ravel().tolist(), right.ravel().tolist()):
            if first is second:
                continue
            try:
                if first != second and not (first != first and second != second):
                    return False
            except Exception:
                return False
        return True
    return bool(np.array_equal(left, right, equal_nan=True))

def same(left, right):
    # RAW BYTES for everything except longdouble/clongdouble, whose x86 80-bit value sits in a
    # 16-byte slot with 6 UNINITIALISED pad bytes (`deadlock-audit-6cukd`), and except object
    # dtypes, which have no byte representation at all - both compare by VALUE.
    if type(left) is not type(right):
        return False, "python type %s vs %s" % (type(left).__name__, type(right).__name__)
    if isinstance(left, tuple):
        if len(left) != len(right):
            return False, "tuple len %d vs %d" % (len(left), len(right))
        for first, second in zip(left, right):
            agree, why = same(first, second)
            if not agree:
                return False, why
        return True, ""
    if isinstance(left, np.ndarray):
        if left.dtype != right.dtype:
            return False, "dtype %s vs %s" % (left.dtype, right.dtype)
        if left.shape != right.shape:
            return False, "shape %s vs %s" % (left.shape, right.shape)
        by_value = left.dtype.hasobject or left.dtype.char in "gG"
        try:
            if by_value:
                return values_agree(left, right), "values"
            return left.tobytes() == right.tobytes(), "bytes"
        except Exception as exception:
            return True, "uncomparable (%s)" % describe(exception)
    try:
        if left != left and right != right:
            return True, "scalar nan"
        return bool(left == right), "scalar"
    except Exception:
        return True, "uncomparable scalar"

def entry_points():
    for name in sorted(name for name in dir(fnp) if not name.startswith("_")):
        if name in DENY:
            continue
        ours = getattr(fnp, name, None)
        theirs = getattr(np, name, None)
        if theirs is None or not callable(ours) or not callable(theirs):
            continue
        if isinstance(ours, type) or isinstance(theirs, type):
            continue
        yield name, ours, theirs

def shapes():
    for probe_name, probe_make in PROBES:
        for arity in (1, 2, 3):
            for position in range(arity):
                for combination in itertools.product(FILLERS, repeat=arity - 1):
                    yield probe_name, probe_make, arity, position, combination

cells = {"both_ok": 0, "both_raise": 0, "ours_raise_theirs_ok": 0, "ours_ok_theirs_raise": 0}
defects, inverse, value_diffs, exc_type_diffs = [], [], [], []
covered_names, admissible_calls = set(), 0
entry_count = 0

for name, ours, theirs in entry_points():
    entry_count += 1
    if TRACE:
        out("trace ok-sweep", name)
    settled = set()
    for probe_name, probe_make, arity, position, combination in shapes():
        # One admissible call per (probe, arity, position) is enough: the fillers only exist
        # to get numpy to accept the call at all.
        key = (probe_name, arity, position)
        if key in settled:
            continue
        arguments = [maker() for _, maker in combination]
        arguments.insert(position, probe_make())
        their_kind, their_result = call(theirs, arguments)
        if their_kind != "ok":
            continue
        settled.add(key)
        arguments = [maker() for _, maker in combination]
        arguments.insert(position, probe_make())
        our_kind, our_result = call(ours, arguments)
        admissible_calls += 1
        covered_names.add(name)
        row = (name, probe_name, arity, position,
               "+".join(label for label, _ in combination) or "-")
        if our_kind == "raise":
            cells["ours_raise_theirs_ok"] += 1
            defects.append(row + (describe(our_result),))
        else:
            cells["both_ok"] += 1
            agree, why = same(their_result, our_result)
            if not agree:
                value_diffs.append(row + (why,))

# The inverse cell needs numpy to RAISE, so it is swept separately: above, a numpy raise meant
# the call was inadmissible and the sweep moved on.
for name, ours, theirs in entry_points():
    if TRACE:
        out("trace raise-sweep", name)
    settled = set()
    for probe_name, probe_make, arity, position, combination in shapes():
        key = (probe_name, arity, position)
        if key in settled:
            continue
        arguments = [maker() for _, maker in combination]
        arguments.insert(position, probe_make())
        their_kind, their_result = call(theirs, arguments)
        if their_kind != "raise":
            continue
        settled.add(key)
        arguments = [maker() for _, maker in combination]
        arguments.insert(position, probe_make())
        our_kind, our_result = call(ours, arguments)
        row = (name, probe_name, arity, position,
               "+".join(label for label, _ in combination) or "-")
        if our_kind == "ok":
            cells["ours_ok_theirs_raise"] += 1
            inverse.append(row + (describe(their_result),))
        else:
            cells["both_raise"] += 1
            if type(our_result) is not type(their_result):
                exc_type_diffs.append(row + ("theirs %s / ours %s" % (
                    type(their_result).__name__, type(our_result).__name__),))

out("")
out("entry points swept: %d | reached by an admissible call: %d | admissible calls: %d"
    % (entry_count, len(covered_names), admissible_calls))
out("2x2: both_ok %d | both_raise %d | OURS RAISED/theirs ok %d | ours ok/theirs raised %d"
    % (cells["both_ok"], cells["both_raise"],
       cells["ours_raise_theirs_ok"], cells["ours_ok_theirs_raise"]))

def report(title, rows, limit=300):
    out("")
    out("%s: %d" % (title, len(rows)))
    for row in rows[:limit]:
        out("  %-24s %-15s arity=%d pos=%d fillers=%-12s %s" % row)
    if len(rows) > limit:
        out("  ... %d more" % (len(rows) - limit))

report("DEFECT - ours RAISED where numpy answered", defects)
report("INVERSE - ours answered where numpy RAISED", inverse)
report("VALUE DIVERGENCE - both answered, results differ", value_diffs)
report("exception TYPE differs (both raised; informational)", exc_type_diffs)
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
