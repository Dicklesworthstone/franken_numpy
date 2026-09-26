"""Run NumPy's OWN test modules with fnp_python swapped in for numpy (the drop-in check).

pytest plugin. After collection it rebinds each collected numpy test module's globals:
`np`/`numpy` (and `numpy.linalg`/`fft`/`random`) become a stand-in whose PUBLIC names
resolve on fnp_python and whose PRIVATE names (`_core`, `_NoValue`, ...) fall back to
numpy, and every name the module imported from those packages is rebound to fnp's object.
An `__import__` hook hands the same stand-in to the test module at import time, so values
captured then (parametrize lists, class attributes) are fnp objects too. numpy itself is
never touched: fnp's delegates look numpy callables up on the live module, so patching
numpy would recurse.

Two lanes, the same tests: FNP_DROPIN=0 is the A/A lane (real numpy); FNP_DROPIN=1 swaps.
A test that PASSES on the A/A lane and fails on the swap lane is a divergence. Run both
with `scripts/run_numpy_dropin_suite.sh`, which then calls `python3 <this file> compare`.

Environment: FNP_DROPIN_SO=<path to the built fnp_python cdylib> (required when swapping).

`np._core.umath` (and a module global bound to `numpy._core.umath`) resolves to a copy of
numpy's umath whose ufuncs are fnp's same-named objects: tests that enumerate umath and then
look names up on `np` exercise fnp consistently. Before that, test_ufunc failed at collection
(its whole module was invisible to this check) and test_umath's ~474
test_unary_spurious_fpexception rows were harness artifacts.

Known harness artifacts (not fnp defects): tests that read private attributes of fnp
objects (`capsule`) fail on the swap lane; test_pickle_withstring unpickles numpy's own
`numpy._core.umath.cos`; TestAdd_newdoc_ufunc hands fnp's ufunc to numpy's private
`_add_newdoc_ufunc`.

By design while fnp is an accelerator over numpy (owner decision rc0923 .11): when fnp
delegates a ufunc call, `__array_ufunc__` / `__array_wrap__` hooks receive NUMPY's ufunc
object, and `__array_function__` / `like=` hooks receive numpy's function, which is what
third-party libraries that imported real numpy key their handler tables on. So
test_ufunc_override, test_ufunc_override_methods, test_wrap, test_multiarray's
test_ufunc_binop_interaction and test_overrides' identity rows (test_override_sum,
test_sum_on_mock_array, TestArrayLike, test_function_like, test_nep35_functions_as_array_functions),
which expect `np.<name>` (fnp's object) under full substitution, fail on the swap lane.

Also by design (fnp is its own package): `__module__` checks that expect 'numpy' on fnp's
objects (test_overrides::test_set_module, test_public_api's check_dir / __module__ rows),
numpy's array-API entry point, `np.version` describing fnp while `np.__version__` stays numpy's
under partial substitution (test_short_version), pickles WRITTEN by numpy that construct numpy's
objects (test_legacy_pickle, test_load_ufunc_pickle), `bitgen_t` ctypes/cffi interfaces
(test_ctypes), and assigning a ufunc's `__doc__` (test_ufunc_docstring).

History: this harness found the defect batch fixed under deadlock-audit-rc0923-epic-71qy3.8
(clip on lists, ufunc protocol, jumped(), RNG pickling, vectorize, isclose NEP 50,
linalg empty-stack panics, ...). Retire it when fnp passes numpy's suite or a stronger
harness replaces it.
"""
import builtins
import importlib.util
import os
import re
import sys
import types

ENABLED = os.environ.get("FNP_DROPIN", "1") == "1"
SO = os.environ.get("FNP_DROPIN_SO", "")
_FNP = None


def _load_fnp():
    global _FNP
    if _FNP is None:
        spec = importlib.util.spec_from_file_location("fnp_python", SO)
        module = importlib.util.module_from_spec(spec)
        sys.modules["fnp_python"] = module
        spec.loader.exec_module(module)
        _FNP = module
    return _FNP


_PATCHED_UMATH = None


def _patched_umath(fnp):
    """`numpy._core.umath` with every ufunc that fnp also exports replaced by fnp's object.
    Tests enumerate this module (test_ufunc's UNARY_UFUNCS, test_umath's UFUNCS) and then
    look names up on `np`; mixing numpy's list with fnp's names failed test_ufunc at
    collection and filled test_umath with ~474 spurious divergences. Other entries stay
    numpy's."""
    global _PATCHED_UMATH
    if _PATCHED_UMATH is None:
        import types

        import numpy

        real = numpy._core.umath
        patched = types.ModuleType(real.__name__, real.__doc__)
        for name, value in vars(real).items():
            replacement = getattr(fnp, name, None) if isinstance(value, numpy.ufunc) else None
            patched.__dict__[name] = replacement if isinstance(replacement, numpy.ufunc) else value
        _PATCHED_UMATH = patched
    return _PATCHED_UMATH


class _CoreSwap:
    """Stand-in for `numpy._core`: numpy's, except that `umath` is `_patched_umath`."""

    def __init__(self, fnp_mod):
        object.__setattr__(self, "_fnp", fnp_mod)

    def __getattr__(self, name):
        import numpy

        if name == "umath":
            return _patched_umath(object.__getattribute__(self, "_fnp"))
        return getattr(numpy._core, name)


class _PublicSwap:
    """Stand-in for `np`: public names resolve on fnp_python, private names on numpy
    (tests reach into numpy internals no drop-in user would touch), except `np._core`,
    whose `umath` carries fnp's ufuncs."""

    def __init__(self, fnp_mod, np_mod):
        object.__setattr__(self, "_fnp", fnp_mod)
        object.__setattr__(self, "_np", np_mod)

    def __getattribute__(self, name):
        # test_ma's TestOptionalArgs calls `numpy.__getattribute__(f)` directly, and that slot
        # never falls back to `__getattr__`: route a name the stand-in lacks the same way.
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return type(self).__getattr__(self, name)

    def __getattr__(self, name):
        import numpy

        np_mod = object.__getattribute__(self, "_np")
        if name == "_core" and np_mod is numpy:
            return _CoreSwap(object.__getattribute__(self, "_fnp"))
        # `np.linalg` / `np.fft` / `np.random` / `np.lib` are stand-ins too, so their private
        # names (`np.linalg._umath_linalg`, `np.lib._nanfunctions_impl`) fall back to numpy.
        if name in ("linalg", "fft", "random", "lib") and np_mod is numpy:
            return _PublicSwap(
                getattr(object.__getattribute__(self, "_fnp"), name), getattr(numpy, name)
            )
        # `np.random.mtrand._rand` is the global RandomState the public `np.random.*` functions
        # draw from, so it is fnp's own when fnp has one: routed to numpy's, test_hot_swap and
        # test_coercion_RandomState_Generator set fnp's bit generator and then inspected numpy's
        # untouched singleton (the only two numpy tests that read `_rand`).
        if name == "_rand" and hasattr(object.__getattribute__(self, "_fnp"), "_rand"):
            return getattr(object.__getattribute__(self, "_fnp"), "_rand")
        if name.startswith("_"):
            return getattr(np_mod, name)
        fnp_mod = object.__getattribute__(self, "_fnp")
        value = getattr(fnp_mod, name)
        # Any other submodule both sides have gets the same split one level down:
        # test_recfunctions reads `np.lib.recfunctions._get_fieldspec` at import, and fnp's
        # module has no such private name, so the whole module (51 tests) failed collection.
        # Only one of fnp's OWN submodules is looked up on numpy's side: a second lookup of any
        # other name ran numpy's module `__getattr__` again, so `np.lib.math` and `np.chararray`
        # emitted their DeprecationWarning twice (fnp alone emits one, as numpy does).
        if isinstance(value, types.ModuleType) and value.__name__.startswith(fnp_mod.__name__.split(".")[0] + "."):
            np_value = getattr(np_mod, name, None)
            if isinstance(np_value, types.ModuleType):
                return _PublicSwap(value, np_value)
        return value

    def __dir__(self):
        # Tests parametrize over `dir(np)` (test_ufunc_types, test_ufunc_noncontiguous): the
        # stand-in's own dir() was empty, so the swap lane collected NOTSET for 212 cases.
        # Same split as __getattr__: public names are fnp's, private names numpy's.
        fnp_mod = object.__getattribute__(self, "_fnp")
        np_mod = object.__getattribute__(self, "_np")
        return sorted(
            {name for name in dir(fnp_mod) if not name.startswith("_")}
            | {name for name in dir(np_mod) if name.startswith("_")}
        )

    @property
    def __dict__(self):
        # `vars(np)` as a namespace (test_arrayprint's `eval(repr(a), vars(np))`): the
        # stand-in's own dict held only `_fnp`/`_np`, so `array` was a NameError. A snapshot
        # with the same split as __getattr__; writes to it do not reach either module.
        fnp_mod = object.__getattribute__(self, "_fnp")
        np_mod = object.__getattribute__(self, "_np")
        namespace = {name: value for name, value in vars(np_mod).items() if name.startswith("_")}
        namespace.update(
            (name, value) for name, value in vars(fnp_mod).items() if not name.startswith("_")
        )
        return namespace


def _swap_globals(module, fnp):
    import numpy

    subs = [(numpy, _PublicSwap(fnp, numpy))]
    for sub in ("linalg", "fft", "random"):
        subs.append((getattr(numpy, sub), _PublicSwap(getattr(fnp, sub), getattr(numpy, sub))))
    namespace = module.__dict__
    for name, value in list(namespace.items()):
        if name.startswith("__"):
            continue
        # `import numpy._core.umath as ncu` (test_umath) binds numpy's module: use the one
        # with fnp's ufuncs, as `np._core.umath` does.
        if value is numpy._core.umath:
            namespace[name] = _patched_umath(fnp)
            continue
        # A module global that IS one of the swapped packages becomes its stand-in.
        package_swap = next((swap for np_mod, swap in subs if value is np_mod), None)
        if package_swap is not None:
            namespace[name] = package_swap
            continue
        # A name imported FROM a swapped package becomes fnp's object of that name.
        for np_mod, swap in subs:
            np_value = getattr(np_mod, name, None)
            if np_value is not None and value is np_value:
                fnp_value = getattr(object.__getattribute__(swap, "_fnp"), name, None)
                if fnp_value is not None and fnp_value is not np_value:
                    namespace[name] = fnp_value
                break


_REAL_IMPORT = builtins.__import__
_TEST_MODULE = re.compile(r"^numpy\.(.+\.)?tests\.test_")
_SWAPPED_PACKAGES = ("numpy", "numpy.linalg", "numpy.fft", "numpy.random")


def _swapping_import(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {}).get("__name__", "")
    caller_file = (globals or {}).get("__file__", "") or ""
    # pytest may import a test module under its bare name, so match the file path too.
    is_numpy_test = bool(_TEST_MODULE.match(caller)) or (
        "/numpy/" in caller_file and "/tests/test_" in caller_file
    )
    if level == 0 and is_numpy_test and name in _SWAPPED_PACKAGES:
        import numpy

        _REAL_IMPORT(name, globals, locals, fromlist, level)
        fnp = _load_fnp()
        if not fromlist:
            return _PublicSwap(fnp, numpy)
        np_target = numpy if name == "numpy" else getattr(numpy, name.split(".")[1])
        fnp_target = fnp if name == "numpy" else getattr(fnp, name.split(".")[1])
        return _PublicSwap(fnp_target, np_target)
    return _REAL_IMPORT(name, globals, locals, fromlist, level)


def pytest_configure(config):
    if ENABLED:
        builtins.__import__ = _swapping_import


def pytest_collection_modifyitems(session, config, items):
    if not ENABLED:
        return
    fnp = _load_fnp()
    seen = set()
    for item in items:
        module = getattr(item, "module", None)
        if module is not None and id(module) not in seen:
            seen.add(id(module))
            _swap_globals(module, fnp)


def pytest_report_header(config):
    return f"fnp drop-in swap: {'ON' if ENABLED else 'OFF (A/A lane)'} so={SO}"


def _load_junit(path):
    import xml.etree.ElementTree as ET

    outcomes = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        key = f"{case.get('classname')}::{case.get('name')}"
        status, message = "pass", ""
        for tag in ("failure", "error"):
            element = case.find(tag)
            if element is not None:
                status = tag
                message = (element.get("message") or "").strip().splitlines()[0][:220] if element.get("message") else ""
        if case.find("skipped") is not None:
            status = "skipped"
        outcomes[key] = (status, message)
    return outcomes


# Divergence classes that are known and OWNED, matched against the test id (module short name +
# "::" + class/test). Anything that matches none of them is an UNOWNED fnp divergence: fix it or
# give it an owner here, never leave it unclassified. `owner` is the bead that decides the class.
KNOWN_DIVERGENCES = (
    (r"test_umath::TestSpecialMethods::test_(wrap|ufunc_override|ufunc_override_methods)$|"
     r"test_multiarray::TestBinop::test_ufunc_binop_interaction$|test_multiarray::TestCAPI::test_import_entry_point",
     "identity", "deadlock-audit-rc0923-epic-71qy3.11",
     "a delegated ufunc call hands NUMPY's ufunc to __array_ufunc__/__array_wrap__ - what libraries that imported numpy key on"),
    (r"test_overrides::(TestNumPyFunctions::test_(override_sum|sum_on_mock_array)$|TestArrayLike::|test_function_like$)",
     "identity", "deadlock-audit-rc0923-epic-71qy3.11",
     "__array_function__ / like= hooks receive numpy's function object, and fnp's dispatcher is its own type"),
    (r"test_overrides::TestNumPyFunctions::test_set_module$|test_public_api::(test_numpy_namespace|test_numpy_linalg|"
     r"test_numpy_fft|test___module___attribute|test___qualname___and___module___attribute|test_array_api_entry_point)$|"
     r"test_regression::TestRegression::test__array_namespace__$|test_scalar_methods::TestDevice::test___array_namespace__",
     "identity", "deadlock-audit-rc0923-epic-71qy3.11",
     "fnp's objects report fnp_python as their module; numpy's scalars and entry point name numpy"),
    (r"::test_legacy_pickle|test_regression::TestRegression::test_load_ufunc_pickle$|test_ufunc::TestUfunc::test_pickle_withstring$",
     "identity", "deadlock-audit-rc0923-epic-71qy3.11",
     "a pickle WRITTEN by numpy reconstructs numpy's own object"),
    (r"test_direct::Test\w+::test_ctypes$",
     "identity", "deadlock-audit-rc0923-epic-71qy3.11",
     "fnp's bit generators have no C bitgen_t for ctypes/cffi"),
    (r"test_numpy_version::test_short_version$",
     "identity", "deadlock-audit-rc0923-epic-71qy3.11",
     "np.version describes fnp while np.__version__ stays numpy's under partial substitution"),
    (r"test_umath::(TestSpecialMethods::test_ufunc_docstring|TestAdd_newdoc_ufunc::test_ufunc_arg)$",
     "identity", "deadlock-audit-rc0923-epic-71qy3.11",
     "fnp's ufunc objects are not numpy.ufunc: no writable __doc__, not accepted by _add_newdoc_ufunc"),
)


def classify(test_id):
    """(kind, owner, reason) for a divergence; kind 'unowned' when no known class matches."""
    for pattern, kind, owner, reason in KNOWN_DIVERGENCES:
        if re.search(pattern, test_id):
            return kind, owner, reason
    return "unowned", "", ""


def compare(aa_path, swap_path, label, json_path=None):
    """Print the tests that pass on the A/A lane and fail on the swap lane, each with its class;
    with `json_path`, also write them as JSON. Returns the count."""
    import json

    aa, swap = _load_junit(aa_path), _load_junit(swap_path)
    divergences = [
        (key, swap[key][1])
        for key in swap
        if swap[key][0] in ("failure", "error") and aa.get(key, ("?",))[0] == "pass"
    ]
    aa_pass = sum(1 for value in aa.values() if value[0] == "pass")
    aa_failed = sum(1 for value in aa.values() if value[0] in ("failure", "error"))
    swap_pass = sum(1 for value in swap.values() if value[0] == "pass")
    short = label.split(".")[-1]
    rows = []
    for key, message in divergences:
        # junit's classname is "" for a module-level test and ".Class" for a class, and a
        # parametrised name may itself contain dots ("[1.0]"): keep "Class::name[params]" whole.
        classname, _, name = key.partition("::")
        classname = classname.rsplit(".", 1)[-1]
        test_id = f"{short}::{classname + '::' if classname else ''}{name}"
        kind, owner, reason = classify(test_id)
        rows.append({"nodeid": test_id, "message": message, "kind": kind, "owner": owner, "reason": reason})
    unowned = sum(1 for row in rows if row["kind"] == "unowned")
    print(f"## {label}: A/A pass {aa_pass}/{len(aa)} | swap pass {swap_pass}/{len(swap)} | "
          f"DIVERGENCES {len(divergences)} (unowned {unowned})")
    for row in rows:
        tag = "UNOWNED" if row["kind"] == "unowned" else f"{row['kind']} -> {row['owner']}"
        print(f"  - {row['nodeid']} :: [{tag}] {row['message']}")
    if json_path:
        with open(json_path, "w") as handle:
            json.dump({"module": label, "aa_pass": aa_pass, "aa_failed": aa_failed, "aa_total": len(aa),
                       "swap_pass": swap_pass, "swap_total": len(swap), "divergences": rows}, handle, indent=1)
    return len(divergences)


def aggregate(out_dir, report_path):
    """Merge the per-module JSON files of one run into a single report with totals."""
    import glob
    import json

    modules = [json.load(open(path)) for path in sorted(glob.glob(f"{out_dir}/*.json"))
               if not path.endswith("/report.json")]
    rows = [row for module in modules for row in module["divergences"]]
    report = {
        "modules": len(modules),
        "aa_pass": sum(module["aa_pass"] for module in modules),
        "aa_failed": sum(module["aa_failed"] for module in modules),
        "swap_pass": sum(module["swap_pass"] for module in modules),
        "divergences": len(rows),
        "unowned": sum(1 for row in rows if row["kind"] == "unowned"),
        "by_owner": {owner: sum(1 for row in rows if row["owner"] == owner)
                     for owner in sorted({row["owner"] for row in rows if row["owner"]})},
        "per_module": modules,
    }
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"REPORT {report_path}: {report['modules']} modules, A/A pass {report['aa_pass']} "
          f"(A/A failed {report['aa_failed']}), divergences {report['divergences']}, "
          f"unowned {report['unowned']}")
    return report


if __name__ == "__main__":
    if len(sys.argv) in (5, 6) and sys.argv[1] == "compare":
        compare(*sys.argv[2:])
    elif len(sys.argv) == 4 and sys.argv[1] == "aggregate":
        aggregate(sys.argv[2], sys.argv[3])
    else:
        sys.exit("usage: numpy_dropin_plugin.py compare <aa.xml> <swap.xml> <label> [<out.json>]\n"
                 "       numpy_dropin_plugin.py aggregate <out_dir> <report.json>")
