"""FrankenNumPy: the ``fnp_python`` extension module, re-exported in full.

maturin wraps a pure-Rust extension in a package whose generated ``__init__`` does
``from .fnp_python import *``. That honours the extension's ``__all__``, which mirrors
``numpy.__all__`` verbatim, so every native name outside it (``Nditer``, ``FromPyFunc``,
``Vectorize``, the exception classes, ``__version__``, ``__numpy_version__``, the
flattened polynomial helpers, ...) would silently vanish from the installed package.
This wrapper copies the extension's whole namespace instead, so ``import fnp_python``
from a wheel is the same surface as loading the bare cdylib. Submodules
(``fnp_python.linalg`` and friends) are registered in ``sys.modules`` by the extension
itself, so dotted imports keep working.
"""

import sys as _sys


def _check_x86_64_avx2_support() -> None:
    import os
    import platform
    import sys

    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        return
    if os.environ.get("FNP_SIMULATE_NO_AVX2") == "1":
        raise ImportError(
            "fnp_python x86_64 binary was compiled with +avx2 and requires an AVX2-capable CPU"
        )
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as _cpuinfo:
                for _line in _cpuinfo:
                    if _line.startswith("flags") and ":" in _line:
                        _flags = _line.split(":", 1)[1].split()
                        if "avx2" not in _flags:
                            raise ImportError(
                                "fnp_python x86_64 binary was compiled with +avx2 and requires an AVX2-capable CPU"
                            )
                        break
        except OSError:
            pass


_check_x86_64_avx2_support()

from . import fnp_python as _extension

_SKIP = frozenset(
    {
        "__builtins__",
        "__cached__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
    }
)

_package = _sys.modules[__name__]
for _name in dir(_extension):
    if _name not in _SKIP:
        setattr(_package, _name, getattr(_extension, _name))

# The extension registers its submodules under its own qualified name. Loaded as
# ``fnp_python.fnp_python`` that is ``fnp_python.fnp_python.linalg``; alias each one to
# ``fnp_python.linalg`` so ``import fnp_python.linalg`` resolves from the wheel exactly
# as it does from the bare cdylib.
_nested_prefix = _extension.__name__ + "."
for _qualified in list(_sys.modules):
    if _qualified.startswith(_nested_prefix):
        _alias = __name__ + "." + _qualified[len(_nested_prefix) :]
        _sys.modules.setdefault(_alias, _sys.modules[_qualified])

del (
    _SKIP,
    _check_x86_64_avx2_support,
    _extension,
    _name,
    _nested_prefix,
    _package,
    _qualified,
    _sys,
)
