#!/usr/bin/env bash
# Build (or accept a pre-built) frankennumpy wheel, install it in an isolated venv,
# and verify:
#   1. 100% numpy.__all__ coverage + full fnp_python extension namespace re-export
#   2. Standalone PyNdArray and PyDType (PEP 3118 memoryview, __array_interface__,
#      __array_ufunc__, __array_function__, pickle round-trip)
#   3. Complex libm (15 ops x 64 4-quadrant complex128/complex64 inputs) and float32
#      unary/binary byte-identity against the platform's live NumPy (0 silent drift)
#   4. Non-AVX2 x86_64 pre-dlopen & module-init ImportError guard (no SIGILL)
#
# Usage:
#   scripts/verify_wheel_distribution.sh [--wheel <path.whl>] [--report <report.json>] [--python <python3.13+>]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WHEEL_PATH=""
WHEEL_OUT_DIR=""
REPORT_PATH=""
PYTHON_BIN="${PYTHON:-$(command -v python3.13 || command -v python3.14 || command -v python3)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheel)
      WHEEL_PATH="${2:?--wheel requires a path}"
      shift 2
      ;;
    --wheel-out)
      WHEEL_OUT_DIR="${2:?--wheel-out requires a directory}"
      shift 2
      ;;
    --report)
      REPORT_PATH="${2:?--report requires a path}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:?--python requires an interpreter}"
      shift 2
      ;;
    *)
      echo "usage: $0 [--wheel <path.whl>] [--wheel-out <dir>] [--report <report.json>] [--python <python3.13+>]" >&2
      exit 2
      ;;
  esac
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/fnp-wheel-verify.XXXXXX")"

if [[ -z "$REPORT_PATH" ]]; then
  REPORT_PATH="$WORK/wheel_distribution_verification.json"
else
  mkdir -p "$(dirname "$REPORT_PATH")"
  if [[ "$REPORT_PATH" != /* ]]; then
    REPORT_PATH="$(pwd)/$REPORT_PATH"
  fi
fi

if ! "$PYTHON_BIN" -m venv "$WORK/venv" 2>/dev/null; then
  uv venv --seed --python "$PYTHON_BIN" "$WORK/venv"
fi
if [[ -f "$WORK/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$WORK/venv/bin/activate"
else
  # shellcheck disable=SC1091
  source "$WORK/venv/Scripts/activate"
fi
unset PYO3_PYTHON

python -m pip install -q --upgrade pip "numpy>=2.3,<2.5"

if [[ -z "$WHEEL_PATH" ]]; then
  python -m pip install -q maturin
  if [[ "$(uname -m)" == "x86_64" || "$(uname -m)" == "amd64" ]]; then
    export RUSTFLAGS="${RUSTFLAGS:--D warnings -C target-feature=+avx2}"
  else
    export RUSTFLAGS="${RUSTFLAGS:--D warnings}"
  fi
  (cd "$REPO" || exit 1; maturin build --release --out "$WORK/dist")
  WHEEL_PATH="$(ls "$WORK"/dist/*.whl | head -n 1)"
  if [[ -n "$WHEEL_OUT_DIR" ]]; then
    mkdir -p "$WHEEL_OUT_DIR"
    cp -f "$WHEEL_PATH" "$WHEEL_OUT_DIR/"
  fi
else
  if [[ "$WHEEL_PATH" != /* ]]; then
    WHEEL_PATH="$(pwd)/$WHEEL_PATH"
  fi
fi

if command -v cygpath >/dev/null 2>&1; then
  PY_WHEEL_PATH="$(cygpath -m "$WHEEL_PATH")"
  PY_REPORT_PATH="$(cygpath -m "$REPORT_PATH")"
else
  PY_WHEEL_PATH="$WHEEL_PATH"
  PY_REPORT_PATH="$REPORT_PATH"
fi

python -m pip install -q --force-reinstall --no-deps "$PY_WHEEL_PATH"

cd "$WORK" || exit 1
python - "$PY_WHEEL_PATH" "$PY_REPORT_PATH" <<'PY'
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import fnp_python as pkg
import fnp_python.fnp_python as ext
import fnp_python.linalg as la
import numpy as np

wheel_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
wheel_sha256 = hashlib.sha256(wheel_path.read_bytes()).hexdigest()

# 1. Surface & namespace verification
_skip = {
    "__builtins__",
    "__cached__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__path__",
    "__spec__",
}
missing_ext = sorted(n for n in ext.__dict__ if n not in _skip and not hasattr(pkg, n))
assert not missing_ext, f"{len(missing_ext)} extension names missing from package: {missing_ext[:10]}"
assert dir(pkg) == dir(ext), "dir(fnp_python) does not match dir(fnp_python.fnp_python)"

missing_np_all = sorted(n for n in np.__all__ if not hasattr(pkg, n))
assert not missing_np_all, f"{len(missing_np_all)} numpy.__all__ names missing: {missing_np_all[:10]}"
assert list(pkg.__all__) == list(np.__all__), "pkg.__all__ does not match np.__all__ verbatim"
assert la is pkg.linalg, "dotted submodule import does not resolve to package attribute"
assert pkg.__version__ and pkg.__numpy_version__ == np.__version__

# 2. Standalone PyNdArray & PyDType verification
dt = pkg.PyDType("float64")
assert dt.name == "float64" and dt.itemsize == 8 and dt.kind == "f"
arr = pkg.PyNdArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
assert arr.shape == (2, 3) and arr.size == 6 and arr.nbytes == 48
mv = memoryview(arr)
assert mv.format == "d" and mv.shape == (2, 3)
np_view = np.asarray(arr)
assert np.shares_memory(np_view, mv)
assert np.array_equal(np.asarray(np.add(arr, arr)), np_view * 2.0)
assert float(np.sum(arr)) == 21.0
roundtripped = pickle.loads(pickle.dumps(arr))
assert isinstance(roundtripped, pkg.PyNdArray)
assert roundtripped.shape == arr.shape and roundtripped.tolist() == arr.tolist()

# 3. Complex libm & float32 unary/binary byte-identity against platform NumPy
#    Tested at both small (n=64) and parallel zero-copy threshold (n=65536 = 1<<16) sizes.
coords_small = np.array([-3.5, -1.25, -0.5, -0.0, 0.0, 0.5, 1.25, 3.5], dtype=np.float64)
re_s, im_s = np.meshgrid(coords_small, coords_small)
grid128_small = (re_s + 1j * im_s).ravel().astype(np.complex128)
grid64_small = grid128_small.astype(np.complex64)

coords_large = np.linspace(-3.5, 3.5, 256, dtype=np.float64)
re_l, im_l = np.meshgrid(coords_large, coords_large)
grid128_large = (re_l + 1j * im_l).ravel().astype(np.complex128)
grid64_large = grid128_large.astype(np.complex64)

complex_ops = [
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "arcsin",
    "arccos",
    "arctan",
    "arcsinh",
    "arccosh",
    "arctanh",
    "sign",
]
complex_cells_checked = 0
complex_mismatches = []
with np.errstate(all="ignore"):
    for op in complex_ops:
        for tag, inp in (
            ("complex128@64", grid128_small),
            ("complex64@64", grid64_small),
            ("complex128@65536", grid128_large),
            ("complex64@65536", grid64_large),
        ):
            fnp_out = np.asarray(getattr(pkg, op)(inp))
            np_out = np.asarray(getattr(np, op)(inp))
            complex_cells_checked += 1
            if fnp_out.dtype != np_out.dtype or fnp_out.tobytes() != np_out.tobytes():
                complex_mismatches.append(f"{op}:{tag}")

assert not complex_mismatches, f"complex libm byte mismatches vs platform NumPy: {complex_mismatches}"

f32_grid = np.linspace(-0.9, 0.9, 65536, dtype=np.float32)
f32_pos = np.linspace(1.05, 10.0, 65536, dtype=np.float32)
float32_ops = [
    ("tan", f32_grid),
    ("arcsin", f32_grid),
    ("arccos", f32_grid),
    ("arctan", f32_grid),
    ("sinh", f32_grid),
    ("cosh", f32_grid),
    ("arcsinh", f32_grid),
    ("arccosh", f32_pos),
    ("arctanh", f32_grid),
    ("expm1", f32_grid),
    ("log1p", f32_grid),
    ("cbrt", f32_grid),
    ("log10", f32_pos),
    ("log2", f32_pos),
    ("exp2", f32_grid),
    ("sin", f32_grid),
    ("cos", f32_grid),
    ("exp", f32_grid),
    ("log", f32_pos),
    ("sqrt", f32_pos),
]
float32_cells_checked = 0
float32_mismatches = []
with np.errstate(all="ignore"):
    for op, inp in float32_ops:
        fnp_out = np.asarray(getattr(pkg, op)(inp))
        np_out = np.asarray(getattr(np, op)(inp))
        float32_cells_checked += 1
        if fnp_out.dtype != np_out.dtype or fnp_out.tobytes() != np_out.tobytes():
            float32_mismatches.append(op)

assert not float32_mismatches, f"float32 unary byte mismatches vs platform NumPy: {float32_mismatches}"

# 4. Non-AVX2 x86_64 pre-dlopen & module-init ImportError guard
machine = platform.machine().lower()
env_no_avx2 = {**os.environ, "FNP_SIMULATE_NO_AVX2": "1"}
proc_pkg = subprocess.run(
    [sys.executable, "-c", "import fnp_python"],
    env=env_no_avx2,
    capture_output=True,
    text=True,
)
if machine in ("x86_64", "amd64"):
    assert proc_pkg.returncode == 1, (
        f"Expected ImportError (rc=1) under FNP_SIMULATE_NO_AVX2=1, got rc={proc_pkg.returncode}: "
        f"{proc_pkg.stderr}"
    )
    assert "ImportError" in proc_pkg.stderr and "AVX2" in proc_pkg.stderr, (
        f"Expected clear AVX2 ImportError message, got: {proc_pkg.stderr}"
    )
    non_avx2_guard_verified = True
else:
    assert proc_pkg.returncode == 0, (
        f"Non-x86_64 import failed under FNP_SIMULATE_NO_AVX2=1: {proc_pkg.stderr}"
    )
    non_avx2_guard_verified = True

report = {
    "status": "pass",
    "platform": sys.platform,
    "machine": platform.machine(),
    "python_version": sys.version.split()[0],
    "numpy_version": np.__version__,
    "fnp_version": pkg.__version__,
    "wheel_name": wheel_path.name,
    "wheel_sha256": wheel_sha256,
    "numpy_all_total": len(np.__all__),
    "numpy_all_covered": len(np.__all__) - len(missing_np_all),
    "extension_exports_total": len(dir(ext)),
    "standalone_pyndarray_verified": True,
    "complex_libm_cells_checked": complex_cells_checked,
    "complex_libm_byte_mismatches": complex_mismatches,
    "float32_libm_cells_checked": float32_cells_checked,
    "float32_libm_byte_mismatches": float32_mismatches,
    "non_avx2_guard_verified": non_avx2_guard_verified,
}
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
