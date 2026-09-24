#!/usr/bin/env bash
# Run numpy's OWN test modules twice - A/A lane (real numpy) and swap lane (fnp_python in
# numpy's place, see scripts/numpy_dropin_plugin.py) - and list the tests that pass on numpy
# but fail with fnp swapped in. Each listed divergence is a drop-in defect candidate.
#
# Usage: scripts/run_numpy_dropin_suite.sh <fnp_python cdylib> <out_dir> [numpy test module ...]
#   PYTHON=<interpreter with numpy + pytest + hypothesis>   (default: python3)
# The interpreter's numpy must be the one fnp was built against for parity (numpy 2.4.x here).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SO="${1:?path to the built fnp_python cdylib}"
OUT="${2:?output directory}"
shift 2
PYTHON="${PYTHON:-python3}"
MODULES=("$@")
if [[ ${#MODULES[@]} -eq 0 ]]; then
  MODULES=(
    numpy._core.tests.test_numeric numpy._core.tests.test_ufunc numpy._core.tests.test_umath
    numpy._core.tests.test_shape_base numpy._core.tests.test_function_base
    numpy._core.tests.test_einsum numpy._core.tests.test_nep50_promotions
    numpy.lib.tests.test_function_base numpy.lib.tests.test_shape_base
    numpy.lib.tests.test_arraysetops numpy.lib.tests.test_nanfunctions
    numpy.lib.tests.test_twodim_base numpy.lib.tests.test_index_tricks
    numpy.lib.tests.test_histograms numpy.lib.tests.test_stride_tricks numpy.lib.tests.test_io
    numpy.linalg.tests.test_linalg numpy.fft.tests.test_pocketfft numpy.fft.tests.test_helper
    numpy.random.tests.test_generator_mt19937 numpy.random.tests.test_random
    numpy.random.tests.test_randomstate
    numpy.polynomial.tests.test_polynomial numpy.polynomial.tests.test_chebyshev
  )
fi
mkdir -p "$OUT"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" FNP_DROPIN_SO="$SO"
# A runaway allocation must fail the test, not the host (test_huge_list_error builds 2^31 items).
ulimit -v 32000000
total=0
for mod in "${MODULES[@]}"; do
  short="${mod//./_}"
  for lane in 0 1; do
    FNP_DROPIN=$lane timeout 1500 "$PYTHON" -m pytest -p numpy_dropin_plugin \
      -p no:cacheprovider --pyargs "$mod" -q --tb=no -o addopts="" \
      -k "not test_huge_list_error" --junitxml="$OUT/${short}_lane$lane.xml" >/dev/null 2>&1
    echo "$mod lane$lane exit=$?" >> "$OUT/exits.txt"
  done
  if [[ -f "$OUT/${short}_lane0.xml" && -f "$OUT/${short}_lane1.xml" ]]; then
    "$PYTHON" "$HERE/numpy_dropin_plugin.py" compare "$OUT/${short}_lane0.xml" \
      "$OUT/${short}_lane1.xml" "$mod" > "$OUT/${short}.txt"
    head -1 "$OUT/${short}.txt"
    n=$(grep -c '^  - ' "$OUT/${short}.txt")
    total=$((total + n))
  else
    echo "## $mod: MISSING REPORT (crash, timeout or collection error - see $OUT/exits.txt)"
  fi
done
echo "TOTAL DIVERGENCES: $total (details in $OUT/*.txt)"
