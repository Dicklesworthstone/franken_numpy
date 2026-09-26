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
    # Added 2026-09-25; the first run over these found 60 divergences (loadtxt validation,
    # np.ma subclass results, bit-generator state/Philox/out=) and a collection failure that
    # hid all 51 of test_recfunctions.
    numpy.lib.tests.test_arraypad numpy.lib.tests.test_type_check numpy.lib.tests.test_ufunclike
    numpy.lib.tests.test_packbits numpy.lib.tests.test_polynomial numpy.lib.tests.test_loadtxt
    numpy.lib.tests.test_recfunctions numpy.lib.tests.test_arrayterator
    numpy.lib.tests.test_regression numpy._core.tests.test_item_selection
    numpy._core.tests.test_datetime numpy._core.tests.test_defchararray
    numpy._core.tests.test_strings numpy._core.tests.test_records
    numpy._core.tests.test_numerictypes numpy._core.tests.test_getlimits
    numpy._core.tests.test_half numpy._core.tests.test_indexing numpy.ma.tests.test_core
    numpy.ma.tests.test_extras numpy.polynomial.tests.test_hermite
    numpy.polynomial.tests.test_hermite_e numpy.polynomial.tests.test_laguerre
    numpy.polynomial.tests.test_legendre numpy.polynomial.tests.test_classes
    numpy.polynomial.tests.test_polyutils numpy.random.tests.test_direct
    numpy.random.tests.test_seed_sequence numpy.random.tests.test_smoke
    numpy.random.tests.test_regression numpy.random.tests.test_randomstate_regression
    numpy.random.tests.test_generator_mt19937_regressions numpy.linalg.tests.test_regression
    # Added 2026-09-25 (third batch, from 51 modules never run before): they found asarray
    # copying ndarray subclasses, fromstring dropping unmatched tokens, no module __getattr__,
    # and missing np.version / np.matlib.
    numpy._core.tests.test_multiarray numpy._core.tests.test_regression numpy._core.tests.test_api
    numpy._core.tests.test_array_coercion numpy._core.tests.test_deprecations
    numpy._core.tests.test_memmap numpy._core.tests.test_longdouble
    numpy._core.tests.test_scalar_methods numpy._core.tests.test_scalar_ctors
    numpy._core.tests.test_umath_complex numpy._core.tests.test_unicode
    numpy._core.tests.test_stringdtype numpy.tests.test_matlib numpy.tests.test_numpy_version
    numpy.matrixlib.tests.test_defmatrix numpy.matrixlib.tests.test_interaction
    numpy.ma.tests.test_subclassing numpy.ma.tests.test_mrecords numpy.testing.tests.test_utils
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
