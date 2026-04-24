#!/usr/bin/env python3
"""Generate a MUST/SHOULD/MAY compliance matrix for fnp-python parity tests.

Scans crates/fnp-python/src/lib.rs for `#[test] fn <name>_matches_numpy_*`
functions, classifies each by numpy surface (ufunc / reducer / constructor /
linalg / fft / random / structured / testing / misc) and requirement level,
and emits a Markdown matrix to crates/fnp-python/COMPLIANCE.generated.md.

Classification rules — see tables below. Rationale: most fnp-python tests
assert bit-for-bit numpy parity on public numpy.* API. Pure-numeric kernels
(ufuncs, reducers, linalg, fft, constructors) that numpy's documentation
defines as numerical contract are MUST. Shape/order/string-rendering paths
that numpy doesn't formally guarantee bit-exact are SHOULD. Debug/inspect
APIs that numpy itself labels "introspective" are MAY.

Usage: python3 scripts/fnp_python_compliance_matrix.py
Exit 0 on success. Writes the matrix to a fixed path so CI can diff it.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
SRC_FILE: Final = REPO_ROOT / "crates" / "fnp-python" / "src" / "lib.rs"
OUT_FILE: Final = REPO_ROOT / "crates" / "fnp-python" / "COMPLIANCE.generated.md"

TEST_RE = re.compile(
    r"^\s*#\[test\]\s*\n\s*fn\s+([a-z_0-9]+_matches_numpy[a-z_0-9]*)\s*\(",
    re.MULTILINE,
)

# A handful of test stems that do NOT use the *_matches_numpy* suffix but
# are still contract-level parity probes. Listed explicitly so the matrix
# can include them.
EXTRA_TESTS: Final = [
    ("frompyfunc_reduce_does_not_delegate_to_numpy_frompyfunc", "runtime_policy"),
    ("vectorize_live_callable_matches_numpy_single_output", "vectorize"),
    ("vectorize_live_callable_matches_numpy_multi_output", "vectorize"),
    ("vectorize_excluded_argument_matches_numpy", "vectorize"),
]

# Domain → (RequirementLevel, one-line summary)
# MUST = numerical contract; SHOULD = shape/order/string; MAY = inspect.
DOMAIN_LEVELS: Final = {
    "ufunc_numeric":       ("MUST",   "Elementwise numerical ufunc parity"),
    "reducer":             ("MUST",   "Axis reducers (sum, mean, var, nan*, quantile…)"),
    "linalg":              ("MUST",   "Linear algebra kernels (passthrough / native)"),
    "fft":                 ("MUST",   "FFT family (fft/ifft/rfft/fftshift…)"),
    "constructor":         ("MUST",   "Array constructors (zeros/ones/arange/linspace…)"),
    "search_sort":         ("MUST",   "Sort / partition / searchsorted / argsort"),
    "set_op":              ("MUST",   "Set ops (union1d / intersect1d / isin / unique)"),
    "random":              ("MUST",   "RNG distributions and bit streams"),
    "math_special":        ("MUST",   "Special-function math (bessel, gamma, beta…)"),
    "reshape_transpose":   ("SHOULD", "Shape / transpose / broadcasting surfaces"),
    "concat_split":        ("SHOULD", "Concatenate / split / stack families"),
    "index_select":        ("SHOULD", "Advanced indexing / take / put / where / choose"),
    "indexing_iter":       ("SHOULD", "Iteration / ogrid / mgrid / broadcast helpers"),
    "structured":          ("SHOULD", "Structured-dtype and record surfaces"),
    "masked":              ("SHOULD", "Masked arrays"),
    "string":              ("SHOULD", "String ufuncs / char functions"),
    "datetime":            ("SHOULD", "datetime64 / timedelta64 arithmetic"),
    "testing":             ("SHOULD", "numpy.testing.assert_* helpers"),
    "dtype_policy":        ("SHOULD", "dtype promotion / casting / can_cast"),
    "runtime_policy":      ("MAY",    "frompyfunc / delegation observability probes"),
    "inspect":             ("MAY",    "__version__ / get_include / show_config"),
    "vectorize":           ("SHOULD", "numpy.vectorize contract"),
    "misc":                ("SHOULD", "Uncategorized numpy surface"),
}

# Test-name → domain. Ordered from most specific to least specific: first
# match wins.
CLASSIFIERS: Final = [
    # FFT surface.
    (re.compile(r"^(fft|ifft|rfft|irfft|hfft|ihfft|fftshift|ifftshift|fftfreq|rfftfreq|rfft2|irfft2|rfftn|irfftn)"), "fft"),
    # Linalg.
    (re.compile(r"^(svd|qr|cholesky|eig|eigh|eigvals|eigvalsh|inv|solve|lstsq|pinv|slogdet|det|matrix_rank|matrix_power|matrix_norm|matrix_transpose|vecdot|multi_dot|tensorsolve|tensorinv|solve_triangular|linalg|kron|vdot)"), "linalg"),
    # RNG.
    (re.compile(r"^(random|rng|pcg64|philox|mt19937|sfc64|generator)"), "random"),
    # Reducers.
    (re.compile(r"^(sum|prod|mean|std|var|median|percentile|quantile|average|cov|corrcoef|min|max|ptp|all|any|count_nonzero|trace|cumsum|cumprod|cummax|cummin|nan(mean|sum|prod|std|var|max|min|argmax|argmin|percentile|quantile|median))"), "reducer"),
    # Constructors.
    (re.compile(r"^(zeros|ones|empty|full|eye|identity|arange|linspace|geomspace|logspace|meshgrid|indices|mgrid|ogrid|copy|frombuffer|fromstring|fromiter|fromfile|loadtxt|genfromtxt|diag|diagflat|tri|tril_indices|triu_indices|tril|triu|vander|array|asarray|asanyarray|asfortranarray|ascontiguousarray|asarray_chkfinite)"), "constructor"),
    # Search / sort / partition.
    (re.compile(r"^(sort|argsort|argmin|argmax|partition|argpartition|searchsorted|lexsort|sort_complex|flatnonzero|argwhere|nonzero)"), "search_sort"),
    # Set ops.
    (re.compile(r"^(union1d|intersect1d|setdiff1d|isin|unique|in1d|ediff1d)"), "set_op"),
    # String surface.
    (re.compile(r"^(str_|string_|chararray|strings_|char_)"), "string"),
    # Datetime surface.
    (re.compile(r"^(datetime|timedelta|busday)"), "datetime"),
    # Structured.
    (re.compile(r"^(recfunctions|structured|record_array)"), "structured"),
    # Masked.
    (re.compile(r"^(masked|ma_|getmask|make_mask|mask_or|compressed)"), "masked"),
    # Testing.
    (re.compile(r"^(testing_|assert_)"), "testing"),
    # dtype policy.
    (re.compile(r"^(astype|can_cast|result_type|promote_types|common_type|min_scalar_type|obj2sctype|issubdtype|isdtype)"), "dtype_policy"),
    # Reshape / transpose / broadcast.
    (re.compile(r"^(reshape|transpose|swapaxes|moveaxis|rollaxis|ravel|flatten|squeeze|expand_dims|broadcast|atleast_|roll|rot90|flip|fliplr|flipud)"), "reshape_transpose"),
    # Concat / stack / split.
    (re.compile(r"^(concatenate|stack|hstack|vstack|dstack|column_stack|row_stack|block|split|hsplit|vsplit|dsplit|array_split|tile|repeat|pad|trim_zeros)"), "concat_split"),
    # Indexing / take / put / where.
    (re.compile(r"^(take|put|putmask|place|compress|extract|where|choose|select|diagonal|fill_diagonal|put_along_axis|take_along_axis|ix_|ravel_multi_index|unravel_index)"), "index_select"),
    # Indexing iter helpers.
    (re.compile(r"^(broadcast_to|broadcast_arrays|nditer|ndindex|ix_helper)"), "indexing_iter"),
    # Math special functions.
    (re.compile(r"^(i0|sinc|kaiser|blackman|hanning|hamming|bartlett|bessel|beta|gamma|lgamma|trapezoid|trapz|interp|digitize|bincount|histogram)"), "math_special"),
    # Numeric ufuncs.
    (re.compile(r"^(add|subtract|multiply|divide|true_divide|floor_divide|mod|remainder|power|pow|abs|absolute|negative|positive|reciprocal|sign|copysign|sqrt|square|cbrt|exp|expm1|exp2|log|log1p|log2|log10|sin|cos|tan|arcsin|arccos|arctan|arctan2|sinh|cosh|tanh|arcsinh|arccosh|arctanh|radians|degrees|rad2deg|deg2rad|hypot|frexp|ldexp|modf|floor|ceil|fix|rint|round|trunc|nan_to_num|isnan|isinf|isfinite|isneginf|isposinf|signbit|spacing|nextafter|isclose|allclose|array_equal|array_equiv|equal|not_equal|less|less_equal|greater|greater_equal|logical_and|logical_or|logical_not|logical_xor|bitwise_and|bitwise_or|bitwise_xor|bitwise_not|left_shift|right_shift|maximum|minimum|fmax|fmin|logaddexp|logaddexp2|clip|conj|conjugate|real|imag|angle|mod|divmod)"), "ufunc_numeric"),
    # frompyfunc / vectorize etc.
    (re.compile(r"^(frompyfunc|vectorize|apply_along|apply_over)"), "runtime_policy"),
    # Inspect.
    (re.compile(r"^(get_include|show_config|show_runtime|version|__version__|lookfor|info|deprecate)"), "inspect"),
]


def classify(name: str) -> str:
    for pattern, domain in CLASSIFIERS:
        if pattern.match(name):
            return domain
    return "misc"


def main() -> int:
    if not SRC_FILE.is_file():
        print(f"source file not found: {SRC_FILE}", file=sys.stderr)
        return 2
    source = SRC_FILE.read_text(encoding="utf-8")
    matches = sorted(set(TEST_RE.findall(source)))
    tagged = [(name, classify(name)) for name in matches]
    for name, domain in EXTRA_TESTS:
        if name not in matches:
            tagged.append((name, domain))

    by_domain: dict[str, list[str]] = defaultdict(list)
    for name, domain in tagged:
        by_domain[domain].append(name)

    by_level: dict[str, int] = defaultdict(int)
    for domain, names in by_domain.items():
        level = DOMAIN_LEVELS.get(domain, ("SHOULD", "Uncategorized"))[0]
        by_level[level] += len(names)

    total = sum(by_level.values())

    lines: list[str] = []
    lines.append("# fnp-python Compliance Matrix (auto-generated)\n")
    lines.append("> Source: `scripts/fnp_python_compliance_matrix.py`\n")
    lines.append(f"> Total parity tests scanned: **{total}**\n")
    lines.append("> Do not edit by hand — regenerate via `python3 scripts/fnp_python_compliance_matrix.py`.\n")
    lines.append("\n## Summary by RequirementLevel\n")
    lines.append("")
    lines.append("| Level | Count | Share |")
    lines.append("|-------|------:|------:|")
    for level in ("MUST", "SHOULD", "MAY"):
        count = by_level.get(level, 0)
        share = (count / total * 100.0) if total else 0.0
        lines.append(f"| {level} | {count} | {share:.1f}% |")
    lines.append(f"| **Total** | **{total}** | 100.0% |")
    lines.append("")
    lines.append("## Per-domain breakdown\n")
    lines.append("| Domain | Level | Tests | Description |")
    lines.append("|--------|:-----:|------:|-------------|")
    for domain in sorted(by_domain):
        level, desc = DOMAIN_LEVELS.get(domain, ("SHOULD", "Uncategorized surface"))
        count = len(by_domain[domain])
        lines.append(f"| `{domain}` | {level} | {count} | {desc} |")
    lines.append("")
    lines.append("## Per-test classification\n")
    lines.append("| Test | Domain | Level |")
    lines.append("|------|--------|:-----:|")
    for name, domain in sorted(tagged):
        level = DOMAIN_LEVELS.get(domain, ("SHOULD", ""))[0]
        lines.append(f"| `{name}` | `{domain}` | {level} |")
    lines.append("")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_FILE}")
    print("domains:")
    for domain in sorted(by_domain):
        level = DOMAIN_LEVELS.get(domain, ("SHOULD", ""))[0]
        print(f"  {domain:<22} level={level:<6} tests={len(by_domain[domain])}")
    print(f"levels: MUST={by_level.get('MUST',0)} SHOULD={by_level.get('SHOULD',0)} MAY={by_level.get('MAY',0)} total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
