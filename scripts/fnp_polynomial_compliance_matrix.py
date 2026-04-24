#!/usr/bin/env python3
"""Generate coverage matrix: fnp_python polynomial surface vs numpy.polynomial.

Static analysis (no numpy / fnp_python import required). Parses:
  - crates/fnp-python/src/lib.rs for #[pyfunction] fns prefixed with
    poly/cheb/leg/herm/herme/lag that wrap numpy.polynomial.*, plus
    the class re-export block (Polynomial, Chebyshev, …, Laguerre).
  - hand-coded NUMPY_POLYNOMIAL_SURFACE: the snapshot of the public
    numpy.polynomial API we target for parity.

Emits crates/fnp-python/POLYNOMIAL_COMPLIANCE.generated.md with:
  - summary counts by MUST / SHOULD / MAY
  - per-name present/missing table
  - exit 0 on success; exit 2 when MUST coverage drops below 95%.

Usage: python3 scripts/fnp_polynomial_compliance_matrix.py

Rationale for the level map: core ops (add, sub, mul, val, roots,
fromroots, der, int, div, pow, line, mulx, trim) are numerical
contract = MUST. Basis conversions (poly2{cheb,herm,…} and reverse)
are SHOULD. Vandermonde / companion / fit helpers are SHOULD because
numpy's algorithm is documented but users rely on Python classes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
SRC_FILE: Final = REPO_ROOT / "crates" / "fnp-python" / "src" / "lib.rs"
OUT_FILE: Final = REPO_ROOT / "crates" / "fnp-python" / "POLYNOMIAL_COMPLIANCE.generated.md"

# Public names exposed under numpy.polynomial + each subpackage (2.0+).
NUMPY_POLYNOMIAL_SURFACE: Final = {
    # Top-level numpy.polynomial classes.
    "Polynomial":         ("MUST",   "class"),
    "Chebyshev":          ("MUST",   "class"),
    "Legendre":           ("MUST",   "class"),
    "Hermite":            ("MUST",   "class"),
    "HermiteE":           ("MUST",   "class"),
    "Laguerre":           ("MUST",   "class"),
    # numpy.polynomial.polynomial (plain power basis).
    "polyadd":            ("MUST",   "polynomial"),
    "polysub":            ("MUST",   "polynomial"),
    "polymul":            ("MUST",   "polynomial"),
    "polyval":            ("MUST",   "polynomial"),
    "polyroots":          ("MUST",   "polynomial"),
    "polyfromroots":      ("MUST",   "polynomial"),
    "polyder":            ("MUST",   "polynomial"),
    "polyint":            ("MUST",   "polynomial"),
    "polydiv":            ("MUST",   "polynomial"),
    "polypow":            ("MUST",   "polynomial"),
    "polyline":           ("SHOULD", "polynomial"),
    "polytrim":           ("SHOULD", "polynomial"),
    "polyvander":         ("SHOULD", "polynomial"),
    "polyvalfromroots":   ("SHOULD", "polynomial"),
    # chebyshev.
    "chebadd":            ("MUST",   "chebyshev"),
    "chebsub":            ("MUST",   "chebyshev"),
    "chebmul":            ("MUST",   "chebyshev"),
    "chebval":            ("MUST",   "chebyshev"),
    "chebroots":          ("MUST",   "chebyshev"),
    "chebfromroots":      ("MUST",   "chebyshev"),
    "chebder":            ("MUST",   "chebyshev"),
    "chebint":            ("MUST",   "chebyshev"),
    "chebdiv":            ("MUST",   "chebyshev"),
    "chebpow":            ("MUST",   "chebyshev"),
    "chebline":           ("SHOULD", "chebyshev"),
    "chebtrim":           ("SHOULD", "chebyshev"),
    "chebmulx":           ("SHOULD", "chebyshev"),
    "chebvander":         ("SHOULD", "chebyshev"),
    "cheb2poly":          ("SHOULD", "chebyshev"),
    "poly2cheb":          ("SHOULD", "chebyshev"),
    # hermite (physicist's).
    "hermadd":            ("MUST",   "hermite"),
    "hermsub":            ("MUST",   "hermite"),
    "hermmul":            ("MUST",   "hermite"),
    "hermval":            ("MUST",   "hermite"),
    "hermroots":          ("MUST",   "hermite"),
    "hermfromroots":      ("MUST",   "hermite"),
    "hermder":            ("MUST",   "hermite"),
    "hermint":            ("MUST",   "hermite"),
    "hermdiv":            ("MUST",   "hermite"),
    "hermpow":            ("MUST",   "hermite"),
    "hermline":           ("SHOULD", "hermite"),
    "hermtrim":           ("SHOULD", "hermite"),
    "hermmulx":           ("SHOULD", "hermite"),
    "hermvander":         ("SHOULD", "hermite"),
    "herm2poly":          ("SHOULD", "hermite"),
    "poly2herm":          ("SHOULD", "hermite"),
    # hermite_e (probabilist's).
    "hermeadd":           ("MUST",   "hermite_e"),
    "hermesub":           ("MUST",   "hermite_e"),
    "hermemul":           ("MUST",   "hermite_e"),
    "hermeval":           ("MUST",   "hermite_e"),
    "hermeroots":         ("MUST",   "hermite_e"),
    "hermefromroots":     ("MUST",   "hermite_e"),
    "hermeder":           ("MUST",   "hermite_e"),
    "hermeint":           ("MUST",   "hermite_e"),
    "hermediv":           ("MUST",   "hermite_e"),
    "hermepow":           ("MUST",   "hermite_e"),
    "hermeline":          ("SHOULD", "hermite_e"),
    "hermetrim":          ("SHOULD", "hermite_e"),
    "hermemulx":          ("SHOULD", "hermite_e"),
    "hermevander":        ("SHOULD", "hermite_e"),
    "herme2poly":         ("SHOULD", "hermite_e"),
    "poly2herme":         ("SHOULD", "hermite_e"),
    # laguerre.
    "lagadd":             ("MUST",   "laguerre"),
    "lagsub":             ("MUST",   "laguerre"),
    "lagmul":             ("MUST",   "laguerre"),
    "lagval":             ("MUST",   "laguerre"),
    "lagroots":           ("MUST",   "laguerre"),
    "lagfromroots":       ("MUST",   "laguerre"),
    "lagder":             ("MUST",   "laguerre"),
    "lagint":             ("MUST",   "laguerre"),
    "lagdiv":             ("MUST",   "laguerre"),
    "lagpow":             ("MUST",   "laguerre"),
    "lagline":            ("SHOULD", "laguerre"),
    "lagtrim":            ("SHOULD", "laguerre"),
    "lagmulx":            ("SHOULD", "laguerre"),
    "lagvander":          ("SHOULD", "laguerre"),
    "lag2poly":           ("SHOULD", "laguerre"),
    "poly2lag":           ("SHOULD", "laguerre"),
    # legendre.
    "legadd":             ("MUST",   "legendre"),
    "legsub":             ("MUST",   "legendre"),
    "legmul":             ("MUST",   "legendre"),
    "legval":             ("MUST",   "legendre"),
    "legroots":           ("MUST",   "legendre"),
    "legfromroots":       ("MUST",   "legendre"),
    "legder":             ("MUST",   "legendre"),
    "legint":             ("MUST",   "legendre"),
    "legdiv":             ("MUST",   "legendre"),
    "legpow":             ("MUST",   "legendre"),
    "legline":            ("SHOULD", "legendre"),
    "legtrim":            ("SHOULD", "legendre"),
    "legmulx":            ("SHOULD", "legendre"),
    "legvander":          ("SHOULD", "legendre"),
    "leg2poly":           ("SHOULD", "legendre"),
    "poly2leg":           ("SHOULD", "legendre"),
}

CLASS_NAMES: Final = frozenset(
    name for name, (_, cat) in NUMPY_POLYNOMIAL_SURFACE.items() if cat == "class"
)

# #[pyfunction]-annotated function names in lib.rs. Matches
#   #[pyfunction]\n...\nfn NAME(
# with optional attributes between (e.g. #[pyo3(signature = ...)]).
FN_DECL_RE = re.compile(
    r"#\[pyfunction\][^\n]*\n(?:#\[pyo3[^\n]*\]\n)*fn\s+([a-z_][a-z0-9_]*)\s*\(",
)
# Look for eager-setattr of each polynomial class on the polynomial
# submodule. The source literal is:
#   polynomial.setattr(name, cls)?
# after "for name in [ ... ]" iteration over the 6 class names.
POLY_CLASS_LOOP_RE = re.compile(
    r"polynomial\.setattr\(name,\s*cls\)\?.*?for\s+name\s+in\s*\[([^\]]*)\]",
    re.DOTALL,
)
# The PEP-562 __getattr__ fallback installed unconditionally lists the
# recognised class names inside a frozenset literal.
POLY_GETATTR_RE = re.compile(
    r"_CLASS_NAMES\s*=\s*frozenset\(\(([^)]*)\)\)",
)


def parse_source() -> set[str]:
    source = SRC_FILE.read_text(encoding="utf-8")
    names: set[str] = set(FN_DECL_RE.findall(source))
    # Retain only names defined in our target surface so unrelated
    # #[pyfunction] fns (e.g. fft, linalg) don't pollute the report.
    surface_names = set(NUMPY_POLYNOMIAL_SURFACE)
    names &= surface_names
    # Pull class re-exports from the polynomial submodule block. We
    # treat a class as present if it appears in either the eager loop
    # body or the __getattr__ frozenset.
    for loop_match in POLY_CLASS_LOOP_RE.finditer(source):
        for tok in re.findall(r'"([A-Za-z]+)"', loop_match.group(1)):
            if tok in CLASS_NAMES:
                names.add(tok)
    for ga_match in POLY_GETATTR_RE.finditer(source):
        for tok in re.findall(r"'([A-Za-z]+)'", ga_match.group(1)):
            if tok in CLASS_NAMES:
                names.add(tok)
    # Belt-and-braces: if the whole polynomial submodule block simply
    # setattrs every numpy.polynomial class by name in a tuple literal,
    # just look for the six class name tokens near the submodule.
    for cls in CLASS_NAMES:
        if f'"{cls}"' in source:
            names.add(cls)
    return names & (surface_names | CLASS_NAMES)


def main() -> int:
    if not SRC_FILE.is_file():
        print(f"source file not found: {SRC_FILE}", file=sys.stderr)
        return 2

    present = parse_source()
    missing = {
        name: level
        for name, (level, _) in NUMPY_POLYNOMIAL_SURFACE.items()
        if name not in present
    }
    present_spec = {
        name: NUMPY_POLYNOMIAL_SURFACE[name]
        for name in NUMPY_POLYNOMIAL_SURFACE
        if name in present
    }

    counts_by_level: dict[str, tuple[int, int]] = {}
    for level in ("MUST", "SHOULD", "MAY"):
        total = sum(1 for (lvl, _) in NUMPY_POLYNOMIAL_SURFACE.values() if lvl == level)
        p_count = sum(1 for (lvl, _) in present_spec.values() if lvl == level)
        counts_by_level[level] = (p_count, total)

    lines: list[str] = [
        "# fnp_python polynomial Compliance Matrix (auto-generated)\n",
        "> Source: `scripts/fnp_polynomial_compliance_matrix.py`  ",
        "> Target: `numpy.polynomial.*` public API (numpy 2.0+ snapshot)  ",
        "> Do not edit by hand — regenerate via the script.\n",
        "",
        "## Summary by RequirementLevel\n",
        "| Level | Present | Total | Coverage |",
        "|-------|--------:|------:|---------:|",
    ]
    for level in ("MUST", "SHOULD", "MAY"):
        p, t = counts_by_level[level]
        pct = (p / t * 100.0) if t else 0.0
        lines.append(f"| {level} | {p} | {t} | {pct:.1f}% |")
    tp = sum(p for p, _ in counts_by_level.values())
    ta = sum(t for _, t in counts_by_level.values())
    tpc = (tp / ta * 100.0) if ta else 0.0
    lines.append(f"| **Total** | **{tp}** | **{ta}** | **{tpc:.1f}%** |")
    lines.append("")
    # Subtree breakdown
    lines.append("## Per-subtree coverage\n")
    lines.append("| Subtree | Present | Total | Coverage |")
    lines.append("|---------|--------:|------:|---------:|")
    subtree_totals: dict[str, tuple[int, int]] = {}
    for name, (lvl, cat) in NUMPY_POLYNOMIAL_SURFACE.items():
        p, t = subtree_totals.get(cat, (0, 0))
        t += 1
        if name in present:
            p += 1
        subtree_totals[cat] = (p, t)
    for cat in sorted(subtree_totals):
        p, t = subtree_totals[cat]
        pct = (p / t * 100.0) if t else 0.0
        lines.append(f"| `{cat}` | {p} | {t} | {pct:.1f}% |")
    lines.append("")
    if missing:
        lines.append("## Missing names\n")
        lines.append("| Name | Level | Subtree |")
        lines.append("|------|:-----:|---------|")
        for name in sorted(missing):
            lvl, cat = NUMPY_POLYNOMIAL_SURFACE[name]
            lines.append(f"| `{name}` | {lvl} | `{cat}` |")
        lines.append("")
    lines.append("## Present names\n")
    lines.append("| Name | Level | Subtree |")
    lines.append("|------|:-----:|---------|")
    for name in sorted(present_spec):
        lvl, cat = present_spec[name]
        lines.append(f"| `{name}` | {lvl} | `{cat}` |")
    lines.append("")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_FILE}")
    for level in ("MUST", "SHOULD", "MAY"):
        p, t = counts_by_level[level]
        pct = (p / t * 100.0) if t else 0.0
        print(f"  {level:<7} {p}/{t}  ({pct:.1f}%)")
    print(f"missing: {sorted(missing)}")
    must_p, must_t = counts_by_level["MUST"]
    must_pct = (must_p / must_t * 100.0) if must_t else 0.0
    if must_pct < 95.0:
        print(f"WARN: MUST coverage {must_pct:.1f}% < 95% target", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
