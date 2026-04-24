#!/usr/bin/env python3
"""Generate coverage matrix: fnp_python.random vs numpy.random.__all__.

Static analysis (no numpy / fnp_python import required). Parses:
  - crates/fnp-python/src/lib.rs for the install() block that binds
    numpy.random legacy top-level aliases onto fnp_python.random.
  - hand-coded NUMPY_RANDOM_ALL: the snapshot of numpy.random.__all__
    we target for parity.

Emits crates/fnp-python/RANDOM_COMPLIANCE.generated.md with:
  - summary counts by MUST / SHOULD / MAY
  - per-name present/missing table
  - exit 0 always; exit 2 if MUST coverage drops below 95%.

Usage: python3 scripts/fnp_random_compliance_matrix.py

Rationale for the level map: distributions (beta, normal, etc.) are
numerical-contract MUST because downstream code relies on them. Seed
management (seed, get_state, set_state) is SHOULD — callers typically
need it but fnp_python may legitimately diverge on internal state
representation per DISC-008. Class re-exports (BitGenerator,
SeedSequence, RandomState, Generator, bit gens) are MUST since user
code does isinstance checks. Deprecated aliases (ranf, sample) are MAY.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
SRC_FILE: Final = REPO_ROOT / "crates" / "fnp-python" / "src" / "lib.rs"
OUT_FILE: Final = REPO_ROOT / "crates" / "fnp-python" / "RANDOM_COMPLIANCE.generated.md"

# Snapshot of numpy.random.__all__ (numpy 2.0+). If numpy adds new
# symbols we'll catch them on next regen.
NUMPY_RANDOM_ALL: Final = {
    # Classes
    "Generator":                    "MUST",
    "RandomState":                  "MUST",
    "SeedSequence":                 "MUST",
    "BitGenerator":                 "MUST",
    "MT19937":                      "MUST",
    "PCG64":                        "MUST",
    "PCG64DXSM":                    "MUST",
    "Philox":                       "MUST",
    "SFC64":                        "MUST",
    "default_rng":                  "MUST",
    # State management
    "seed":                         "SHOULD",
    "get_state":                    "SHOULD",
    "set_state":                    "SHOULD",
    # Basic random
    "rand":                         "MUST",
    "randn":                        "MUST",
    "randint":                      "MUST",
    "random":                       "MUST",
    "random_sample":                "MUST",
    "random_integers":              "MUST",
    "tomaxint":                     "MUST",
    "bytes":                        "MUST",
    "choice":                       "MUST",
    "shuffle":                      "MUST",
    "permutation":                  "MUST",
    # Deprecated aliases
    "ranf":                         "MAY",
    "sample":                       "MAY",
    # Distributions
    "beta":                         "MUST",
    "binomial":                     "MUST",
    "chisquare":                    "MUST",
    "dirichlet":                    "MUST",
    "exponential":                  "MUST",
    "f":                            "MUST",
    "gamma":                        "MUST",
    "geometric":                    "MUST",
    "gumbel":                       "MUST",
    "hypergeometric":               "MUST",
    "laplace":                      "MUST",
    "logistic":                     "MUST",
    "lognormal":                    "MUST",
    "logseries":                    "MUST",
    "multinomial":                  "MUST",
    "multivariate_normal":          "MUST",
    "negative_binomial":            "MUST",
    "noncentral_chisquare":         "MUST",
    "noncentral_f":                 "MUST",
    "normal":                       "MUST",
    "pareto":                       "MUST",
    "poisson":                      "MUST",
    "power":                        "MUST",
    "rayleigh":                     "MUST",
    "standard_cauchy":              "MUST",
    "standard_exponential":         "MUST",
    "standard_gamma":               "MUST",
    "standard_normal":              "MUST",
    "standard_t":                   "MUST",
    "triangular":                   "MUST",
    "uniform":                      "MUST",
    "vonmises":                     "MUST",
    "wald":                         "MUST",
    "weibull":                      "MUST",
    "zipf":                         "MUST",
}

# Classes that fnp-python adds via pyo3 add_class! — extracted by regex.
CLASS_ADD_RE = re.compile(
    r"random\.add_class::<Py([A-Za-z0-9_]+)>\(\)\?",
)
# Class name-map: the #[pyclass(name = "X")] value, extracted from the
# struct definition. For this generator we use a hard-coded map because
# the pyclass name differs from the Rust struct name.
PYCLASS_NAMES: Final = {
    "PySeedSequence":       "SeedSequence",
    "PyRandomGenerator":    "Generator",
    "PyRandomState":        "RandomState",
    "PyMt19937":            "MT19937",
    "PyPcg64":              "PCG64",
    "PyPcg64Dxsm":          "PCG64DXSM",
    "PyPhilox":             "Philox",
    "PySfc64":              "SFC64",
}
# Functions added via add_function on the random submodule.
FN_ADD_RE = re.compile(
    r"random\.add_function\(wrap_pyfunction!\(([a-z_][a-z0-9_]*),\s*&random\)",
)
# The install() helper iterates a tuple of legacy alias names to bind.
# Pull them out by finding the for-name-in block and the two explicit
# alias assignments at the end.
INSTALL_BLOCK_RE = re.compile(
    r"\"def install\(mod, RandomState\):.*?for name in \((?P<names>.*?)\).*?\"",
    re.DOTALL,
)
MOD_SETATTR_RE = re.compile(r"mod\.([a-z_][a-z0-9_]*)\s*=", re.MULTILINE)


def parse_pyrandomstate_methods(source: str) -> set[str]:
    """Return the set of methods actually implemented on PyRandomState."""
    start = source.find("impl PyRandomState {")
    if start < 0:
        return set()
    # Find the matching closing brace (simple depth counter).
    depth = 0
    end = start
    for i, ch in enumerate(source[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    block = source[start:end]
    # Methods are declared as `fn <name>(` with optional `&mut self` etc.
    return set(re.findall(r"\n\s*fn\s+([a-z_][a-z0-9_]*)\s*\(", block))


def parse_source() -> set[str]:
    source = SRC_FILE.read_text(encoding="utf-8")
    names: set[str] = set()
    for cls_rust in CLASS_ADD_RE.findall(source):
        pyname = PYCLASS_NAMES.get(f"Py{cls_rust}", cls_rust)
        names.add(pyname)
    for fn in FN_ADD_RE.findall(source):
        names.add(fn)
    # BitGenerator is installed via setattr or via a lazy __getattr__.
    if 'random.setattr("BitGenerator"' in source:
        names.add("BitGenerator")
    if (
        "if name == 'BitGenerator'" in source
        and "import numpy.random" in source
    ):
        names.add("BitGenerator")
    # install() binds aliases from a tuple, guarded by hasattr(_rand, name).
    # Only count alias names whose PyRandomState method actually exists.
    rs_methods = parse_pyrandomstate_methods(source)
    install_match = INSTALL_BLOCK_RE.search(source)
    if install_match:
        raw = install_match.group("names")
        for alias in re.findall(r"'([a-z_][a-z0-9_]*)'", raw):
            if alias in rs_methods:
                names.add(alias)
    # Explicit aliases (ranf = _rand.random_sample, sample = _rand.random_sample).
    for alias_name in MOD_SETATTR_RE.findall(source):
        # Only count when the target actually resolves (random_sample on PyRandomState).
        if "random_sample" in rs_methods:
            names.add(alias_name)
    return names


def main() -> int:
    if not SRC_FILE.is_file():
        print(f"source file not found: {SRC_FILE}", file=sys.stderr)
        return 2

    present = parse_source()
    missing = {name: level for name, level in NUMPY_RANDOM_ALL.items() if name not in present}
    present_names = {name: level for name, level in NUMPY_RANDOM_ALL.items() if name in present}

    counts_by_level: dict[str, tuple[int, int]] = {}
    for level in ("MUST", "SHOULD", "MAY"):
        total = sum(1 for v in NUMPY_RANDOM_ALL.values() if v == level)
        present_count = sum(1 for v in present_names.values() if v == level)
        counts_by_level[level] = (present_count, total)

    lines: list[str] = []
    lines.append("# fnp_python.random Compliance Matrix (auto-generated)\n")
    lines.append("> Source: `scripts/fnp_random_compliance_matrix.py`  \n")
    lines.append("> Target: `numpy.random.__all__` (numpy 2.0+ snapshot)  \n")
    lines.append("> Do not edit by hand — regenerate via the script.\n")
    lines.append("")
    lines.append("## Summary by RequirementLevel\n")
    lines.append("| Level | Present | Total | Coverage |")
    lines.append("|-------|--------:|------:|---------:|")
    for level in ("MUST", "SHOULD", "MAY"):
        p, t = counts_by_level[level]
        pct = (p / t * 100.0) if t else 0.0
        lines.append(f"| {level} | {p} | {t} | {pct:.1f}% |")
    total_present = sum(p for p, _ in counts_by_level.values())
    total_all = sum(t for _, t in counts_by_level.values())
    total_pct = (total_present / total_all * 100.0) if total_all else 0.0
    lines.append(f"| **Total** | **{total_present}** | **{total_all}** | **{total_pct:.1f}%** |")
    lines.append("")
    if missing:
        lines.append("## Missing names\n")
        lines.append("| Name | Level |")
        lines.append("|------|:-----:|")
        for name in sorted(missing):
            lines.append(f"| `{name}` | {missing[name]} |")
        lines.append("")
    lines.append("## Present names\n")
    lines.append("| Name | Level |")
    lines.append("|------|:-----:|")
    for name in sorted(present_names):
        lines.append(f"| `{name}` | {present_names[name]} |")
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
