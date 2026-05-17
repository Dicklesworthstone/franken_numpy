# fnp-* reality-check — numpy.__all__ coverage — 2026-05-13 (touched 2026-05-16)

Comparing the Python-facing surface exposed by `fnp_python` against `numpy.__all__` (NumPy 2.x). Refreshed after the May 2026 parity wave closed the original gap; lock-in is structural via the `fnp_python_covers_full_numpy_all` conformance test, which iterates `numpy.__all__` and fails CI if any name regresses. **This is now a steady-state baseline plus regression-protection note, not a gap list.**

## Vision (from README.md § The Solution, item 1)

> "Absolute behavioral compatibility with legacy NumPy. Not a subset, not 'inspired by.' The full API, edge cases and all."

The measuring stick is `numpy.__all__` — the 499 names NumPy publishes as its public Python surface.

## Headline numbers (current)

- `numpy.__all__` names: **499**
- `numpy.__all__` ∩ `fnp_python`: **499** (**100.0%** of numpy's top-level surface is reachable as `fnp_python.<name>`)
- `numpy.__all__` \ `fnp_python` (gap): **0** (0%)
- Names in `fnp_python` that are NOT in `numpy.__all__`: ~130 internal helpers and flat-namespace aliases (`linalg_*`, `ma_*`, etc.) used by conformance tests; not user-facing surface. The live count is reported by the API coverage gate (`exports / covered / excluded / missing`) — at the 2026-05-13 snapshot: `exports=633`, `covered=599`, `excluded=34`, `missing=0`.

## Coverage progression

| Date       | Covered / 499 | Share | Notes                                              |
|------------|---------------|-------|----------------------------------------------------|
| 2026-04-22 | 216           | 43.3% | Initial audit baseline (this document, prior rev). |
| 2026-05-09 | ~480          | ~96%  | Bulk wrapper wave landed ufunc/array/random/poly.  |
| 2026-05-13 | **499**       | **100.0%** | mgrid/ogrid + strings/char/rec/emath/matrixlib + index-trick + errstate + diag + remaining attrs + core/f2py. |

## How parity is maintained

The 100% coverage is structurally enforced by a conformance test that fails fast if any future change drops a re-export:

```
crates/fnp-python/tests/conformance_remaining_top_level_attrs.rs::
    fnp_python_covers_full_numpy_all
```

This test iterates `np.__all__` at run-time and asserts every name is reachable via the live `fnp_python` module. If a refactor accidentally drops a re-export, CI fails with the exact list of newly-missing names.

## Architectural choices that drove the close-out

1. **Engine vs surface separation.** The `fnp_python` crate is a parity-oracle surface — by design, the heavy lifting lives in `fnp-ufunc`, `fnp-ndarray`, `fnp-linalg`, `fnp-io`, `fnp-iter`, `fnp-random` Rust crates. `fnp_python` wires those to Python and falls back to numpy where there is no engine substitute. That makes most of the 499 names cheap to expose.
2. **Re-export-where-safe pattern.** Submodules whose semantics are pure-numpy state (`numpy.strings`, `numpy.char`, `numpy.rec`, `numpy.emath`, `numpy.matrixlib`, `numpy.ma`, `numpy.testing`, `numpy.typing`, `numpy.ctypeslib`, `numpy.core`, `numpy.f2py`, plus class instances like `numpy.mgrid`/`numpy.ogrid` and constants like `numpy.s_`/`numpy.True_`) are re-exported verbatim via `m.add(name, &numpy.getattr(name))`. This keeps fnp_python identity-equal to numpy for those surfaces, with zero maintenance and 100% upstream parity (including version-gated and deprecation behaviors).
3. **Native fast-paths with numpy fallback.** Performance-relevant wrappers (sum, mean, var, sort, partition, fft.*, linalg.*, polynomial.*) implement the common shape on the Rust engine and fall back to numpy for unusual kwarg combinations. This preserves drop-in semantics while delivering native-speed for the hot paths.

## Conformance coverage gates

Beyond `fnp_python_covers_full_numpy_all`, the parity wave landed dozens of focused conformance tests under `crates/fnp-python/tests/conformance_*.rs`. The API coverage gate `cargo run -p fnp-conformance --bin run_fnp_python_api_coverage -- --fail-on-missing` reports `exports=633 covered=599 missing=0 excluded=34 orphan_suites=0` as of 2026-05-13.

## What's still NOT in scope here

This audit measures Python-surface parity. It does not measure:

- **Behavioral parity** for every kwarg permutation (covered by per-function conformance tests in `crates/fnp-python/tests/conformance_*.rs`).
- **Engine quality** for ufuncs that take the native Rust fast-path (covered by `crates/fnp-ufunc/tests/*` and the conformance harness in `crates/fnp-conformance`).
- **Performance parity** vs numpy (separately tracked under `tests/artifacts/perf/`).
- **Diagnostic / exception-class parity** (tracked under the `franken_numpy-33vtd` diagnostic-parity wave).

## Reproduction

```bash
python3 - <<'PY'
import importlib.util, os
import numpy as np
candidates = []
for d in ('/data/projects/.cargo-target-fnp-pinkdesert-verify/debug/deps',
          '/data/projects/.cargo-target-fnp-cc-array-api/debug/deps'):
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        if f.startswith('libfnp_python') and f.endswith('.so'):
            candidates.append(os.path.join(d, f))
candidates.sort()
spec = importlib.util.spec_from_file_location('fnp_python', candidates[-1])
fnp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fnp)

missing = [n for n in np.__all__ if not hasattr(fnp, n)]
print(f'{len(np.__all__) - len(missing)}/{len(np.__all__)} = '
      f'{100*(len(np.__all__) - len(missing))/len(np.__all__):.1f}%')
print(f'Missing: {missing}')
PY
```

## Notes

- The lock-in conformance test runs against the *live* numpy version on the build host, so the parity guarantee holds as numpy evolves. New names that numpy adds to `__all__` will fail the test until they are explicitly added to the re-export block in `crates/fnp-python/src/lib.rs`.
- Prior version of this document (2026-04-22) recorded 43.3% coverage and proposed a "close the gap" multi-session program. That program is now complete; the bead trail is in `.beads/issues.jsonl` (search for `franken_numpy-vek3z`, `ghsx4`, `bntjh`, `tmg0c`, `4t0ql`, `xdxvn`, `r1xmi`, `cp8xw`, `t3eb4`, `dm9bn`, `0xpje`).
