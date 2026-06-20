# UFunc Boolean-Index Same-Host Verification

Date: 2026-06-20
Agent: BlackThrush / cod-b
Bead: franken_numpy-ixs5y.251

Decision: KEEP, no source changes in this closeout.

Decision host:
- FNP Criterion: RCH-selected `vmi1149989`.
- NumPy comparator: direct SSH on `vmi1149989` as `ubuntu`.
- NumPy: 2.2.4, Python: 3.13.7.
- Thread env: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`.

Performance:

| Row | FNP ns/iter | NumPy median ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `boolean_index_f64_masked_sparse/100000` | 43,634 | 99,813 | 0.437x | 2.29x |
| `boolean_index_f64_masked_sparse/1000000` | 628,093 | 1,355,257 | 0.463x | 2.16x |

Ledger: 2 wins, 0 losses, 0 neutral.

Artifacts:
- `fnp_boolean_index_hz2.txt`: Criterion output; filename kept from the initial
  hz2 preference, but RCH selected `vmi1149989`.
- `numpy_boolean_index_vmi1149989_retry2.txt`: successful NumPy comparator.
- `numpy_boolean_index_vmi1149989.txt`: invalid quoting probe, no timings.
- `numpy_boolean_index_vmi1149989_retry.txt`: invalid f-string quoting probe, no
  timings.
- `test_boolean_index_vmi1149989.txt`: focused correctness.
- `test_fnp_ufunc_full_vmi1149989.txt`: full `fnp-ufunc` tests; RCH selected
  `hz2`.
- `check_fnp_ufunc_vmi_or_hz.txt`: `cargo check -p fnp-ufunc --all-targets`;
  RCH selected `vmi1153651`.
- `build_release_fnp_ufunc_vmi_or_hz.txt`: `cargo build -p fnp-ufunc --release`;
  RCH selected `vmi1153651`.
- `clippy_fnp_ufunc_vmi_or_hz.txt`: invalid clippy attempt; missing component on
  `vmi1149989`.
- `clippy_fnp_ufunc_vmi1149989_after_component_install.txt`: clippy pass after
  installing pinned nightly clippy on `vmi1149989`.

Validation summary:
- Focused boolean-index tests: PASS, 4/4.
- Full `fnp-ufunc` tests: PASS, 2244 passed, 0 failed, 41 ignored, integration
  tests green, doctests ignored as expected.
- `cargo check -p fnp-ufunc --all-targets`: PASS.
- `cargo build -p fnp-ufunc --release`: PASS.
- `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`: PASS after worker
  component repair.

Retry predicate: do not retune this wrapper path without fresh losing evidence.
Future work must target a lower primitive such as mask representation, mask
decode traffic, or a distinct sidecar-preserving gather core.
