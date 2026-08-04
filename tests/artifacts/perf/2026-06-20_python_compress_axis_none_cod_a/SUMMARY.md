# 2026-06-20 Python compress axis=None bitmask gather

Agent: `YellowElk` / `cod-a`
Parent bead: `franken_numpy-ixs5y`
Crate/API: `fnp-python` / `np.compress(condition, a)` flat `axis=None`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`

## Decision

SHIP.

The retained lever replaces speculative per-element stores in `compact_typed`
with an 8-lane condition mask and trailing-zero selected-lane gathers. Sparse
branching and small NumPy delegation were measured and reverted.

## Counted Worker

Performance rows were counted only from `vmi1167313`.

| Row | Baseline FNP | Baseline FNP/NumPy | Final FNP | Final/Old | Final/NumPy |
|---|---:|---:|---:|---:|---:|
| `compress_f64_axis_none_100000` | 167,603 ns | 1.215x loss | 142,735 ns | 0.852x win | 1.015x neutral/noisy |
| `compress_f64_axis_none_1000000` | 1,902,857 ns | 0.792x win | 1,853,998 ns | 0.974x win | 0.805x win |

Final old/new gate: 2 wins / 0 losses / 0 neutral.
Final NumPy gate: 1 win / 0 losses / 1 neutral.

## Rejected Probes

| Probe | Old/New FNP | Reason |
|---|---:|---|
| Sparse kept-only branch | 0 wins / 2 losses / 0 neutral | Regressed 100k to 1.080x old and 1M to 1.044x old. |
| Small NumPy delegate | 0 wins / 2 losses / 0 neutral | Regressed 100k to 1.030x old and 1M to 1.044x old; raw NumPy win was not accepted. |

## Validation

- `rch exec -- cargo test -p fnp-python --test conformance_compress_choose_diagonal compress -- --nocapture`: 13 passed, 0 failed.
- `rch exec -- cargo check -p fnp-python --lib --bench criterion_python_surface`: passed with three inherited warnings.
- `rch exec -- cargo build -p fnp-python --release`: passed with the same inherited warnings.
- `cargo fmt -p fnp-python -- --check`: reports broad pre-existing `fnp-python` rustfmt drift; new benchmark hunk was manually aligned.
- `rch exec -- cargo clippy -p fnp-python --lib --bench criterion_python_surface -- -D warnings`: failed on 35 existing `fnp-python` lint errors outside this hunk.
- `ubs crates/fnp-python/src/lib.rs crates/fnp-python/benches/criterion_python_surface.rs docs/NEGATIVE_EVIDENCE.md docs/RELEASE_READINESS_SCORECARD.md tests/artifacts/perf/2026-06-20_python_compress_axis_none_cod_a/SUMMARY.md`: completed with broad existing `fnp-python` scanner findings.
- `git diff --check`: passed.

## Raw Logs

- `baseline_compress_axis_none.txt`
- `candidate_sparse_branch_compress_axis_none.txt`
- `candidate_small_numpy_delegate_compress_axis_none.txt` (local fallback interrupted; not counted)
- `candidate_small_numpy_delegate_compress_axis_none_remote.txt`
- `candidate_bitmask_gather_compress_axis_none.txt`
- `test_conformance_compress.txt`
- `check_fnp_python_lib_bench.txt`
- `build_fnp_python_release.txt`
- `fmt_fnp_python_check.txt`
- `clippy_fnp_python_lib_bench.txt`
- `ubs_changed_files.txt`
- `git_diff_check.txt`
