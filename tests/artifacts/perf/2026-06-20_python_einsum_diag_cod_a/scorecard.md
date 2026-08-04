# 2026-06-20 Python einsum diagonal gauntlet

Scope:
- Parent bead: `franken_numpy-ixs5y`.
- Agent: `BlackThrush` / `cod-a`.
- Crate: `fnp-python`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Target gap: `fnp_einsum_diag_f64_4000`, previously a Python-boundary loss versus `numpy.einsum("ii->i", a)`.

Lever kept:
- Add an early exact-NumPy-ndarray f64 single-operand diagonal/trace gate before dtype-policy probing.
- Use `PyBuffer<f64>` metadata for dtype/shape/readonly checks, a cached `numpy.ndarray` type object for exact type gating, and interned Python method/keyword names for the diagonal writable-view construction.
- Preserve fallback for kwargs other than `optimize`, non-f64 inputs, non-exact ndarray inputs, non-square matrices, non-contiguous trace inputs, and all general einsum forms.

Commands:
- Baseline local fallback invoked through rch: `AGENT_NAME=BlackThrush RCH_WORKER=hz2 RCH_WORKERS=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_einsum_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1`
- Candidate/final local: `AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a cargo bench -p fnp-python --bench criterion_python_surface -- python_einsum_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1`
- Compile check: `AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a cargo check -p fnp-python --lib --bench criterion_python_surface`
- RCH conformance: `AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-python --test conformance_einsum`
- Final RCH head-to-head: `AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_einsum_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1`
- RCH release build: `AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo build -p fnp-python --release`

Decision table:

| Workload | Evidence | FNP median | NumPy median | FNP/NumPy | FNP vs baseline | Verdict |
|---|---|---:|---:|---:|---:|---|
| `trace_f64_4000` | local baseline | 18.425 us | 15.014 us | 1.227x | 1.000x | baseline loss |
| `diag_f64_4000` | local baseline | 4.5756 us | 1.0796 us | 4.238x | 1.000x | baseline loss |
| `trace_f64_4000` | local final | 15.296 us | 15.852 us | 0.965x | 0.830x | keep |
| `diag_f64_4000` | local final | 883.98 ns | 1.0942 us | 0.808x | 0.193x | keep |
| `trace_f64_4000` | rch final, `vmi1227854` | 5.9900 us | 5.2275 us | 1.146x | n/a | residual negative evidence |
| `diag_f64_4000` | rch final, `vmi1227854` | 805.39 ns | 889.51 ns | 0.905x | n/a | keep |

Intermediate candidates:

| Candidate | Diagonal FNP | Diagonal NumPy | FNP/NumPy | Verdict |
|---|---:|---:|---:|---|
| typed buffer gate + string type check | 1.2799 us | 1.0142 us | 1.262x | improved but still loss |
| cached ndarray type gate | 1.0609 us | 1.0194 us | 1.041x | neutral/slight loss |
| cached ndarray type + interned method names | 883.98 ns | 1.0942 us | 0.808x | keep |

Correctness:
- `rch exec -- cargo test -p fnp-python --test conformance_einsum`: 28 passed, 0 failed.
- The existing `einsum_f64_single_operand_diagonal_view_and_trace_golden_sha256` test covers writable diagonal view semantics and trace parity for the touched path.
- `rch exec -- cargo build -p fnp-python --release`: passed on `vmi1149989`, with the same pre-existing warnings.

Known verification limits:
- `cargo check -p fnp-python --benches` reaches unrelated pre-existing `#[cfg(test)]` call-site drift in the lib test target.
- `cargo fmt -p fnp-python -- --check` reports broad pre-existing formatting drift across `fnp-python` and tests; no formatter was run because it would rewrite unrelated files.
- `cargo check -p fnp-python --lib --bench criterion_python_surface` passes but reports pre-existing warnings for `StackHelperKind::{Depth, Column}`, `extract_mask_operand`, and `count_valid_elements`.

Retry predicate:
- Do not retry wrapper-level pre-policy diagonal dispatch; this run supersedes it with cached exact-type and interned-name proof.
- Next credible retry must remove or bypass the remaining `diagonal()+setflags(write=True)` Python method dispatch while preserving NumPy writable-view semantics. The rch trace residual should be treated separately from the diagonal keep.
