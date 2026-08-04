# fnp-ufunc percentile_method medium gate check

Agent: `YellowElk` / `cod-b`
Parent bead: `franken_numpy-ixs5y`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Decision

No production cutoff change. The current `percentile_method(q, None, Linear)`
path is already faster than NumPy on the same OVH host for the checked medium
sizes, so the suspected `1 << 17` gate is not a current loss.

## Evidence

| Row | Current FNP median ms | NumPy median ms | FNP/NumPy | Verdict |
|---|---:|---:|---:|---|
| n=131072 | 0.494989 | 0.883377 | 0.560x | win |
| n=262144 | 0.380504 | 1.707563 | 0.223x | win |
| n=524288 | 0.672502 | 3.923541 | 0.171x | win |

Scorecard vs NumPy: **3 wins / 0 losses / 0 neutral**.

## Final Validation

- `cargo check -p fnp-ufunc --all-targets` passed on `hz1` after the final
  source state.
- `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed on `hz1`.
  The first clippy run exposed an existing current-tree `trapezoid` last-axis
  range-loop lint; the final tree includes the iterator-equivalent rewrite and
  the release `trapezoid` filter passed.
- `cargo test -p fnp-ufunc percentile --release -- --nocapture` passed
  33 tests with 5 ignored perf probes on `hz1`.
- `cargo test -p fnp-ufunc trapezoid --release -- --nocapture` passed
  13 tests on `hz1`.
- `cargo fmt --check -p fnp-ufunc` still reports broad pre-existing formatting
  drift outside this evidence slice; no crate-wide formatting normalization was
  performed.

## Commands

- `AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-ufunc percentile_method_medium_gate_report --release -- --ignored --nocapture`
- `ssh fmd 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -'`
- `AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo check -p fnp-ufunc --all-targets`
- `AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-ufunc percentile --release -- --nocapture`
- `AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-ufunc trapezoid --release -- --nocapture`
- `cargo fmt --check -p fnp-ufunc`

## Notes

- RCH selected `ovh-a`; the direct NumPy comparator used the configured `fmd`
  alias for the same host.
- A trial `PERCENTILE_M_GLOBAL_PARALLEL_MIN = 1 << 19` run was not counted
  because RCH moved it to `hz1`; the source cutoff was restored to `1 << 17`.
