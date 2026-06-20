# 2026-06-20 fnp-python einsum trace scalar-builder gauntlet

Agent: YellowElk / cod-b  
Parent bead: `franken_numpy-ixs5y`  
Crate: `fnp-python`  
Target dir: `/data/projects/.rch-targets/franken_numpy-cod-b`  
Decision: keep

## Target Gap

The previous same-worker `vmi1227854` final row for the cached-buffer einsum diagonal pass left `fnp_einsum_trace_f64_4000` slower than NumPy:

| Row | Prior FNP | Prior NumPy | Prior FNP/NumPy | State |
| --- | ---: | ---: | ---: | --- |
| `fnp_einsum_trace_f64_4000` | 5.9900 us | 5.2275 us | 1.146x | loss |
| `fnp_einsum_diag_f64_4000` | 805.39 ns | 889.51 ns | 0.905x | win |

## Lever

Replace the f64 scalar-return builder's 0-D ndarray materialization with a cached `numpy.float64` type lookup via `PyOnceLock`, and route the direct f64 `trace` fast path through that same helper. This keeps behavior in NumPy scalar space while removing the temporary ndarray path from scalar trace results.

## Measured Result

Command:

```bash
AGENT_NAME=YellowElk RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b \
  rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_einsum_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher
```

RCH worker: `vmi1227854`  
Artifact: `candidate_rch_vmi1227854.txt`

| Row | FNP | NumPy | FNP/NumPy | Outcome |
| --- | ---: | ---: | ---: | --- |
| `einsum_trace_f64_4000` | 4,838 ns | 6,242 ns | 0.775x | win |
| `einsum_diag_f64_4000` | 860 ns | 939 ns | 0.916x | win |
| `einsum_reduce_all_f64_1000` | 95,143 ns | 94,139 ns | 1.011x | neutral/loss, non-target |
| `einsum_reduce_rows_f64_1000` | 90,580 ns | 93,613 ns | 0.968x | win |
| `einsum_reduce_cols_f64_1000` | 109,933 ns | 198,288 ns | 0.554x | win |

Target-decision rows: 2 wins / 0 losses / 0 neutral.  
Full observed boundary sweep: 4 wins / 1 loss-or-neutral / 0 neutral if `reduce_all` is counted strictly.

Trace moved from 1.146x slower than NumPy to 0.775x of NumPy on the same RCH worker. FNP trace also moved from 5.990 us to 4.838 us, a 0.808x old/new ratio. The diagonal support row remains faster than NumPy; its FNP absolute time is 1.068x slower than the prior artifact, so it is recorded as preserved-win noise rather than a fresh diagonal improvement.

## Validation

```bash
AGENT_NAME=YellowElk RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b \
  rch exec -- cargo test -p fnp-python --test conformance_einsum
```

Result: pass, 28/28 tests. Worker selected by RCH: `vmi1153651`.

```bash
AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b \
  rch exec -- cargo check -p fnp-python --lib --bench criterion_python_surface
```

Result: pass with pre-existing warnings in `fnp-python`.

```bash
AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b \
  rch exec -- cargo build -p fnp-python --release
```

Result: pass with the same pre-existing warnings.

```bash
AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b \
  rch exec -- cargo clippy -p fnp-python --lib --bench criterion_python_surface -- -D warnings
```

Result: blocked by broad pre-existing `fnp-python` lint inventory, including dead code, range-loop, too-many-arguments, range contains, NaN equality idioms, collapsible-if, redundant guard/closure, manual checked ops, and `chars().next()` comparisons. The clippy log does not mention `build_f64_scalar` or `NUMPY_FLOAT64_TYPE`.

```bash
AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b \
  cargo fmt -p fnp-python -- --check
```

Result: blocked by broad pre-existing rustfmt drift in `fnp-python`; the fmt diff log does not mention `build_f64_scalar` or `NUMPY_FLOAT64_TYPE`.

```bash
ubs crates/fnp-python/src/lib.rs
```

Result: scanner did not finish within the allotted interactive window for the single large `fnp-python` source file and was interrupted after more than three minutes. The artifact contains the partial scanner startup log and no emitted finding.

## Retry Predicate

Do not retry the scalar-builder lever unless it is paired with a stronger scalar-return contract proof or a different scalar constructor mechanism. Next performance work should target the remaining observed `reduce_all` near-loss, a deeper non-boundary einsum kernel, or linalg rows that still lose after the latest column-norm no-ship.
