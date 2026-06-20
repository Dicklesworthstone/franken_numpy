# 2026-06-20 fnp-linalg column-norm strip-mine no-ship

Run identity:
- Parent bead: `franken_numpy-ixs5y`.
- Agent: `BlackThrush` / `cod-a`.
- Crate scope: `fnp-linalg` only.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Workload: direct Rust `matrix_norm_nxn_orders/(one|neg_one)/(256|512|1024)`.

## Verdict

No-ship. The 8-column strip-mined accumulation was behavior-preserving under the
focused bit guard, but it did not meet the measured keep gate:

- `vmi1149989` old FNP -> candidate FNP: 5 wins / 1 loss / 0 neutral.
- `hz1` candidate FNP -> prior `hz1` NumPy comparator: 0 wins / 6 losses / 0 neutral.
- Direct same-host NumPy refresh on `hz1` failed with SSH auth denial.
- Source hunk reverted; only evidence artifacts and ledger updates remain.

## Candidate

The attempted lever unrolled the cache-linear column-sum fill by eight adjacent
columns per row, preserving each column's row-order addition and returning the
same canonical `NaN` on the first NaN-containing strip. This tested the
graveyard/vectorization hypothesis that adjacent-column strip mining would
reduce iterator overhead and improve memory-level parallelism without changing
the public `ord="1"` / `ord="-1"` semantics.

## Same-Worker Rust Delta on `vmi1149989`

| Workload | Baseline FNP ns | Candidate FNP ns | Candidate/Baseline | Verdict |
|---|---:|---:|---:|---|
| `one/256` | 28388 | 23372 | 0.823x | win |
| `neg_one/256` | 26724 | 27721 | 1.037x | loss |
| `one/512` | 113473 | 106512 | 0.939x | win |
| `neg_one/512` | 111496 | 103362 | 0.927x | win |
| `one/1024` | 530381 | 409582 | 0.772x | win |
| `neg_one/1024` | 632365 | 412535 | 0.652x | win |

## NumPy Comparator Evidence

The only fresh paired Rust old/candidate run landed on `vmi1149989`, but direct
NumPy capture on that host failed with SSH auth denial. The non-compilation
`rch exec -- python3` probe ran locally on `thinkstation1`, not remotely, so it
is routing evidence only.

| Workload | Candidate FNP ns (`vmi1149989`) | Local NumPy ns (`thinkstation1`) | Candidate/NumPy | Counted? |
|---|---:|---:|---:|---|
| `one/256` | 23372 | 29345 | 0.796x | routing only |
| `neg_one/256` | 27721 | 26140 | 1.060x | routing only |
| `one/512` | 106512 | 96573 | 1.103x | routing only |
| `neg_one/512` | 103362 | 113425 | 0.911x | routing only |
| `one/1024` | 409582 | 416639 | 0.983x | routing only |
| `neg_one/1024` | 412535 | 359040 | 1.149x | routing only |

The repeat candidate run landed on `hz1` and was slower than the prior direct
`hz1` NumPy column-norm comparator from
`tests/artifacts/perf/2026-06-20_linalg_batch_vs_numpy/numpy_column_hz1.txt`.
This is also not counted as a same-time comparator, but it rejects the idea that
the strip mine is robust across the available RCH lanes.

| Workload | Candidate FNP ns (`hz1`) | Prior NumPy ns (`hz1`) | Candidate/NumPy | Verdict |
|---|---:|---:|---:|---|
| `one/256` | 50646 | 40921 | 1.238x | loss |
| `neg_one/256` | 50689 | 40940 | 1.238x | loss |
| `one/512` | 211885 | 147264 | 1.439x | loss |
| `neg_one/512` | 213556 | 145528 | 1.468x | loss |
| `one/1024` | 836943 | 506356 | 1.653x | loss |
| `neg_one/1024` | 830032 | 503971 | 1.647x | loss |

## Validation

- `rch exec -- cargo test -p fnp-linalg matrix_norm_column_reduction_matches_strided_reference_bits -- --nocapture`: pass, 1 focused test passed on `hz1`.
- `rch exec -- cargo build -p fnp-linalg --release`: pass on `vmi1293453` after reverting the production hunk.
- `rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'matrix_norm_nxn_orders/(one|neg_one)/(256|512|1024)'`: baseline and candidate artifacts captured.
- `ubs $(git diff --name-only --cached)`: exit 0; evidence bundle contains docs/logs only, so UBS reported no recognizable source languages.
- Direct same-host NumPy attempts on `vmi1149989` and `hz1`: blocked by SSH auth denial.

## Decision

The candidate was reverted because it had one same-worker Rust regression, no
valid same-worker NumPy win table, and a repeat RCH-lane run that lost every row
against the available NumPy context. The next credible retry should use a real
portable-SIMD absolute-value lane or a generated microkernel selected by an
empirical per-size gate, and it must capture same-host NumPy ratios before any
production hunk is kept.
