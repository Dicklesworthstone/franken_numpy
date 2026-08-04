# 2026-06-21 roll bulk-copy follow-up scorecard

Agent: `YellowElk` / `cod-b`
Bead: `franken_numpy-ixs5y.280`
Target crate: `fnp-python`
Warm target dir: `/data/projects/.rch-targets/franken_numpy-cod-b`

Current-main correction: `origin/main` already contains the roll memcpy landing
in `84f52074` and artifact closeout `24b3d258`. This scorecard adds a reusable
Criterion row and follow-up evidence only; no production source is kept by this
commit.

## Hypothesis

The fresh roll/compress profile showed flattened `np.roll` as a real loss:
FrankenNumPy was 1.36x-1.43x slower than NumPy on serial head-to-head runs even
though the Python surface already had zero-copy roll fast paths. The radical
lever was to replace per-element `Cell` loops in the flattened f64 and generic
byte roll paths with two contiguous copies into a NumPy-owned output array.

Mapped skills and sources:
- `/alien-graveyard`: vectorized execution and cache-local bulk movement.
- `/alien-artifact-coding`: concrete artifact is a two-copy exact relocation
  path, not an approximate algorithm.
- `/extreme-software-optimization`: one lever, same-worker measurement, revert
  unless it beats NumPy.
- `/running-the-gauntlet-on-your-rust-port`: compare directly against NumPy and
  preserve behavior before considering a keep.
- `/profiling-software-performance`: target an already-measured loss class.

## Commands

Baseline and candidates used the same bench filter:

```bash
AGENT_NAME=YellowElk RAYON_NUM_THREADS=1 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b \
rch exec -- cargo bench -p fnp-python --bench criterion_python_surface \
roll_f64_axis_none -- --sample-size 10 --measurement-time 3 \
--warm-up-time 1 --output-format bencher
```

Note: this pinned Cargo rejects `cargo bench --release`; Criterion bench profile
is already optimized, so the counted runs use `cargo bench`.

## Results

| Run | Worker | FNP | NumPy | FNP/NumPy | Verdict |
|---|---|---:|---:|---:|---|
| baseline current code | `ovh-a` | 2,067,710 ns | 1,419,763 ns | 1.456x | loss |
| candidate 1: bulk `copy_from_slice` | `ovh-a` | 1,409,862 ns | 1,388,494 ns | 1.015x | neutral |
| current `84f52074`: f64 `empty_like` plus byte bulk-copy | `ovh-a` | 1,516,753 ns | 1,417,947 ns | 1.070x | loss |

Candidate 1 improved FrankenNumPy's own flattened-roll time by about 1.47x
relative to the baseline, but the decision rule is direct NumPy dominance. The
remaining 1.015x ratio is neutral/slight loss. The current landing recheck is
1.070x on `ovh-a`, which conflicts with the landed commit's separate `.probe`
parity evidence and should be treated as follow-up warning evidence, not a
single-run revert trigger.

## Decision

No production code in this closeout. Keep only the benchmark row plus this
evidence so the lane is reproducible. Do not revert `84f52074` from this single
ovh-a recheck; require repeated same-worker proof before changing the landed roll
path again.

Win/loss/neutral for candidate trials: **0 / 1 / 1**.

Retry predicate: do not repeat the Cell-loop-to-bulk-copy family or the
`empty_like` output variant as a standalone roll optimization. A future attempt
must either prove the landed code across repeated same-worker runs or use a
different allocation/ownership route that beats NumPy in the same head-to-head
run.

## Validation

```bash
AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b \
rch exec -- cargo build -p fnp-python --bench criterion_python_surface --release
```

Result: green on `hz2`; existing `fnp-python` warnings only
(`StackHelperKind::{Depth, Column}`, `extract_mask_operand`, and
`count_valid_elements`).

```bash
AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b \
rch exec -- cargo test -p fnp-python --test conformance_array_transform roll -- --nocapture
```

Result: green on `hz2`, 10 passed / 0 failed.

## Caveats

An early candidate run on `vmi1264463` was interrupted after a long silent wait
and is not counted. The counted baseline and candidate measurements above are
all from `ovh-a`.
