# 2026-06-20 fnp-linalg batched row-sum norm vs NumPy

Bead: `franken_numpy-ixs5y.239`
Agent: `BlackThrush` / `cod-b`
Crate: `fnp-linalg`
Worker: `hz1` for FrankenNumPy Criterion; explicit NumPy comparator over `ssh hz1` on host `hetzner1`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Decision

Keep and close the already-landed direct batched row-sum lane-fill path for
`batch_matrix_norm(..., ord="inf")` and `ord="-inf"`.

This verification did not add a source hunk. It measured the existing direct
row-sum lane path against NumPy on the same worker host, recorded the result,
and kept the path because all measured target rows were decisive wins.

## Head-to-Head

| Workload | FrankenNumPy median | NumPy median | FNP/NumPy | NumPy/FNP | Verdict |
|---|---:|---:|---:|---:|---|
| `inf_4096x8x8` | 86,647 ns | 1,021,869 ns | 0.085x | 11.79x faster | Win |
| `inf_1024x32x32` | 180,783 ns | 1,347,235 ns | 0.134x | 7.45x faster | Win |
| `-inf_4096x8x8` | 84,239 ns | 1,044,963 ns | 0.081x | 12.40x faster | Win |
| `-inf_1024x32x32` | 181,321 ns | 1,320,966 ns | 0.137x | 7.29x faster | Win |

Scorecard: win/loss/neutral = 4/0/0.

## Commands

FrankenNumPy Criterion:

```bash
RCH_WORKER=hz1 RCH_WORKERS=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg batch_matrix_norm_row_sum -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

NumPy comparator:

```bash
ssh hz1 'python3 -c ...'
```

The comparator reported:

```text
host hetzner1
python 3.14.4
numpy 2.3.5
```

An earlier `rch exec -- python3 -c ...` attempt printed the RCH
non-compilation warning and did not identify a selected worker. That row is
routing evidence only and is not counted in the keep decision.

## Validation

Passed:

```bash
RCH_WORKER=hz1 RCH_WORKERS=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg batch_matrix_norm_row_sum_direct_lane_fill_matches_per_lane_reference_bits -- --nocapture
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo check -p fnp-linalg --all-targets
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo build -p fnp-linalg --release
```

`cargo fmt -p fnp-linalg -- --check` failed because of broad pre-existing
format drift in untouched `fnp-linalg` benches, examples, and source regions.
No source hunk was added in this verification slice, so the drift is recorded
but not normalized here.

## Isomorphism Proof

- Ordering preserved: each lane computes row sums in the same row-major row and
  column order as the per-lane `matrix_norm_nxn` reference.
- Floating-point schedule preserved inside each row sum.
- `ord="inf"` and `ord="-inf"` selection semantics are preserved for NaN and
  signed zero through the focused bit-preservation test.
- RNG, tie-breaking, aliasing, and mutation surfaces are not involved.

## Retry Predicate

Do not retry this bead unless future same-worker evidence regresses the current
row-sum direct lane path, or a materially different row-sum shape exposes a
fresh NumPy loss.
