# 2026-06-20 fnp-linalg kron identity RHS vs NumPy

Bead: `franken_numpy-ixs5y.236`
Agent: `BlackThrush` / `cod-b`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Verdict

Keep and close the existing nonnegative identity-RHS `kron_nxn` specialization.

- Same-worker `hz2` final FrankenNumPy vs NumPy: 2 wins / 0 losses / 0 neutral.
- No source hunk was added in this verification slice.
- Generic dense kron and other structured RHS classes remain separate future
  targets.

## Scored Rows

| Workload | FrankenNumPy ns | NumPy ns | FNP/NumPy | Verdict |
|---|---:|---:|---:|---|
| `kron_nxn/kron_64x64_4x4_eye` | 30,314 | 173,371 | 0.175x | Win |
| `kron_nxn/kron_128x128_4x4_eye_nonnegative_fast_path` | 230,786 | 859,101 | 0.269x | Win |

## Commands

- `RCH_WORKER=hz2 RCH_REQUIRE_REMOTE=1 RCH_DAEMON_WAIT_RESPONSE_TIMEOUT_SECS=240 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg kron_nxn -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `ssh hz2 'cd /data/projects/franken_numpy && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python3 - <<PY ... PY'`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg kron_ -- --nocapture`

## Validation

- Focused kron tests: pass, 4/4.
- `cargo check -p fnp-linalg --all-targets`: pass on the same no-source linalg
  tree during the immediately preceding column-sum verification.
- `cargo clippy -p fnp-linalg --all-targets -- -D warnings`: pass on the same
  no-source linalg tree.
- `cargo build -p fnp-linalg --release`: pass on the same no-source linalg tree.
- `cargo fmt --package fnp-linalg -- --check`: known broad pre-existing drift
  outside this no-source verification slice.

## Negative Evidence

- The current specialization pays only for exact identity RHS with finite
  nonnegative LHS. Do not widen it into negative, signed-zero, NaN, Inf, or
  non-identity RHS domains without new bit-preserving proof.
- A credible next kron lever needs a distinct structure class, such as diagonal
  RHS, sparse block masks, or separable Kronecker chains.
