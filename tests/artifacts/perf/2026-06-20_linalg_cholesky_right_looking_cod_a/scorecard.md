# Small-N Cholesky Ordered-Dot Verification

Date: 2026-06-20
Agent: YellowElk / cod-a
Parent bead: franken_numpy-ixs5y
Source under verification: 856c38cb

The artifact directory name reflects the initial right-looking Cholesky
hypothesis. The committed source under verification is narrower: a 4-wide
ordered dot helper routed only for unblocked `cholesky_nxn` n=16..32.

## Rust Parent/Current Gate

Command:

```bash
AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a \
  rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg \
  'cholesky_nxn|batch_cholesky' -- --sample-size 20 --warm-up-time 1 \
  --measurement-time 3 --output-format bencher
```

Parent commit: 586f3459
Current commit: 856c38cb
Comparable worker: vmi1153651

| Row | Parent | Current | Current/Parent | Verdict |
|---|---:|---:|---:|---|
| cholesky_nxn/16 | 2,186 ns | 1,901 ns | 0.870x | owned win |
| cholesky_nxn/32 | 11,091 ns | 9,747 ns | 0.879x | owned win |
| cholesky_nxn/64 | 70,817 ns | 70,754 ns | 0.999x | neutral |
| cholesky_nxn/128 | 319,742 ns | 306,706 ns | 0.959x | noisy non-owned |
| cholesky_nxn/256 | 1,868,544 ns | 2,052,584 ns | 1.098x | noisy non-owned loss |
| cholesky_nxn/512 | 35,042,224 ns | 24,653,195 ns | 0.704x | noisy non-owned |
| cholesky_nxn/768 | 107,262,958 ns | 96,512,158 ns | 0.900x | noisy non-owned |
| batch_cholesky/64x128x128 | 20,565,511 ns | 24,253,715 ns | 1.179x | noisy non-owned loss |
| batch_cholesky/16x256x256 | 22,245,367 ns | 78,519,429 ns | 3.529x | noisy non-owned loss |

Owned Rust rows: 2 wins / 0 losses / 0 neutral.

## Python NumPy Comparator

Build command:

```bash
AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a \
  rch exec -- cargo build -p fnp-python --release --features python-extension
```

Build worker: vmi1152480
Comparator module: current-head `libfnp_python.so` staged as `fnp_python.so`.

| Row | NumPy | FNP | FNP/NumPy | Match | Verdict |
|---|---:|---:|---:|---|---|
| B=4000 d=8 | 1.18 ms | 1.31 ms | 1.11x | true | loss |
| B=2000 d=16 | 1.52 ms | 9.83 ms | 6.46x | true | owned loss |
| B=1000 d=32 | 3.60 ms | 19.63 ms | 5.46x | true | owned loss |
| B=500 d=64 | 6.08 ms | 38.11 ms | 6.27x | true | loss |
| B=200 d=100 | 6.62 ms | 9.65 ms | 1.46x | true | loss |
| B=64 d=200 | 7.64 ms | 12.78 ms | 1.67x | true | loss |
| B=10000 d=4 | 2.06 ms | 1.54 ms | 0.75x | true | win |

Current Python rows vs NumPy: 1 win / 6 losses / 0 neutral.
Owned Python-facing rows vs NumPy: 0 wins / 2 losses / 0 neutral.

## Decision

Keep the already-present narrow Rust micro-win, but do not treat it as a NumPy
performance closeout. The residual is structural: Python stacked Cholesky still
needs batched layout work, packed-panel kernels, or generated fixed-size d=16/32
specializations with same-window NumPy proof.

## Validation

- `rch exec -- cargo test -p fnp-linalg cholesky_ -- --nocapture`: passed on
  vmi1293453. Totals: 21 unit tests, 4 conformance tests, 2 golden tests, 1
  metamorphic test, and 4 solve tests passed.
- `rch exec -- cargo check -p fnp-linalg --all-targets`: passed on hz1.
- `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`:
  passed on hz1.
- `rch exec -- cargo build -p fnp-linalg --release`: passed on vmi1149989.
- `rch exec -- cargo build -p fnp-python --release --features
  python-extension`: passed on vmi1152480 with three pre-existing
  `fnp-python` warnings.
