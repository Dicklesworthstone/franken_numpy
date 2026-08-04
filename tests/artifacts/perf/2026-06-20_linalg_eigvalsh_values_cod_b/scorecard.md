# 2026-06-20 `fnp-linalg` eigvalsh mid-band row-dot reducer

Bead: `franken_numpy-ixs5y.275`
Agent: `YellowElk` / `cod-b`
Worker: `hz1` for Rust Criterion and direct NumPy comparator
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Decision

Keep the gated row-dot serial panel matvec for `192 <= n < 384`.

The direct row-dot form is bit-identical to the former half-symmetric scatter
walk when the dense symmetric work matrix has mirrored entries: each output row
still sums contributions in ascending column order. The full row-dot layout is
SIMD/cache friendlier for the 256-class reducer, but the ungated attempt
regressed smaller rows and badly hurt 512. The final source therefore keeps the
old half-symmetric path below 192 and at 384+.

## Performance

| Workload | Baseline FNP ns | Final FNP ns | NumPy median ns | Final/Baseline | Final/NumPy | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `eigvalsh_nxn/size/64` | 261856 | 270106 | 254157 | 1.032x | 1.063x | neutral/noise, below row-dot gate |
| `eigvalsh_nxn/size/128` | 1995299 | 1896797 | 1280690 | 0.951x | 1.481x | residual loss; below row-dot gate |
| `eigvalsh_nxn/size/256` | 17636268 | 12969460 | 7380748 | 0.735x | 1.757x | keep win vs FNP, residual NumPy loss |
| `eigvalsh_nxn/size/512` | not rebaselined | 59840882 | 49987519 | n/a | 1.197x | guard row; row-dot disabled |

Rejected probes:

| Probe | Workload | Probe FNP ns | Baseline/final comparator | Ratio | Verdict |
|---|---:|---:|---:|---:|---|
| Ungated row-dot | `eigvalsh_nxn/size/64` | 287434 | 261856 baseline | 1.098x | reject regression |
| Ungated row-dot | `eigvalsh_nxn/size/128` | 2103644 | 1995299 baseline | 1.054x | reject regression |
| Ungated row-dot | `eigvalsh_nxn/size/256` | 12580950 | 17636268 baseline | 0.713x | useful signal |
| Row-dot enabled at 512 | `eigvalsh_nxn/size/512` | 88449167 | 59840882 final old-path guard | 1.478x | reject; upper gate required |

## Profile Context

`tridiag_eigvals_qr_perf_report` on `hz1` confirmed the QR chase is already the
optimized scaled-hypot path: 1.30x, 1.31x, and 1.27x faster than the old libm
`hypot` path at n=256/512/768. The remaining end-to-end gap is therefore still
the dense tridiagonal reducer, not the QR micro-loop.

## Validation

- `rch exec -- cargo test -p fnp-linalg tridiag --release`: pass on
  RCH-selected `vmi1153651`; 7 passed, 0 failed, 4 ignored. This covers the
  bit-equivalence test, blocked/unblocked checks, parallel-matvec check, and
  `tridiag_rank2k_fused_update_preserves_spectra_and_golden_sha256`.
- `rch exec -- cargo check -p fnp-linalg --all-targets`: pass on
  RCH-selected `vmi1152480`.
- `rch exec -- cargo build -p fnp-linalg --release`: pass on RCH-selected
  `vmi1152480`.
- `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`: pass
  on `hz1`.
- `git diff --check`: pass.
- `cargo fmt -p fnp-linalg -- --check`: known gap from broad pre-existing
  rustfmt drift in benches/examples and unrelated source regions.
- `ubs` over changed source/doc paths: known gap from broad existing
  `fnp-linalg/src/lib.rs` inventory, not a row-dot-hunk-specific finding.

## Retry Predicate

Do not retry ungated full row-dot matvec. It helps the 256-class dense reducer
but loses at 64/128 and 512. Future work should target the remaining NumPy loss
with a deeper values-only tridiagonal reducer, a true band-stage primitive, or a
size-generated 128-specific reducer that can improve `cond_nxn/128` without
reopening the rejected panel-width, active-window deflation, or sub-1024 Rayon
matvec families.
