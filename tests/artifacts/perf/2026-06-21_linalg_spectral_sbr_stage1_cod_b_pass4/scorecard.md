# 2026-06-21 linalg spectral SBR stage-1 probe

Agent: `YellowElk` / `cod-b`
Parent bead: `franken_numpy-ixs5y`
Target dir: `/data/projects/.rch-targets/franken_numpy-cod-b`

## Same-worker `ovh-a` ratios

| Probe | FNP ns | NumPy ns | FNP/NumPy | Verdict |
|---|---:|---:|---:|---|
| Current `eigvalsh_nxn/128` | 1,315,452 | 631,765 | 2.082x | loss |
| Current `cond_nxn/128` | 993,887 | 961,374 | 1.034x | neutral |
| Current `eigvalsh_nxn/512` | 68,791,964 | 27,470,726 | 2.504x | loss |
| Existing `sbr_stage1_band_nxn/512` vs NumPy full `eigvalsh/512` | 19,948,921 | 27,470,726 | 0.726x | incomplete primitive |

Counted API scorecard: win/loss/neutral = **0/2/1**.

## Interpretation

The API gap is still `eigvalsh`, not `cond` in this same-worker run. Existing
SBR stage 1 is promising because it costs less than NumPy's complete 512x512
eigensolve, but it does not produce eigenvalues. It should not be wired into
`eigvalsh_nxn` until there is a true band-to-tridiagonal stage or a band-aware
eigenvalue path.

At `n=512`, the measured budget to beat NumPy after stage 1 is approximately:

```text
27,470,726 ns NumPy eigvalsh_512
-19,948,921 ns SBR stage1_512
= 7,521,805 ns for stage2 + tridiagonal eigenvalues
```

## Files

- `fnp_linalg_probe.txt`: RCH `ovh-a` current `eigvalsh_128`, `cond_128`, and
  `sbr_stage1_512`.
- `numpy_ovh_a_probe.txt`: direct `ovh-a` NumPy 2.2.4 comparator with BLAS
  threads pinned to 1.
- `fnp_linalg_ovh_a_eigvalsh512.txt`: direct `ovh-a` Rust `eigvalsh_512`
  comparator using the already synced RCH workspace/target.
- `fnp_linalg_probe_large.txt`: routing-only cross-worker probe on
  `vmi1227854`; not used for same-worker ratios.
- `validation_test_sbr_stage1.txt`: `cargo test -p fnp-linalg sbr_stage1
  --release`, passed.
- `validation_check_fnp_linalg.txt`: `cargo check -p fnp-linalg --all-targets`,
  passed.
- `validation_clippy_fnp_linalg.txt`: `cargo clippy -p fnp-linalg
  --all-targets -- -D warnings`, passed.
- `validation_build_fnp_linalg_release.txt`: `cargo build -p fnp-linalg
  --release`, passed.
