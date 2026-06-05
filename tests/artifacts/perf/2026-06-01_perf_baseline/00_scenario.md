# Perf Campaign — Scenario Definition (DEFINE phase)

- **Run ID:** `2026-06-01_perf_baseline`
- **Owner:** BlackThrush (claude-code / opus-4.8) — PROFILING pass (measurement only)
- **Pass type:** measurement-only; hand off to optimizer agents via perf-tagged beads. No optimization in this pass.
- **Git SHA at baseline:** see `fingerprint_local.json` (`git_sha`); campaign branch `main`.

## Scenario

FrankenNumPy's core numeric op families exercised at representative array
sizes — the same fixture set as the 2026-05-05 baseline so progression is
comparable. Driver: `crates/fnp-conformance/benches/criterion_core_ops.rs`
(group `core_ops`), plus `crates/fnp-ufunc/benches/elementwise.rs` and
`transcendental.rs` for the size-scaling sweeps.

| Op | Fixture | Why it matters |
|----|---------|----------------|
| `ufunc_add_broadcast` | 1024×1024 ⊕ 1024 (row broadcast), 1M elem | prior #1 hotspot; broadcasting is on every arithmetic path |
| `reduce_sum_axis1` | 1024×1024 | reduction kernel; control vs add (same element count) |
| `matmul` | 256×256 · 256×256 | O(n³) numeric kernel, no BLAS |
| `sort_quicksort` | 1M f64 | O(n log n) data-movement kernel |
| `fft` | 65 536 complex | Cooley–Tukey transform |
| `astype f64→i32` | 1024×1024 | cast/transfer loop |
| `reshape` | 1024×1024→2048×512 | metadata/stride path (near-free expected) |
| `io_npy save+load` | 512×512 f64 | serialize/parse round-trip |
| elementwise add/sub/mul/div | sweep 100 → 1e6 | per-element vs fixed-overhead scaling law |

## Success metric

Primary: **per-op median latency + element/byte throughput**, ranked to
identify the top hotspots by absolute cost and by throughput deficit
relative to the cheapest op moving the same element count (the
add-vs-reduce gap is the canonical signal).

Secondary: **scaling law** from the 100→1e6 elementwise sweep — slope ≈
per-element cost, intercept ≈ fixed dispatch/alloc overhead. Classifies each
hotspot as throughput-bound (slope) vs overhead-bound (intercept).

Reported per op: criterion median ± MAD, and (for the size sweeps) p-estimates.
Criterion default = 100 samples/measurement (≥ the 20-run floor).

## Build profile (profilable)

`[profile.release-perf]` and `[profile.bench]` pinned to
`debug = "line-tables-only"`, `strip = false`, `opt-level = 3`; bench binaries
built with `RUSTFLAGS="-C force-frame-pointers=yes"` so samply/perf attribution
works. We do NOT profile the size-optimized release binary.

## Methodology notes (honesty gate)

- **Build offload vs run host.** Compilation is offloaded through the rch
  PreToolUse hook (avoids local build-storm contention with the 2 peer agents).
  Bench *execution* runs **locally** on the Threadripper 5975WX, `taskset`-pinned,
  because rch's worker pool is heterogeneous and partly unhealthy (ts1 offline,
  mixed CPUs) — running the statistical baseline across rotating workers would
  break same-host comparability. Same-host discipline > literal "run on rch".
- **Governor = powersave** (not tuned; kernel tuning requires sudo and the skill
  says ask first). Treated as a caveat: absolute numbers may carry P-state
  jitter; the *ranking* and *scaling slopes* are robust to it. Re-baseline under
  `performance` governor is a follow-up if the optimizers need tighter CIs.
- **perf_event_paranoid = 1** → samply user-space sampling is available for
  flamegraph attribution of the top hotspot(s).
- One lever per run; criterion outlier rejection + MAD reported.

## Hand-off contract

Artifacts in this dir: `fingerprint_local.json`, `10_baseline.md` (raw criterion
medians), `20_hotspot_table.md` (ranked + evidence paths), `30_hypothesis_ledger.md`.
Top hotspots filed as `perf`-tagged beads (`br create --type=task -p 2`) with
scenario + baseline numbers + evidence + hypothesized fix in the body, for the
extreme-software-optimization agents to score and act on.
