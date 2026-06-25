# Performance Release Readiness Scorecard

Scope: rolling gauntlet verification of measured FrankenNumPy performance slices
against original NumPy.

## 2026-06-24 CreamEagle fnp-python nanmean axis=0 streaming fused pass keep (53rd win)

| Area | Score | Verdict |
|---|---:|---|
| `nanmean(axis=0)` 4096x512 vs NumPy | 10/10 | `0.424x` NumPy time (2.36x faster) |
| `nanmean(axis=0)` 50000x64 vs NumPy | 10/10 | `0.342x` NumPy time (2.92x faster) |
| Behavior gate | 9/10 | Native only for f64 C-contiguous axis=0 (>=2-D); non-axis-0 single non-trailing axes and non-f64 defer (all-NaN column handled inline as NaN + warning) |
| Conformance gate | 10/10 | `conformance_nan_funcs` 37/37 (new nanmean axis=0 test, incl all-NaN column) |
| Tool hygiene | 7/10 | Per-crate build+bench+test clean via RCH; pre-existing crate warnings unchanged |

Evidence:
- Agent `CreamEagle`; lever: `try_zerocopy_f64_nanmean_axis0` single fused streaming pass
  (per-column NaN->0 sum + non-NaN count; numpy reduces axis=0 sequentially -> bit-exact).
- numpy materializes a NaN->0 copy + mask + two reduces (~19x slower than plain mean(0));
  this streams once. Benched input all-finite, so it understates the gap (numpy on
  10%-NaN data ~6.8 ms vs ~3.3 ms here).
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc
  rch exec -- cargo bench -p fnp-python --profile release --bench
  criterion_python_surface var_axis0 -- --sample-size 20 --warm-up-time 1
  --measurement-time 3 --output-format bencher`.
- Artifact directory: `tests/artifacts/perf/2026-06-24_nanmean_axis0_cod_b/`.

Decision:
- Release-ready keep for the f64 C-contiguous axis=0 nanmean rows.
- Non-axis-0 single non-trailing axes remain on NumPy.

## 2026-06-24 CreamEagle fnp-python nanvar/nanstd axis=0 streaming keep (52nd win)

| Area | Score | Verdict |
|---|---:|---|
| `nanvar(axis=0)` 4096x512 vs NumPy | 10/10 | `0.462x` NumPy time (2.16x faster) |
| `nanstd(axis=0)` 4096x512 vs NumPy | 10/10 | `0.456x` NumPy time (2.19x faster) |
| `nanvar(axis=0)` 50000x64 vs NumPy | 10/10 | `0.382x` NumPy time (2.62x faster) |
| `nanstd(axis=0)` 50000x64 vs NumPy | 10/10 | `0.382x` NumPy time (2.68x faster) |
| Behavior gate | 9/10 | Native only for f64 C-contiguous axis=0 (>=2-D), ddof>=0; any count<=ddof column (all-NaN) and everything else defer to NumPy |
| Conformance gate | 10/10 | `conformance_nan_funcs` 36/36 (new nanvar+nanstd axis=0 test, incl all-NaN-column defer) |
| Tool hygiene | 7/10 | Per-crate build+bench+test clean via RCH; pre-existing crate warnings unchanged |

Evidence:
- Agent `CreamEagle`; lever: `try_zerocopy_f64_nanvar_axis0` serial streaming (NaN-skip +
  per-column count; numpy reduces axis=0 sequentially, verified bit-exact).
- numpy materializes NaN->0 copy + mask + count + a-mean + squared temps; this streams the
  array twice with no temporary. Benched input is all-finite so it understates the gap
  (numpy.nanvar(axis=0) on 10%-NaN data ~14 ms vs ~7 ms here).
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc
  rch exec -- cargo bench -p fnp-python --profile release --bench
  criterion_python_surface var_axis0 -- --sample-size 20 --warm-up-time 1
  --measurement-time 3 --output-format bencher`.
- Artifact directory: `tests/artifacts/perf/2026-06-24_nanvar_nanstd_axis0_cod_b/`.

Decision:
- Release-ready keep for the f64 C-contiguous axis=0 nanvar/nanstd rows.
- All-NaN/under-ddof columns and non-axis-0 single non-trailing axes remain on NumPy.

## 2026-06-24 CreamEagle fnp-python var/std axis=0 streaming two-pass keep (51st win)

| Area | Score | Verdict |
|---|---:|---|
| `var(axis=0)` 4096x512 vs NumPy | 10/10 | `0.244x` NumPy time (4.10x faster) |
| `std(axis=0)` 4096x512 vs NumPy | 10/10 | `0.230x` NumPy time (4.35x faster) |
| `var(axis=0)` 50000x64 vs NumPy | 10/10 | `0.268x` NumPy time (3.73x faster) |
| `std(axis=0)` 50000x64 vs NumPy | 10/10 | `0.243x` NumPy time (4.12x faster) |
| Behavior gate | 9/10 | Native only for f64 C-contiguous axis normalizing to 0 (>=2-D), native ddof, no out/dtype; M<=ddof and everything else defer to NumPy |
| Conformance gate | 10/10 | `conformance_var` 17/17 (new var+std axis=0 test) + `conformance_std` 15/15 |
| Tool hygiene | 7/10 | Per-crate build+bench+test clean via RCH; pre-existing crate warnings unchanged |

Evidence:
- Agent `CreamEagle`; lever: `try_zerocopy_f64_var_axis0` serial cache-friendly streaming
  two-pass (numpy reduces axis=0 SEQUENTIALLY, verified, so bit-exact).
- numpy materializes a-mean + squared whole-array temps + 2 sequential reduces; this
  streams the array twice with no temporary. Column-block parallelism rejected (cache-hostile
  on row-major: ~2.6x slower + noisy) — serial streaming is bandwidth-optimal.
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc
  rch exec -- cargo bench -p fnp-python --profile release --bench
  criterion_python_surface var_axis0 -- --sample-size 20 --warm-up-time 1
  --measurement-time 3 --output-format bencher`.
- Artifact directory: `tests/artifacts/perf/2026-06-24_var_std_axis0_cod_b/`.

Decision:
- Release-ready keep for the f64 C-contiguous axis=0 var/std rows.
- M<=ddof, non-axis-0 single non-trailing axes, and non-f64 remain on NumPy.

## 2026-06-24 CreamEagle fnp-python gradient non-last-axis row-combine keep (50th win, MODEST)

| Area | Score | Verdict |
|---|---:|---|
| `gradient(axis=0)` 4096x1024 vs NumPy | 7/10 | `0.722x` NumPy time (1.39x faster) — modest, bandwidth-bound |
| `gradient(axis=0)` 1024x4096 vs NumPy | 7/10 | `0.753x` NumPy time (1.33x faster) — modest, bandwidth-bound |
| Behavior gate | 9/10 | Native only for f64 C-contiguous non-last single int axis, uniform scalar spacing, edge_order=1; edge_order=2 / coord spacing / axis=None-on-ND / non-f64 defer |
| Conformance gate | 10/10 | `conformance_gradient` 23/23 (new strided-axis bit-exact test) + `conformance_diff_gradient` 12/12 |
| Tool hygiene | 7/10 | Per-crate build+bench+test clean via RCH; pre-existing crate warnings unchanged |

Evidence:
- Agent `CreamEagle`; lever: `try_zerocopy_f64_gradient_strided_axis` (per-output-row
  vectorized combine of two input rows; parallel across outer*n rows; direct buffer write).
- numpy.gradient(axis=0) materializes whole-array slice temporaries; the op is
  memory-bandwidth-bound (~96 MB traffic for 4M f64) so temp-avoidance gives ~1.4x — near
  the achievable ceiling, honestly recorded as modest (not the family's 3-11x).
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc
  rch exec -- cargo bench -p fnp-python --profile release --bench
  criterion_python_surface gradient_axis -- --sample-size 10 --warm-up-time 1
  --measurement-time 3 --output-format bencher`.
- Artifact directory: `tests/artifacts/perf/2026-06-24_gradient_strided_axis_cod_b/`.

Decision:
- Release-ready keep for the non-last-axis f64 uniform-spacing edge_order=1 gradient.
- edge_order=2 / coordinate spacing / axis=None-on-ND list return remain on NumPy.

## 2026-06-24 CreamEagle fnp-python nanvar/nanstd multi-axis trailing fold keep (49th win)

| Area | Score | Verdict |
|---|---:|---|
| `nanvar(axis=(-2,-1))` 4096x16x16 vs NumPy | 10/10 | `0.088x` NumPy time (11.37x faster) |
| `nanstd(axis=(-2,-1))` 4096x16x16 vs NumPy | 10/10 | `0.194x` NumPy time (5.17x faster) |
| `nanvar(axis=(-2,-1))` 2048x32x32 vs NumPy | 10/10 | `0.111x` NumPy time (9.00x faster) |
| `nanstd(axis=(-2,-1))` 2048x32x32 vs NumPy | 10/10 | `0.120x` NumPy time (8.35x faster) |
| Behavior gate | 9/10 | Native only for f64 C-contiguous axis tuple == trailing `[ndim-k..ndim)`, ddof>=0; non-trailing/duplicate axes + all-NaN/under-ddof blocks defer to NumPy |
| Conformance gate | 10/10 | `conformance_nan_funcs` 35/35 (new nanvar+nanstd multi-axis test, incl all-NaN block defer) |
| Tool hygiene | 7/10 | Per-crate build+bench+test clean via RCH; pre-existing crate warnings unchanged |

Evidence:
- Agent `CreamEagle`; lever: generalized `try_zerocopy_f64_nanvar_axis` to accept a
  trailing-axis tuple (per-block "lane" = product of trailing dims; symmetric so sorted).
- numpy materializes isnan mask + where temp + squared temp + multi-axis reduce
  (single-threaded, ~7-14 ms); the multi-axis reduce over a contiguous trailing block
  is bit-identical to a flat per-block pairwise nansum/count + sqr-dev fold.
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc
  rch exec -- cargo bench -p fnp-python --profile release --bench
  criterion_python_surface var_multiaxis -- --sample-size 10 --warm-up-time 1
  --measurement-time 3 --output-format bencher`.
- Artifact directory: `tests/artifacts/perf/2026-06-24_nanvar_nanstd_multiaxis_cod_b/`.

Decision:
- Release-ready keep for the contiguous trailing-axes f64 nanvar/nanstd rows.
- Non-trailing/strided axes and all-NaN/under-ddof blocks remain on NumPy.

## 2026-06-24 CreamEagle fnp-python var/std multi-axis trailing two-pass keep (48th win)

| Area | Score | Verdict |
|---|---:|---|
| `var(axis=(-2,-1))` 4096x16x16 vs NumPy | 10/10 | `0.342x` NumPy time (2.93x faster) |
| `std(axis=(-2,-1))` 4096x16x16 vs NumPy | 10/10 | `0.360x` NumPy time (2.78x faster) |
| `var(axis=(-2,-1))` 2048x32x32 vs NumPy | 10/10 | `0.120x` NumPy time (8.30x faster) |
| `std(axis=(-2,-1))` 2048x32x32 vs NumPy | 10/10 | `0.100x` NumPy time (9.96x faster) |
| Behavior gate | 9/10 | Native only for f64 C-contiguous axis tuple == trailing `[ndim-k..ndim)`, native ddof, no out/dtype; non-trailing/duplicate axes, non-finite blocks defer to NumPy |
| Conformance gate | 10/10 | `conformance_var` 16/16 (new var+std multi-axis test) + `conformance_std` 15/15 |
| Tool hygiene | 7/10 | Per-crate build+bench+test clean via RCH; pre-existing crate warnings unchanged |

Evidence:
- Agent `CreamEagle`; lever: generalized `try_zerocopy_f64_var_axis` to accept a
  trailing-axis tuple (per-block "lane" = product of trailing dims; symmetric so sorted).
- numpy materializes mean-broadcast + squared temp + multi-axis reduce (single-threaded);
  the multi-axis reduce over a contiguous trailing block is bit-identical to a flat
  per-block two-pass.
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc
  rch exec -- cargo bench -p fnp-python --profile release --bench
  criterion_python_surface var_multiaxis -- --sample-size 10 --warm-up-time 1
  --measurement-time 3 --output-format bencher`.
- Artifact directory: `tests/artifacts/perf/2026-06-24_std_var_multiaxis_cod_b/`.

Decision:
- Release-ready keep for the contiguous trailing-axes f64 var/std rows.
- Non-trailing/strided axes and non-finite blocks remain on NumPy.

## 2026-06-24 CreamEagle fnp-python linalg.norm induced matrix p-norm (ord ±1/±inf) keep (47th win)

| Area | Score | Verdict |
|---|---:|---|
| `linalg.norm(ord=inf, axis=(-2,-1))` 4096x16x16 vs NumPy | 10/10 | `0.214x` NumPy time (4.66x faster) |
| `linalg.norm(ord=inf, axis=(-2,-1))` 2048x32x32 vs NumPy | 10/10 | `0.137x` NumPy time (7.32x faster) |
| `linalg.norm(ord=1, axis=(-2,-1))` 4096x16x16 vs NumPy | 10/10 | `0.271x` NumPy time (3.69x faster) |
| `linalg.norm(ord=1, axis=(-2,-1))` 2048x32x32 vs NumPy | 10/10 | `0.379x` NumPy time (2.64x faster) |
| Behavior gate | 9/10 | Native only for f64 C-contiguous ord {1,-1,inf,-inf} over exact trailing-order 2-tuple axis; reversed/non-trailing axes + other dtypes defer to NumPy |
| Conformance gate | 10/10 | `conformance_linalg_basic` 61/61 (new 13-case bit-exact induced-matrix-norm parity test) |
| Tool hygiene | 7/10 | Per-crate build+bench+test clean via RCH; pre-existing crate warnings unchanged |

Evidence:
- Agent `CreamEagle`; lever: `try_zerocopy_f64_matrix_norm_lastaxes`
  (`MatrixNormKind` ±inf=row abs-sum, ±1=col abs-sum; per-row contiguous /
  per-col gathered `pairwise_abs_f64` + NaN-prop max/min), ord 1/-1/±inf + exact
  trailing 2-tuple-axis dispatch in `norm()`.
- numpy materializes abs(x) + a per-row/col add.reduce + max/min (3 single-threaded
  passes); each row/col abs-sum is bit-identical to a contiguous pairwise reduce.
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc
  rch exec -- cargo bench -p fnp-python --profile release --bench
  criterion_python_surface norm_frobenius -- --sample-size 10 --warm-up-time 1
  --measurement-time 3 --output-format bencher`.
- Artifact directory: `tests/artifacts/perf/2026-06-24_linalg_norm_matrix_induced_cod_b/`.

Decision:
- Release-ready keep for the contiguous trailing-2-axis f64 induced matrix p-norm rows.
- ord=0/general p and reversed/non-trailing axes remain on NumPy.

## 2026-06-24 CreamEagle fnp-python linalg.norm batched Frobenius (trailing 2-axis) keep (46th win)

| Area | Score | Verdict |
|---|---:|---|
| `linalg.norm(axis=(-2,-1))` 4096x16x16 vs NumPy | 10/10 | `0.285x` NumPy time (3.51x faster) |
| `linalg.norm(axis=(-2,-1))` 2048x32x32 vs NumPy | 10/10 | `0.129x` NumPy time (7.76x faster) |
| Behavior gate | 9/10 | Native only for f64 C-contiguous ord None/'fro'/'f' over trailing 2-tuple axis; axis=None (BLAS dot) and non-trailing axes defer to NumPy |
| Conformance gate | 10/10 | `conformance_linalg_basic` 60/60 (new bit-exact Frobenius trailing-axes parity test) |
| Tool hygiene | 7/10 | Per-crate build+bench+test clean via RCH; pre-existing crate warnings unchanged |

Evidence:
- Agent `CreamEagle`; lever: `try_zerocopy_f64_frobenius_lastaxes` (per-M*N-block
  `pairwise_sq_f64` + sqrt, parallel across blocks), ord None/'fro'/'f' + 2-tuple
  trailing-axis dispatch in `norm()`.
- numpy `sqrt(add.reduce((x*x).real, axis=(row,col)))` materializes the squared temp
  + single-threaded 2-axis reduce; numpy's 2-axis reduce is bit-identical to a flat
  pairwise over the contiguous M*N block (maxulp 0.0, verified).
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc
  rch exec -- cargo bench -p fnp-python --profile release --bench
  criterion_python_surface norm_frobenius -- --sample-size 10 --warm-up-time 1
  --measurement-time 3 --output-format bencher`.
- Artifact directory: `tests/artifacts/perf/2026-06-24_linalg_norm_frobenius_cod_b/`.

Decision:
- Release-ready keep for the contiguous trailing-2-axis f64 Frobenius rows.
- Matrix ord=1/inf (column/row abs-sum) and non-trailing axes remain on NumPy.

## 2026-06-24 CreamEagle fnp-python linalg.norm last-axis +-inf (ord=±inf) keep (45th win)

| Area | Score | Verdict |
|---|---:|---|
| `linalg.norm(ord=inf, axis=-1)` 4096x512 vs NumPy | 10/10 | `0.107x` NumPy time (9.37x faster) |
| `linalg.norm(ord=inf, axis=-1)` 8192x1024 vs NumPy | 10/10 | `0.145x` NumPy time (6.89x faster) |
| Behavior gate | 9/10 | Native only for f64 C-contiguous last-axis `ord in {+inf,-inf}`; other orders/axes/dtypes defer to NumPy |
| Conformance gate | 10/10 | `conformance_linalg_basic` 59/59 (new bit-exact +-inf axis parity test) |
| Tool hygiene | 7/10 | Per-crate build+bench+test clean via RCH; pre-existing crate warnings unchanged |

Evidence:
- Agent `CreamEagle`; lever: `lane_extreme_abs_f64` (NaN-propagating max/min of |x|)
  + `VectorNormKind::{MaxAbs,MinAbs}`, `ord=±inf` dispatch in `norm()`.
- Third member of the last-axis vector-norm fold family (L2 `6355309e`, L1 `657a1137`).
  NumPy `abs(x).max/min(axis)` materializes abs temp + a separate reduce (two passes).
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc
  rch exec -- cargo bench -p fnp-python --profile release --bench
  criterion_python_surface norm_axis -- --sample-size 10 --warm-up-time 1
  --measurement-time 3 --output-format bencher`.
- Artifact directory: `tests/artifacts/perf/2026-06-24_linalg_norm_inf_axis_cod_b/`.

Decision:
- Release-ready keep for the exact contiguous last-axis f64 `ord=±inf` rows.
- `ord=0` and general `ord=p` non-last axes remain on the NumPy fallback.

## 2026-06-24 CreamEagle fnp-python linalg.norm last-axis L1 (ord=1) keep (44th win)

| Area | Score | Verdict |
|---|---:|---|
| `linalg.norm(ord=1, axis=-1)` 4096x512 vs NumPy | 10/10 | `0.243x` NumPy time (4.11x faster) |
| `linalg.norm(ord=1, axis=-1)` 8192x1024 vs NumPy | 10/10 | `0.081x` NumPy time (12.38x faster) |
| Behavior gate | 9/10 | Native only for f64 C-contiguous last-axis `ord in {1,1.0}`; all other orders/axes/dtypes defer to NumPy |
| Conformance gate | 10/10 | `conformance_linalg_basic` 58/58 (new bit-exact L1 axis parity test) |
| Tool hygiene | 7/10 | Per-crate build+bench+test clean via RCH; pre-existing crate warnings unchanged |

Evidence:
- Agent `CreamEagle`; source lever: `pairwise_abs_f64` + `VectorNormKind::L1`
  threaded into `try_zerocopy_f64_vector_norm_axis`, `ord=1` dispatch in `norm()`.
- Extends commit `6355309e` (which covered `ord in {None,2}`) to the still-delegated
  `ord=1` vector L1 norm; NumPy materializes `abs(x)` then `add.reduce(.., axis)`.
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc
  rch exec -- cargo bench -p fnp-python --profile release --bench
  criterion_python_surface norm_axis -- --sample-size 10 --warm-up-time 1
  --measurement-time 3 --output-format bencher`.
- Artifact directory: `tests/artifacts/perf/2026-06-24_linalg_norm_l1_axis_cod_b/`.

Decision:
- Release-ready keep for the exact contiguous last-axis f64 `ord=1` row.
- Do not broaden to `ord=inf/-inf/0/p` or non-last axes without a separate
  proof/benchmark pass; those remain on the NumPy fallback.

## 2026-06-24 CreamEagle fnp-python linalg.norm last-axis native keep (43rd win)

| Area | Score | Verdict |
|---|---:|---|
| `linalg.norm(axis=-1)` 4096x512 vs NumPy | 10/10 | `0.136x` NumPy time (7.35x faster) |
| `linalg.norm(axis=-1)` 8192x1024 vs NumPy | 10/10 | `0.0585x` NumPy time (17.09x faster) |
| Behavior gate | 9/10 | Native only for C-contiguous f64 last-axis vector 2-norm; unsupported orders/axes defer to NumPy |
| Conformance gate | 10/10 | Existing norm filter, dedicated axis-norm parity test, and per-crate all-targets check passed through RCH |
| Tool hygiene | 8/10 | Existing default warnings remain outside this lever |

Evidence:
- Agent `CreamEagle`; source lever: `try_zerocopy_f64_vector_norm_axis` wired
  into `np.linalg.norm` for `ord=None`/`2` and single last-axis reductions.
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b
  rch exec -- cargo bench -p fnp-python --profile release --bench
  criterion_python_surface norm_axis -- --sample-size 10 --warm-up-time 1
  --measurement-time 3 --output-format bencher`.
- Artifact directory:
  `tests/artifacts/perf/2026-06-24_norm_axis_cod_b/`.

Decision:
- Release-ready keep for the exact contiguous last-axis f64 vector 2-norm row.
- Do not broaden to matrix norms, tuple axes, non-f64, or non-contiguous inputs
  without a separate proof/benchmark pass.

## 2026-06-24 CreamEagle fnp-python std/var last-axis native two-pass keep (42nd win)

| Area | Score | Verdict |
|---|---:|---|
| `var(axis=-1)` 4096x512 vs NumPy | 10/10 | `0.114x` NumPy time (8.77x faster) |
| `std(axis=-1)` 4096x512 vs NumPy | 10/10 | `0.117x` NumPy time (8.53x faster) |
| `var(axis=-1)` 8192x1024 vs NumPy | 10/10 | `0.082x` NumPy time (12.20x faster) |
| `std(axis=-1)` 8192x1024 vs NumPy | 10/10 | `0.084x` NumPy time (11.93x faster) |
| Behavior gate | 9/10 | Native only for finite C-contiguous f64 last-axis rows; unsupported/special cases defer to NumPy |
| Conformance gate | 10/10 | `conformance_std` 15/15 and `conformance_var` 15/15 passed through RCH |
| Tool hygiene | 7/10 | Per-crate check passed; fmt/clippy still expose pre-existing crate warnings/drift outside this lever |

Evidence:
- Agent `CreamEagle`; source lever: `try_zerocopy_f64_var_axis` wired into
  `py_std`/`var` after the existing flat native path.
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b
  rch exec -- cargo bench -p fnp-python --profile release --bench
  criterion_python_surface std_var_axis -- --sample-size 10 --warm-up-time 1
  --measurement-time 3 --output-format bencher`.
- Artifact directory:
  `tests/artifacts/perf/2026-06-24_std_var_axis_cod_b/`.

Decision:
- Release-ready keep for the exact contiguous last-axis f64 `std`/`var` row.
- Do not broaden to non-last axes or non-finite lanes without a separate
  proof/benchmark pass; those remain on the NumPy fallback.

## 2026-06-24 BlackThrush fnp-python std/var flat native two-pass keep (41st win)

| Area | Score | Verdict |
|---|---:|---|
| `var`/`std` N=8M vs NumPy | 10/10 | No-alloc two-pass pairwise fold is `0.17x` NumPy time (5.8x faster) |
| `var`/`std` N=16M vs NumPy | 10/10 | `0.18x` NumPy time; numpy materializes two whole-array temps |
| `var`/`std` N=1M / 100k vs NumPy | 8/10 | `0.74-0.78x` / `0.87x` — wins, gap narrows in-cache |
| Small N=1000 vs NumPy | 9/10 | `0.17x` (6x) — no small-array regression |
| Bit-exactness vs NumPy | 10/10 | 0 mismatches over 14 shapes × {ddof 0,1,2} × {var,std}; same pairwise/blocksize |
| Special-value safety | 9/10 | NaN/Inf/all-NaN/f32/int/2D all correctly defer to numpy |
| Conformance/build gates | 9/10 | conformance_var 15/15, conformance_std 15/15; fmt/clippy clean on hunk |

Evidence:
- Agent `BlackThrush`; source lever: `compute_f64_var_flat` wired into `py_std`/`var`
  before the numpy delegate (axis=None, C-contiguous f64, no out/dtype/keepdims, native
  ddof). Reuses `pairwise_simd_f64` + `pairwise_sqr_dev_f64` (the nanvar kernel).
- Ledger: `docs/NEGATIVE_EVIDENCE.md` "std/var flat … 41st win".
- The 3 `conformance_statistics` cov/corrcoef failures are PRE-EXISTING (golden-drift +
  cov-y-ddof 1-ULP) — confirmed identical (26 passed / 3 failed) with the change stashed.

## 2026-06-21 cod-b fnp-linalg batch_cholesky n=64 direct-write keep

| Area | Score | Verdict |
|---|---:|---|
| Current `500x64x64` vs NumPy | 8/10 | Already a same-worker win at `0.331x` NumPy time |
| Candidate `500x64x64` vs current | 9/10 | Direct-write widened path improves current by `0.685x` |
| Candidate `500x64x64` vs NumPy | 10/10 | Same-worker candidate is `0.227x` NumPy time |
| `1000x32x32` guard | 8/10 | Existing branch remains a win: `0.921x` vs current, `0.154x` vs NumPy |
| `64x128x128` guard | 7/10 | Still beats NumPy, but branch is not reached and the baseline row was noisy |
| Conformance/build gates | 9/10 | Focused bit-identity test, check, clippy, release build, and diff check passed |
| Tool hygiene | 7/10 | Scoped fmt/UBS remain blocked by pre-existing linalg-wide drift/noise outside this hunk |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-b`.
- Artifact directory:
  `tests/artifacts/perf/2026-06-21_linalg_batch_cholesky64_direct_write_cod_b/`.
- Source lever: `CHOL_DIRECT_WRITE_MAX_N` widened from `32` to `64`; the
  byte-identity regression test now includes `n = 64`.
- Same-worker `ovh-a` current rows: `1000x32x32 = 806,310 ns`,
  `500x64x64 = 3,587,449 ns`, `64x128x128 = 3,460,608 ns`.
- Same-worker `ovh-a` candidate rows: `1000x32x32 = 742,740 ns`,
  `500x64x64 = 2,457,592 ns`, `64x128x128 = 1,757,799 ns`.
- Direct `ovh-a` NumPy rows with Python `3.13.7`, NumPy `2.2.4`, and BLAS
  threads pinned to 1: `1000x32x32 = 4,827,292 ns`, `500x64x64 =
  10,837,794 ns`, `64x128x128 = 8,874,177 ns`.
- Counted ratios: candidate vs NumPy = `0.154x`, `0.227x`, `0.198x`; candidate
  vs current = `0.921x`, `0.685x`, `0.508x`.
- Counted scorecard: current vs NumPy **3/0/0**, candidate vs NumPy **3/0/0**,
  candidate vs current **3/0/0**.
- Final scoped gates: `cargo test -p fnp-linalg
  batch_cholesky_scratch_matches_per_lane_cholesky_nxn_bits -- --nocapture`,
  `cargo check -p fnp-linalg --all-targets`, `cargo clippy -p fnp-linalg
  --all-targets -- -D warnings`, `cargo build -p fnp-linalg --release`, and
  `git diff --check` passed via RCH/per-crate workflow.

Decision:
- Release-ready keep. This is a small, behavior-preserving allocation-elision
  lever on a measured batch Cholesky row that already dominates NumPy and now
  widens the lead.
- Do not infer that `n >= 128` should use the same threshold. That boundary is
  a blocked-Cholesky algorithm decision and needs separate proof.

---

## 2026-06-21 cod-b fnp-linalg SBR Stage-1 Spectral No-Ship

| Area | Score | Verdict |
|---|---:|---|
| Current `eigvalsh_nxn/128` vs NumPy | 2/10 | Same-worker `ovh-a` remains a `2.082x` loss |
| Current `cond_nxn/128` vs NumPy | 5/10 | Same-worker `ovh-a` is neutral at `1.034x`; not the target gap this pass |
| Current `eigvalsh_nxn/512` vs NumPy | 2/10 | Same-worker `ovh-a` remains a `2.504x` loss |
| SBR stage-1 feasibility | 6/10 | Stage 1 alone is `0.726x` of NumPy full `eigvalsh_512`, but not an API result |
| Source discipline | 10/10 | No linalg source was kept; no stage-1-only dispatch shipped |
| Retry guidance | 9/10 | Routes to true band-to-tridiagonal / band-aware eigvalsh, away from threshold and post-sort tweaks |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-b`.
- Artifact directory:
  `tests/artifacts/perf/2026-06-21_linalg_spectral_sbr_stage1_cod_b_pass4/`.
- RCH current rows on `ovh-a`: `eigvalsh_nxn/128 = 1,315,452 ns`,
  `cond_nxn/128 = 993,887 ns`, `sbr_stage1_band_nxn/512 = 19,948,921 ns`.
- Direct `ovh-a` NumPy rows: `eigvalsh_128 = 631,765 ns`,
  `eigvalsh_512 = 27,470,726 ns`, `cond_128 = 961,374 ns` with Python
  `3.13.7`, NumPy `2.2.4`, and BLAS threads pinned to 1.
- Direct `ovh-a` Rust `eigvalsh_nxn/512` row from the synced RCH workspace:
  `68,791,964 ns`.
- Counted current API scorecard vs NumPy: win/loss/neutral = **0/2/1**.
- Cross-worker routing-only row on `vmi1227854`: `eigvalsh_nxn/512 =
  42,176,502 ns`, `sbr_stage1_band_nxn/1024 = 135,221,960 ns`.
- Final scoped gates: `cargo test -p fnp-linalg sbr_stage1 --release`,
  `cargo check -p fnp-linalg --all-targets`, `cargo clippy -p fnp-linalg
  --all-targets -- -D warnings`, `cargo build -p fnp-linalg --release`,
  `git diff --check`, and `ubs` on the markdown docs/scorecard passed.

Decision:
- No release-ready improvement. Keep no source change.
- SBR remains the radical route, but the next shippable lever must be a true
  stage-2 band-to-tridiagonal reducer or band-aware eigvalsh path. A stage-1-only
  dispatch or dense-band fallback would not attack the dominant work.

---

## 2026-06-21 cod-b fnp-linalg Eigvalsh/Cond 128 Unblocked Reducer No-Ship

| Area | Score | Verdict |
|---|---:|---|
| Current `eigvalsh_nxn/128` vs NumPy | 2/10 | Confirmed current losses on `hz1` (`2.092x`) and `ovh-a` (`1.969x`) |
| Current `cond_nxn/128` vs NumPy | 3/10 | Confirmed current losses on `hz1` (`1.303x`) and `ovh-a` (`1.216x`) |
| Exact-128 unblocked values-only reducer | 0/10 | Candidate lost to NumPy by `5.280x` and `2.502x` on `vmi1153651` |
| Revert discipline | 10/10 | `crates/fnp-linalg/src/lib.rs` returned to zero diff; evidence only |
| Final per-crate gates | 8/10 | `test tridiag --release`, `check`, `clippy -D warnings`, release build, and `git diff --check` passed |
| Retry guidance | 8/10 | Routes away from unblocked exact-128, threshold, sort, direct-extrema, and row-dot families |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-b`.
- Artifact directory:
  `tests/artifacts/perf/2026-06-21_linalg_eigvalsh_cond128_cod_b_pass3/`.
- Current `hz1` rows: `eigvalsh_nxn/128 = 1,906,955 ns` vs NumPy
  `911,490 ns` (`2.092x` loss); `cond_nxn/128 = 1,787,593 ns` vs NumPy
  `1,372,420 ns` (`1.303x` loss).
- Current `ovh-a` rerun: `eigvalsh_nxn/128 = 1,318,349 ns` vs NumPy
  `669,516 ns` (`1.969x` loss); `cond_nxn/128 = 1,226,881 ns` vs NumPy
  `1,009,183 ns` (`1.216x` loss).
- Candidate source trial: exact values-only `n == 128` tridiagonalization routed
  to the existing unblocked Householder reducer while eigenvector paths and all
  other sizes stayed blocked.
- Candidate `vmi1153651` rows: `eigvalsh_nxn/128 = 4,243,947 ns` vs NumPy
  `803,699 ns` (`5.280x` loss); `cond_nxn/128 = 3,856,139 ns` vs NumPy
  `1,541,118 ns` (`2.502x` loss).
- Final scoped gates: `cargo test -p fnp-linalg tridiag --release` passed 7
  tests with 4 ignored probes; `cargo check -p fnp-linalg --all-targets`
  passed; `cargo clippy -p fnp-linalg --all-targets -- -D warnings` passed;
  `cargo build -p fnp-linalg --release` passed; `git diff --check` passed.
- `cargo fmt -p fnp-linalg --check` remains blocked by broad pre-existing
  formatting drift outside this evidence slice.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`; no new
  `.scratch` worktree.

Decision:
- No release-ready improvement. Keep no linalg source.
- The target gap remains open, but this route is rejected. A future attempt needs
  a shared-work tridiagonal eigensolver, true two-stage band reduction, or a
  genuinely generated 128-specialized reducer with paired proof.

---

## 2026-06-21 cod-b fnp-ufunc percentile_method medium gate check

| Area | Score | Verdict |
|---|---:|---|
| Current `percentile_method(None, linear)` vs NumPy | 9/10 | Same-host OVH rows are 3 wins, 0 losses, 0 neutral |
| Candidate cutoff change | 2/10 | Not kept; RCH moved candidate to another worker before paired proof |
| Probe coverage | 8/10 | Added ignored medium-N timing probe for 131K/262K/524K rows |
| Revert discipline | 9/10 | `PERCENTILE_M_GLOBAL_PARALLEL_MIN` restored to `1 << 17`; production path unchanged |
| Final per-crate gates | 8/10 | `check`, `clippy -D warnings`, percentile release filter, and trapezoid release filter passed |
| Retry guidance | 8/10 | Do not raise this gate unless a paired same-worker regression appears |

Evidence:
- Agent/bead: `YellowElk` / `cod-b`, parent `franken_numpy-ixs5y`.
- Artifact directory:
  `tests/artifacts/perf/2026-06-21_ufunc_percentile_method_gate_cod_b/`.
- Rust probe:
  `AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-ufunc percentile_method_medium_gate_report --release -- --ignored --nocapture`
  on RCH worker `ovh-a`.
- NumPy comparator: `ssh fmd` on the same OVH host, Python `3.13.7`,
  NumPy `2.2.4`, single-thread env.
- Ratios: n=131072 `0.560x`, n=262144 `0.223x`, n=524288 `0.171x`.
- Final `fnp-ufunc` gates: `cargo check -p fnp-ufunc --all-targets` passed;
  `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed after a
  behavior-preserving iterator rewrite in the current-tree `trapezoid` last-axis
  loop; `cargo test -p fnp-ufunc percentile --release` passed 33 tests with 5
  ignored perf probes; `cargo test -p fnp-ufunc trapezoid --release` passed 13
  tests.
- `cargo fmt --check -p fnp-ufunc` remains blocked by broad pre-existing
  formatting drift outside this evidence slice.

Decision:
- Current `percentile_method(axis=None, method=linear)` is release-ready for
  the checked medium rows.
- Keep the cutoff unchanged. The next BOLD-VERIFY target should be a measured
  current loss, not this already-winning method path.

---

## 2026-06-21 cod-b fnp-linalg Matrix Norm Column Current Win

| Area | Score | Verdict |
|---|---:|---|
| Current `matrix_norm_nxn_orders/(one|neg_one)` vs NumPy | 10/10 | Same-worker `vmi1152480` row is 8 wins, 0 losses, 0 neutral |
| Largest remaining row | 9/10 | Worst current ratio is still a win: `neg_one/128 = 0.811x` |
| Deep-size column rows | 10/10 | 256-1024 rows are `0.185x` to `0.275x` FNP/NumPy |
| Source discipline | 10/10 | No source change was needed or kept |
| Focused conformance | 9/10 | Column reduction bit-preservation test passed in release mode |
| Release build | 9/10 | `cargo build -p fnp-linalg --release` passed through RCH |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-b`.
- Artifact directory:
  `tests/artifacts/perf/2026-06-21_linalg_matrix_norm_column_cod_b_pass2/`.
- Current Rust bench:
  `AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'matrix_norm_nxn_orders/(one|neg_one)' -- --sample-size 12 --warm-up-time 1 --measurement-time 2 --output-format bencher`.
- RCH selected `vmi1152480`; direct NumPy comparator ran on the same worker with
  Python `3.13.7` and NumPy `2.4.6`.
- Ratios: `one/128 0.799x`, `neg_one/128 0.811x`, `one/256 0.221x`,
  `neg_one/256 0.185x`, `one/512 0.263x`, `neg_one/512 0.275x`,
  `one/1024 0.270x`, `neg_one/1024 0.267x`.
- Scorecard vs NumPy: win/loss/neutral = **8/0/0**.
- Focused conformance:
  `cargo test -p fnp-linalg matrix_norm_column_reduction_matches_strided_reference_bits --release -- --nocapture`
  passed on RCH-selected `vmi1153651`.
- Release build:
  `cargo build -p fnp-linalg --release` passed on RCH-selected `vmi1152480`.

Decision:
- Release-ready current-code proof. The older column-sum residual is stale on
  current `main`.
- No source hunk is kept. Do not reopen the allocation-only stack-threshold or
  NaN-prefilter families for this lane.

---

## 2026-06-21 cod-a fnp-python Sort Complex Real-f64 Gate Keep

| Area | Score | Verdict |
|---|---:|---|
| High-thread `sort_complex` vs NumPy | 9/10 | Fresh current rerun wins both rows: `0.975x` at 200k and `0.150x` at 1M |
| Low-thread guard | 8/10 | Forced 4-thread rerun stays neutral: `1.024x` and `1.017x` |
| Revert discipline | 9/10 | Rejected direct-output-only, combined scan/copy, and sub-8-thread native sort variants |
| Focused conformance | 9/10 | `conformance_sort_search` filtered rows passed; signed-zero/NaN unit guards added for broader lib-unit runs |
| Release build readiness | 8/10 | `cargo build -p fnp-python --release` passed through RCH with known warnings |
| Retry guidance | 8/10 | Ledger routes away from Python complex list construction and ungated parallel sort |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Source: `crates/fnp-python/src/lib.rs`, exact 1-D C-contiguous `float64`
  `sort_complex` path builds `complex128` output directly and sorts real values
  with stable Rayon sorting only at the high-thread crossover.
- Bench rows added in
  `crates/fnp-python/benches/criterion_python_surface.rs` under
  `python_sort_complex_boundary`.
- Baseline old native export on `hz1`: 200k `53,011,981 ns` vs NumPy
  `2,177,501 ns` (`24.345x` loss); 1M `294,361,340 ns` vs NumPy
  `12,954,676 ns` (`22.722x` loss).
- Rejected direct-output-only candidate: 200k `8.068x` loss, 1M `6.673x`
  loss.
- Rejected combined scan/copy candidate: 200k `1.628x` loss, 1M `1.304x`
  loss.
- Final high-thread keep on `ovh-a`: 200k `1,457,650 ns` vs NumPy
  `1,456,064 ns` (`1.001x` neutral); 1M `6,538,178 ns` vs NumPy
  `8,520,745 ns` (`0.767x` win).
- Final forced fallback with `RAYON_NUM_THREADS=4` on `ovh-a`: 200k
  `1,457,144 ns` vs NumPy `1,451,943 ns` (`1.004x` neutral); 1M
  `8,476,034 ns` vs NumPy `8,475,550 ns` (`1.000x` neutral).
- Fresh current rerun high-thread on `hz2`: 200k `8,681,463 ns` vs NumPy
  `8,900,824 ns` (`0.975x` win); 1M `8,101,862 ns` vs NumPy
  `53,997,830 ns` (`0.150x` win).
- Fresh current rerun forced fallback with `RAYON_NUM_THREADS=4` on `hz1`:
  200k `2,215,800 ns` vs NumPy `2,164,453 ns` (`1.024x` neutral); 1M
  `12,702,714 ns` vs NumPy `12,486,455 ns` (`1.017x` neutral).
- Release build:
  `AGENT_NAME=cod-a CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo build -p fnp-python --release`
  passed on `hz1` with the existing three `fnp-python` warnings.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`; no new
  `.scratch` worktree.

Decision:
- Release-ready for exact 1-D C-contiguous real-f64 `sort_complex` at the
  measured high-thread crossover.
- Keep low-thread and NaN-bearing rows delegated to NumPy; signed-zero/NaN
  behavior stays guarded by focused conformance.
- Next work should target remaining measured losses rather than broadening this
  path without a dtype/shape-specific crossover sweep.

---

## 2026-06-21 cod-b fnp-linalg Eigvalsh 128 Sturm Bisection No-Ship

| Area | Score | Verdict |
|---|---:|---|
| Current `eigvalsh_nxn/128` vs NumPy | 2/10 | Fresh `hz2` row is still a 2.059x NumPy loss |
| Sturm bisection candidate | 0/10 | Regressed to 3.322x slower than current and 6.842x slower than NumPy |
| Correctness guard | 8/10 | Candidate matched QR reference within `1e-9` before revert |
| Revert discipline | 9/10 | Production `fnp-linalg` source returned to baseline; evidence only |
| Retry guidance | 8/10 | Routes away from per-eigenvalue bisection and rejected microfamilies |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-b`.
- Artifact directory:
  `tests/artifacts/perf/2026-06-21_linalg_eigvalsh128_cod_b_pass2/`.
- Current baseline on `hz2`: `eigvalsh_nxn/size/128 = 1,545,094 ns`.
- Direct NumPy comparator via `ssh hz2`: NumPy `2.3.5`, median `750,348 ns`.
- Current FNP/NumPy ratio: `2.059x` loss.
- Candidate: exact-`n==128` Sturm-count bisection for all tridiagonal
  eigenvalues after the existing blocked reduction.
- Candidate result on `hz2`: `5,133,686 ns`; candidate/current `3.322x`
  regression; candidate/NumPy `6.842x` loss.
- Candidate focused test passed, then the source hunk and temporary test were
  removed.
- Final focused gates after revert: `cargo test -p fnp-linalg tridiag --release`
  passed 7/7 with 4 ignored timing reports; `cargo build -p fnp-linalg --release`
  passed; `git diff --check` passed. `cargo fmt --check -p fnp-linalg` still
  reports broad pre-existing linalg formatting drift and was not normalized.

Decision:
- No release-ready improvement. Keep no source.
- Do not retry full-spectrum per-eigenvalue bisection for this residual.
- Next credible `eigvalsh_nxn/128` attempt needs shared-work tridiagonal
  eigensolver work, true two-stage band-to-tridiagonal work, or a generated
  128-specific reducer that avoids the already rejected threshold/sort/deflation
  and row-dot families.

---

## 2026-06-21 cod-b fnp-python Matrix Power Lazy Fallback Keep

| Area | Score | Verdict |
|---|---:|---|
| `matrix_power(A, 1)` vs NumPy | 9/10 | Current residual `2.428x` loss became `0.354x` win on patched-source rerun |
| Sibling boundary row | 7/10 | `matrix_power(A, 0)` remained delegated; final row was noisy `1.089x`, repeat was `1.022x` |
| Revert discipline | 9/10 | Kept only lazy fallback construction; rejected broader cached fallback and eager-fallback variants |
| Focused conformance | 9/10 | `conformance_linalg_advanced matrix_power` passed 5/5 |
| Release build | 8/10 | `cargo build -p fnp-python --release` passed through `rch` with existing warnings |
| Hygiene gates | 6/10 | `cargo fmt --check -p fnp-python` still blocked by broad pre-existing formatting drift |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-b`.
- Source: `crates/fnp-python/src/lib.rs`, `matrix_power` fallback construction moved
  after the exact-ndarray `n == 1` identity return.
- Artifact directory:
  `tests/artifacts/perf/2026-06-21_fnp_python_matrix_power_cod_b/`.
- Current baseline (`ovh-a`, before lazy fallback): `n0` `143,942 ns` vs
  `143,410 ns` (`1.004x`); `n1` `1,413 ns` vs `582 ns` (`2.428x` loss).
- Final patched-source rerun (`vmi1153651`): `n0` `1,301,266 ns` vs
  `1,195,201 ns` (`1.089x`); `n1` `503 ns` vs `1,422 ns` (`0.354x`).
- Independent patched-source repeat (`hz1`): `n0` `300,795 ns` vs `294,294 ns`
  (`1.022x`); `n1` `263 ns` vs `676 ns` (`0.389x`).
- Rejected candidates: Vec-shape direct return (`2.044x` loss), broader cached
  fallback (`0.365x` but too wide), tuple-shape eager fallback (`2.167x` loss).
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`; no new
  `.scratch` worktree.

Decision:
- Release-ready for the exact NumPy ndarray `matrix_power(A, 1)` boundary.
- Keep `n==0` delegated to NumPy; do not reopen native identity allocation in this
  commit.
- Future work should target a fresh measured loss rather than broadening the
  fallback helper into a general NumPy call cache.

---

## 2026-06-21 cod-a fnp-python Matrix Power n==1 Alias Keep

| Area | Score | Verdict |
|---|---:|---|
| `matrix_power(A, 1)` vs NumPy | 9/10 | Same-worker `hz1` row improved from `2.779x` loss to `0.409x` win |
| Sibling boundary row | 8/10 | `matrix_power(A, 0)` stayed neutral/win: `0.963x` before, `0.931x` candidate |
| Focused conformance | 9/10 | `conformance_linalg_advanced matrix_power` passed 5/5 |
| Release build | 8/10 | `cargo build -p fnp-python --release` passed through `rch` |
| Revert discipline | 9/10 | One source hunk only; invalid/subclass/object-stack paths still defer to NumPy |
| Hygiene gates | 6/10 | Whole-file `rustfmt --check`, UBS, and unit-test filter remain blocked by pre-existing `fnp-python` drift/debt |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Source: `crates/fnp-python/src/lib.rs`, `matrix_power` exact-ndarray `n == 1`
  short-cut before the extraction/native multiply path.
- Counted bench worker: `hz1`; command:
  `rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- matrix_power_delegate --output-format bencher`.
- Before row (`hz1`, same turn before source hunk): `n0` `297,768 ns` vs
  `309,071 ns` (`0.963x`); `n1` `1,834 ns` vs `660 ns` (`2.779x` loss).
- Candidate row (`hz1`): `n0` `279,617 ns` vs `300,364 ns` (`0.931x`);
  `n1` `277 ns` vs `677 ns` (`0.409x`).
- Conformance:
  `rch exec -- cargo test -p fnp-python --test conformance_linalg_advanced matrix_power -- --nocapture`
  passed 5/5.
- Release build:
  `rch exec -- cargo build -p fnp-python --release` passed.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`; no new
  `.scratch` worktree.

Decision:
- Release-ready for exact NumPy ndarray `matrix_power(A, 1)`.
- This closes the micro-dispatch loss exposed by the earlier 2-D linalg delegate
  scorecard without reopening native GEMM/LAPACK lanes.
- Keep monitoring broader `fnp-python` hygiene separately; do not mix unrelated
  formatting or stale unit-test signature repairs into this perf commit.

---

## 2026-06-21 cod-b fnp-python Compress Mask Count/Compaction Keep

| Area | Score | Verdict |
|---|---:|---|
| `compress_f64_axis_none` vs NumPy | 9/10 | 2 wins, 0 losses; candidate ratios 0.363x and 0.498x |
| Revert discipline | 8/10 | Failed first 16-lane attempt was fixed before keep; no regression hunk retained |
| Focused conformance | 8/10 | Filtered `compress` shard passed 13/13; full shard's lone failure is unrelated `choose` parity |
| Release build | 8/10 | `cargo build -p fnp-python --release` passed through `rch` |
| Hygiene gates | 6/10 | UBS/fmt report broad pre-existing `fnp-python` debt; no broad cleanup mixed into this perf commit |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-b`.
- Source: `crates/fnp-python/src/lib.rs`, flat f64 `compress` fast path and
  generic typed mask compactor.
- Artifact directory:
  `tests/artifacts/perf/2026-06-21_fnp_python_compress_cod_b/`.
- Baseline (`hz1`): `compress_f64_axis_none_100000` FNP/NumPy `1.123x`;
  `compress_f64_axis_none_1000000` FNP/NumPy `1.077x`.
- Candidate (`vmi1149989`, same process FNP vs NumPy): 100K row
  `62,745 ns` vs `172,737 ns` (`0.363x`); 1M row `883,588 ns` vs
  `1,773,287 ns` (`0.498x`).
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`; no new
  `.scratch` worktree.

Decision:
- Release-ready for this exact flat f64 `compress(axis=None)` row.
- Cross-worker candidate-vs-baseline movement is not used as proof; the proof is
  the candidate head-to-head ratio against NumPy in the same Criterion process.
- Next target should be a current measured loss, not another pass over the
  already-fixed 8-lane branch.

---

## 2026-06-21 cod-a fnp-python 2-D Linalg Delegate Criterion Recheck

| Area | Score | Verdict |
|---|---:|---|
| 2-D `eigvalsh`/`eigh`/`cholesky` delegate rows | 8/10 | 2 wins, 0 losses, 4 neutral; old dense Python-surface loss remains closed |
| `matrix_power` boundary rows | 5/10 | `n=0` parity; `n=1` exposes a 2.407x micro-dispatch loss |
| Guard linalg boundary rows | 9/10 | Batch `slogdet`/`inv`/`solve` still dominate NumPy; batched Cholesky stays parity/win |
| Revert discipline | 9/10 | Kept benchmark rows only; no production edit while `fnp-python/src/lib.rs` is peer-dirty |
| Focused conformance | 9/10 | `conformance_linalg*` release shards passed 69/69 |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Added exact 2-D delegate rows to
  `crates/fnp-python/benches/criterion_python_surface.rs`.
- Counted bench worker: `ovh-a`; command:
  `rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --
  python_linalg_boundary --output-format bencher`.
- New delegate ratios: `eigvalsh` n=200 `1.002x`, `eigvalsh` n=800 `1.011x`,
  `eigh` n=200 `0.886x`, `eigh` n=800 `0.996x`, `cholesky` n=200 `0.906x`,
  `cholesky` n=800 `0.997x`.
- `matrix_power(A, 0)` n=800 is `1.015x`; `matrix_power(A, 1)` n=800 is
  `2.407x`, but only `1,401 ns` versus NumPy `582 ns`.
- Same run guard score across all linalg boundary pairs:
  **11 wins / 1 loss / 9 neutral**.
- Focused conformance used `rch` with no admissible worker and fell back locally,
  still using `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`:
  `conformance_linalg` 1/1, `conformance_linalg_advanced` 29/29,
  `conformance_linalg_decomp` 39/39.

Decision:
- Treat dense 2-D `eigvalsh`/`eigh`/`cholesky` as release-ready for the Python
  surface; do not reopen native-kernel work for those exact ndarray rows.
- Keep no production source change from this pass. The remaining measured loss
  is a narrow `matrix_power(A, 1)` wrapper dispatch floor and should wait until
  `crates/fnp-python/src/lib.rs` is free from peer-owned compress work.

---

## 2026-06-21 cod-a fnp-python Batch Inv/Solve Current Recheck

| Area | Score | Verdict |
|---|---:|---|
| Python `inv` batch rows vs NumPy | 9/10 | 3 wins, 0 losses, 0 neutral |
| Python `solve` guard rows vs NumPy | 9/10 | 2 wins, 0 losses, 0 neutral |
| Revert discipline | 9/10 | Rejected source-kernel edit; kept only benchmark rows |
| Focused conformance | 8/10 | `conformance_linalg` 1/1 passed on `ovh-a` |
| Same-process comparator freshness | 8/10 | FNP and NumPy ran inside the same Criterion bench process |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Added `fnp_inv`/`numpy_inv` rows to
  `crates/fnp-python/benches/criterion_python_surface.rs`.
- Counted `inv` worker: `ovh-a`; command:
  `rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --
  inv_f64 --output-format bencher`.
- `inv` FNP/NumPy ratios: batch8192 4x4 `0.155x`, batch64 128x128
  `0.067x`, batch16 256x256 `0.134x`.
- Counted `solve` guard worker: `vmi1149989`; command:
  `rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --
  solve_f64_batch8192_4x4 --output-format bencher`.
- `solve` FNP/NumPy ratios: vector RHS `0.231x`, matrix RHS `0.252x`.

Decision:
- Mark the previous batch `inv` / `solve` light-lane loss routing as stale for
  the Python-boundary API rows measured here.
- No source kernel edit. A generated direct small-N inverse/solve path would be
  premature without a fresh same-process NumPy loss.
- Route future BOLD-VERIFY work to a current measured residual outside this
  closed slice (`eigvalsh_nxn/128`, architectural `sqrt` zero-init, or
  peer-owned Python wrapper lanes).

---

## 2026-06-21 cod-b fnp-linalg Matrix Norm Current Recheck

| Area | Score | Verdict |
|---|---:|---|
| Current `matrix_norm_nxn_orders` 1/-1 rows | 8/10 | Current head wins all six checked rows versus prior direct NumPy and local routing comparator |
| Revert discipline | 9/10 | No source hunk attempted; avoided the already-rejected scalar strip-mine family |
| Focused conformance/build | 8/10 | Focused column-reduction bit test and `fnp-linalg` release build passed through `rch` |
| Same-host NumPy comparator freshness | 4/10 | Fresh same-host Python was blocked by SSH auth; local comparator is routing-only |

Evidence:
- Bead: `franken_numpy-ixs5y.281`; agent `YellowElk` / `cod-b`.
- Current Rust bench worker: `vmi1152480`; command:
  `rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg
  'matrix_norm_nxn_orders/(one|neg_one)/(256|512|1024)'`.
- Current FNP medians: `one/256` 7,743 ns, `neg_one/256` 5,207 ns,
  `one/512` 26,211 ns, `neg_one/512` 25,737 ns, `one/1024` 99,936 ns,
  `neg_one/1024` 98,382 ns.
- Against the prior direct `hz2` NumPy rows, current FNP/NumPy ratios are
  `0.279x`, `0.184x`, `0.253x`, `0.250x`, `0.252x`, and `0.250x`.
- SSH to the selected worker was denied, and `rch exec -- python3` runs locally
  for non-compilation commands, so the fresh `thinkstation1` NumPy 2.4.3 ratios
  are recorded only as cross-host routing evidence.

Decision:
- Keep no source change. Mark the previous matrix-norm 1/-1 column-reduction
  gap as stale at current head.
- Route future work to a fresh measured loss, not another scalar strip-mine or
  allocation-only matrix-norm retune.

---

## 2026-06-21 cod-a fnp-linalg Spectral Cond No-Ship Recheck

| Area | Score | Verdict |
|---|---:|---|
| `cond_nxn/128` target gap | 2/10 | Still 1.115x slower than NumPy after candidate |
| `eigvalsh_nxn/128` adjacent gap | 2/10 | Still 1.820x slower than NumPy after candidate |
| Guard rows already winning vs NumPy | 7/10 | `cond_nxn` 64/256/512 stayed wins, but this does not close the target |
| Revert discipline | 9/10 | Scan/sort elision source was reverted after neutral target result |
| Focused conformance | 8/10 | `cond_p_spectral_symmetric` focused release test passed |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Counted worker: `hz2`; candidate ran directly in existing warm RCH target
  `.rch-target-hz2-pool-f4ecbc5a8032ed7eb8c61438ab6b2cc8`.
- Current target baseline: `cond_nxn/128 = 1,242,314 ns`; NumPy `1,110,135 ns`;
  current FNP/NumPy `1.119x`.
- Candidate target: `cond_nxn/128 = 1,237,760 ns`; candidate/current `0.996x`;
  candidate/NumPy `1.115x`.
- Adjacent `eigvalsh_nxn/128`: candidate `1,359,806 ns` vs NumPy `747,108 ns`,
  `1.820x`.

Decision:
- No release-ready improvement from this slice.
- Keep no source. Route the next spectral attempt to a deeper reduction or
  eigensolver primitive, not scan elision or eigenvalue postprocessing.

---

## 2026-06-21 cod-a fnp-random PCG Current Recheck

| Area | Score | Verdict |
|---|---:|---|
| Current PCG head-to-head ratio-vs-NumPy | 9/10 | 10 wins, 0 losses, 0 neutral rows |
| `Generator::bytes` stale-gap closure | 9/10 | Both byte rows now win after `.265` direct final-buffer append/fill |
| Revert discipline | 9/10 | `.257` intermediate word-vector bytes path remains rejected and absent |
| Focused conformance/build | 8/10 | Per-crate `fnp-random` gates rerun for this docs recheck |

Evidence:
- Bead: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- RCH worker for counted bench: `vmi1152480`; command:
  `rch exec -- cargo bench -p fnp-random --bench random_vs_numpy --
  --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format
  bencher`.
- Ratio table: raw `fill_u64` 0.390x and 0.460x, `Generator::bytes` 0.522x
  and 0.268x, `gumbel` 0.299x and 0.172x, `laplace` 0.314x and 0.139x,
  full-range `uint8` 0.932x and 0.293x.

Decision:
- Mark the previous "current `Generator::bytes` parity/perf gap" as stale.
- No source change and no revert in this pass.
- Route future BOLD-VERIFY work to active measured losses outside this closed PCG
  cluster.

---

## 2026-06-21 cod-a fnp-python Linalg Boundary Reverify

| Area | Score | Verdict |
|---|---:|---|
| Python linalg boundary ratio-vs-NumPy | 9/10 | 6 wins, 0 losses, 2 neutral rows |
| Delegate behavior boundary | 9/10 | Exact 2-D LAPACK-shaped ndarray calls delegated; batched/native winning paths preserved |
| Focused conformance | 8/10 | `conformance_linalg` 1/1 and `conformance_linalg_decomp` 39/39 pass; advanced shard 28/29 with only missing SciPy |
| Current dirty-worktree independence | 6/10 | Later filtered rerun blocked by unowned `fnp-ufunc` unsafe edit, not by linalg delegate behavior |

Evidence:
- Bead: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- RCH worker for counted bench: `vmi1149989`; command:
  `rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --
  python_linalg_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1
  --output-format bencher`.
- Ratio table: `slogdet` 0.331x, `solve` vec 0.367x, `solve` mat2 0.469x,
  `cholesky` 4x4 1.010x, 8x8 0.870x, 16x16 0.853x, 32x32 0.919x,
  64x64 0.989x.
- Counted conformance: `conformance_linalg` 1/1 PASS; `conformance_linalg_decomp`
  39/39 PASS. `conformance_linalg_advanced` passed 28/29 and stopped only because
  `solve_triangular_complex` imports `scipy`, which was not installed on the
  worker.

Decision:
- Mark the previous code-only 2-D dense-linalg delegate rows as superseded by
  measured evidence for this focused boundary slice.
- No source change and no revert in this cod-a pass.
- Remaining target gaps are not this wrapper cliff; route future work to the
  measured kernel/batching losses (`batch_inv`, `batch_solve`, and native
  `eigvalsh_nxn/128`) with a different primitive.

---

## 2026-06-21 fnp-python matrix_power n=0/1 Boundary Delegate Code-Only Slice

| Area | Score | Verdict |
|---|---:|---|
| `matrix_power` exact ndarray `n=0` and `n=1` | 3/10 | Code-only pending bench |
| `matrix_power` powers `>=2` native path | 8/10 | Left unchanged |
| Focused conformance | 0/10 | Pending disk recovery |
| Fresh Criterion ratio-vs-NumPy | 0/10 | Pending disk recovery |

Evidence:
- Bead: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Code-only lever: delegate exact NumPy ndarray boundary exponents `0` and `1`
  to `numpy.linalg.matrix_power` before extracting into Rust.
- Rationale: NumPy's boundary paths need only shape/dtype for `n=0` and return
  the asarray result for `n=1`; the previous wrapper paid an avoidable full
  matrix extract plus finite scan first.
- No new cargo build/bench/test/check was started under the 45G disk-low
  instruction.

Decision:
- Keep as a pending-bench code-only commit, not a measured win.
- Next admissible turn must run focused `fnp-python` Criterion rows and
  `matrix_power` conformance, then either score the ratio-vs-NumPy or revert on
  ~0 gain/regression.

---

## 2026-06-21 fnp-python cholesky 2-D Delegate Code-Only Slice

| Area | Score | Verdict |
|---|---:|---|
| `fnp_python.linalg.cholesky` exact ndarray real 2-D square inputs | 3/10 | Code-only pending bench |
| Stacked / non-ndarray cholesky paths | 8/10 | Left unchanged |
| Focused conformance | 0/10 | Pending disk recovery |
| Fresh Criterion ratio-vs-NumPy | 0/10 | Pending disk recovery |

Evidence:
- Bead: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Existing disk-low probe recorded the native 2-D `cholesky` path losing to
  NumPy by `2.95x` at 200x200 and `6.28x` at 800x800.
- Code-only lever: delegate exact NumPy ndarray real 2-D square inputs to
  `numpy.linalg.cholesky` before extracting into Rust; preserve `upper` fallback
  semantics and keep stacked / non-ndarray paths unchanged.
- No new cargo build/bench/test/check was started after the 48G disk-low
  instruction. Agent Mail writes are blocked by the corrupt DB circuit breaker.

Decision:
- Keep as a pending-bench code-only commit, not a measured win.
- Next admissible turn must run focused `fnp-python` Criterion rows and cholesky
  conformance, then either score the ratio-vs-NumPy or revert on ~0 gain.

---

## 2026-06-21 fnp-python 2-D Eigh Delegate Code-Only Candidate

| Area | Score | Verdict |
|---|---:|---|
| Existing 2-D `eigh` native path | 2/10 | Not release-ready; prior Python-surface native path loses 4.18x@200 and 4.05x@800 vs NumPy |
| Delegate candidate source | 5/10 | Code-only candidate is already on `main` via `76712a2b`; exact real 2-D square float `ndarray` routes to NumPy before extraction |
| Batched `batch_eigh` preservation | 6/10 | Source path untouched; guard benchmark still pending |
| Validation status | 1/10 | No direct cargo build/bench run and no focused conformance; targeted UBS failed on broad pre-existing inventory |

Evidence:
- Bead: `franken_numpy-ixs5y.278`.
- Existing measured native ratios: `np.linalg.eigh` real 2-D square float
  ndarray loses `4.18x` at n=200 and `4.05x` at n=800.
- Source change: remote `main` already applied the same metadata-only
  shape/dtype peek used by `eigvalsh` in `76712a2b`; matching 2-D inputs call
  `numpy.linalg.eigh(..., UPLO=UPLO)` before Rust extraction. The duplicate
  local hunk from bead `.278` was skipped during rebase.
- No after-ratio is recorded yet. Build, focused conformance, and head-to-head
  bench are pending the next disk-safe turn.
- Targeted UBS on the changed file set exited nonzero from existing
  `fnp-python` findings; it did not identify the new `eigh` hunk as the cause.
- Agent Mail reservation could not be recorded because the database corruption
  circuit breaker refused writes; coordination is via `docs/NEGATIVE_EVIDENCE.md`.

Decision:
- Keep the upstream code-only delegate candidate for the next validation slice.
- Do not mark the row release-ready until 2-D `eigh` after-ratios are measured
  and the batched native guard still routes correctly.
- If validation fails, revert the single wrapper hunk and keep the ledger entry
  as negative evidence.

---

## 2026-06-21 Linalg Eigvalsh 128 Values-Only Reducer Probe

| Area | Score | Verdict |
|---|---:|---|
| `eigvalsh_nxn/128` current row | 2/10 | Not release-ready; 1.937x slower than NumPy on `vmi1149989` |
| Tail-local small-n reducer matvec | 0/10 | Rejected; paired direct A/B regressed 1.066x |
| Tridiagonal correctness gates | 9/10 | Focused release tests passed |
| Source/revert discipline | 9/10 | No production linalg diff kept |

Evidence:
- Artifact: `tests/artifacts/perf/2026-06-21_linalg_eigvalsh128_values_reducer_cod_b/`.
- Current baseline: `eigvalsh_nxn/size/128 = 1,372,654 ns`; same-worker NumPy
  median `708,451 ns`; FNP/NumPy `1.937x`.
- Candidate first run: `1,295,452 ns`, still `1.829x` NumPy and within baseline
  noise.
- Paired direct repeat on `vmi1149989`: baseline `1,295,211 ns`, candidate
  `1,380,393 ns`; candidate/baseline `1.066x` regression.
- `cargo test -p fnp-linalg tridiag --release` passed; QR profile stayed on the
  already-optimized scaled-hypot path.

Decision:
- Keep no source from this probe.
- Keep the negative evidence and route `eigvalsh_nxn/128` to a different
  reducer/eigensolver primitive.

---

## 2026-06-20 fnp-python Einsum Reduce-All Scalar Builder

| Area | Score | Verdict |
|---|---:|---|
| `fnp_einsum_reduce_all_f64_1000` | 8/10 | Release-ready current win for this worker |
| Existing f64 single-operand reduction conformance | 9/10 | Golden SHA and scalar parity green |
| Adjacent `reduce_rows_f64_1000` guard | 4/10 | Needs separate focused recheck; candidate run was 1.035x slower than NumPy |
| `fnp-python` all-targets/clippy/fmt hygiene | 3/10 | Blocked by pre-existing unrelated crate debt |

Evidence:
- Artifact: `tests/artifacts/perf/2026-06-20_python_einsum_reduce_all_scalar_cod_a/`.
- Same-worker `vmi1149989` target row:
  baseline `119,524 ns` vs NumPy `115,252 ns` (`1.037x` loss);
  candidate `100,778 ns` vs NumPy `104,427 ns` (`0.965x` win);
  candidate/old FNP `0.843x`.
- Guard rows from the same Criterion group:
  trace `0.754x`, diagonal `0.775x`, reduce rows `1.035x`, reduce cols `0.350x`
  candidate FNP/NumPy.
- `cargo test -p fnp-python --test conformance_einsum` passed after RCH failed
  open locally; `cargo build -p fnp-python --release` passed on `hz1`.
- `cargo check -p fnp-python --all-targets`, clippy, and fmt remain blocked by
  pre-existing unrelated `fnp-python` debt recorded in the artifact logs.
- Bounded UBS on the changed Rust file exited nonzero from broad existing
  `fnp-python` inventory; `git diff --check` passed.

Decision:
- Keep the exact-contiguous-f64 `einsum("ij->")` scalar builder fast path.
- Treat this single-operand reduce-all row as a current measured win rather than
  an active gap.
- Recheck row reductions separately before acting on the candidate-run
  row-guard loss; this patch does not alter that branch's source path.

---

## 2026-06-20 Linalg Symmetric Spectral / Batch Eigvalsh Bold-Verify

| Area | Score | Verdict |
|---|---:|---|
| `batch_eigvalsh` 64x128x128 and 16x256x256 rows | 9/10 | Release-ready current win |
| `cond_nxn` 128 and 512 exact-symmetric rows | 8/10 | Current win on this worker |
| `cond_nxn` 64 and 256 exact-symmetric rows | 3/10 | Not release-ready; still slower than NumPy |
| `eigvalsh_nxn/128` | 2/10 | Not release-ready; 3.081x slower than NumPy |
| Lanczos/power extremal-cond shortcut | 0/10 | Rejected before source edit; clustered spectra made residuals too loose |

Evidence:
- Artifact: `tests/artifacts/perf/2026-06-20_linalg_cond_lanczos_cod_a/`.
- Same-worker `vmi1227854` current FNP/NumPy ratios:
  `eigvalsh_nxn/128 = 3.081x`,
  `cond_nxn/64 = 1.409x`,
  `cond_nxn/128 = 0.859x`,
  `cond_nxn/256 = 1.428x`,
  `cond_nxn/512 = 0.431x`,
  `batch_eigvalsh/64x128x128 = 0.577x`,
  `batch_eigvalsh/16x256x256 = 0.0057x`.
- The initial Python comparator file is retained but invalid because `rch exec`
  ran it locally; counted NumPy rows come from direct SSH on `vmi1227854`.
- Fresh QR profile passed and reported the current scaled-hypot values-only QR
  path remains 1.24x-1.25x faster than the old libm-hypot path.
- Production `crates/fnp-linalg/src/lib.rs` source remains unchanged.

Decision:
- Keep no new linalg source from this slice.
- Treat batch eigvalsh as a current measured win, not a gap.
- Route the remaining spectral work to a deeper reduction/eigensolver primitive:
  dsytrd-class blocked Householder, two-stage tridiagonalization, or a fully
  convergent tridiagonal eigensolver. Do not repeat sort, threshold,
  direct-extrema, or fixed-iteration extremal shortcuts for this class.

---

## 2026-06-20 Linalg Spectral Bold-Verify

| Area | Score | Verdict |
|---|---:|---|
| Current `batch_cholesky` 64/128/256 rows | 9/10 | Release-ready current win |
| `eigvalsh_nxn/128` | 2/10 | Not release-ready; 3.051x slower than NumPy |
| `cond_nxn/128` | 4/10 | Not release-ready; 1.583x slower than NumPy |
| Small-threshold / sort / cond-extrema probes | 0/10 | Rejected and reverted |

Evidence:
- Artifact: `tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_cod_a/`.
- Current batch Cholesky FNP/NumPy ratios: `0.281x`, `0.226x`, `0.152x`.
- Current spectral FNP/NumPy ratios on `vmi1227854`: `eigvalsh_nxn/128 = 3.051x`,
  `cond_nxn/128 = 1.583x`.
- Rejected probes:
  `TRIDIAG_BLOCK_MIN=192` failed golden digest,
  `cond_nxn` direct extrema regressed paired A/B by `1.026x`,
  `eigvalsh_nxn sort_unstable_by` regressed paired A/B by `1.113x`.
- Production `crates/fnp-linalg/src/lib.rs` source returned to baseline after
  rejected probes.

Decision:
- Keep no new linalg source from this slice.
- Preserve the batch-Cholesky benchmark evidence as a current win.
- Route next spectral work to a deeper tridiagonal reduction/QR primitive; do
  not retry threshold, sort, or post-processing-only levers for this loss class.

---

## 2026-06-19 Random PCG Backlog

## Summary

| Area | Score | Verdict |
|---|---:|---|
| `franken_numpy-ixs5y.255` parallel PCG raw `fill_u64` | 9/10 | Release-ready keep |
| `franken_numpy-ixs5y.257` PCG bytes word-fill | 0/10 | Rejected and reverted |
| Current `Generator::bytes` direct final-buffer path | 9/10 | Fresh 2026-06-21 rerun: 2/0/0 wins vs NumPy |
| `franken_numpy-ixs5y.250` parallel PCG gumbel inverse-CDF fill | 9/10 | Release-ready keep |
| `franken_numpy-ixs5y.253` parallel PCG laplace inverse-CDF fill | 9/10 | Release-ready keep |

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| Crate bench build | Pass | `rch exec -- cargo check -p fnp-random --benches` |
| Raw-fill conformance | Pass | `parallel_pcg_fill_u64_matches_serial_stream_state` |
| Bytes stream conformance | Pass | `bytes_large_calls_match_serial_uint32_stream_state` |
| Gumbel stream conformance | Pass | `parallel_pcg_gumbel_matches_serial_stream_state` |
| Laplace stream conformance | Pass | `parallel_pcg_laplace_matches_serial_stream_state` |
| Distribution live NumPy oracle | Pass | `gumbel_matches_live_numpy_oracle`, `laplace_matches_live_numpy_oracle` |
| Head-to-head Criterion vs NumPy | Pass | 2026-06-21 current rerun is 10/0/0 vs NumPy |
| Head-to-head distribution Criterion vs NumPy | Pass | `tests/artifacts/perf/2026-06-19_random_vs_numpy_pcg_distributions/` |
| Negative-evidence ledger | Updated | `docs/NEGATIVE_EVIDENCE.md` |
| Required reverts | Done | Removed `.257` production word-fill path from `Generator::bytes` |

## Decision

Keep `.255`: final-code `fill_u64` is faster than NumPy by 3.72x at 100k u64 and 2.03x at 1M u64 on `hz1`.

Reject `.257`: pre-revert bytes word-fill was slower than NumPy by 1.64x at 100k bytes and 1.99x at 1M bytes on `ovh-a`. The production optimization was reverted. The later `.265` direct final-buffer append/fill path supersedes the old serial gap; the 2026-06-21 current rerun measured `Generator::bytes` as 0.522x NumPy at 100k and 0.268x NumPy at 1M.

Keep `.250`: final-code PCG64 gumbel is faster than NumPy by 6.01x at 100k f64 and 7.15x at 1M f64 on `ovh-a`.

Keep `.253`: final-code PCG64 laplace is faster than NumPy by 6.76x at 100k f64 and 8.67x at 1M f64 on `ovh-a`.

---

## 2026-06-21 cod-a fnp-linalg Eigvalsh 128 Current-Loss Reverify

| Area | Score | Verdict |
|---|---:|---|
| Current `eigvalsh_nxn/128` vs NumPy | 2/10 | Same-host `ovh-a`/`fmd` row is `2.912x` slower than NumPy |
| Exact-128 blocked route | 5/10 | Already present on `main`; no source hunk to keep |
| Revert discipline | 10/10 | No `fnp-linalg` source diff kept |
| Focused conformance | 9/10 | Filtered release `eigvalsh` tests passed: 7 unit rows plus 3 golden rows |
| Release build | 9/10 | `cargo build -p fnp-linalg --release` passed through RCH |
| Retry guidance | 8/10 | Routes future work to a deeper tridiagonal eigensolver/reducer, not shallow gates |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Current Rust row on RCH-selected `ovh-a`: `eigvalsh_nxn/size/128 =
  1,908,101 ns`.
- Same-host NumPy comparator through `ssh fmd`, Python `3.13.7`, NumPy `2.2.4`,
  single-thread BLAS env: median `655,420 ns`.
- Ratio-vs-NumPy: `2.912x` loss; win/loss/neutral = **0/1/0**.
- Focused tests:
  `cargo test -p fnp-linalg eigvalsh --release -- --nocapture` passed on
  RCH-selected `vmi1227854`.
- Release build:
  `cargo build -p fnp-linalg --release` passed on RCH-selected `vmi1293453`.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`; no new
  `.scratch` worktree.

Decision:
- No release-ready improvement was available from the exact-128 blocked route
  because that route is already current. Keep no source.
- The target remains open for a deeper symmetric spectral primitive: shared-work
  tridiagonal eigensolver, true band-to-tridiagonal stage, or generated
  128-specific reducer with paired proof.

---

## 2026-06-21 cod-a fnp-linalg Terminal-2x2 QR No-Ship

| Area | Score | Verdict |
|---|---:|---|
| Current six-row linalg slice vs NumPy | 1/10 | Same-worker `vmi1227854` remains **0/6/0** vs NumPy |
| Terminal-2x2 QR candidate vs current | 0/10 | Same-worker candidate regressed all six measured rows |
| Revert discipline | 10/10 | Candidate source hunk was removed; final linalg source is unchanged |
| Disk discipline | 10/10 | Used existing `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`; no new `.scratch` |
| Retry guidance | 8/10 | Routes away from QR tail cleanup and toward reducer/eigensolver replacement |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- RCH same-worker baseline on `vmi1227854`: `eigvalsh_nxn/64 = 204,702 ns`,
  `eigvalsh_nxn/128 = 1,313,136 ns`, `eigvalsh_nxn/256 = 8,099,070 ns`,
  `cond_nxn/64 = 156,445 ns`, `cond_nxn/128 = 1,162,411 ns`,
  `cond_nxn/256 = 7,369,744 ns`.
- Same-host NumPy on `vmi1227854`, Python `3.13.7`, NumPy `2.4.6`, BLAS
  threads pinned to 1: `eigvalsh` 64/128/256 = `161,342 / 465,138 /
  1,987,180 ns`; `cond` 64/128/256 = `131,617 / 764,155 / 4,544,545 ns`.
- Current ratio-vs-NumPy: `eigvalsh` 64/128/256 = `1.269x / 2.823x /
  4.076x`; `cond` 64/128/256 = `1.189x / 1.521x / 1.622x`.
- Candidate terminal-2x2 QR rows on the same worker: `211,842 / 1,376,577 /
  9,645,038 ns` for `eigvalsh`; `175,060 / 1,208,742 / 8,700,746 ns` for
  `cond`.
- Candidate/current: `1.035x / 1.048x / 1.191x` for `eigvalsh`; `1.119x /
  1.040x / 1.181x` for `cond`.

Decision:
- No release-ready improvement. Keep no source change.
- Do not retry terminal 2x2 QR deflation, QR tail cleanup, sorting-only changes,
  or shallow active-window gates as standalone levers. The next credible path is
  a real reducer/eigensolver replacement: band-to-tridiagonal stage 2,
  band-aware eigvalsh, or generated fixed-size reducer with paired NumPy proof.

---

## 2026-06-21 cod-b fnp-linalg Exact Tridiagonal Eigvalsh Keep

| Area | Score | Verdict |
|---|---:|---|
| Exact tridiagonal old FNP vs NumPy | 2/10 | Same-worker old path lost all rows: `2.066x / 2.532x / 2.274x` NumPy time |
| Exact tridiagonal final vs old FNP | 10/10 | Same-worker final is `0.283x / 0.185x / 0.127x` old FNP time |
| Exact tridiagonal final vs NumPy | 10/10 | Same-worker final beats NumPy on all measured rows: `0.584x / 0.468x / 0.288x` |
| Dense spectral frontier | 4/10 | Dense SPD `eigvalsh_nxn` loss remains open; this keep covers exact tridiagonal inputs |
| Conformance/check/clippy/build | 9/10 | Focused eigvalsh tests, fast-path tests, check, clippy, and release build passed through RCH |
| Disk discipline | 10/10 | Used existing `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`; no new `.scratch` |
| Validation caveats | 7/10 | Workspace fmt and UBS remain blocked by unrelated pre-existing drift/noise |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-b`.
- Source lever: exact symmetric tridiagonal inputs to `eigvalsh_nxn` now extract
  diagonal/off-diagonal arrays and enter the existing tridiagonal QR eigensolver
  directly, skipping dense Householder tridiagonalization.
- Same-worker `vmi1149989` old FNP rows: `128 = 1,477,437 ns`, `256 =
  9,305,490 ns`, `512 = 49,845,148 ns`.
- Same-worker `vmi1149989` final FNP rows: `128 = 417,718 ns`, `256 =
  1,721,791 ns`, `512 = 6,320,658 ns`.
- Same-worker direct NumPy rows with NumPy `2.2.4` and BLAS threads pinned to 1:
  `128 = 715,137 ns`, `256 = 3,675,686 ns`, `512 = 21,924,302 ns`.
- Counted scorecard: old FNP vs NumPy **0/3/0**, final FNP vs old FNP
  **3/0/0**, final FNP vs NumPy **3/0/0**.
- RCH `hz1` candidate sanity rows: `535,984 / 1,941,455 / 8,016,470 ns`.
- Rejected micro-variant: a two-pass delayed-allocation helper regressed the
  `128` row to `635,968 ns` on `vmi1149989`; reverted before final.
- Final scoped gates: `cargo test -p fnp-linalg eigvalsh --release`,
  `cargo test -p fnp-linalg exact_tridiagonal -- --nocapture`,
  `cargo check -p fnp-linalg --all-targets`, and `cargo clippy -p fnp-linalg
  --all-targets -- -D warnings`, and `cargo build -p fnp-linalg --release`
  passed through RCH/per-crate workflow.

Decision:
- Release-ready keep for exact symmetric tridiagonal inputs.
- Do not extend this to approximate bands, asymmetric inputs, or dense SPD cases
  without fresh parity and same-worker NumPy proof. Dense `eigvalsh_nxn` still
  needs a deeper reducer/eigensolver replacement.

---

## 2026-06-21 cod-a fnp-linalg Diagonal Eigvalsh QR-Skip No-Ship

| Area | Score | Verdict |
|---|---:|---|
| Current diagonal `eigvalsh` vs NumPy | 10/10 | Same-worker current is `0.030x / 0.019x / 0.014x` NumPy time |
| Candidate QR-skip vs current | 0/10 | Candidate regressed all rows: `1.324x / 1.252x / 1.212x` current time |
| Revert discipline | 10/10 | Candidate source hunk was removed; final linalg source is unchanged |
| Benchmark coverage | 8/10 | Added focused `eigvalsh_diagonal_nxn` Criterion rows for 128/256/512 |
| Disk discipline | 10/10 | Used existing `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`; no new `.scratch` |
| Retry guidance | 8/10 | Routes away from diagonal flag cleanup and back to dense spectral primitives |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Current RCH rows on same worker `ovh-a`: `12,132 / 51,756 / 281,984 ns`
  for 128/256/512 descending diagonal matrices.
- Candidate QR-skip rows on same worker `ovh-a`: `16,057 / 64,813 /
  341,859 ns`.
- Direct same-host NumPy rows on `fmd`, NumPy `2.2.4`, BLAS threads pinned to
  1: `405,480 / 2,707,520 / 19,579,503 ns`.
- Counted scorecard: current FNP vs NumPy **3/0/0**, candidate vs current FNP
  **0/3/0**, candidate vs NumPy **3/0/0**.

Decision:
- No release-ready source improvement. Keep the focused benchmark and evidence;
  keep no linalg source hunk.
- Do not retry diagonal QR-skip unless future profiles show zero-offdiagonal QR
  deflation became expensive. Dense `eigvalsh_nxn` remains the real open gap.
