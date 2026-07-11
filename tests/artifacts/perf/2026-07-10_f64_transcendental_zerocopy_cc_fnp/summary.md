# f64 transcendental zero-copy fused-defer lever (cc_fnp, 2026-07-10)

Bead: deadlock-audit-scy0o. Sibling finding: deadlock-audit-tvy7o (warning-surface gap).

## Profile basis (ranked hotspot, from code + ledger + fresh baseline below)

- f64 sin/cos/tan/tanh/expm1/log1p/arcsin/... route: `zerocopy_f64_unary_flat`
  rejects them (not in the copy-equivalent list) -> `extract_precise_numeric_array`
  (cold mmap copy, ~16k minor faults/4M call per the in-tree comment) ->
  `UFuncArray::try_elementwise_unary` (vec![0.0; n] zero-init + parallel compute,
  per-op gate 1<<15) -> `build_numpy_array_from_ufunc` (third copy).
- numpy's f64 transcendental ufuncs are single-threaded scalar-libm-class calls
  on this fleet; fnp's native path already wins at 4M (sin 0.60x, ledger
  2026-06-21) DESPITE the three copies; the 2026-06-22 sweep recorded medium-N
  (2^20) losses of 1.4-1.5x for sin/cos/expm1 with a "per-op gate + quiet box"
  retry predicate.
- The ledger "copy buffer to Vec before parallel" antipattern thread has 3 win
  families (cheap unary b88b1995, binary fa71f8d2, argextreme); the
  error-tracking transcendentals are the unapplied remainder, held back only by
  the event surface, which the sqrt fused-defer precedent (b40ff37b) solves.

## One lever

Extend `zerocopy_f64_unary_flat` to 15 transcendental ops: direct parallel
scalar-libm map for the no-event ops (sin/cos/tan/arctan/arcsinh/tanh/cbrt),
sqrt-style fused compute+detect for the error-tracking ops
(expm1/log1p/sinh/cosh/arcsin/arccos/arctanh/arccosh) with predicates that
EXACTLY mirror fnp-ufunc note_unary_float_errors; any would-be event defers the
whole call to the unchanged UFuncArray path (event recording identical).
Parallel gate 1<<15 = the native path's parallel_min_len for these ops (regime
unchanged; only the copies disappear). Values bit-identical: same
UnaryOp::apply scalar libm call per element, disjoint chunk writes.

## Runs

- baseline_bench_run1.txt: pre-change fnp-vs-numpy median gate. Worker
  vmi1227854, binary sha256 2539a24a..., RAYON_NUM_THREADS=4, numpy 2.4.6
  (AVX2/FMA3 runtime, no AVX-512), 1139.7s, exit 0.

### Baseline (effect = numpy/fnp medians; null = numpy A/A):

| row | numpy ms | fnp ms | effect | null | verdict |
|---|---:|---:|---:|---:|---|
| f64_sin_262k | 3.232 | 1.817 | 1.813 | 1.000 | fnp already wins |
| f64_sin_1m | 11.436 | 5.399 | 2.094 | 1.013 | win |
| f64_sin_4m | 47.505 | 29.323 | 1.592 | 1.001 | win |
| f64_cos_1m | 11.399 | 5.330 | 2.232 | 1.002 | win |
| f64_tan_1m | 17.563 | 6.911 | 2.560 | 1.001 | win |
| f64_tanh_1m | 5.415 | 7.248 | **0.757** | 1.007 | **LOSS 1.32x — the ranked hotspot** |
| f64_expm1_262k | 3.088 | 1.888 | 1.594 | 1.013 | win |
| f64_expm1_1m | 12.606 | 6.917 | 1.807 | 0.987 | win |

Baseline findings: (1) the 2026-06-22 "medium-N sin/cos/expm1 1.4-1.5x loss"
DEFER row is refuted on a quiet worker — current tree already wins those; the
live loss is tanh (numpy 2.4.6 vectorized f64 tanh, single-threaded, beats
fnp's copy-taxed parallel native path). (2) TRANSCENDENTAL_PARITY probes:
sin/cos/tan/expm1 fnp==numpy BYTE-EXACT on this worker; tanh diverges in
~38-40% of elements TODAY (pre-lever, last-ULP class) -> bead
deadlock-audit-d4mc2. Warning-surface finding -> bead deadlock-audit-tvy7o.

### Method note (negative evidence, tooling)

`cargo fmt -p fnp-python` must NOT be run in this repo: the tree is not
rustfmt-clean (fmt --check is not one of the effective gates; git diff --check
is). A package-wide fmt reformatted ~7k lines across 13 files and had to be
reconstructed from HEAD + the intended edits (whitespace-hash verification is
defeated by rustfmt trailing-comma churn; reconstruction from git show is the
reliable recovery).

- candidate_bench_run2.txt: post-change, same command/group. Worker vmi1149989,
  binary sha256 04badf51..., RAYON_NUM_THREADS=4. Candidate parity probes:
  sin/cos/tan/expm1 byte_equal=true THROUGH THE NEW ROUTE; tanh
  byte_equal=false with EXACTLY the baseline's diff counts
  (99992/399921/1601470, same first-diff indices) on a different worker+binary
  — the route change altered zero output bits.

### Candidate run2 (effect = numpy/fnp, all verdicts WIN, nulls 0.98-1.02):

| row | numpy ms | fnp ms | effect | baseline effect | movement |
|---|---:|---:|---:|---:|---|
| f64_sin_262k | 3.174 | 1.597 | 2.065 | 1.813 | +14% |
| f64_sin_1m | 12.991 | 3.984 | 3.156 | 2.094 | +51% |
| f64_sin_4m | 47.630 | 14.065 | 3.342 | 1.592 | +110% (fnp self 29.3->14.1ms) |
| f64_cos_1m | 12.209 | 4.092 | 2.857 | 2.232 | +28% |
| f64_tan_1m | 17.369 | 5.244 | 3.368 | 2.560 | +32% |
| f64_tanh_1m | 12.015 | 9.172 | **1.372** | **0.757 LOSS** | **FLIPPED** |
| f64_expm1_262k | 3.376 | 1.836 | 1.784 | 1.594 | +12% |
| f64_expm1_1m | 13.366 | 5.462 | 2.460 | 1.807 | +36% |

- candidate_bench_run3.txt: second candidate replication. Worker hz1 (third
  worker class this session), RAYON_NUM_THREADS=4. Parity probes IDENTICAL to
  both prior workers (sin/cos/tan/expm1 byte-equal through the new route; tanh
  diff counts exactly 99992/399921/1601470 again — deterministic, numpy-side).

### Candidate run3, hz1 (effect = numpy/fnp, all verdicts WIN, nulls 0.995-1.004):

| row | numpy ms | fnp ms | effect |
|---|---:|---:|---:|
| f64_sin_262k | 3.291 | 1.894 | 1.746 |
| f64_sin_1m | 13.121 | 7.115 | 1.846 |
| f64_sin_4m | 52.636 | 26.952 | 2.134 |
| f64_cos_1m | 12.437 | 5.754 | 2.174 |
| f64_tan_1m | 19.716 | 6.532 | 3.071 |
| f64_tanh_1m | 14.341 | 9.859 | **1.613** (flip replicated) |
| f64_expm1_262k | 3.507 | 1.468 | 2.386 |
| f64_expm1_1m | 14.143 | 9.591 | 1.482 |

DECISION: SHIP. 16/16 WIN rows across two candidate workers vs pinned in-run
nulls; tanh loss (baseline 0.757) flipped on both (1.372 / 1.613); bit-identity
proven by construction + route-equality conformance + identical cross-worker
parity byte patterns.
- conformance_unary_ops_candidate.txt: route-equality + defer + specials battery.

### Validation (candidate)

- `cargo check -p fnp-python`: GREEN (remote hz2, 68.4s).
- `cargo test -p fnp-python --release --test conformance_unary_ops`: 18/18
  GREEN remote, including the 3 new tests:
  f64_transcendental_zerocopy_route_matches_native_route_and_numpy (array
  route vs list route BYTE equality at 257 + 100,001 elements x 15 ops, plus
  numpy allclose), f64_transcendental_error_inputs_defer_byte_exactly (planted
  invalid/divide/over/under values -> byte-equal to the native reference),
  f64_transcendental_special_values_and_layouts_match (NaN/+-inf/+-0.0 route
  equality, 0-d scalar handling, strided delegation).
- `cargo clippy -p fnp-python --lib -- -D warnings -A dead_code` (dead_code
  allowed for the documented pre-existing fnp-ufunc nan_filtered lint):
  remaining errors are all PRE-EXISTING lib lints (deprecated pyo3 downcast,
  unused vars at lines 367..68540); ZERO findings inside the edited
  7300-7800 region.
- UBS on the changed test/bench files: criticals only at pre-existing lines
  (161/241 of conformance_unary_ops.rs — whole-file heuristic class).
