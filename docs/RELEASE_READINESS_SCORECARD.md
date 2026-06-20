# FrankenNumPy Release-Readiness Scorecard

This is a rolling gauntlet scorecard. It summarizes measured evidence for the
current verification slice and does not certify the whole project for release.

## 2026-06-20 - Batch Cholesky 8-Lane SoA No-Ship Slice

Scope:
- Bead: `franken_numpy-ixs5y.272`.
- Parent bead measured: `franken_numpy-ixs5y`.
- Crate/API: `fnp-linalg::batch_cholesky`.
- Worker proof: `hz1` for same-worker old/new Criterion; `hz2` for bit proof.
- Artifact: `tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_generated_cod_b/`.
- Candidate: temporary 8-lane SoA SIMD layout for `n=16/32/64` batched
  Cholesky groups, intended to avoid the prior per-k gather/scatter SIMD failure.

| Gate | Result | Evidence |
|---|---|---|
| Candidate correctness | PASS | SoA route matched per-lane `cholesky_nxn` bits for d=16/32/64. |
| Candidate performance | FAIL | Same-worker target rows were 1 win / 2 losses: 0.934x, 1.651x, and 1.131x candidate/old. |
| Guardrails | FAIL / NOISY | Non-routed n=128 and n=256 rows were also slower in the same run window: 1.837x and 1.437x candidate/old. |
| NumPy comparison | BLOCKED / NOT COUNTED | Direct same-host Python on `hz1` failed with SSH authentication denial; local Python lacks importable `fnp_python`. |
| Revert discipline | PASS | Candidate source was reverted; `crates/fnp-linalg/src/lib.rs` has no remaining diff. |
| Post-revert correctness | PASS | `rch exec -- cargo test -p fnp-linalg batch_cholesky -- --nocapture` passed 2 tests, 0 failed, 1 ignored. |
| Bench health | PASS | `rch exec -- cargo check -p fnp-linalg --benches` passed with the retained d=16/32/64 batch benchmark rows. |
| Evidence durability | PASS | Baseline/candidate logs, blocked NumPy attempt, compile/test logs, ratios, and retry predicate are recorded. |

Cluster score: **58 / 100**

Score rationale:
- +18 correctness: the candidate's bit-equivalence test passed before revert.
- +16 evidence discipline: same-worker old/new rows and the blocked NumPy
  comparator are recorded.
- +10 revert hygiene: no regressing production source remains.
- +8 benchmark coverage: d=16/32/64 batch Cholesky rows now stay visible in the
  per-crate Criterion harness.
- +6 routing clarity: temporary SoA register layout is ruled out.
- -40 performance: two of three routed rows regressed, and the non-routed
  guardrails were worse in the same run window.

Current release posture:
- `batch_cholesky` / Python stacked `fnp.linalg.cholesky` remains a confirmed
  high-priority NumPy gap.
- Do not retry temporary SoA lane grouping or per-k gather/scatter SIMD. The
  next credible route needs a true packed-panel batched layout, a serial-winning
  safe vector dot primitive for d=32/d=64, or a LAPACK-class blocked per-lane
  kernel with same-host NumPy capture.

## 2026-06-20 - Medium Cholesky Lower-Triangular Threshold No-Ship Slice

Scope:
- Bead: `franken_numpy-ixs5y.271`.
- Parent bead measured: `franken_numpy-ixs5y`.
- Crate/API: `fnp-linalg::batch_cholesky`.
- Worker proof: `vmi1264463`.
- Artifact: `tests/artifacts/perf/2026-06-20_linalg_cholesky_triangular_medium_cod_b/`.
- Candidate: route medium Cholesky trailing updates with `trail >= 64` through
  the existing lower-triangular packed update instead of the full square product.

| Gate | Result | Evidence |
|---|---|---|
| Candidate correctness | PASS | `cholesky_mid_panel` golden guards passed with 2 tests, 0 failed. |
| Candidate performance | FAIL | Same-worker candidate regressed both target rows: 2.078x and 1.524x candidate/baseline. |
| NumPy comparison | BLOCKED / NOT COUNTED | Direct Python attempts to `root@38.242.209.154` and `ubuntu@38.242.209.154` failed with SSH authentication denial; no local Python fallback is counted. |
| Revert discipline | PASS | Candidate source was reverted; `crates/fnp-linalg/src/lib.rs` has no remaining diff. |
| Post-revert correctness | PASS | `rch exec -- cargo test -p fnp-linalg batch_cholesky -- --nocapture` passed 2 tests, 0 failed, 1 ignored. |
| Evidence durability | PASS | Raw baseline/candidate logs, blocked NumPy attempts, validation logs, ratios, and retry predicate are recorded. |

Cluster score: **54 / 100**

Score rationale:
- +18 correctness: candidate golden guards passed and post-revert batch tests
  are green.
- +16 evidence discipline: same-worker old/new rows and failed same-host NumPy
  capture attempts are recorded.
- +10 revert hygiene: no regressing production hunk remains.
- +10 routing clarity: medium-trail threshold lowering is now ruled out.
- -46 performance: both measured target rows regressed and no NumPy keep proof
  was warranted.

Current release posture:
- `batch_cholesky` remains a confirmed high-priority gap.
- Do not retry this as triangular-update threshold lowering. The next credible
  route is generated fixed-size batched panels, a safe SIMD dot primitive that
  beats the scalar panel solve, or a LAPACK-class blocked Cholesky kernel with
  same-host NumPy capture and zero medium-row regressions.

## 2026-06-20 - Python Stacked Cholesky Delegate Keep Slice

Scope:
- Parent bead measured: `franken_numpy-ixs5y`.
- Crate/API: `fnp-python` / `fnp.linalg.cholesky` stacked-SPD arrays.
- Worker proof: same-worker baseline and final candidate on `vmi1152480`.
- Artifact:
  `tests/artifacts/perf/2026-06-20_linalg_cholesky_python_delegate_cod_a/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | Final candidate is 4 wins / 0 material losses / 1 neutral versus NumPy. FNP/NumPy ratios: 4x4 `0.700x`, 8x8 `0.944x`, 16x16 `0.988x`, 32x32 `1.026x` neutral/noisy, 64x64 `0.966x`. |
| Old/new FNP regression gate | PASS | Final candidate is faster than old FNP on every measured row, with New/Old FNP ratios from `0.267x` to `0.837x`. |
| Targeted correctness | PASS | `rch exec -- cargo test -p fnp-python --test conformance_linalg_decomp cholesky -- --nocapture` passed 6/6 after adding the stacked 4x4 SPD route. |
| Crate compile health | PASS WITH WARNINGS | `rch exec -- cargo check -p fnp-python --lib --bench criterion_python_surface` passed; local post-format rerun passed. Three inherited `fnp-python` warnings remain. |
| Release build health | PASS WITH WARNINGS | `rch exec -- cargo build -p fnp-python --release` passed with the same inherited warnings. |
| Clippy health | KNOWN GAP | RCH selected a worker missing clippy for the pinned nightly; local clippy with `-D warnings` failed on broad pre-existing `fnp-python` lint inventory outside this hunk. |
| Formatting health | KNOWN GAP | `cargo fmt -p fnp-python -- --check` fails on broad pre-existing crate drift; the touched Cholesky hunk was manually aligned with rustfmt output. |
| UBS | KNOWN GAP | UBS over the changed files completed nonzero with broad existing findings in `fnp-python`, not a new Cholesky-specific finding. |
| Evidence durability | PASS | Baseline, intermediate rejects, final candidate, conformance, check, release build, clippy/fmt/UBS caveats, ratios, and retry predicates are recorded in the artifact bundle and negative-evidence ledger. |

Cluster score: **84 / 100**

Score rationale:
- +36 performance: the selected stacked-Cholesky loss class moved from 0/5
  wins versus NumPy to 4/0/1, and every row improved versus old FNP.
- +18 correctness: the focused linalg decomposition conformance shard passed
  including the optimized stacked 4x4 route.
- +14 build health: per-crate check and release build pass.
- +12 evidence discipline: same-worker baseline/candidate proof, intermediate
  negative evidence, exact ratios, and retry predicates are recorded.
- +8 source discipline: the route is gated to exact stacked NumPy ndarrays with
  default lower-triangle semantics; upper and unsupported inputs retain fallback
  behavior.
- -4 residual performance: the 32x32 row is a noise-band neutral rather than a
  clean win.
- -20 residual hygiene: broad `fnp-python` lib-test, clippy, rustfmt, and UBS
  debt remains outside this hunk.

Current release posture:
- Python stacked `fnp.linalg.cholesky` at the measured 4x4..64x64 sizes is now
  a measured keep for this slice and no longer a current material NumPy loss.
- `fnp-python` is still not globally release-clean because unrelated lib-test
  compile errors and broad lint/format/UBS debt remain open.

## 2026-06-20 - Small-N Cholesky Ordered-Dot Mixed Slice

Scope:
- Parent bead measured: `franken_numpy-ixs5y`.
- Crate/API: `fnp-linalg::cholesky_nxn`, plus Python `fnp.linalg.cholesky`
  stacked-SPD comparator.
- Source commit under verification: `856c38cb`.
- Worker proof: Rust parent/current pair on `vmi1153651`; Python extension build
  on `vmi1152480`; Python comparator local with the built current-head `.so`.
- Artifact: `tests/artifacts/perf/2026-06-20_linalg_cholesky_right_looking_cod_a/`.

| Gate | Result | Evidence |
|---|---|---|
| Owned Rust performance | PASS | `cholesky_nxn/16` and `/32` improved to 0.870x and 0.879x of parent on `vmi1153651`. |
| Head-to-head performance vs NumPy | FAILING RESIDUAL | Current Python stacked Cholesky remains 1 win / 6 losses vs NumPy; owned d=16 and d=32 rows are still 6.46x and 5.46x slower. |
| Broad regression guardrail | MIXED/NOISY | Non-owned blocked and batch rows were noisy, including apparent losses on paths the helper does not route through. They are recorded but not attributed as causal wins. |
| Correctness | PASS | `rch exec -- cargo test -p fnp-linalg cholesky_ -- --nocapture` passed current-head unit, conformance, golden, metamorphic, and solve rows; Python comparator reported `match=True` for every measured row. |
| Crate compile health | PASS | `rch exec -- cargo check -p fnp-linalg --all-targets` passed. |
| Clippy health | PASS | `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings` passed. |
| Release build health | PASS WITH WARNINGS | `rch exec -- cargo build -p fnp-linalg --release` passed. `rch exec -- cargo build -p fnp-python --release --features python-extension` also passed, with three pre-existing `fnp-python` warnings. |
| Evidence durability | PASS | Parent/current Rust logs, current-head Python comparator, extension build log, win/loss ratios, and retry predicate are in the artifact bundle and negative-evidence ledger. |

Cluster score: **62 / 100**

Score rationale:
- +18 performance: the owned Rust direct rows improved by about 12-13%.
- +16 correctness/build: Python extension build passed and all comparator rows
  matched NumPy.
- +14 evidence discipline: same-worker Rust parent/current proof and Python
  NumPy ratios are recorded.
- +8 source discipline: the helper is gated to n=16..32 and leaves larger
  blocked Cholesky routing alone.
- +6 routing clarity: scalar micro-tuning is now separated from the real
  stacked-SPD NumPy gap.
- -38 residual performance: current Python stacked Cholesky still loses badly to
  NumPy on 6 of 7 rows, including the owned d=16/d=32 cases.

Current release posture:
- `cholesky_nxn` small-N ordered dot is a narrow Rust keep already present in
  `main`, not a release-level NumPy performance closeout.
- `batch_cholesky` / Python stacked `fnp.linalg.cholesky` remains a confirmed
  high-priority gap. Next work should target batched layout or generated
  fixed-size kernels, not another scalar-loop micro-tweak.

## 2026-06-20 - Linalg Column-Norm SIMD Keep Slice

Scope:
- Parent bead measured: `franken_numpy-ixs5y`.
- Crate: `fnp-linalg`.
- Worker proof: `vmi1227854`.
- Artifact: `tests/artifacts/perf/2026-06-20_linalg_column_norm_simd_cod_a/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | Target rows `n>=256` are 6/6 wins vs same-host NumPy, with New/NumPy ratios from 0.255x to 0.334x. |
| Old/new regression gate | PASS WITH NOISY GUARDRAIL | 7/8 observed rows improved vs old FNP; `neg_one/128` moved 1.047x slower but stayed faster than NumPy and is below the SIMD threshold. |
| Targeted correctness | PASS | `rch exec -- cargo test -p fnp-linalg matrix_norm_column_reduction_matches_strided_reference_bits -- --nocapture` passed. |
| Crate compile health | PASS | `rch exec -- cargo check -p fnp-linalg --all-targets` passed. |
| Clippy health | PASS | `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings` passed. |
| Release build health | PASS | `rch exec -- cargo build -p fnp-linalg --release` passed. |
| Formatting health | KNOWN GAP | Crate formatting remains blocked by broad pre-existing rustfmt drift; the touched SIMD line was manually adjusted to rustfmt's reported shape. |
| UBS | KNOWN GAP | `ubs crates/fnp-linalg/src/lib.rs` reports broad pre-existing inventory unrelated to the matrix-norm helper; UBS internal fmt/clippy/check sub-gates were green. |
| Evidence durability | PASS | Win/loss/neutral ratios, NumPy comparator metadata, validation commands, no-ship Cholesky probe, and retry predicates are recorded in `docs/NEGATIVE_EVIDENCE.md` and per-run scorecards. |

Cluster score: **88 / 100**

Score rationale:
- +36 performance: the selected residual moved from losing to NumPy at
  256/512/1024 to a clear same-host win.
- +18 correctness: the bit-reference matrix norm test passed after the final
  selector split.
- +14 compile health: check, clippy, and release build are green crate-scoped.
- +12 evidence discipline: same-worker old/new proof and same-host NumPy
  comparator are recorded with ratios.
- +8 source discipline: the SIMD helper is isolated so sub-256 scalar routing
  stays outside the widened lane.
- -12 residual hygiene: crate-wide fmt and UBS still have broad pre-existing
  debt outside this hunk.

Current release posture:
- `matrix_norm_nxn_orders` 1/-1 rows at 256, 512, and 1024 are **measured
  keep** for this slice and now beat same-host NumPy.
- Cholesky remains a gap; the const-specialization probe was reverted as too
  small/noisy.

## 2026-06-20 - Cholesky Const-Specialization No-Ship Slice

Scope:
- Crate: `fnp-linalg`.
- API: `cholesky_nxn` / `batch_cholesky`.
- Evidence: `tests/artifacts/perf/2026-06-20_linalg_cholesky_const_specialize_cod_a/`.

| Gate | Result | Evidence |
|---|---|---|
| Candidate correctness | PASS | Focused bit-reference test passed while the candidate existed. |
| Candidate compile health | PASS | `cargo check -p fnp-linalg --all-targets` passed while the candidate existed. |
| Candidate performance | NO-SHIP | Owned small-N rows improved only 4.9%-8.1%; broader apparent wins were treated as worker noise because they were not routed through the const-specialized path. |
| NumPy comparison | NOT RUN | The old/new proof was too small/noisy to justify a NumPy keep claim. |
| Revert discipline | PASS | Candidate source and test were removed before commit; no Cholesky source hunk remains. |

Cluster score: **61 / 100**

Current release posture:
- Do not retry small fixed-N Cholesky const specialization.
- Next Cholesky work must be an actual medium-matrix layout/algorithm change
  with same-window Rust and NumPy proof.
## 2026-06-20 - Batch Cholesky Validation-Hoist No-Ship Slice

Scope:
- Parent bead: `franken_numpy-ixs5y`.
- Crate: `fnp-linalg`.
- Evidence: `tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_validation_hoist_cod_b/`.

| Gate | Result | Evidence |
|---|---|---|
| Candidate performance | FAIL | Same-worker `vmi1153651` broad gate regressed both rows: 1.150x and 3.451x candidate/baseline. |
| Candidate correctness compile | PASS | `rch exec -- cargo check -p fnp-linalg --lib` passed before revert. |
| Revert discipline | PASS | Candidate source was reverted; `crates/fnp-linalg/src/lib.rs` has no remaining diff. |
| Post-revert correctness | PASS | `rch exec -- cargo test -p fnp-linalg batch_cholesky_ -- --nocapture` passed 2 tests, 0 failed, 1 ignored. |
| Post-revert build | PASS | `rch exec -- cargo build -p fnp-linalg --release` passed. |
| Ledger discipline | PASS | Negative evidence records win/loss/neutral, old/new ratios, validation, and retry predicate. |

Cluster score: **56 / 100**

Score rationale:
- +18 correctness/build: candidate compiled; post-revert focused tests and
  release build passed.
- +18 evidence discipline: same-worker baseline and candidate rows are stored.
- +10 revert hygiene: no regressing source was kept.
- +10 retry clarity: finite-scan hoisting is now ruled out alongside prior
  allocation, threshold, and f64x4 gather/scatter no-ships.
- -44 performance: the candidate regressed both measured rows and did not earn
  a NumPy rerun.

Current release posture:
- `batch_cholesky` remains a confirmed medium stacked-SPD performance gap.
- Next attempt needs a structurally different Cholesky kernel with same-window
  proof across medium rows and n>=128 rows.

## 2026-06-20 - Python Einsum Reduce-All Current-Head Rerun Slice

Scope:
- Parent bead: `franken_numpy-ixs5y`.
- Crate: `fnp-python`.
- Worker proof: `vmi1293453`.
- Evidence: `tests/artifacts/perf/2026-06-20_python_einsum_reduce_all_cod_b/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | Current-head rerun was 5/0/0 versus NumPy; `einsum_reduce_all_f64_1000` was 0.730x NumPy time. |
| Source discipline | PASS | No source edit was made because the target was already a current win. |
| Routing discipline | PASS | Former near-loss was removed from the active target list and rerouted to deeper measured losers. |
| Evidence durability | PASS | RCH benchmark log and per-run scorecard are stored under the artifact directory. |

Cluster score: **78 / 100**

Score rationale:
- +35 performance: all observed einsum boundary rows beat NumPy on this worker.
- +18 evidence discipline: exact ratios and worker are recorded.
- +15 source discipline: no speculative source change was made after baseline
  invalidated the target gap.
- +10 routing clarity: scalar-builder and diagonal families are not reopened
  without fresh losing evidence.

Current release posture:
- `einsum_reduce_all_f64_1000` is not an active current-head gap from this rerun.
- Continue targeting confirmed losers such as `batch_cholesky`, not this row.
## 2026-06-20 - Python Einsum Trace Scalar-Builder Keep Slice

Scope:
- Parent bead measured: `franken_numpy-ixs5y`.
- Crate: `fnp-python`.
- Worker proof: `vmi1227854`.
- Artifact: `tests/artifacts/perf/2026-06-20_python_einsum_trace_cod_b/`.
- Source commit: `eb64c4d5`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | Trace moved from a prior 1.146x slower residual to 0.775x NumPy time on `vmi1227854`; diagonal remains faster at 0.916x NumPy time. |
| Full observed boundary sweep | MIXED | 4 wins and 1 non-target loss-or-neutral: trace, diagonal, reduce-rows, and reduce-cols win; reduce-all is 1.011x slower than NumPy. |
| Targeted correctness | PASS | `rch exec -- cargo test -p fnp-python --test conformance_einsum` passed 28/28. |
| Crate compile health | PASS WITH WARNINGS | `rch exec -- cargo check -p fnp-python --lib --bench criterion_python_surface` and `rch exec -- cargo build -p fnp-python --release` passed with pre-existing warnings. |
| Clippy health | KNOWN GAP | Crate-scoped clippy remains blocked by broad pre-existing `fnp-python` lint debt; the log does not mention the scalar-builder helper. |
| Formatting health | KNOWN GAP | `cargo fmt -p fnp-python -- --check` reports broad pre-existing rustfmt drift; the touched helper is not in the fmt diff. |
| UBS | TIMED OUT | `ubs crates/fnp-python/src/lib.rs` ran for more than three minutes on the single large file and was interrupted with no emitted finding. |
| Evidence durability | PASS | RCH benchmark, conformance, check, release build, clippy/fmt/UBS caveats, diff check, and per-run scorecard are stored under the artifact directory. |

Cluster score: **84 / 100**

Score rationale:
- +34 performance: the targeted trace residual now beats NumPy on the same RCH
  worker, and the diagonal support row remains a win.
- +18 correctness: focused einsum conformance passed all scalar, trace, view,
  and keyword/path tests.
- +14 reproducibility: same-worker prior residual, candidate RCH proof, target
  directory, and exact commands are recorded.
- +14 ledger discipline: wins, non-target loss, old/new ratio, validation
  caveats, and retry predicate are recorded.
- +4 source discipline: a narrow scalar-builder hunk replaced a temporary 0-D
  ndarray construction path without widening dispatch semantics.
- -16 residual health: `reduce_all` is still a visible Python-boundary near-loss,
  and broad `fnp-python` clippy/fmt/UBS health remains blocked by pre-existing
  inventory.

Current release posture:
- `fnp_einsum_trace_f64_4000` is **measured keep** for this scalar-builder pass.
- `einsum_reduce_all_f64_1000` remains the next visible residual in the observed
  boundary sweep; deeper einsum kernel work should target that or a fresh loser,
  not the superseded diagonal shortcut family.

## 2026-06-20 - Batch Cholesky SIMD-Across-Lanes No-Ship Slice

Scope:
- Crate: `fnp-linalg`.
- API: `batch_cholesky` / Python `fnp.cholesky` stacked SPD matrices.
- Evidence: `tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_simd_cod_a/`.

| Gate | Result | Evidence |
|---|---|---|
| Baseline vs NumPy | FAILING BASELINE | Existing code lost 0/7 measured Python rows vs NumPy 2.4.3, with medium rows 4.67x-19.65x slower. |
| Candidate correctness | PASS | Focused RCH test passed 3 batch_cholesky tests, including bit-exact n=16/32/64 proof against scalar per-lane `cholesky_nxn`. |
| Candidate performance | FAIL | Same-worker RCH Criterion on `ovh-a` regressed `64x128x128` by 45.662% and `16x256x256` by 16.109%. |
| Revert discipline | PASS | Candidate source was removed before commit; no production change kept. |
| Ledger discipline | PASS | Win/loss/neutral ratios, commands, failure reason, and retry predicate recorded in `docs/NEGATIVE_EVIDENCE.md`. |

Cluster score: **58 / 100**

Score rationale:
- +20 correctness: the candidate preserved scalar Cholesky bits on focused tests.
- +20 evidence discipline: baseline, candidate, and retry predicate are recorded.
- +10 revert hygiene: no regressing source was kept.
- +8 reproducibility: RCH target dir and same-worker Criterion rows are recorded.
- -42 performance: the candidate regressed the broad Rust gate and did not earn
  a NumPy rerun.

Current release posture:
- `batch_cholesky` remains a confirmed performance gap for medium stacked SPD
  matrices.
- Do not retry f64x4 gather/scatter SIMD. The next Cholesky attempt needs a
  different layout or blocked batched-panel algorithm with same-window proof
  across medium and n>=128 rows.

## 2026-06-20 - UFunc Boolean-Index Verification Slice

Scope:
- Recent code-first pending backlog measured and closed:
  `franken_numpy-ixs5y.251`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.2.4.
- Same-worker decision host: `vmi1149989`.
- Evidence: `tests/artifacts/perf/2026-06-20_ufunc_boolean_index_vs_numpy_cod_b/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | 2/2 measured rows faster than NumPy; speedups were 2.29x and 2.16x. |
| Noise discipline | PASS | FNP Criterion ran through RCH on `vmi1149989`; NumPy comparator ran directly on the same host and reported host/version/load metadata. |
| Targeted correctness | PASS | Focused `boolean_index` test filter passed 4/4 tests including the golden SHA-256 bit-parity guard. |
| Full crate conformance | PASS | `cargo test -p fnp-ufunc` passed via RCH: 2244 passed, 0 failed, 41 ignored, integration tests green, doctests ignored as expected. |
| Crate compile health | PASS | `cargo check -p fnp-ufunc --all-targets` passed through RCH. |
| Clippy health | PASS | `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed on `vmi1149989` after installing the missing pinned nightly clippy component there. |
| Release build health | PASS | `cargo build -p fnp-ufunc --release` passed through RCH. |
| Formatting health | NOT RERUN | No source files were edited in this verification slice; prior broad formatting drift was outside this evidence closeout. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and the per-run scorecard. |

Cluster score: **90 / 100**

Score rationale:
- +35 performance: both target rows beat NumPy by more than 2x on the same host.
- +20 correctness: focused boolean-index SHA guard and the full `fnp-ufunc`
  suite passed.
- +15 reproducibility: same-host FNP and NumPy timing evidence with host,
  version, and load metadata.
- +15 ledger discipline: wins, invalid probes, validation, retry predicate, and
  artifact path recorded.
- +8 no-source discipline: existing implementation verified without new
  implementation churn.
- -3 residual validation: formatting was not rerun because this closeout edited
  only docs/artifacts/bead state.

Current release posture:
- `franken_numpy-ixs5y.251` is **measured keep**, not pending.
- Large flat F64 `a[mask]` with sparse truthy mask is ahead of NumPy for the
  measured rows; future work should move below this wrapper into mask storage or
  gather-core primitives only if fresh profiles expose a loss.

## 2026-06-20 - UFunc where_nonzero Verification Slice

Scope:
- Recent code-first pending backlog measured and closed: `franken_numpy-ixs5y.247`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.3.5.
- Same-worker decision host: `hz2` / `hetzner2`.
- Evidence: `tests/artifacts/perf/2026-06-20_ufunc_where_nonzero_vs_numpy/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | 2/2 measured rows faster than NumPy; speedups were 4.00x and 6.88x. |
| Noise discipline | PASS | FNP Criterion ran on `hz2`; NumPy comparator ran directly on `hz2` and reported host/version metadata. |
| Targeted correctness | PASS | `where_nonzero_f64_parallel_matches_serial_reference_and_golden_sha256` passed. |
| Full crate conformance | PASS | `cargo test -p fnp-ufunc` passed after the rounded Legendre golden was repaired: 2244 passed, 0 failed, 41 ignored, plus green integration tests and doctests. |
| Crate compile health | PASS | `cargo check -p fnp-ufunc --all-targets` passed through RCH. |
| Clippy health | PASS | `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed on `hz1` after one worker missing-component miss. |
| Release build health | PASS | `cargo build -p fnp-ufunc --release` passed through RCH. |
| Formatting health | KNOWN GAP | `cargo fmt --package fnp-ufunc -- --check` reports broad pre-existing drift outside this slice; the refreshed artifact does not flag the new Legendre row. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and the per-run scorecard. |

Cluster score: **91 / 100**

Score rationale:
- +35 performance: both target rows beat NumPy decisively.
- +20 correctness: focused where_nonzero SHA guard and the full `fnp-ufunc`
  suite passed.
- +15 reproducibility: same-worker FNP and NumPy evidence.
- +15 ledger discipline: all rows, validation, retry predicate, and artifact
  path recorded.
- +8 test hygiene: the full-suite failure was traced to a rounded NumPy golden
  row and repaired with full f64 coefficients.
- -2 residual validation: format drift remains in untouched `fnp-ufunc`
  regions and was not normalized in this slice.

Current release posture:
- `franken_numpy-ixs5y.247` is **measured keep**, not pending.
- Large F64 2-D `where_nonzero` coordinate gather is ahead of NumPy for the
  measured sparse shapes; future work should target a distinct coordinate
  primitive rather than retuning this same chunk-gather family.

## 2026-06-20 - Linalg Kron Identity Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.236`.
- Crate: `fnp-linalg`.
- Reference: NumPy 2.3.5.
- Same-worker decision host: `hz2` / `hetzner2`.
- Evidence: `tests/artifacts/perf/2026-06-20_linalg_kron_identity_vs_numpy/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | 2/2 measured rows faster than NumPy; speedups were 5.72x and 3.72x. |
| Noise discipline | PASS | FNP Criterion ran on `hz2`; NumPy comparator ran directly on `hz2` and reported host/version metadata. |
| Targeted correctness | PASS | `cargo test -p fnp-linalg kron_ -- --nocapture` passed 4 focused kron tests. |
| Crate compile health | PASS | Same no-source linalg tree passed `cargo check -p fnp-linalg --all-targets` in the preceding verification slice. |
| Clippy health | PASS | Same no-source linalg tree passed `cargo clippy -p fnp-linalg --all-targets -- -D warnings`. |
| Release build health | PASS | Same no-source linalg tree passed `cargo build -p fnp-linalg --release`. |
| Formatting health | KNOWN GAP | `cargo fmt --package fnp-linalg -- --check` reports broad pre-existing drift outside this no-source verification slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and the per-run scorecard. |

Cluster score: **86 / 100**

Score rationale:
- +35 performance: both target rows beat NumPy decisively.
- +20 correctness: focused identity, fallback, scalar, and golden kron tests
  passed.
- +15 reproducibility: same-worker FNP and NumPy evidence.
- +15 ledger discipline: all rows, validation, retry predicate, and artifact
  path recorded.
- +3 no-source discipline: existing implementation verified without extra churn.
- -2 residual validation: format drift remains in untouched `fnp-linalg`
  regions and was not normalized in this slice.

Current release posture:
- `franken_numpy-ixs5y.236` is **measured keep**, not pending.
- Structured identity-RHS `kron_nxn` is ahead of NumPy for the measured block
  operator shapes; generic kron and other structured RHS classes remain separate
  future work.

## 2026-06-20 - Linalg Batched Column-Sum Norm Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.240`.
- Crate: `fnp-linalg`.
- Reference: NumPy 2.3.5.
- Same-worker decision host: `hz2` / `hetzner2`.
- Evidence: `tests/artifacts/perf/2026-06-20_linalg_batch_column_sum_vs_numpy/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | 4/4 measured rows faster than NumPy; speedups ranged from 9.06x to 12.58x. |
| Noise discipline | PASS | FNP Criterion ran on `hz2`; NumPy comparator ran directly on `hz2` and reported host/version metadata. |
| Targeted correctness | PASS | `batch_matrix_norm_column_sum_direct_lane_fill_matches_per_lane_reference_bits` passed. |
| Crate compile health | PASS | `cargo check -p fnp-linalg --all-targets` passed through RCH after one worker-local `SIGILL` retry. |
| Clippy health | PASS | `cargo clippy -p fnp-linalg --all-targets -- -D warnings` passed through RCH. |
| Release build health | PASS | `cargo build -p fnp-linalg --release` passed through RCH. |
| Formatting health | KNOWN GAP | `cargo fmt --package fnp-linalg -- --check` reports broad pre-existing drift outside this no-source verification slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and the per-run scorecard. |

Cluster score: **88 / 100**

Score rationale:
- +40 performance: all target rows beat NumPy decisively.
- +20 correctness: focused bit-preservation guard passed.
- +15 reproducibility: same-worker FNP and NumPy evidence, with cross-host and
  unsupported Python-offload attempts excluded from scoring.
- +15 ledger discipline: all rows, invalid probes, validation, retry predicate,
  and artifact path recorded.
- -2 residual validation: format drift remains in untouched `fnp-linalg`
  regions and was not normalized in this slice.

Current release posture:
- `franken_numpy-ixs5y.240` is **measured keep**, not pending.
- This slice further improves confidence in stacked matrix norm diagnostics; it
  is not project-wide release certification.

## 2026-06-20 - Linalg Batched Row-Sum Norm Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.239`.
- Crate: `fnp-linalg`.
- Reference: NumPy 2.3.5.
- Same-worker decision host: `hz1` / `hetzner1`.
- Evidence: `tests/artifacts/perf/2026-06-20_linalg_batch_row_sum_vs_numpy/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | 4/4 measured rows faster than NumPy; speedups ranged from 7.29x to 12.40x. |
| Noise discipline | PASS | FNP Criterion ran on `hz1`; NumPy comparator ran directly on `hz1` and reported host/version metadata. |
| Targeted correctness | PASS | `batch_matrix_norm_row_sum_direct_lane_fill_matches_per_lane_reference_bits` passed. |
| Crate compile health | PASS | `cargo check -p fnp-linalg --all-targets` passed through RCH. |
| Clippy health | PASS | `cargo clippy -p fnp-linalg --all-targets -- -D warnings` passed through RCH. |
| Release build health | PASS | `cargo build -p fnp-linalg --release` passed through RCH. |
| Formatting health | KNOWN GAP | `cargo fmt -p fnp-linalg -- --check` reports broad pre-existing drift outside this no-source verification slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and the per-run scorecard. |

Cluster score: **88 / 100**

Score rationale:
- +40 performance: all target rows beat NumPy decisively.
- +20 correctness: focused bit-preservation guard passed.
- +15 reproducibility: same-worker FNP and NumPy evidence, with the invalid
  `rch exec -- python3` comparator attempt excluded.
- +15 ledger discipline: all rows, validation, retry predicate, and artifact
  path recorded.
- -2 residual validation: format drift remains in untouched `fnp-linalg`
  regions and was not normalized in this slice.

Current release posture:
- `franken_numpy-ixs5y.239` is **measured keep**, not pending.
- This slice improves confidence in stacked matrix norm diagnostics; it is not
  project-wide release certification.

## 2026-06-19 - Ufunc Flatnonzero Rejection Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.245` via
  cod-a verification bead `franken_numpy-ixs5y.260`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.4.3.
- Same-host decision machine: `thinkstation1`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | MIXED, REJECT CANDIDATE | Candidate was 1.07x slower at 100k and 3.57x faster at 1M; the 100k regression failed the keep gate. |
| Final code performance | PASS | Post-revert serial sidecar path is 3.21x faster than NumPy at 100k and 3.18x faster at 1M. |
| Revert discipline | PASS | Removed the `.245` parallel flatnonzero production fast path after measurement. |
| Targeted correctness | PASS | Pre-revert and post-revert flatnonzero golden guards passed remotely through `rch`. |
| Crate compile health | PASS | `rch exec -- cargo check -p fnp-ufunc --all-targets` passed with `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`. |
| Clippy health | PASS | `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed. |
| Formatting health | KNOWN GAP | `cargo fmt --check` and `cargo fmt -p fnp-ufunc -- --check` still report broad pre-existing format drift outside this slice. |
| UBS health | KNOWN GAP | UBS reports the established broad `fnp-ufunc` inventory; no references to the touched flatnonzero lines were reported. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and `tests/artifacts/perf/2026-06-19_ufunc_flatnonzero_vs_numpy/`. |

Cluster score: **78 / 100**

Score rationale:
- +20 correctness: candidate and post-revert golden guards passed.
- +18 performance: final code beats NumPy on both measured rows, but the
  measured candidate itself had a 100k regression.
- +15 reproducibility: same-host local FNP vs NumPy evidence is recorded, with
  remote correctness evidence kept separate.
- +15 ledger discipline: candidate loss/win, final-code rows, and retry
  predicate are recorded.
- +15 revert discipline: mixed/regressing production fast path removed.
- -5 residual validation: fmt and UBS still have broad pre-existing
  `fnp-ufunc` inventory outside this slice.

Current release posture:
- `franken_numpy-ixs5y.245` is **measured rejected for the parallel branch**, not
  pending.
- The serial exact int64 sidecar export path remains the accepted final code for
  this workload and is faster than NumPy in the measured rows.

## 2026-06-19 - Ufunc Extract Rejection Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.244` via
  cod-a verification bead `franken_numpy-ixs5y.259`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.4.3.
- Same-host decision machine: `thinkstation1`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | FAIL | Candidate was 2.18x slower at 100k and 1.22x slower at 1M. |
| Revert discipline | PASS | Removed the `.244` parallel extract production fast path after measurement. |
| Targeted correctness | PASS | Post-revert extract and boolean-index golden guards passed remotely through `rch`. |
| Crate compile health | PASS | `rch exec -- cargo check -p fnp-ufunc --all-targets` passed with `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`. |
| Clippy health | PASS | `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed. |
| Formatting health | KNOWN GAP | `cargo fmt --check` and `cargo fmt -p fnp-ufunc -- --check` still report broad pre-existing format drift in untouched regions. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and `tests/artifacts/perf/2026-06-19_ufunc_extract_vs_numpy/`. |

Cluster score: **62 / 100**

Score rationale:
- +20 correctness: candidate and post-revert golden guards passed.
- +14 reproducibility: same-host local FNP vs NumPy evidence is recorded, with
  remote correctness/build evidence kept separate.
- +15 ledger discipline: reject, final-code gap, and retry predicate are recorded.
- +15 revert discipline: losing production fast path removed.
- -38 performance: the measured candidate lost to NumPy on both rows; the final
  serial path is still slower than NumPy at 1M.
- -4 formatting residual: package/workspace fmt drift remains outside this
  slice.

Current release posture:
- `franken_numpy-ixs5y.244` is **measured rejected**, not pending.
- `extract(condition, arr)` is acceptable only as the reverted serial path for
  this slice; the 1M sparse-mask row remains an open performance gap versus
  NumPy.

## 2026-06-19 - Random PCG Distribution Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.250` and
  `franken_numpy-ixs5y.253`.
- Crate: `fnp-random`.
- Reference: NumPy 2.4.3.
- Worker: `ovh-a` via `rch exec`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | 4/4 measured size rows faster than NumPy median; speedups ranged from 6.01x to 8.67x. |
| Noise discipline | PASS | Criterion `bencher` output recorded all rows with sample size 10 and the same worker for FNP and NumPy rows. |
| Targeted correctness | PASS | PCG gumbel/laplace stream-state guards and live NumPy oracle checks passed. |
| Crate bench compile health | PASS | `rch exec -- cargo check -p fnp-random --benches` passed with requested `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`. |
| Revert decision | PASS | No revert required; no measured row was neutral or regressed in this slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` with retry predicates and artifacts under `tests/artifacts/perf/2026-06-19_random_vs_numpy_pcg_distributions/`. |

Cluster score: **90 / 100**

Score rationale:
- +40 performance: all target rows beat NumPy by at least 6.01x.
- +20 correctness: stream-state guards and live NumPy oracle checks passed.
- +15 reproducibility: same worker and explicit target-dir rewrite recorded.
- +15 ledger discipline: every result and retry predicate recorded.
- -10 project-wide release gap: this is not a full workspace gauntlet, full
  conformance, or 10-round convergence run.

## 2026-06-19 - Ufunc Data-Movement Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.256` and
  `franken_numpy-ixs5y.258`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.4.3.
- Worker: `thinkstation1` via `rch exec`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | 4/4 measured size rows faster than NumPy median; speedups ranged from 1.19x to 1.73x. |
| Noise discipline | PARTIAL PASS | 3/4 batched NumPy rows were at or near the 5% CV gate; 1M insert stayed noisy but still had NumPy minimum slower than FNP Criterion upper CI. |
| Targeted correctness | PASS | Both new golden SHA guards ran and passed with real test execution. |
| Crate compile health | PASS | `rch exec -- cargo check -p fnp-ufunc` passed with `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`. |
| Revert decision | PASS | No revert required; no measured row was neutral or regressed in this slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` with retry predicates. |

Cluster score: **82 / 100**

Score rationale:
- +35 performance: all target rows beat NumPy medians, with one weak-but-positive
  delete row and one noisy-but-positive insert row.
- +20 correctness: targeted golden guards passed.
- +15 reproducibility: same rch worker and explicit target dir recorded.
- +12 ledger discipline: every result, discarded noisy attempt, and retry
  predicate recorded.
- -18 project-wide release gap: this is not a full workspace gauntlet, full
  conformance, or 10-round convergence run.

Current release posture:
- `fnp-ufunc` data-movement cluster is **measured keep** for the verified rows.
- Project-wide release certification remains **not certified** from this slice
  alone; continue converting `code-first batch-test pending` beads into measured
  ledger entries before claiming global performance dominance.

## 2026-06-19 - Ufunc Compress Rejection Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.249`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.4.3.
- Same-host decision machine: `thinkstation1`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | FAIL | Candidate was 7.15x slower at 100k and 2.05x slower at 1M. |
| Revert discipline | PASS | Removed the `.249` parallel compress production fast path after measurement. |
| Targeted correctness | PASS AFTER GUARD FIX | Bitwise guard passed remotely after replacing a `NaN`-unsafe assertion; post-revert compress tests passed locally. |
| Crate compile health | PASS | `cargo check -p fnp-ufunc` passed with `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`. |
| Clippy health | PASS | `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed after replacing an approximate `2/pi` literal with `std::f64::consts::FRAC_2_PI`. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and `tests/artifacts/perf/2026-06-19_ufunc_selection_vs_numpy/`. |

Cluster score: **64 / 100**

Score rationale:
- +20 correctness: candidate guard passed after the test fix and post-revert tests passed.
- +14 reproducibility: same-host local FNP vs NumPy evidence is recorded, with remote routing evidence kept separate.
- +15 ledger discipline: reject, final-code gap, and retry predicate are recorded.
- +15 revert discipline: regressing production fast path removed.
- -36 performance: the measured candidate lost to NumPy and regressed local Criterion history.

Current release posture:
- `franken_numpy-ixs5y.249` is **measured rejected**, not pending.
- `compress(condition, axis=None)` remains an open performance gap versus NumPy.

## 2026-06-19 - Ufunc Count Nonzero Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.246`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.4.3.
- Same-host decision machine: `thinkstation1`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS AFTER NARROWING | Original 16k activation was 3.56x slower than NumPy at 100k; final threshold was 4.69x faster at 100k and 3.48x faster at 1M. |
| Revert discipline | PASS | The regressing 16k activation threshold was removed; final code raises activation to `1 << 19` and keeps 4096-element chunks for large rows. |
| Targeted correctness | PASS | `count_nonzero_f64_parallel_matches_serial_reference_and_golden_sha256` passed after the intentional threshold-fixture digest update. |
| Crate compile health | PASS | `cargo check -p fnp-ufunc` passed with `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`. |
| Clippy health | PASS | `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed. |
| Formatting health | WARN | `cargo fmt --check` still reports broad pre-existing workspace formatting drift outside this slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and `tests/artifacts/perf/2026-06-19_ufunc_count_nonzero_vs_numpy/`. |

Cluster score: **84 / 100**

Score rationale:
- +34 performance: the final code beats NumPy on both measured rows, but the
  first measurement found a real 100k regression that had to be narrowed away.
- +20 correctness: targeted golden guard passed after the intentional fixture
  digest update.
- +15 reproducibility: same-host FNP and NumPy timings and explicit target dir
  are recorded.
- +15 ledger discipline: every win, loss, weakened neutral-ish correction, and
  retry predicate is recorded.
- -16 project-wide release gap: this is one verified pending optimization, not a
  full workspace gauntlet or global release certification.

Current release posture:
- `franken_numpy-ixs5y.246` is **measured keep after narrowing**, not pending.
- Continue converting the remaining batch-test backlog into measured rows before
  claiming broader `fnp-ufunc` performance readiness.

## 2026-06-19 - Ufunc Argwhere Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.248`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.4.3.
- Same-host decision machine: `thinkstation1`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | Final code was 3.04x faster at 512x512 and 4.79x faster at 1024x1024. |
| Noise discipline | PASS WITH CAVEAT | NumPy CV was above 10%, but NumPy minimum still exceeded the FNP Criterion upper bound on both rows. |
| Targeted correctness | PASS | `argwhere_f64_parallel_matches_serial_reference_and_golden_sha256` passed. |
| Crate compile health | PASS | `cargo check -p fnp-ufunc` passed with `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`. |
| Clippy health | PASS | `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed. |
| Formatting health | WARN | `cargo fmt --check` still reports broad pre-existing workspace formatting drift outside this slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and `tests/artifacts/perf/2026-06-19_ufunc_argwhere_vs_numpy/`. |

Cluster score: **86 / 100**

Score rationale:
- +38 performance: both measured rows clear NumPy by more than 3x.
- +18 correctness: targeted interleaved-coordinate golden guard passed.
- +15 reproducibility: same-host FNP and NumPy timings and explicit target dir
  are recorded.
- +15 ledger discipline: every row and retry predicate is recorded.
- -14 project-wide release gap: this is one verified pending optimization, not a
  full workspace gauntlet or global release certification.

Current release posture:
- `franken_numpy-ixs5y.248` is **measured keep**, not pending.
- Continue converting the remaining batch-test backlog into measured rows before
  claiming broader `fnp-ufunc` performance readiness.

## 2026-06-20 - Linalg Column Norm Reject Slice

Scope:
- Parent bead measured: `franken_numpy-ixs5y`.
- Crate: `fnp-linalg`.
- Worker: `hz2`.
- Artifact: `tests/artifacts/perf/2026-06-20_linalg_column_norm_prefilter_stack256/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | FAIL | Existing final code still loses to NumPy on the 256-1024 `ord=1/-1` matrix norm rows; the rejected candidates were 1.047x-1.848x NumPy time. |
| Candidate vs current FNP | FAIL | Whole-matrix NaN prefilter regressed all target rows; stack256-only regressed `neg_one/512` by 1.056x and `one/1024` by 1.018x while giving only a small `neg_one/256` gain. |
| Targeted correctness | PASS | `matrix_norm_column_reduction_matches_strided_reference_bits` passed for both candidates. |
| Revert discipline | PASS | Both production source changes were removed after measurement. |
| Evidence durability | PASS | No-ship table is recorded in `docs/NEGATIVE_EVIDENCE.md` and the artifact scorecard. |

Cluster score: **54 / 100**

Score rationale:
- +20 correctness: focused column-reduction reference test passed.
- +14 reproducibility: same-worker `hz2` FNP and NumPy rows are recorded with crate-scoped RCH commands.
- +15 ledger discipline: both failed levers and retry predicates are recorded.
- +15 revert discipline: no regressing source remains in production.
- -10 performance: no kept win; the 256-1024 column-norm gap remains open.

Current release posture:
- The column-norm residual remains **open**.
- Do not retry whole-matrix NaN prefilter or stack256-only. A future keep needs SIMD or strip-mined multi-column accumulation with same-worker proof against NumPy and no target-row regressions.

## 2026-06-20 - Python Einsum Diagonal Keep Slice

Scope:
- Parent bead measured: `franken_numpy-ixs5y`.
- Crate: `fnp-python`.
- Worker proof: final remote bench on `vmi1227854`; local baseline/final in `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Artifact: `tests/artifacts/perf/2026-06-20_python_einsum_diag_cod_a/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS FOR DIAGONAL | Local final diagonal is 0.808x NumPy time; rch final diagonal is 0.905x NumPy time. |
| Candidate vs current FNP | PASS | Local final diagonal is 0.193x baseline FNP time; local trace is 0.830x baseline FNP time. |
| Negative evidence discipline | PASS | Two intermediate candidates and the rch trace residual are recorded in `docs/NEGATIVE_EVIDENCE.md`. |
| Targeted correctness | PASS | `rch exec -- cargo test -p fnp-python --test conformance_einsum` passed 28/28. |
| Crate compile health | PASS WITH WARNINGS | `cargo check -p fnp-python --lib --bench criterion_python_surface` passed locally and `rch exec -- cargo build -p fnp-python --release` passed on `vmi1149989`; pre-existing warnings remain. |
| Formatting health | WARN | `cargo fmt -p fnp-python -- --check` reports broad pre-existing formatting drift; no formatter was run because it would rewrite unrelated files. |
| Broader all-target health | WARN | `cargo check -p fnp-python --benches` reaches unrelated pre-existing lib-test call-site drift. |
| Evidence durability | PASS | Baseline, candidates, rch conformance, final rch bench, and artifact scorecard are stored under the artifact directory. |

Cluster score: **82 / 100**

Score rationale:
- +32 performance: the target diagonal row now beats NumPy locally and on rch; trace improves locally but remains slower than NumPy on the rch worker.
- +18 correctness: focused einsum conformance and writable-view golden tests passed.
- +15 reproducibility: local baseline/final and final rch paired FNP-vs-NumPy rows are recorded.
- +15 ledger discipline: every candidate and residual loss is recorded.
- +2 integration hygiene: compile passes for the non-test library and exact bench target, but pre-existing warnings and drift block broader clean claims.
- -18 project-wide release gap: this is one verified Python-boundary keep, not a full workspace release certification.

Current release posture:
- `fnp_einsum_diag_f64_4000` is **measured keep**, not pending.
- The rch trace residual is **not closed**; future work should only target trace if it removes remaining scalar construction or view/fallback overhead without weakening diagonal writability semantics.

## 2026-06-20 - Linalg Column Norm Strip-Mine Reject Slice

Scope:
- Parent bead measured: `franken_numpy-ixs5y`.
- Crate: `fnp-linalg`.
- Artifact: `tests/artifacts/perf/2026-06-20_linalg_column_norm_stripmine_cod_a/`.
- Candidate: 8-column strip-mined cache-linear column accumulation for `matrix_norm_nxn_orders/(one|neg_one)/(256|512|1024)`.

| Gate | Result | Evidence |
|---|---|---|
| Candidate vs current FNP | FAIL | `vmi1149989` showed 5 wins / 1 loss / 0 neutral; `neg_one/256` regressed 1.037x. |
| Head-to-head performance vs NumPy | FAIL | Direct same-host NumPy capture was blocked by SSH auth; local NumPy was routing-only and mixed, while repeat `hz1` candidate rows lost 1.238x-1.653x against the available `hz1` NumPy context. |
| Targeted correctness | PASS | `matrix_norm_column_reduction_matches_strided_reference_bits` passed on `hz1`. |
| Revert discipline | PASS | The production source hunk was removed after measurement. |
| Evidence durability | PASS | Raw RCH logs, failed NumPy capture attempts, artifact scorecard, and this ledger entry are stored. |

Cluster score: **51 / 100**

Score rationale:
- +18 correctness: the focused bit-reference guard passed.
- +12 reproducibility: raw RCH logs capture selected workers, target-dir rewriting, and exact command lines.
- +15 ledger discipline: wins, losses, blocked NumPy proof, and retry predicate are recorded.
- +15 revert discipline: no mixed or regressing production hunk remains.
- -9 performance: the only same-worker Rust delta had a regression and the repeat worker context rejected robustness.

Current release posture:
- The column-norm residual remains **open**.
- Do not retry scalar manual strip mining. The next credible route is a real SIMD lane or generated size-specialized column microkernel with same-host NumPy capture and zero target-row regressions.

## 2026-06-20 - Batch Cholesky Blocked Ordered-Dot Reject Slice

Scope:
- Parent bead measured: `franken_numpy-ixs5y`; child bead: `franken_numpy-ixs5y.270`.
- Crate: `fnp-linalg`.
- Artifact: `tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_ordered_dot_cod_b/`.
- Candidate: extend ordered 4-wide scalar dot helpers into blocked Cholesky diagonal and panel updates for `batch_cholesky`.

| Gate | Result | Evidence |
|---|---|---|
| Candidate vs current FNP | FAIL | Same-worker `vmi1153651` Criterion was mixed: `64x128x128` improved to 0.914x baseline, but `16x256x256` regressed to 1.064x baseline. |
| Head-to-head performance vs NumPy | BLOCKED / NOT COUNTED | Direct Python on `vmi1153651` failed with SSH authentication denial; `rch exec -- python3` ran locally on `thinkstation1`, so no same-host NumPy ratio is counted. |
| Targeted correctness | PASS | `rch exec -- cargo test -p fnp-linalg cholesky_ -- --nocapture` passed on `vmi1153651`, including the Cholesky golden and metamorphic filtered tests. |
| Crate compile/lint health | PASS | `rch exec -- cargo check -p fnp-linalg --all-targets` passed on `vmi1149989`; `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings` passed on `vmi1227854`. |
| Formatting health | WARN | `cargo fmt -p fnp-linalg -- --check` reports pre-existing formatting drift across `fnp-linalg` benches/examples and `src/lib.rs`; no formatter was run to avoid unrelated churn. |
| Revert discipline | PASS | The candidate source hunk was removed; `crates/fnp-linalg/src/lib.rs` has no cod-b diff after the run. |
| Evidence durability | PASS | The no-ship table is recorded in `docs/NEGATIVE_EVIDENCE.md` and the artifact scorecard. |

Cluster score: **49 / 100**

Score rationale:
- +18 correctness: focused Cholesky unit, golden, and metamorphic filters passed.
- +10 reproducibility: same-worker RCH baseline and candidate numbers are recorded, but raw local `tee` artifacts were not preserved by the RCH wrapper.
- +15 ledger discipline: the win, loss, blocked NumPy comparator, and retry predicate are recorded.
- +10 crate health: `fnp-linalg` check and clippy pass.
- +15 revert discipline: no mixed-regression source remains.
- -9 performance: one target row regressed, and no same-host NumPy comparator was captured.
- -10 formatting drift: existing `fnp-linalg` rustfmt drift prevents a clean format gate for this slice.

Current release posture:
- `batch_cholesky` remains **open**.
- Do not retry allocation elimination, gate tuning, validation hoist, small const specialization, f64x4 across-lane gather/scatter, or the blocked-path ordered scalar helper as standalone levers. The next credible route is a real blocked/batched panel kernel, generated size-specialized microkernel, or safe SIMD dot primitive with same-host NumPy capture and zero target-row regressions.

## 2026-06-20 - Batch Cholesky Direct-Write n=16/32 Keep Slice

Scope:
- Parent bead measured: `franken_numpy-ixs5y`; child bead: `franken_numpy-ixs5y.273`.
- Crate: `fnp-linalg`.
- Artifact: `tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_direct_write_cod_b/`.
- Candidate: route `batch_cholesky` for `n <= 32` through direct output writes, using the ordered scalar dot helper for bit identity with `cholesky_nxn`.

| Gate | Result | Evidence |
|---|---|---|
| Affected-row performance vs current FNP | PASS | Same-worker `vmi1227854`: `2000x16x16` improved to 0.786x baseline; `1000x32x32` improved to 0.716x baseline. Repeat candidate routing rows were 0.841x and 0.751x. |
| Head-to-head performance vs NumPy | PASS | Same-host `vmi1227854` candidate rows are 0.183x and 0.239x NumPy on the affected sizes; all five measured rows remain faster than NumPy. |
| Guard-row accounting | WARN | The measured `n>=64` guard rows were slower than the paired baseline, but this patch is gated to `n <= 32`, so those losses are recorded as noisy shared-worker evidence rather than attributed source regressions. |
| Targeted correctness | PASS | `batch_cholesky_scratch_matches_per_lane_cholesky_nxn_bits` passed remotely after extending the cases through `n=16` and `n=32`. |
| Crate compile/lint health | PASS | `rch exec -- cargo check -j 1 -p fnp-linalg --all-targets` and `rch exec -- cargo clippy -j 1 -p fnp-linalg --all-targets -- -D warnings` both passed on `vmi1149989`. |
| Formatting health | WARN | `cargo fmt -p fnp-linalg -- --check` still fails on broad pre-existing `fnp-linalg` rustfmt drift; no formatter was run. `git diff --check` passed. |
| UBS scan | WARN | `ubs` on the changed files exited nonzero on the existing broad `fnp-linalg/src/lib.rs` inventory; the scan was captured in `ubs_changed_files.txt` and did not isolate a new candidate-specific finding. |
| Evidence durability | PASS | Baselines, candidates, NumPy comparators, source patch, validation logs, and the artifact scorecard are stored under the artifact directory; ledger row includes every win/loss/neutral ratio. |

Cluster score: **78 / 100**

Score rationale:
- +24 performance: the two branch-affected rows are clear same-worker wins and both dominate same-host NumPy.
- +18 correctness: direct-write output remains bit-identical to per-lane `cholesky_nxn` for all tested small sizes including `16/32`.
- +14 reproducibility: a paired `vmi1227854` baseline/candidate/NumPy table is captured, but worker selection required retries and `vmi1153651` remained too noisy for a paired keep proof.
- +15 ledger discipline: the guard losses, unpaired baselines, worker-selection issue, and retry predicate are recorded.
- +12 crate health: focused test, check, and clippy pass remotely.
- -5 formatting: pre-existing rustfmt drift still blocks a clean format gate.

Current release posture:
- `batch_cholesky` direct-write small-matrix path is **measured keep** through `n=32`.
- The Python stacked Cholesky and `n>=64` linalg lanes remain open for separate branch-specific work.
