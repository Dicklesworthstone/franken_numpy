# Reality-Check Audit: README.md + AGENTS.md Claims vs Live Tree

**Date:** 2026-09-03
**Auditor:** Main (omp session)
**Tree state:** HEAD `343a09f0` (2026-09-03), shared multi-agent checkout — counts are a snapshot; the fleet lands changes continuously.
**Method:** Static verification only (no builds — 59G disk rule): `git ls-files` + `rg` counts, workspace manifests, `br` tracker, `gh run` job results, one live Python probe. Same methodology as the stated counting commands in both documents wherever they state one.

## Summary

Audited every machine-checkable claim in README.md (2,461 lines) and AGENTS.md against the live tree, tracker, and GitHub CI. **26 claim classes verified exact** (including the CI status block, verified job-by-job against run 33719040154). **12 gaps filed as beads** — all documentation drift, no code defects found by this pass. The three headline architectural claims hold: (1) 9/10 crates carry `#![forbid(unsafe_code)]` with `fnp-python` the lone exception, (2) the structural surface lock `fnp_python_covers_full_numpy_all` exists and runs against live numpy, (3) CI G1/G9 green, G2 red on ledger hygiene, G3–G8 blocked — exactly as the README badge and status block state.

---

## Gaps Found

### GAP-001: `pub fn` count stale in both documents

**Claim:** 1,575 (README lines 325, 2271; AGENTS "1,575 `pub fn` declarations"), methodology "under `crates/*/src/**/*.rs`, excluding `src/bin/`".
**Reality:** **1,626** by the identical methodology (`git ls-files crates | src/ minus /src/bin/`, `rg -c 'pub fn '`).
**Delta:** +51 since the count was banked.
**Severity:** Medium (both docs quote it as a verified live count).

### GAP-002: README Test Coverage section contradicts itself on conformance shard count

**Claim:** Line 1675: "236 integration test files … `fnp-python` 137 (**133 conformance shards** + metamorphic_array_ops + e2e_workflow + golden_native_functions + 1 helper)". Line 1680 (same section): "…**191 dedicated conformance shards**". AGENTS: "133 dedicated `conformance_*.rs` parity shards".
**Reality:** **192** `conformance_*.rs` under `crates/fnp-python/tests/`; **198** files total in that directory; **240** integration test files under `crates/*/tests/` (nested included, matching the stated `find` methodology).
**Severity:** Medium (internal contradiction inside one section; the 133/137 numbers are from May).

### GAP-003: AGENTS.md "Current state" block is a stale snapshot

**Claim (AGENTS, dated 2026-05-16):** 6,392 tests; ~1,417 closed beads (2026-05-19); 133 shards; `codebase_hygiene.rs` "8 #[test] functions"; cost-note "fnp-ufunc (2,191 tests) … fnp-python (2,127 tests)".
**Reality:** **8,691** tests; **2,786** closed beads; **192** shards; hygiene file carries **13** `#[test]` functions; fnp-ufunc **2,472**, fnp-python **3,637**. README line 1626 also still says the hygiene file has "8 #[test] functions" — stale in both docs.
**Severity:** Medium (the block presents May numbers under a "Current state" header).

### GAP-004: Workspace dependency pins drifted (README line 403)

**Claim:** serde 1.0.228 (3 consumers), serde_json 1.0.149 (3), criterion 0.6 with html_reports (7 consumer bench crates), pyo3 0.28.3 (1).
**Reality (root Cargo.toml):** serde **1.0.229**, serde_json **1.0.151**, criterion **0.8.2**, pyo3 0.28.3 ✓. Consumer counts: serde/serde_json 3 ✓ each; criterion **9** crates reference it (all but fnp-runtime).
**Severity:** Medium (pinned-version claims are the doc's own precision contract).

### GAP-005: Bench inventory stale by ~9x

**Claim:** README line 1951: "7 bench files" with a named list (no fnp-python benches listed).
**Reality:** **61** bench `.rs` files across 9 crates (59 `[[bench]]` targets + 2 helpers), including 25 `criterion_python_*.rs` targets + `benches/common/mod.rs` in fnp-python — the harness AGENTS.md's entire perf section depends on.
**Severity:** Medium (understates the perf-evidence surface; list omits the crate AGENTS.md references most).

### GAP-006: Fuzz counts inconsistent across the docs

**Claim:** README Fuzzing section + `docs/FUZZING.md`: "7 fuzz crates, **27** fuzz targets, **~200** curated seed corpus files (as of 2026-05-16/17)". README line 86: "**30** fuzz targets".
**Reality:** 7 crates ✓; **30** targets (table drift: fnp-linalg 3→**5**, fnp-ufunc 3→**4**; others match); **261** corpus files.
**Severity:** Low-Medium (line 86 is already correct; section + FUZZING.md lag).

### GAP-007: "13 registered classes" is wrong — 12 are registered

**Claim:** README line 416: "13 registered classes". Lines 327/703: "12 PyO3 classes are registered" (with the 12-name list).
**Reality:** **12** `add_class` registrations (4 top-level: NditerStep, Nditer, FromPyFunc, Vectorize; 8 in `random`). The error's mechanism: **13** `pyclass` declarations exist in the source; one is not registered. `mgrid`/`ogrid`/`r_`/`c_` singletons verified at lib.rs:118673-118676.
**Severity:** Low.

### GAP-008: Source-anchor lines rotted (README "Reading the Source Code")

**Claim → Reality:**
- `crates/fnp-ndarray/src/lib.rs` "(1,531 lines)" → **2,079** (contradicts the README's own LOC table, which says 2,079).
- `elementwise_binary` "around line 5920" → **6390**.
- `reduce_sum_values` "around line 25684" → **32277** (~6.6k lines off).
- fnp-python "single 182k-line file" → **180,069** (and shrinking; see Observations).
**Severity:** Low (navigation aid).

### GAP-009: CI description inaccuracies (README CI Gate Topology)

1. **Claim:** "a concurrency group cancels in-progress runs when a new commit lands on the same ref (`concurrency.cancel-in-progress: true`)".
   **Reality:** `cancel-in-progress: ${{ github.event_name != 'push' }}` — pushes to main are **never** cancelled (deliberate, per the ci.yml comment: push groups are keyed by SHA).
2. **Claim:** CI "sets up Python 3.12 + `pip install --upgrade pip numpy`".
   **Reality:** G2/G3 install **`numpy<2.5`** (explicit cap; the oracle floor logic lives elsewhere).
3. Trivial: ci.yml's own comment says `RUST_TOOLCHAIN` is at "line 24"; it is at line 47.
**Severity:** Low (precision, not structure — gate names, ordering, `needs:` chain, G9 wheel job, env pins for G6/G7 all verified correct).

### GAP-010: fnp-random dependency claim omits rayon

**Claim:** README lines 574/2149/2437: fnp-random "keeps only intra-workspace `fnp-ndarray` plus `getrandom`".
**Reality:** `crates/fnp-random/Cargo.toml` default features = `["ndarray", "getrandom", "rayon"]` — **rayon** is a default-on dependency (data-parallel jump-ahead fills; all three optional, `default-features = false` still gives the bare generator core).
**Severity:** Low (the "minimal dependency graph" selling point is understated-by-omission).

### GAP-011: Timeline commit counts presented as live are stale

**Claim:** README line 2415: "~650 commits in May 2026 alone, ~1,880 commits since 2026-04-01 … (These are live numbers; the repository keeps moving.)"
**Reality:** May 2026: **1,381**; since 2026-04-01: **7,010**; total: 7,336 ("7,300+ total" ✓). Related: "roughly 1,230 landed `perf(...)` commits" (June–July campaign) → **1,602** perf-subject commits in that window.
**Severity:** Low (narrative numbers, explicitly labeled live).

### GAP-012: The "499/499" headline is oracle-version-sensitive and no doc pins the version

**Claim:** "499/499 names" (README badge + 6 body sites; AGENTS).
**Reality:** Live probe on this host (numpy **2.3.5**): `len(numpy.__all__)` = **501**. The claimed 499 corresponds to a different numpy (fleet beads reference numpy 2.4.3 probes; CI pins `numpy<2.5`). The structural lock test iterates whatever numpy is on the build host, so the *mechanism* is sound — but the printed number is a snapshot against an unnamed oracle version. **Live coverage not re-run:** no cdylib is built on this host and building fnp-python is declined under the disk rule; no stored `run_fnp_python_api_coverage` artifact was found, so "exports=633 covered=599 missing=0" is likewise not re-verified.
**Severity:** Medium (the headline number should name its numpy pin or say "as measured against numpy <x> on <date>").

---

## Observations (no bead required)

- **LOC table:** 7/10 crates **exact** to the line (dtype 4,697; ndarray 2,079; iter 4,030; linalg 23,865; random 21,056; io 12,353; runtime 1,672). In-flight drift: fnp-python 182,027→180,259 (−1,768), fnp-ufunc +71, fnp-conformance +78. Total "roughly 402k" → 400,743.
- **#[test] counts:** README's 8,688 (2026-09-02) → **8,691** live (+3: fnp-ufunc 2,471→2,472, fnp-python 3,635→3,637). Remaining 8 crates match exactly — the table is genuinely current.
- **"216 rows lack worker markers":** independent recount over `docs/NEGATIVE_EVIDENCE.md` (1,465 `##` entries; AGENTS "1,000+" ✓) finds **213** rows dated 2026-08-16..31 lacking `host=`/`worker=`/`harness=` — corroborates the G2 failure description within counting-method noise.
- **Stray tracked file:** `crates/bench_identity.rs` (ASCII, at `crates/` root, not inside any crate). Tracked in git. Needs an owner decision; not touched or deleted (Rule 1).
- **AGENTS profile section:** verified exact — no `[profile.release]`; only `release-perf`, `bench`, `bench-fast` exist.

## Verified Claims (accurate, evidence-backed)

| Claim | Verification |
|---|---|
| 10 workspace members; 7 fuzz crates excluded | Root Cargo.toml members + exclude lists |
| `nightly-2026-08-25` in rust-toolchain.toml AND ci.yml `RUST_TOOLCHAIN` | Both files read; identical |
| 9/10 crates `#![forbid(unsafe_code)]`, fnp-python lone exception | `rg -l` over `**/src/lib.rs`: exactly the 9 non-python crates |
| `fnp_python_covers_full_numpy_all` structural lock | `conformance_remaining_top_level_attrs.rs:269` |
| Hygiene tests exist incl. `test_count_sanity_check` (>6,000), `no_allow_unused_in_library_code`, `no_unsafe_code_blocks_or_items` | codebase_hygiene.rs:115/147/161 |
| 48 binaries under `crates/fnp-conformance/src/bin/` | `git ls-files` count = 48 |
| 3 sibling `conformance_*.rs` files (dtype, linalg, ndarray) | git ls-files list |
| 12 PyO3 classes registered; mgrid/ogrid/r_/c_ singletons | lib.rs:118669-118676, 118744-118751 |
| MAX_HEADER_BYTES 65,536 / MAX_ARCHIVE_MEMBERS 4,096 / MAX_ARCHIVE_UNCOMPRESSED_BYTES 2 GiB / MAX_TEXT_ELEMENTS 16,777,216 | fnp-io/src/lib.rs:42-51 |
| `pub const fn promote` in fnp-dtype | src/lib.rs:209 |
| COMPENSATED_SUM_MIN_LEN = 1,000,000 | fnp-ufunc/src/lib.rs:53 |
| 12 threat classes in security_control_checks_v1.yaml; six required log fields | artifacts/contracts/…yaml, `threat_class` count = 12 |
| 9 P2C packets FNP-P2C-001..009 | artifacts/phase2c/ listing |
| pyproject.toml + maturin (module fnp_python, features python-extension, requires-python >=3.13, numpy>=2.3 floor); python/fnp_python/__init__.py | File read in full |
| All 12 referenced docs exist; all referenced scripts/e2e/*.sh exist | git ls-files |
| Workspace edition 2024, version 0.2.0 | Root Cargo.toml |
| No `[profile.release]`; release-perf/bench/bench-fast only | Profile section scan |
| v0.2.0 tag dated 2026-07-11; 7,300+ total commits | git tag / rev-list (7,336) |
| 2,782 closed beads as of 2026-09-02 | Live: 2,786 (+4, consistent drift); 38 open |
| DIVERGENCES.md: 0 active rows | File header + empty table |
| serde/serde_json 3 consumers each; pyo3 0.28.3 sole consumer fnp-python | Cargo.toml grep |
| CI status block 2026-09-03: G1 passed first time since 2026-02-26, G9 passed, G2 failed, G3–G8 skipped | `gh run view 33719040154`: G1 success, G9 success, G2 failure, G3–G8 skipped — exact |
| G2 red cause: ledger_hygiene rows missing worker markers | 213-row recount (above) |
| #[test] 8,688 (2026-09-02 count) | 8,691 live, +3 fleet drift; per-crate table otherwise exact |

## Test Count Verification

| Crate | README claim (2026-09-02) | Live (2026-09-03) | Δ |
|---|---:|---:|---:|
| fnp-ufunc | 2,471 | 2,472 | +1 |
| fnp-python | 3,635 | 3,637 | +2 |
| fnp-conformance | 395 | 395 | 0 |
| fnp-random | 477 | 477 | 0 |
| fnp-linalg | 459 | 459 | 0 |
| fnp-io | 407 | 407 | 0 |
| fnp-dtype | 276 | 276 | 0 |
| fnp-ndarray | 231 | 231 | 0 |
| fnp-iter | 203 | 203 | 0 |
| fnp-runtime | 134 | 134 | 0 |
| **Total** | **8,688** | **8,691** | **+3** |

## CI Verification (run 33719040154, commit b049c562)

| Job | Conclusion |
|---|---|
| G1: fmt + lint | success |
| G2: unit + property | **failure** |
| G9: wheel builds and imports | success |
| G3–G8 | skipped (needs-chained after G2) |

Two later pushes on 2026-09-03 (33784210979, 33785824930) also conclude `failure` overall — G2 remains red as documented.

## Beads Filed

See `br` titles prefixed `[reality-check]` created 2026-09-03 (GAP-001 … GAP-012 mapping listed in each bead body). No files were deleted; the only new file is this artifact.
