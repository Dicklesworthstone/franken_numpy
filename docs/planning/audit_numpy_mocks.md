# fnp-* mock-code-finder audit — 2026-04-22 (refreshed 2026-05-14)

Scanned `fnp-dtype`, `fnp-ndarray`, `fnp-iter`, `fnp-ufunc`, `fnp-linalg`, `fnp-random`, `fnp-io`, `fnp-runtime`, `fnp-conformance`, `fnp-python` — production code only (`#[cfg(test)]` blocks excluded). Scope covers both `crates/fnp-*/src/*.rs` (lib code) and `crates/fnp-*/src/bin/*.rs` (CLI tools); test-only `crates/*/tests/` files are out of scope (they ARE test code by definition). See the Reproduction Recipe at the bottom of this file for the exact `find` / `rg` invocations. Audit by CC agent in franken_numpy swarm, invoking the `mock-code-finder` skill.

## Summary

**Still zero real stubs/mocks/TODOs across all 10 `fnp-*` crates** (the 7 numeric impl crates — `fnp-dtype`, `fnp-ndarray`, `fnp-iter`, `fnp-ufunc`, `fnp-linalg`, `fnp-random`, `fnp-io` — plus `fnp-runtime` for mode/policy orchestration, `fnp-python` bindings, and `fnp-conformance` harness) as of the 2026-05-14 refresh. AGENTS.md characterises remaining gaps as "parity debt, not feature cuts"; this audit confirms that at the code level. The headline finding has held through the May 2026 parity wave that took `numpy.__all__` coverage from 43.3% to 100%.

> **Structural enforcement.** The findings here are also enforced automatically by `crates/fnp-conformance/tests/codebase_hygiene.rs` — 8 `#[test]` functions that fail CI when the corresponding marker appears in the codebase: `no_unimplemented_macros` (no `unimplemented!`), `no_todo_macros` (no `todo!`), `no_stub_comments`, `no_not_implemented_panics` (no `panic!("not implemented")`), `no_fixme_hack_markers` (no `FIXME`/`HACK`/`XXX` in comments), `no_dbg_macros_in_library_code`, `no_allow_unused_in_library_code` (no `#[allow(unused_*)]` in `crates/*/src/`), and `test_count_sanity_check` (asserts the workspace has >6,000 `#[test]` functions). All 8 verified passing on 2026-05-20 via `cargo test -p fnp-conformance --test codebase_hygiene`. This audit document is the human-readable companion; the test is the structural lock-in. Sibling integration tests in the same `tests/` directory cover related invariants: `concurrency_safety.rs` (verifies fnp-conformance's 4+ static Mutex/OnceLock combinations are thread-safe and deadlock-free), `numpy_reference_ops.rs` (oracle-driven reference-op cross-checks), `profiling_baseline.rs` (captures p50/p95/p99 latencies + environment fingerprint), and `smoke.rs` (end-to-end run of `run_all_core_suites` + `run_smoke`).

The cosmetic `.unwrap()` inventory has grown with the codebase: from **43 sites** at the original audit (2026-04-22) to **115 sites** in `fnp-conformance/src/**` alone (re-verified 2026-05-17: still exactly 115 — `ufunc_differential.rs:65`, `lib.rs:47`, `bin/run_oracle_drift_matrix.rs:2`, `oracle_drift_matrix.rs:1`). Crate growth (304,694 Rust lines vs 254,570 in April — re-verified 2026-05-20 via `find crates -name '*.rs' -not -path '*/fuzz/*' -not -path '*/target/*' | xargs wc -l`; 5-line drift since 2026-05-17 = 0.0016%, all from the per-fuzz `rust-toolchain.toml` + `fnp-iter` Cargo.toml comment additions plus one bench comment refresh) explains the increase; all checked sites remain on statically-correct invariants (fixture/parser code, non-empty Vec, matching DType, constant seeds) and not on user-reachable paths.

## Detection matrix

| Detection | Count | Notes |
|---|---|---|
| `TODO` / `FIXME` / `HACK` / `XXX` / `STUB` / `PLACEHOLDER` / `MOCK` / `DUMMY` / `FAKE` keywords | **0** | Only hit across all crates: `crates/fnp-conformance/src/bin/validate_phase2c_packet.rs:41` — `<FNP-P2C-XXX>` inside a CLI usage help string (user-input placeholder, not code). |
| `todo!()` / `unimplemented!()` / `panic!("not implemented")` | **0** | — |
| Empty function bodies / `Ok(())`-only stub returns | **0** | — |
| `sleep()` / `thread::sleep` / fake-work patterns | **0** | One legitimate `std::time::Duration::from_secs(10)` in `crates/fnp-conformance/src/raptorq_artifacts.rs:245` (RaptorQ `block_timeout` config value). |
| pyo3 / numpy delegation stubs in impl crates | **0** | `fnp-python` delegates by design (it is the parity oracle surface); every other `fnp-*` crate is self-contained Rust. |
| `numpy.*` references in impl crates | docstrings + 1 legitimate embed | All references are either doc comments explaining numpy semantics, or the embedded-Python snippet in `fnp-io` that reads `numpy.lib.format.open_memmap` for memmap parity. |
| Production `.unwrap()` — impl crates (excl. `fnp-conformance`) | **42** at 2026-09-24 (fnp-python 26, fnp-linalg 10, fnp-ufunc 6) | Census and per-site classes in §D below. The einsum site of §A was rewritten to `let Some(...) = ... else { return Err(...) }`. |
| Production `.unwrap()` — `fnp-conformance` fixture/oracle code | **42** at April audit; **115** at 2026-05-14 refresh | Most growth is in the diagnostic-oracle and structured-dtype-corpus expansion that landed under the `33vtd` epic. Pattern unchanged: still all on statically-correct invariants in fixture-capture code that never runs on production paths. **Note:** this 115 figure is fixture/test-harness code only — the 9 non-conformance impl crates remain at **zero** production unwraps (see prior row). See §B below. |
| AST-grep scan for suspiciously short functions (`fn $NAME($$$) -> $RET { $SINGLE }`) | hits are all legitimate | `default()` constructors, `Display::fmt`, simple accessors like `all_numeric_dtypes()` and `is_malformed_probability_input`. None are stubs. |
| `#[allow(unused_*)]` in `crates/*/src/` library code | **0** | Structurally enforced by `codebase_hygiene::no_allow_unused_in_library_code`. The `unused_*` family of warnings is exactly what catches stale code paths that would otherwise rot silently — suppressing it would defeat the no-stubs invariant. |
| `dbg!()` macro in `crates/*/src/` library code | **0** | Structurally enforced by `codebase_hygiene::no_dbg_macros_in_library_code`. (The `println!`/`eprintln!` row above covers the broader print-family scan.) |
| `println!` / `eprintln!` / `dbg!` in production code | **4** | All legitimate: `crates/fnp-conformance/src/lib.rs:21319`/`21324` (e2e step progress to stderr), `crates/fnp-conformance/src/lib.rs:21874` (warning when a log line fails parsing in the diagnostic harness), and `crates/fnp-ufunc/src/lib.rs:394` (NumPy-parity `errstate(over='print')` mode — the intended behavior, not debug output). No `dbg!` macros anywhere. |
| `panic!()` in `#[cfg(test)]` oracle/test helpers | **3 in fnp-linalg, plus matching patterns in fnp-ufunc test blocks** | All in test-only code, separate from the stub-flavored row above (which is enforced structurally by `codebase_hygiene::no_not_implemented_panics`). fnp-linalg sites: `crates/fnp-linalg/src/lib.rs:9621`/`9701`/`9744` — three `panic!("unknown oracle payload kind: {kind}")` arms in test-only oracle parsing helpers (inside `#[cfg(test)]` block starting at line 5302, helper fns like `numpy_oracle_pinv_tolerance_aliases`). Several similar patterns exist in fnp-ufunc test blocks (`match err { …, other => panic!("unexpected error: {other:?}") }`). These are deliberate test-failure mechanisms, not stubs. |
| `.expect()` calls in production code (impl crates excl. `fnp-conformance`) | **109** at 2026-09-24 (fnp-python 95, fnp-ufunc 10, fnp-io 2, fnp-random 1, fnp-random-core 1), plus 19 `unreachable!` | Census in §D below: comments, strings and every `#[cfg(test)]` item stripped by brace matching (the 2026-05-17 figure counted only up to the FIRST `#[cfg(test)]` marker per file). |

## Historical findings (originally drafted as beads — resolution status as of 2026-05-14)

The April audit drafted three beads "to file when DB contention clears." DB contention is long resolved (1220+ beads filed since). Status of each:

  - **Bead 1 (einsum unwrap):** **RESOLVED ORGANICALLY.** The `.unwrap()` was rewritten to `let Some((prefix, _)) = sub.split_once("...") else { return Err(...) }` at `crates/fnp-ufunc/src/lib.rs:19013`. No bead was filed; the fix landed as part of broader cleanup.
  - **Bead 2 (fixture unwrap cluster):** **TRACKED ONLY HERE — no bead filed.** Site count grew from 42 → 115 (see refreshed counts above). Recommendation unchanged: cosmetic, no real-mock signal, low-priority `.expect()` migration. Deliberately not filed as a bead because it would clutter the tracker with low-impact style debt; this row is the canonical record. Would be appropriate as a multi-hour batch task only.
  - **Bead 3 (audit record bead):** **NOT NEEDED.** The audit document itself (this file) is now referenced from README.md, CHANGELOG.md, and the structural lock-in conformance test commentary in `crates/fnp-python/src/lib.rs`. The auditability function is served without a bead pointer.

### A. Bead 1 — single einsum parser unwrap (RESOLVED in 2026-05 cleanup)

- **Title:** `[MOCK] fnp-ufunc einsum split_once unwrap should use expect()`
- **Type:** `task`
- **Priority:** `3` (low — purely cosmetic)
- **Site:** `crates/fnp-ufunc/src/lib.rs:17893`
- **Code:**

  ```rust
  let (prefix, _) = sub.split_once("...").unwrap();
  ```

- **Why it is not a real mock:** The branch is gated by a prior `contains("...")` check, so `split_once` is statically guaranteed to succeed. A panic here indicates a bug upstream, not a missing implementation.
- **Proposed fix:** Replace with `.expect("einsum: '...' expansion is guarded by the contains-check upstream")`. Strictly diagnostic, no semantic change.

### B. Bead 2 — fnp-conformance fixture `.unwrap()` cluster (42 sites)

- **Title:** `[MOCK] fnp-conformance fixture unwraps should use expect() with context (42 sites)`
- **Type:** `task`
- **Priority:** `3` (low — purely cosmetic)
- **Sites:**
  - `crates/fnp-conformance/src/lib.rs` — **25 sites**, all the pattern `UFuncArray::new(vec![v.len()], v, DType::F64).unwrap()` (lines 13147, 13151, 13155, 13159, 13177, 13181, 13185, 13189, 13207, 13211, 13215, 13219, 13233, 13237, 13241, 13245, 13271, 13275, 13279, 13283, 13297, 13301, 13305, 13309, 13319).
  - `crates/fnp-conformance/src/bin/dump_expected.rs` — **17 sites**, mostly `Generator::from_pcg64_dxsm(seed).unwrap()` plus a few distribution-specific helper unwraps (`dirichlet`, `noncentral_chisquare`, `noncentral_f`, `zipf`, etc.).
- **Why they are not real mocks:** All unwrap on statically-correct invariants — non-empty `Vec`, matching `DType::F64`, constant seeds. They are in oracle-capture/benchmark-fixture code that is never hit on production paths; they are only reached during conformance capture and evidence dumps.
- **Proposed fix:** Replace with `.expect("fnp-conformance fixture: <reason>")` so a future refactor that breaks the invariant surfaces a readable message instead of a bare "called `Option::unwrap()` on a `None` value".

### C. Bead 3 — audit result record

- **Title:** `[MOCK-AUDIT] fnp-* clean scan: zero stubs/mocks/TODOs, 43 cosmetic unwraps`
- **Type:** `docs`
- **Priority:** `4` (backlog)
- **Body:** Pointer to this file (`audit_numpy_mocks.md`) and a short summary for auditability.

## Detection commands (reproducible)

```bash
# Keyword scan (covers all 10 fnp-* crates including fnp-python).
# Scope per Summary above: lib + bin only, no tests/. The -g filters
# exclude integration-test dirs and codebase_hygiene.rs (which contains
# the keywords as regex strings used to detect them — would self-match).
rg -n --type rust "TODO|FIXME|HACK|XXX|STUB|PLACEHOLDER|MOCK|DUMMY|FAKE" \
  -g '!*/tests/*' \
  -g '!codebase_hygiene.rs' \
  crates/fnp-dtype crates/fnp-ndarray crates/fnp-iter crates/fnp-ufunc \
  crates/fnp-linalg crates/fnp-random crates/fnp-io crates/fnp-conformance \
  crates/fnp-runtime crates/fnp-python

# Unimplemented macros
rg -n --type rust "unimplemented!|todo!\(|panic!\(\"not implemented" crates/fnp-*/src

# Production unwrap count per file (strips from first #[cfg(test)])
for f in crates/fnp-*/src/*.rs crates/fnp-*/src/bin/*.rs; do
  [ -f "$f" ] || continue
  prod_unwraps=$(awk '/^#\[cfg\(test\)\]/ {exit} /\.unwrap\(\)/' "$f" | wc -l)
  [ "$prod_unwraps" -gt 0 ] && echo "$prod_unwraps $f"
done | sort -rn

# Structural scan (all 10 fnp-* impl crates including fnp-python + fnp-conformance)
ast-grep run -l Rust -p 'fn $NAME($$$) -> $RET { $SINGLE }' --json \
  | jq -r '.[] | select(.file | test("crates/fnp-(dtype|ndarray|iter|ufunc|linalg|random|io|runtime|python|conformance)/src")) | "\(.file):\(.range.start.line)"'
```

## D. Panic-site census — 2026-09-24 (bead `deadlock-audit-rc0923-epic-71qy3.20`)

Every `.unwrap()`, `.expect(`, `unreachable!` and `panic!` in non-test library code (comments and
string literals stripped, `#[cfg(test)]` items dropped by brace matching, `src/bin` skipped):
199 sites. Each was classified by reading its guard, and every "reachable" claim was backed by a
repro run against a live fnp build and numpy 2.4.3:

- **a — 147 sites: no input reaches them.** The proof is the guard or invariant at the site
  (early return on empty/0-d input, a slice of exactly 8 bytes, a matching dtype arm, ...).
- **b — 38 sites, FIXED: a NaN written between a route's NaN pre-scan and its comparator.** The
  NaN-screened sort / argsort / unique / searchsorted / sort_complex routes compared with
  `partial_cmp(..).expect("no NaN")`; another thread's `np.copyto` (which drops the GIL) reproduced
  25 of them, a patched `np.empty` 13 more. numpy returned under the same race; fnp raised
  PanicException. All 59 such comparators now use `NanLastCmp` (numpy's NaN-last order, -0.0 ==
  0.0), argmin's NaN re-scan no longer `expect`s the NaN it saw, and
  `nan_screened_sort_routes_never_panic_when_the_operand_changes_after_the_screen` is the probe.
  Five sites classed (a) share the same exposure and received the same fix.
- **b? — 2 sites, fixed with the same change:** the flat f64/f32 argsort fallbacks. They are only
  reached above 2^32 elements, so they were never reproduced.
- **b-api — 1 site:** `fnp-random-core` `generate_state_u64` (see bead `.14`). It is reachable
  from its public Rust API only, and nothing depends on that crate.
- **c — 10 sites, FIXED:** fnp-linalg batch `first_err` Mutex `lock()` / `into_inner()` unwraps
  are now poison-tolerant.
- **hook — 1 site:** the alloc-error hook's `panic!`, which is caught and raised as
  `MemoryError`.

The hostile-input runs behind this census found no panic from any single-threaded call. They did
find eight numpy divergences next to these sites, each fixed and probed by
`panic_audit_neighbour_divergences_match_numpy` and
`strings_and_char_functions_match_numpy_on_zero_dim_and_scalar_operands`:
- 0-d string operands in 29 strings/char functions (79 cells);
- `ravel_multi_index` overflow wrapping silently;
- `ediff1d` float `to_end` on a list;
- `take`/`put` with uint64 indices;
- `histogram_bin_edges(bins=None)`;
- `linalg.cholesky(upper=None / 1)`.

<details><summary>All 199 sites (line numbers at commit 37dda281)</summary>

| site (at 37dda281) | kind | class | function | status |
|---|---|---|---|---|
| `crates/fnp-conformance/src/fnp_python_api_coverage.rs:201` | `.expect` | a | `build_api_coverage_report_from_inputs` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13206` | `.expect` | a | `poly_family_add` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13211` | `.expect` | a | `poly_family_add` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13216` | `.expect` | a | `poly_family_add` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13221` | `.expect` | a | `poly_family_add` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13240` | `.expect` | a | `poly_family_sub` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13245` | `.expect` | a | `poly_family_sub` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13250` | `.expect` | a | `poly_family_sub` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13255` | `.expect` | a | `poly_family_sub` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13274` | `.expect` | a | `poly_family_mul` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13279` | `.expect` | a | `poly_family_mul` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13284` | `.expect` | a | `poly_family_mul` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13289` | `.expect` | a | `poly_family_mul` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13304` | `.expect` | a | `poly_family_deriv` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13309` | `.expect` | a | `poly_family_deriv` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13314` | `.expect` | a | `poly_family_deriv` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13319` | `.expect` | a | `poly_family_deriv` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13346` | `.expect` | a | `poly_family_deriv_m` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13351` | `.expect` | a | `poly_family_deriv_m` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13356` | `.expect` | a | `poly_family_deriv_m` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13361` | `.expect` | a | `poly_family_deriv_m` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13376` | `.expect` | a | `poly_family_int` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13381` | `.expect` | a | `poly_family_int` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13386` | `.expect` | a | `poly_family_int` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13391` | `.expect` | a | `poly_family_int` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/lib.rs:13402` | `.expect` | a | `poly_family_val` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/oracle_drift_matrix.rs:379` | `.expect` | a | `value_hash` | unreachable (proof in the site's guard) |
| `crates/fnp-conformance/src/oracle_drift_matrix.rs:383` | `.expect` | a | `value_hash` | unreachable (proof in the site's guard) |
| `crates/fnp-io/src/lib.rs:4243` | `.expect` | a | `<SpaceWildcardFields as Iterator>::next` | unreachable (proof in the site's guard) |
| `crates/fnp-io/src/lib.rs:4480` | `.expect` | a | `push_i64_decimal` | unreachable (proof in the site's guard) |
| `crates/fnp-linalg/src/lib.rs:10965` | `.unwrap()` | c | `batch inv/det/solve first_err Mutex` | FIXED - poison-tolerant |
| `crates/fnp-linalg/src/lib.rs:10976` | `.unwrap()` | c | `batch inv/det/solve first_err Mutex` | FIXED - poison-tolerant |
| `crates/fnp-linalg/src/lib.rs:11025` | `.unwrap()` | c | `batch inv/det/solve first_err Mutex` | FIXED - poison-tolerant |
| `crates/fnp-linalg/src/lib.rs:11037` | `.unwrap()` | c | `batch inv/det/solve first_err Mutex` | FIXED - poison-tolerant |
| `crates/fnp-linalg/src/lib.rs:11085` | `.unwrap()` | c | `batch inv/det/solve first_err Mutex` | FIXED - poison-tolerant |
| `crates/fnp-linalg/src/lib.rs:11097` | `.unwrap()` | c | `batch inv/det/solve first_err Mutex` | FIXED - poison-tolerant |
| `crates/fnp-linalg/src/lib.rs:11360` | `.unwrap()` | c | `batch inv/det/solve first_err Mutex` | FIXED - poison-tolerant |
| `crates/fnp-linalg/src/lib.rs:11371` | `.unwrap()` | c | `batch inv/det/solve first_err Mutex` | FIXED - poison-tolerant |
| `crates/fnp-linalg/src/lib.rs:11586` | `.unwrap()` | c | `batch inv/det/solve first_err Mutex` | FIXED - poison-tolerant |
| `crates/fnp-linalg/src/lib.rs:11596` | `.unwrap()` | c | `batch inv/det/solve first_err Mutex` | FIXED - poison-tolerant |
| `crates/fnp-python/src/lib.rs:7931` | `unreachable!` | a | `extract_take_indices` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:8925` | `unreachable!` | a | `storage_from_numeric_text_tokens` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:8944` | `unreachable!` | a | `storage_from_numeric_text_tokens` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:13703` | `.expect` | a | `try_zerocopy_f16_argextreme_flat` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:18167` | `.expect` | a | `try_zerocopy_f64_select` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:18406` | `.expect` | a | `try_zerocopy_int_select` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:26371` | `.expect` | a | `nonzero_nd_parallel_typed` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:29473` | `.expect` | a | `try_native_repeat_scalar` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:29481` | `.expect` | a | `try_native_repeat_scalar` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:29489` | `.expect` | a | `try_native_repeat_scalar` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:31987` | `.expect` | a | `int_matrix_power_typed` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:32412` | `.expect` | a | `cholesky` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:32414` | `.expect` | a | `cholesky` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:32832` | `unreachable!` | a | `try_zerocopy_f64_eigvalsh_diagonal` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:41218` | `.expect` | a | `try_zerocopy_histogram_f16` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:41223` | `.expect` | a | `try_zerocopy_histogram_f16` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:42794` | `.expect` | a | `diff1_pend_arm! generated fn (diff1_pend_f64 / diff1_pend_i64 ...)` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:42798` | `.expect` | a | `diff1_pend_arm! generated fn (diff1_pend_f64 / diff1_pend_i64 ...)` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:42803` | `.expect` | a | `diff1_pend_arm! generated fn (diff1_pend_f64 / diff1_pend_i64 ...)` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:43288` | `.unwrap()` | a | `histogram_bin_edges` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:53039` | `unreachable!` | a | `try_zerocopy_f64_vector_norm_axis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:53068` | `unreachable!` | a | `try_zerocopy_f64_vector_norm_axis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:53202` | `unreachable!` | a | `try_zerocopy_f32_vector_norm_axis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:53267` | `unreachable!` | a | `try_zerocopy_f32_vector_norm_axis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:53295` | `unreachable!` | a | `try_zerocopy_f32_vector_norm_axis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:54741` | `.unwrap()` | a | `try_zerocopy_f32_nanarg_lastaxis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:54829` | `.unwrap()` | a | `try_zerocopy_f64_nanarg_lastaxis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:57675` | `.unwrap()` | a | `wide_int_table_bounds` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:57680` | `.unwrap()` | a | `wide_int_table_bounds` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:59951` | `.expect` | a | `c128_pair_cmp` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:59955` | `.expect` | a | `c128_pair_cmp` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:65070` | `.unwrap()` | a | `try_native_unwrap_default` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:69051` | `.expect` | a | `loadtxt` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:69099` | `.expect` | a | `loadtxt` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:69227` | `unreachable!` | a | `loadtxt` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:69499` | `unreachable!` | a | `loadtxt` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:69516` | `unreachable!` | a | `loadtxt` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:69755` | `.expect` | a | `genfromtxt` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:69808` | `.expect` | a | `genfromtxt` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:69944` | `unreachable!` | a | `genfromtxt` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:69961` | `unreachable!` | a | `genfromtxt` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:73300` | `.expect` | b | `try_zerocopy_f64_sort_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:73388` | `.expect` | b | `try_zerocopy_c128_sort_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:73389` | `.expect` | b | `try_zerocopy_c128_sort_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:73450` | `.expect` | b | `try_zerocopy_c128_unique_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:73451` | `.expect` | b | `try_zerocopy_c128_unique_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:73588` | `.expect` | b | `try_zerocopy_c128_searchsorted` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:73589` | `.expect` | b | `try_zerocopy_c128_searchsorted` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:73817` | `.expect` | b | `try_zerocopy_c64_searchsorted` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:73818` | `.expect` | b | `try_zerocopy_c64_searchsorted` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:73990` | `.expect` | b | `try_zerocopy_c64_unique_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:73991` | `.expect` | b | `try_zerocopy_c64_unique_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74180` | `.expect` | b | `try_zerocopy_c128_sort_lastaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74181` | `.expect` | b | `try_zerocopy_c128_sort_lastaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74258` | `.expect` | b | `try_zerocopy_c128_sort_axis0` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74259` | `.expect` | b | `try_zerocopy_c128_sort_axis0` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74378` | `.expect` | b | `try_zerocopy_c128_sort_midaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74379` | `.expect` | b | `try_zerocopy_c128_sort_midaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74490` | `.expect` | b | `try_zerocopy_c64_sort_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74491` | `.expect` | b | `try_zerocopy_c64_sort_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74575` | `.expect` | b | `try_zerocopy_c64_sort_lastaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74576` | `.expect` | b | `try_zerocopy_c64_sort_lastaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74650` | `.expect` | b | `try_zerocopy_c64_sort_axis0` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74651` | `.expect` | b | `try_zerocopy_c64_sort_axis0` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74767` | `.expect` | b | `try_zerocopy_c64_sort_midaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:74768` | `.expect` | b | `try_zerocopy_c64_sort_midaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:76294` | `.expect` | a | `try_packed_string_union1d` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:76409` | `.expect` | a | `try_packed_string_setxor` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:78505` | `.unwrap()` | a | `try_native_datetime_sort_axes` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:79026` | `.expect` | b | `try_zerocopy_f64_sort_lastaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:79111` | `.expect` | a | `try_zerocopy_f64_sort_axis0` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:79224` | `.expect` | a | `try_zerocopy_f64_sort_midaxis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:79885` | `.expect` | b? | `try_zerocopy_f64_argsort_flat` | FIXED - same pattern, not reproduced (needs >2^32 elements) |
| `crates/fnp-python/src/lib.rs:79960` | `.expect` | b? | `try_zerocopy_f32_argsort_flat` | FIXED - same pattern, not reproduced (needs >2^32 elements) |
| `crates/fnp-python/src/lib.rs:80063` | `.expect` | b | `try_zerocopy_c128_argsort_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:80064` | `.expect` | b | `try_zerocopy_c128_argsort_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:80162` | `.expect` | b | `try_zerocopy_c64_argsort_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:80163` | `.expect` | b | `try_zerocopy_c64_argsort_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:80263` | `.expect` | b | `try_zerocopy_c64_argsort_lastaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:80264` | `.expect` | b | `try_zerocopy_c64_argsort_lastaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:80396` | `.expect` | b | `try_zerocopy_c128_argsort_lastaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:80397` | `.expect` | b | `try_zerocopy_c128_argsort_lastaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:80497` | `.expect` | a | `try_zerocopy_c128_argsort_axis0` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:80498` | `.expect` | a | `try_zerocopy_c128_argsort_axis0` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:80623` | `.expect` | a | `try_zerocopy_c128_argsort_midaxis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:80624` | `.expect` | a | `try_zerocopy_c128_argsort_midaxis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:80768` | `.expect` | a | `try_zerocopy_c64_argsort_axis0` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:80769` | `.expect` | a | `try_zerocopy_c64_argsort_axis0` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:80891` | `.expect` | a | `try_zerocopy_c64_argsort_midaxis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:80892` | `.expect` | a | `try_zerocopy_c64_argsort_midaxis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:82917` | `.expect` | b | `try_zerocopy_f64_argsort_lastaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:83012` | `.expect` | a | `try_zerocopy_f64_argsort_axis0` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:83137` | `.expect` | a | `try_zerocopy_f64_argsort_midaxis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:83253` | `.expect` | b | `try_zerocopy_f32_argsort_lastaxis` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:83340` | `.expect` | a | `try_zerocopy_f32_argsort_axis0` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:83457` | `.expect` | a | `try_zerocopy_f32_argsort_midaxis` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:83920` | `.expect` | b | `try_zerocopy_f64_sort_complex_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:83923` | `.expect` | b | `try_zerocopy_f64_sort_complex_flat` | FIXED - race/TOCTOU |
| `crates/fnp-python/src/lib.rs:84325` | `.expect` | a | `try_native_f16_multi_quantile_histogram` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:84326` | `.expect` | a | `try_native_f16_multi_quantile_histogram` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:88679` | `.unwrap()` | a | `try_zerocopy_ravel_c` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:92420` | `.unwrap()` | a | `minmax_int_typed` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:93599` | `.expect` | a | `parallel_arg_extremum_f64` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:93620` | `.expect` | a | `parallel_arg_extremum_f64` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:96700` | `.expect` | a | `try_zerocopy_f64_argextreme` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:96902` | `.unwrap()` | a | `try_zerocopy_lastaxis_argextreme` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:96938` | `.unwrap()` | a | `try_zerocopy_lastaxis_argextreme` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:97356` | `.unwrap()` | a | `try_zerocopy_bool_argextreme_flat` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:103245` | `.unwrap()` | a | `try_einsum_transpose_view` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:104620` | `.expect` | a | `einsum_int_chain_spec_matches` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:104622` | `.expect` | a | `einsum_int_chain_spec_matches` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:104632` | `.expect` | a | `einsum_int_chain_spec_matches` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:104659` | `.expect` | a | `einsum_int_batched_chain_spec_matches` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:104672` | `.expect` | a | `einsum_int_batched_chain_spec_matches` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:104674` | `.expect` | a | `einsum_int_batched_chain_spec_matches` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:104684` | `.expect` | a | `einsum_int_batched_chain_spec_matches` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:105460` | `.unwrap()` | a | `try_zerocopy_f64_einsum_outer_2vec` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:105461` | `.unwrap()` | a | `try_zerocopy_f64_einsum_outer_2vec` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:106151` | `.expect` | a | `try_zerocopy_f64_unique_flat` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:106242` | `.unwrap()` | a | `transform_field` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:107789` | `.expect` | a | `try_native_unique_rows_lexsort_f64` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:107894` | `.expect` | a | `try_native_unique_rows_lexsort_f32` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:108562` | `.expect` | a | `try_native_unique_rows_lexsort_f64_full` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:108721` | `.expect` | a | `try_native_unique_rows_lexsort_f32_full` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:111324` | `.unwrap()` | a | `try_zerocopy_unicode_strip` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:115188` | `.unwrap()` | a | `argextreme_typed` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:115273` | `.unwrap()` | a | `ptp_typed` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:115274` | `.unwrap()` | a | `ptp_typed` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:115424` | `.unwrap()` | a | `ptp_axis_typed` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:117524` | `.expect` | a | `try_zerocopy_block_2d_grid` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:121461` | `.unwrap()` | a | `ediff1d` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:121463` | `.unwrap()` | a | `ediff1d` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:121466` | `.unwrap()` | a | `ediff1d` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:121466` | `.unwrap()` | a | `ediff1d` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:121611` | `.unwrap()` | a | `ediff1d` | unreachable (proof in the site's guard) |
| `crates/fnp-python/src/lib.rs:121721` | `panic!` | hook | `alloc error hook` | intentional (alloc_error_hook -> MemoryError) |
| `crates/fnp-random-core/src/lib.rs:106` | `.expect` | b-api | `SeedSequence::generate_state_u64` | pub Rust API only (unused crate, see .14) |
| `crates/fnp-random/src/lib.rs:6707` | `.expect` | a | `Generator::permuted` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:4381` | `.expect` | a | `reduce_frompyfunc_values` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:5125` | `unreachable!` | a | `UFuncArray::from_storage_with_dtype` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:5266` | `unreachable!` | a | `UFuncArray::write_integer_mutation` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:9083` | `.expect` | a | `UFuncArray::reduce_sum_axes` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:9110` | `.expect` | a | `UFuncArray::reduce_prod_axes` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:9137` | `.expect` | a | `UFuncArray::reduce_min_axes` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:9164` | `.expect` | a | `UFuncArray::reduce_max_axes` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:9191` | `.expect` | a | `UFuncArray::reduce_mean_axes` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:15481` | `unreachable!` | a | `UFuncArray::complex_logical_predicate` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:17480` | `.expect` | a | `UFuncArray::any_axes` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:17506` | `.expect` | a | `UFuncArray::all_axes` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:25216` | `.unwrap()` | a | `UFuncArray::ptp` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:25243` | `.unwrap()` | a | `UFuncArray::ptp` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:30474` | `unreachable!` | a | `select_percentile_method` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:30498` | `.unwrap()` | a | `i64_membership_mask` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:30499` | `.unwrap()` | a | `i64_membership_mask` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:30522` | `.unwrap()` | a | `u64_membership_mask` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:30523` | `.unwrap()` | a | `u64_membership_mask` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:30738` | `unreachable!` | a | `par_select_percentile` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:30786` | `.expect` | a | `par_select_percentiles_linear` | unreachable (proof in the site's guard) |
| `crates/fnp-ufunc/src/lib.rs:30790` | `.expect` | a | `par_select_percentiles_linear` | unreachable (proof in the site's guard) |

</details>

## Notes

- The DB is currently jammed (14 concurrent `br create` processes across agent projects plus a lingering `br show franken_numpy-p6qy` zombie from earlier in the session). The three beads drafted above will be batch-filed from this file in a later tick when the DB frees up.
- This audit complements but does not substitute for `fnp-conformance` differential coverage — a file can pass this mock audit and still be parity-incomplete. Differential coverage is tracked separately via the oracle-capture pipeline.
