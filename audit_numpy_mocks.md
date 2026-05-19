# fnp-* mock-code-finder audit — 2026-04-22 (refreshed 2026-05-14)

Scanned `fnp-dtype`, `fnp-ndarray`, `fnp-iter`, `fnp-ufunc`, `fnp-linalg`, `fnp-random`, `fnp-io`, `fnp-runtime`, `fnp-conformance`, `fnp-python` — production code only (`#[cfg(test)]` blocks excluded). Scope covers both `crates/fnp-*/src/*.rs` (lib code) and `crates/fnp-*/src/bin/*.rs` (CLI tools); test-only `crates/*/tests/` files are out of scope (they ARE test code by definition). See the Reproduction Recipe at the bottom of this file for the exact `find` / `rg` invocations. Audit by CC agent in franken_numpy swarm, invoking the `mock-code-finder` skill.

## Summary

**Still zero real stubs/mocks/TODOs across all 10 `fnp-*` crates** (the 7 numeric impl crates — `fnp-dtype`, `fnp-ndarray`, `fnp-iter`, `fnp-ufunc`, `fnp-linalg`, `fnp-random`, `fnp-io` — plus `fnp-runtime` for mode/policy orchestration, `fnp-python` bindings, and `fnp-conformance` harness) as of the 2026-05-14 refresh. AGENTS.md characterises remaining gaps as "parity debt, not feature cuts"; this audit confirms that at the code level. The headline finding has held through the May 2026 parity wave that took `numpy.__all__` coverage from 43.3% to 100%.

> **Structural enforcement.** The findings here are also enforced automatically by `crates/fnp-conformance/tests/codebase_hygiene.rs` — 8 `#[test]` functions that fail CI when the corresponding marker appears in the codebase: `no_unimplemented_macros` (no `unimplemented!`), `no_todo_macros` (no `todo!`), `no_stub_comments`, `no_not_implemented_panics` (no `panic!("not implemented")`), `no_fixme_hack_markers` (no `FIXME`/`HACK`/`XXX` in comments), `no_dbg_macros_in_library_code`, `no_allow_unused_in_library_code` (no `#[allow(unused_*)]` in `crates/*/src/`), and `test_count_sanity_check` (asserts the workspace has >6,000 `#[test]` functions). All 8 verified passing on 2026-05-20 via `cargo test -p fnp-conformance --test codebase_hygiene`. This audit document is the human-readable companion; the test is the structural lock-in. Sibling integration tests in the same `tests/` directory cover related invariants: `concurrency_safety.rs` (verifies fnp-conformance's 4+ static Mutex/OnceLock combinations are thread-safe and deadlock-free), `numpy_reference_ops.rs` (oracle-driven reference-op cross-checks), `profiling_baseline.rs` (captures p50/p95/p99 latencies + environment fingerprint), and `smoke.rs` (end-to-end run of `run_all_core_suites` + `run_smoke`).

The cosmetic `.unwrap()` inventory has grown with the codebase: from **43 sites** at the original audit (2026-04-22) to **115 sites** in `fnp-conformance/src/**` alone (re-verified 2026-05-17: still exactly 115 — `ufunc_differential.rs:65`, `lib.rs:47`, `bin/run_oracle_drift_matrix.rs:2`, `oracle_drift_matrix.rs:1`). Crate growth (304,689 Rust lines vs 254,570 in April — re-verified 2026-05-17 via `find crates -name '*.rs' -not -path '*/fuzz/*' -not -path '*/target/*' | xargs wc -l`) explains the increase; all checked sites remain on statically-correct invariants (fixture/parser code, non-empty Vec, matching DType, constant seeds) and not on user-reachable paths.

## Detection matrix

| Detection | Count | Notes |
|---|---|---|
| `TODO` / `FIXME` / `HACK` / `XXX` / `STUB` / `PLACEHOLDER` / `MOCK` / `DUMMY` / `FAKE` keywords | **0** | Only hit across all crates: `crates/fnp-conformance/src/bin/validate_phase2c_packet.rs:41` — `<FNP-P2C-XXX>` inside a CLI usage help string (user-input placeholder, not code). |
| `todo!()` / `unimplemented!()` / `panic!("not implemented")` | **0** | — |
| Empty function bodies / `Ok(())`-only stub returns | **0** | — |
| `sleep()` / `thread::sleep` / fake-work patterns | **0** | One legitimate `std::time::Duration::from_secs(10)` in `crates/fnp-conformance/src/raptorq_artifacts.rs:245` (RaptorQ `block_timeout` config value). |
| pyo3 / numpy delegation stubs in impl crates | **0** | `fnp-python` delegates by design (it is the parity oracle surface); every other `fnp-*` crate is self-contained Rust. |
| `numpy.*` references in impl crates | docstrings + 1 legitimate embed | All references are either doc comments explaining numpy semantics, or the embedded-Python snippet in `fnp-io` that reads `numpy.lib.format.open_memmap` for memmap parity. |
| Production `.unwrap()` — impl crates (excl. `fnp-conformance`) | **1** at April audit; current scan unchanged (other unwraps now under `#[cfg(test)]`) | See §A below. The einsum site has since been rewritten to `let Some(...) = ... else { return Err(...) }` — no longer an unwrap (see `crates/fnp-ufunc/src/lib.rs:19013`). |
| Production `.unwrap()` — `fnp-conformance` fixture/oracle code | **42** at April audit; **115** at 2026-05-14 refresh | Most growth is in the diagnostic-oracle and structured-dtype-corpus expansion that landed under the `33vtd` epic. Pattern unchanged: still all on statically-correct invariants in fixture-capture code that never runs on production paths. **Note:** this 115 figure is fixture/test-harness code only — the 9 non-conformance impl crates remain at **zero** production unwraps (see prior row). See §B below. |
| AST-grep scan for suspiciously short functions (`fn $NAME($$$) -> $RET { $SINGLE }`) | hits are all legitimate | `default()` constructors, `Display::fmt`, simple accessors like `all_numeric_dtypes()` and `is_malformed_probability_input`. None are stubs. |
| `#[allow(unused_*)]` in `crates/*/src/` library code | **0** | Structurally enforced by `codebase_hygiene::no_allow_unused_in_library_code`. The `unused_*` family of warnings is exactly what catches stale code paths that would otherwise rot silently — suppressing it would defeat the no-stubs invariant. |
| `dbg!()` macro in `crates/*/src/` library code | **0** | Structurally enforced by `codebase_hygiene::no_dbg_macros_in_library_code`. (The `println!`/`eprintln!` row above covers the broader print-family scan.) |
| `println!` / `eprintln!` / `dbg!` in production code | **4** | All legitimate: `crates/fnp-conformance/src/lib.rs:21319`/`21324` (e2e step progress to stderr), `crates/fnp-conformance/src/lib.rs:21874` (warning when a log line fails parsing in the diagnostic harness), and `crates/fnp-ufunc/src/lib.rs:394` (NumPy-parity `errstate(over='print')` mode — the intended behavior, not debug output). No `dbg!` macros anywhere. |
| `panic!()` in `#[cfg(test)]` oracle/test helpers | **3 in fnp-linalg, plus matching patterns in fnp-ufunc test blocks** | All in test-only code, separate from the stub-flavored row above (which is enforced structurally by `codebase_hygiene::no_not_implemented_panics`). fnp-linalg sites: `crates/fnp-linalg/src/lib.rs:9621`/`9701`/`9744` — three `panic!("unknown oracle payload kind: {kind}")` arms in test-only oracle parsing helpers (inside `#[cfg(test)]` block starting at line 5302, helper fns like `numpy_oracle_pinv_tolerance_aliases`). Several similar patterns exist in fnp-ufunc test blocks (`match err { …, other => panic!("unexpected error: {other:?}") }`). These are deliberate test-failure mechanisms, not stubs. |
| `.expect()` calls in production code (impl crates excl. `fnp-conformance`) | **0** in production paths | Re-verified 2026-05-17 via grep + `#[cfg(test)]`-start cross-check. Per-crate counts (all sit after the crate's `#[cfg(test)]` boundary): fnp-dtype 1, fnp-ndarray 71, fnp-iter 95, fnp-ufunc 768, fnp-linalg 249, fnp-random 430, fnp-io 163, fnp-runtime 3 = ~1,780 total but **zero outside test blocks**. (`fnp-conformance` fixture/oracle code uses `.unwrap()` instead — see prior row.) |

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

## Notes

- The DB is currently jammed (14 concurrent `br create` processes across agent projects plus a lingering `br show franken_numpy-p6qy` zombie from earlier in the session). The three beads drafted above will be batch-filed from this file in a later tick when the DB frees up.
- This audit complements but does not substitute for `fnp-conformance` differential coverage — a file can pass this mock audit and still be parity-incomplete. Differential coverage is tracked separately via the oracle-capture pipeline.
