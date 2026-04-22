# fnp-* mock-code-finder audit — 2026-04-22

Scanned `fnp-dtype`, `fnp-ndarray`, `fnp-iter`, `fnp-ufunc`, `fnp-linalg`, `fnp-random`, `fnp-io`, `fnp-runtime`, `fnp-conformance`, `fnp-python` — production code only (`#[cfg(test)]` blocks excluded). Audit by CC agent in franken_numpy swarm, invoking the `mock-code-finder` skill.

## Summary

**Zero real stubs/mocks/TODOs in the numpy implementation.** AGENTS.md characterises remaining gaps as "parity debt, not feature cuts"; this audit confirms that at the code level. The only findings are ~43 cosmetic `.unwrap()` calls in fixture/parser code that should be `.expect("...")` for better panic diagnostics.

## Detection matrix

| Detection | Count | Notes |
|---|---|---|
| `TODO` / `FIXME` / `HACK` / `XXX` / `STUB` / `PLACEHOLDER` / `MOCK` / `DUMMY` / `FAKE` keywords | **0** | Only hit across all crates: `crates/fnp-conformance/src/bin/validate_phase2c_packet.rs:41` — `<FNP-P2C-XXX>` inside a CLI usage help string (user-input placeholder, not code). |
| `todo!()` / `unimplemented!()` / `panic!("not implemented")` | **0** | — |
| Empty function bodies / `Ok(())`-only stub returns | **0** | — |
| `sleep()` / `thread::sleep` / fake-work patterns | **0** | One legitimate `std::time::Duration::from_secs(10)` in `crates/fnp-conformance/src/raptorq_artifacts.rs:245` (RaptorQ `block_timeout` config value). |
| pyo3 / numpy delegation stubs in impl crates | **0** | `fnp-python` delegates by design (it is the parity oracle surface); every other `fnp-*` crate is self-contained Rust. |
| `numpy.*` references in impl crates | docstrings + 1 legitimate embed | All references are either doc comments explaining numpy semantics, or the embedded-Python snippet in `fnp-io` that reads `numpy.lib.format.open_memmap` for memmap parity. |
| Production `.unwrap()` — impl crates (excl. `fnp-conformance`) | **1** | See §A below. |
| Production `.unwrap()` — `fnp-conformance` fixture/oracle code | **42** | See §B below. |
| AST-grep scan for suspiciously short functions (`fn $NAME($$$) -> $RET { $SINGLE }`) | hits are all legitimate | `default()` constructors, `Display::fmt`, simple accessors like `all_numeric_dtypes()` and `is_malformed_probability_input`. None are stubs. |

## Findings (draft beads — to file when DB contention clears)

### A. Bead 1 — single einsum parser unwrap

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
# Keyword scan
rg -n --type rust "TODO|FIXME|HACK|XXX|STUB|PLACEHOLDER|MOCK|DUMMY|FAKE" \
  crates/fnp-dtype crates/fnp-ndarray crates/fnp-iter crates/fnp-ufunc \
  crates/fnp-linalg crates/fnp-random crates/fnp-io crates/fnp-conformance \
  crates/fnp-runtime

# Unimplemented macros
rg -n --type rust "unimplemented!|todo!\(|panic!\(\"not implemented" crates/fnp-*/src

# Production unwrap count per file (strips from first #[cfg(test)])
for f in crates/fnp-*/src/*.rs crates/fnp-*/src/bin/*.rs; do
  [ -f "$f" ] || continue
  prod_unwraps=$(awk '/^#\[cfg\(test\)\]/ {exit} /\.unwrap\(\)/' "$f" | wc -l)
  [ "$prod_unwraps" -gt 0 ] && echo "$prod_unwraps $f"
done | sort -rn

# Structural scan
ast-grep run -l Rust -p 'fn $NAME($$$) -> $RET { $SINGLE }' --json \
  | jq -r '.[] | select(.file | test("crates/fnp-(dtype|ndarray|iter|ufunc|linalg|random|io|runtime)/src")) | "\(.file):\(.range.start.line)"'
```

## Notes

- The DB is currently jammed (14 concurrent `br create` processes across agent projects plus a lingering `br show franken_numpy-p6qy` zombie from earlier in the session). The three beads drafted above will be batch-filed from this file in a later tick when the DB frees up.
- This audit complements but does not substitute for `fnp-conformance` differential coverage — a file can pass this mock audit and still be parity-incomplete. Differential coverage is tracked separately via the oracle-capture pipeline.
