# FrankenNumPy Divergence Ledger

This ledger is the machine-readable handoff point for diagnostic gates. A diagnostic
case may only mark a behavior as an accepted intentional divergence when the case
references an `intentional` ledger entry here. Ordinary compatibility gaps belong
in `parity_debt` rows with a follow-up bead.

Current policy: no fnp-python diagnostic mismatch is accepted as an intentional
NumPy divergence. The rows below are active tracked parity debt discovered while
building the diagnostic parity wave.

| ID | Disposition | Surface | Affected behavior | NumPy scope | Strict behavior | Hardened behavior | Follow-up | Evidence |
|---|---|---|---|---|---|---|---|---|
| `PD-09EPN` | parity_debt | fnp-python indexing/text IO diagnostics | Candidate diagnostic cases diverged for `take(..., mode="not-a-mode")`, `compress(..., axis out of bounds)`, and `loadtxt(io.StringIO("a b"))`. | NumPy 2.x exception class and warning/error behavior. | Parity debt: strict mode must preserve NumPy exception classes and text parsing failure behavior. | Same parity debt; hardened mode may add bounded diagnostics but must preserve public exception class behavior. | `franken_numpy-09epn` | `crates/fnp-python/tests/conformance_diagnostics.rs`; `franken_numpy-33vtd.4` |

Resolved warning-debt note: `franken_numpy-2f6l4` restored diagnostic coverage
for divide/remainder/mod/fmod zero-divisor warnings, empty mean/var warnings,
and all-NaN nanmean/nanstd warnings. It no longer has an active parity-debt row.

## Checker

Run the ledger gate with:

```bash
rch exec -- cargo run -p fnp-conformance --bin run_divergence_ledger -- --fail-on-missing
```

The checker also accepts `--case-json <path>` for diagnostic oracle case files. Any
case that sets `intentional_divergence` must reference an `intentional` ledger entry,
otherwise the gate fails closed.
