# FrankenNumPy Divergence Ledger

This ledger is the machine-readable handoff point for diagnostic gates. A diagnostic
case may only mark a behavior as an accepted intentional divergence when the case
references an `intentional` ledger entry here. Ordinary compatibility gaps belong
in `parity_debt` rows with a follow-up bead.

Current policy: no fnp-python diagnostic mismatch is accepted as an intentional
NumPy divergence. The rows below are tracked parity debt discovered while building
the diagnostic parity wave.

| ID | Disposition | Surface | Affected behavior | NumPy scope | Strict behavior | Hardened behavior | Follow-up | Evidence |
|---|---|---|---|---|---|---|---|---|
| `PD-2F6L4` | parity_debt | fnp-python arithmetic/statistics diagnostics | Native fnp-python paths suppress selected NumPy `RuntimeWarning` categories for divide/remainder/fmod float division by zero, empty mean/var, and all-NaN nanmean/nanstd cases. | NumPy 2.x warning category/count/order behavior. | Parity debt: strict mode must eventually emit the same warning categories in the same order. | Same parity debt; hardened mode may add bounded audit context but must not hide NumPy warnings. | `franken_numpy-2f6l4` | `crates/fnp-python/tests/conformance_diagnostics.rs`; `franken_numpy-33vtd.4` |
| `PD-09EPN` | parity_debt | fnp-python indexing/text IO diagnostics | Candidate diagnostic cases diverged for `take(..., mode="not-a-mode")`, `compress(..., axis out of bounds)`, and `loadtxt(io.StringIO("a b"))`. | NumPy 2.x exception class and warning/error behavior. | Parity debt: strict mode must preserve NumPy exception classes and text parsing failure behavior. | Same parity debt; hardened mode may add bounded diagnostics but must preserve public exception class behavior. | `franken_numpy-09epn` | `crates/fnp-python/tests/conformance_diagnostics.rs`; `franken_numpy-33vtd.4` |

## Checker

Run the ledger gate with:

```bash
rch exec -- cargo run -p fnp-conformance --bin run_divergence_ledger -- --fail-on-missing
```

The checker also accepts `--case-json <path>` for diagnostic oracle case files. Any
case that sets `intentional_divergence` must reference an `intentional` ledger entry,
otherwise the gate fails closed.
