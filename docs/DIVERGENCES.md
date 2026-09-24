# FrankenNumPy Divergence Ledger

This ledger is the machine-readable handoff point for diagnostic gates. A diagnostic
case may only mark a behavior as an accepted intentional divergence when the case
references an `intentional` ledger entry here. Ordinary compatibility gaps belong
in `parity_debt` rows with a follow-up bead.

Current policy: no fnp-python diagnostic mismatch is accepted as an intentional
NumPy divergence. The table below lists active tracked parity debt discovered
while building the diagnostic parity wave.

**Active rows: 1** (as of 2026-09-24): one intentional divergence that exists ONLY in Hardened
mode; Strict mode has none. The resolution notes below the table record beads that previously
held entries; they remain for provenance.

| ID | Disposition | Surface | Affected behavior | NumPy scope | Strict behavior | Hardened behavior | Follow-up | Evidence |
|---|---|---|---|---|---|---|---|---|
| DIV-HARDENED-LINALG-NONFINITE | intentional | fnp_python.linalg svd, qr, cholesky, lstsq, solve, inv, det, slogdet, eigh, eigvals, eigvalsh, eig, pinv, matrix_rank (and their top-level aliases) | an operand containing inf or NaN | NumPy answers inconsistently: svd / pinv raise LinAlgError("SVD did not converge"), inv / solve / det / slogdet return NaN-filled results with a RuntimeWarning, an inf-only matrix can yield a finite pinv | NumPy-identical: no check is made, NumPy's outcome is returned | raises LinAlgError("OP: array must not contain infs or NaNs (hardened mode)") before any work and records a `linalg_nonfinite_operand` / `full_validate` runtime decision | deadlock-audit-rc0923-epic-71qy3.10 (further hardened guards) | crates/fnp-python/tests/conformance_runtime_mode.rs::hardened_mode_rejects_nonfinite_linalg_operands_strict_matches_numpy |

Resolved no-seed RNG note: `franken_numpy-iqo31` changed `SeedMaterial::None`
and no-seed `default_rng()` from the fixed `DEFAULT_RNG_SEED` stream to a fresh
`SeedSequence` initialized from OS entropy. Explicit seed material remains
deterministic and bit-for-bit reproducible.

Resolved warning-debt note: `franken_numpy-2f6l4` restored diagnostic coverage
for divide/remainder/mod/fmod zero-divisor warnings, empty mean/var warnings,
and all-NaN nanmean/nanstd warnings. It no longer has an active parity-debt row.

Resolved indexing/text-IO diagnostic note: `franken_numpy-09epn` restored
diagnostic coverage for `take(..., mode="not-a-mode")`, `compress(..., axis out
of bounds)`, and `loadtxt(io.StringIO("a b"))`. It no longer has an active
parity-debt row.

## Checker

Run the ledger gate with:

```bash
rch exec -- cargo run -p fnp-conformance --bin run_divergence_ledger -- --fail-on-missing
```

The checker also accepts `--case-json <path>` for diagnostic oracle case files. Any
case that sets `intentional_divergence` must reference an `intentional` ledger entry,
otherwise the gate fails closed.
