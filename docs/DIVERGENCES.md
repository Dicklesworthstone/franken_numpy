# FrankenNumPy Divergence Ledger

This ledger is the machine-readable handoff point for diagnostic gates. A diagnostic
case may only mark a behavior as an accepted intentional divergence when the case
references an `intentional` ledger entry here. Ordinary compatibility gaps belong
in `parity_debt` rows with a follow-up bead.

Current policy: no fnp-python diagnostic mismatch is accepted as an intentional
NumPy divergence. The table below lists active tracked parity debt discovered
while building the diagnostic parity wave.

**Active rows: 1** (as of 2026-05-16). The two resolution notes below the table
record beads that previously held entries; they remain for provenance.

| ID | Disposition | Surface | Affected behavior | NumPy scope | Strict behavior | Hardened behavior | Follow-up | Evidence |
|---|---|---|---|---|---|---|---|---|
| `franken_numpy-ucc2o` | parity_debt | `fnp-random` `SeedMaterial::None` / no-seed `default_rng()` | Sourcing entropy for an unseeded RNG | NumPy's `default_rng()` sources from OS entropy (getrandom / CryptGenRandom) → fresh sequence per process | Uses fixed `DEFAULT_RNG_SEED = 0xC0DE_CAFE_F00D_BAAD` (deterministic) | Same as strict | Decide whether to add `getrandom` as an external crates.io dep (fnp-random currently has none — only intra-workspace fnp-ndarray) or keep deterministic-default and document loudly | `crates/fnp-random/src/lib.rs:256` (verified 2026-05-16, byte-for-byte match incl. underscore placement), `:1997, :5354`; README RNG State Serialization section updated 2026-05-16 to flag the divergence |

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
