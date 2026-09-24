# FrankenNumPy Divergence Ledger

This ledger is the machine-readable handoff point for diagnostic gates. A diagnostic
case may only mark a behavior as an accepted intentional divergence when the case
references an `intentional` ledger entry here. Ordinary compatibility gaps belong
in `parity_debt` rows with a follow-up bead.

Current policy: no fnp-python diagnostic mismatch (exception class, warning category) is
accepted as an intentional NumPy divergence.

This is the ONLY divergence ledger. Every test that tolerates a NumPy divergence — an
`#[ignore]` whose reason names a parity gap, parity debt, divergence, `DISC-` or upstream
defect, or any `ExpectedFail("...")` — must cite a row id below, and every row must name a
`path.rs::test_fn` probe that exists. `repository_markers_and_ledger_rows_agree` in
`crates/fnp-conformance/src/divergence_ledger.rs` enforces both; it runs in CI G2, and
`run_divergence_ledger` prints the same audit.

**Active rows: 4** (as of 2026-09-24): two intentional (one Hardened-only; one an FMA rounding
contract), one parity debt (host-dependent last-bit sums), one upstream defect fnp declines to
copy. The resolution notes below the table record former entries and why they closed.

| ID | Disposition | Surface | Affected behavior | NumPy scope | Strict behavior | Hardened behavior | Follow-up | Evidence |
|---|---|---|---|---|---|---|---|---|
| DIV-HARDENED-LINALG-NONFINITE | intentional | fnp_python.linalg svd, qr, cholesky, lstsq, solve, inv, det, slogdet, eigh, eigvals, eigvalsh, eig, pinv, matrix_rank (and their top-level aliases) | an operand containing inf or NaN | NumPy answers inconsistently: svd / pinv raise LinAlgError("SVD did not converge"), inv / solve / det / slogdet return NaN-filled results with a RuntimeWarning, an inf-only matrix can yield a finite pinv | NumPy-identical: no check is made, NumPy's outcome is returned | raises LinAlgError("OP: array must not contain infs or NaNs (hardened mode)") before any work and records a `linalg_nonfinite_operand` / `full_validate` runtime decision | deadlock-audit-rc0923-epic-71qy3.10 (further hardened guards) | crates/fnp-python/tests/conformance_runtime_mode.rs::hardened_mode_rejects_nonfinite_linalg_operands_strict_matches_numpy |
| DIV-COV-GRAM-NO-FMA | intentional | fnp_python cov / corrcoef native Gram path (contiguous float64) | entries can differ from NumPy in the last bit (bounded relative deviation 1e-12; observed ~1e-16) | NumPy computes `dot(X, X.T)` through BLAS dgemm, whose micro-kernels are FMA-contracted and chosen by shape, so NumPy's own bytes vary with shape and BLAS build | fnp accumulates without FMA, the workspace-wide bit-reproducibility rule; results are equal within 1e-12 relative | same as strict | none; reopen if NumPy's cov stops depending on the BLAS kernel | crates/fnp-python/tests/conformance_statistics.rs::cov_native_fast_path_matches_numpy_across_shape_ddof_bias, crates/fnp-python/tests/conformance_statistics.rs::cov_corrcoef_long_observation_ufunc_gate_matches_numpy_within_fma_bound |
| PD-F64-FLAT-SUM-ISA | parity_debt | fnp_python nansum / var / std with axis=None over contiguous float64 | on some hosts the result differs from NumPy in the last bit: nansum from n=131072, var/std from ~2M elements and on every N-D flattened input | NumPy's pairwise-sum leaf keeps a number of partial accumulators set by the SIMD width it dispatches to (AVX-512 vs AVX2), so its bits depend on the host | fnp's base_sum_simd reproduces one leaf layout: bit-identical where NumPy dispatches that layout, last-bit different elsewhere (all 12 probe rows pass on a non-AVX-512 host with numpy 2.4.3, 2026-09-24) | same as strict | deadlock-audit-rc0923-epic-71qy3.29 | crates/fnp-python/tests/conformance_var.rs::f64_var_flat_byte_parity_probe_vs_numpy (`#[ignore]`d until .29 lands) |
| UD-F16-SORT-X86SIMDSORT | upstream_drift | fnp_python sort / unique on float16 | fnp returns ascending output where NumPy does not | numpy 2.3.x on AVX-512 hosts: the x86-simd-sort fp16 qsort emits non-ascending output (observed on hz2; fleet workers on numpy 2.4.3 are clean) | fnp output is correctly sorted and byte-equal to NumPy's own float32-widened sort; on hosts without the defect it is byte-equal to NumPy directly | same as strict | deadlock-audit-f7qjf (closed: the affected version left the fleet; the upstream report needs an external account and is not filed) | crates/fnp-python/tests/conformance_sorting.rs::f16_sort_flat_widening_matches_numpy, crates/fnp-python/tests/conformance_unravel_unique.rs::f16_unique_presence_table_bit_exact |

Merged ledger note (2026-09-24, bead deadlock-audit-rc0923-epic-71qy3.16):
`crates/fnp-conformance/DISCREPANCIES.md` was a second ledger with twelve `DISC-` entries.
Each one was re-probed at the `fnp_python` surface against numpy 2.4.3 using the f11c7752 build.
None is an active NumPy divergence:
- DISC-001 (`empty()` zero-fills): NumPy's content is unspecified, so any content matches.
- DISC-002 (Unicode 15.1 width tables): no crate depends on `unicode-width` any more.
- DISC-003 (error message text): ten natively-raised errors (reshape, add, concatenate, sum
  axis, take, matmul, sort axis, transpose, zeros, arange) have byte-identical messages.
  Message parity is checked per surface by the conformance shards, not by a blanket row.
- DISC-004 (multivariate_normal via Cholesky): `Generator.multivariate_normal` delegates to NumPy.
- DISC-005 (multivariate_hypergeometric sequential draws): seed-exact with NumPy for both methods.
- The probe for DISC-004 and DISC-005 is
  `crates/fnp-python/tests/conformance_random.rs::multivariate_distributions_are_seed_exact_with_numpy`.
- DISC-006 (interleaved complex storage) and DISC-007 (f64 arithmetic in `UFuncArray`) describe
  the Rust `fnp-ufunc` API, not NumPy behaviour. At the Python surface, complex multiply and
  int64/uint64 add, multiply, subtract, floor_divide and sum above 2**53 are byte-identical to
  NumPy (see also `conformance_sum.rs::sum_large_integer_flat_parallel_is_bit_exact`).
- DISC-008 was already RESOLVED.
- DISC-009 (svd/qr/norm LAPACK bits) is a delegation policy: those surfaces return NumPy's result.
- DISC-010 (min/max signed-zero ties) and DISC-012 (uint8+int8 promotion): their tests were
  re-enabled and pass.
- DISC-011 (signed-zero accumulation in dot/inner/vdot/matmul/tensordot): all five of its
  `#[ignore]`d tests pass at the Python surface and were re-enabled in this bead.

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
