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

**Active rows: 9** (as of 2026-10-08): seven intentional (four Hardened-only; three rounding
contracts: cov/corrcoef, einsum and the stacked linalg kernels) and two upstream defects fnp declines
to copy. The resolution notes below the table record former entries and why they closed.

| ID | Disposition | Surface | Affected behavior | NumPy scope | Strict behavior | Hardened behavior | Follow-up | Evidence |
|---|---|---|---|---|---|---|---|---|
| DIV-HARDENED-LINALG-NONFINITE | intentional | fnp_python.linalg svd, qr, cholesky, lstsq, solve, inv, det, slogdet, eigh, eigvals, eigvalsh, eig, pinv, matrix_rank (and their top-level aliases) | an operand containing inf or NaN | NumPy answers inconsistently: svd / pinv raise LinAlgError("SVD did not converge"), inv / solve / det / slogdet return NaN-filled results with a RuntimeWarning, an inf-only matrix can yield a finite pinv | NumPy-identical: no check is made, NumPy's outcome is returned | raises LinAlgError("OP: array must not contain infs or NaNs (hardened mode)") before any work and records a `linalg_nonfinite_operand` / `full_validate` runtime decision | deadlock-audit-rc0923-epic-71qy3.10 (further hardened guards) | crates/fnp-python/tests/conformance_runtime_mode.rs::hardened_mode_rejects_nonfinite_linalg_operands_strict_matches_numpy |
| DIV-HARDENED-ADMISSION-CAP | intentional | fnp_python creation routines empty, zeros, ones, full, zeros_like / ones_like / full_like (with `shape=`), eye, identity, tri, arange, linspace, logspace, geomspace, indices, repeat, tile, resize | one requested array larger than the hardened admission cap (default 4 GiB; `FNP_HARDENED_MAX_ARRAY_BYTES`, a positive byte count - anything else fails the import) | NumPy attempts the allocation and raises MemoryError only if the OS refuses it; under Linux overcommit a large `empty` / `zeros` usually succeeds and the process dies later, when its pages are touched | NumPy-identical: no check is made | raises MemoryError("OP: requested array of N bytes exceeds the hardened admission cap of M bytes (FNP_HARDENED_MAX_ARRAY_BYTES)") before allocating and records an `admission_cap_exceeded` / `full_validate` runtime decision; spec section 16 ("Shape bomb": hardened enforces stricter admission caps) | deadlock-audit-rc0923-epic-71qy3.10 (further hardened guards) | crates/fnp-python/tests/conformance_runtime_mode.rs::hardened_mode_caps_shape_bomb_allocations_strict_matches_numpy |
| DIV-HARDENED-LOAD-BOUNDS | intentional | fnp_python.load of `.npy` and `.npz`, from a path or a file-like object, with or without mmap_mode | a file that violates an fnp-io bound: bad magic or version, a header over 65,536 bytes or not a well-formed header dictionary, a shape of rank > 32 or an overflowing element count, a payload shorter than the header declares (checked where the source length is known: a path, a real file object, a BytesIO), an `.npz` with more than 4,096 members or more than 2 GiB decoded, or one failing fnp-io's ZIP checks (local/central disagreement, overlap, traversal names, a method other than STORE/DEFLATE, a bad CRC-32, data descriptors) | NumPy reads the header with its own parser and ALLOCATES THE DECLARED ARRAY BEFORE READING THE PAYLOAD (a 16-byte payload under a 2^40-element header asks for 8 TiB); its `zipfile` reader opens any member count and inflates any declared size | NumPy-identical: no check is made (a path `.npy` still decodes natively, but anything its header cannot promise is NumPy's) | raises ValueError("load: refused by the hardened fnp-io bounds (REASON): DETAIL") before NumPy sees the bytes and records a runtime decision under fnp-io's reason code (`io_header_schema_invalid`, `io_read_payload_incomplete`, `io_npz_archive_contract_violation`, ...) / `full_validate`; a pickle or any non-npy/npz file stays NumPy's | none | crates/fnp-python/tests/conformance_io.rs::hardened_load_enforces_fnp_io_bounds_and_strict_load_matches_numpy |
| DIV-HARDENED-SPAWN-BUDGET | intentional | fnp_python.random SeedSequence.spawn, and every bit generator's spawn (which calls it) | more than 4096 children in one spawn call | NumPy spawns any count (its loop index is a uint32_t) | NumPy-identical: the children, spawn keys and draws numpy produces (the packet-007 cap applied in every mode until deadlock-audit-r8eqg) | raises ValueError("SeedSequence.spawn(N): more than 4096 children per call exceeds the hardened spawn budget (packet-007)") and records an `rng_seedsequence_spawn_contract_violation` / `full_validate` runtime decision - the cap artifacts/phase2c/FNP-P2C-007/risk_note.md lists under "Explicit bounded caps (hardened policy path)" | deadlock-audit-rc0923-epic-71qy3.10 (further hardened guards) | crates/fnp-python/tests/conformance_runtime_mode.rs::hardened_mode_keeps_the_seed_sequence_spawn_budget, crates/fnp-python/tests/conformance_random.rs::seed_sequence_spawns_past_the_packet_budget_like_numpy |
| DIV-COV-GRAM-NO-FMA | intentional | fnp_python cov / corrcoef native Gram path (contiguous float64, and integer inputs, which numpy and fnp both compute in float64), engaged only where it beats numpy: small Gram work (n_vars < 48), 256 <= n_vars < 512 with n_obs >= 2048, and n_vars >= 512 only when numpy's BLAS is serial; every other shape is numpy's and byte-identical | entries can differ from NumPy in the last bits: below 256 variables within 1e-12 relative per entry; from 256 variables within 1e-12 of sqrt(c_ii * c_jj) (observed <= 1.3e-15), since near-zero off-diagonal entries make a per-entry relative bound meaningless (2.0e-11 at 256x256) | NumPy computes `dot(X, X.T)` through BLAS dgemm, whose micro-kernels are FMA-contracted and chosen by shape, so NumPy's own bytes vary with shape and BLAS build | fnp accumulates without FMA, the workspace-wide bit-reproducibility rule; results are equal within the bounds above | same as strict | none; reopen if NumPy's cov stops depending on the BLAS kernel | crates/fnp-python/tests/conformance_statistics.rs::cov_native_fast_path_matches_numpy_across_shape_ddof_bias, crates/fnp-python/tests/conformance_statistics.rs::cov_corrcoef_large_n_vars_gate_delegates_losses_and_bounds_native_gram |
| DIV-BATCHED-LINALG-NO-LAPACK | intentional | fnp_python.linalg on a finite float64 STACK (ndim >= 3, ndarray or nested list) inside each route's measured win gate: inv (n < 16, or batch < 256), solve (batch >= 1024, or >= 256 with n <= 4, or a broadcast A), eigvalsh (n <= 6, or >= 2^18 elements), cholesky (n <= 3), pinv (n <= 24, or >= 2^18 elements), svdvals / cond / matrix_rank / norm and matrix_norm with ord 2, -2 or 'nuc' (n <= 8, or >= 2^18 elements) | values can differ from NumPy in the last bits, within a norm-wise bound per lane (C = 32): inv, solve, pinv, cholesky, cond, norm -2 within C * cond2(A) * k * eps * max(abs(numpy result)); eigvalsh, svdvals, norm 2, 'nuc' within C * k * eps * norm2(A) (measured worst 6.15 / 3.0 of the C = 1 scale); matrix_rank equal on generic input and may differ when the smallest singular value is within about 1x of the tolerance; type, dtype, shape, layout and exceptions match | NumPy loops LAPACK per lane (getrf/getri, gesv, syevd, potrf, gesdd), whose blocked, FMA-contracted operation order depends on the BLAS build | fnp's per-lane kernels factor in their own order without FMA (the workspace bit-reproducibility rule), in parallel across lanes; det, slogdet, eigh, tensorinv and non-finite stacks are NumPy's (their native kernels returned wrong answers, bead deadlock-audit-41n96) | same as strict | none; reopen if a stacked route needs NumPy's exact bytes | crates/fnp-python/tests/conformance_linalg_decomp.rs::stacked_native_linalg_stays_within_the_divergence_bounds |
| DIV-EINSUM-FLOAT-NO-FMA | intentional | fnp_python einsum with optimize=False and a float64 result, two or more operands with a summed index (matrix-product, dot, row-dot and transposed-product spellings); and np.matvec / np.vecmat of a float64 2-D matrix and 1-D vector, which run the same contraction kernel (bead deadlock-audit-6y5wp; found unledgered 2026-10-08) | the value can differ from NumPy in the last bits (bounded: 1e-12 of the largest magnitude); type, dtype, shape and layout match | NumPy's einsum inner loops (`sum_of_products_*` in einsum_sumprod.c.src) accumulate with `npyv_muladd` - FMA-contracted, in lanes whose count follows the host ISA | fnp contracts without FMA, the workspace-wide bit-reproducibility rule. A float32 or complex result is NumPy's own call (float32 was computed in float64 and rounded once until bead deadlock-audit-vc4p4); optimize=True calls, integer / bool / float16 / float32 / complex operands and every other product function (matmul, dot, inner, vdot, tensordot, kron, trace) are byte-identical to NumPy | same as strict | none; reopen if NumPy's einsum stops depending on FMA and SIMD width | crates/fnp-python/tests/conformance_einsum.rs::matrix_products_and_einsum_match_numpy_bytes_or_the_documented_fma_bound |
| UD-F16-SORT-X86SIMDSORT | upstream_drift | fnp_python sort / unique on float16 | fnp returns ascending output where NumPy does not | numpy 2.3.x on AVX-512 hosts: the x86-simd-sort fp16 qsort emits non-ascending output (observed on hz2; fleet workers on numpy 2.4.3 are clean) | fnp output is correctly sorted and byte-equal to NumPy's own float32-widened sort; on hosts without the defect it is byte-equal to NumPy directly | same as strict | deadlock-audit-f7qjf (closed: the affected version left the fleet; the upstream report needs an external account and is not filed) | crates/fnp-python/tests/conformance_sorting.rs::f16_sort_flat_widening_matches_numpy, crates/fnp-python/tests/conformance_unravel_unique.rs::f16_unique_presence_table_bit_exact |
| UD-LEGACY-ZIPF-LARGE-A | upstream_drift | fnp_python.random.RandomState.zipf, scalar and array `a` | `a >= 1025`, `inf` included | numpy's `legacy_random_zipf` (installed 2.4.3 and the vendored source) never returns there: `b = 2**(a - 1)` overflows, every candidate is 1 and its acceptance test is NaN, so the call spins until the process is killed | 1 for each such entry, without a draw, as numpy's modern `random_zipf` answers the same `a` (its guard notes the chance of a value above 1 is below 3e-309); below 1025 the draws are numpy's legacy draws byte for byte | same as strict | none; reopen if numpy's legacy kernel gains the modern guard | crates/fnp-python/tests/conformance_random.rs::legacy_zipf_past_a_1025_answers_one_where_numpy_never_returns |

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

Resolved PD-F64-FLAT-SUM-ISA (2026-09-24, bead deadlock-audit-rc0923-epic-71qy3.29): the
host-dependent last-bit sums were a NumPy VERSION effect, not a SIMD-width one. NumPy's pairwise
leaf is fixed C code (eight accumulators, `loops_utils.h.src`); what changed is that NumPy before
2.3 sums a contiguous run in 8192-element buffer chunks and 2.3+ sums it as one tree. On one host
numpy 2.2.6 and 1.26.4 differ from 2.3.5/2.4.3/2.4.4 from n = 8193 on. Every native route that
evaluates the tree now consults the runtime witness `float_pairwise_tree_matches_numpy` (only flat
sum/mean did before) and declines to NumPy where it fails, and the nan-reductions' sequential
cold path now delegates too. The probe is un-ignored as
`crates/fnp-python/tests/conformance_var.rs::pairwise_tree_reductions_are_bit_identical_to_the_installed_numpy`
(99 cells: 0 mismatches under numpy 2.4.3 and 2.2.6; 40 under 2.2.6 before the fix).

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
