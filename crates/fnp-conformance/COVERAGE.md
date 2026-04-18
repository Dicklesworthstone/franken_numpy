# Conformance Coverage Matrix

> Generated: 2026-04-18 | Status: Living document tracking differential conformance against NumPy oracle

## Coverage Accounting

| Domain | Differential | Metamorphic | Adversarial | Total Cases | Coverage Assessment |
|--------|:-----------:|:-----------:|:-----------:|:-----------:|---------------------|
| Ufunc operations | 381 | 25 | 60 | 466 | Good - core ops well covered |
| Signal processing | 295 | — | — | 295 | Good - extensive conv/corr/fft |
| Polynomials | 88 | — | — | 88 | Good - all 5 families |
| I/O (npy/npz/text) | 66 | 35 | 33 | 134 | Good - parser boundaries covered |
| Linear algebra | 56 | 25 | 35 | 116 | Adequate - core decompositions |
| String arrays | 52 | — | — | 52 | Adequate - 33 char functions |
| RNG | 22 | 23 | 41 | 86 | Adequate + 39 statistical |
| FFT | 30 | — | — | 30 | Adequate - transform families |
| Datetime/timedelta | 34 | — | — | 34 | Adequate - arithmetic + busday |
| Masked arrays | 27 | — | — | 27 | Adequate - reshape/concat/fill |
| Iterator/transfer | 31 | 16 | 16 | 63 | Adequate - transfer/overlap/flatiter edges covered |
| Shape/stride (SCE) | 36 | — | — | 36 | Adequate - 0-D, empty, negative-stride covered |
| Dtype promotion | 198 | — | — | 198 | Good - full 14×14 type matrix coverage |
| Runtime policy | 23 | — | 8 | 31 | Adequate - risk thresholds, boundaries covered |

## Priority Coverage Gaps

### 1. Dtype Promotion Matrix (COMPLETE)

**Current state:** 198 cases covering full 14×14 type matrix
**Coverage:** 100% of type pairs (bool, int8-64, uint8-64, float16-64, complex64-128)
**Generated:** Programmatically via NumPy's np.result_type oracle

No remaining gaps.

### 2. Shape/Stride Calculus Engine (MEDIUM)

**Current state:** 36 cases covering core edge cases
**Covered:** 0-D scalars, empty arrays, negative strides, Fortran order, various item sizes

Remaining gaps:
- Very large shapes (overflow checks)
- Non-contiguous reshape failures
- Multi-axis transpose permutations beyond 4D

### 3. Runtime Policy Dual-Mode (MEDIUM)

**Current state:** 31 cases (23 policy + 8 adversarial)
**Covered:** All CompatibilityClass variants, risk thresholds at boundaries (0.0, 0.69, 0.7, 0.71, 1.0), variable thresholds

Remaining gaps:
- Override audit event logging
- Evidence ledger serialization

### 4. Iterator/Transfer System (LOW)

**Current state:** 63 cases (31 diff + 16 meta + 16 adversarial)
**Covered:** Transfer class selection, overlap copy direction, flatiter read/write, error cases

Remaining gaps:
- Multi-operand iteration
- External loop chunking
- Seek/reset mid-iteration

## Fixture Provenance

| Fixture | Generator | Version | Last Regenerated |
|---------|-----------|---------|------------------|
| ufunc_input_cases.json | capture_numpy_oracle | NumPy 2.x | 2026-04-15 |
| signal_differential_cases.json | capture_numpy_oracle | NumPy 2.x | 2026-04-15 |
| polynomial_differential_cases.json | capture_numpy_oracle | NumPy 2.x | 2026-04-15 |
| dtype_promotion_cases.json | manual | — | 2026-03-12 |
| shape_stride_cases.json | manual | — | 2026-04-11 |
| runtime_policy_cases.json | manual | — | 2026-03-12 |

## Test-to-Fixture Mapping

| Test Function | Fixture Source | Pass Criteria |
|---------------|----------------|---------------|
| `test_ufunc_differential_*` | ufunc_input_cases.json | Shape + dtype + values match oracle |
| `test_dtype_promotion_*` | dtype_promotion_cases.json | `promote(lhs, rhs) == expected` |
| `test_shape_stride_*` | shape_stride_cases.json | Broadcast shape + strides match |
| `test_rng_differential_*` | rng_differential_cases.json | Bit-exact sequence match |
| `test_io_differential_*` | io_differential_cases.json | Parse + roundtrip match |

## Conformance Targets

Per the testing-conformance-harnesses skill:
- **MUST clause coverage target:** ≥95%
- **Current MUST coverage:** ~85% (estimated from fixture gaps)
- **Score < 0.95 = NOT conformant** (shipping with known gaps documented)

## Next Actions

1. [ ] Expand dtype_promotion_cases.json to cover all 18 DType pairs systematically
2. [ ] Add shape_stride_cases for 0-D, empty, negative-stride edge cases
3. [ ] Add runtime_policy_cases for all CompatibilityClass variants
4. [ ] Add iterator differential cases for seek/reset/external-loop
5. [ ] Wire up automated fixture regeneration from oracle
6. [ ] Add CI coverage tracking (case count trends)

## References

- [FEATURE_PARITY.md](../../FEATURE_PARITY.md) - Overall parity status
- [fixtures/README.md](fixtures/README.md) - Fixture file descriptions
- [DISCREPANCIES.md](DISCREPANCIES.md) - Intentional divergences from NumPy
