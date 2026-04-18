# Conformance Coverage Matrix

> Generated: 2026-04-18 | Status: Living document tracking differential conformance against NumPy oracle

## Coverage Accounting

| Domain | Differential | Metamorphic | Adversarial | Total Cases | Coverage Assessment |
|--------|:-----------:|:-----------:|:-----------:|:-----------:|---------------------|
| Ufunc operations | 381 | 36 | 60 | 477 | Good - core ops, associativity, distributivity, identity, inverse |
| Signal processing | 295 | — | — | 295 | Good - extensive conv/corr/fft |
| Polynomials | 88 | — | — | 88 | Good - all 5 families |
| I/O (npy/npz/text) | 66 | 35 | 33 | 134 | Good - parser boundaries covered |
| Linear algebra | 63 | 35 | 35 | 133 | Good - decompositions, inv identity, det transpose, eig trace |
| String arrays | 60 | — | — | 60 | Good - 33 char functions, empty/whitespace edge cases |
| RNG | 33 | 23 | 41 | 97 | Good - distributions, shuffle, choice + 39 statistical |
| FFT | 38 | — | — | 38 | Good - transform families, edge sizes, 2D non-square |
| Datetime/timedelta | 41 | — | — | 41 | Good - arithmetic, busday, NaT, broadcast, abs |
| Masked arrays | 36 | — | — | 36 | Good - reshape/concat/fill/broadcast/all-masked/argmin-max |
| Iterator/transfer | 42 | 16 | 16 | 74 | Good - multi-operand, external loop, seek/reset covered |
| Shape/stride (SCE) | 47 | — | — | 47 | Good - 0-D, empty, negative-stride, large shapes, 5D/6D transpose covered |
| Dtype promotion | 198 | — | — | 198 | Good - full 14×14 type matrix coverage |
| Runtime policy | 23 | — | 15 | 38 | Good - adversarial edge cases, clamping, injection attempts |

## Priority Coverage Gaps

### 1. Dtype Promotion Matrix (COMPLETE)

**Current state:** 198 cases covering full 14×14 type matrix
**Coverage:** 100% of type pairs (bool, int8-64, uint8-64, float16-64, complex64-128)
**Generated:** Programmatically via NumPy's np.result_type oracle

No remaining gaps.

### 2. Shape/Stride Calculus Engine (COMPLETE)

**Current state:** 47 cases covering comprehensive edge cases
**Covered:** 0-D scalars, empty arrays, negative strides, Fortran order, various item sizes, large shapes (4G+ elements), 5D/6D transpose permutations, non-contiguous views

No remaining gaps.

### 3. Runtime Policy Dual-Mode (MEDIUM)

**Current state:** 31 cases (23 policy + 8 adversarial)
**Covered:** All CompatibilityClass variants, risk thresholds at boundaries (0.0, 0.69, 0.7, 0.71, 1.0), variable thresholds

Remaining gaps:
- Override audit event logging
- Evidence ledger serialization

### 4. Iterator/Transfer System (COMPLETE)

**Current state:** 74 cases (42 diff + 16 meta + 16 adversarial)
**Covered:** Transfer class selection, overlap copy direction, flatiter read/write, error cases, multi-operand broadcast iteration, external loop chunking, seek/reset mid-iteration

No remaining gaps.

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

1. [x] Expand dtype_promotion_cases.json to cover all 14×14 type matrix (198 cases)
2. [x] Add shape_stride_cases for 0-D, empty, negative-stride, 5D/6D transpose (47 cases)
3. [x] Add runtime_policy_cases for all CompatibilityClass variants + adversarial (38 cases)
4. [x] Add iterator differential cases for seek/reset/external-loop/multi-operand (74 cases)
5. [ ] Wire up automated fixture regeneration from oracle
6. [ ] Add CI coverage tracking (case count trends)

## References

- [FEATURE_PARITY.md](../../FEATURE_PARITY.md) - Overall parity status
- [fixtures/README.md](fixtures/README.md) - Fixture file descriptions
- [DISCREPANCIES.md](DISCREPANCIES.md) - Intentional divergences from NumPy
