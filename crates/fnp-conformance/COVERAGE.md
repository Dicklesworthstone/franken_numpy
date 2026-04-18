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
| Iterator/transfer | 22 | 16 | 16 | 54 | Thin - complex NDIter system |
| Shape/stride (SCE) | 22 | — | — | 22 | **Thin** - critical engine |
| Dtype promotion | 15 | — | — | 15 | **Thin** - 15/324 pairs (4.6%) |
| Runtime policy | 12 | — | 8 | 20 | **Thin** - dual-mode system |

## Priority Coverage Gaps

### 1. Dtype Promotion Matrix (CRITICAL)

**Current state:** 15 cases covering only 5 types (bool, i32, i64, f32, f64)
**Required:** 324 pairs from 18 DType variants
**Gap:** Missing u8, u16, u32, u64, i8, i16, f16, complex64, complex128

Key missing edge cases:
- `u64 + any_signed -> f64` (counterintuitive widening rule)
- `f16 + i16 -> f32` (mantissa overflow rule)
- `complex64 + i32 -> complex128` (mirrors float widening)
- All unsigned/signed cross-promotion

### 2. Shape/Stride Calculus Engine (CRITICAL)

**Current state:** 22 cases
**Gap:** SCE is the "non-negotiable compatibility kernel" but has thin coverage

Missing edge cases:
- 0-D arrays (scalar views)
- Empty arrays (shape with 0 dimension)
- Negative strides (reverse views)
- Very large shapes (overflow checks)
- Non-contiguous reshape failures
- Multi-axis transpose permutations
- Broadcast with negative strides

### 3. Runtime Policy Dual-Mode (HIGH)

**Current state:** 20 cases (12 policy + 8 adversarial)
**Gap:** Complex strict/hardened mode-split with risk-aware decisions

Missing coverage:
- All `CompatibilityClass` variants exercised
- Risk score thresholds at boundaries
- Override audit event logging
- Evidence ledger serialization

### 4. Iterator/Transfer System (MEDIUM)

**Current state:** 54 cases
**Gap:** NDIter is a complex state machine

Missing coverage:
- Multi-operand iteration
- External loop chunking
- Seek operations
- Reset mid-iteration
- Overlap copy direction selection

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
