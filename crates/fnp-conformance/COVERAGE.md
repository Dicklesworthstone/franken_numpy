# Conformance Coverage Matrix

> Generated: 2026-04-18 | Status: Living document tracking differential conformance against NumPy oracle

## Coverage Accounting

| Domain | Differential | Metamorphic | Adversarial | Total Cases | Coverage Assessment |
|--------|:-----------:|:-----------:|:-----------:|:-----------:|---------------------|
| Ufunc operations | 381 | 33 | 62 | 476 | Good - core ops well covered, 3D commutative, identity scalar |
| Signal processing | 305 | 12 | — | 317 | Good - extensive conv/corr/fft, atleast_nd, reciprocal/sign edge cases, convolve commutative/distributive |
| Polynomials | 96 | 12 | — | 108 | Good - all 5 families, edge cases, add-zero, sub-self, int/der edge, commutative/identity metamorphic |
| I/O (npy/npz/text) | 70 | 40 | 35 | 145 | Good - parser boundaries, complex128, quoted fields, F-order, single-member npz |
| Linear algebra | 67 | 30 | 37 | 134 | Good - core decompositions, 1x1 edge cases, identity, wide QR, fractional solve |
| String arrays | 71 | 12 | — | 83 | Good - 33 char functions, empty/pattern edge cases, case/title conversion, isdigit, idempotence/involution metamorphic |
| RNG | 26 | 28 | 43 | 97 | Adequate + 39 statistical, small bound, zero-fill, single-step jump |
| FFT | 47 | 12 | 12 | 71 | Good - transform families, edge sizes, 2D/3D, fftn/ifftn, shifts, fftfreq, inverse/linearity/parseval metamorphic, edge-case adversarial |
| Datetime/timedelta | 48 | 12 | — | 60 | Good - arithmetic, busday, NaT, broadcast, abs, comparisons, neg, mul-zero, add/sub inverse metamorphic |
| Masked arrays | 44 | 12 | — | 56 | Good - reshape/concat/fill/broadcast/all-masked/argmin-max/axis-aware, min/max axis partial, mask preservation metamorphic |
| Iterator/transfer | 37 | 21 | 18 | 76 | Good - transfer/overlap/flatiter/broadcast, single-element, copy-overlap flags |
| Shape/stride (SCE) | 51 | 12 | — | 63 | Good - 0-D, empty, negative-stride, large shapes, 5D/6D transpose, scalar-to-5D, F-order, reshape/transpose roundtrip metamorphic |
| Dtype promotion | 198 | 17 | 20 | 235 | Good - full 14×14 type matrix + metamorphic + adversarial, title-case/trailing-space |
| Runtime policy | 37 | — | 12 | 49 | Good - risk thresholds, boundaries, override audit, unknown-class, exact-threshold |

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

### 3. Runtime Policy Dual-Mode (COMPLETE)

**Current state:** 43 cases (23 policy + 8 adversarial + 12 override audit)
**Covered:** All CompatibilityClass variants, risk thresholds at boundaries (0.0, 0.69, 0.7, 0.71, 1.0), variable thresholds, override audit events

No remaining gaps.

### 4. Iterator/Transfer System (GOOD)

**Current state:** 65 cases (33 diff + 16 meta + 16 adversarial) + oracle smoke tests
**Covered:** Transfer class selection, overlap copy direction, flatiter read/write, multi-operand broadcast planning, error cases
**Oracle-verified in smoke.rs:** External loop chunking, seek via set_multi_index/set_iterindex

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
- **Current MUST coverage:** 100% (155/155 conformance tests passing)
- **Score < 0.95 = NOT conformant** (shipping with known gaps documented)

## Next Actions

1. [x] ~~Expand dtype_promotion_cases.json to cover all 18 DType pairs systematically~~ (DONE: 198 cases, full 14×14 matrix)
2. [x] ~~Add shape_stride_cases for 0-D, empty, negative-stride edge cases~~ (DONE: 47 cases)
3. [x] ~~Add runtime_policy_cases for all CompatibilityClass variants~~ (DONE: 31 cases)
4. [x] ~~Add iterator differential cases for seek/reset/external-loop~~ (DONE: covered in smoke.rs oracle tests)
5. [ ] Wire up automated fixture regeneration from oracle
6. [ ] Add CI coverage tracking (case count trends)

## References

- [FEATURE_PARITY.md](../../FEATURE_PARITY.md) - Overall parity status
- [fixtures/README.md](fixtures/README.md) - Fixture file descriptions
- [DISCREPANCIES.md](DISCREPANCIES.md) - Intentional divergences from NumPy
