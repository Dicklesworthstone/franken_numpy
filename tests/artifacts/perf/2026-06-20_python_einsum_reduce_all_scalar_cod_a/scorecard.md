# fnp-python einsum reduce-all scalar builder - 2026-06-20

Bead: `franken_numpy-ixs5y.276`

Target:
- `python_einsum_boundary/fnp_einsum_reduce_all_f64_1000`
- Exact contiguous `float64` ndarray, `einsum("ij->", a)`.

## Decision

SHIP.

The candidate bypasses the temporary 0-D `UFuncArray` and returns directly
through the cached `numpy.float64` scalar builder in the single-operand f64
reduce-all fast path.

## Same-worker Criterion evidence

Worker: `vmi1149989`

Command:
`rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_einsum_boundary --sample-size 10 --measurement-time 3 --warm-up-time 1 --output-format bencher`

| Row | Baseline FNP ns | Baseline NumPy ns | Baseline FNP/NumPy | Candidate FNP ns | Candidate NumPy ns | Candidate FNP/NumPy | Candidate/Old FNP | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| trace f64 4000 | 5,431 | 8,017 | 0.677x | 5,102 | 6,763 | 0.754x | 0.939x | guard win |
| diag f64 4000 | 1,045 | 1,158 | 0.902x | 833 | 1,075 | 0.775x | 0.797x | guard win |
| reduce-all f64 1000 | 119,524 | 115,252 | 1.037x | 100,778 | 104,427 | 0.965x | 0.843x | keep |
| reduce-rows f64 1000 | 105,463 | 165,079 | 0.639x | 103,688 | 100,144 | 1.035x | 0.983x | noisy guard loss |
| reduce-cols f64 1000 | 148,799 | 489,469 | 0.304x | 113,290 | 323,885 | 0.350x | 0.761x | guard win |

## Validation

Passed:
- `cargo test -p fnp-python --test conformance_einsum`
  - RCH had no admissible workers and failed open locally.
  - 28 passed, 0 failed.
- `rch exec -- cargo build -p fnp-python --release`
  - Passed on `hz1`.

Known pre-existing blockers:
- `cargo fmt --package fnp-python --check`
  - Failed on broad existing formatting drift; formatter was not applied.
- `rch exec -- cargo check -p fnp-python --all-targets`
  - Failed on existing lib-test call sites for wrapper functions whose
    signatures now take `(py, args, kwargs)`.
- `rch exec -- cargo clippy -p fnp-python --all-targets -- -D warnings`
  - Failed on the same all-targets errors plus existing lint inventory.
- `ubs crates/fnp-python/src/lib.rs ...`
  - Exited nonzero after 202s on broad existing `fnp-python` inventory.
  - No finding was specific to the edited scalar-return line.
- `git diff --check`
  - Passed.

## Retry predicate

Do not retry wrapper-only work for `einsum("ij->")` unless a fresh same-worker
bench shows the row regressed again. If `reduce_rows_f64_1000` remains a loss in
a focused row-only A/B, claim a separate row-reduction bead.
