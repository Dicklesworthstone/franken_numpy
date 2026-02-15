# FNP-P2C-008 Legacy Anchor Map

Packet: `FNP-P2C-008`  
Subsystem: `linalg bridge first wave`

## Validator Token Fields

packet_id: `FNP-P2C-008`

legacy_paths:
- `legacy_numpy_code/numpy/numpy/linalg/_linalg.py`
- `legacy_numpy_code/numpy/numpy/linalg/umath_linalg.cpp`
- `legacy_numpy_code/numpy/numpy/linalg/lapack_litemodule.c`
- `legacy_numpy_code/numpy/numpy/linalg/tests/test_linalg.py`
- `legacy_numpy_code/numpy/numpy/linalg/tests/test_regression.py`

legacy_symbols:
- `solve`
- `inv`
- `cholesky`
- `qr`
- `svd`
- `eig`
- `eigvals`
- `eigh`
- `lstsq`
- `matrix_rank`

## Scope

This map captures concrete legacy NumPy linear-algebra anchors (Python API wrappers, gufunc kernels, and lapack-lite bridge points) and binds them to planned Rust boundaries for clean-room `fnp-linalg` implementation.

## Legacy-to-Rust Anchor Table

| Legacy path | Symbol/anchor | Role in observable behavior | Planned Rust boundary |
|---|---|---|---|
| `legacy_numpy_code/numpy/numpy/linalg/__init__.py:1` | `numpy.linalg` surface export | canonical linalg API surface and symbol availability contract | `crates/fnp-linalg` API facade boundary |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:115` | `LinAlgError` + `_raise_linalgerror_*` helpers | error taxonomy for singular/non-convergence/invalid-shape paths | `linalg_error_taxonomy` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:363` | `solve` | square-system solve contract and singular-matrix failure behavior | `solver_ops::solve` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:536` | `inv` | matrix inversion contract and singular/ill-conditioned failure family | `solver_ops::inv` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:2418` | `lstsq` | least-squares tuple contract (`x`, residuals, rank, singular values) | `least_squares_ops::lstsq` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:778` | `cholesky` | positive-definite factorization contract (upper/lower selection) | `factorization_ops::cholesky` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:965` | `qr` | QR mode contracts (`reduced`, `complete`, raw/r) and shape outputs | `factorization_ops::qr` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:1170` | `eigvals` | general eigenvalue contract and non-convergence behavior | `spectral_ops::eigvals` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:1362` | `eig` | eigenvalue/eigenvector contract and output-shape family | `spectral_ops::eig` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:1515` | `eigh` / `eigvalsh` paths | Hermitian eigen decomposition contract with `UPLO` branches | `spectral_ops::eigh` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:1668` | `svd` / `svdvals` | SVD mode contracts and non-convergence behavior | `factorization_ops::svd` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:2272` | `slogdet` | signed log-determinant contract | `norm_det_rank_ops::slogdet` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:2356` | `det` | determinant contract over stacked matrices | `norm_det_rank_ops::det` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:2599` | `norm` | vector/matrix norm contract and axis/order handling | `norm_det_rank_ops::norm` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:2035` | `matrix_rank` | rank/tolerance contract and singular-value threshold handling | `norm_det_rank_ops::matrix_rank` |
| `legacy_numpy_code/numpy/numpy/linalg/_linalg.py:2154` | `pinv` | pseudo-inverse tolerance and hermitian-flag contract | `solver_ops::pinv` |
| `legacy_numpy_code/numpy/numpy/linalg/umath_linalg.cpp:1765` | `solve` gufunc kernel | core batched solve kernel path and error propagation | `backend_bridge::solve_kernel` |
| `legacy_numpy_code/numpy/numpy/linalg/umath_linalg.cpp:1839` | `inv` gufunc kernel | core inversion kernel path | `backend_bridge::inv_kernel` |
| `legacy_numpy_code/numpy/numpy/linalg/umath_linalg.cpp:2010` | `cholesky` kernel | cholesky lower/upper kernel dispatch | `backend_bridge::cholesky_kernel` |
| `legacy_numpy_code/numpy/numpy/linalg/umath_linalg.cpp:2494` | `eig_wrapper` / `eig` / `eigvals` | eigen kernel family and convergence/error behavior | `backend_bridge::eig_kernels` |
| `legacy_numpy_code/numpy/numpy/linalg/umath_linalg.cpp:2999` | `svd_wrapper` / `svd_*` | SVD kernel family and mode-specific output behavior | `backend_bridge::svd_kernels` |
| `legacy_numpy_code/numpy/numpy/linalg/umath_linalg.cpp:3362` | `qr_r_raw` / `qr_reduced` / `qr_complete` | QR kernel family and mode dispatch | `backend_bridge::qr_kernels` |
| `legacy_numpy_code/numpy/numpy/linalg/umath_linalg.cpp:4125` | `lstsq` kernel | least-squares kernel path and residual/rank outputs | `backend_bridge::lstsq_kernel` |
| `legacy_numpy_code/numpy/numpy/linalg/umath_linalg.cpp:4284` | gufunc registration table | public gufunc name/signature registration contract | `fnp-linalg` kernel registry boundary |
| `legacy_numpy_code/numpy/numpy/linalg/lapack_litemodule.c:170` | `lapack_lite_dgeqrf` / related wrappers | lapack-lite adapter semantics and parameter validation behavior | `lapack_adapter` compatibility seam |
| `legacy_numpy_code/numpy/numpy/linalg/lapack_litemodule.c:351` | `lapack_lite_xerbla` | low-level LAPACK error hook pathway | `backend_error_hook` |

## Oracle Test Anchors

| Test path | Anchor | Behavior family |
|---|---|---|
| `legacy_numpy_code/numpy/numpy/linalg/tests/test_linalg.py:524` | solve invalid-shape/singular assertions | solve failure-class contract |
| `legacy_numpy_code/numpy/numpy/linalg/tests/test_linalg.py:818` | `test_singular` family | inversion/condition/singularity behavior |
| `legacy_numpy_code/numpy/numpy/linalg/tests/test_linalg.py:1753` | `test_qr_empty` | QR edge-shape behavior |
| `legacy_numpy_code/numpy/numpy/linalg/tests/test_linalg.py:1991` | `test_xerbla_override` | backend error hook consistency |
| `legacy_numpy_code/numpy/numpy/linalg/tests/test_linalg.py:2212` | eig/eigh failure assertions | spectral non-convergence/failure classes |
| `legacy_numpy_code/numpy/numpy/linalg/tests/test_regression.py:54` | `test_svd_build` | SVD stability/regression behavior |
| `legacy_numpy_code/numpy/numpy/linalg/tests/test_regression.py:143` | `test_lstsq_complex_larger_rhs` | least-squares complex/shape behavior |
| `legacy_numpy_code/numpy/numpy/linalg/tests/test_regression.py:156` | `test_cholesky_empty_array` | cholesky empty/edge behavior |

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Notes for Follow-on Packet Steps

- Strict/hardened split must remain explicit for linalg error-class and tolerance boundaries.
- Backend adapter choices must preserve deterministic observable behavior (shape/value/error classes).
- Differential corpus should prioritize singular, near-singular, non-convergence, and mixed-dtype tolerance-edge fixtures before optimization work.
