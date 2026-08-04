#!/usr/bin/env python3
"""correctness_sweep_vs_numpy.py — reusable correctness regression guard for fnp-python.

Encodes the SUBTLE comparators that the 2026-06 BOLD-VERIFY sessions needed to
catch bugs the Rust conformance suite missed. Run against a built fnp_python.so
(no cargo needed) for a fast post-build correctness check that complements
`cargo test conformance_*`.

Usage:
    PYTHONPATH=.probe python3 scripts/correctness_sweep_vs_numpy.py

Exit code = number of FAILs (0 = clean).

Why this exists (hard-won lessons, do not weaken the comparators):
1. eig/eigvals SILENT-WRONG bug: a native iterative QR returned the unconverged
   diagonal on timeout -> 11/120 random real matrices had WRONG eigenvalues. The
   conformance suite MISSED it because it only used symmetric/diagonal inputs, and
   naive comparators give false results: sort_complex diff and greedy nearest-match
   both mis-pair eigenvalues. The CORRECT, order-independent comparator is
   POWER-SUM INVARIANTS: sum(lambda^k) == trace(A^k) for k=1,2,3 (Newton's
   identities). Use RANDOM (non-symmetric) + adversarial corpora.
2. View-returning ops (transpose/rollaxis/ravel/diagonal) must return a VIEW:
   np.shares_memory(result, input) must be True (materializing a copy is both
   ~1e4x slower AND a semantics bug).
3. Selection/rearrangement ops (choice/take/permutation) must preserve INPUT DTYPE
   (compute indices then gather) — a common bug is extracting to f64 and returning
   float64 for int/complex inputs.
4. Special-value parity: singular factorization -> LinAlgError; cond(singular) ->
   +inf (numpy converts the 0/0 NaN to inf when input is finite).
"""
import sys
import numpy as np

try:
    import fnp_python as f
except Exception as e:  # pragma: no cover
    print(f"cannot import fnp_python (set PYTHONPATH to the built .so dir): {e}")
    sys.exit(2)

# fnp exposes linalg fns both top-level and under f.linalg; eig is linalg-only.
_eig = getattr(f, "eig", None) or f.linalg.eig
_eigvals = getattr(f, "eigvals", None) or f.linalg.eigvals

fails = []


def ok(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  [{detail}]"))
    if not cond:
        fails.append(name)


def power_sums(vals, k):
    return complex(np.sum(np.asarray(vals, dtype=complex) ** k))


def main():
    rng = np.random.default_rng(12345)

    # 1. eig / eigvals power-sum invariants on RANDOM real matrices (the missed bug)
    bad_eigvals = 0
    bad_eig = 0
    for _ in range(60):
        n = int(rng.integers(3, 40))
        A = rng.standard_normal((n, n))
        traces = [np.trace(np.linalg.matrix_power(A, k)) for k in (1, 2, 3)]
        try:
            w = _eigvals(A)
            if not all(abs(power_sums(w, k) - traces[k - 1]) <= 1e-6 * (1 + abs(traces[k - 1]))
                       for k in (1, 2, 3)):
                bad_eigvals += 1
        except Exception:
            bad_eigvals += 1
        try:
            w2 = _eig(A)[0]
            if not all(abs(power_sums(w2, k) - traces[k - 1]) <= 1e-6 * (1 + abs(traces[k - 1]))
                       for k in (1, 2, 3)):
                bad_eig += 1
        except Exception:
            bad_eig += 1
    ok("eigvals power-sum invariants (60 random real)", bad_eigvals == 0, f"{bad_eigvals} wrong")
    ok("eig power-sum invariants (60 random real)", bad_eig == 0, f"{bad_eig} wrong")

    # symmetric eigvalsh: values match numpy (sorted ascending)
    S = rng.standard_normal((50, 50)); S = S + S.T
    ok("eigvalsh values == numpy", np.allclose(np.sort(np.asarray(f.eigvalsh(S))),
                                               np.sort(np.linalg.eigvalsh(S)), atol=1e-8))

    # 2. linalg special-value / error parity
    def raises_linalgerror(fn):
        try:
            fn(); return False
        except np.linalg.LinAlgError:
            return True
        except Exception:
            return False
    ok("inv(singular) -> LinAlgError", raises_linalgerror(lambda: f.inv(np.ones((4, 4)))))
    ok("solve(singular) -> LinAlgError", raises_linalgerror(lambda: f.solve(np.ones((3, 3)), np.ones(3))))
    ok("cholesky(non-PD) -> LinAlgError", raises_linalgerror(lambda: f.cholesky(-np.eye(3))))
    dn = f.det(np.array([[np.nan, 1.0], [2.0, 3.0]]))
    ok("det(nan-entry) is nan (numpy parity)", np.isnan(np.asarray(dn, dtype=float)))
    # cond(singular finite) -> +inf
    try:
        c = float(np.asarray(f.cond(np.ones((4, 4)))))
        ok("cond(singular finite) -> +inf", np.isinf(c) and c > 0, f"got {c}")
    except Exception as e:
        ok("cond(singular finite) -> +inf", False, str(e)[:30])

    # 3. view-op aliasing contract
    Mv = rng.standard_normal((40, 60))
    for name, vf in [
        ("matrix_transpose shares_memory", lambda: f.matrix_transpose(Mv)),
        ("rollaxis shares_memory", lambda: f.rollaxis(Mv.reshape(4, 10, 60), 2, 0)),
        ("ravel shares_memory", lambda: f.ravel(Mv)),
        ("diagonal shares_memory", lambda: f.diagonal(Mv)),
    ]:
        try:
            ok(name, np.shares_memory(np.asarray(vf()), Mv))
        except Exception as e:
            ok(name, False, str(e)[:30])

    # 4. selection/rearrangement ops preserve INPUT dtype (not coerced to f64)
    gen = np.random.default_rng(7)
    ints = np.arange(1000, dtype=np.int64)
    try:
        r = f.random.Generator(f.random.PCG64(0)).choice(ints, size=10) if hasattr(f, "random") else None
    except Exception:
        r = None
    if r is not None:
        ok("Generator.choice preserves int64 dtype", np.asarray(r).dtype == np.int64,
           str(np.asarray(r).dtype))
    take_res = np.asarray(f.take(ints, np.array([1, 5, 9])))
    ok("take preserves int64 dtype", take_res.dtype == np.int64, str(take_res.dtype))

    # 5. reduction-over-comparison early-exit ops match numpy (array_equal/allclose)
    a = rng.standard_normal(100000); b = a.copy(); b[50000] += 1e-3
    ok("array_equal(differ) == False", f.array_equal(a, b) == False)
    ok("array_equal(same) == True", f.array_equal(a, a.copy()) == True)
    ok("allclose nan equal_nan", f.allclose(np.array([np.nan, 1.0]), np.array([np.nan, 1.0]), equal_nan=True) == True)

    # 6. value-parity for THIS SESSION'S shipped perf wins (guard against regressing them)
    sig = rng.standard_normal(50000); ker = rng.standard_normal(16)
    for mode in ("full", "same", "valid"):
        ok(f"convolve({mode}) == numpy",
           np.allclose(np.asarray(f.convolve(sig, ker, mode)), np.convolve(sig, ker, mode), atol=1e-9))
        ok(f"correlate({mode}) == numpy",
           np.allclose(np.asarray(f.correlate(sig, ker, mode)), np.correlate(sig, ker, mode), atol=1e-9))
    p, q = rng.standard_normal(5000), rng.standard_normal(5000)
    ok("cov(a,b) == numpy", np.allclose(np.asarray(f.cov(p, q)), np.cov(p, q), atol=1e-10))
    ok("corrcoef(a,b) == numpy", np.allclose(np.asarray(f.corrcoef(p, q)), np.corrcoef(p, q), atol=1e-10))
    ok("concat(alias) == numpy", np.array_equal(np.asarray(f.concat([p, q])), np.concat([p, q])))
    ok("atan2(alias) == arctan2", np.allclose(np.asarray(f.atan2(p, q)), np.arctan2(p, q), atol=1e-12))

    print(f"\nFAILs: {len(fails)}" + ("" if not fails else "  -> " + ", ".join(fails)))
    sys.exit(min(len(fails), 125))


if __name__ == "__main__":
    main()
