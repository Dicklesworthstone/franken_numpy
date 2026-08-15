//! Wide-key companion shard for `conformance_setops`.
//!
//! Holds the large-n packed-Latin-1 string probes. Their sizes are LOAD-BEARING:
//! the packed-u64 `(key, index)` sort path and the two-word wide-key branch only
//! engage above the native dispatch gates (n >= 1<<17), so a smaller input would
//! silently exercise a different route. They are separated rather than shrunk or
//! ignored — coverage is unchanged and every test still runs by default. Split
//! under bead `deadlock-audit-syi8e`.
//!
//! THIS SEPARATION IS CATEGORICAL, NOT A PERFORMANCE FIX — do not cite it as one.
//! Bead `deadlock-audit-syi8e` was filed claiming these three each ran over a
//! minute (2026-08-09). Re-measured 2026-08-15 on rch worker hz2 with
//! `cargo test --report-time`, they are the three CHEAPEST tests in the setops
//! family — 2.140s, 1.731s and 1.039s — and the parent shard completes in 23.12s,
//! nowhere near a 120s budget. The likely reconciliation is that the wide-key
//! string set-op work landed in between and closed them. What survives is the
//! grouping: these are the packed/wide-key string routes, kept together so a
//! future regression on that path is obvious in one binary.
//!
//! RUNTIME IS HOST-SCOPED — never quote a runtime without naming the host that
//! observed it; a sibling shard measured 9.03s, 183.46s and 617.62s for identical
//! tests on three hosts. When running under a time cap CHECK THAT THE BINARY
//! REPORTED — a cap kills a shard mid-execution and prints no `test result:` line
//! for it, which reads exactly like a pass if you only grep for failures.

mod common;

use common::with_fnp_and_numpy;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn unique_and_sort_string_packed_latin1_large_matches_numpy() {
    // Large-n fixed-width Latin-1 U8/S6 arrays (n >= 1<<17) take the packed-u64 (key, index)
    // sort path in both unique and sort. Packed key order == codepoint order; unique first-of-run
    // and sorted record sequence must be byte-exact vs numpy.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(19)\n\
                 n = 300_000\n\
                 u8 = rng.integers(97, 123, (n, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
                 s6 = np.array([bytes(r) for r in rng.integers(97, 105, (60_000, 6), dtype=np.uint8)], dtype='S6')\n\
                 s6 = np.tile(s6, 5)[:n]\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        let array_equal = numpy.getattr("array_equal")?;
        for name in ["u8", "s6"] {
            let arr = ns
                .get_item(name)?
                .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing arr"))?;
            for op in ["unique", "sort"] {
                let ours = module.getattr(op)?.call1((&arr,))?;
                let theirs = numpy.getattr(op)?.call1((&arr,))?;
                let equal: bool = array_equal.call1((&ours, &theirs))?.extract()?;
                assert!(equal, "packed Latin-1 {name} {op} diverged from numpy");
            }
        }
        Ok(())
    });
}

#[test]
fn unique_packed_wide_latin1_u9_u16_matches_numpy() {
    // U16 records no longer fit the packed-u64 path. The two-word key captures all 16
    // Latin-1 codepoints in NumPy lexicographic order while retaining the original index
    // as the deterministic tie-break used by the unchanged gather/dedup pipeline.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(281)\n\
                 n = 300_000\n\
                 u9 = rng.integers(97, 123, (n, 9), dtype=np.uint32).reshape(-1).view('U9')\n\
                 u16 = rng.integers(97, 123, (n, 16), dtype=np.uint32).reshape(-1).view('U16')\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        for name in ["u9", "u16"] {
            let arr = ns.get_item(name)?.ok_or_else(|| {
                pyo3::exceptions::PyAssertionError::new_err(format!("missing {name}"))
            })?;
            let ours = module.getattr("unique")?.call1((&arr,))?;
            let theirs = numpy.getattr("unique")?.call1((&arr,))?;
            let equal: bool = numpy
                .getattr("array_equal")?
                .call1((&ours, &theirs))?
                .extract()?;
            assert!(
                equal,
                "packed-wide Latin-1 {name} unique diverged from numpy"
            );
            assert_eq!(
                ours.getattr("dtype")?.str()?.to_string(),
                theirs.getattr("dtype")?.str()?.to_string()
            );
            let ours_bytes: Vec<u8> = ours.call_method0("tobytes")?.extract()?;
            let theirs_bytes: Vec<u8> = theirs.call_method0("tobytes")?.extract()?;
            assert_eq!(
                ours_bytes, theirs_bytes,
                "{name} unique output bytes diverged"
            );
        }
        Ok(())
    });
}

#[test]
fn unique_full_string_packed_latin1_large_matches_numpy() {
    // unique(..., return_index/inverse/counts) on large Latin-1 U8/S6 takes the packed-u64
    // (key, index) path; first-occurrence index, inverse map, and counts must all be byte-exact.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(23)\n\
                 n = 300_000\n\
                 u8 = rng.integers(97, 101, (n, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
                 s6 = np.array([bytes(r) for r in rng.integers(97, 101, (40_000, 6), dtype=np.uint8)], dtype='S6')\n\
                 s6 = np.tile(s6, 8)[:n]\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        let array_equal = numpy.getattr("array_equal")?;
        let kw = PyDict::new(py);
        kw.set_item("return_index", true)?;
        kw.set_item("return_inverse", true)?;
        kw.set_item("return_counts", true)?;
        for name in ["u8", "s6"] {
            let arr = ns
                .get_item(name)?
                .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing arr"))?;
            let ours = module.getattr("unique")?.call((&arr,), Some(&kw))?;
            let theirs = numpy.getattr("unique")?.call((&arr,), Some(&kw))?;
            for k in 0..4 {
                let o = ours.get_item(k)?;
                let t = theirs.get_item(k)?;
                let equal: bool = array_equal.call1((&o, &t))?.extract()?;
                assert!(
                    equal,
                    "packed Latin-1 {name} unique_full field {k} diverged from numpy"
                );
            }
        }
        Ok(())
    });
}
