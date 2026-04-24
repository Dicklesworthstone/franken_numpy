//! Shared infrastructure for the fnp-python conformance matrix.
//!
//! Each family-level conformance test (conformance_array_creation.rs,
//! conformance_arithmetic_ops.rs, …) drives the same differential loop:
//! call `fnp_python.<fn>(*args, **kwargs)`, call the matching
//! `numpy.<fn>(*args, **kwargs)`, compare outputs, emit a JSON-line
//! verdict, and aggregate results per RequirementLevel into a
//! markdown-ready compliance summary.
//!
//! The harness is opinionated: every case declares its RequirementLevel
//! (Must / Should / May) and its intended comparison mode (Strict for
//! integer / boolean / dtype / shape exactness, Close for float arrays
//! where 1 ULP divergence is tolerated, Surface for tuple / scalar /
//! structured dtype parity, Error for raise-alignment tests). Unknown
//! divergences become FAIL; documented divergences flip to XFAIL by
//! returning CaseOutcome::ExpectedFail from the builder closure.
//!
//! MUST failures abort the run; SHOULD / MAY failures print but let the
//! suite continue.

#![allow(dead_code)]

use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use fnp_python::fnp_python;

/// Serialize Python test runs across the pyo3 auto-initialize
/// interpreter. Cargo runs integration tests on multiple threads; the
/// embedded CPython VM is single-threaded, and numpy's own module state
/// is not thread-safe in our re-entrant helpers.
pub static PY_MUTEX: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequirementLevel {
    /// Fundamental contract of the function — any divergence is a bug.
    Must,
    /// Widely-relied-upon behavior (e.g. dtype preservation, layout).
    Should,
    /// Niche or legacy surface; divergence only matters for specific
    /// callers.
    May,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareMode {
    /// Exact equality of shape, dtype, and value string representation.
    /// Use for integer / boolean arrays and dtype-class checks.
    Strict,
    /// Float-array match under numpy.allclose semantics; shapes and
    /// dtype names must still be identical.
    Close,
    /// Tuple / structured / scalar surface — each component strict.
    Surface,
    /// Both implementations must raise the same exception type.
    Error,
}

#[derive(Debug)]
pub enum CaseOutcome {
    Pass,
    Fail(String),
    /// Documented intentional divergence (e.g. DISCREPANCIES entry).
    ExpectedFail(&'static str),
}

pub struct Totals {
    pub must_total: AtomicUsize,
    pub must_pass: AtomicUsize,
    pub must_xfail: AtomicUsize,
    pub should_total: AtomicUsize,
    pub should_pass: AtomicUsize,
    pub should_xfail: AtomicUsize,
    pub may_total: AtomicUsize,
    pub may_pass: AtomicUsize,
    pub may_xfail: AtomicUsize,
    pub fail_count: AtomicUsize,
}

impl Totals {
    pub const fn new() -> Self {
        Self {
            must_total: AtomicUsize::new(0),
            must_pass: AtomicUsize::new(0),
            must_xfail: AtomicUsize::new(0),
            should_total: AtomicUsize::new(0),
            should_pass: AtomicUsize::new(0),
            should_xfail: AtomicUsize::new(0),
            may_total: AtomicUsize::new(0),
            may_pass: AtomicUsize::new(0),
            may_xfail: AtomicUsize::new(0),
            fail_count: AtomicUsize::new(0),
        }
    }

    fn record(&self, level: RequirementLevel, outcome: &CaseOutcome) {
        let (total, pass, xfail) = match level {
            RequirementLevel::Must => (&self.must_total, &self.must_pass, &self.must_xfail),
            RequirementLevel::Should => (&self.should_total, &self.should_pass, &self.should_xfail),
            RequirementLevel::May => (&self.may_total, &self.may_pass, &self.may_xfail),
        };
        total.fetch_add(1, Ordering::Relaxed);
        match outcome {
            CaseOutcome::Pass => {
                pass.fetch_add(1, Ordering::Relaxed);
            }
            CaseOutcome::ExpectedFail(_) => {
                xfail.fetch_add(1, Ordering::Relaxed);
            }
            CaseOutcome::Fail(_) => {
                self.fail_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn summarize(&self, family: &str) -> String {
        let pct = |pass: usize, total: usize| -> String {
            if total == 0 {
                "n/a".to_string()
            } else {
                format!("{:.1}%", (pass as f64 / total as f64) * 100.0)
            }
        };
        let must_total = self.must_total.load(Ordering::Relaxed);
        let must_pass = self.must_pass.load(Ordering::Relaxed);
        let must_xfail = self.must_xfail.load(Ordering::Relaxed);
        let should_total = self.should_total.load(Ordering::Relaxed);
        let should_pass = self.should_pass.load(Ordering::Relaxed);
        let should_xfail = self.should_xfail.load(Ordering::Relaxed);
        let may_total = self.may_total.load(Ordering::Relaxed);
        let may_pass = self.may_pass.load(Ordering::Relaxed);
        let may_xfail = self.may_xfail.load(Ordering::Relaxed);
        format!(
            "| {family} | MUST {must_pass}/{must_total} ({} pass, {must_xfail} xfail) | SHOULD {should_pass}/{should_total} ({} pass, {should_xfail} xfail) | MAY {may_pass}/{may_total} ({} pass, {may_xfail} xfail) |",
            pct(must_pass, must_total),
            pct(should_pass, should_total),
            pct(may_pass, may_total),
        )
    }
}

/// Initialize pyo3, import numpy, and build the fnp_python module — all
/// operations a conformance test needs to compare outputs.
pub fn with_fnp_and_numpy<F>(f: F)
where
    F: for<'py> FnOnce(Python<'py>, Bound<'py, PyModule>, Bound<'py, PyModule>) -> PyResult<()>,
{
    let _guard = PY_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    Python::initialize();
    Python::attach(|py| {
        if py.import("numpy").is_err() {
            // numpy not present in the embedded interpreter — treat as
            // pass. Local / rch flows install numpy before running.
            return;
        }
        let module = PyModule::new(py, "fnp_python_conformance").unwrap();
        fnp_python(&module).unwrap();
        let numpy = py.import("numpy").unwrap();
        f(py, module, numpy).unwrap();
    });
}

/// Run a single parity case: call both implementations with the same
/// args, compare per the requested mode, record the outcome.
#[allow(clippy::too_many_arguments)]
pub fn run_case<F, G>(
    py: Python<'_>,
    module: &Bound<'_, PyModule>,
    numpy: &Bound<'_, PyModule>,
    id: &str,
    function: &str,
    level: RequirementLevel,
    mode: CompareMode,
    totals: &Totals,
    build_args: F,
    build_kwargs: G,
) where
    F: for<'py> Fn(Python<'py>) -> PyResult<Bound<'py, PyTuple>>,
    G: for<'py> Fn(Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>>,
{
    let args_ours = build_args(py).expect("args builder failed (ours)");
    let kwargs_ours = build_kwargs(py).expect("kwargs builder failed (ours)");
    // Rebuild args/kwargs for the numpy call (some builders materialize
    // StringIO or iterators that are consumed on first use).
    let args_theirs = build_args(py).expect("args builder failed (theirs)");
    let kwargs_theirs = build_kwargs(py).expect("kwargs builder failed (theirs)");

    let our_fn = module.getattr(function).expect("fnp_python missing fn");
    let numpy_fn = numpy.getattr(function).expect("numpy missing fn");

    let ours = our_fn.call(&args_ours, kwargs_ours.as_ref());
    let theirs = numpy_fn.call(&args_theirs, kwargs_theirs.as_ref());

    let outcome = compare(py, mode, ours, theirs);
    emit_verdict(id, function, level, &outcome);
    totals.record(level, &outcome);
    if let CaseOutcome::Fail(reason) = &outcome
        && level == RequirementLevel::Must
    {
        panic!("MUST clause {id} ({function}) failed: {reason}");
    }
}

fn compare(
    py: Python<'_>,
    mode: CompareMode,
    ours: PyResult<pyo3::Bound<'_, pyo3::types::PyAny>>,
    theirs: PyResult<pyo3::Bound<'_, pyo3::types::PyAny>>,
) -> CaseOutcome {
    match (ours, theirs) {
        (Ok(a), Ok(b)) => match mode {
            CompareMode::Strict => compare_strict(py, &a, &b),
            CompareMode::Close => compare_close(py, &a, &b),
            CompareMode::Surface => compare_surface(py, &a, &b),
            CompareMode::Error => {
                CaseOutcome::Fail("expected both to raise but both succeeded".to_string())
            }
        },
        (Err(a), Err(b)) => {
            if matches!(mode, CompareMode::Error) {
                compare_error_types(py, &a, &b)
            } else {
                // Both raised under a non-error mode → surface the pair so
                // we can adapt the case to CompareMode::Error if this was
                // expected.
                let a_str = format!(
                    "{}: {}",
                    a.get_type(py)
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_default(),
                    a
                );
                let b_str = format!(
                    "{}: {}",
                    b.get_type(py)
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_default(),
                    b
                );
                if a_str == b_str {
                    CaseOutcome::Pass
                } else {
                    CaseOutcome::Fail(format!(
                        "both raised but errors differ: ours={a_str} theirs={b_str}"
                    ))
                }
            }
        }
        (Ok(_), Err(e)) => CaseOutcome::Fail(format!("ours succeeded, theirs raised: {e}")),
        (Err(e), Ok(_)) => CaseOutcome::Fail(format!("ours raised, theirs succeeded: {e}")),
    }
}

fn compare_strict(
    py: Python<'_>,
    ours: &Bound<'_, pyo3::types::PyAny>,
    theirs: &Bound<'_, pyo3::types::PyAny>,
) -> CaseOutcome {
    // Compare dtype + shape + repr. `repr` on an ndarray is stable
    // enough to catch value mismatches without float tolerance.
    let ours_dtype = fetch_dtype_name(py, ours);
    let theirs_dtype = fetch_dtype_name(py, theirs);
    if ours_dtype != theirs_dtype {
        return CaseOutcome::Fail(format!(
            "dtype mismatch: ours={ours_dtype:?} theirs={theirs_dtype:?}"
        ));
    }
    let ours_shape = fetch_shape(py, ours);
    let theirs_shape = fetch_shape(py, theirs);
    if ours_shape != theirs_shape {
        return CaseOutcome::Fail(format!(
            "shape mismatch: ours={ours_shape:?} theirs={theirs_shape:?}"
        ));
    }
    let ours_repr = pyobject_repr(py, ours);
    let theirs_repr = pyobject_repr(py, theirs);
    if ours_repr != theirs_repr {
        return CaseOutcome::Fail(format!(
            "value repr mismatch: ours={ours_repr} theirs={theirs_repr}"
        ));
    }
    CaseOutcome::Pass
}

fn compare_close(
    py: Python<'_>,
    ours: &Bound<'_, pyo3::types::PyAny>,
    theirs: &Bound<'_, pyo3::types::PyAny>,
) -> CaseOutcome {
    let ours_dtype = fetch_dtype_name(py, ours);
    let theirs_dtype = fetch_dtype_name(py, theirs);
    if ours_dtype != theirs_dtype {
        return CaseOutcome::Fail(format!(
            "dtype mismatch: ours={ours_dtype:?} theirs={theirs_dtype:?}"
        ));
    }
    let ours_shape = fetch_shape(py, ours);
    let theirs_shape = fetch_shape(py, theirs);
    if ours_shape != theirs_shape {
        return CaseOutcome::Fail(format!(
            "shape mismatch: ours={ours_shape:?} theirs={theirs_shape:?}"
        ));
    }
    // numpy.allclose with equal_nan=True — close enough for the ULP-level
    // drift that integer-reducing / geometric-stepping native ports can
    // introduce.
    let numpy = match py.import("numpy") {
        Ok(np) => np,
        Err(_) => return CaseOutcome::Fail("numpy import failed during compare".into()),
    };
    let kwargs = PyDict::new(py);
    kwargs.set_item("equal_nan", true).unwrap();
    let allclose = numpy.getattr("allclose").unwrap();
    match allclose.call((ours, theirs), Some(&kwargs)) {
        Ok(verdict) => match verdict.extract::<bool>() {
            Ok(true) => CaseOutcome::Pass,
            Ok(false) => CaseOutcome::Fail(format!(
                "values drift beyond allclose default tol: ours={} theirs={}",
                pyobject_repr(py, ours),
                pyobject_repr(py, theirs)
            )),
            Err(e) => CaseOutcome::Fail(format!("allclose non-bool result: {e}")),
        },
        Err(e) => CaseOutcome::Fail(format!("allclose raised: {e}")),
    }
}

fn compare_surface(
    py: Python<'_>,
    ours: &Bound<'_, pyo3::types::PyAny>,
    theirs: &Bound<'_, pyo3::types::PyAny>,
) -> CaseOutcome {
    let ours_repr = pyobject_repr(py, ours);
    let theirs_repr = pyobject_repr(py, theirs);
    if ours_repr == theirs_repr {
        CaseOutcome::Pass
    } else {
        CaseOutcome::Fail(format!(
            "surface repr mismatch: ours={ours_repr} theirs={theirs_repr}"
        ))
    }
}

fn compare_error_types(py: Python<'_>, ours: &PyErr, theirs: &PyErr) -> CaseOutcome {
    let ours_type = ours
        .get_type(py)
        .name()
        .map(|n| n.to_string())
        .unwrap_or_default();
    let theirs_type = theirs
        .get_type(py)
        .name()
        .map(|n| n.to_string())
        .unwrap_or_default();
    if ours_type == theirs_type {
        CaseOutcome::Pass
    } else {
        CaseOutcome::Fail(format!(
            "error type mismatch: ours={ours_type} theirs={theirs_type}"
        ))
    }
}

fn fetch_dtype_name(_py: Python<'_>, obj: &Bound<'_, pyo3::types::PyAny>) -> Option<String> {
    obj.getattr("dtype")
        .ok()
        .and_then(|d| d.getattr("name").ok())
        .and_then(|n| n.extract::<String>().ok())
        .or_else(|| obj.get_type().name().ok()?.extract::<String>().ok())
}

fn fetch_shape(py: Python<'_>, obj: &Bound<'_, pyo3::types::PyAny>) -> Option<Vec<usize>> {
    let _ = py;
    obj.getattr("shape")
        .ok()
        .and_then(|s| s.extract::<Vec<usize>>().ok())
}

fn pyobject_repr(py: Python<'_>, obj: &Bound<'_, pyo3::types::PyAny>) -> String {
    let _ = py;
    obj.repr()
        .ok()
        .and_then(|r| r.extract::<String>().ok())
        .unwrap_or_else(|| "<repr failed>".to_string())
}

fn emit_verdict(id: &str, function: &str, level: RequirementLevel, outcome: &CaseOutcome) {
    let (verdict, detail) = match outcome {
        CaseOutcome::Pass => ("PASS", String::new()),
        CaseOutcome::ExpectedFail(reason) => ("XFAIL", (*reason).to_string()),
        CaseOutcome::Fail(reason) => ("FAIL", reason.clone()),
    };
    let level_str = match level {
        RequirementLevel::Must => "MUST",
        RequirementLevel::Should => "SHOULD",
        RequirementLevel::May => "MAY",
    };
    let detail_json = json_escape(&detail);
    eprintln!(
        "{{\"id\":\"{id}\",\"fn\":\"{function}\",\"level\":\"{level_str}\",\"verdict\":\"{verdict}\",\"detail\":\"{detail_json}\"}}"
    );
}

/// Minimal JSON string-escape. detail strings come from compare
/// helpers; they can carry quotes, backslashes, and newlines.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}
