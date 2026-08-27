//! BROAD vs-LIVE-NUMPY LOSS SWEEP, and the contract that follows it.
//!
//! WHY THIS BINARY EXISTS. Every other bench in this crate measures a cell someone already
//! suspected. That is how the campaign found its wins, and it is also why the standing
//! worst-cell figure went eight days and four levers stale before anyone re-measured it: no
//! instrument here ANSWERS "which op is worst right now" without being told where to look.
//! This one does. It discovers the comparable surface at runtime, times both arms from Rust
//! in one invocation, and ranks every cell by `numpy_ns / fnp_ns`.
//!
//! TWO GROUPS, and the split is deliberate:
//!
//!   `bench_vs_numpy_loss_sweep`      TRIAGE. Interleaved min-of-K, no A/A null, no CI. It
//!                                    exists to RANK, and a rank is all it may be quoted
//!                                    for. Every line it prints says so.
//!   `bench_vs_numpy_worst_contract`  DECISION. The full dual-null median-CI contract on
//!                                    ONE case named by `FNP_SWEEP_CASE`, so the op the
//!                                    sweep nominates can be certified without a rebuild.
//!
//! WHAT THE SWEEP CANNOT DO, stated before its numbers are read: a min-of-K ratio taken
//! once is not a decidable effect. It has no null, so it cannot tell an op that is slow
//! from an op that was unlucky, and this host is shared. Nothing from the sweep goes in the
//! ledger as a vs-incumbent claim. It nominates; the contract decides.
//!
//! DISCOVERY RATHER THAN A HAND-WRITTEN LIST, because a hand-written list can only contain
//! ops someone already thought about. The setup walks `dir(fnp) & dir(numpy)`, tries each
//! name against a ladder of operand tuples, and keeps a case only when BOTH libraries accept
//! the call AND return equal results. An op that disagrees is a correctness question, not a
//! performance one, and is reported separately rather than timed.

use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyList, PyListMethods, PyModule, PyTuple, PyTupleMethods};
use std::hint::black_box;
use std::time::Instant;

#[path = "common/mod.rs"]
mod common;
use common::*;

/// Hostname, sanitised, so every printed row names the machine it was taken on. Duplicated
/// from `criterion_python_elementwise.rs` rather than shared: it lives in a bench BODY there,
/// not in `common`, and moving it would touch a file two other agents hold open tonight.
fn measurement_worker() -> String {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .chars()
                .map(|character| {
                    if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                        character
                    } else {
                        '_'
                    }
                })
                .collect()
        })
        .unwrap_or_else(|| "unavailable".to_owned())
}

/// Operand ladder plus the discovery walk. Shared verbatim by both groups so the contract
/// group measures exactly the case the sweep ranked - a second, subtly different corpus is
/// how a "confirmation" ends up confirming something else.
///
/// THE DENYLIST IS NOT TUNING. Each entry is excluded for a reason that would make its
/// timing meaningless or its call unsafe to repeat: RNG draws are not reproducible between
/// arms, IO touches the filesystem, `sort`/`argsort`/`searchsorted` are owned by other
/// agents' live beads right now, and the printing/allocation helpers have no incumbent to
/// compare against.
const SWEEP_SETUP: &str = r#"
import numpy as np

rng = np.random.default_rng(20260826)
N = 1 << 16

f64  = rng.standard_normal(N)
f64b = rng.standard_normal(N) + 3.0
pos  = np.abs(f64) + 0.5
posb = np.abs(f64b) + 0.5
f32  = f64.astype(np.float32)
f32b = f64b.astype(np.float32)
f16  = f64.astype(np.float16)
i64  = rng.integers(-1000, 1000, N, dtype=np.int64)
i64b = rng.integers(1, 1000, N, dtype=np.int64)
i32  = i64.astype(np.int32)
i32b = i64b.astype(np.int32)
u8   = (i64 % 251).astype(np.uint8)
boo  = i64 > 0
boob = i64b > 500
c128 = f64 + 1j * f64b
c64  = c128.astype(np.complex64)
m2   = rng.standard_normal((256, 256))
m2b  = rng.standard_normal((256, 256))
i2d  = rng.integers(-1000, 1000, (256, 256), dtype=np.int64)
small = f64[:1024]
smallb = f64b[:1024]

LADDER = [
    ("f64",        (f64,)),
    ("f64_f64",    (f64, f64b)),
    ("f64_scalar", (f64, 2.0)),
    ("pos",        (pos,)),
    ("pos_pos",    (pos, posb)),
    ("f32",        (f32,)),
    ("f32_f32",    (f32, f32b)),
    ("f16",        (f16,)),
    ("i64",        (i64,)),
    ("i64_i64",    (i64, i64b)),
    ("i32",        (i32,)),
    ("i32_i32",    (i32, i32b)),
    ("u8",         (u8,)),
    ("bool",       (boo,)),
    ("bool_bool",  (boo, boob)),
    ("c128",       (c128,)),
    ("c128_c128",  (c128, c128)),
    ("c64",        (c64,)),
    ("m2",         (m2,)),
    ("m2_m2",      (m2, m2b)),
    ("i2d",        (i2d,)),
    ("small",      (small,)),
    ("small_small",(small, smallb)),
    ("f64_axis0",  (m2, 0)),
]

# TINY MIRROR OF THE LADDER, AND IT IS A SAFETY DEVICE, NOT AN OPTIMISATION.
#
# The first version of this sweep discovered by calling each candidate at FULL ladder size
# and only then applying a cost cap. That order is unsound: `outer` and `kron` return an
# output QUADRATIC in their input, so `np.outer(f64, f64b)` at N = 65536 asks for 4.29e9
# elements - 34 GB - before any cap can look at it. The run reached 16 GB RSS on a shared
# host and had to be killed. Discovery now happens entirely down here, where the same call
# costs kilobytes, and an op is promoted to the full corpus only after its OUTPUT EXPANSION
# is known to be bounded.
T = 256
tf64  = f64[:T].copy()
tf64b = f64b[:T].copy()
tpos  = pos[:T].copy()
tposb = posb[:T].copy()
tf32  = f32[:T].copy()
tf32b = f32b[:T].copy()
tf16  = f16[:T].copy()
ti64  = i64[:T].copy()
ti64b = i64b[:T].copy()
ti32  = i32[:T].copy()
ti32b = i32b[:T].copy()
tu8   = u8[:T].copy()
tboo  = boo[:T].copy()
tboob = boob[:T].copy()
tc128 = c128[:T].copy()
tc64  = c64[:T].copy()
tm2   = m2[:16, :16].copy()
tm2b  = m2b[:16, :16].copy()
ti2d  = i2d[:16, :16].copy()

TINY = {
    "f64":         (tf64,),
    "f64_f64":     (tf64, tf64b),
    "f64_scalar":  (tf64, 2.0),
    "pos":         (tpos,),
    "pos_pos":     (tpos, tposb),
    "f32":         (tf32,),
    "f32_f32":     (tf32, tf32b),
    "f16":         (tf16,),
    "i64":         (ti64,),
    "i64_i64":     (ti64, ti64b),
    "i32":         (ti32,),
    "i32_i32":     (ti32, ti32b),
    "u8":          (tu8,),
    "bool":        (tboo,),
    "bool_bool":   (tboo, tboob),
    "c128":        (tc128,),
    "c128_c128":   (tc128, tc128),
    "c64":         (tc64,),
    "m2":          (tm2,),
    "m2_m2":       (tm2, tm2b),
    "i2d":         (ti2d,),
    "small":       (tf64,),
    "small_small": (tf64, tf64b),
    "f64_axis0":   (tm2, 0),
}

MAX_EXPANSION = 8.0
SWEEP_COST_CAP_NS = 3_000_000


def _elements(value):
    try:
        if isinstance(value, tuple):
            return sum(_elements(v) for v in value)
        return int(np.asarray(value).size)
    except Exception:
        return 0


DENY = {
    # owned by live beads held by other agents this session
    "sort", "argsort", "searchsorted", "msort",
    # not reproducible between arms
    "random", "seed", "default_rng", "shuffle", "permutation", "choice",
    # filesystem / process side effects
    "load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt",
    "genfromtxt", "fromfile", "tofile", "memmap", "DataSource", "get_include",
    "show_config", "info", "source", "lookfor", "seterr", "seterrcall",
    "set_printoptions", "get_printoptions", "printoptions", "errstate",
    # no incumbent semantics worth timing / mutate their operand in place
    "put", "putmask", "place", "copyto", "fill_diagonal", "resize", "shares_memory",
    "may_share_memory", "set_string_function", "deprecate", "disp",
}


def _agree(x, y):
    """Equal shape, equal dtype, equal bytes - with NaN counted equal to NaN.

    Deliberately STRICTER than `allclose`: this sweep ranks performance, and two arms that
    differ in the last ulp are not the same computation. A near-miss should surface as a
    disagreement to be looked at, not be averaged into a ratio.
    """
    try:
        if isinstance(x, tuple) or isinstance(y, tuple):
            if not (isinstance(x, tuple) and isinstance(y, tuple)):
                return False
            return len(x) == len(y) and all(_agree(a, b) for a, b in zip(x, y))
        xa = np.asarray(x)
        ya = np.asarray(y)
        if xa.shape != ya.shape or xa.dtype != ya.dtype:
            return False
        if xa.dtype.kind in "fc":
            return bool(np.array_equal(xa, ya, equal_nan=True))
        return bool(np.array_equal(xa, ya))
    except Exception:
        return False


def discover(fnp_module):
    """Return (cases, disagreements, skipped).

    Three gates, in this order, and the order is the point:

      1. TINY ACCEPT  - both libraries take the same call on a 256-element corpus.
      2. TINY AGREE   - and return equal results there. A disagreement is a correctness
                        question, recorded and never timed.
      3. EXPANSION    - the tiny output is at most MAX_EXPANSION times the tiny input. An
                        op that expands more than that is quadratic-or-worse in its
                        operand and is measured on the SMALL corpus instead of the full
                        one, because promoting it is what tried to allocate 34 GB.

    Only then is the case re-verified on its real operands and cost-probed, so no
    unbounded call is ever made.
    """
    import time

    cases = []
    disagree = []
    skipped = []
    shared = sorted(set(dir(fnp_module)) & set(dir(np)))
    for name in shared:
        if name.startswith("_") or name in DENY:
            continue
        ours = getattr(fnp_module, name, None)
        theirs = getattr(np, name, None)
        if not callable(ours) or not callable(theirs) or isinstance(theirs, type):
            continue
        for label, args in LADDER:
            tiny_args = TINY.get(label)
            if tiny_args is None:
                continue
            try:
                tiny_expected = theirs(*tiny_args)
                tiny_actual = ours(*tiny_args)
            except Exception:
                continue
            if not _agree(tiny_expected, tiny_actual):
                disagree.append((name, label))
                break

            expansion = _elements(tiny_expected) / max(1.0, float(_elements(tiny_args)))
            if expansion > MAX_EXPANSION:
                # Quadratic-or-worse: keep it, but on the tiny operands only.
                chosen_label = "%s[%s@tiny]" % (name, label)
                chosen_args = tiny_args
            else:
                chosen_label = "%s[%s]" % (name, label)
                chosen_args = args

            try:
                expected = theirs(*chosen_args)
                actual = ours(*chosen_args)
            except Exception:
                skipped.append((name, label, "raised_at_full_size"))
                break
            if not _agree(expected, actual):
                disagree.append((name, label + "@full_only"))
                break

            best = None
            for _ in range(3):
                t0 = time.perf_counter_ns()
                theirs(*chosen_args)
                dt = time.perf_counter_ns() - t0
                best = dt if best is None else min(best, dt)
            if best > SWEEP_COST_CAP_NS:
                skipped.append((name, label, "numpy_arm_%dns_over_budget" % best))
                break

            cases.append((chosen_label, name, chosen_args))
            break

    # `take` needs an index operand, so it cannot enter the one-operand generic
    # ladder above.  These are runtime-built public-call cases, not a special
    # value table: container form (list and tuple) and length both vary while
    # the source is the same ordinary f64 array.  The 1024-index row prevents a
    # list10-only implementation from qualifying as a general container route.
    take_source = np.linspace(-7.5, 7.5, N, dtype=np.float64)
    take_cases = [
        ("take[list10]", [0, 17, 1024, N // 2, -1, 7, N - 2, 91, 4096, 33]),
        ("take[tuple10]", (0, 17, 1024, N // 2, -1, 7, N - 2, 91, 4096, 33)),
        ("take[list1024]", list(range(0, N, N // 1024))),
    ]
    for label, indices in take_cases:
        try:
            expected = np.take(take_source, indices)
            actual = fnp_module.take(take_source, indices)
        except Exception:
            skipped.append(("take", label, "raised"))
            continue
        if not _agree(expected, actual):
            disagree.append(("take", label))
            continue
        cases.append((label, "take", (take_source, indices)))
    return cases, disagree, skipped
"#;

/// Build the case list on the interpreter and hand back `(cases, disagreements, slow)`.
fn discover_cases<'py>(
    py: Python<'py>,
    module: &pyo3::Bound<'py, PyModule>,
) -> (
    pyo3::Bound<'py, PyList>,
    pyo3::Bound<'py, PyList>,
    pyo3::Bound<'py, PyList>,
) {
    let ns = PyDict::new(py);
    ns.set_item("fnp", module).expect("bind fnp module");
    py.run(
        std::ffi::CString::new(SWEEP_SETUP).unwrap().as_c_str(),
        Some(&ns),
        Some(&ns),
    )
    .expect("sweep setup");
    py.run(
        std::ffi::CString::new("CASES, DISAGREE, SLOW = discover(fnp)\n")
            .unwrap()
            .as_c_str(),
        Some(&ns),
        Some(&ns),
    )
    .expect("case discovery");
    (
        ns.get_item("CASES")
            .expect("CASES")
            .cast_into::<PyList>()
            .expect("CASES is a list"),
        ns.get_item("DISAGREE")
            .expect("DISAGREE")
            .cast_into::<PyList>()
            .expect("DISAGREE is a list"),
        ns.get_item("SLOW")
            .expect("SLOW")
            .cast_into::<PyList>()
            .expect("SLOW is a list"),
    )
}

/// Rebuild one of the public `take` container cases without first discovering
/// every unrelated NumPy operation.  The generic discovery corpus deliberately
/// contains expensive numerical operators; using it merely to retrieve a
/// runtime-selected `take[...]` row can evaluate an unrelated quadratic call
/// before the contract even starts.  These labels are also emitted by
/// `discover`, and the inputs retain the list/tuple and length variation that
/// prevents a list10-only route from qualifying.
fn take_contract_case<'py>(
    py: Python<'py>,
    wanted: &str,
) -> Option<(String, String, pyo3::Bound<'py, PyTuple>)> {
    let indices = match wanted {
        "take[list10]" => "[0, 17, 1024, N // 2, -1, 7, N - 2, 91, 4096, 33]",
        "take[tuple10]" => "(0, 17, 1024, N // 2, -1, 7, N - 2, 91, 4096, 33)",
        "take[list1024]" => "list(range(0, N, N // 1024))",
        _ => return None,
    };
    let ns = PyDict::new(py);
    py.run(
        std::ffi::CString::new(format!(
            "import numpy as np\nN = 1 << 16\ntake_source = np.linspace(-7.5, 7.5, N, dtype=np.float64)\ntake_indices = {indices}\n"
        ))
        .expect("take contract setup CString")
        .as_c_str(),
        Some(&ns),
        Some(&ns),
    )
    .expect("take contract setup");
    let source = ns.get_item("take_source").expect("take source");
    let indices = ns.get_item("take_indices").expect("take indices");
    let args = PyTuple::new(py, [&source, &indices]).expect("take contract arguments");
    Some((wanted.to_owned(), "take".to_owned(), args))
}

/// Build finite same-shape f64 `fmod` calls directly for the dual-null contract.
///
/// `fmod` needs two operands and therefore cannot enter the generic one-operand
/// sweep ladder.  These two sizes use different dividend/divisor distributions;
/// they exercise the public ufunc rather than a private kernel and keep the
/// divisor non-zero so the finite hot path, not NumPy's warning fallback, is timed.
fn fmod_contract_case<'py>(
    py: Python<'py>,
    wanted: &str,
) -> Option<(String, String, pyo3::Bound<'py, PyTuple>)> {
    let n = match wanted {
        "fmod[finite-mixed4096]" => 4_096,
        "fmod[finite-mixed65536]" => 65_536,
        _ => return None,
    };
    let ns = PyDict::new(py);
    py.run(
        std::ffi::CString::new(format!(
            "import numpy as np\nN = {n}\ni = np.arange(N, dtype=np.float64)\nfmod_lhs = (i * 1.61803398875) - (N * 0.75)\nfmod_rhs = 0.125 + np.mod(i * 17.0 + 3.0, 101.0) / 7.0\n"
        ))
        .expect("fmod contract setup CString")
        .as_c_str(),
        Some(&ns),
        Some(&ns),
    )
    .expect("fmod contract setup");
    let lhs = ns.get_item("fmod_lhs").expect("fmod lhs");
    let rhs = ns.get_item("fmod_rhs").expect("fmod rhs");
    let args = PyTuple::new(py, [&lhs, &rhs]).expect("fmod contract arguments");
    Some((wanted.to_owned(), "fmod".to_owned(), args))
}

/// Structural checksum that survives tuples, bools and non-summable dtypes.
///
/// `.sum()` is the usual idiom in this crate and it does NOT generalise: it raises on a
/// tuple return, silently counts `True` as 1 for boolean results, and overflows for large
/// integer outputs. Hashing the bytes of every returned array compares what the caller
/// actually receives.
fn structural_checksum(value: &pyo3::Bound<'_, pyo3::PyAny>) -> u64 {
    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    let mut absorb = |bytes: &[u8]| {
        for &byte in bytes {
            state = (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    let fold = |v: &pyo3::Bound<'_, pyo3::PyAny>, absorb: &mut dyn FnMut(&[u8])| {
        // ARRAYS HASH THEIR BYTES; SCALARS HASH THEIR VALUE, and the split has to be on
        // `ndim` rather than on "does `tobytes` exist". `np.allclose` returns a PYTHON
        // `bool` while `fnp.allclose` returns a `np.bool_`: the numpy scalar has
        // `tobytes` and the Python one does not, so a `tobytes`-first probe hashed one
        // arm's raw byte and the other arm's `"True"` and called two identical answers
        // different. Normalising 0-d and scalar returns through `item()` makes the two
        // spellings of the same value hash the same, which is what a checksum is for.
        // (The return-type difference is real and worth its own look; it is not a
        // performance question and must not abort a performance sweep.)
        let is_array = v
            .getattr("ndim")
            .and_then(|n| n.extract::<usize>())
            .map(|n| n > 0)
            .unwrap_or(false);
        if is_array
            && let Ok(bytes) = v
                .call_method0("tobytes")
                .and_then(|b| b.extract::<Vec<u8>>())
        {
            absorb(&bytes);
            return;
        }
        let scalar = v.call_method0("item").unwrap_or_else(|_| v.clone());
        if let Ok(text) = scalar.str().map(|s| s.to_string()) {
            absorb(text.as_bytes());
        }
    };
    if let Ok(items) = value.cast::<PyTuple>() {
        for item in items.iter() {
            fold(&item, &mut absorb);
        }
    } else {
        fold(value, &mut absorb);
    }
    state
}

/// Interleaved min-of-K on both arms. ABBA order so a monotone drift over the window lands
/// on both arms equally rather than on whichever went second.
fn min_of_k(
    incumbent: &dyn Fn() -> (u128, u64),
    candidate: &dyn Fn() -> (u128, u64),
    k: usize,
) -> (u128, u128, u64, u64) {
    let mut best_a = u128::MAX;
    let mut best_b = u128::MAX;
    let mut sum_a = 0_u64;
    let mut sum_b = 0_u64;
    for round in 0..k {
        if round % 2 == 0 {
            let (a, ca) = incumbent();
            let (b, cb) = candidate();
            best_a = best_a.min(a);
            best_b = best_b.min(b);
            sum_a = ca;
            sum_b = cb;
        } else {
            let (b, cb) = candidate();
            let (a, ca) = incumbent();
            best_a = best_a.min(a);
            best_b = best_b.min(b);
            sum_a = ca;
            sum_b = cb;
        }
    }
    (best_a, best_b, sum_a, sum_b)
}

fn bench_vs_numpy_loss_sweep(_c: &mut Criterion) {
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_loss_sweep").expect("bench module");
        fnp_python(&module).expect("initialize fnp module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let numpy_version: String = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract()
            .expect("numpy version string");

        let (cases, disagree, slow) = discover_cases(py, &module);
        println!(
            "SWEEP_DISCOVERY numpy_version={numpy_version} worker={} \
             comparable_cases={} disagreements={} skipped={} \
             statistic=interleaved_min_of_k k=25 \
             THIS_GROUP_IS_TRIAGE_ONLY_no_null_no_ci_not_ledger_quotable=true",
            measurement_worker(),
            cases.len(),
            disagree.len(),
            slow.len(),
        );
        for item in disagree.iter() {
            println!("SWEEP_DISAGREEMENT case={item} note=correctness_question_not_timed");
        }
        for item in slow.iter() {
            println!("SWEEP_SKIPPED case={item}");
        }

        let mut ranked: Vec<(f64, String, u128, u128)> = Vec::with_capacity(cases.len());
        for case in cases.iter() {
            let label: String = case.get_item(0).expect("label").extract().expect("label");
            let name: String = case.get_item(1).expect("name").extract().expect("name");
            let args = case
                .get_item(2)
                .expect("args")
                .cast_into::<PyTuple>()
                .expect("args tuple");

            let ours = module.getattr(name.as_str()).expect("fnp callable");
            let theirs = numpy.getattr(name.as_str()).expect("numpy callable");

            let incumbent = || {
                let started = Instant::now();
                let r = theirs.call1(&args).expect("numpy arm");
                let elapsed = started.elapsed().as_nanos();
                let sum = structural_checksum(&r);
                black_box(&r);
                (elapsed, sum)
            };
            let candidate = || {
                let started = Instant::now();
                let r = ours.call1(&args).expect("fnp arm");
                let elapsed = started.elapsed().as_nanos();
                let sum = structural_checksum(&r);
                black_box(&r);
                (elapsed, sum)
            };

            // Warm both arms before either is timed: a first call can fault in pages, fill
            // a dispatch cache or resolve a lazy import, and whichever arm ran first would
            // otherwise be charged for all of it.
            for _ in 0..3 {
                black_box(incumbent());
                black_box(candidate());
            }
            let (numpy_ns, fnp_ns, sum_a, sum_b) = min_of_k(&incumbent, &candidate, 25);
            // NON-FATAL BY DESIGN. A single unstable op must not destroy a 260-case sweep:
            // the point of this group is the RANKING, and one case that cannot be checksummed
            // consistently is a finding to report, not a reason to lose the other 259. It is
            // dropped from the ranking rather than timed, because a cell whose arms return
            // different values is not measuring one computation.
            if sum_a != sum_b {
                println!(
                    "SWEEP_UNSTABLE case={label} numpy_checksum={sum_a:016x} \
                     fnp_checksum={sum_b:016x} note=dropped_from_ranking_arms_differ_under_repetition"
                );
                continue;
            }
            let ratio = numpy_ns as f64 / fnp_ns as f64;
            println!(
                "SWEEP_CELL case={label} numpy_ns={numpy_ns} fnp_ns={fnp_ns} \
                 ratio_numpy_over_fnp={ratio:.6} checksum={sum_a:016x} triage_only=true"
            );
            ranked.push((ratio, label, numpy_ns, fnp_ns));
        }

        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite ratios"));
        println!("SWEEP_RANKING_WORST_FIRST count={}", ranked.len());
        for (rank, (ratio, label, numpy_ns, fnp_ns)) in ranked.iter().take(25).enumerate() {
            println!(
                "SWEEP_RANK {rank} case={label} ratio_numpy_over_fnp={ratio:.6} \
                 slower_by={:.4}x numpy_ns={numpy_ns} fnp_ns={fnp_ns}",
                1.0 / ratio
            );
        }
    });
}

/// Full dual-null contract on ONE case, named by `FNP_SWEEP_CASE` (the sweep's `case=`
/// label). Same corpus, same discovery, same call - so this certifies the cell the sweep
/// ranked rather than a lookalike built separately.
fn bench_vs_numpy_worst_contract(_c: &mut Criterion) {
    let Ok(wanted) = std::env::var("FNP_SWEEP_CASE") else {
        println!(
            "SWEEP_CONTRACT skipped=true reason=FNP_SWEEP_CASE_unset \
             note=set_it_to_a_case_label_printed_by_the_sweep"
        );
        return;
    };
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_loss_contract").expect("bench module");
        fnp_python(&module).expect("initialize fnp module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let numpy_version: String = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract()
            .expect("numpy version string");

        let (label, name, args) = if let Some(case) = take_contract_case(py, &wanted) {
            case
        } else if let Some(case) = fmod_contract_case(py, &wanted) {
            case
        } else {
            let (cases, _, _) = discover_cases(py, &module);
            let mut chosen = None;
            for case in cases.iter() {
                let label: String = case.get_item(0).expect("label").extract().expect("label");
                if label == wanted {
                    let name: String = case.get_item(1).expect("name").extract().expect("name");
                    let args = case
                        .get_item(2)
                        .expect("args")
                        .cast_into::<PyTuple>()
                        .expect("args tuple");
                    chosen = Some((label, name, args));
                    break;
                }
            }
            chosen.unwrap_or_else(|| {
                panic!("FNP_SWEEP_CASE={wanted:?} matched no discovered case; run the sweep group")
            })
        };

        let ours = module.getattr(name.as_str()).expect("fnp callable");
        let theirs = numpy.getattr(name.as_str()).expect("numpy callable");
        assert!(
            !ours.is(&theirs),
            "dispatch trap: fnp.{name} resolved to the NumPy callable"
        );
        common::report_numpy_incumbent_identity(py, name.as_str(), &theirs);

        let expected = theirs.call1(&args).expect("numpy probe");
        let actual = ours.call1(&args).expect("fnp probe");
        assert_eq!(
            structural_checksum(&expected),
            structural_checksum(&actual),
            "{label}: arms disagree before timing"
        );
        println!(
            "SWEEP_CONTRACT_PARITY case={label} op={name} numpy_version={numpy_version} \
             exact_bytes=passed checksum={:016x}",
            structural_checksum(&expected)
        );

        let mut observe_incumbent = || {
            let started = Instant::now();
            let r = theirs.call1(&args).expect("numpy arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: structural_checksum(&r),
            }
        };
        let mut observe_candidate = || {
            let started = Instant::now();
            let r = ours.call1(&args).expect("fnp arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: structural_checksum(&r),
            }
        };
        let row = format!("python_loss_sweep_{}_vs_numpy", name);
        let (effect, incumbent_null, candidate_null) = common::run_dual_null_median_ci_contract(
            &row,
            &mut observe_incumbent,
            &mut observe_candidate,
        );
        let verdict = common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
        println!(
            "SWEEP_CONTRACT_RESULT case={label} op={name} verdict={verdict} \
             incumbent_median_ns={:.3} candidate_median_ns={:.3} \
             ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
             incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
             candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
             incumbent=numpy_live_same_invocation",
            effect.arm_a_median_ns,
            effect.arm_b_median_ns,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            incumbent_null.ratio_median,
            incumbent_null.ratio_ci_low,
            incumbent_null.ratio_ci_high,
            candidate_null.ratio_median,
            candidate_null.ratio_ci_low,
            candidate_null.ratio_ci_high,
        );
    });
}

fn main() {
    common::gated_main_with_source(
        include_str!("criterion_python_loss_sweep.rs"),
        &[
            ("bench_vs_numpy_loss_sweep", bench_vs_numpy_loss_sweep),
            (
                "bench_vs_numpy_worst_contract",
                bench_vs_numpy_worst_contract,
            ),
        ],
    );
}
