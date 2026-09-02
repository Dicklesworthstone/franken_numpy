//! Conformance tests for numpy statistical functions against NumPy oracle.
//!
//! Tests corrcoef, cov, average.

use std::process::Command;

fn numpy_oracle(script: &str) -> Result<String, String> {
    let output = Command::new("python3")
        .args(["-c", script])
        .output()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NumPy oracle failed: {stderr}\nScript: {script}"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

mod support;
use support::fnp_script;

fn indent_python(body: &str) -> String {
    body.lines().map(|line| format!("    {line}\n")).collect()
}

/// The outcome normalizer, shared by the verbatim-comparison scripts and by the
/// FMA-bounded value comparison below, so the two can never drift apart in what
/// they capture from a result.
const NORMALIZE_PY: &str = r#"def normalize(value):
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [normalize(item) for item in value]}
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if np.isscalar(value):
        scalar_type = type(value).__name__
        scalar_dtype = str(value.dtype) if hasattr(value, "dtype") else None
        scalar_value = value.item() if hasattr(value, "item") else value
        return {
            "kind": "scalar",
            "type": scalar_type,
            "dtype": scalar_dtype,
            "value": scalar_value,
        }
    return {"kind": "object", "type": type(value).__name__, "repr": repr(value)}
"#;

fn stats_outcome_body(body: &str) -> String {
    let indented = indent_python(body);
    r#"import json

__NORMALIZE__
try:
__BODY__    print(json.dumps(
        {"status": "ok", "result": normalize(result)},
        sort_keys=True,
        default=str,
    ))
except Exception as exc:
    message = str(exc).splitlines()[0] if str(exc) else ""
    print(json.dumps(
        {"status": "err", "type": type(exc).__name__, "message": message},
        sort_keys=True,
        default=str,
    ))
"#
    .replace("__NORMALIZE__", NORMALIZE_PY)
    .replace("__BODY__", &indented)
}

fn numpy_average_outcome_script(body: &str) -> String {
    numpy_stats_outcome_script(body)
}

fn fnp_average_outcome_script(body: &str) -> String {
    fnp_stats_outcome_script(body)
}

fn numpy_stats_outcome_script(body: &str) -> String {
    format!(
        "import numpy as np\n\
         MODULE = np\n\
         {}",
        stats_outcome_body(body)
    )
}

fn fnp_stats_outcome_script(body: &str) -> String {
    fnp_script(format!("MODULE = fnp\n{}", stats_outcome_body(body)))
}

/// Builds a single-process script that evaluates `body` against BOTH numpy and
/// fnp and reports the comparison in four lines:
///
/// 1. `STRUCT_OK` / `STRUCT_MISMATCH` — status, error type/message, kind, dtype,
///    shape and leaf count compared verbatim. None of those are FMA-affected, so
///    exactness there is correct and is retained.
/// 2. worst deviation over the numeric leaves, expressed in ULPs of the result
///    dtype (`inf` when a leaf is not comparable at all).
/// 3. how many numeric leaves were actually compared.
/// 4. both skeletons (two lines), so a failure names what diverged.
///
/// Both arms run in one interpreter against one numpy build: the ULP figure is a
/// property of the two implementations, not of two processes.
fn stats_value_comparison_script(body: &str) -> String {
    let indented: String = body
        .lines()
        .map(|line| format!("        {line}\n"))
        .collect();
    fnp_script(
        r#"import json
import math

__NORMALIZE__
def outcome(MODULE):
    try:
__BODY__        return {"status": "ok", "result": normalize(result)}
    except Exception as exc:
        message = str(exc).splitlines()[0] if str(exc) else ""
        return {"status": "err", "type": type(exc).__name__, "message": message}

def leaves_of(node):
    kind = node.get("kind")
    if kind == "ndarray":
        found = []
        def walk(value):
            if isinstance(value, list):
                for item in value:
                    walk(item)
            else:
                found.append((value, node["dtype"]))
        walk(node["values"])
        return (
            {
                "kind": "ndarray",
                "dtype": node["dtype"],
                "shape": node["shape"],
                "leaf_count": len(found),
            },
            found,
        )
    if kind == "scalar":
        return (
            {"kind": "scalar", "type": node["type"], "dtype": node["dtype"]},
            [(node["value"], node["dtype"])],
        )
    if kind == "tuple":
        skeletons = []
        found = []
        for item in node["items"]:
            skeleton, items = leaves_of(item)
            skeletons.append(skeleton)
            found.extend(items)
        return ({"kind": "tuple", "items": skeletons}, found)
    return (node, [])

def split_outcome(out):
    if out.get("status") != "ok":
        return (out, [])
    skeleton, found = leaves_of(out["result"])
    return ({"status": "ok", "result": skeleton}, found)

def eps_of(dtype):
    # Integer, bool and object leaves have no rounding budget: None means the
    # comparison falls back to exact equality rather than a tolerance.
    try:
        return float(np.finfo(np.dtype(dtype)).eps)
    except Exception:
        return None

def deviation_ulps(ours, theirs, dtype):
    eps = eps_of(dtype)
    numeric = (
        eps is not None
        and not isinstance(ours, bool)
        and not isinstance(theirs, bool)
        and isinstance(ours, (int, float))
        and isinstance(theirs, (int, float))
    )
    if not numeric:
        return 0.0 if ours == theirs else float("inf")
    if math.isnan(ours) and math.isnan(theirs):
        return 0.0
    if not math.isfinite(ours) or not math.isfinite(theirs):
        return 0.0 if ours == theirs else float("inf")
    scale = max(abs(theirs), 1e-300)
    return abs(ours - theirs) / scale / eps

numpy_skeleton, numpy_leaves = split_outcome(outcome(np))
fnp_skeleton, fnp_leaves = split_outcome(outcome(fnp))
numpy_json = json.dumps(numpy_skeleton, sort_keys=True, default=str)
fnp_json = json.dumps(fnp_skeleton, sort_keys=True, default=str)

if numpy_json != fnp_json:
    print("STRUCT_MISMATCH")
    print("inf")
    print(0)
else:
    print("STRUCT_OK")
    worst = 0.0
    for (ours, _), (theirs, dtype) in zip(fnp_leaves, numpy_leaves):
        worst = max(worst, deviation_ulps(ours, theirs, dtype))
    print(repr(worst))
    print(len(numpy_leaves))
print("numpy: " + numpy_json)
print("fnp:   " + fnp_json)
"#
        .replace("__NORMALIZE__", NORMALIZE_PY)
        .replace("__BODY__", &indented),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// corrcoef
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn corrcoef_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 6, 7, 8, 7])
result = fnp.corrcoef(x, y)
expected = np.corrcoef(x, y)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "corrcoef 1d should match numpy");
    Ok(())
}

#[test]
fn corrcoef_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.corrcoef(x)
expected = np.corrcoef(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "corrcoef 2d should match numpy");
    Ok(())
}

#[test]
fn corrcoef_rowvar_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.corrcoef(x, rowvar=False)
expected = np.corrcoef(x, rowvar=False)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "corrcoef rowvar=False should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cov
// ─────────────────────────────────────────────────────────────────────────────

/// What a container-keyword case is allowed to assert.
#[derive(Clone, Copy)]
enum StatsCaseContract {
    /// Compare the whole normalized outcome verbatim. Correct for error surfaces:
    /// an exception type and message are not FMA-affected, so exactness there
    /// costs nothing and catches a wrong error class immediately.
    ExactOutcome,
    /// Numeric cov/corrcoef output. Status, dtype, shape and leaf count are still
    /// compared exactly; only the numbers get the tolerance. `leaf_count` pins how
    /// many numbers actually get compared, so a `normalize` regression that emits
    /// an empty value list cannot make the case pass having checked nothing.
    FmaBoundedValues { leaf_count: usize },
}

/// Deviation budget for the container-keyword numeric cases, in ULPs of the
/// RESULT dtype.
///
/// These cases go through the same FMA-contracted path as the sweeps guarded by
/// [`COV_FMA_RELATIVE_BOUND`] — numpy's `cov` computes `dot(X, X.T)` as a 2-D
/// matmul, which reaches BLAS dgemm and contracts, while our Gram accumulates in
/// plain no-FMA Rust. On `cov([1,2,4], y=[2,1,0], ddof=0)` entry `[0][0]` that is
/// one rounding per term versus two: 1.5555555555555554 (ours) against
/// 1.5555555555555556 (numpy). Bit-equality is not merely unachieved here, it is
/// not well-defined — BLAS picks micro-kernels by shape, so the "correct" bytes
/// move with shape, BLAS build and host. This test failed on vmi1227854 and
/// passed on hz2 with no change to it.
///
/// The budget is expressed in ULPs rather than reusing `COV_FMA_RELATIVE_BOUND`
/// directly because one case returns float32, where 1e-12 sits five orders BELOW
/// the dtype's own resolution and would silently re-impose the bit-equality
/// demand this contract exists to remove. 64 ULPs is ~20x the couple of
/// contraction roundings reachable at these sizes (every case here has 2-3
/// observations) and ~11 orders below where a wrong formula lands, so a real
/// defect still fails loudly. The observed worst is printed on every run.
const COV_CONTAINER_VALUE_ULP_BOUND: f64 = 64.0;

#[test]
fn cov_corrcoef_python_container_keyword_outcomes_match_numpy() -> Result<(), String> {
    let cases = [
        (
            "cov rowvar false bias",
            "result = MODULE.cov(((1.0, 2.0, 3.0), (2.0, 4.0, 6.0)), rowvar=False, bias=True)",
            StatsCaseContract::FmaBoundedValues { leaf_count: 9 },
        ),
        (
            "cov y ddof",
            "result = MODULE.cov([1.0, 2.0, 4.0], y=[2.0, 1.0, 0.0], ddof=0)",
            StatsCaseContract::FmaBoundedValues { leaf_count: 4 },
        ),
        (
            "cov fweights aweights",
            "result = MODULE.cov(
    np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]]),
    fweights=[1, 2, 1],
    aweights=[1.0, 0.5, 2.0],
)",
            StatsCaseContract::FmaBoundedValues { leaf_count: 4 },
        ),
        (
            "cov weight shape error",
            "result = MODULE.cov([1.0, 2.0], fweights=[1, 2, 3])",
            StatsCaseContract::ExactOutcome,
        ),
        (
            "corrcoef rowvar false dtype",
            "result = MODULE.corrcoef(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]]), rowvar=False, dtype=np.float32)",
            StatsCaseContract::FmaBoundedValues { leaf_count: 4 },
        ),
        (
            // corrcoef's bias/ddof were REMOVED in numpy 2.4 (2.3.5 still accepts
            // them, warns, and ignores them). The two behaviours are incompatible
            // and the fleet runs both, so the contract is not a fixed outcome — it
            // is "whatever the installed numpy does", which is why these compare
            // exactly rather than under a tolerance. fnp forwards the call the
            // moment either is supplied, so numpy's own TypeError or its own bytes
            // are what comes back (deadlock-audit-hp9u2). Before that fix fnp
            // accepted ddof and returned a value on every build.
            "corrcoef y ddof compatibility",
            "result = MODULE.corrcoef([1.0, 2.0, 3.0], y=[3.0, 2.0, 1.0], ddof=0)",
            StatsCaseContract::ExactOutcome,
        ),
        (
            // bias travels with ddof through the same removal, and the native path
            // ignored it the same way. Covered separately so a fix that forwards
            // only ddof is caught.
            "corrcoef bias compatibility",
            "result = MODULE.corrcoef([1.0, 2.0, 3.0], y=[3.0, 2.0, 1.0], bias=True)",
            StatsCaseContract::ExactOutcome,
        ),
        (
            // The 2-D rowvar form takes a different route inside fnp than the
            // two-1-D-operand form above, so it needs its own case: forwarding must
            // not depend on which fast path the input would otherwise have hit.
            "corrcoef rowvar ddof compatibility",
            "result = MODULE.corrcoef(np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]), ddof=1)",
            StatsCaseContract::ExactOutcome,
        ),
        (
            // The contrast case that keeps the change honest: cov KEEPS bias and
            // ddof on every numpy including the vendored 2.5.0.dev0 oracle, so it
            // must still compute them, not inherit corrcoef's forwarding.
            "cov bias still supported",
            "result = MODULE.cov([1.0, 2.0, 3.0], y=[3.0, 2.0, 1.0], bias=True)",
            StatsCaseContract::FmaBoundedValues { leaf_count: 4 },
        ),
        (
            "corrcoef y shape error",
            "result = MODULE.corrcoef([1.0, 2.0], y=[1.0, 2.0, 3.0])",
            StatsCaseContract::ExactOutcome,
        ),
    ];

    for (name, body, contract) in cases {
        match contract {
            StatsCaseContract::ExactOutcome => {
                let numpy_result = numpy_oracle(&numpy_stats_outcome_script(body))?;
                let fnp_result = numpy_oracle(&fnp_stats_outcome_script(body))?;

                assert_eq!(
                    fnp_result, numpy_result,
                    "cov/corrcoef outcome mismatch for {name}\n\
                     numpy: {numpy_result}\nfnp:   {fnp_result}"
                );
            }
            StatsCaseContract::FmaBoundedValues { leaf_count } => {
                let output = numpy_oracle(&stats_value_comparison_script(body))?;
                let mut lines = output.lines();

                assert_eq!(
                    lines.next().unwrap_or_default(),
                    "STRUCT_OK",
                    "cov/corrcoef outcome structure mismatch for {name} — status, error \
                     type/message, dtype, shape and leaf count are compared exactly and \
                     one of them differs, which the FMA bound does not excuse\n{output}"
                );

                let worst_ulps: f64 = lines.next().unwrap_or_default().parse().map_err(|e| {
                    format!("could not parse worst ULP deviation for {name}: {e}; output: {output}")
                })?;
                let compared: usize = lines.next().unwrap_or_default().parse().map_err(|e| {
                    format!("could not parse compared leaf count for {name}: {e}; output: {output}")
                })?;

                assert_eq!(
                    compared, leaf_count,
                    "{name} compared {compared} numbers, expected {leaf_count} — the tolerance \
                     is only meaningful over the values it actually reaches\n{output}"
                );
                assert!(
                    worst_ulps <= COV_CONTAINER_VALUE_ULP_BOUND,
                    "{name} exceeded the named FMA-contraction budget: observed \
                     {worst_ulps:.3} ULP > {COV_CONTAINER_VALUE_ULP_BOUND} ULP of the result \
                     dtype. That is far more than a contraction difference, so suspect the \
                     formula, not the rounding\n{output}"
                );
                eprintln!(
                    "cov/corrcoef container case {name}: worst {worst_ulps:.3} ULP over {compared} values"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn cov_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 6, 7, 8, 7])
result = fnp.cov(x, y)
expected = np.cov(x, y)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cov 1d should match numpy");
    Ok(())
}

#[test]
fn cov_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([[0, 2], [1, 1], [2, 0]]).T
result = fnp.cov(x)
expected = np.cov(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cov 2d should match numpy");
    Ok(())
}

#[test]
fn cov_rowvar_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([[0, 2], [1, 1], [2, 0]])
result = fnp.cov(x, rowvar=False)
expected = np.cov(x, rowvar=False)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cov rowvar=False should match numpy");
    Ok(())
}

#[test]
fn cov_ddof() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4, 5])
result = fnp.cov(x, ddof=0)
expected = np.cov(x, ddof=0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cov ddof should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// average
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn average_python_container_keyword_outcomes_match_numpy() -> Result<(), String> {
    let cases = [
        ("list input scalar", "result = MODULE.average([1, 2, 3, 4])"),
        (
            "tuple input axis weights returned",
            "result = MODULE.average(
    ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
    axis=1,
    weights=[1.0, 2.0, 3.0],
    returned=True,
)",
        ),
        (
            "keepdims keyword",
            "result = MODULE.average(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0, keepdims=True)",
        ),
        (
            "zero weights error type",
            "result = MODULE.average([1.0, 2.0, 3.0], weights=[0.0, 0.0, 0.0])",
        ),
    ];

    for (name, body) in cases {
        let numpy_result = numpy_oracle(&numpy_average_outcome_script(body))?;
        let fnp_result = numpy_oracle(&fnp_average_outcome_script(body))?;

        assert_eq!(
            fnp_result, numpy_result,
            "average outcome mismatch for {name}\nnumpy: {numpy_result}\nfnp:   {fnp_result}"
        );
    }
    Ok(())
}

#[test]
fn average_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.average(a)
expected = np.average(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "average basic should match numpy");
    Ok(())
}

#[test]
fn average_with_weights() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
w = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
result = fnp.average(a, weights=w)
expected = np.average(a, weights=w)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "average with weights should match numpy"
    );
    Ok(())
}

#[test]
fn average_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.average(a, axis=0)
expected = np.average(a, axis=0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "average 2d axis=0 should match numpy"
    );
    Ok(())
}

#[test]
fn average_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.average(a, axis=1)
expected = np.average(a, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "average 2d axis=1 should match numpy"
    );
    Ok(())
}

#[test]
fn average_returned() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
w = np.array([1, 2, 3, 4, 5])
result, sum_weights = fnp.average(a, weights=w, returned=True)
expected, exp_sum = np.average(a, weights=w, returned=True)
print(np.allclose(result, expected) and np.allclose(sum_weights, exp_sum))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "average returned should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn corrcoef_diagonal_is_one() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 6, 7, 8, 7])
result = fnp.corrcoef(x, y)
# Diagonal should be 1.0 (correlation of variable with itself)
print(np.allclose(np.diag(result), 1.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "corrcoef diagonal should be 1");
    Ok(())
}

#[test]
fn cov_variance_on_diagonal() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4, 5])
cov_xx = fnp.cov(x)
var_x = np.var(x, ddof=1)  # cov uses ddof=1 by default
print(np.allclose(cov_xx, var_x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cov of single array should equal variance"
    );
    Ok(())
}

#[test]
fn average_no_weights_equals_mean() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
avg = fnp.average(a)
mean = np.mean(a)
print(np.allclose(avg, mean))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "average without weights should equal mean"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cov_with_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
fnp_result = fnp.cov(x)
np_result = np.cov(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cov with nan should match numpy");
    Ok(())
}

#[test]
fn corrcoef_with_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
y = np.array([5.0, 6.0, np.nan, 8.0, 9.0])
fnp_result = fnp.corrcoef(x, y)
np_result = np.corrcoef(x, y)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "corrcoef with nan should match numpy"
    );
    Ok(())
}

#[test]
fn cov_constant_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
fnp_result = fnp.cov(x)
np_result = np.cov(x)
# Constant array has zero variance, cov returns 0
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cov constant array should match numpy"
    );
    Ok(())
}

#[test]
fn corrcoef_constant_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
fnp_result = fnp.corrcoef(x, y)
np_result = np.corrcoef(x, y)
# Constant array has zero std, corrcoef produces nan
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "corrcoef constant array should match numpy"
    );
    Ok(())
}

#[test]
fn average_with_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
fnp_result = fnp.average(a)
np_result = np.average(a)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "average with nan should match numpy");
    Ok(())
}

#[test]
fn average_with_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.inf, 3.0, 4.0, 5.0])
fnp_result = fnp.average(a)
np_result = np.average(a)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "average with inf should match numpy");
    Ok(())
}

#[test]
fn average_zero_weights() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
w = np.array([1.0, 0.0, 1.0])
fnp_result = fnp.average(a, weights=w)
np_result = np.average(a, weights=w)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "average with zero weights should match numpy"
    );
    Ok(())
}

#[test]
fn cov_single_observation() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x = np.array([5.0])
fnp_result = fnp.cov(x)
np_result = np.cov(x)
# Single observation: ddof=1 leads to division by zero -> nan
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cov single observation should match numpy"
    );
    Ok(())
}

/// THE NAMED BOUND FOR THE FMA-AFFECTED cov/corrcoef PATH.
///
/// These tests used to pin a sha256 of fnp's own output bytes, i.e. they asserted
/// BIT-EQUALITY, while a third test on the same operation asserted only
/// `allclose`. That was a contradiction in the specification, and the exact-bit
/// side of it was asking for something arithmetic cannot deliver.
///
/// PROVEN, not assumed. NumPy's `cov` computes `dot(X, X.T)` as a 2-D matmul, so
/// it goes through BLAS dgemm, whose inner loop is FMA-contracted. Reproduced on
/// the failing case `cov([1,2,4], y=[2,1,0], ddof=0)`, entry [0][0], centred
/// operands identical on both sides:
///
///   accumulate with separate multiply+add (our kernel): 4.666666666666666
///     -> / 3 = 1.5555555555555554  == what fnp returns
///   accumulate with `math.fma` (a dgemm inner loop):    4.666666666666667
///     -> / 3 = 1.5555555555555556  == what numpy.cov returns
///
/// One rounding per term versus two. A no-FMA implementation cannot reproduce it
/// on any host, so exact-bit equality against numpy's cov is unachievable rather
/// than merely unachieved. Worse, it is not even a fixed target: an explicit
/// 1x3 dot of the same data yields the NO-FMA value while cov's internal 2x3 dot
/// yields the FMA one, because BLAS selects micro-kernels by shape — so the
/// "correct" bytes vary with shape and with the installed BLAS build.
///
/// The honest contract is therefore a bounded relative deviation. 1e-12 is ~4
/// orders of magnitude above the accumulated contraction difference at these
/// sizes and ~3 orders BELOW the 1e-9 `allclose` already in use, so it still
/// fails loudly on a wrong formula (which lands at 1e-2..1e-1, not 1e-15) while
/// tolerating the rounding no one can control. The observed worst deviation is
/// printed on every run, so silent drift toward the bound is visible.
///
/// This is correcting a gate that asserted the impossible, not weakening one:
/// the parity assertion against numpy is unchanged and still runs.
const COV_FMA_RELATIVE_BOUND: f64 = 1e-12;

#[test]
fn cov_native_fast_path_matches_numpy_across_shape_ddof_bias() -> Result<(), String> {
    // Locks the zero-copy parallel-Gram fast path (rowvar=True, no y, contiguous f64):
    // it must match numpy.cov within tolerance across variable/observation counts,
    // ddof, and bias. (Reassociated dot sums -> allclose, not bit-exact, like the prior
    // matmul path.)
    let script = fnp_script(
        r#"
ok = True
worst = 0.0

def reldev(f, n):
    f = np.asarray(f, dtype=np.float64); n = np.asarray(n, dtype=np.float64)
    m = np.isfinite(f) & np.isfinite(n)
    if not m.any():
        return 0.0
    scale = np.maximum(np.abs(n[m]), 1e-300)
    return float(np.max(np.abs(f[m] - n[m]) / scale))

rng = np.random.default_rng(3)
for shape in [(50, 2000), (5, 30), (1, 100), (3, 3), (10, 11), (200, 500)]:
    X = rng.standard_normal(shape)
    for kw in [{}, {"bias": True}, {"ddof": 0}, {"ddof": 2}, {"rowvar": True}]:
        f = np.asarray(fnp.cov(X, **kw)); n = np.asarray(np.cov(X, **kw))
        if f.shape != n.shape or not np.allclose(f, n, rtol=1e-9, atol=1e-12, equal_nan=True):
            ok = False
        worst = max(worst, reldev(f, n))
print(ok)
print(f"{worst:.3e}")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    assert_eq!(
        lines.next().unwrap_or_default(),
        "True",
        "cov fast path must match numpy across shape/ddof/bias"
    );
    let worst: f64 =
        lines.next().unwrap_or_default().parse().map_err(|e| {
            format!("could not parse worst relative deviation: {e}; output: {result}")
        })?;
    assert!(
        worst <= COV_FMA_RELATIVE_BOUND,
        "cov fast path exceeded the named FMA-contraction bound: observed {worst:.3e} > \
         {COV_FMA_RELATIVE_BOUND:.0e}. That is far larger than a contraction difference, \
         so suspect the formula, not the rounding. Output: {result}"
    );
    Ok(())
}

#[test]
fn corrcoef_native_fast_path_matches_numpy_across_shapes() -> Result<(), String> {
    // Locks the zero-copy parallel-Gram corrcoef fast path (rowvar=True, no y, f64):
    // cov via the shared Gram core, then normalize by diagonal stddevs and clip to
    // [-1, 1] — must match numpy.corrcoef within tolerance across variable/observation
    // counts, including the 1-D (scalar 1.0) case.
    let script = fnp_script(
        r#"
ok = True
rng = np.random.default_rng(5)
for shape in [(50, 2000), (5, 30), (1, 100), (3, 3), (10, 11), (200, 500)]:
    X = rng.standard_normal(shape)
    f = np.asarray(fnp.corrcoef(X)); n = np.asarray(np.corrcoef(X))
    if f.shape != n.shape or not np.allclose(f, n, rtol=1e-9, atol=1e-12, equal_nan=True):
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "corrcoef fast path must match numpy across shapes"
    );
    Ok(())
}

#[test]
fn cov_corrcoef_long_observation_ufunc_gate_matches_numpy_within_fma_bound() -> Result<(), String> {
    // Locks the long-observation UFuncArray route for rowvar=True/no-y f64 inputs.
    // The route intentionally changes the accumulation tree, so equality is by
    // NumPy-compatible allclose within COV_FMA_RELATIVE_BOUND. It previously also
    // pinned a sha256 of fnp's output bytes; that pin asserted bit-equality with a
    // BLAS-dgemm result and was unachievable — see COV_FMA_RELATIVE_BOUND.
    let script = fnp_script(
        r#"
def reldev(f, n):
    f = np.asarray(f, dtype=np.float64); n = np.asarray(n, dtype=np.float64)
    m = np.isfinite(f) & np.isfinite(n)
    if not m.any():
        return 0.0
    scale = np.maximum(np.abs(n[m]), 1e-300)
    return float(np.max(np.abs(f[m] - n[m]) / scale))

rng = np.random.default_rng(13)
X = rng.standard_normal((50, 5000))
f_cov = np.asarray(fnp.cov(X))
n_cov = np.asarray(np.cov(X))
f_corr = np.asarray(fnp.corrcoef(X))
n_corr = np.asarray(np.corrcoef(X))
ok = (
    f_cov.shape == n_cov.shape
    and f_corr.shape == n_corr.shape
    and np.allclose(f_cov, n_cov, rtol=1e-9, atol=1e-12, equal_nan=True)
    and np.allclose(f_corr, n_corr, rtol=1e-9, atol=1e-12, equal_nan=True)
)
worst = max(reldev(f_cov, n_cov), reldev(f_corr, n_corr))
print(ok)
print(f"{worst:.3e}")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    assert_eq!(
        lines.next().unwrap_or_default(),
        "True",
        "long-observation cov/corrcoef route must match numpy"
    );
    let worst: f64 =
        lines.next().unwrap_or_default().parse().map_err(|e| {
            format!("could not parse worst relative deviation: {e}; output: {result}")
        })?;
    assert!(
        worst <= COV_FMA_RELATIVE_BOUND,
        "long-observation cov/corrcoef exceeded the named FMA-contraction bound: \
         observed {worst:.3e} > {COV_FMA_RELATIVE_BOUND:.0e}. Output: {result}"
    );
    Ok(())
}

#[test]
fn cov_corrcoef_orientation_and_scalar_edge_cases_match_numpy() -> Result<(), String> {
    // Regression for two parity gaps: (1) a genuine 2-D (1, N) input with rowvar=False
    // is N variables -> (N, N), not a scalar (the old shape[0]!=1 guard wrongly skipped
    // the transpose); a true 1-D input stays one variable. (2) cov/corrcoef of a single
    // variable squeezes to a 0-d scalar, not (1, 1).
    let script = fnp_script(
        r#"
ok = True
rng = np.random.default_rng(0)
for fn in ("cov", "corrcoef"):
    ffn = getattr(fnp, fn); nfn = getattr(np, fn)
    for shape in [(1, 5), (5, 1), (1, 100), (3, 5), (100,), (4, 4), (1, 1)]:
        X = rng.standard_normal(shape)
        for kw in ({}, {"rowvar": False}, {"rowvar": True}):
            f = np.asarray(ffn(X, **kw)); n = np.asarray(nfn(X, **kw))
            if f.shape != n.shape or not np.allclose(f, n, rtol=1e-9, atol=1e-12, equal_nan=True):
                ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cov/corrcoef orientation + single-variable scalar must match numpy"
    );
    Ok(())
}

#[test]
fn int_cov_corrcoef_via_f64_conversion_bit_exact_matches_numpy() -> Result<(), String> {
    // Integer/bool cov and corrcoef convert once to f64 (numpy's own chain
    // converts to result_type(m, f64) before any arithmetic - pinned incl.
    // 2^62-scale values) and ride the converged f64 Gram-lane kernels.
    // Weights/dtype-override forms and small inputs keep prior behavior.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(223)
verdicts = []
M = rng.integers(-1000, 1000, (128, 20_000))
Y = rng.integers(-1000, 1000, (128, 20_000))
cases = [
    ("cov", dict()), ("cov", dict(rowvar=False)), ("cov", dict(bias=True)),
    ("cov", dict(ddof=0)), ("corrcoef", dict()), ("corrcoef", dict(rowvar=False)),
]
for fname, kw in cases:
    ff = getattr(fnp, fname); nf = getattr(np, fname)
    r = ff(M, **kw); e = nf(M, **kw)
    ra = np.asarray(r); ea = np.asarray(e)
    if ra.dtype != ea.dtype or ra.shape != ea.shape or ra.tobytes() != ea.tobytes():
        verdicts.append(f"FAIL {fname} {kw}")
# two-operand y: int stays on prior behavior (conversion is single-operand
# only - the two-operand native Gram path is not byte-level for converted
# inputs); parity asserted at the surface's established allclose level.
r2 = np.asarray(fnp.cov(M, Y)); e2 = np.asarray(np.cov(M, Y))
if r2.shape != e2.shape or not np.allclose(r2, e2, rtol=1e-10):
    verdicts.append("FAIL cov two-operand allclose")
r2 = np.asarray(fnp.corrcoef(M, Y)); e2 = np.asarray(np.corrcoef(M, Y))
if r2.shape != e2.shape or not np.allclose(r2, e2, rtol=1e-10):
    verdicts.append("FAIL corrcoef two-operand allclose")
# bool + huge values (conversion unconditional)
B = rng.random((64, 20_000)) > 0.6
if np.asarray(fnp.cov(B)).tobytes() != np.asarray(np.cov(B)).tobytes():
    verdicts.append("FAIL bool cov")
H = rng.integers(-2**62, 2**62, (32, 10_000))
if np.asarray(fnp.cov(H)).tobytes() != np.asarray(np.cov(H)).tobytes():
    verdicts.append("FAIL huge-value cov")
# fweights form stays byte-identical (conversion is numpy-transparent)
fw = rng.integers(1, 5, 20_000)
if np.asarray(fnp.cov(M, fweights=fw)).tobytes() != np.asarray(np.cov(M, fweights=fw)).tobytes():
    verdicts.append("FAIL fweights parity")
# small input keeps prior behavior
S = rng.integers(-100, 100, (8, 50))
if np.asarray(fnp.cov(S)).tobytes() != np.asarray(np.cov(S)).tobytes():
    verdicts.append("FAIL small parity")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.integers(-1000, 1000, (256, 100_000))
tn = best(lambda: np.cov(W)); tf = best(lambda: fnp.cov(W))
print(f"COV_INT_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn = best(lambda: np.corrcoef(W)); tf = best(lambda: fnp.corrcoef(W))
print(f"CORRCOEF_INT_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces COV/CORRCOEF_INT_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "int cov/corrcoef via f64 conversion must be bit-identical to numpy: {result}"
    );
    Ok(())
}
