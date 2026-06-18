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

fn fnp_script(body: String) -> String {
    let library_name = format!(
        "{}fnp_python{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let module_path = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join(&library_name)))
        .unwrap_or_else(|| library_name.into());
    let module_literal = format!("{module_path:?}");
    format!(
        "import importlib.util\n\
         import numpy as np\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}

fn indent_python(body: &str) -> String {
    body.lines().map(|line| format!("    {line}\n")).collect()
}

fn average_outcome_body(body: &str) -> String {
    let indented = indent_python(body);
    r#"import json

def normalize(value):
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
    .replace("__BODY__", &indented)
}

fn numpy_average_outcome_script(body: &str) -> String {
    format!(
        "import numpy as np\n\
         MODULE = np\n\
         {}",
        average_outcome_body(body)
    )
}

fn fnp_average_outcome_script(body: &str) -> String {
    fnp_script(format!("MODULE = fnp\n{}", average_outcome_body(body)))
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
        (
            "list input scalar",
            "result = MODULE.average([1, 2, 3, 4])",
        ),
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

#[test]
fn cov_native_fast_path_matches_numpy_across_shape_ddof_bias() -> Result<(), String> {
    // Locks the zero-copy parallel-Gram fast path (rowvar=True, no y, contiguous f64):
    // it must match numpy.cov within tolerance across variable/observation counts,
    // ddof, and bias. (Reassociated dot sums -> allclose, not bit-exact, like the prior
    // matmul path.)
    let script = fnp_script(
        r#"
import hashlib
ok = True
proof = bytearray()
rng = np.random.default_rng(3)
for shape in [(50, 2000), (5, 30), (1, 100), (3, 3), (10, 11), (200, 500)]:
    X = rng.standard_normal(shape)
    for kw in [{}, {"bias": True}, {"ddof": 0}, {"ddof": 2}, {"rowvar": True}]:
        f = np.asarray(fnp.cov(X, **kw)); n = np.asarray(np.cov(X, **kw))
        if f.shape != n.shape or not np.allclose(f, n, rtol=1e-9, atol=1e-12, equal_nan=True):
            ok = False
        proof.extend(str(f.shape).encode())
        proof.extend(str(f.dtype).encode())
        proof.extend(np.ascontiguousarray(f).view(np.uint8).tobytes())
print(ok)
print(hashlib.sha256(proof).hexdigest())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().next().unwrap_or_default(),
        "True",
        "cov fast path must match numpy across shape/ddof/bias"
    );
    assert_eq!(
        result.lines().nth(1).unwrap_or_default(),
        "80bb612d7fb9522ca878001af0fd6ba17a68eebb4a4e117006caf69a71fa3387",
        "cov fast path golden sha256 drifted"
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
fn cov_corrcoef_long_observation_ufunc_gate_matches_numpy_sha256() -> Result<(), String> {
    // Locks the long-observation UFuncArray route for rowvar=True/no-y f64 inputs.
    // The route intentionally changes the accumulation tree, so equality is by
    // NumPy-compatible allclose and the deterministic fnp output bytes are pinned.
    let script = fnp_script(
        r#"
import hashlib
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
proof = bytearray()
for arr in (f_cov, f_corr):
    proof.extend(str(arr.shape).encode())
    proof.extend(str(arr.dtype).encode())
    proof.extend(np.ascontiguousarray(arr).view(np.uint8).tobytes())
print(ok)
print(hashlib.sha256(proof).hexdigest())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().next().unwrap_or_default(),
        "True",
        "long-observation cov/corrcoef route must match numpy"
    );
    assert_eq!(
        result.lines().nth(1).unwrap_or_default(),
        "72b7359608b619263e2194f0f0d802f27eb25cdbcf133acbecee0a2fa2919667",
        "long-observation cov/corrcoef golden sha256 drifted"
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
