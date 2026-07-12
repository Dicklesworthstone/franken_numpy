//! Conformance tests for numpy einsum and einsum_path against NumPy oracle.
//!
//! Tests einsum, einsum_path.

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

fn outcome_body(body: &str) -> String {
    let indented = indent_python(body);
    r#"import json

def normalize(value):
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [normalize(item) for item in value],
        }
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
__BODY__    payload = {"status": "ok", "result": normalize(result)}
    if "out" in locals():
        payload["out"] = normalize(out)
        payload["result_is_out"] = result is out
    print(json.dumps(payload, sort_keys=True, default=str))
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

fn numpy_outcome_script(body: &str) -> String {
    format!(
        "import numpy as np\n\
         MODULE = np\n\
         {}",
        outcome_body(body)
    )
}

fn fnp_outcome_script(body: &str) -> String {
    fnp_script(format!("MODULE = fnp\n{}", outcome_body(body)))
}

#[test]
fn einsum_keyword_and_path_outcomes_match_numpy() -> Result<(), String> {
    let cases = [
        (
            "scalar contraction",
            "result = MODULE.einsum('i,i->', [1, 2, 3], [4, 5, 6])",
        ),
        (
            "dtype order casting",
            "result = MODULE.einsum(
    'ij,jk->ik',
    np.array([[1, 2], [3, 4]], dtype=np.int32),
    np.array([[5, 6], [7, 8]], dtype=np.int32),
    dtype=np.float64,
    casting='unsafe',
    order='F',
)",
        ),
        (
            "optimize out forwarding",
            "out = np.empty((2, 2), dtype=np.float64)
result = MODULE.einsum(
    'ij,jk->ik',
    np.array([[1.0, 2.0], [3.0, 4.0]]),
    np.array([[5.0, 6.0], [7.0, 8.0]]),
    optimize='greedy',
    out=out,
)",
        ),
        (
            "path tuple output",
            "result = MODULE.einsum_path(
    'ij,jk,kl->il',
    np.ones((2, 3)),
    np.ones((3, 4)),
    np.ones((4, 2)),
    optimize='optimal',
)",
        ),
        (
            "subscript dimension error",
            "result = MODULE.einsum('ij,j->i', np.arange(3), np.arange(3))",
        ),
    ];

    for (name, body) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script(body))?;
        let fnp_result = numpy_oracle(&fnp_outcome_script(body))?;

        assert_eq!(
            fnp_result, numpy_result,
            "einsum outcome mismatch for {name}\nnumpy: {numpy_result}\nfnp:   {fnp_result}"
        );
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// einsum - basic operations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn einsum_trace() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(9).reshape(3, 3)
result = fnp.einsum('ii', a)
expected = np.einsum('ii', a)
print(np.array_equal(result, expected) and result == np.trace(a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "einsum trace should match numpy");
    Ok(())
}

#[test]
fn einsum_diag() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(9).reshape(3, 3)
result = fnp.einsum('ii->i', a)
expected = np.einsum('ii->i', a)
print(np.array_equal(result, expected) and np.array_equal(result, np.diag(a)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "einsum diag should match numpy");
    Ok(())
}

#[test]
fn einsum_f64_single_operand_diagonal_view_and_trace_golden_sha256() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib
n = 32
a = np.arange(n * n, dtype=np.float64).reshape(n, n)
diag = fnp.einsum('ii->i', a)
diag[0] = -123.5
trace = fnp.einsum('ii', a)
expected_diag = np.einsum('ii->i', a)
expected_trace = np.einsum('ii', a)
h = hashlib.sha256()
h.update(np.asarray(trace).tobytes())
h.update(np.asarray(diag).copy(order='C').tobytes())
h.update(str(diag.strides).encode())
parity = (
    np.array_equal(diag, expected_diag)
    and trace == expected_trace
    and type(trace).__name__ == type(expected_trace).__name__
    and np.shares_memory(a, diag)
    and diag.flags.writeable
    and a[0, 0] == -123.5
)
print(parity)
print(h.hexdigest())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    assert_eq!(
        lines.next(),
        Some("True"),
        "einsum f64 diagonal/trace parity failed: {result}"
    );
    assert_eq!(
        lines.next(),
        Some("9dc21300099f7dec79fc2a202b2c654f5785d714d7f56550904b001c1c64e72d"),
        "einsum f64 diagonal/trace golden digest drifted: {result}"
    );
    Ok(())
}

#[test]
fn einsum_f64_trace_edge_bits_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def bits(value):
    return int(np.asarray(value, dtype=np.float64).view(np.uint64))

cases = [
    np.array([[-0.0]], dtype=np.float64),
    np.array([[np.nan]], dtype=np.float64),
    np.array([[np.inf]], dtype=np.float64),
    np.array([[-np.inf]], dtype=np.float64),
    np.arange(9, dtype=np.float64).reshape(3, 3).T,
]
ok = True
for a in cases:
    ours = fnp.einsum('ii', a)
    expected = np.einsum('ii', a)
    ok = ok and type(ours).__name__ == type(expected).__name__
    ok = ok and bits(ours) == bits(expected)
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "einsum f64 trace edge bits should match numpy"
    );
    Ok(())
}

#[test]
fn einsum_sum_all() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(12).reshape(3, 4)
result = fnp.einsum('ij->', a)
expected = np.einsum('ij->', a)
print(result == expected == np.sum(a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "einsum sum all should match numpy");
    Ok(())
}

#[test]
fn einsum_sum_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(12).reshape(3, 4)
result = fnp.einsum('ij->j', a)
expected = np.einsum('ij->j', a)
print(np.array_equal(result, expected) and np.array_equal(result, np.sum(a, axis=0)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "einsum sum axis 0 should match numpy"
    );
    Ok(())
}

#[test]
fn einsum_f64_single_operand_reduction_fast_path_golden_sha256() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib
a = (np.arange(63 * 67).astype(np.float64) * 0.125 - 7.0).reshape(63, 67)
subs = ['ij->', 'ij->i', 'ij->j']
h = hashlib.sha256()
ok = True
for sub in subs:
    ours = np.asarray(fnp.einsum(sub, a))
    theirs = np.asarray(np.einsum(sub, a))
    ok = ok and ours.dtype == theirs.dtype
    ok = ok and ours.shape == theirs.shape
    ok = ok and np.allclose(ours, theirs)
    h.update(sub.encode())
    h.update(str(ours.dtype).encode())
    h.update(str(ours.shape).encode())
    h.update(ours.tobytes())
for shape in [(0, 7), (5, 0), (0, 0)]:
    empty = np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape)
    for sub in subs:
        ours = np.asarray(fnp.einsum(sub, empty))
        theirs = np.asarray(np.einsum(sub, empty))
        ok = ok and ours.dtype == theirs.dtype
        ok = ok and ours.shape == theirs.shape
        ok = ok and np.allclose(ours, theirs)
print(ok)
print(h.hexdigest())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    assert_eq!(
        lines.next(),
        Some("True"),
        "einsum f64 single-operand reduction parity failed: {result}"
    );
    assert_eq!(
        lines.next(),
        Some("a20af4420189d8fe967cddcb2277893bd8ed7f1940779c4d0d0e3c02e1982999"),
        "einsum f64 single-operand reduction golden digest drifted: {result}"
    );
    Ok(())
}

#[test]
fn einsum_single_operand_reductions_preserve_numpy_dtype_and_golden_sha256() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib
cases = [
    ('float32', np.array([[1.25, 2.5, 3.75], [4.5, 5.25, 6.0]], dtype=np.float32)),
    ('int32', np.array([[2000000000, 2000000000, 2000000000], [1, 2, 3]], dtype=np.int32)),
    ('int64', np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)),
]
subs = ['ij->i', 'ij->j', 'ij->']
h = hashlib.sha256()
ok = True
for name, a in cases:
    for sub in subs:
        ours = fnp.einsum(sub, a)
        theirs = np.einsum(sub, a)
        ours_arr = np.asarray(ours)
        theirs_arr = np.asarray(theirs)
        ok = ok and ours_arr.dtype == theirs_arr.dtype
        ok = ok and ours_arr.shape == theirs_arr.shape
        ok = ok and np.array_equal(ours_arr, theirs_arr)
        h.update(name.encode())
        h.update(sub.encode())
        h.update(str(ours_arr.dtype).encode())
        h.update(str(ours_arr.shape).encode())
        h.update(ours_arr.tobytes())
print(ok)
print(h.hexdigest())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    assert_eq!(
        lines.next(),
        Some("True"),
        "einsum reduction dtype/value parity failed: {result}"
    );
    assert_eq!(
        lines.next(),
        Some("f61b08086facbf414740fa48bc71fb1b7ac563be07ff1d40596bc511638a28ee"),
        "einsum reduction dtype golden digest drifted: {result}"
    );
    Ok(())
}

#[test]
fn einsum_transpose() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(12).reshape(3, 4)
result = fnp.einsum('ij->ji', a)
expected = np.einsum('ij->ji', a)
print(np.array_equal(result, expected) and np.array_equal(result, a.T))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "einsum transpose should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// einsum - two operand operations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn einsum_matmul() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
result = fnp.einsum('ij,jk->ik', a, b)
expected = np.einsum('ij,jk->ik', a, b)
print(np.array_equal(result, expected) and np.array_equal(result, a @ b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "einsum matmul should match numpy");
    Ok(())
}

#[test]
fn einsum_inner() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.einsum('i,i->', a, b)
expected = np.einsum('i,i->', a, b)
print(result == expected == np.inner(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "einsum inner should match numpy");
    Ok(())
}

#[test]
fn einsum_outer() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5])
result = fnp.einsum('i,j->ij', a, b)
expected = np.einsum('i,j->ij', a, b)
print(np.array_equal(result, expected) and np.array_equal(result, np.outer(a, b)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "einsum outer should match numpy");
    Ok(())
}

#[test]
fn einsum_hadamard() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(2, 3)
result = fnp.einsum('ij,ij->ij', a, b)
expected = np.einsum('ij,ij->ij', a, b)
print(np.array_equal(result, expected) and np.array_equal(result, a * b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "einsum hadamard should match numpy");
    Ok(())
}

#[test]
fn einsum_batch_matmul() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
b = np.arange(32).reshape(2, 4, 4)
result = fnp.einsum('nij,njk->nik', a, b)
expected = np.einsum('nij,njk->nik', a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "einsum batch matmul should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// einsum - with keyword arguments
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn einsum_optimize_greedy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.random.rand(5, 10)
b = np.random.rand(10, 15)
c = np.random.rand(15, 5)
result = fnp.einsum('ij,jk,kl->il', a, b, c, optimize='greedy')
expected = np.einsum('ij,jk,kl->il', a, b, c, optimize='greedy')
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "einsum with optimize=greedy should match numpy"
    );
    Ok(())
}

#[test]
fn einsum_out_parameter() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
out_fnp = np.zeros((2, 2))
out_np = np.zeros((2, 2))
fnp.einsum('ij,jk->ik', a, b, out=out_fnp)
np.einsum('ij,jk->ik', a, b, out=out_np)
print(np.array_equal(out_fnp, out_np))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "einsum with out parameter should match numpy"
    );
    Ok(())
}

#[test]
fn einsum_dtype_casting() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
b = np.array([4, 5, 6], dtype='int32')
result = fnp.einsum('i,i->', a, b, dtype='float64')
expected = np.einsum('i,i->', a, b, dtype='float64')
print(result == expected and isinstance(result, np.floating))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "einsum with dtype should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// einsum_path
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn einsum_path_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.random.rand(5, 10)
b = np.random.rand(10, 15)
c = np.random.rand(15, 5)
path_fnp, info_fnp = fnp.einsum_path('ij,jk,kl->il', a, b, c)
path_np, info_np = np.einsum_path('ij,jk,kl->il', a, b, c)
print(path_fnp == path_np)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "einsum_path basic should match numpy"
    );
    Ok(())
}

#[test]
fn einsum_path_greedy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.random.rand(5, 10)
b = np.random.rand(10, 15)
c = np.random.rand(15, 5)
path_fnp, _ = fnp.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')
path_np, _ = np.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')
print(path_fnp == path_np)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "einsum_path greedy should match numpy"
    );
    Ok(())
}

#[test]
fn einsum_path_optimal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.random.rand(3, 4)
b = np.random.rand(4, 5)
c = np.random.rand(5, 3)
path_fnp, _ = fnp.einsum_path('ij,jk,kl->il', a, b, c, optimize='optimal')
path_np, _ = np.einsum_path('ij,jk,kl->il', a, b, c, optimize='optimal')
print(path_fnp == path_np)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "einsum_path optimal should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// einsum - implicit mode
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn einsum_implicit_sum() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(12).reshape(3, 4)
result = fnp.einsum('ij', a)
expected = np.einsum('ij', a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "einsum implicit mode should match numpy"
    );
    Ok(())
}

#[test]
fn einsum_implicit_matmul() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
result = fnp.einsum('ij,jk', a, b)
expected = np.einsum('ij,jk', a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "einsum implicit matmul should match numpy"
    );
    Ok(())
}

#[test]
fn einsum_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(9, dtype=np.float64).reshape(3, 3)
fnp_result = fnp.einsum('ii', a)
np_result = np.einsum('ii', a)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "einsum scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn einsum_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j]], dtype=np.complex128)
b = np.array([[1, 2+1j], [3-1j, 4]], dtype=np.complex128)
fnp_result = fnp.einsum('ij,jk->ik', a, b)
np_result = np.einsum('ij,jk->ik', a, b)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "einsum complex should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Error behavior tests
// ─────────────────────────────────────────────────────────────────────────────

fn classify_error(script: &str) -> String {
    let output = std::process::Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("python3 should be available");
    if output.status.success() {
        "ok".to_string()
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("ValueError")
            || stderr.contains("invalid")
            || stderr.contains("subscript")
        {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn einsum_invalid_subscript_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3)
fnp.einsum('xyz', a)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
np.einsum('xyz', a)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "einsum invalid subscript should raise same error as numpy"
    );
}

#[test]
fn einsum_dimension_mismatch_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3)
b = fnp.arange(20).reshape(4, 5)
fnp.einsum('ij,jk->ik', a, b)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
b = np.arange(20).reshape(4, 5)
np.einsum('ij,jk->ik', a, b)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "einsum dimension mismatch should raise same error as numpy"
    );
}

#[test]
fn einsum_preserves_operand_dtype_like_numpy() -> Result<(), String> {
    // Regression: the native kernel extracts operands to f64, so without the dtype policy
    // every einsum result was float64. numpy keeps the promoted input dtype — float32 ->
    // float32 (run native, cast result), and integer/bool/float16 -> defer to numpy (exact
    // dtype + integer overflow wraparound). Covers reductions, permutations, trace, GEMM.
    let script = fnp_script(
        r#"
ok = True
rng = np.random.default_rng(0)
for dt in (np.float64, np.float32, np.float16, np.int8, np.int32, np.int64, np.uint8, np.bool_):
    if np.issubdtype(dt, np.floating):
        A = (rng.standard_normal((40, 50)) * 3).astype(dt); B = (rng.standard_normal((50, 30)) * 3).astype(dt)
        S = (rng.standard_normal((8, 8))).astype(dt)
        rt = 1e-3 if dt == np.float16 else (1e-4 if dt == np.float32 else 1e-9)
    elif dt == np.bool_:
        A = rng.integers(0, 2, (40, 50)).astype(dt); B = rng.integers(0, 2, (50, 30)).astype(dt)
        S = rng.integers(0, 2, (8, 8)).astype(dt); rt = 0
    else:
        A = rng.integers(0, 30, (40, 50)).astype(dt); B = rng.integers(0, 30, (50, 30)).astype(dt)
        S = rng.integers(0, 9, (8, 8)).astype(dt); rt = 0
    cases = [("ij->i", (A,)), ("ij->j", (A,)), ("ij->", (A,)), ("ji->ij", (A,)),
             ("ij->ij", (A,)), ("ii", (S,)), ("ij,jk->ik", (A, B))]
    for p, ops in cases:
        f = np.asarray(fnp.einsum(p, *ops)); n = np.asarray(np.einsum(p, *ops))
        if f.dtype != n.dtype or f.shape != n.shape or not np.allclose(f, n, rtol=rt, atol=1e-3, equal_nan=True):
            ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "einsum must preserve operand dtype like numpy"
    );
    Ok(())
}

#[test]
fn f16_einsum_matmul_per_step_contract_bit_exact() -> Result<(), String> {
    // np.einsum's f16 matmul-shaped contraction accumulates each output element
    // as acc = f16(f32(acc) + f32(a_ij)*f32(b_jk)) over j IN ORDER — a PER-STEP
    // narrow to f16 (ledger recon 2026-07-11; it differs from np.matmul's
    // f32-accumulate-once contract by huge margins, so the tiled GEMM must NOT
    // be reused). The native kernel must be byte-identical to the live oracle
    // across shapes including MR-tile tails, and this test LOCKS the contract
    // on the fleet's numpy version: if a future numpy changes its loop, this
    // fails loudly instead of shipping a stale contract.
    let script = fnp_script(
        r#"
ok = True
rng = np.random.default_rng(20260711)
for (m, k, n) in ((128, 96, 144), (512, 512, 512), (2, 400, 400), (514, 40, 40), (65, 130, 257), (3, 7, 4)):
    a = (rng.standard_normal((m, k)) * 0.3).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.3).astype(np.float16)
    r = fnp.einsum('ij,jk->ik', a, b); e = np.einsum('ij,jk->ik', a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape
    ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# inf/nan propagation through the per-step chain
a = (rng.standard_normal((128, 128)) * 0.3).astype(np.float16)
b = (rng.standard_normal((128, 128)) * 0.3).astype(np.float16)
a[0, 0] = np.float16(np.inf); b[0, 1] = np.float16(np.nan); a[3, 5] = np.float16(-np.inf)
r = fnp.einsum('ij,jk->ik', a, b); e = np.einsum('ij,jk->ik', a, b)
ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# alternate letters, same canonical form
r = fnp.einsum('ab,bc->ac', a, b); e = np.einsum('ab,bc->ac', a, b)
ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# below-gate and non-matmul specs stay on the numpy path (trivially byte-equal)
sm_a = (rng.standard_normal((8, 8)) * 0.3).astype(np.float16)
sm_b = (rng.standard_normal((8, 8)) * 0.3).astype(np.float16)
ok = ok and fnp.einsum('ij,jk->ik', sm_a, sm_b).tobytes() == np.einsum('ij,jk->ik', sm_a, sm_b).tobytes()
ok = ok and fnp.einsum('ij,jk->ki', a, b).tobytes() == np.einsum('ij,jk->ki', a, b).tobytes()
ok = ok and np.float16(fnp.einsum('ij,ij->', a, b)).tobytes() == np.float16(np.einsum('ij,ij->', a, b)).tobytes()
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "f16 einsum matmul-shaped contraction must be bit-identical to numpy's per-step contract: {result}"
    );
    Ok(())
}

#[test]
fn f16_einsum_transposed_contract_bit_exact() -> Result<(), String> {
    // np.einsum's f16 TRANSPOSED matmul spec ('ij,lj->il', the a@b.T idiom) is a
    // DIFFERENT contract class from the per-step chain above: with the contracted
    // index last-axis-contiguous on both operands the unbuffered loop dispatches
    // half_sum_of_products_contig_contig_outstride0_two, which for half is ALWAYS
    // the scalar C fallback (NPYV_CHK=0): f32 accum over blocks of 4 with the
    // left-associated tree accum += ((ab0+ab1)+ab2)+ab3, one-at-a-time tail, one
    // final narrow f16(f32(0) + accum). Pinned by source read of
    // einsum_sumprod.c.src (ledger 2026-07-11) after black-box guesses failed at
    // 2-3/400. This test LOCKS the contract on the fleet's numpy version: if a
    // future numpy changes the loop (e.g. gains a half SIMD path), fail loudly.
    let script = fnp_script(
        r#"
ok = True
rng = np.random.default_rng(20260711)
# (m, k, n): a is (m,k), b is (n,k), out (m,n). Covers MR-row tails (m%4),
# k%4 tails 1/2/3, k<4, the k=7 discriminating case, and long-k chains.
for (m, k, n) in ((128, 96, 144), (512, 512, 512), (2, 400, 400), (514, 40, 40), (65, 130, 257), (129, 7, 300), (129, 517, 5), (61, 2, 3000), (77, 2001, 2)):
    a = (rng.standard_normal((m, k)) * 0.3).astype(np.float16)
    b = (rng.standard_normal((n, k)) * 0.3).astype(np.float16)
    r = fnp.einsum('ij,lj->il', a, b); e = np.einsum('ij,lj->il', a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape
    ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# mixed scales stress rounding of the block-tree adds
a = (rng.standard_normal((96, 130)) * rng.choice([0.01, 1.0, 100.0], (96, 1))).astype(np.float16)
b = (rng.standard_normal((80, 130)) * rng.choice([0.01, 1.0, 100.0], (80, 1))).astype(np.float16)
r = fnp.einsum('ij,lj->il', a, b); e = np.einsum('ij,lj->il', a, b)
ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# inf/nan propagation + f16 overflow-to-inf through the wide accumulator
a = (rng.standard_normal((128, 128)) * 0.3).astype(np.float16)
b = (rng.standard_normal((128, 128)) * 0.3).astype(np.float16)
a[0, 0] = np.float16(np.inf); b[1, 0] = np.float16(np.nan); a[3, 5] = np.float16(-np.inf)
a[7, :] = np.float16(60000); b[9, :] = np.float16(60000)
r = fnp.einsum('ij,lj->il', a, b); e = np.einsum('ij,lj->il', a, b)
ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# alternate letters, same canonical form
r = fnp.einsum('ab,cb->ac', a, b); e = np.einsum('ab,cb->ac', a, b)
ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# below-gate stays on the numpy path (trivially byte-equal)
sm_a = (rng.standard_normal((8, 8)) * 0.3).astype(np.float16)
sm_b = (rng.standard_normal((8, 8)) * 0.3).astype(np.float16)
ok = ok and fnp.einsum('ij,lj->il', sm_a, sm_b).tobytes() == np.einsum('ij,lj->il', sm_a, sm_b).tobytes()
# non-contiguous operand defers to numpy (buffered path, not our contract)
at = np.asfortranarray(a)
r = fnp.einsum('ij,lj->il', at, b); e = np.einsum('ij,lj->il', at, b)
ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# adjacent specs stay byte-exact: 'ij,lj->li' is now captured by this kernel's
# operand-swap arm; 'ji,jl->il' by the gram kernel.
ok = ok and fnp.einsum('ij,lj->li', a, b).tobytes() == np.einsum('ij,lj->li', a, b).tobytes()
ok = ok and fnp.einsum('ji,jl->il', a, b).tobytes() == np.einsum('ji,jl->il', a, b).tobytes()
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "f16 einsum transposed contraction must be bit-identical to numpy's blocked-4 wide contract: {result}"
    );
    Ok(())
}

#[test]
fn f16_einsum_gram_contract_bit_exact() -> Result<(), String> {
    // np.einsum's f16 GRAM spec ('ji,jl->il', the a.T@b idiom) is a THIRD
    // contract class: with the contracted index FIRST on both operands the
    // unbuffered loop dispatches half_sum_of_products_stride0_contig_outcontig
    // -> sum_of_products_muladd's scalar half path (NPYV_CHK=0), a
    // PER-STEP-NARROW muladd chain per output element over ascending j:
    //   acc = f16(f32(a[j,i]) * f32(b[j,l]) + f32(acc)), acc0 = f16(+0.0).
    // Pinned by source read + a 112-case local sweep (ledger 2026-07-12).
    // This test LOCKS the contract on the fleet's numpy version.
    let script = fnp_script(
        r#"
ok = True
rng = np.random.default_rng(20260712)
# (k, m, n): a is (k,m), b is (k,n), out (m,n). Covers m%4 MR tails, k=1/2/3
# short chains, the k=7 discriminator, long-k chains, and skinny outputs.
for (k, m, n) in ((96, 128, 144), (512, 512, 512), (400, 2, 400), (40, 514, 40), (130, 65, 257), (7, 129, 300), (517, 129, 5), (2, 61, 3000), (2001, 77, 2), (1, 600, 600), (3, 500, 200)):
    a = (rng.standard_normal((k, m)) * 0.3).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.3).astype(np.float16)
    r = fnp.einsum('ji,jl->il', a, b); e = np.einsum('ji,jl->il', a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape
    ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# mixed scales stress the per-step rounding
a = (rng.standard_normal((130, 96)) * rng.choice([0.01, 1.0, 100.0], (130, 1))).astype(np.float16)
b = (rng.standard_normal((130, 80)) * rng.choice([0.01, 1.0, 100.0], (130, 1))).astype(np.float16)
r = fnp.einsum('ji,jl->il', a, b); e = np.einsum('ji,jl->il', a, b)
ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# inf/nan propagation + mid-chain f16 overflow-to-inf (per-step narrow saturates)
a = (rng.standard_normal((128, 128)) * 0.3).astype(np.float16)
b = (rng.standard_normal((128, 128)) * 0.3).astype(np.float16)
a[0, 0] = np.float16(np.inf); b[1, 0] = np.float16(np.nan); a[3, 5] = np.float16(-np.inf)
a[:, 7] = np.float16(60000); b[:, 9] = np.float16(60000)
r = fnp.einsum('ji,jl->il', a, b); e = np.einsum('ji,jl->il', a, b)
ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# alternate letters, same canonical form
r = fnp.einsum('ab,ac->bc', a, b); e = np.einsum('ab,ac->bc', a, b)
ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# below-gate stays on the numpy path (trivially byte-equal)
sm_a = (rng.standard_normal((8, 8)) * 0.3).astype(np.float16)
sm_b = (rng.standard_normal((8, 8)) * 0.3).astype(np.float16)
ok = ok and fnp.einsum('ji,jl->il', sm_a, sm_b).tobytes() == np.einsum('ji,jl->il', sm_a, sm_b).tobytes()
# non-contiguous operand defers to numpy (buffered path, not our contract)
at = np.asfortranarray(a)
r = fnp.einsum('ji,jl->il', at, b); e = np.einsum('ji,jl->il', at, b)
ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# adjacent specs stay byte-exact: 'ji,jl->li' is now captured by this kernel's
# operand-swap arm; the plain/transposed idioms by their own kernels.
ok = ok and fnp.einsum('ji,jl->li', a, b).tobytes() == np.einsum('ji,jl->li', a, b).tobytes()
ok = ok and fnp.einsum('ij,jl->il', a, b).tobytes() == np.einsum('ij,jl->il', a, b).tobytes()
ok = ok and fnp.einsum('ij,lj->il', a, b).tobytes() == np.einsum('ij,lj->il', a, b).tobytes()
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "f16 einsum gram contraction must be bit-identical to numpy's per-step muladd-row contract: {result}"
    );
    Ok(())
}

#[test]
fn f16_einsum_output_transposed_variants_bit_exact() -> Result<(), String> {
    // The output-transposed forms of both idioms reuse the sibling kernels
    // with operands swapped: 'ij,lj->li' = transposed kernel(b, a) (same
    // blocked-4 tree, product commuted); 'ji,jl->li' = gram kernel(b, a)
    // (same per-step muladd-row chain, product commuted). IEEE f32 multiply
    // is commutative, so both are byte-exact by construction; verified 16/16
    // local (swapped_output_verify.py). Non-square shapes exercise the
    // swapped output dims; tails cover the MR and k%4 edges.
    let script = fnp_script(
        r#"
ok = True
rng = np.random.default_rng(20260713)
for (m, k, n) in ((128, 96, 144), (2, 400, 400), (65, 130, 257), (129, 7, 300), (61, 2, 3000), (77, 2001, 2)):
    a = (rng.standard_normal((m, k)) * 0.3).astype(np.float16)
    b = (rng.standard_normal((n, k)) * 0.3).astype(np.float16)
    r = fnp.einsum('ij,lj->li', a, b); e = np.einsum('ij,lj->li', a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape
    ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
    ag = (rng.standard_normal((k, m)) * 0.3).astype(np.float16)
    bg = (rng.standard_normal((k, n)) * 0.3).astype(np.float16)
    r = fnp.einsum('ji,jl->li', ag, bg); e = np.einsum('ji,jl->li', ag, bg)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape
    ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# inf/nan + overflow through both swapped routes
a = (rng.standard_normal((96, 128)) * 0.3).astype(np.float16)
b = (rng.standard_normal((80, 128)) * 0.3).astype(np.float16)
a[0, 0] = np.float16(np.inf); b[1, 0] = np.float16(np.nan); a[7, :] = np.float16(60000)
r = fnp.einsum('ij,lj->li', a, b); e = np.einsum('ij,lj->li', a, b)
ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
ag = np.ascontiguousarray(a.T); bg = np.ascontiguousarray(b.T)
r = fnp.einsum('ji,jl->li', ag, bg); e = np.einsum('ji,jl->li', ag, bg)
ok = ok and bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all())
# alternate letters + below-gate numpy path
ok = ok and fnp.einsum('ab,cb->ca', a, b).tobytes() == np.einsum('ab,cb->ca', a, b).tobytes()
sm_a = (rng.standard_normal((8, 8)) * 0.3).astype(np.float16)
sm_b = (rng.standard_normal((8, 8)) * 0.3).astype(np.float16)
ok = ok and fnp.einsum('ij,lj->li', sm_a, sm_b).tobytes() == np.einsum('ij,lj->li', sm_a, sm_b).tobytes()
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "f16 einsum output-transposed variants must be bit-identical via the operand-swapped kernels: {result}"
    );
    Ok(())
}

#[test]
fn f16_einsum_batched_matmul_contract_bit_exact() -> Result<(), String> {
    // np.einsum's f16 BATCHED matmul spec ('bij,bjk->bik') runs the plain
    // spec's per-step-narrow chain per batch slice:
    //   acc = f16(f32(acc) + f32(a[b,i,j]) * f32(x[b,j,l])), acc0 = f16(+0.0)
    // over ascending j, with NO buffering chunk at any k (verified locally to
    // k=9000 across 8192 on numpy 2.2.4/2.4.3/2.4.6). This test LOCKS the
    // contract on the fleet's numpy version.
    let script = fnp_script(
        r#"
verdicts = []
def check(name, r, e):
    if r.dtype != e.dtype or r.shape != e.shape:
        verdicts.append(f"FAIL {name} dtype/shape")
    elif not bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all()):
        verdicts.append(f"FAIL {name} bytes")
rng = np.random.default_rng(20260712)
# (B, m, k, n): covers m%4 MR tails, k tails, batch-of-1, skinny dims, long-k.
for (B, m, k, n) in ((8, 64, 96, 80), (2, 512, 128, 96), (16, 17, 33, 29), (1, 128, 512, 64), (3, 2, 8193, 5), (4, 129, 7, 65)):
    a = (rng.standard_normal((B, m, k)) * 0.3).astype(np.float16)
    x = (rng.standard_normal((B, k, n)) * 0.3).astype(np.float16)
    check(f"B={B},m={m},k={k},n={n}", fnp.einsum('bij,bjk->bik', a, x), np.einsum('bij,bjk->bik', a, x))
# mixed scales + inf/nan + mid-chain overflow
a = (rng.standard_normal((4, 96, 130)) * rng.choice([0.01, 1.0, 100.0], (4, 96, 1))).astype(np.float16)
x = (rng.standard_normal((4, 130, 80)) * 0.3).astype(np.float16)
a[0, 0, 0] = np.float16(np.inf); x[1, 5, 3] = np.float16(np.nan); a[2, 7, :] = np.float16(60000); x[2, :, 9] = np.float16(60000)
check("specials", fnp.einsum('bij,bjk->bik', a, x), np.einsum('bij,bjk->bik', a, x))
# alternate letters, same canonical form
check("alt_letters", fnp.einsum('qrs,qst->qrt', a, x), np.einsum('qrs,qst->qrt', a, x))
# below-gate stays on the numpy path
sm_a = (rng.standard_normal((2, 8, 8)) * 0.3).astype(np.float16)
sm_x = (rng.standard_normal((2, 8, 8)) * 0.3).astype(np.float16)
check("below_gate", fnp.einsum('bij,bjk->bik', sm_a, sm_x), np.einsum('bij,bjk->bik', sm_a, sm_x))
# non-contiguous defers; adjacent specs not captured stay byte-exact via numpy
at = np.asfortranarray(a)
check("f_order", fnp.einsum('bij,bjk->bik', at, x), np.einsum('bij,bjk->bik', at, x))
check("bcast_spec", fnp.einsum('bij,jk->bik', a, x[0]), np.einsum('bij,jk->bik', a, x[0]))
check("sum_batch", fnp.einsum('bij,bjk->ik', a, x), np.einsum('bij,bjk->ik', a, x))
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "f16 einsum batched matmul must be bit-identical to numpy's per-batch per-step contract: {result}"
    );
    Ok(())
}

#[test]
fn f16_einsum_batched_transposed_contract_bit_exact() -> Result<(), String> {
    // np.einsum's f16 BATCHED TRANSPOSED spec ('bij,blj->bil') with B>1 and
    // m,n>1 runs the BUFFERED ndim-4 path: every output element folds
    // per-8192-chunk blocked-4 wide f32 trees through an f16 store/reload.
    // B==1 / m==1 / n==1 coalesce to the ndim-3 unbuffered single-tree
    // contract (different bytes) and defer to numpy. Verified on numpy
    // 2.2.4/2.4.3/2.4.6 locally; this test LOCKS both the chunk contract and
    // the coalescing boundary on the fleet's numpy version.
    let script = fnp_script(
        r#"
verdicts = []
def check(name, r, e):
    if r.dtype != e.dtype or r.shape != e.shape:
        verdicts.append(f"FAIL {name} dtype/shape")
    elif not bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all()):
        verdicts.append(f"FAIL {name} bytes")
rng = np.random.default_rng(20260713)
# (B, m, k, n): chunk boundaries (8192/8193/16389), MR tails, k tails.
for (B, m, k, n) in ((8, 64, 96, 80), (2, 256, 128, 96), (2, 3, 8193, 4), (3, 5, 9000, 3), (2, 2, 16389, 2), (16, 17, 33, 29), (4, 129, 7, 65)):
    a = (rng.standard_normal((B, m, k)) * 0.3).astype(np.float16)
    x = (rng.standard_normal((B, n, k)) * 0.3).astype(np.float16)
    check(f"B={B},m={m},k={k},n={n}", fnp.einsum('bij,blj->bil', a, x), np.einsum('bij,blj->bil', a, x))
# coalescing exclusions: B==1 / m==1 / n==1 stay on the numpy path (different contract)
for (B, m, k, n) in ((1, 64, 9000, 48), (3, 1, 9000, 48), (3, 64, 9000, 1)):
    a = (rng.standard_normal((B, m, k)) * 0.3).astype(np.float16)
    x = (rng.standard_normal((B, n, k)) * 0.3).astype(np.float16)
    check(f"coalesce B={B},m={m},n={n}", fnp.einsum('bij,blj->bil', a, x), np.einsum('bij,blj->bil', a, x))
# mixed scales + inf/nan + overflow through the chunk folds
a = (rng.standard_normal((3, 96, 130)) * rng.choice([0.01, 1.0, 100.0], (3, 96, 1))).astype(np.float16)
x = (rng.standard_normal((3, 80, 130)) * 0.3).astype(np.float16)
a[0, 0, 0] = np.float16(np.inf); x[1, 5, 3] = np.float16(np.nan); a[2, 7, :] = np.float16(60000); x[2, 9, :] = np.float16(60000)
check("specials", fnp.einsum('bij,blj->bil', a, x), np.einsum('bij,blj->bil', a, x))
# alternate letters + below-gate
check("alt_letters", fnp.einsum('qrs,qts->qrt', a, x), np.einsum('qrs,qts->qrt', a, x))
sm_a = (rng.standard_normal((2, 8, 8)) * 0.3).astype(np.float16)
sm_x = (rng.standard_normal((2, 8, 8)) * 0.3).astype(np.float16)
check("below_gate", fnp.einsum('bij,blj->bil', sm_a, sm_x), np.einsum('bij,blj->bil', sm_a, sm_x))
# F-order defers; adjacent specs stay byte-exact via their own routes
at = np.asfortranarray(a)
check("f_order", fnp.einsum('bij,blj->bil', at, x), np.einsum('bij,blj->bil', at, x))
check("out_swapped", fnp.einsum('bij,blj->bli', a, x), np.einsum('bij,blj->bli', a, x))
check("sum_batch", fnp.einsum('bij,blj->il', a, x), np.einsum('bij,blj->il', a, x))
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "f16 einsum batched transposed must be bit-identical to numpy's buffered chunk-fold contract: {result}"
    );
    Ok(())
}

#[test]
fn f16_einsum_batched_gram_contract_bit_exact() -> Result<(), String> {
    // np.einsum's f16 BATCHED GRAM spec ('bji,bjl->bil') runs the 2-op gram
    // per-step-narrow muladd-row chain per batch slice:
    //   acc = f16(f32(a[b,j,i]) * f32(x[b,j,l]) + f32(acc)), acc0 = f16(+0.0)
    // over ascending j. The per-step class is CHUNK-IMMUNE: ndim-4 buffering
    // and B==1 coalescing produce identical bytes (verified locally on numpy
    // 2.2.4/2.4.3/2.4.6 incl k=8193/9000, B=1 vs B>1, n=9000 row-chunking).
    // This test LOCKS the contract on the fleet's numpy version.
    let script = fnp_script(
        r#"
verdicts = []
def check(name, r, e):
    if r.dtype != e.dtype or r.shape != e.shape:
        verdicts.append(f"FAIL {name} dtype/shape")
    elif not bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all()):
        verdicts.append(f"FAIL {name} bytes")
rng = np.random.default_rng(20260713)
# (B, k, m, n): MR/k tails, chunk-straddling k, B=1 coalescing, skinny dims.
for (B, k, m, n) in ((8, 96, 64, 80), (2, 128, 256, 96), (2, 8193, 3, 4), (1, 9000, 4, 3), (2, 33, 17, 29), (4, 7, 129, 65), (2, 130, 3, 9000)):
    a = (rng.standard_normal((B, k, m)) * 0.3).astype(np.float16)
    x = (rng.standard_normal((B, k, n)) * 0.3).astype(np.float16)
    check(f"B={B},k={k},m={m},n={n}", fnp.einsum('bji,bjl->bil', a, x), np.einsum('bji,bjl->bil', a, x))
# mixed scales + inf/nan + overflow through the per-step chains
a = (rng.standard_normal((3, 130, 96)) * rng.choice([0.01, 1.0, 100.0], (3, 130, 1))).astype(np.float16)
x = (rng.standard_normal((3, 130, 80)) * 0.3).astype(np.float16)
a[0, 0, 0] = np.float16(np.inf); x[1, 5, 3] = np.float16(np.nan); a[2, :, 7] = np.float16(60000); x[2, :, 9] = np.float16(60000)
check("specials", fnp.einsum('bji,bjl->bil', a, x), np.einsum('bji,bjl->bil', a, x))
# alternate letters + below-gate
check("alt_letters", fnp.einsum('qsr,qst->qrt', a, x), np.einsum('qsr,qst->qrt', a, x))
sm_a = (rng.standard_normal((2, 8, 8)) * 0.3).astype(np.float16)
sm_x = (rng.standard_normal((2, 8, 8)) * 0.3).astype(np.float16)
check("below_gate", fnp.einsum('bji,bjl->bil', sm_a, sm_x), np.einsum('bji,bjl->bil', sm_a, sm_x))
# F-order defers; adjacent specs stay byte-exact via their own routes
at = np.asfortranarray(a)
check("f_order", fnp.einsum('bji,bjl->bil', at, x), np.einsum('bji,bjl->bil', at, x))
check("out_swapped", fnp.einsum('bji,bjl->bli', a, x), np.einsum('bji,bjl->bli', a, x))
check("sum_batch", fnp.einsum('bji,bjl->il', a, x), np.einsum('bji,bjl->il', a, x))
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "f16 einsum batched gram must be bit-identical to numpy's per-batch per-step muladd-row contract: {result}"
    );
    Ok(())
}

#[test]
fn f16_einsum_dot_1d_buffered_contract_bit_exact() -> Result<(), String> {
    // np.einsum's f16 1-D dot ('j,j->') runs contig_contig_outstride0_two once
    // per 8192-element nditer buffer, folding each chunk's blocked-4 f32 tree
    // through an f16 store/reload on the scalar output:
    //   out = f16(f32(out) + tree(chunk)), out0 = f16(+0.0).
    // Pinned by source read + a 22-case local sweep across the buffer
    // boundaries (ledger 2026-07-12). This test LOCKS the chunking contract
    // on the fleet's numpy version - if a future numpy grows the buffer or
    // fuses the fold, fail loudly. k values straddle 8192 multiples; the
    // native gate is 1<<19, so sub-gate ks also exercise the numpy path.
    let script = fnp_script(
        r#"
verdicts = []
def check(name, r, e):
    # House NaN rule (sibling batteries): any-nan == any-nan; payload bits of
    # f32->f16 NaN conversion are unspecified and differ between casts.
    rs, es = np.float16(r), np.float16(e)
    if type(r).__name__ != type(e).__name__:
        verdicts.append(f"FAIL {name} type {type(r).__name__} vs {type(e).__name__}")
    elif not (rs.tobytes() == es.tobytes() or (np.isnan(rs) and np.isnan(es))):
        verdicts.append(f"FAIL {name} bytes {rs!r} vs {es!r}")
rng = np.random.default_rng(20260712)
for k in (524288, 524291, 1048576, 1048573, 2097152, 600000):
    for scale in (0.3, 30.0):
        a = (rng.standard_normal(k) * scale).astype(np.float16)
        b = (rng.standard_normal(k) * scale).astype(np.float16)
        check(f"k={k},x{scale}", fnp.einsum('j,j->', a, b), np.einsum('j,j->', a, b))
# inf/nan propagation + mid-fold overflow through the f16 store/reload
a = (rng.standard_normal(1048576) * 0.3).astype(np.float16)
b = (rng.standard_normal(1048576) * 0.3).astype(np.float16)
a[123] = np.float16(np.inf); b[8192 * 3 + 5] = np.float16(np.nan)
check("inf_nan", fnp.einsum('j,j->', a, b), np.einsum('j,j->', a, b))
af = np.full(1048576, np.float16(60000)); bf = np.full(1048576, np.float16(60000))
check("overflow", fnp.einsum('j,j->', af, bf), np.einsum('j,j->', af, bf))
# alternate letter + below-gate numpy path + adjacent specs not captured
check("alt_letter", fnp.einsum('q,q->', a, b), np.einsum('q,q->', a, b))
sm = (rng.standard_normal(1000) * 0.3).astype(np.float16)
check("below_gate", fnp.einsum('j,j->', sm, sm), np.einsum('j,j->', sm, sm))
if fnp.einsum('j,j->j', a, b).tobytes() != np.einsum('j,j->j', a, b).tobytes():
    verdicts.append("FAIL elementwise j,j->j")
check("implicit_jj", fnp.einsum('j,j', a, b), np.einsum('j,j', a, b))
# non-contiguous defers to numpy
check("strided", fnp.einsum('j,j->', a[::2], b[::2]), np.einsum('j,j->', a[::2], b[::2]))
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "f16 einsum 1-D dot must be bit-identical to numpy's buffered chunk-fold contract: {result}"
    );
    Ok(())
}
