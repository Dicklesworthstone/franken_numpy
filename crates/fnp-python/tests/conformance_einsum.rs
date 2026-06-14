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
