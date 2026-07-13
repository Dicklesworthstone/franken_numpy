//! Conformance tests for numpy.matmul against NumPy oracle.
//!
//! Tests matmul (matrix multiplication).

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

#[test]
fn matmul_2d_matrices() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul 2D matrices should match numpy"
    );
    Ok(())
}

#[test]
fn matmul_vector_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.arange(6).reshape(3, 2)
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul vector-matrix should match numpy"
    );
    Ok(())
}

#[test]
fn matmul_matrix_vector() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.array([1.0, 2.0, 3.0])
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul matrix-vector should match numpy"
    );
    Ok(())
}

#[test]
fn matmul_batch() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
b = np.arange(16).reshape(2, 4, 2)
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul batch should match numpy");
    Ok(())
}

#[test]
fn matmul_operator() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
result = fnp.matmul(a, b)
expected = a @ b
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul should match @ operator");
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
            || stderr.contains("matmul:")
            || stderr.contains("not aligned")
        {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn matmul_dimension_mismatch_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3)
b = fnp.arange(10).reshape(5, 2)
fnp.matmul(a, b)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
b = np.arange(10).reshape(5, 2)
np.matmul(a, b)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "matmul dimension mismatch should raise same error as numpy"
    );
}

#[test]
fn matmul_1d_mismatch_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(3)
b = fnp.arange(5)
fnp.matmul(a, b)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(3)
b = np.arange(5)
np.matmul(a, b)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "matmul 1D vector mismatch should raise same error as numpy"
    );
}

#[test]
fn matmul_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[np.inf, 1.0], [np.nan, 2.0]])
b = np.array([[1.0, 2.0], [3.0, 4.0]])
fnp_result = fnp.matmul(a, b)
np_result = np.matmul(a, b)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul special values should match numpy"
    );
    Ok(())
}

#[test]
fn matmul_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
b = np.array([[5+5j, 6+6j], [7+7j, 8+8j]], dtype=np.complex128)
fnp_result = fnp.matmul(a, b)
np_result = np.matmul(a, b)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul complex should match numpy");
    Ok(())
}

#[test]
fn matmul_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
identity = np.eye(3)
fnp_result = fnp.matmul(a, identity)
np_result = np.matmul(a, identity)
# Matmul with identity should return original (or close to it)
print(np.allclose(fnp_result, a) and np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul identity should match numpy");
    Ok(())
}

#[test]
fn matmul_broadcast_batch() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(12).reshape(3, 2, 2)
b = np.arange(4).reshape(2, 2)  # broadcast this
fnp_result = fnp.matmul(a, b)
np_result = np.matmul(a, b)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul broadcast batch should match numpy"
    );
    Ok(())
}

#[test]
#[ignore = "PARITY GAP: fnp accumulator returns -0.0, NumPy returns 0.0. See DISC-011."]
fn matmul_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Matmul signed-zero parity (accumulation in matrix multiplication)
# Each output element is a dot product
tests = [
    # 1D @ 1D (dot product)
    (np.array([1.0, -0.0]), np.array([-0.0, 1.0])),
    (np.array([-0.0, -0.0]), np.array([1.0, 1.0])),
    # 2D @ 2D with signed zeros
    (np.array([[1.0, -0.0], [-0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])),
]
all_pass = True
for a, b in tests:
    fnp_result = fnp.matmul(a, b)
    np_result = np.matmul(a, b)
    fnp_signs = np.signbit(fnp_result)
    np_signs = np.signbit(np_result)
    if not np.array_equal(fnp_signs, np_signs):
        print(f"FAIL: matmul signbit mismatch")
        print(f"  fnp signbit={fnp_signs.tolist()}")
        print(f"  np signbit={np_signs.tolist()}")
        all_pass = False
    if not np.allclose(fnp_result, np_result):
        print(f"FAIL: matmul values mismatch")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_matmul_native_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy integer matmul uses a naive serial loop (no BLAS); the native parallel ikj
    // GEMM must be byte-identical — incl. overflow WRAP (numpy accumulates in the input
    // dtype with wrapping multiply+add). Sizes straddle the 1<<18 work gate.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(11)
ok = True
# bit-exactness across widths + shapes (square, rectangular, non-square inner)
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint8]:
    # (2, 400, 400): whole output smaller than one MR=4 row block;
    # (514, 40, 40): MR=4 tail of 2 rows; (65, 130, 257): tail of 1 row.
    for (m, k, n) in [(96, 96, 96), (128, 200, 96), (65, 130, 257), (2, 400, 400), (514, 40, 40)]:
        info = np.iinfo(dt)
        a = rng.integers(info.min // 2, info.max // 2, (m, k)).astype(dt)
        b = rng.integers(info.min // 2, info.max // 2, (k, n)).astype(dt)
        r = fnp.matmul(a, b); e = np.matmul(a, b)
        ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
        # @ operator routes through matmul too
        ok = ok and (a @ b).tobytes() == e.tobytes()

# explicit overflow-wrap case (int64): large values so products+sums wrap
a = np.full((80, 80), 5_000_000_000, dtype=np.int64)
b = np.full((80, 80), 5_000_000_000, dtype=np.int64)
ok = ok and fnp.matmul(a, b).tobytes() == np.matmul(a, b).tobytes()

# int8 saturating-wrap case
a = np.full((70, 70), 100, dtype=np.int8)
b = np.full((70, 70), 100, dtype=np.int8)
ok = ok and fnp.matmul(a, b).tobytes() == np.matmul(a, b).tobytes()

print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native integer matmul must be bit-identical to numpy (incl. wrap): {result}"
    );
    Ok(())
}

#[test]
fn bool_matmul_native_bitpacked_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy bool matmul is a scalar early-exit loop; the native bitpacked OR-AND
    // GEMM (64 pairs per word-AND) must be byte-identical: 0/1 output bytes and
    // C `&&` truthiness (any nonzero byte is True - pinned on view-created
    // degenerate bytes), across densities, shapes, and every routed entry point
    // (matmul/@/dot/inner/tensordot/multi_dot). Batched, matvec, and below-gate
    // shapes delegate to numpy and must stay byte-identical too.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(23)
verdicts = []
for dens in [0.0, 0.01, 0.1, 0.5, 1.0]:
    for (m, k, n) in [(96, 96, 96), (128, 200, 96), (65, 130, 257), (2, 400, 400), (514, 40, 40), (100, 513, 77)]:
        a = rng.random((m, k)) < dens
        b = rng.random((k, n)) < dens
        r = fnp.matmul(a, b); e = np.matmul(a, b)
        if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
            verdicts.append(f"FAIL matmul dens={dens} shape=({m},{k},{n})")
        if (a @ b).tobytes() != e.tobytes():
            verdicts.append(f"FAIL @ dens={dens} shape=({m},{k},{n})")
# degenerate non-0/1 bool bytes (view-created): numpy is logical (!=0), 0/1 out
a8 = (rng.integers(0, 4, (90, 90)) * 64).astype(np.uint8)
b8 = (rng.integers(0, 4, (90, 90)) * 64).astype(np.uint8)
if fnp.matmul(a8.view(bool), b8.view(bool)).tobytes() != np.matmul(a8.view(bool), b8.view(bool)).tobytes():
    verdicts.append("FAIL degenerate-byte matmul")
# sibling entry points route the same kernel
a = rng.random((150, 150)) > 0.9
b = rng.random((150, 150)) > 0.9
c = rng.random((150, 150)) > 0.9
if fnp.dot(a, b).tobytes() != np.dot(a, b).tobytes():
    verdicts.append("FAIL dot")
if fnp.inner(a, b).tobytes() != np.inner(a, b).tobytes():
    verdicts.append("FAIL inner square")
ar = rng.random((200, 300)) > 0.5
br = rng.random((150, 300)) > 0.5
if fnp.inner(ar, br).tobytes() != np.inner(ar, br).tobytes():
    verdicts.append("FAIL inner rect")
if fnp.tensordot(a, b, axes=1).tobytes() != np.tensordot(a, b, axes=1).tobytes():
    verdicts.append("FAIL tensordot 2d axes=1")
t3a = rng.random((40, 50, 60)) > 0.8
t3b = rng.random((60, 30, 20)) > 0.8
if fnp.tensordot(t3a, t3b, axes=1).tobytes() != np.tensordot(t3a, t3b, axes=1).tobytes():
    verdicts.append("FAIL tensordot 3d axes=1")
if fnp.multi_dot([a, b, c]).tobytes() != np.linalg.multi_dot([a, b, c]).tobytes():
    verdicts.append("FAIL multi_dot")
# batched (>=3-D matching batch dims) routes the bitpacked kernel too
for dens in [0.0, 0.05, 0.5, 1.0]:
    for shape_a, shape_b in [((8, 96, 96), (8, 96, 96)), ((5, 40, 130), (5, 130, 77)), ((2, 3, 64, 100), (2, 3, 100, 48))]:
        a3 = rng.random(shape_a) < dens
        b3 = rng.random(shape_b) < dens
        r = fnp.matmul(a3, b3); e = np.matmul(a3, b3)
        if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
            verdicts.append(f"FAIL batched dens={dens} shape={shape_a}")
# batched einsum spec + chain route the same batched kernel
e3a = rng.random((8, 128, 128)) > 0.9
e3b = rng.random((8, 128, 128)) > 0.9
if fnp.einsum("abc,acd->abd", e3a, e3b).tobytes() != np.einsum("abc,acd->abd", e3a, e3b).tobytes():
    verdicts.append("FAIL batched einsum")
if fnp.einsum("abc,acd,ade->abe", e3a, e3b, e3a).tobytes() != np.einsum("abc,acd,ade->abe", e3a, e3b, e3a).tobytes():
    verdicts.append("FAIL batched einsum chain")
# broadcast-batch (mismatched batch dims) stays a byte-identical delegate
bba = rng.random((4, 96, 96)) > 0.9
bbb = rng.random((96, 96)) > 0.9
if fnp.matmul(bba, bbb).tobytes() != np.matmul(bba, bbb).tobytes():
    verdicts.append("FAIL broadcast-batch delegate")
v = rng.random(96) > 0.5
if fnp.matmul(a[:96, :96], v).tobytes() != np.matmul(a[:96, :96], v).tobytes():
    verdicts.append("FAIL matvec delegate")
sa = rng.random((10, 10)) > 0.5
if fnp.matmul(sa, sa).tobytes() != np.matmul(sa, sa).tobytes():
    verdicts.append("FAIL below-gate delegate")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

wa = rng.random((1024, 1024)) > 0.9
wb = rng.random((1024, 1024)) > 0.9
tn = best(lambda: np.matmul(wa, wb)); tf = best(lambda: fnp.matmul(wa, wb))
print(f"MATMUL_BOOL_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn = best(lambda: np.dot(wa, wb)); tf = best(lambda: fnp.dot(wa, wb))
print(f"DOT_BOOL_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn = best(lambda: np.inner(wa, wb)); tf = best(lambda: fnp.inner(wa, wb))
print(f"INNER_BOOL_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
w3a = rng.random((16, 256, 256)) > 0.9
w3b = rng.random((16, 256, 256)) > 0.9
tn = best(lambda: np.matmul(w3a, w3b)); tf = best(lambda: fnp.matmul(w3a, w3b))
print(f"MATMUL_BOOL_BATCHED_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn = best(lambda: np.einsum("abc,acd->abd", w3a, w3b))
tf = best(lambda: fnp.einsum("abc,acd->abd", w3a, w3b))
print(f"EINSUM_BOOL_BATCHED_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces MATMUL/DOT/INNER_BOOL_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "native bitpacked bool matmul must be bit-identical to numpy across entry points: {result}"
    );
    Ok(())
}

#[test]
fn int_batched_matmul_native_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // Batched (>=3-D) integer matmul: numpy uses a naive per-slice serial loop. The
    // native parallel batched GEMM must be byte-identical incl. 4-D batch + overflow wrap.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(17)
ok = True
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32]:
    info = np.iinfo(dt)
    # 3-D matching batch
    a = rng.integers(info.min // 2, info.max // 2, (8, 40, 70)).astype(dt)
    b = rng.integers(info.min // 2, info.max // 2, (8, 70, 33)).astype(dt)
    r = fnp.matmul(a, b); e = np.matmul(a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
    ok = ok and (a @ b).tobytes() == e.tobytes()
# 4-D batch
a = rng.integers(-1000, 1000, (3, 4, 32, 48)).astype(np.int64)
b = rng.integers(-1000, 1000, (3, 4, 48, 24)).astype(np.int64)
ok = ok and fnp.matmul(a, b).tobytes() == np.matmul(a, b).tobytes()
# overflow wrap, batched
a = np.full((6, 60, 60), 5_000_000_000, dtype=np.int64)
b = np.full((6, 60, 60), 5_000_000_000, dtype=np.int64)
ok = ok and fnp.matmul(a, b).tobytes() == np.matmul(a, b).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native batched integer matmul must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_multi_dot_native_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy integer multi_dot is a chain of no-BLAS matmuls (slow). The native int GEMM
    // chain is bit-exact (matrix mult over Z/2^w is associative) incl. overflow wrap and
    // varying chain lengths/shapes.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(37)
ok = True
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32]:
    info = np.iinfo(dt)
    # square chains of varying length
    for L in [2, 3, 5]:
        mats = [rng.integers(info.min // 8, info.max // 8, (96, 96)).astype(dt) for _ in range(L)]
        r = fnp.multi_dot(mats); e = np.linalg.multi_dot(mats)
        ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
    # non-square conforming chain (m,k)(k,p)(p,n)
    a = rng.integers(info.min // 8, info.max // 8, (80, 130)).astype(dt)
    b = rng.integers(info.min // 8, info.max // 8, (130, 64)).astype(dt)
    c = rng.integers(info.min // 8, info.max // 8, (64, 100)).astype(dt)
    r = fnp.multi_dot([a, b, c]); e = np.linalg.multi_dot([a, b, c])
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# overflow wrap (int64)
mats = [np.full((90, 90), 5_000_000_000, dtype=np.int64) for _ in range(3)]
ok = ok and fnp.multi_dot(mats).tobytes() == np.linalg.multi_dot(mats).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native integer multi_dot must be bit-identical to numpy: {result}"
    );
    Ok(())
}
