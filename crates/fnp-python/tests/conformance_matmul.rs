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
        last, "True",
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
fn broadcast_batch_int_bool_matmul_bit_exact_matches_numpy() -> Result<(), String> {
    // (B.., m, k) @ (k, n) for int/bool: numpy runs its serial no-BLAS loop per
    // slice; the zero-copy (B*m, k) reshape arm must be byte-identical through
    // matmul, @, and dot (each pinned against its own numpy function). The
    // (m, k) @ (B.., k, n) mirror routes the shared-A batched kernels
    // (matmul-only: np.dot's 2-D @ N-D layout differs and must stay a
    // delegate, pinned below). Below-gate shapes stay byte-identical delegates.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(31)
verdicts = []
for dt in [np.int64, np.int32, np.int8, np.uint64]:
    info = np.iinfo(dt)
    for shape_a, shape_b in [((8, 64, 96), (96, 80)), ((2, 3, 50, 64), (64, 40))]:
        a = rng.integers(info.min // 2, info.max // 2, shape_a).astype(dt)
        b = rng.integers(info.min // 2, info.max // 2, shape_b).astype(dt)
        r = fnp.matmul(a, b); e = np.matmul(a, b)
        if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
            verdicts.append(f"FAIL matmul {dt.__name__} {shape_a}")
        if (a @ b).tobytes() != e.tobytes():
            verdicts.append(f"FAIL @ {dt.__name__} {shape_a}")
        rd = fnp.dot(a, b); ed = np.dot(a, b)
        if rd.dtype != ed.dtype or rd.shape != ed.shape or rd.tobytes() != ed.tobytes():
            verdicts.append(f"FAIL dot {dt.__name__} {shape_a}")
# overflow wrap through the broadcast arm
a = np.full((6, 60, 60), 5_000_000_000, dtype=np.int64)
b = np.full((60, 60), 5_000_000_000, dtype=np.int64)
if fnp.matmul(a, b).tobytes() != np.matmul(a, b).tobytes():
    verdicts.append("FAIL overflow wrap")
for dens in [0.0, 0.05, 0.5, 1.0]:
    for shape_a, shape_b in [((16, 96, 96), (96, 96)), ((5, 40, 130), (130, 77))]:
        a = rng.random(shape_a) < dens
        b = rng.random(shape_b) < dens
        r = fnp.matmul(a, b); e = np.matmul(a, b)
        if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
            verdicts.append(f"FAIL bool matmul dens={dens} {shape_a}")
ab = rng.random((16, 96, 96)) > 0.9
bb = rng.random((96, 96)) > 0.9
if fnp.dot(ab, bb).tobytes() != np.dot(ab, bb).tobytes():
    verdicts.append("FAIL bool dot broadcast")
# mirror direction (m,k)@(B..,k,n): shared-A batched kernels
for dt in [np.int64, np.int32, np.int8, np.uint64]:
    info = np.iinfo(dt)
    for shape_a, shape_b in [((96, 96), (8, 96, 80)), ((50, 64), (2, 3, 64, 40))]:
        ma = rng.integers(info.min // 2, info.max // 2, shape_a).astype(dt)
        mb = rng.integers(info.min // 2, info.max // 2, shape_b).astype(dt)
        r = fnp.matmul(ma, mb); e = np.matmul(ma, mb)
        if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
            verdicts.append(f"FAIL mirror {dt.__name__} {shape_b}")
        if (ma @ mb).tobytes() != e.tobytes():
            verdicts.append(f"FAIL mirror @ {dt.__name__} {shape_b}")
ma = np.full((60, 60), 5_000_000_000, dtype=np.int64)
mb = np.full((6, 60, 60), 5_000_000_000, dtype=np.int64)
if fnp.matmul(ma, mb).tobytes() != np.matmul(ma, mb).tobytes():
    verdicts.append("FAIL mirror overflow wrap")
for dens in [0.0, 0.05, 0.5, 1.0]:
    mba = rng.random((96, 130)) < dens
    mbb = rng.random((16, 130, 77)) < dens
    r = fnp.matmul(mba, mbb); e = np.matmul(mba, mbb)
    if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL bool mirror dens={dens}")
# np.dot(a >=2-D, b >=3-D) contracts to dot's OWN layout ((m, B.., n)): the
# shared-A kernel + transpose-copy arm must match np.dot byte-for-byte
for dt in [np.int64, np.int32, np.int8, np.uint64]:
    info = np.iinfo(dt)
    for shape_a, shape_b in [((96, 96), (8, 96, 80)), ((50, 64), (2, 3, 64, 40)), ((4, 50, 64), (2, 3, 64, 40))]:
        da = rng.integers(info.min // 2, info.max // 2, shape_a).astype(dt)
        db = rng.integers(info.min // 2, info.max // 2, shape_b).astype(dt)
        rd = fnp.dot(da, db); ed = np.dot(da, db)
        if rd.dtype != ed.dtype or rd.shape != ed.shape or rd.tobytes() != ed.tobytes():
            verdicts.append(f"FAIL dot layout {dt.__name__} {shape_a}@{shape_b}")
da = np.full((60, 60), 5_000_000_000, dtype=np.int64)
db = np.full((6, 60, 60), 5_000_000_000, dtype=np.int64)
if fnp.dot(da, db).tobytes() != np.dot(da, db).tobytes():
    verdicts.append("FAIL dot layout overflow wrap")
for dens in [0.0, 0.05, 0.5, 1.0]:
    dba = rng.random((96, 130)) < dens
    dbb = rng.random((16, 130, 77)) < dens
    rd = fnp.dot(dba, dbb); ed = np.dot(dba, dbb)
    if rd.dtype != ed.dtype or rd.shape != ed.shape or rd.tobytes() != ed.tobytes():
        verdicts.append(f"FAIL bool dot layout dens={dens}")
# 1-D a stays a byte-identical delegate
va = rng.integers(-1000, 1000, 96).astype(np.int64)
vb = rng.integers(-1000, 1000, (8, 96, 80)).astype(np.int64)
if fnp.dot(va, vb).tobytes() != np.dot(va, vb).tobytes():
    verdicts.append("FAIL dot 1-D a delegate")
# non-contiguous operands through the broadcast/mirror/dot-layout arms
nc3 = rng.integers(-1000, 1000, (8, 96, 96)).astype(np.int64)
nc2 = rng.integers(-1000, 1000, (96, 96)).astype(np.int64)
if fnp.matmul(nc3.swapaxes(-1, -2), nc2).tobytes() != np.matmul(nc3.swapaxes(-1, -2), nc2).tobytes():
    verdicts.append("FAIL broadcast swapaxes x1")
if fnp.matmul(nc3, nc2.T).tobytes() != np.matmul(nc3, nc2.T).tobytes():
    verdicts.append("FAIL broadcast non-contig x2")
if fnp.matmul(nc2.T, nc3).tobytes() != np.matmul(nc2.T, nc3).tobytes():
    verdicts.append("FAIL mirror non-contig a")
if fnp.matmul(nc2, nc3.swapaxes(-1, -2)).tobytes() != np.matmul(nc2, nc3.swapaxes(-1, -2)).tobytes():
    verdicts.append("FAIL mirror swapaxes b")
if fnp.dot(nc2.T, nc3).tobytes() != np.dot(nc2.T, nc3).tobytes():
    verdicts.append("FAIL dot-layout non-contig a")
ncb = rng.random((8, 96, 96)) > 0.9
ncb2 = rng.random((96, 96)) > 0.9
if fnp.matmul(ncb.swapaxes(-1, -2), ncb2).tobytes() != np.matmul(ncb.swapaxes(-1, -2), ncb2).tobytes():
    verdicts.append("FAIL bool broadcast swapaxes")
# below-gate non-contig broadcast stays a copy-free delegate
sm3 = rng.integers(-5, 5, (2, 12, 12)).astype(np.int64)
sm2 = rng.integers(-5, 5, (12, 12)).astype(np.int64)
if fnp.matmul(sm3.swapaxes(-1, -2), sm2).tobytes() != np.matmul(sm3.swapaxes(-1, -2), sm2).tobytes():
    verdicts.append("FAIL below-gate non-contig broadcast delegate")
sa = rng.integers(-5, 5, (2, 10, 10)).astype(np.int64)
sb = rng.integers(-5, 5, (10, 10)).astype(np.int64)
if fnp.matmul(sa, sb).tobytes() != np.matmul(sa, sb).tobytes():
    verdicts.append("FAIL below-gate delegate")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

wa = rng.integers(-1000, 1000, (16, 256, 256)).astype(np.int64)
wb = rng.integers(-1000, 1000, (256, 256)).astype(np.int64)
tn = best(lambda: np.matmul(wa, wb)); tf = best(lambda: fnp.matmul(wa, wb))
print(f"MATMUL_INT_BROADCAST_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
wba = rng.random((16, 256, 256)) > 0.9
wbb = rng.random((256, 256)) > 0.9
tn = best(lambda: np.matmul(wba, wbb)); tf = best(lambda: fnp.matmul(wba, wbb))
print(f"MATMUL_BOOL_BROADCAST_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn = best(lambda: np.matmul(wb, wa)); tf = best(lambda: fnp.matmul(wb, wa))
print(f"MATMUL_INT_MIRROR_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn = best(lambda: np.matmul(wbb, wba)); tf = best(lambda: fnp.matmul(wbb, wba))
print(f"MATMUL_BOOL_MIRROR_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn = best(lambda: np.dot(wb, wa)); tf = best(lambda: fnp.dot(wb, wa))
print(f"DOT_INT_2D3D_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn = best(lambda: np.matmul(wa.swapaxes(-1, -2), wb))
tf = best(lambda: fnp.matmul(wa.swapaxes(-1, -2), wb))
print(f"MATMUL_INT_BROADCAST_NC_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn = best(lambda: np.dot(wbb, wba)); tf = best(lambda: fnp.dot(wbb, wba))
print(f"DOT_BOOL_2D3D_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces MATMUL_*_BROADCAST_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "broadcast-batch int/bool matmul must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn noncontig_int_bool_matmul_contiguate_bit_exact_matches_numpy() -> Result<(), String> {
    // Non-contiguous int/bool GEMM operands (transposed grams A @ B.T, strided
    // and F-order views, negative strides, batched swapaxes) now take ONE
    // ascontiguousarray copy after the work gate and route the native kernels;
    // numpy's serial loop on a strided int operand measured 2652ms at 1024^2.
    // Values are identical either way, so every row pins bytes against numpy.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(53)
verdicts = []
for dt in [np.int64, np.int32, np.uint8]:
    info = np.iinfo(dt)
    A = rng.integers(info.min // 2, info.max // 2, (200, 200)).astype(dt)
    B = rng.integers(info.min // 2, info.max // 2, (200, 200)).astype(dt)
    W = rng.integers(info.min // 2, info.max // 2, (200, 400)).astype(dt)
    cases = [
        ("A@B.T", A, B.T), ("A.T@B", A.T, B), ("A.T@B.T", A.T, B.T),
        ("strided", W[:, ::2], B), ("neg-stride", A[::-1], B),
        ("F-order", np.asfortranarray(A), B),
    ]
    for name, x, y in cases:
        r = fnp.matmul(x, y); e = np.matmul(x, y)
        if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
            verdicts.append(f"FAIL matmul {name} {dt.__name__}")
        rd = fnp.dot(x, y); ed = np.dot(x, y)
        if rd.tobytes() != ed.tobytes():
            verdicts.append(f"FAIL dot {name} {dt.__name__}")
Ab = rng.random((200, 200)) > 0.9
Bb = rng.random((200, 400)) > 0.9
if fnp.matmul(Ab, Ab.T).tobytes() != np.matmul(Ab, Ab.T).tobytes():
    verdicts.append("FAIL bool A@A.T")
if fnp.matmul(Bb[:, ::2], Ab).tobytes() != np.matmul(Bb[:, ::2], Ab).tobytes():
    verdicts.append("FAIL bool strided")
# batched swapaxes views route the batched dispatcher's contiguation
A3 = rng.integers(-1000, 1000, (8, 96, 96)).astype(np.int64)
B3 = rng.integers(-1000, 1000, (8, 96, 96)).astype(np.int64)
r = fnp.matmul(A3.swapaxes(-1, -2), B3); e = np.matmul(A3.swapaxes(-1, -2), B3)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL batched swapaxes")
b3 = rng.random((8, 96, 96)) > 0.9
r = fnp.matmul(b3, b3.swapaxes(-1, -2)); e = np.matmul(b3, b3.swapaxes(-1, -2))
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL bool batched swapaxes")
# below-gate non-contiguous stays a byte-identical delegate (no copy paid)
S = rng.integers(-5, 5, (12, 12)).astype(np.int64)
if fnp.matmul(S, S.T).tobytes() != np.matmul(S, S.T).tobytes():
    verdicts.append("FAIL below-gate non-contig delegate")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

G = rng.integers(-1000, 1000, (1024, 1024))
tn = best(lambda: np.matmul(G, G.T)); tf = best(lambda: fnp.matmul(G, G.T))
print(f"MATMUL_INT_GRAM_T_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
Ws = rng.integers(-1000, 1000, (1024, 2048))[:, ::2]
tn = best(lambda: np.matmul(Ws, G)); tf = best(lambda: fnp.matmul(Ws, G))
print(f"MATMUL_INT_STRIDED_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
Gb = rng.random((1024, 1024)) > 0.9
tn = best(lambda: np.matmul(Gb, Gb.T)); tf = best(lambda: fnp.matmul(Gb, Gb.T))
print(f"MATMUL_BOOL_GRAM_T_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces MATMUL_*_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "non-contiguous int/bool matmul must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_vecmat_native_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // v @ A (1-D x 2-D) int: numpy's strided column walk measured 70.4ms at
    // 4096^2 vs 10.1ms for its own matvec; the row-major block-accumulation
    // kernel must be byte-identical through matmul, @, and dot. matvec (A @ v),
    // bool, and below-gate stay byte-identical delegates.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(61)
verdicts = []
for dt in [np.int64, np.int32, np.int8, np.uint64]:
    info = np.iinfo(dt)
    for (k, n) in [(512, 512), (300, 1000), (1000, 700)]:
        v = rng.integers(info.min // 2, info.max // 2, k).astype(dt)
        A = rng.integers(info.min // 2, info.max // 2, (k, n)).astype(dt)
        r = fnp.matmul(v, A); e = np.matmul(v, A)
        if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
            verdicts.append(f"FAIL matmul {dt.__name__} ({k},{n})")
        if (v @ A).tobytes() != e.tobytes():
            verdicts.append(f"FAIL @ {dt.__name__} ({k},{n})")
        rd = fnp.dot(v, A); ed = np.dot(v, A)
        if rd.tobytes() != ed.tobytes():
            verdicts.append(f"FAIL dot {dt.__name__} ({k},{n})")
# overflow wrap through the kernel
v = np.full(600, 5_000_000_000, dtype=np.int64)
A = np.full((600, 600), 5_000_000_000, dtype=np.int64)
if fnp.matmul(v, A).tobytes() != np.matmul(v, A).tobytes():
    verdicts.append("FAIL overflow wrap")
# non-contiguous v (strided column view) and A (transposed)
M = rng.integers(-1000, 1000, (600, 600)).astype(np.int64)
vs = M[:, 0]
if fnp.matmul(vs, M).tobytes() != np.matmul(vs, M).tobytes():
    verdicts.append("FAIL strided v")
if fnp.matmul(M[0], M.T).tobytes() != np.matmul(M[0], M.T).tobytes():
    verdicts.append("FAIL non-contig A")
# delegates: matvec, bool vecmat, below-gate
mv = rng.integers(-1000, 1000, 600).astype(np.int64)
if fnp.matmul(M, mv).tobytes() != np.matmul(M, mv).tobytes():
    verdicts.append("FAIL matvec delegate")
vb = rng.random(600) > 0.5
Ab = rng.random((600, 600)) > 0.9
if fnp.matmul(vb, Ab).tobytes() != np.matmul(vb, Ab).tobytes():
    verdicts.append("FAIL bool vecmat delegate")
sv = rng.integers(-5, 5, 100).astype(np.int64)
sA = rng.integers(-5, 5, (100, 100)).astype(np.int64)
if fnp.matmul(sv, sA).tobytes() != np.matmul(sv, sA).tobytes():
    verdicts.append("FAIL below-gate delegate")

def best(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.integers(-1000, 1000, (4096, 4096))
w = rng.integers(-1000, 1000, 4096)
tn = best(lambda: np.matmul(w, W)); tf = best(lambda: fnp.matmul(w, W))
print(f"VECMAT_INT_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces VECMAT_INT_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "native int vecmat must be bit-identical to numpy incl. delegates: {result}"
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
