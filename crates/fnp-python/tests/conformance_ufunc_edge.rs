//! Conformance tests for ufunc edge cases (empty arrays, identity values).
//!
//! Tests verify PyUFunc reduce/accumulate/outer behaviors match NumPy on
//! edge cases like empty arrays and identity element semantics.

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
fn add_reduce_empty_returns_zero() {
    let fnp_result = numpy_oracle(&fnp_script("print(float(fnp.add.reduce([])))".to_string()))
        .expect("fnp add.reduce");
    let np_result =
        numpy_oracle("import numpy as np; print(float(np.add.reduce([])))").expect("np");
    assert_eq!(fnp_result, np_result, "add.reduce([]) should return 0");
    assert_eq!(fnp_result.trim(), "0.0", "identity is 0");
}

#[test]
fn multiply_reduce_empty_returns_one() {
    let fnp_result = numpy_oracle(&fnp_script(
        "print(float(fnp.multiply.reduce([])))".to_string(),
    ))
    .expect("fnp multiply.reduce");
    let np_result =
        numpy_oracle("import numpy as np; print(float(np.multiply.reduce([])))").expect("np");
    assert_eq!(fnp_result, np_result, "multiply.reduce([]) should return 1");
    assert_eq!(fnp_result.trim(), "1.0", "identity is 1");
}

#[test]
fn add_accumulate_empty_returns_empty() {
    let fnp_result = numpy_oracle(&fnp_script(
        "print(len(fnp.add.accumulate([])))".to_string(),
    ))
    .expect("fnp add.accumulate");
    let np_result =
        numpy_oracle("import numpy as np; print(len(np.add.accumulate([])))").expect("np");
    assert_eq!(
        fnp_result, np_result,
        "add.accumulate([]).shape should match"
    );
    assert_eq!(fnp_result.trim(), "0", "empty array");
}

#[test]
fn add_outer_empty_first() {
    let fnp_result = numpy_oracle(&fnp_script(
        "print(fnp.add.outer([], [1,2,3]).shape)".to_string(),
    ))
    .expect("fnp add.outer");
    let np_result =
        numpy_oracle("import numpy as np; print(np.add.outer([], [1,2,3]).shape)").expect("np");
    assert_eq!(
        fnp_result, np_result,
        "add.outer([],arr).shape should match"
    );
}

#[test]
fn add_outer_empty_second() {
    let fnp_result = numpy_oracle(&fnp_script(
        "print(fnp.add.outer([1,2,3], []).shape)".to_string(),
    ))
    .expect("fnp add.outer");
    let np_result =
        numpy_oracle("import numpy as np; print(np.add.outer([1,2,3], []).shape)").expect("np");
    assert_eq!(
        fnp_result, np_result,
        "add.outer(arr,[]).shape should match"
    );
}

#[test]
fn add_reduce_keepdims_empty() {
    let fnp_result = numpy_oracle(&fnp_script(
        "arr = np.array([]).reshape(0,3); print(fnp.add.reduce(arr, axis=0, keepdims=True).shape)"
            .to_string(),
    ))
    .expect("fnp");
    let np_result = numpy_oracle(
        "import numpy as np; arr = np.array([]).reshape(0,3); print(np.add.reduce(arr, axis=0, keepdims=True).shape)",
    )
    .expect("np");
    assert_eq!(fnp_result, np_result, "reduce keepdims on empty");
}

#[test]
fn add_identity_is_zero() {
    let fnp_result = numpy_oracle(&fnp_script("print(fnp.add.identity)".to_string())).expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(np.add.identity)").expect("np");
    assert_eq!(fnp_result, np_result, "add.identity should be 0");
}

#[test]
fn multiply_identity_is_one() {
    let fnp_result =
        numpy_oracle(&fnp_script("print(fnp.multiply.identity)".to_string())).expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(np.multiply.identity)").expect("np");
    assert_eq!(fnp_result, np_result, "multiply.identity should be 1");
}

#[test]
fn maximum_identity_is_none() {
    let fnp_result =
        numpy_oracle(&fnp_script("print(fnp.maximum.identity)".to_string())).expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(np.maximum.identity)").expect("np");
    assert_eq!(fnp_result, np_result, "maximum.identity should be None");
}

#[test]
fn add_nin_is_two() {
    let fnp_result = numpy_oracle(&fnp_script("print(fnp.add.nin)".to_string())).expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(np.add.nin)").expect("np");
    assert_eq!(fnp_result, np_result, "add.nin should be 2");
}

#[test]
fn add_nout_is_one() {
    let fnp_result = numpy_oracle(&fnp_script("print(fnp.add.nout)".to_string())).expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(np.add.nout)").expect("np");
    assert_eq!(fnp_result, np_result, "add.nout should be 1");
}

#[test]
fn add_nargs_is_three() {
    let fnp_result = numpy_oracle(&fnp_script("print(fnp.add.nargs)".to_string())).expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(np.add.nargs)").expect("np");
    assert_eq!(fnp_result, np_result, "add.nargs should be 3");
}

#[test]
fn ufunc_signature_has_x1_x2() {
    let fnp_result = numpy_oracle(&fnp_script(
        "import inspect; sig = inspect.signature(fnp.add); print('x1' in sig.parameters and 'x2' in sig.parameters)".to_string(),
    ))
    .expect("fnp");
    let np_result = numpy_oracle(
        "import numpy as np; import inspect; sig = inspect.signature(np.add); print('x1' in sig.parameters and 'x2' in sig.parameters)",
    )
    .expect("np");
    assert_eq!(fnp_result, np_result, "signature should have x1,x2");
    assert_eq!(fnp_result.trim(), "True");
}

#[test]
fn add_reduce_with_initial() {
    let fnp_result = numpy_oracle(&fnp_script(
        "print(float(fnp.add.reduce([1,2,3], initial=10)))".to_string(),
    ))
    .expect("fnp");
    let np_result =
        numpy_oracle("import numpy as np; print(float(np.add.reduce([1,2,3], initial=10)))")
            .expect("np");
    assert_eq!(fnp_result, np_result, "reduce with initial");
    assert_eq!(fnp_result.trim(), "16.0");
}

#[test]
fn add_reduce_empty_with_initial() {
    let fnp_result = numpy_oracle(&fnp_script(
        "print(float(fnp.add.reduce([], initial=42)))".to_string(),
    ))
    .expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(float(np.add.reduce([], initial=42)))")
        .expect("np");
    assert_eq!(fnp_result, np_result, "reduce empty with initial");
    assert_eq!(fnp_result.trim(), "42.0");
}

#[test]
fn maximum_minimum_accumulate_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // Above the 1<<21 gate the native flat f64 maximum/minimum.accumulate runs the
    // two-pass parallel prefix. max/min are associative and np_fmax/np_fmin replicate
    // numpy's tie rule (return SECOND arg) + NaN propagation, so it must be byte-exact
    // to numpy's serial accumulate — incl. signed-zero ties and a propagating NaN.
    let script = fnp_script(
        r#"
n = (1 << 21) + 65
rng = np.random.default_rng(0)

# random data with signed zeros + a NaN that must propagate forward
x = rng.standard_normal(n)
x[10] = -0.0; x[11] = 0.0; x[12] = -0.0; x[13] = 0.0
x[1234] = np.nan
ok = True
for ufm, npf in [(fnp.maximum, np.maximum), (fnp.minimum, np.minimum)]:
    a = ufm.accumulate(x); e = npf.accumulate(x)
    ok = ok and a.dtype == e.dtype and a.shape == e.shape and a.tobytes() == e.tobytes()

# signed-zero-heavy, no NaN (stresses the tie bit-pattern across block boundaries)
y = np.full(n, -0.0)
y[1::3] = 1.5
y[2::5] = -2.5
y[::7] = 0.0
for ufm, npf in [(fnp.maximum, np.maximum), (fnp.minimum, np.minimum)]:
    a = ufm.accumulate(y); e = npf.accumulate(y)
    ok = ok and a.tobytes() == e.tobytes()

print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large parallel maximum/minimum.accumulate must be bit-identical to numpy"
    );
    Ok(())
}

#[test]
fn accumulate_extremum_f32_int_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // f32 + integer maximum/minimum.accumulate share the generic two-pass prefix.
    // f32: same NaN/signed-zero rules as f64. Integer: no NaN/promotion, output dtype
    // == input dtype. Must be byte-identical to numpy's serial accumulate above the gate.
    let script = fnp_script(
        r#"
n = (1 << 21) + 65
rng = np.random.default_rng(1)
ok = True

# float32 with signed zeros + propagating NaN
xf = rng.standard_normal(n).astype(np.float32)
xf[20] = np.float32(-0.0); xf[21] = np.float32(0.0); xf[22] = np.float32(-0.0)
xf[9999] = np.float32(np.nan)
for ufm, npf in [(fnp.maximum, np.maximum), (fnp.minimum, np.minimum)]:
    a = ufm.accumulate(xf); e = npf.accumulate(xf)
    ok = ok and a.dtype == e.dtype and a.shape == e.shape and a.tobytes() == e.tobytes()

# integer running max/min across several widths (output dtype preserved)
for dt in [np.int64, np.int32, np.int16, np.uint32, np.uint8]:
    xi = (rng.integers(-50, 50, n)).astype(dt)
    for ufm, npf in [(fnp.maximum, np.maximum), (fnp.minimum, np.minimum)]:
        a = ufm.accumulate(xi); e = npf.accumulate(xi)
        ok = ok and a.dtype == e.dtype and a.tobytes() == e.tobytes()

print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large f32/int maximum/minimum.accumulate must be bit-identical to numpy"
    );
    Ok(())
}

#[test]
fn add_multiply_accumulate_int_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // integer/bool add.accumulate routes to the parallel cumsum path and
    // multiply.accumulate to cumprod. Both must stay byte-identical to numpy's own
    // add/multiply.accumulate (same int64/uint64 promotion + overflow wrap) above the
    // 1<<21 gate, AND match np.cumsum / np.cumprod.
    let script = fnp_script(
        r#"
n = (1 << 21) + 65
rng = np.random.default_rng(2)
ok = True

# add.accumulate over several int/bool widths, incl. an overflow-wrap case
for x in [
    rng.integers(-50, 50, n).astype(np.int64),
    rng.integers(-50, 50, n).astype(np.int32),
    rng.integers(0, 7, n).astype(np.uint8),
    (rng.integers(0, 2, n)).astype(np.bool_),
    np.full(n, 9_000_000_000_000_000_000, dtype=np.int64),
]:
    a = fnp.add.accumulate(x); e = np.add.accumulate(x)
    ok = ok and a.dtype == e.dtype and a.shape == e.shape and a.tobytes() == e.tobytes()
    ok = ok and a.tobytes() == np.cumsum(x).tobytes()

# multiply.accumulate (wrapping product)
for x in [
    np.where(np.arange(n) % 11 == 0, np.int64(-3), np.int64(2)),
    rng.integers(0, 3, n).astype(np.uint32),
]:
    a = fnp.multiply.accumulate(x); e = np.multiply.accumulate(x)
    ok = ok and a.dtype == e.dtype and a.tobytes() == e.tobytes()
    ok = ok and a.tobytes() == np.cumprod(x).tobytes()

print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large int add/multiply.accumulate must be bit-identical to numpy + cumsum/cumprod"
    );
    Ok(())
}

#[test]
fn bitwise_accumulate_int_bool_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // bitwise_and/or/xor.accumulate (int/uint/bool) route to the two-pass parallel
    // prefix (associative, no promotion). Must be byte-identical to numpy above the gate.
    let script = fnp_script(
        r#"
n = (1 << 21) + 65
rng = np.random.default_rng(3)
ok = True
for fnp_uf, np_uf in [
    (fnp.bitwise_or, np.bitwise_or),
    (fnp.bitwise_and, np.bitwise_and),
    (fnp.bitwise_xor, np.bitwise_xor),
]:
    for dt in [np.int64, np.int32, np.uint8, np.uint64]:
        x = rng.integers(0, 1 << 20, n).astype(dt)
        a = fnp_uf.accumulate(x); e = np_uf.accumulate(x)
        ok = ok and a.dtype == e.dtype and a.shape == e.shape and a.tobytes() == e.tobytes()
    # bool inputs (running any/all/parity)
    xb = (rng.integers(0, 2, n)).astype(np.bool_)
    a = fnp_uf.accumulate(xb); e = np_uf.accumulate(xb)
    ok = ok and a.dtype == e.dtype and a.tobytes() == e.tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large bitwise_*.accumulate must be bit-identical to numpy",
    );
    Ok(())
}

#[test]
fn f16_binary_add_mul_sub_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy has no native f16 ALU (widen->op->narrow); the native parallel widen->op->
    // narrow must be byte-identical incl. inf/nan/-0.0/overflow, above the 1<<20 gate.
    let script = fnp_script(
        r#"
n = (1 << 20) + 257
rng = np.random.default_rng(41)
a = rng.standard_normal(n).astype(np.float16)
b = (rng.standard_normal(n) + 1.5).astype(np.float16)
# seed special values
a[0] = np.float16(np.inf); a[1] = np.float16(-np.inf); a[2] = np.float16(np.nan)
a[3] = np.float16(-0.0);  a[4] = np.float16(65504.0); b[4] = np.float16(2.0)  # overflow -> inf
b[5] = np.float16(0.0)
ok = True
for fnp_op, np_op in [(fnp.add, np.add), (fnp.multiply, np.multiply), (fnp.subtract, np.subtract)]:
    r = fnp_op(a, b); e = np_op(a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# operator forms (a + b etc.) route through the same ufuncs
ok = ok and (a + b).tobytes() == np.add(a, b).tobytes()
ok = ok and (a * b).tobytes() == np.multiply(a, b).tobytes()
ok = ok and (a - b).tobytes() == np.subtract(a, b).tobytes()
# 2-D same-shape
a2 = a[:1 << 20].reshape(1024, 1024); b2 = b[:1 << 20].reshape(1024, 1024)
ok = ok and fnp.add(a2, b2).tobytes() == np.add(a2, b2).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 add/multiply/subtract must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn f32_fmod_copysign_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy runs f32 binary ufuncs single-threaded; there was no f32 binary zero-copy path
    // (only f64). The native parallel f32 kernel for fmod (lhs % rhs = IEEE fmodf) and copysign
    // (sign-bit copy) must be byte-identical to numpy above the 1<<21 gate, incl. inf/nan/-0.0.
    let script = fnp_script(
        r#"
n = (1 << 21) + 257
rng = np.random.default_rng(7)
a = (rng.standard_normal(n) * 1e3).astype(np.float32)
# fmod path defers on any zero divisor, so keep divisors strictly non-zero to exercise the kernel
b = (rng.standard_normal(n) * 7.0).astype(np.float32)
b[np.abs(b) < 1e-3] = np.float32(1.5)
# seed special values (no zero divisor)
a[0]=np.float32(np.inf); a[1]=np.float32(-np.inf); a[2]=np.float32(np.nan); a[3]=np.float32(-0.0)
b[2]=np.float32(3.0); b[3]=np.float32(-2.0)
ok = True
r = fnp.fmod(a, b); e = np.fmod(a, b)
ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# remainder (floored-mod, sign of divisor) — bit-identical to numpy in f32
r = fnp.remainder(a, b); e = np.remainder(a, b)
ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# copysign over the same arrays (every f32 input, incl -0.0/nan/inf signs)
r = fnp.copysign(a, b); e = np.copysign(a, b)
ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# nextafter (bit-step toward the other operand) — bit-identical in f32
r = fnp.nextafter(a, b); e = np.nextafter(a, b)
ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# 2-D shape preserved
a2 = a[:1 << 21].reshape(2048, 1024); b2 = b[:1 << 21].reshape(2048, 1024)
ok = ok and fnp.fmod(a2, b2).tobytes() == np.fmod(a2, b2).tobytes()
ok = ok and fnp.fmod(a2, b2).shape == np.fmod(a2, b2).shape
# zero-divisor case defers to numpy -> still byte-identical
bz = b.copy(); bz[5] = np.float32(0.0)
ok = ok and fnp.fmod(a, bz).tobytes() == np.fmod(a, bz).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f32 fmod/copysign must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_divmod_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy runs integer divmod single-threaded; the native parallel kernel produces both the
    // floored quotient and floored remainder in one pass and must be byte-identical for every
    // width incl mixed signs / INT_MIN, above the gate. Zero divisor must defer to numpy.
    let script = fnp_script(
        r#"
import warnings
n = (1 << 18) + 257
rng = np.random.default_rng(19)
ok = True
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8]:
    info = np.iinfo(dt)
    a = rng.integers(info.min, info.max, n, dtype=dt)
    b = rng.integers(info.min, info.max, n, dtype=dt)
    b[b == 0] = dt(1)
    a[0]=dt(7); b[0]=dt(-3) if info.min < 0 else dt(3)
    a[1]=info.min; b[1]=dt(-1) if info.min < 0 else dt(1)
    a[2]=info.min; b[2]=info.max
    q, r = fnp.divmod(a, b); eq, er = np.divmod(a, b)
    ok = ok and q.dtype == eq.dtype and q.tobytes() == eq.tobytes()
    ok = ok and r.dtype == er.dtype and r.tobytes() == er.tobytes()
    # identity a == q*b + r (in the wrapping ring)
    ok = ok and ((q * b + r).astype(dt).tobytes() == a.tobytes())
# zero divisor defers to numpy
az = rng.integers(1, 1000, n, dtype=np.int64); bz = rng.integers(1, 7, n, dtype=np.int64); bz[5] = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    q, r = fnp.divmod(az, bz); eq, er = np.divmod(az, bz)
    ok = ok and q.tobytes() == eq.tobytes() and r.tobytes() == er.tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native int divmod must be bit-identical to numpy incl mixed signs / INT_MIN: {result}"
    );
    Ok(())
}

#[test]
fn int_remainder_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy runs integer a%b as a single-threaded element loop; the native parallel floored-
    // remainder kernel (sign of divisor) must be byte-identical for every width incl mixed
    // signs, above the gate. A zero divisor must still defer to numpy (0 + RuntimeWarning).
    let script = fnp_script(
        r#"
import warnings
n = (1 << 18) + 257
rng = np.random.default_rng(17)
ok = True
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8]:
    info = np.iinfo(dt)
    a = rng.integers(info.min, info.max, n, dtype=dt)
    b = rng.integers(info.min, info.max, n, dtype=dt)
    b[b == 0] = dt(1)
    a[0]=dt(7);  b[0]=dt(-3) if info.min < 0 else dt(3)   # mixed-sign floored remainder
    a[1]=dt(-7) if info.min < 0 else dt(7); b[1]=dt(3)
    a[2]=info.min; b[2]=info.max
    for opname in ("remainder", "mod"):
        r = getattr(fnp, opname)(a, b); e = getattr(np, opname)(a, b)
        ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# % operator routes through the same ufunc
a64 = rng.integers(-10**9, 10**9, n, dtype=np.int64); b64 = rng.integers(1, 1000, n, dtype=np.int64)
ok = ok and (a64 % b64).tobytes() == np.remainder(a64, b64).tobytes()
# zero divisor defers to numpy (0 + warning)
bz = b64.copy(); bz[3] = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ok = ok and fnp.remainder(a64, bz).tobytes() == np.remainder(a64, bz).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native int remainder/mod must be bit-identical to numpy incl mixed signs: {result}"
    );
    Ok(())
}

#[test]
fn int_floordiv_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy runs integer a//b as a single-threaded element loop; the native parallel floored-
    // division kernel must be byte-identical for every width incl mixed signs and INT_MIN//-1,
    // above the gate. A zero divisor must still defer to numpy (0 + RuntimeWarning).
    let script = fnp_script(
        r#"
import warnings
n = (1 << 18) + 257
rng = np.random.default_rng(15)
ok = True
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8]:
    info = np.iinfo(dt)
    a = rng.integers(info.min, info.max, n, dtype=dt)
    b = rng.integers(info.min, info.max, n, dtype=dt)
    b[b == 0] = dt(1)               # non-zero divisors to exercise the kernel
    a[0]=info.min; b[0]=dt(-1) if info.min < 0 else dt(1)   # INT_MIN // -1 wrap
    a[1]=dt(7);  b[1]=dt(-3) if info.min < 0 else dt(3)     # mixed sign floor
    a[2]=info.min; b[2]=info.max
    r = fnp.floor_divide(a, b); e = np.floor_divide(a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# zero divisor defers to numpy -> still byte-identical (0 + warning, suppressed)
az = rng.integers(1, 1000, n, dtype=np.int64); bz = rng.integers(1, 7, n, dtype=np.int64)
bz[9] = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ok = ok and fnp.floor_divide(az, bz).tobytes() == np.floor_divide(az, bz).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native int floor_divide must be bit-identical to numpy incl INT_MIN//-1 and mixed signs: {result}"
    );
    Ok(())
}

#[test]
fn int_power_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy runs integer a**b as a single-threaded element loop; the native parallel wrapping
    // repeated-squaring kernel must be byte-identical for every width (overflow wraps mod 2^w,
    // 0**0==1, negative base) above the gate. Negative exponents must still defer to numpy's
    // ValueError (tested separately).
    let script = fnp_script(
        r#"
n = (1 << 18) + 257
rng = np.random.default_rng(13)
ok = True
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8]:
    info = np.iinfo(dt)
    a = rng.integers(info.min, info.max, n, dtype=dt)
    b = rng.integers(0, 12, n, dtype=dt)  # non-negative exponents
    a[0]=dt(0); b[0]=dt(0)      # 0**0 == 1
    a[1]=dt(0); b[1]=dt(5)      # 0**5 == 0
    a[2]=info.max; b[2]=dt(3)   # overflow wrap
    if info.min < 0:
        a[3]=dt(-2); b[3]=dt(7) # negative base
        a[4]=info.min; b[4]=dt(2)
    r = fnp.power(a, b); e = np.power(a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# negative exponent must defer to numpy and raise ValueError (same as numpy)
aa = rng.integers(1, 5, n, dtype=np.int64); bb = rng.integers(0, 4, n, dtype=np.int64)
bb[7] = -1
raised = False
try:
    fnp.power(aa, bb)
except ValueError:
    raised = True
ok = ok and raised
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native int power must be bit-identical to numpy (and defer negative exponents): {result}"
    );
    Ok(())
}

#[test]
fn int_gcd_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy np.gcd is a single-threaded element loop; the native parallel Euclid kernel must be
    // byte-identical for every integer width, incl signed INT_MIN (|INT_MIN| wraps in two's
    // complement) and zeros, above the 1<<18 gate.
    let script = fnp_script(
        r#"
n = (1 << 18) + 257
rng = np.random.default_rng(9)
ok = True
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8]:
    info = np.iinfo(dt)
    a = rng.integers(info.min, info.max, n, dtype=dt)
    b = rng.integers(info.min, info.max, n, dtype=dt)
    # seed edges: INT_MIN, INT_MAX, 0, +-1, equal pairs
    a[0]=info.min; b[0]=info.min
    a[1]=info.min; b[1]=dt(0)
    a[2]=dt(0);    b[2]=dt(0)
    a[3]=info.max; b[3]=info.min
    a[4]=dt(0);    b[4]=info.max
    r = fnp.gcd(a, b); e = np.gcd(a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
    # lcm: (|a|/gcd)*|b| with dtype wrap (incl overflow + INT_MIN), bit-identical to numpy
    r = fnp.lcm(a, b); e = np.lcm(a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# 2-D shape preserved
a2 = rng.integers(1, 10**9, (600, 600)).astype(np.int64)
b2 = rng.integers(1, 10**9, (600, 600)).astype(np.int64)
ok = ok and fnp.gcd(a2, b2).tobytes() == np.gcd(a2, b2).tobytes()
ok = ok and fnp.gcd(a2, b2).shape == np.gcd(a2, b2).shape
ok = ok and fnp.lcm(a2, b2).tobytes() == np.lcm(a2, b2).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native int gcd must be bit-identical to numpy across all widths incl INT_MIN: {result}"
    );
    Ok(())
}

#[test]
fn f16_round_decimals0_full_domain_bit_exact_matches_numpy() -> Result<(), String> {
    // np.round(f16, decimals=0) == round-half-even (rint); numpy widens f16->f32. The native
    // parallel widen-rint kernel must be byte-identical over the FULL f16 domain. round and
    // around share the dispatcher.
    let script = fnp_script(
        r#"
patterns = np.arange(65536, dtype=np.uint16).view(np.float16)
x = np.tile(patterns, ((1 << 20) // patterns.size) + 2)
ok = True
for fnp_op, np_op in [(fnp.round, np.round), (fnp.around, np.around)]:
    r = fnp_op(x); e = np_op(x)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# 2-D shape preserved
x2 = np.tile(patterns, (1 << 20) // patterns.size).reshape(-1, patterns.size)
ok = ok and fnp.round(x2).tobytes() == np.round(x2).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 round(decimals=0) must be bit-identical to numpy over the full domain: {result}"
    );
    Ok(())
}

#[test]
fn f32_spacing_full_domain_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy f32 spacing is single-threaded; the native direct f32 bit formula (ULP at f32
    // precision) must be byte-identical over the FULL f32 domain (sampled), incl 0/-0/inf/nan/
    // subnormal, tiled past the gate.
    let script = fnp_script(
        r#"
patterns = np.arange(0, 2**32, 1 << 11, dtype=np.uint32).view(np.float32)   # ~2M f32 samples
ok = True
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = fnp.spacing(patterns); e = np.spacing(patterns)
ok = ok and r.dtype == e.dtype and r.shape == e.shape
ok = ok and ((r.view(np.uint32) == e.view(np.uint32)) | (np.isnan(r) & np.isnan(e))).all()
# explicit specials + 2-D
sp = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 1e38, 1e-40], dtype=np.float32)
sp = np.tile(sp, ((1 << 18) // sp.size) + 2)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r2 = fnp.spacing(sp); e2 = np.spacing(sp)
ok = ok and bool(((r2.view(np.uint32) == e2.view(np.uint32)) | (np.isnan(r2) & np.isnan(e2))).all())
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f32 spacing must be bit-identical to numpy over the full domain: {result}"
    );
    Ok(())
}

#[test]
fn f32_ldexp_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy f32 ldexp (scalbnf) is single-threaded; the native widen->exact-pow2-scale->narrow is
    // a single rounding (== scalbnf), bit-identical across the exponent range incl 0/-0/inf/nan
    // mantissas and subnormal/overflow exponents, above the gate.
    let script = fnp_script(
        r#"
n = (1 << 20) + 257
rng = np.random.default_rng(79)
x = rng.standard_normal(n).astype(np.float32)
e = rng.integers(-300, 300, n).astype(np.int32)
x[0]=np.float32(0.0); x[1]=np.float32(-0.0); x[2]=np.float32(np.inf); x[3]=np.float32(-np.inf); x[4]=np.float32(np.nan)
x[5]=np.float32(1e38); e[5]=np.int32(40)        # overflow -> inf
x[6]=np.float32(1e-30); e[6]=np.int32(-40)      # underflow -> 0/subnormal
e[0]=np.int32(100); e[2]=np.int32(7)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = fnp.ldexp(x, e); ee = np.ldexp(x, e)
ok = r.dtype == ee.dtype and r.shape == ee.shape
ok = ok and ((r.view(np.uint32) == ee.view(np.uint32)) | (np.isnan(r) & np.isnan(ee))).all()
# 2-D shape preserved
x2 = x[:1 << 20].reshape(1024, 1024); e2 = e[:1 << 20].reshape(1024, 1024)
r2 = fnp.ldexp(x2, e2); ee2 = np.ldexp(x2, e2)
ok = ok and ((r2.view(np.uint32) == ee2.view(np.uint32)) | (np.isnan(r2) & np.isnan(ee2))).all()
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f32 ldexp must be bit-identical to numpy across the exponent range: {result}"
    );
    Ok(())
}

#[test]
fn f32_polyval_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy polyval (Horner) is single-threaded; for f32 coeffs + f32 x the result is f32 with an
    // in-f32 Horner. The native parallel per-element f32 Horner must be byte-identical, above the
    // gate, for several degrees + incl inf/nan/-0.0 x values.
    let script = fnp_script(
        r#"
n = (1 << 18) + 257
rng = np.random.default_rng(73)
ok = True
for deg in (1, 4, 11):
    p = rng.standard_normal(deg).astype(np.float32)
    x = (rng.standard_normal(n) * 3.0).astype(np.float32)
    x[0] = np.float32(np.inf); x[1] = np.float32(-np.inf); x[2] = np.float32(np.nan); x[3] = np.float32(-0.0)
    r = fnp.polyval(p, x); e = np.polyval(p, x)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# 2-D x shape preserved
x2 = (rng.standard_normal((512, 512)) * 2.0).astype(np.float32)
p = rng.standard_normal(6).astype(np.float32)
ok = ok and fnp.polyval(p, x2).tobytes() == np.polyval(p, x2).tobytes()
ok = ok and fnp.polyval(p, x2).shape == np.polyval(p, x2).shape
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f32 polyval must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn complex_multiply_divide_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy runs complex multiply / divide single-threaded. The native parallel kernels (FMA
    // multiply for complex128, Smith divide for both dtypes) must be BYTE-identical to numpy above
    // the gate, incl the full inf/nan/-0.0 specials grid for multiply. Divide defers to numpy on a
    // zero complex divisor (numpy's div-by-zero recovery differs), so the random divisor is kept
    // non-zero. (complex64 multiply + complex square delegate to numpy — byte-identical trivially.)
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
ok = True
for cdt, fdt in (("complex128", "float64"), ("complex64", "float32")):
    n = (1 << 20) + 257
    rng = np.random.default_rng(91)
    a = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(cdt)
    # divisor magnitude bounded away from zero so divide takes the native Smith path
    b = ((rng.standard_normal(n) + 2.5) + 1j * (rng.standard_normal(n) + 2.5)).astype(cdt)
    for r, e in ((fnp.multiply(a, b), np.multiply(a, b)),
                 (fnp.divide(a, b), np.divide(a, b)),
                 (fnp.square(a), np.square(a))):
        ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
    # 2-D shape preserved (native multiply for c128, delegated for c64; native divide both)
    a2 = a[:1 << 20].reshape(1024, 1024); b2 = b[:1 << 20].reshape(1024, 1024)
    ok = ok and fnp.multiply(a2, b2).tobytes() == np.multiply(a2, b2).tobytes()
    ok = ok and fnp.multiply(a2, b2).shape == np.multiply(a2, b2).shape
    ok = ok and fnp.divide(a2, b2).tobytes() == np.divide(a2, b2).tobytes()
    # full inf/nan/-0.0 specials grid for multiply, tiled past the 1<<20 multiply gate
    sp = np.array([0.0, -0.0, 1.0, -2.5, np.inf, -np.inf, np.nan], dtype=fdt)
    gr = np.array([complex(x, y) for x in sp for y in sp], dtype=cdt)
    A = np.tile(gr, ((1 << 20) // gr.size) + 64)
    B = np.tile(gr[::-1], ((1 << 20) // gr.size) + 64)
    rm = fnp.multiply(A, B); em = np.multiply(A, B)
    ok = ok and ((rm.view(fdt) == em.view(fdt)) | (np.isnan(rm.view(fdt)) & np.isnan(em.view(fdt)))).all()
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native complex multiply/divide must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn f16_divide_widen_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy has no f16 ALU: it widens f16->f32, divides, narrows (round-to-nearest-even). The native
    // parallel widen-divide-narrow must be BYTE-identical above the gate, incl inf/nan/-0.0 numerators
    // and the full f16 domain divided by a fixed divisor set. A zero divisor defers to numpy.
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
n = (1 << 20) + 257
rng = np.random.default_rng(97)
a = (rng.standard_normal(n) * 4).astype(np.float16)
b = (rng.standard_normal(n) * 4).astype(np.float16)
b[b == 0] = np.float16(1.0)                 # non-zero divisor -> native path
a[0]=np.float16(np.inf); a[1]=np.float16(-np.inf); a[2]=np.float16(np.nan); a[3]=np.float16(-0.0)
r = fnp.divide(a, b); e = np.divide(a, b)
ok = r.dtype == e.dtype and r.shape == e.shape
ok = ok and ((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all()
# 2-D shape preserved
a2 = a[:1 << 20].reshape(1024, 1024); b2 = b[:1 << 20].reshape(1024, 1024)
r2 = fnp.divide(a2, b2); e2 = np.divide(a2, b2)
ok = ok and ((r2.view(np.uint16) == e2.view(np.uint16)) | (np.isnan(r2) & np.isnan(e2))).all()
# full f16 domain (tiled past the gate) divided by a fixed divisor set
allf = np.arange(0, 65536, dtype=np.uint16).view(np.float16)
A = np.tile(allf, ((1 << 20) // allf.size) + 2)
for d in (np.float16(1.0), np.float16(-3.5), np.float16(7.0), np.float16(0.5)):
    B = np.full(A.size, d, dtype=np.float16)
    rr = fnp.divide(A, B); ee = np.divide(A, B)
    ok = ok and bool(((rr.view(np.uint16) == ee.view(np.uint16)) | (np.isnan(rr) & np.isnan(ee))).all())
# zero divisor defers to numpy and still matches (inf/nan + RuntimeWarning suppressed)
bz = b.copy(); bz[5] = np.float16(0.0); bz[6] = np.float16(-0.0)
rz = fnp.divide(a, bz); ez = np.divide(a, bz)
ok = ok and bool(((rz.view(np.uint16) == ez.view(np.uint16)) | (np.isnan(rz) & np.isnan(ez))).all())
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 divide must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn f16_floor_divide_widen_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy's f16 floor_divide widens f16->f32, runs the npy_divmod float floor_divide (fmod-
    // corrected, NOT floor(a/b)), narrows. The native parallel divmod replication (div=(a-fmod)/b,
    // floor-sign adjust, round-to-nearest-int, signed-zero from a/b) must be BYTE-identical above
    // the gate over the full f16 domain divided by an f16 divisor set incl inf/nan numerators. A
    // zero divisor defers to numpy.
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
n = (1 << 20) + 257
rng = np.random.default_rng(101)
a = (rng.standard_normal(n) * 6).astype(np.float16)
b = (rng.standard_normal(n) * 6).astype(np.float16)
b[b == 0] = np.float16(1.0)
a[0]=np.float16(np.inf); a[1]=np.float16(-np.inf); a[2]=np.float16(np.nan); a[3]=np.float16(-0.0)
r = fnp.floor_divide(a, b); e = np.floor_divide(a, b)
ok = r.dtype == e.dtype and r.shape == e.shape
ok = ok and ((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all()
# 2-D shape preserved
a2 = a[:1 << 20].reshape(1024, 1024); b2 = b[:1 << 20].reshape(1024, 1024)
r2 = fnp.floor_divide(a2, b2); e2 = np.floor_divide(a2, b2)
ok = ok and ((r2.view(np.uint16) == e2.view(np.uint16)) | (np.isnan(r2) & np.isnan(e2))).all()
# full f16 domain (tiled past the gate) // f16 divisor set (signed-zero + divmod-correction cases)
allf = np.arange(0, 65536, dtype=np.uint16).view(np.float16)
A = np.tile(allf, ((1 << 20) // allf.size) + 2)
for d in (np.float16(0.1), np.float16(-7.5), np.float16(3.0), np.float16(0.3),
          np.float16(np.inf), np.float16(-0.001), np.float16(1.0)):
    B = np.full(A.size, d, dtype=np.float16)
    rr = fnp.floor_divide(A, B); ee = np.floor_divide(A, B)
    ok = ok and bool(((rr.view(np.uint16) == ee.view(np.uint16)) | (np.isnan(rr) & np.isnan(ee))).all())
# zero divisor defers to numpy and still matches
bz = b.copy(); bz[5] = np.float16(0.0); bz[6] = np.float16(-0.0)
rz = fnp.floor_divide(a, bz); ez = np.floor_divide(a, bz)
ok = ok and bool(((rz.view(np.uint16) == ez.view(np.uint16)) | (np.isnan(rz) & np.isnan(ez))).all())
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 floor_divide must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn f16_matmul_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy has no f16 BLAS: a @ b widens to f32, accumulates each output element over k=0..K-1 in
    // order, narrows to f16 once. The native parallel GEMM (per-output f32 accumulation in the same
    // k-order, single narrow) must be BYTE-identical to np.matmul / np.dot across square, rectangular,
    // matvec and large-K shapes, above the gate.
    let script = fnp_script(
        r#"
ok = True
for (M, K, N, seed) in ((512,512,512,1),(300,700,200,2),(128,2000,64,3),(2000,3,2000,4),(64,4096,1,5)):
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((M, K)) * 0.3).astype(np.float16)
    b = (rng.standard_normal((K, N)) * 0.3).astype(np.float16)
    rm = fnp.matmul(a, b); em = np.matmul(a, b)
    ok = ok and rm.dtype == em.dtype and rm.shape == em.shape
    ok = ok and ((rm.view(np.uint16) == em.view(np.uint16)) | (np.isnan(rm) & np.isnan(em))).all()
    rd = fnp.dot(a, b); ed = np.dot(a, b)
    ok = ok and ((rd.view(np.uint16) == ed.view(np.uint16)) | (np.isnan(rd) & np.isnan(ed))).all()
# inf/nan elements propagate identically
rng = np.random.default_rng(6)
a = (rng.standard_normal((256, 256)) * 0.3).astype(np.float16); b = (rng.standard_normal((256, 256)) * 0.3).astype(np.float16)
a[0, 0] = np.float16(np.inf); b[0, 1] = np.float16(np.nan)
rm = fnp.matmul(a, b); em = np.matmul(a, b)
ok = ok and bool(((rm.view(np.uint16) == em.view(np.uint16)) | (np.isnan(rm) & np.isnan(em))).all())
# BATCHED (>=3-D) f16 matmul, matching batch dims
for shp in ((8, 128, 128), (4, 3, 64, 64)):
    rng = np.random.default_rng(sum(shp))
    *bd, M, K = shp
    a = (rng.standard_normal(shp) * 0.3).astype(np.float16)
    b = (rng.standard_normal((*bd, K, M)) * 0.3).astype(np.float16)
    rb = fnp.matmul(a, b); eb = np.matmul(a, b)
    ok = ok and rb.dtype == eb.dtype and rb.shape == eb.shape
    ok = ok and bool(((rb.view(np.uint16) == eb.view(np.uint16)) | (np.isnan(rb) & np.isnan(eb))).all())
# BROADCAST batched: (B,m,k)@(k,n) [b shared] and (m,k)@(B,k,n) [a shared]
rng = np.random.default_rng(77)
a = (rng.standard_normal((32, 128, 128)) * 0.3).astype(np.float16); b2d = (rng.standard_normal((128, 96)) * 0.3).astype(np.float16)
rb = fnp.matmul(a, b2d); eb = np.matmul(a, b2d)
ok = ok and rb.shape == eb.shape and bool(((rb.view(np.uint16) == eb.view(np.uint16)) | (np.isnan(rb) & np.isnan(eb))).all())
a2d = (rng.standard_normal((96, 128)) * 0.3).astype(np.float16); b = (rng.standard_normal((32, 128, 64)) * 0.3).astype(np.float16)
rb2 = fnp.matmul(a2d, b); eb2 = np.matmul(a2d, b)
ok = ok and rb2.shape == eb2.shape and bool(((rb2.view(np.uint16) == eb2.view(np.uint16)) | (np.isnan(rb2) & np.isnan(eb2))).all())
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 matmul must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn f16_tensordot_inner_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy has no f16 BLAS, so tensordot(axes>=1) and inner run the slow naive widen matmul. Both
    // flatten to the native f16 GEMM (tensordot via contiguous reshape; inner via a @ contiguous(b.T)),
    // so the result must be BYTE-identical to np.tensordot / np.inner across axes specs and shapes.
    let script = fnp_script(
        r#"
ok = True
rng = np.random.default_rng(31)
# tensordot axes=1 (2-D) and axes=2 (4-D contracting 2)
a = (rng.standard_normal((256, 256)) * 0.3).astype(np.float16); b = (rng.standard_normal((256, 256)) * 0.3).astype(np.float16)
rt = fnp.tensordot(a, b, axes=1); et = np.tensordot(a, b, axes=1)
ok = ok and rt.dtype == et.dtype and rt.shape == et.shape
ok = ok and ((rt.view(np.uint16) == et.view(np.uint16)) | (np.isnan(rt) & np.isnan(et))).all()
a4 = (rng.standard_normal((20, 16, 8, 8)) * 0.3).astype(np.float16); b4 = (rng.standard_normal((8, 8, 12)) * 0.3).astype(np.float16)
rt2 = fnp.tensordot(a4, b4, axes=2); et2 = np.tensordot(a4, b4, axes=2)
ok = ok and rt2.shape == et2.shape and ((rt2.view(np.uint16) == et2.view(np.uint16)) | (np.isnan(rt2) & np.isnan(et2))).all()
# inner: contracts shared last axis
ai = (rng.standard_normal((300, 64)) * 0.3).astype(np.float16); bi = (rng.standard_normal((200, 64)) * 0.3).astype(np.float16)
ri = fnp.inner(ai, bi); ei = np.inner(ai, bi)
ok = ok and ri.dtype == ei.dtype and ri.shape == ei.shape
ok = ok and ((ri.view(np.uint16) == ei.view(np.uint16)) | (np.isnan(ri) & np.isnan(ei))).all()
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 tensordot/inner must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn f32_int_flat_sort_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy sorts every dtype single-threaded; the native parallel flat sort (rayon
    // par_sort_unstable over a fresh copy) must be BYTE-identical to np.sort above the gate.
    // Integers are byte-exact unconditionally (Ord == numpy ascending value order, incl signed
    // two's-complement, with ties/duplicates). f32 is byte-exact for no-NaN/no--0.0 input.
    let script = fnp_script(
        r#"
n = (1 << 20) + 257
rng = np.random.default_rng(123)
ok = True
# integer dtypes incl negatives, duplicates, full range
for dt in ("int8","int16","int32","int64","uint8","uint16","uint32","uint64"):
    info = np.iinfo(dt)
    a = rng.integers(info.min, info.max, n, dtype=dt, endpoint=True)
    a[:500] = a[500:1000]  # duplicates
    r = fnp.sort(a); e = np.sort(a)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# float32 (no NaN, no -0.0): includes +inf/-inf, +0.0, duplicates
f = (rng.standard_normal(n) * 10).astype(np.float32)
f[0] = np.float32(np.inf); f[1] = np.float32(-np.inf); f[2] = np.float32(0.0)
f[:300] = f[300:600]
r = fnp.sort(f); e = np.sort(f)
ok = ok and r.dtype == e.dtype and r.tobytes() == e.tobytes()
# f32 with NaN/-0.0 must DELEGATE and still match numpy
fn_ = f.copy(); fn_[5] = np.float32(np.nan); fn_[6] = np.float32(-0.0)
rn = fnp.sort(fn_); en = np.sort(fn_)
ok = ok and bool(((rn.view(np.uint32) == en.view(np.uint32)) | (np.isnan(rn) & np.isnan(en))).all())
# integer LAST-AXIS sort (2-D, many wide lanes) incl duplicates, default + explicit axis=-1, and a stable kind
for dt in ("int32", "int64", "uint32", "uint64"):
    info = np.iinfo(dt)
    m2 = rng.integers(info.min, info.max, (4096, 256), dtype=dt, endpoint=True)
    m2[:, :40] = m2[:, 40:80]  # per-lane duplicates
    r = fnp.sort(m2); e = np.sort(m2)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
    r2 = fnp.sort(m2, axis=-1); e2 = np.sort(m2, axis=-1)
    ok = ok and r2.tobytes() == e2.tobytes()
    rk = fnp.sort(m2, kind="stable"); ek = np.sort(m2, kind="stable")
    ok = ok and rk.tobytes() == ek.tobytes()
    # AXIS-0 (column) sort, 2-D (rows x cols), incl per-column duplicates
    c2 = rng.integers(info.min, info.max, (1024, 512), dtype=dt, endpoint=True)
    c2[:40, :] = c2[40:80, :]  # per-column duplicates
    ra = fnp.sort(c2, axis=0); ea = np.sort(c2, axis=0)
    ok = ok and ra.dtype == ea.dtype and ra.shape == ea.shape and ra.tobytes() == ea.tobytes()
    # MIDDLE-axis sort, 3-D (outer x alen x inner), incl per-lane duplicates along the sorted axis
    m3 = rng.integers(info.min, info.max, (64, 256, 64), dtype=dt, endpoint=True)
    m3[:, :40, :] = m3[:, 40:80, :]  # per-lane duplicates
    rm = fnp.sort(m3, axis=1); em = np.sort(m3, axis=1)
    ok = ok and rm.dtype == em.dtype and rm.shape == em.shape and rm.tobytes() == em.tobytes()
# COMPLEX128 VALUE sort (np.sort): flat distinct (perm real), flat with full dups, NaN/-0.0 delegate
cv = (rng.permutation(n).astype(np.float64) + 1j * rng.standard_normal(n)).astype(np.complex128)
rcv = fnp.sort(cv); ecv = np.sort(cv)
ok = ok and rcv.dtype == ecv.dtype and rcv.shape == ecv.shape and rcv.tobytes() == ecv.tobytes()
# full (re,im) duplicates: equal complex = identical bytes -> value sort still byte-exact (no tie-defer)
cvd = (rng.integers(0, 20, n).astype(np.float64) + 1j * rng.integers(0, 20, n).astype(np.float64)).astype(np.complex128)
ok = ok and fnp.sort(cvd).tobytes() == np.sort(cvd).tobytes()
# complex with NaN -> DELEGATE, still match (NaN-at-end ordering)
cvn = cv.copy(); cvn[9] = complex(np.nan, 2.0)
ok = ok and bool(((fnp.sort(cvn).view(np.float64) == np.sort(cvn).view(np.float64)) | (np.isnan(fnp.sort(cvn).view(np.float64)) & np.isnan(np.sort(cvn).view(np.float64)))).all())
# complex with -0.0 -> DELEGATE, still match
cvz = cv.copy(); cvz[3] = complex(-0.0, 1.0)
ok = ok and fnp.sort(cvz).tobytes() == np.sort(cvz).tobytes()
# COMPLEX128 LAST-AXIS value sort, 2-D distinct-real per lane
cvm = np.stack([rng.permutation(256).astype(np.float64) + 1j * rng.standard_normal(256) for _ in range(4096)]).astype(np.complex128)
ok = ok and fnp.sort(cvm).tobytes() == np.sort(cvm).tobytes()
# COMPLEX64 VALUE sort (np.sort): flat distinct (perm real < 2^24 = exact f32), full dups, NaN/-0.0 delegate, last-axis
c6 = (rng.permutation(n).astype(np.float32) + 1j * rng.standard_normal(n).astype(np.float32)).astype(np.complex64)
r6 = fnp.sort(c6); e6 = np.sort(c6)
ok = ok and r6.dtype == e6.dtype and r6.shape == e6.shape and r6.tobytes() == e6.tobytes()
c6d = (rng.integers(0, 20, n).astype(np.float32) + 1j * rng.integers(0, 20, n).astype(np.float32)).astype(np.complex64)
ok = ok and fnp.sort(c6d).tobytes() == np.sort(c6d).tobytes()  # full dups byte-exact
c6n = c6.copy(); c6n[9] = np.complex64(complex(np.nan, 2.0))
ok = ok and bool(((fnp.sort(c6n).view(np.float32) == np.sort(c6n).view(np.float32)) | (np.isnan(fnp.sort(c6n).view(np.float32)) & np.isnan(np.sort(c6n).view(np.float32)))).all())
c6z = c6.copy(); c6z[3] = np.complex64(complex(-0.0, 1.0))
ok = ok and fnp.sort(c6z).tobytes() == np.sort(c6z).tobytes()  # -0.0 delegate, still match
c6m = np.stack([rng.permutation(256).astype(np.float32) + 1j * rng.standard_normal(256).astype(np.float32) for _ in range(4096)]).astype(np.complex64)
ok = ok and fnp.sort(c6m).tobytes() == np.sort(c6m).tobytes()  # last-axis
# COMPLEX128 VALUE sort AXIS0 + MIDAXIS (gather/scatter), distinct-real per lane + full-dup byte-exact
cva0 = np.stack([rng.permutation(256).astype(np.float64) + 1j * rng.standard_normal(256) for _ in range(4096)], axis=1).astype(np.complex128)
ok = ok and fnp.sort(cva0, axis=0).tobytes() == np.sort(cva0, axis=0).tobytes()
cvmid_re = np.argsort(rng.standard_normal((64, 256, 64)), axis=1).astype(np.float64)
cvmid = (cvmid_re + 1j * rng.standard_normal((64, 256, 64))).astype(np.complex128)
ok = ok and fnp.sort(cvmid, axis=1).tobytes() == np.sort(cvmid, axis=1).tobytes()
# axis0 full (re,im) dups -> still byte-exact (value sort, no tie-defer)
cvad = (rng.integers(0, 20, (256, 4096)).astype(np.float64) + 1j * rng.integers(0, 20, (256, 4096)).astype(np.float64)).astype(np.complex128)
ok = ok and fnp.sort(cvad, axis=0).tobytes() == np.sort(cvad, axis=0).tobytes()
# COMPLEX64 VALUE sort AXIS0 + MIDAXIS (f32 gather/scatter), distinct-real per lane
c6va0 = np.stack([rng.permutation(256).astype(np.float32) + 1j * rng.standard_normal(256).astype(np.float32) for _ in range(4096)], axis=1).astype(np.complex64)
ok = ok and fnp.sort(c6va0, axis=0).tobytes() == np.sort(c6va0, axis=0).tobytes()
c6vmid_re = np.argsort(rng.standard_normal((64, 256, 64)), axis=1).astype(np.float32)
c6vmid = (c6vmid_re + 1j * rng.standard_normal((64, 256, 64)).astype(np.float32)).astype(np.complex64)
ok = ok and fnp.sort(c6vmid, axis=1).tobytes() == np.sort(c6vmid, axis=1).tobytes()
# DATETIME64 / TIMEDELTA64 flat VALUE sort (int64-backed): distinct + dups byte-exact; NaT delegate
dvs = rng.permutation(n).astype("datetime64[s]")
ok = ok and fnp.sort(dvs).dtype == np.sort(dvs).dtype and fnp.sort(dvs).tobytes() == np.sort(dvs).tobytes()
tvs = rng.integers(0, 1000, n).astype("timedelta64[s]")  # heavy dups (value sort, no tie-defer)
ok = ok and fnp.sort(tvs).tobytes() == np.sort(tvs).tobytes()
dvn = dvs.copy(); dvn[5] = np.datetime64("NaT")  # NaT -> delegate
ok = ok and fnp.sort(dvn).tobytes() == np.sort(dvn).tobytes()
# DATETIME64/TIMEDELTA64 VALUE sort AXES (last/axis0): distinct + heavy-dups byte-exact (no tie-defer); NaT delegate
dvl = np.stack([rng.permutation(256) for _ in range(4096)]).astype("datetime64[s]")  # last-axis distinct
ok = ok and fnp.sort(dvl).dtype == np.sort(dvl).dtype and fnp.sort(dvl).tobytes() == np.sort(dvl).tobytes()
tvl = rng.integers(0, 50, (4096, 256)).astype("timedelta64[s]")  # last-axis HEAVY dups -> byte-exact, no defer
ok = ok and fnp.sort(tvl).tobytes() == np.sort(tvl).tobytes()
dva0 = np.stack([rng.permutation(256) for _ in range(4096)], axis=1).astype("datetime64[s]")  # axis-0 distinct
ok = ok and fnp.sort(dva0, axis=0).dtype == np.sort(dva0, axis=0).dtype and fnp.sort(dva0, axis=0).tobytes() == np.sort(dva0, axis=0).tobytes()
dvln = dvl.copy(); dvln[3, 7] = np.datetime64("NaT")  # NaT in a lane -> whole-op delegate, still match
ok = ok and fnp.sort(dvln).tobytes() == np.sort(dvln).tobytes()
dvm = np.argsort(rng.standard_normal((64, 256, 64)), axis=1).astype("datetime64[s]")  # middle-axis distinct
ok = ok and fnp.sort(dvm, axis=1).dtype == np.sort(dvm, axis=1).dtype and fnp.sort(dvm, axis=1).tobytes() == np.sort(dvm, axis=1).tobytes()
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f32/int flat sort must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_argsort_flat_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy argsort is single-threaded introsort. The native parallel argsort (sort a [0..n] index
    // permutation by value) is byte-identical to np.argsort for DISTINCT integer values (unique
    // permutation); ties defer to numpy (unstable order is algorithm-specific). 4-/8-byte ints.
    let script = fnp_script(
        r#"
n = (1 << 20) + 257
rng = np.random.default_rng(131)
ok = True
# DISTINCT values (shuffled arange-like) -> native path, byte-exact
for dt in ("int32", "int64", "uint32", "uint64"):
    a = rng.permutation(n).astype(dt)  # distinct 0..n-1 permuted
    r = fnp.argsort(a); e = np.argsort(a)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
    # verify it actually sorts
    ok = ok and bool((a[r] == np.sort(a)).all())
# DUPLICATES -> must DELEGATE to numpy and still match exactly
ad = rng.integers(0, 1000, n, dtype=np.int64)  # heavy ties
rd = fnp.argsort(ad); ed = np.argsort(ad)
ok = ok and rd.tobytes() == ed.tobytes()
# FLOAT32 flat argsort, DISTINCT (perm 0..n-1 < 2^24 = exact f32) -> native path, byte-exact
af = rng.permutation(n).astype(np.float32)
rf = fnp.argsort(af); ef = np.argsort(af)
ok = ok and rf.dtype == ef.dtype and rf.shape == ef.shape and rf.tobytes() == ef.tobytes()
ok = ok and bool((af[rf] == np.sort(af)).all())
# FLOAT32 with NaN -> must DELEGATE (numpy NaN-at-end ordering) and still match
anf = rng.standard_normal(n).astype(np.float32); anf[5] = np.nan; anf[n // 2] = np.nan
ok = ok and fnp.argsort(anf).tobytes() == np.argsort(anf).tobytes()
# LAST-AXIS argsort, 2-D, distinct per-lane values
for dt in ("int32", "int64", "uint32", "uint64"):
    m = np.stack([rng.permutation(256).astype(dt) for _ in range(4096)])  # each lane distinct
    r2 = fnp.argsort(m); e2 = np.argsort(m)
    ok = ok and r2.dtype == e2.dtype and r2.shape == e2.shape and r2.tobytes() == e2.tobytes()
# 2-D with per-lane ties -> delegate, still match
mt = rng.integers(0, 50, (4096, 256), dtype=np.int64)
ok = ok and fnp.argsort(mt).tobytes() == np.argsort(mt).tobytes()
# AXIS-0 argsort, 2-D, distinct per-COLUMN values (each column a permutation of 0..rows-1)
for dt in ("int32", "int64", "uint32", "uint64"):
    m0 = np.stack([rng.permutation(256).astype(dt) for _ in range(4096)], axis=1)  # (256,4096)
    r3 = fnp.argsort(m0, axis=0); e3 = np.argsort(m0, axis=0)
    ok = ok and r3.dtype == e3.dtype and r3.shape == e3.shape and r3.tobytes() == e3.tobytes()
# axis-0 with per-column ties -> delegate, still match
mt0 = rng.integers(0, 50, (256, 4096), dtype=np.int64)
ok = ok and fnp.argsort(mt0, axis=0).tobytes() == np.argsort(mt0, axis=0).tobytes()
# MIDDLE-AXIS argsort, 3-D (64,256,64), distinct per-lane along axis=1
for dt in ("int32", "int64", "uint32", "uint64"):
    mm = np.argsort(rng.standard_normal((64, 256, 64)), axis=1).astype(dt)  # each axis-1 lane a perm
    r4 = fnp.argsort(mm, axis=1); e4 = np.argsort(mm, axis=1)
    ok = ok and r4.dtype == e4.dtype and r4.shape == e4.shape and r4.tobytes() == e4.tobytes()
# middle-axis with per-lane ties -> delegate, still match
mtm = rng.integers(0, 30, (64, 256, 64), dtype=np.int64)
ok = ok and fnp.argsort(mtm, axis=1).tobytes() == np.argsort(mtm, axis=1).tobytes()
# FLOAT32 axis variants, distinct per-lane (perm < 2^24 = exact f32) -> native, byte-exact
mf_last = np.stack([rng.permutation(256).astype(np.float32) for _ in range(4096)])  # last-axis
ok = ok and fnp.argsort(mf_last).tobytes() == np.argsort(mf_last).tobytes()
mf_ax0 = np.stack([rng.permutation(256).astype(np.float32) for _ in range(4096)], axis=1)  # (256,4096)
ok = ok and fnp.argsort(mf_ax0, axis=0).tobytes() == np.argsort(mf_ax0, axis=0).tobytes()
mf_mid = np.argsort(rng.standard_normal((64, 256, 64)), axis=1).astype(np.float32)  # middle axis
ok = ok and fnp.argsort(mf_mid, axis=1).tobytes() == np.argsort(mf_mid, axis=1).tobytes()
# f32 NaN per-lane -> delegate, still match (last-axis)
mfn = rng.standard_normal((4096, 256)).astype(np.float32); mfn[0, 3] = np.nan
ok = ok and fnp.argsort(mfn).tobytes() == np.argsort(mfn).tobytes()
# COMPLEX128 flat argsort: distinct real parts (permutation) so lexicographic (re,im) is tie-free
cre = rng.permutation(n).astype(np.float64); cim = rng.standard_normal(n)
cc = (cre + 1j * cim).astype(np.complex128)
rc = fnp.argsort(cc); ec = np.argsort(cc)
ok = ok and rc.dtype == ec.dtype and rc.shape == ec.shape and rc.tobytes() == ec.tobytes()
ok = ok and bool((cc[rc] == np.sort(cc)).all())
# complex with ties on real, broken by imag (still all distinct (re,im)) -> native, byte-exact
cre2 = rng.integers(0, 8, n).astype(np.float64); cim2 = rng.permutation(n).astype(np.float64)
cc2 = (cre2 + 1j * cim2).astype(np.complex128)  # re repeats, im distinct -> (re,im) distinct
ok = ok and fnp.argsort(cc2).tobytes() == np.argsort(cc2).tobytes()
# complex with full (re,im) duplicates -> DELEGATE (tie), still match
ccd = rng.integers(0, 50, n).astype(np.float64) + 1j * rng.integers(0, 50, n).astype(np.float64)
ok = ok and fnp.argsort(ccd.astype(np.complex128)).tobytes() == np.argsort(ccd.astype(np.complex128)).tobytes()
# complex with NaN -> DELEGATE, still match
ccn = cc.copy(); ccn[7] = complex(np.nan, 1.0)
ok = ok and fnp.argsort(ccn).tobytes() == np.argsort(ccn).tobytes()
# COMPLEX128 axis variants: distinct real parts per lane (tie-free lexicographic) -> native, byte-exact
cm_last = np.stack([rng.permutation(256).astype(np.float64) + 1j * rng.standard_normal(256) for _ in range(4096)])
cm_last = cm_last.astype(np.complex128)  # last-axis: each row distinct real
ok = ok and fnp.argsort(cm_last).tobytes() == np.argsort(cm_last).tobytes()
cm_ax0 = np.stack([rng.permutation(256).astype(np.float64) + 1j * rng.standard_normal(256) for _ in range(4096)], axis=1)
cm_ax0 = cm_ax0.astype(np.complex128)  # axis0: each column distinct real
ok = ok and fnp.argsort(cm_ax0, axis=0).tobytes() == np.argsort(cm_ax0, axis=0).tobytes()
cm_mid_re = np.argsort(rng.standard_normal((64, 256, 64)), axis=1).astype(np.float64)  # axis-1 lane = perm of real
cm_mid = (cm_mid_re + 1j * rng.standard_normal((64, 256, 64))).astype(np.complex128)
ok = ok and fnp.argsort(cm_mid, axis=1).tobytes() == np.argsort(cm_mid, axis=1).tobytes()
# c128 axis with full dup (tie) -> delegate, still match (last-axis)
cmd = (rng.integers(0, 16, (4096, 256)).astype(np.float64) + 1j * rng.integers(0, 16, (4096, 256)).astype(np.float64)).astype(np.complex128)
ok = ok and fnp.argsort(cmd).tobytes() == np.argsort(cmd).tobytes()
# COMPLEX64 argsort: flat distinct-real (perm), full-dup defer, NaN defer, last-axis distinct-per-lane
c6a = (rng.permutation(n).astype(np.float32) + 1j * rng.standard_normal(n).astype(np.float32)).astype(np.complex64)
r6a = fnp.argsort(c6a); e6a = np.argsort(c6a)
ok = ok and r6a.dtype == e6a.dtype and r6a.shape == e6a.shape and r6a.tobytes() == e6a.tobytes()
ok = ok and bool((c6a[r6a] == np.sort(c6a)).all())
c6ad = (rng.integers(0, 40, n).astype(np.float32) + 1j * rng.integers(0, 40, n).astype(np.float32)).astype(np.complex64)
ok = ok and fnp.argsort(c6ad).tobytes() == np.argsort(c6ad).tobytes()  # full dup tie -> delegate
c6an = c6a.copy(); c6an[7] = np.complex64(complex(np.nan, 1.0))
ok = ok and fnp.argsort(c6an).tobytes() == np.argsort(c6an).tobytes()  # NaN -> delegate
c6al = np.stack([rng.permutation(256).astype(np.float32) + 1j * rng.standard_normal(256).astype(np.float32) for _ in range(4096)]).astype(np.complex64)
ok = ok and fnp.argsort(c6al).tobytes() == np.argsort(c6al).tobytes()  # last-axis distinct-per-lane
# COMPLEX64 argsort AXIS0 + MIDAXIS, distinct-real per lane
c6a0 = np.stack([rng.permutation(256).astype(np.float32) + 1j * rng.standard_normal(256).astype(np.float32) for _ in range(4096)], axis=1).astype(np.complex64)
ok = ok and fnp.argsort(c6a0, axis=0).tobytes() == np.argsort(c6a0, axis=0).tobytes()
c6mid_re = np.argsort(rng.standard_normal((64, 256, 64)), axis=1).astype(np.float32)
c6mid = (c6mid_re + 1j * rng.standard_normal((64, 256, 64)).astype(np.float32)).astype(np.complex64)
ok = ok and fnp.argsort(c6mid, axis=1).tobytes() == np.argsort(c6mid, axis=1).tobytes()
# DATETIME64 / TIMEDELTA64 flat argsort (int64-backed): distinct -> native byte-exact; NaT/dup -> delegate
dts = rng.permutation(n).astype("datetime64[s]")  # distinct ticks
ok = ok and fnp.argsort(dts).tobytes() == np.argsort(dts).tobytes()
tds = rng.permutation(n).astype("timedelta64[s]")
ok = ok and fnp.argsort(tds).tobytes() == np.argsort(tds).tobytes()
dtd = rng.integers(0, 1000, n).astype("datetime64[s]")  # heavy ties -> delegate
ok = ok and fnp.argsort(dtd).tobytes() == np.argsort(dtd).tobytes()
dtn = dts.copy(); dtn[5] = np.datetime64("NaT"); dtn[n // 2] = np.datetime64("NaT")  # NaT -> delegate
ok = ok and fnp.argsort(dtn).tobytes() == np.argsort(dtn).tobytes()
# DATETIME64 argsort AXES (last/axis0/mid), distinct-per-lane via int64 view
dml = np.stack([rng.permutation(256) for _ in range(4096)]).astype("datetime64[s]")  # last-axis
ok = ok and fnp.argsort(dml).tobytes() == np.argsort(dml).tobytes()
dma0 = np.stack([rng.permutation(256) for _ in range(4096)], axis=1).astype("datetime64[s]")  # axis0
ok = ok and fnp.argsort(dma0, axis=0).tobytes() == np.argsort(dma0, axis=0).tobytes()
dmm = np.argsort(rng.standard_normal((64, 256, 64)), axis=1).astype("datetime64[s]")  # middle axis
ok = ok and fnp.argsort(dmm, axis=1).tobytes() == np.argsort(dmm, axis=1).tobytes()
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native int flat argsort must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn char_case_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy char.upper/lower/swapcase run single-threaded per element; the native ASCII path maps
    // codepoints in parallel above the gate. Must be BYTE-identical to numpy for all-ASCII input,
    // and DELEGATE (still match) on any non-ASCII codepoint (numpy uses full-Unicode casing that
    // can change width, e.g. 'ß'.upper()=='SS').
    let script = fnp_script(
        r#"
ok = True
# large all-ASCII U16 array (1M strings x 16 codepoints = 16M >> gate)
base = np.array(["aZ_bY9-cX_%d" % (i % 89) for i in range(1000)], dtype="<U16")
a = np.tile(base, 1000 + 1)[: (1 << 20) + 257]
for op in ("upper", "lower", "swapcase", "capitalize", "title"):
    r = getattr(fnp.char, op)(a); e = getattr(np.char, op)(a)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# char.translate: 1:1 ASCII codepoint remap (parallel lookup), large array
tbl = str.maketrans("abcdXYZ9", "ABCDxyz0")
rt = fnp.char.translate(a, tbl); et = np.char.translate(a, tbl)
ok = ok and rt.dtype == et.dtype and rt.shape == et.shape and rt.tobytes() == et.tobytes()
# char.add: element-wise concat (fixed output width wa+wb), same-shape large arrays
b = np.tile(np.array(["_QR%d" % (i % 71) for i in range(1000)], dtype="<U8"), 1000 + 1)[: (1 << 20) + 257]
rad = fnp.char.add(a, b); ead = np.char.add(a, b)
ok = ok and rad.dtype == ead.dtype and rad.shape == ead.shape and rad.tobytes() == ead.tobytes()
# char.add works for ANY unicode (pure concat, no casing) + embedded nulls
ua = np.tile(np.array(["café", "x\x00y"], dtype="<U6"), ((1 << 20) // 2) + 2)
ub = np.tile(np.array(["ü9", "Z"], dtype="<U4"), ((1 << 20) // 2) + 2)
rau = fnp.char.add(ua, ub); eau = np.char.add(ua, ub)
ok = ok and rau.dtype == eau.dtype and rau.tobytes() == eau.tobytes()
# np.strings.add too
ok = ok and fnp.strings.add(a, b).tobytes() == np.strings.add(a, b).tobytes()
# char.strip/lstrip/rstrip (whitespace, fixed width), large array with leading/trailing ws + tabs
sb = np.array(["  hi \t", "\nx\ty\r", "   ", "abc", "\x1cQ\x1f", "  e f  "], dtype="<U10")
sa = np.tile(sb, ((1 << 20) // sb.size) + 2)
for op in ("strip", "lstrip", "rstrip"):
    rs = getattr(fnp.char, op)(sa); es = getattr(np.char, op)(sa)
    ok = ok and rs.dtype == es.dtype and rs.shape == es.shape and rs.tobytes() == es.tobytes()
    ok = ok and getattr(fnp.strings, op)(sa).tobytes() == getattr(np.strings, op)(sa).tobytes()
# chars-arg strip must DELEGATE and still match
ok = ok and fnp.char.strip(sa, "h").tobytes() == np.char.strip(sa, "h").tobytes()
# char.replace: expansion / contraction / same-len / no-match / overlapping pattern; output width = max result
rb = np.array(["aXbXc", "XXXX", "nomatch", "XXX", "", "oXo"], dtype="<U10")
ra2 = np.tile(rb, ((1 << 20) // rb.size) + 2)
for old, new in (("X", "YZ"), ("XX", "Y"), ("X", "Q"), ("Z", "W"), ("o", "")):
    rr = fnp.char.replace(ra2, old, new); er = np.char.replace(ra2, old, new)
    ok = ok and rr.dtype == er.dtype and rr.shape == er.shape and rr.tobytes() == er.tobytes()
    ok = ok and fnp.strings.replace(ra2, old, new).tobytes() == np.strings.replace(ra2, old, new).tobytes()
# count arg + non-ASCII old must DELEGATE and still match
ok = ok and fnp.char.replace(ra2, "X", "YZ", 1).tobytes() == np.char.replace(ra2, "X", "YZ", 1).tobytes()
# char.multiply: repeat content n times, output width = max_content*n; works for any unicode
mb = np.array(["ab", "cde", "f", "", "café"], dtype="<U6")
ma = np.tile(mb, ((1 << 20) // mb.size) + 2)
for k in (1, 2, 3, 5):
    rm2 = fnp.char.multiply(ma, k); em2 = np.char.multiply(ma, k)
    ok = ok and rm2.dtype == em2.dtype and rm2.shape == em2.shape and rm2.tobytes() == em2.tobytes()
    ok = ok and fnp.strings.multiply(ma, k).tobytes() == np.strings.multiply(ma, k).tobytes()
# n<=0 must DELEGATE and still match
ok = ok and fnp.char.multiply(ma, 0).tobytes() == np.char.multiply(ma, 0).tobytes()
# is* bool predicates (fixed bool output): mixed alpha/digit/alnum/space/empty content
pb = np.array(["abc", "ABC", "a1b", "123", "   ", "", "a b", "x9", "  ", "9z"], dtype="<U4")
pa = np.tile(pb, ((1 << 20) // pb.size) + 2)
for op in ("isalpha", "isdigit", "isalnum", "isspace"):
    rp = getattr(fnp.char, op)(pa); ep = getattr(np.char, op)(pa)
    ok = ok and rp.dtype == ep.dtype and rp.shape == ep.shape and rp.tobytes() == ep.tobytes()
    ok = ok and getattr(fnp.strings, op)(pa).tobytes() == getattr(np.strings, op)(pa).tobytes()
# non-ASCII (é is alpha) must DELEGATE and still match
up = np.tile(np.array(["café", "123", "  "], dtype="<U5"), ((1 << 20) // 3) + 2)
ok = ok and fnp.char.isalpha(up).tobytes() == np.char.isalpha(up).tobytes()
# non-ASCII must delegate to numpy and still match (full-Unicode casing)
u = np.tile(np.array(["café_StraßE", "ÀÉÎ_xyz"], dtype="<U16"), ((1 << 20) // 2) + 2)
for op in ("upper", "lower"):
    r = getattr(fnp.char, op)(u); e = getattr(np.char, op)(u)
    ok = ok and r.tobytes() == e.tobytes()
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native parallel char upper/lower/swapcase must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn f32_searchsorted_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy searchsorted is a single-threaded cold-cache binary search per query; the parallel
    // per-query lower/upper-bound search must return the identical intp index array for both
    // sides, incl exact-match ties and -inf/inf, above the gate. NaN defers to numpy.
    let script = fnp_script(
        r#"
n = (1 << 21) + 257
rng = np.random.default_rng(71)
a = np.sort(rng.standard_normal(500000).astype(np.float32))
v = (rng.standard_normal(n) * 2.0).astype(np.float32)
v[:1000] = a[rng.integers(0, a.size, 1000)]   # exact-match ties
v[1] = np.float32(np.inf); v[2] = np.float32(-np.inf)
ok = True
for side in ("left", "right"):
    r = fnp.searchsorted(a, v, side=side); e = np.searchsorted(a, v, side=side)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# 2-D query shape preserved
v2 = v[:1 << 21].reshape(1024, 2048)
ok = ok and fnp.searchsorted(a, v2).tobytes() == np.searchsorted(a, v2).tobytes()
ok = ok and fnp.searchsorted(a, v2).shape == np.searchsorted(a, v2).shape
# NaN in query defers to numpy and still matches
vn = v.copy(); vn[5] = np.float32(np.nan)
ok = ok and fnp.searchsorted(a, vn).tobytes() == np.searchsorted(a, vn).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f32 searchsorted must be bit-identical to numpy (both sides, ties, NaN defer): {result}"
    );
    Ok(())
}

#[test]
fn timedelta_remainder_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // td % td -> timedelta64 (same unit). For same-unit non-NaT non-zero operands it equals the
    // int64 floored remainder of the raw counts viewed back to timedelta. NaT / zero divisor defer
    // to numpy (which returns NaT). Must be byte-identical above the 1<<18 gate.
    let script = fnp_script(
        r#"
import warnings
n = (1 << 18) + 257
rng = np.random.default_rng(67)
a = rng.integers(-10**7, 10**7, n).astype('timedelta64[s]')
b = rng.integers(-10**7, 10**7, n).astype('timedelta64[s]')
b[b == np.timedelta64(0, 's')] = np.timedelta64(1, 's')
a[0] = np.timedelta64(7, 's'); b[0] = np.timedelta64(-3, 's')   # mixed-sign floored remainder
ok = True
r = a % b; e = np.remainder(a, b)
ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
ok = ok and np.remainder(a, b).tobytes() == e.tobytes()
# NaT present + zero divisor -> defer to numpy (NaT), still match
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    an = a.copy(); an[5] = np.timedelta64('NaT')
    ok = ok and (an % b).tobytes() == np.remainder(an, b).tobytes()
    bz = b.copy(); bz[9] = np.timedelta64(0, 's')
    ok = ok and (a % bz).tobytes() == np.remainder(a, bz).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native timedelta64 remainder must be bit-identical to numpy (incl NaT/zero defer): {result}"
    );
    Ok(())
}

#[test]
fn timedelta_floordiv_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // td // td -> int64. numpy runs it single-threaded with per-element NaT handling; for same-
    // unit non-NaT non-zero operands it equals int64 floor_divide of the raw counts. NaT and zero
    // divisor defer to numpy (which returns 0). Must be byte-identical above the 1<<18 gate.
    let script = fnp_script(
        r#"
import warnings
n = (1 << 18) + 257
rng = np.random.default_rng(61)
a = rng.integers(-10**7, 10**7, n).astype('timedelta64[s]')
b = rng.integers(-10**7, 10**7, n).astype('timedelta64[s]')
b[b == np.timedelta64(0, 's')] = np.timedelta64(1, 's')   # non-zero divisors -> exercise kernel
a[0] = np.timedelta64(7, 's'); b[0] = np.timedelta64(-3, 's')   # mixed-sign floor
ok = True
r = a // b; e = np.floor_divide(a, b)
ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
ok = ok and np.floor_divide(a, b).tobytes() == e.tobytes()
# NaT present -> defers to numpy (numpy returns 0), still matches
an = a.copy(); an[5] = np.timedelta64('NaT')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ok = ok and (an // b).tobytes() == np.floor_divide(an, b).tobytes()
    # zero divisor -> defers to numpy
    bz = b.copy(); bz[9] = np.timedelta64(0, 's')
    ok = ok and (a // bz).tobytes() == np.floor_divide(a, bz).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native timedelta64 floor_divide must be bit-identical to numpy (incl NaT/zero defer): {result}"
    );
    Ok(())
}

#[test]
fn f16_nextafter_full_domain_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy widens f16->f32 for nextafter; the native uint16 bit-step must be byte-identical over
    // the FULL f16 domain for several scalar targets b (incl 0/-0/inf/nan), tiled past the gate.
    // numpy f16 returns x1's bits on the equal case (incl signed zeros).
    let script = fnp_script(
        r#"
patterns = np.arange(65536, dtype=np.uint16).view(np.float16)
reps = ((1 << 20) // patterns.size) + 2
x = np.tile(patterns, reps)
ok = True
for bv in [1.0, -1.0, 0.0, -0.0, np.inf, -np.inf, 0.5, 65504.0]:
    b = np.full(x.size, np.float16(bv), dtype=np.float16)
    r = fnp.nextafter(x, b); e = np.nextafter(x, b)
    same = ((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all()
    ok = ok and bool(same) and r.dtype == e.dtype
# elementwise pair (both arrays) + 2-D
y = patterns[::-1]
xt = np.tile(patterns, reps); yt = np.tile(y, reps)
r = fnp.nextafter(xt, yt); e = np.nextafter(xt, yt)
ok = ok and bool(((r.view(np.uint16)==e.view(np.uint16))|(np.isnan(r)&np.isnan(e))).all())
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 nextafter must be bit-identical to numpy over the full domain: {result}"
    );
    Ok(())
}

#[test]
fn f16_copysign_heaviside_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy widens f16->f32 for copysign/heaviside; the native parallel kernels (copysign =
    // uint16 sign-bit copy, heaviside = widen-piecewise) must be byte-identical incl inf/nan/-0.0,
    // above the 1<<20 gate.
    let script = fnp_script(
        r#"
n = (1 << 20) + 257
rng = np.random.default_rng(59)
a = (rng.standard_normal(n) * 10.0).astype(np.float16)
b = (rng.standard_normal(n) * 3.0).astype(np.float16)
a[0]=np.float16(np.inf); a[1]=np.float16(-np.inf); a[2]=np.float16(np.nan); a[3]=np.float16(-0.0); a[4]=np.float16(0.0)
b[0]=np.float16(-1.0); b[2]=np.float16(7.0); b[4]=np.float16(2.5)
ok = True
r = fnp.copysign(a, b); e = np.copysign(a, b)
ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
r = fnp.heaviside(a, b); e = np.heaviside(a, b)
ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# 2-D shape preserved
a2 = a[:1 << 20].reshape(1024, 1024); b2 = b[:1 << 20].reshape(1024, 1024)
ok = ok and fnp.copysign(a2, b2).tobytes() == np.copysign(a2, b2).tobytes()
ok = ok and fnp.heaviside(a2, b2).tobytes() == np.heaviside(a2, b2).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 copysign/heaviside must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn f16_fmod_remainder_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy widens f16->f32 for fmod/remainder (the slowest f16 binary ops); the native parallel
    // widen-op-narrow must be byte-identical incl inf/nan/-0.0/mixed-signs, above the 1<<20 gate.
    // A zero divisor defers to numpy (warning + nan/0) and must still match.
    let script = fnp_script(
        r#"
import warnings
n = (1 << 20) + 257
rng = np.random.default_rng(53)
a = (rng.standard_normal(n) * 100.0).astype(np.float16)
b = (rng.standard_normal(n) * 7.0).astype(np.float16)
b[np.abs(b) < 0.05] = np.float16(1.5)   # non-zero divisors to exercise the kernel
a[0]=np.float16(np.inf); a[1]=np.float16(-np.inf); a[2]=np.float16(np.nan); a[3]=np.float16(-0.0)
b[2]=np.float16(3.0); b[3]=np.float16(-2.0)
a[4]=np.float16(7.0);  b[4]=np.float16(-3.0)   # mixed-sign floored remainder
ok = True
r = fnp.fmod(a, b); e = np.fmod(a, b)
ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
r = fnp.remainder(a, b); e = np.remainder(a, b)
ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
ok = ok and (a % b).tobytes() == np.remainder(a, b).tobytes()   # operator form
# zero divisor defers to numpy (warning suppressed) and still matches
bz = b.copy(); bz[5] = np.float16(0.0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ok = ok and fnp.fmod(a, bz).tobytes() == np.fmod(a, bz).tobytes()
    ok = ok and fnp.remainder(a, bz).tobytes() == np.remainder(a, bz).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 fmod/remainder must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn f16_nan_to_num_full_domain_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy widens f16->f32 for nan_to_num; the native uint16 bit-replacement must be byte-
    // identical over the FULL f16 domain (every nan/inf/finite pattern), default args AND custom
    // nan/posinf/neginf, tiled past the 1<<20 gate.
    let script = fnp_script(
        r#"
patterns = np.arange(65536, dtype=np.uint16).view(np.float16)
x = np.tile(patterns, ((1 << 20) // patterns.size) + 2)
ok = True
# default args
r = fnp.nan_to_num(x); e = np.nan_to_num(x)
ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# custom nan/posinf/neginf
r = fnp.nan_to_num(x, nan=2.0, posinf=100.0, neginf=-50.0)
e = np.nan_to_num(x, nan=2.0, posinf=100.0, neginf=-50.0)
ok = ok and r.tobytes() == e.tobytes()
# 2-D shape preserved
x2 = np.tile(patterns, (1 << 20) // patterns.size).reshape(-1, patterns.size)
ok = ok and fnp.nan_to_num(x2).tobytes() == np.nan_to_num(x2).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 nan_to_num must be bit-identical to numpy over the full domain: {result}"
    );
    Ok(())
}

#[test]
fn f16_clip_scalar_bounds_full_domain_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy widens f16->f32 to clip; the native parallel uint16-view clamp must be byte-identical
    // over the ENTIRE f16 domain (all 65536 patterns, incl NaN/inf/-0.0) for several scalar bound
    // pairs (incl zero / reversed bounds), tiled past the 1<<20 gate.
    let script = fnp_script(
        r#"
patterns = np.arange(65536, dtype=np.uint16).view(np.float16)
x = np.tile(patterns, ((1 << 20) // patterns.size) + 2)
ok = True
for lo, hi in [(-0.5, 0.5), (0.0, 1.0), (-1.0, 0.0), (-0.0, 0.0), (0.3, 0.7), (2.0, -1.0)]:
    r = fnp.clip(x, lo, hi); e = np.clip(x, lo, hi)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# 2-D shape preserved
x2 = np.tile(patterns, (1 << 20) // patterns.size).reshape(-1, patterns.size)
ok = ok and fnp.clip(x2, -0.5, 0.5).tobytes() == np.clip(x2, -0.5, 0.5).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 clip must be bit-identical to numpy over the full domain: {result}"
    );
    Ok(())
}

#[test]
fn f16_nonlast_axis_argmin_argmax_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy widens f16->f32 per column for argmin/argmax along a non-last axis; the native strided
    // parallel scan must return the identical first-occurrence index array (axis=0 of a 2-D, and a
    // middle axis of a 3-D), above the gate. A NaN anywhere defers the whole call to numpy.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(47)
ok = True
# 2-D axis=0 (outer==1 -> column-block case), rows*cols > 1<<20
m = (rng.standard_normal((1100, 1000)) * 50.0).astype(np.float16)
m[10] = m[0]   # ties down a column
m[0, 5] = np.float16(0.0); m[1, 5] = np.float16(-0.0)
for fnp_op, np_op in [(fnp.argmax, np.argmax), (fnp.argmin, np.argmin)]:
    r = fnp_op(m, axis=0); e = np_op(m, axis=0)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# 3-D middle axis (outer>1)
t = (rng.standard_normal((40, 700, 40)) * 30.0).astype(np.float16)  # 1.12M
for fnp_op, np_op in [(fnp.argmax, np.argmax), (fnp.argmin, np.argmin)]:
    r = fnp_op(t, axis=1); e = np_op(t, axis=1)
    ok = ok and r.shape == e.shape and r.tobytes() == e.tobytes()
# NaN defers to numpy and still matches
mn = m.copy(); mn[7, 3] = np.float16(np.nan)
ok = ok and fnp.argmax(mn, axis=0).tobytes() == np.argmax(mn, axis=0).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 non-last-axis argmin/argmax must match numpy first-occurrence index: {result}"
    );
    Ok(())
}

#[test]
fn f16_lastaxis_argmin_argmax_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy widens f16->f32 per lane for argmin/argmax(axis=-1); the native per-lane uint16-view
    // scan must return the identical first-occurrence index array, above the 1<<20 gate. A NaN
    // lane defers the whole call to numpy (first-NaN index) and must still match.
    let script = fnp_script(
        r#"
rows, cols = 4096, 300   # rows*cols > 1<<20
rng = np.random.default_rng(43)
a = (rng.standard_normal((rows, cols)) * 50.0).astype(np.float16)
# ties + signed zeros within lanes
a[:, 10] = a[:, 0]
a[0, 5] = np.float16(0.0); a[0, 6] = np.float16(-0.0)
a[1, :3] = np.float16(np.inf)
ok = True
for fnp_op, np_op in [(fnp.argmax, np.argmax), (fnp.argmin, np.argmin)]:
    r = fnp_op(a, axis=-1); e = np_op(a, axis=-1)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
    r1 = fnp_op(a, axis=1); e1 = np_op(a, axis=1)   # axis=1 == last axis here
    ok = ok and r1.tobytes() == e1.tobytes()
# NaN lane defers to numpy and still matches (first-NaN index)
an = a.copy(); an[2, 7] = np.float16(np.nan); an[2, 1] = np.float16(np.nan)
ok = ok and fnp.argmax(an, axis=-1).tobytes() == np.argmax(an, axis=-1).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 last-axis argmin/argmax must match numpy first-occurrence index (incl NaN-lane defer): {result}"
    );
    Ok(())
}

#[test]
fn f16_flat_argmin_argmax_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy widens f16->f32 to scan for argmin/argmax; the native parallel uint16-view scan
    // returns the identical first-occurrence index, above the 1<<20 gate. NaN defers to numpy
    // (first-NaN index) and must still match.
    let script = fnp_script(
        r#"
n = (1 << 20) + 257
rng = np.random.default_rng(41)
ok = True
for scale in (1.0, 30.0, 400.0):
    x = (rng.standard_normal(n) * scale).astype(np.float16)
    # force ties + signed zeros (index-based, must keep first occurrence)
    x[100:110] = x.max(); x[200:210] = x.min()
    x[300] = np.float16(0.0); x[301] = np.float16(-0.0)
    x[5] = np.float16(np.inf); x[6] = np.float16(-np.inf)
    for fnp_op, np_op in [(fnp.argmax, np.argmax), (fnp.argmin, np.argmin)]:
        r = fnp_op(x); e = np_op(x)
        ok = ok and int(r) == int(e)
# NaN defers to numpy (first-NaN index) and still matches
nan_arr = (rng.standard_normal(n) * 30.0).astype(np.float16); nan_arr[123] = np.float16(np.nan); nan_arr[7] = np.float16(np.nan)
ok = ok and int(fnp.argmax(nan_arr)) == int(np.argmax(nan_arr))
ok = ok and int(fnp.argmin(nan_arr)) == int(np.argmin(nan_arr))
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 argmin/argmax must match numpy first-occurrence index (kernel + NaN defer): {result}"
    );
    Ok(())
}

#[test]
fn f16_flat_ptp_reduction_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy widens f16->f32 for ptp (max-min); the native parallel one-pass max-min reduce
    // narrowed to f16 must be byte-identical, above the 1<<20 gate. NaN defers to numpy.
    let script = fnp_script(
        r#"
n = (1 << 20) + 257
rng = np.random.default_rng(37)
ok = True
for scale in (1.0, 50.0, 500.0):
    x = (rng.standard_normal(n) * scale).astype(np.float16)
    x[5] = np.float16(np.inf); x[6] = np.float16(-np.inf)
    r = fnp.ptp(x); e = np.ptp(x)
    ok = ok and r.dtype == e.dtype and r.view(np.uint16) == e.view(np.uint16)
# all-equal -> ptp 0; mixed signed zeros -> ptp 0
z = np.full(n, np.float16(3.0)); z[: n // 2] = np.float16(-0.0); z[n // 2 :] = np.float16(0.0)
ok = ok and fnp.ptp(z).view(np.uint16) == np.ptp(z).view(np.uint16)
# NaN defers to numpy and still matches
nan_arr = (rng.standard_normal(n) * 50.0).astype(np.float16); nan_arr[9] = np.float16(np.nan)
ok = ok and np.isnan(fnp.ptp(nan_arr)) and np.isnan(np.ptp(nan_arr))
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 ptp must be bit-identical to numpy (kernel + NaN defer): {result}"
    );
    Ok(())
}

#[test]
fn f16_flat_min_max_reduction_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy widens f16->f32 to reduce; the native parallel f32-fold reduce narrowed to f16 must
    // be byte-identical for the no-NaN / non-zero-extremum case, above the 1<<20 gate. NaN and
    // zero-extremum arrays defer to numpy and must still match.
    let script = fnp_script(
        r#"
n = (1 << 20) + 257
rng = np.random.default_rng(31)
ok = True
# kernel path: shift away from zero so the extremum is non-zero, no NaN
a = (rng.standard_normal(n) * 100.0 + 500.0).astype(np.float16)   # all positive, max != 0
b = (-(rng.standard_normal(n) * 100.0 + 500.0)).astype(np.float16) # all negative, min != 0
a[123] = np.float16(np.inf); b[123] = np.float16(-np.inf)
for arr in (a, b):
    for fnp_op, np_op in [(fnp.max, np.max), (fnp.min, np.min), (fnp.amax, np.amax), (fnp.amin, np.amin)]:
        r = fnp_op(arr); e = np_op(arr)
        ok = ok and r.dtype == e.dtype and r.view(np.uint16) == e.view(np.uint16)
# defer paths still match numpy: NaN present, and zero extremum
nan_arr = a.copy(); nan_arr[7] = np.float16(np.nan)
ok = ok and np.isnan(fnp.max(nan_arr)) and np.isnan(np.max(nan_arr))
zero_arr = (-np.abs(rng.standard_normal(n)).astype(np.float16))  # all <= 0, max is 0
ok = ok and fnp.max(zero_arr).view(np.uint16) == np.max(zero_arr).view(np.uint16)
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 flat min/max reduction must be bit-identical to numpy (kernel + defer): {result}"
    );
    Ok(())
}

#[test]
fn f16_axis_min_max_reduction_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy widens f16->f32 per lane to reduce min/max along an axis (and strides for non-last
    // axes). The native uint16-view per-lane parallel f32-fold reduce narrowed to f16 must be
    // byte-identical for the no-NaN / non-zero-extremum case on the LAST axis, AXIS 0, and a MIDDLE
    // axis. NaN-present and zero-extremum arrays defer to numpy and must still match byte-for-byte.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(41)
ok = True
# kernel path: shifted positive so extrema != 0, no NaN. 2-D (last + axis0) and 3-D (middle), all 1<<20.
m2 = (rng.standard_normal((4096, 256)) * 50 + 300).astype(np.float16)   # all positive
m3 = (rng.standard_normal((64, 256, 64)) * 50 + 300).astype(np.float16)
for fnp_op, np_op in [(fnp.min, np.min), (fnp.max, np.max), (fnp.amin, np.amin), (fnp.amax, np.amax)]:
    for arr, ax in [(m2, -1), (m2, 0), (m3, 1)]:
        r = fnp_op(arr, axis=ax); e = np_op(arr, axis=ax)
        ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.view(np.uint16).tobytes() == e.view(np.uint16).tobytes()
# negative array (min != 0) along the last axis
mn = (-(rng.standard_normal((4096, 256)) * 50 + 300)).astype(np.float16)
ok = ok and fnp.min(mn, axis=-1).view(np.uint16).tobytes() == np.min(mn, axis=-1).view(np.uint16).tobytes()
# NaN present in a lane -> defer to numpy (NaN propagation), still byte-identical
nan2 = m2.copy(); nan2[3, 7] = np.float16(np.nan)
ok = ok and fnp.max(nan2, axis=-1).view(np.uint16).tobytes() == np.max(nan2, axis=-1).view(np.uint16).tobytes()
# per-lane zero extremum -> defer (+0/-0 ambiguity), still byte-identical
z2 = (-np.abs(rng.standard_normal((4096, 256))).astype(np.float16))  # all <= 0, per-lane max is 0
ok = ok and fnp.max(z2, axis=-1).view(np.uint16).tobytes() == np.max(z2, axis=-1).view(np.uint16).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 axis min/max reduction must be bit-identical to numpy (last/axis0/middle + defer): {result}"
    );
    Ok(())
}

#[test]
fn f16_axis_ptp_reduction_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy widens f16->f32 for BOTH max and min passes then subtracts (the slowest f16 reduction),
    // striding for non-last axes. The native uint16-view per-lane parallel max-min narrowed to f16
    // must be byte-identical on the LAST axis, AXIS 0, and a MIDDLE axis. A NaN-present array defers
    // to numpy and must still match byte-for-byte. ptp is non-negative so no signed-zero tie.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(42)
ok = True
m2 = (rng.standard_normal((4096, 256)) * 50 + 300).astype(np.float16)
m3 = (rng.standard_normal((64, 256, 64)) * 50 + 300).astype(np.float16)
for arr, ax in [(m2, -1), (m2, 0), (m3, 1)]:
    r = fnp.ptp(arr, axis=ax); e = np.ptp(arr, axis=ax)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.view(np.uint16).tobytes() == e.view(np.uint16).tobytes()
# NaN present in a lane -> defer to numpy, still byte-identical
nan2 = m2.copy(); nan2[3, 7] = np.float16(np.nan)
ok = ok and fnp.ptp(nan2, axis=-1).view(np.uint16).tobytes() == np.ptp(nan2, axis=-1).view(np.uint16).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 axis ptp reduction must be bit-identical to numpy (last/axis0/middle + defer): {result}"
    );
    Ok(())
}

#[test]
fn f16_nanmin_nanmax_flat_and_axis_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy has no f16 ALU, so np.nanmin/np.nanmax of float16 widens f16->f32 to reduce while
    // skipping NaN (~32ms@16M, ~5x f64). The native uint16-view skip-NaN reduce narrowed to f16 must
    // be byte-identical (FLAT + LAST axis + AXIS 0 + MIDDLE axis). All-NaN lanes (numpy NaN + warning)
    // and zero-extremum lanes defer to numpy and must still match byte-for-byte.
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(43)
ok = True
n = (1 << 20) + 257
# FLAT: shifted positive (extrema != 0), sparse NaN, not all-NaN
a = (np.abs(rng.standard_normal(n)) * 50 + 300).astype(np.float16); a[::997] = np.float16(np.nan)
for fnp_op, np_op in [(fnp.nanmin, np.nanmin), (fnp.nanmax, np.nanmax)]:
    r = fnp_op(a); e = np_op(a)
    ok = ok and r.dtype == e.dtype and r.view(np.uint16) == e.view(np.uint16)
# AXIS: 2-D (last + axis0) + 3-D (middle), sparse NaN, no all-NaN lane
m2 = (np.abs(rng.standard_normal((4096, 256))) * 50 + 300).astype(np.float16); m2[::7, ::13] = np.float16(np.nan)
m3 = (np.abs(rng.standard_normal((64, 256, 64))) * 50 + 300).astype(np.float16); m3[::3, ::11, ::5] = np.float16(np.nan)
for fnp_op, np_op in [(fnp.nanmin, np.nanmin), (fnp.nanmax, np.nanmax)]:
    for arr, ax in [(m2, -1), (m2, 0), (m3, 1)]:
        r = fnp_op(arr, axis=ax); e = np_op(arr, axis=ax)
        ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.view(np.uint16).tobytes() == e.view(np.uint16).tobytes()
# all-NaN lane -> defer to numpy (NaN + "All-NaN slice" warning), still byte-identical
mall = m2.copy(); mall[5, :] = np.float16(np.nan)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ok = ok and fnp.nanmin(mall, axis=-1).view(np.uint16).tobytes() == np.nanmin(mall, axis=-1).view(np.uint16).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 nanmin/nanmax (flat + last/axis0/middle + defer) must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn f16_axis_cumsum_cumprod_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy has no f16 ALU, so np.cumsum/np.cumprod of float16 widens f16->f32 per element then
    // narrows back to f16 EACH STEP (the accumulator is f16). The native uint16-view per-lane scan
    // carries the SAME f16-narrowed accumulator, parallel across independent lanes, so it is
    // byte-identical on the LAST axis, AXIS 0, and a MIDDLE axis. NaN and inf propagate exactly.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(44)
ok = True
# cumsum: small magnitudes so partial sums stay finite/meaningful
s2 = (rng.standard_normal((4096, 256)) * 0.1).astype(np.float16)
s3 = (rng.standard_normal((64, 256, 64)) * 0.1).astype(np.float16)
for arr, ax in [(s2, -1), (s2, 0), (s3, 1)]:
    r = fnp.cumsum(arr, axis=ax); e = np.cumsum(arr, axis=ax)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.view(np.uint16).tobytes() == e.view(np.uint16).tobytes()
# cumprod: values near 1.0 so the running product stays O(1) (not all-underflow-to-0)
p2 = (1.0 + rng.standard_normal((4096, 256)) * 0.03).astype(np.float16)
p3 = (1.0 + rng.standard_normal((64, 256, 64)) * 0.03).astype(np.float16)
for arr, ax in [(p2, -1), (p2, 0), (p3, 1)]:
    r = fnp.cumprod(arr, axis=ax); e = np.cumprod(arr, axis=ax)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.view(np.uint16).tobytes() == e.view(np.uint16).tobytes()
# NaN propagation: once a NaN appears the rest of the lane is NaN. Compare NaN-positionally + finite-exact.
mn = s2.copy(); mn[3, 7] = np.float16(np.nan)
rf = fnp.cumsum(mn, axis=-1); en = np.cumsum(mn, axis=-1)
both_nan = np.isnan(rf) & np.isnan(en)
fin_eq = (~np.isnan(rf)) & (~np.isnan(en)) & (rf.view(np.uint16) == en.view(np.uint16))
ok = ok and bool((both_nan | fin_eq).all())
# overflow to +inf (deterministic 0x7c00): 1000 per step overflows f16 max (65504) mid-lane
mb = (np.ones((4096, 256)) * 1000).astype(np.float16)
ok = ok and fnp.cumsum(mb, axis=-1).view(np.uint16).tobytes() == np.cumsum(mb, axis=-1).view(np.uint16).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 axis cumsum/cumprod must be bit-identical to numpy (last/axis0/middle + NaN/inf): {result}"
    );
    Ok(())
}

#[test]
fn f16_axis_nancumsum_nancumprod_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy has no f16 ALU; np.nancumsum/np.nancumprod of float16 widens f16->f32, replaces NaN with
    // the identity (0 for sum, 1 for prod), accumulates, and narrows back to f16 each step. The native
    // uint16-view per-lane scan with the same skip-NaN/identity rule, parallel across independent
    // lanes, must be byte-identical on the LAST axis, AXIS 0, and a MIDDLE axis (incl an all-NaN lane,
    // which nancumsum turns into all-zeros). Deterministic -> no defers.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(46)
ok = True
s2 = (rng.standard_normal((4096, 256)) * 0.1).astype(np.float16); s2[::7, ::13] = np.float16(np.nan)
s3 = (rng.standard_normal((64, 256, 64)) * 0.1).astype(np.float16); s3[::3, ::11, ::5] = np.float16(np.nan)
for arr, ax in [(s2, -1), (s2, 0), (s3, 1)]:
    r = fnp.nancumsum(arr, axis=ax); e = np.nancumsum(arr, axis=ax)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.view(np.uint16).tobytes() == e.view(np.uint16).tobytes()
p2 = (1.0 + rng.standard_normal((4096, 256)) * 0.03).astype(np.float16); p2[::7, ::13] = np.float16(np.nan)
p3 = (1.0 + rng.standard_normal((64, 256, 64)) * 0.03).astype(np.float16); p3[::3, ::11, ::5] = np.float16(np.nan)
for arr, ax in [(p2, -1), (p2, 0), (p3, 1)]:
    r = fnp.nancumprod(arr, axis=ax); e = np.nancumprod(arr, axis=ax)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.view(np.uint16).tobytes() == e.view(np.uint16).tobytes()
# all-NaN lane -> nancumsum is all-zeros (identity carries); must match
mall = s2.copy(); mall[5, :] = np.float16(np.nan)
ok = ok and fnp.nancumsum(mall, axis=-1).view(np.uint16).tobytes() == np.nancumsum(mall, axis=-1).view(np.uint16).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 axis nancumsum/nancumprod must be bit-identical to numpy (last/axis0/middle + all-NaN): {result}"
    );
    Ok(())
}

#[test]
fn cumsum_cumprod_axis0_large_2d_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy runs cumsum/cumprod single-threaded for every dtype. fnp's per-axis scan kernels
    // parallelize across OUTER blocks, but AXIS 0 has outer==1, so it ran SERIAL. The new transpose
    // column-parallel path (outer==1, inner>=2, total>=1<<18) must be byte-identical to numpy for
    // f64/f32/int cumsum+cumprod AND f64 nancumsum/nancumprod (skip_nan / -0.0 semantics preserved).
    let script = fnp_script(
        r#"
rng = np.random.default_rng(45)
ok = True
shp = (512, 512)  # 262144 == 1<<18, engages the transpose path; axis-0 -> outer==1, inner==512
for dt in (np.float64, np.float32):
    a = (rng.standard_normal(shp) * 0.5).astype(dt)
    ok = ok and fnp.cumsum(a, axis=0).dtype == np.cumsum(a, axis=0).dtype and fnp.cumsum(a, axis=0).tobytes() == np.cumsum(a, axis=0).tobytes()
    p = (1.0 + rng.standard_normal(shp) * 0.01).astype(dt)
    ok = ok and fnp.cumprod(p, axis=0).tobytes() == np.cumprod(p, axis=0).tobytes()
for dt in (np.int64, np.int32, np.uint64, np.uint32):
    ai = rng.integers(-1000, 1000, shp).astype(dt)
    ok = ok and fnp.cumsum(ai, axis=0).dtype == np.cumsum(ai, axis=0).dtype and fnp.cumsum(ai, axis=0).tobytes() == np.cumsum(ai, axis=0).tobytes()
    pi = rng.integers(0, 4, shp).astype(dt)
    ok = ok and fnp.cumprod(pi, axis=0).tobytes() == np.cumprod(pi, axis=0).tobytes()
# f64 nancumsum/nancumprod axis-0 (skip_nan path) with sparse NaN, not all-NaN columns
an = (rng.standard_normal(shp) * 0.5); an[::97, ::13] = np.nan
ok = ok and fnp.nancumsum(an, axis=0).tobytes() == np.nancumsum(an, axis=0).tobytes()
ap = (1.0 + an * 0.01)
ok = ok and fnp.nancumprod(ap, axis=0).tobytes() == np.nancumprod(ap, axis=0).tobytes()
# -0.0 first-row preservation (plain cumsum, no skip): a column starting with -0.0
z = np.zeros(shp); z[0, :] = -0.0; z[1:, :] = rng.standard_normal((shp[0]-1, shp[1])) * 0.5
ok = ok and fnp.cumsum(z, axis=0).tobytes() == np.cumsum(z, axis=0).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native axis-0 transpose-parallel cumsum/cumprod must be bit-identical to numpy (f64/f32/int + nan + -0.0): {result}"
    );
    Ok(())
}

#[test]
fn f16_modf_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy has no f16 ALU; np.modf(f16) widens, splits into (trunc(x), x-trunc(x) signed), narrows
    // both. The native parallel uint16-view split must be byte-identical for finite inputs (native)
    // and delegate inf/nan (numpy's special-value/warning surface) while still matching byte-for-byte.
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(49)
ok = True
n = (1 << 20) + 257
a = (rng.standard_normal(n) * 100).astype(np.float16)  # finite, fractional -> native path
fr, ip = fnp.modf(a); efr, eip = np.modf(a)
ok = ok and fr.dtype == efr.dtype and ip.dtype == eip.dtype
ok = ok and fr.view(np.uint16).tobytes() == efr.view(np.uint16).tobytes()
ok = ok and ip.view(np.uint16).tobytes() == eip.view(np.uint16).tobytes()
# negative + -0.0 / integer (-0.0 sign of frac) handling
a2 = (-(rng.standard_normal(n) * 50)).astype(np.float16); a2[0] = np.float16(-0.0); a2[1] = np.float16(-2.0)
fr2, ip2 = fnp.modf(a2); efr2, eip2 = np.modf(a2)
ok = ok and fr2.view(np.uint16).tobytes() == efr2.view(np.uint16).tobytes()
ok = ok and ip2.view(np.uint16).tobytes() == eip2.view(np.uint16).tobytes()
# inf/nan present -> defer to numpy, still byte-identical (NaN-positional)
an = a.copy(); an[3] = np.float16(np.inf); an[5] = np.float16(np.nan); an[7] = np.float16(-np.inf)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    frn, ipn = fnp.modf(an); efrn, eipn = np.modf(an)
ok = ok and bool(((frn.view(np.uint16) == efrn.view(np.uint16)) | (np.isnan(frn) & np.isnan(efrn))).all())
ok = ok and bool(((ipn.view(np.uint16) == eipn.view(np.uint16)) | (np.isnan(ipn) & np.isnan(eipn))).all())
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 modf must be byte-identical to numpy (finite native + inf/nan defer + signs): {result}"
    );
    Ok(())
}

#[test]
fn f16_reciprocal_full_domain_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy has no f16 ALU; np.reciprocal(f16) widens f16->f32, divides 1/x, narrows. f32 division is
    // IEEE correctly-rounded (no libm), so narrow(1/widen) is byte-identical to numpy over the ENTIRE
    // f16 domain. The native path defers any element whose f16 reciprocal overflows (|x| < 1/65504 ->
    // "overflow") or x==0 (-> "divide by zero") so numpy's warning surface is reproduced.
    let script = fnp_script(
        r#"
import warnings
ok = True
allf16 = np.arange(65536, dtype=np.uint16).view(np.float16)
af = allf16.astype(np.float32)
# NATIVE path, EXHAUSTIVE: every f16 whose reciprocal does NOT overflow (finite, |x| safely above
# 1/65504) tiled past the 1<<20 gate -> the kernel runs (no defer) and must be byte-identical.
safe = allf16[np.isfinite(allf16) & (np.abs(af) >= (1.0 / 65504.0) * 1.01)]
engx = np.tile(safe, (1 << 20) // safe.size + 2)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rx = fnp.reciprocal(engx); ex = np.reciprocal(engx)
ok = ok and rx.dtype == ex.dtype and rx.view(np.uint16).tobytes() == ex.view(np.uint16).tobytes()
# DEFER path: the full f16 domain contains zeros/tiny (overflow) -> whole call delegates to numpy;
# inf->0, nan->nan. Must stay byte-identical (NaN-positional).
full = np.tile(allf16, (1 << 20) // 65536 + 2)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rf = fnp.reciprocal(full); ef = np.reciprocal(full)
both_nan = np.isnan(rf) & np.isnan(ef)
ok = ok and bool((both_nan | (rf.view(np.uint16) == ef.view(np.uint16))).all())
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 reciprocal must be bit-identical to numpy over the full domain (native + defer): {result}"
    );
    Ok(())
}

#[test]
fn f16_predicate_isnan_isinf_isfinite_signbit_full_domain_matches_numpy() -> Result<(), String> {
    // numpy widens f16 to f32 for isnan/isinf/isfinite/signbit; the native parallel uint16
    // bit-check must produce the identical bool array over the ENTIRE f16 domain (all 65536
    // patterns, tiled past the 1<<20 gate).
    let script = fnp_script(
        r#"
patterns = np.arange(65536, dtype=np.uint16).view(np.float16)
x = np.tile(patterns, ((1 << 20) // patterns.size) + 2)
ok = True
for fnp_op, np_op in [(fnp.isnan, np.isnan), (fnp.isinf, np.isinf),
                      (fnp.isfinite, np.isfinite), (fnp.signbit, np.signbit)]:
    r = fnp_op(x); e = np_op(x)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# 2-D shape preserved
x2 = np.tile(patterns, (1 << 20) // patterns.size).reshape(-1, patterns.size)
ok = ok and fnp.isnan(x2).tobytes() == np.isnan(x2).tobytes()
ok = ok and fnp.isnan(x2).shape == np.isnan(x2).shape
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 isnan/isinf/isfinite/signbit must match numpy over the full domain: {result}"
    );
    Ok(())
}

#[test]
fn f16_ordered_comparison_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy widens f16 to f32 for ordered comparisons (compute-bound). The native parallel
    // widen-compare must produce the identical bool array, incl NaN (all ordered comparisons
    // with NaN are False), inf, and signed zeros, above the 1<<20 gate.
    let script = fnp_script(
        r#"
n = (1 << 20) + 257
rng = np.random.default_rng(29)
a = rng.standard_normal(n).astype(np.float16)
b = rng.standard_normal(n).astype(np.float16)
# force some equal elements + specials
b[: n // 4] = a[: n // 4]
av = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, np.nan, 1.0], dtype=np.float16)
bv = np.array([1.0, np.inf, -np.inf, -0.0, 0.0, np.nan, np.nan], dtype=np.float16)
a[:7] = av; b[:7] = bv
ok = True
for fnp_op, np_op in [(fnp.greater, np.greater), (fnp.less, np.less),
                      (fnp.greater_equal, np.greater_equal), (fnp.less_equal, np.less_equal)]:
    r = fnp_op(a, b); e = np_op(a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# operator forms route through the same ufuncs
ok = ok and (a > b).tobytes() == np.greater(a, b).tobytes()
ok = ok and (a <= b).tobytes() == np.less_equal(a, b).tobytes()
# 2-D shape preserved
a2 = a[:1 << 20].reshape(1024, 1024); b2 = b[:1 << 20].reshape(1024, 1024)
ok = ok and fnp.greater(a2, b2).tobytes() == np.greater(a2, b2).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 ordered comparisons must be bit-identical to numpy incl NaN: {result}"
    );
    Ok(())
}

#[test]
fn f16_binary_maximum_minimum_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy has no native f16 ALU; maximum/minimum widen->f32->op->narrow (compute-bound). The
    // native parallel kernel must be byte-identical incl NaN-bit propagation (LHS first), signed
    // zeros (returns LHS on equal), and inf, above the 1<<20 gate.
    let script = fnp_script(
        r#"
n = (1 << 20) + 257
rng = np.random.default_rng(23)
a = rng.standard_normal(n).astype(np.float16)
b = rng.standard_normal(n).astype(np.float16)
# seed special pairs (canonical + non-canonical nan, inf, signed zeros)
av = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -0.0], dtype=np.float16)
bv = np.array([1.0, -np.inf, np.inf, -0.0, 0.0, np.nan, 0.0], dtype=np.float16)
a[:7] = av; b[:7] = bv
# non-canonical nan bit patterns in the LHS
a.view(np.uint16)[7] = 0x7e01; b[7] = np.float16(5.0)
a.view(np.uint16)[8] = np.uint16(0x0001); b.view(np.uint16)[8] = np.uint16(0x7e05)  # b non-canonical nan
ok = True
for fnp_op, np_op in [(fnp.maximum, np.maximum), (fnp.minimum, np.minimum)]:
    r = fnp_op(a, b); e = np_op(a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# 2-D shape preserved
a2 = a[:1 << 20].reshape(1024, 1024); b2 = b[:1 << 20].reshape(1024, 1024)
ok = ok and fnp.maximum(a2, b2).tobytes() == np.maximum(a2, b2).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 maximum/minimum must be bit-identical to numpy incl NaN-bit propagation: {result}"
    );
    Ok(())
}

#[test]
fn f16_unary_sqrt_square_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy has no native f16 ALU; sqrt/square widen->f32->op->narrow (compute-bound). The native
    // parallel widen path is bit-exact for the warning-free common case (sqrt of non-negatives,
    // square of |x|<256) and DEFERS to numpy when a warning would fire (sqrt of a negative ->
    // invalid; square overflow). Both kernel and defer paths must be byte-identical to numpy.
    let script = fnp_script(
        r#"
import warnings
# all non-negative f16 bit patterns for sqrt: positive finite + +inf (skip nan/negatives)
allp = np.arange(65536, dtype=np.uint16).view(np.float16)
pos = allp[(~np.isnan(allp)) & (allp >= np.float16(0))]
reps = ((1 << 20) // pos.size) + 2
xs = np.tile(pos, reps)
ok = True
r = fnp.sqrt(xs); e = np.sqrt(xs)
ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# square: values with |x| < 256 (warning-free), tiled past the gate; include negatives and -0.0
mod = allp[(~np.isnan(allp)) & (np.abs(allp) < np.float16(256))]
xsq = np.tile(mod, ((1 << 20) // mod.size) + 2)
r = fnp.square(xsq); e = np.square(xsq)
ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# DEFER paths still match numpy exactly (suppress the expected warnings)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    neg = np.tile(allp[~np.isnan(allp)], 17)          # contains negatives -> sqrt defers
    ok = ok and fnp.sqrt(neg).tobytes() == np.sqrt(neg).tobytes()
    big = (np.tile(np.array([300.0, 1.0, -400.0, 2.0], dtype=np.float16), (1 << 19)))  # |x|>=256 -> square defers
    ok = ok and fnp.square(big).tobytes() == np.square(big).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 sqrt/square must be bit-identical to numpy (kernel + defer paths): {result}"
    );
    Ok(())
}

#[test]
fn f16_unary_floor_ceil_trunc_rint_parallel_full_domain_bit_exact_matches_numpy() -> Result<(), String>
{
    // numpy has no native f16 ALU: floor/ceil/trunc/rint widen->f32->op->narrow (compute-bound).
    // The native parallel widen path must be byte-identical to numpy over EVERY f16 bit pattern
    // (all 65536, incl. every nan/inf/subnormal/-0.0), tiled past the 1<<20 gate to engage the
    // parallel kernel. Also a 2-D case to exercise the shape-preserving reshape.
    let script = fnp_script(
        r#"
patterns = np.arange(65536, dtype=np.uint16).view(np.float16)
reps = ((1 << 20) // patterns.size) + 2  # > gate
x = np.tile(patterns, reps)
ok = True
for fnp_op, np_op in [(fnp.floor, np.floor), (fnp.ceil, np.ceil),
                      (fnp.trunc, np.trunc), (fnp.rint, np.rint)]:
    r = fnp_op(x); e = np_op(x)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# 2-D shape preserved
x2 = np.tile(patterns, (1 << 20) // patterns.size).reshape(-1, patterns.size)
ok = ok and fnp.floor(x2).tobytes() == np.floor(x2).tobytes()
ok = ok and fnp.floor(x2).shape == np.floor(x2).shape
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 floor/ceil/trunc/rint must be bit-identical to numpy over the full domain: {result}"
    );
    Ok(())
}
