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
