//! Conformance tests for ufunc edge cases (empty arrays, identity values).
//!
//! Tests verify PyUFunc reduce/accumulate/outer behaviors match NumPy on
//! edge cases like empty arrays and identity element semantics.

use std::process::Command;

fn numpy_oracle(script: &str) -> Result<String, String> {
    let py = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let output = Command::new(&py)
        .args(["-c", script])
        .output()
        .map_err(|error| format!("{py} should be available: {error}\nScript: {script}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NumPy oracle failed: {stderr}\nScript: {script}"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

mod support;
use support::{fnp_script, fnp_script_with};

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
fn f32_searchsorted_small_native_path_is_nan_and_signed_zero_exact() -> Result<(), String> {
    // This is a planted negative for the f32 dispatch path: the old implementation
    // pre-scanned either buffer for NaN and then delegated to np.searchsorted. Poison
    // that delegate after capturing NumPy's expected bytes, so this only passes when
    // the small f32 array-needle route itself implements NumPy's NaN-last ordering.
    // The finite case carries both signed zeros and duplicate ties; the NaN-haystack
    // and NaN-needle cases exercise the ordering predicate that a plain PartialOrd
    // binary search gets wrong.
    let script = fnp_script(
        r#"
finite_a = np.array([-np.inf, -2.0, -0.0, 0.0, 0.0, 1.0, np.inf], dtype=np.float32)
finite_v = np.array([-0.0, 0.0, 0.5, np.inf], dtype=np.float32)
nan_a = np.array([-np.inf, -1.0, -0.0, 0.0, 1.0, np.inf, np.nan, np.nan], dtype=np.float32)
nan_v = np.array([-0.0, 0.0, 0.5, np.inf, np.nan], dtype=np.float32)
cases = ((finite_a, finite_v), (nan_a, nan_v))
expected = [
    (np.searchsorted(a, v, side="left").copy(), np.searchsorted(a, v, side="right").copy())
    for a, v in cases
]
original = np.searchsorted
def delegated_searchsorted(*args, **kwargs):
    raise AssertionError("f32 native array-needle path delegated to numpy.searchsorted")
np.searchsorted = delegated_searchsorted
try:
    actual = [
        (fnp.searchsorted(a, v, side="left"), fnp.searchsorted(a, v, side="right"))
        for a, v in cases
    ]
finally:
    np.searchsorted = original
ok = all(
    left.dtype == expected_left.dtype
    and right.dtype == expected_right.dtype
    and left.shape == expected_left.shape
    and right.shape == expected_right.shape
    and left.tobytes() == expected_left.tobytes()
    and right.tobytes() == expected_right.tobytes()
    for (left, right), (expected_left, expected_right) in zip(actual, expected)
)
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "small f32 searchsorted must keep signed-zero and NaN ordering native and exact: {result}"
    );
    Ok(())
}

#[test]
fn f16_searchsorted_cumulative_table_matches_numpy_and_defers_edge_inputs() -> Result<(), String> {
    // The native f16 table path is only valid for a C-contiguous sorted finite
    // haystack. Build that order from the bit-order key instead of float16 sort
    // so this test remains independent of the fleet's known fp16 sort defect.
    // Queries cover every finite payload, including subnormals, infinities, and
    // both signed zeros; NaN and noncontiguous cases exercise the widening defer.
    let script = fnp_script(
        r#"
all_bits = np.arange(65536, dtype=np.uint16)
finite_bits = all_bits[(all_bits & np.uint16(0x7fff)) <= np.uint16(0x7c00)]
magnitude = finite_bits & np.uint16(0x7fff)
keys = np.where(
    magnitude == 0,
    np.uint16(0x8000),
    np.where((finite_bits & np.uint16(0x8000)) == 0, finite_bits | np.uint16(0x8000), ~finite_bits),
)
a = finite_bits[np.argsort(keys, kind='stable')].view(np.float16)
n = 129 * 131  # 16,899: above the table gate and exactly reshapeable below
v = np.resize(finite_bits, n).view(np.float16).reshape(129, -1)
ok = True
for side in ('left', 'right'):
    got = fnp.searchsorted(a, v, side=side)
    expected = np.searchsorted(a, v, side=side)
    ok = ok and got.dtype == expected.dtype and got.shape == expected.shape and got.tobytes() == expected.tobytes()

# C-contiguity is part of the table admission contract; the established f32
# widening path must still preserve the exact result for strided inputs.
strided_a = a[::2]
strided_v = v.ravel()[::2]
for side in ('left', 'right'):
    got = fnp.searchsorted(strided_a, strided_v, side=side)
    expected = np.searchsorted(strided_a, strided_v, side=side)
    ok = ok and got.dtype == expected.dtype and got.shape == expected.shape and got.tobytes() == expected.tobytes()

# Numeric-order verification is also an admission requirement. NumPy permits
# an unsorted haystack but its observable answer is not a prefix-count table;
# this must take the established fallback rather than silently table-reordering.
unsorted_a = a.copy()
unsorted_a[[17, 18]] = unsorted_a[[18, 17]]
for side in ('left', 'right'):
    got = fnp.searchsorted(unsorted_a, v, side=side)
    expected = np.searchsorted(unsorted_a, v, side=side)
    ok = ok and got.dtype == expected.dtype and got.shape == expected.shape and got.tobytes() == expected.tobytes()

# `a` is required to be 1-D by NumPy. A C-contiguous matrix exposes a flat
# buffer too, so the table path must defer before flattening its haystack.
matrix_a = a.reshape(2, -1)
for side in ('left', 'right'):
    try:
        np.searchsorted(matrix_a, v, side=side)
    except Exception as exc:
        expected_error = type(exc).__name__
    else:
        expected_error = None
    try:
        fnp.searchsorted(matrix_a, v, side=side)
    except Exception as exc:
        got_error = type(exc).__name__
    else:
        got_error = None
    ok = ok and expected_error == 'ValueError' and got_error == expected_error

# NaN ordering is deliberately owned by NumPy/the widening fallback rather than
# the finite table. Distinct payloads cover both haystack and query deferral.
a_nan = np.concatenate((a, np.array([np.uint16(0x7e01)], dtype=np.uint16).view(np.float16)))
v_nan = v.copy()
v_nan.view(np.uint16).ravel()[7] = np.uint16(0xfe11)
for side in ('left', 'right'):
    got = fnp.searchsorted(a_nan, v, side=side)
    expected = np.searchsorted(a_nan, v, side=side)
    ok = ok and got.dtype == expected.dtype and got.shape == expected.shape and got.tobytes() == expected.tobytes()

    got = fnp.searchsorted(a_nan, v_nan, side=side)
    expected = np.searchsorted(a_nan, v_nan, side=side)
    ok = ok and got.dtype == expected.dtype and got.shape == expected.shape and got.tobytes() == expected.tobytes()
print(bool(ok))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "f16 cumulative-table searchsorted and its edge deferrals must match numpy: {result}"
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
fn complex_cumsum_lastaxis_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy's complex cumsum is a single-threaded sequential dependency chain; the real & imaginary
    // parts accumulate independently, so per contiguous lane it is two interleaved sequential real
    // prefix sums. The native parallel-across-lanes scan must be byte-identical for complex128 and
    // complex64 on the last axis (2-D + 3-D), incl NaN/inf propagation; axis-0 delegates to numpy.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(52)
ok = True
for dt, rname in [(np.complex128, np.float64), (np.complex64, np.float32)]:
    m = (rng.standard_normal((4096, 256)) + 1j * rng.standard_normal((4096, 256))).astype(dt)
    for ax in (1, -1):
        r = fnp.cumsum(m, axis=ax); e = np.cumsum(m, axis=ax)
        ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.view(rname).tobytes() == e.view(rname).tobytes()
    m3 = (rng.standard_normal((64, 64, 256)) + 1j * rng.standard_normal((64, 64, 256))).astype(dt)
    r3 = fnp.cumsum(m3, axis=2); e3 = np.cumsum(m3, axis=2)
    ok = ok and r3.shape == e3.shape and r3.view(rname).tobytes() == e3.view(rname).tobytes()
    # NaN/inf propagation along a lane
    mn = m.copy(); mn[3, 7] = complex(np.nan, 1.0); mn[5, 0] = complex(np.inf, np.nan)
    rn = fnp.cumsum(mn, axis=1); en = np.cumsum(mn, axis=1)
    ok = ok and bool(((rn.view(rname) == en.view(rname)) | (np.isnan(rn.view(rname)) & np.isnan(en.view(rname)))).all())
# axis-0 (non-last) -> delegate, still byte-identical
ma0 = (rng.standard_normal((256, 4096)) + 1j * rng.standard_normal((256, 4096))).astype(np.complex128)
ok = ok and fnp.cumsum(ma0, axis=0).view(np.float64).tobytes() == np.cumsum(ma0, axis=0).view(np.float64).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native complex last-axis cumsum must be byte-identical to numpy (c128/c64 + 3-D + NaN/inf + axis0 defer): {result}"
    );
    Ok(())
}

#[test]
fn complex_cumprod_lastaxis_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy's complex cumprod is multiply.accumulate on complex: a single-threaded sequential
    // dependency chain carrying one complex accumulator per lane via the naive cmul. The native
    // parallel-across-lanes scan must be byte-identical for complex128 and complex64 on the last axis
    // (2-D + 3-D), incl overflow->inf and NaN/inf propagation; axis-0 (non-last) delegates to numpy.
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(53)
ok = True
for dt, rname in [(np.complex128, np.float64), (np.complex64, np.float32)]:
    m = (rng.standard_normal((4096, 256)) + 1j * rng.standard_normal((4096, 256))).astype(dt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ax in (1, -1):
            r = fnp.cumprod(m, axis=ax); e = np.cumprod(m, axis=ax)
            ok = ok and r.dtype == e.dtype and r.shape == e.shape and bool(((r.view(rname) == e.view(rname)) | (np.isnan(r.view(rname)) & np.isnan(e.view(rname)))).all())
        m3 = (rng.standard_normal((64, 64, 256)) + 1j * rng.standard_normal((64, 64, 256))).astype(dt)
        r3 = fnp.cumprod(m3, axis=2); e3 = np.cumprod(m3, axis=2)
        ok = ok and r3.shape == e3.shape and bool(((r3.view(rname) == e3.view(rname)) | (np.isnan(r3.view(rname)) & np.isnan(e3.view(rname)))).all())
        # NaN/inf propagation along a lane
        mn = m.copy(); mn[3, 7] = complex(np.nan, 1.0); mn[5, 0] = complex(np.inf, np.nan); mn[6, 2] = complex(0.0, np.inf)
        rn = fnp.cumprod(mn, axis=1); en = np.cumprod(mn, axis=1)
        ok = ok and bool(((rn.view(rname) == en.view(rname)) | (np.isnan(rn.view(rname)) & np.isnan(en.view(rname)))).all())
# axis-0 (non-last) -> delegate, still byte-identical
ma0 = (rng.standard_normal((256, 4096)) + 1j * rng.standard_normal((256, 4096))).astype(np.complex128)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r0 = fnp.cumprod(ma0, axis=0); e0 = np.cumprod(ma0, axis=0)
ok = ok and bool(((r0.view(np.float64) == e0.view(np.float64)) | (np.isnan(r0.view(np.float64)) & np.isnan(e0.view(np.float64)))).all())
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native complex last-axis cumprod must be byte-identical to numpy (c128/c64 + 3-D + NaN/inf + axis0 defer): {result}"
    );
    Ok(())
}

#[test]
fn complex_nancumulative_lastaxis_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy's complex nancumsum/nancumprod = cum* with every NaN-complex (re OR im NaN) replaced by the
    // identity (0+0j sum / 1+0j prod) on a single-threaded chain. The native per-lane parallel nan-scan
    // must be byte-identical for c128/c64 on the last axis (2-D + 3-D), incl first-element-NaN,
    // (nan,nan), (inf,nan). The 1M-element axis-0 fixture crosses the native
    // gather/scan/scatter threshold for complex128 nancumprod and locks the
    // delegated combinations to the same byte-level behavior.
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(55)
ok = True
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for dt, rname in [(np.complex128, np.float64), (np.complex64, np.float32)]:
        m = (rng.standard_normal((4096, 256)) + 1j * rng.standard_normal((4096, 256))).astype(dt)
        # inject NaN-complex incl first-element of some lanes
        m[0, 0] = complex(np.nan, 1.0); m[3, 0] = complex(2.0, np.nan); m[5, 7] = complex(np.nan, np.nan); m[9, 1] = complex(np.inf, np.nan)
        for fn_fnp, fn_np in [(fnp.nancumsum, np.nancumsum), (fnp.nancumprod, np.nancumprod)]:
            for ax in (1, -1):
                r = fn_fnp(m, axis=ax); e = fn_np(m, axis=ax)
                ok = ok and r.dtype == e.dtype and r.shape == e.shape and bool(((r.view(rname) == e.view(rname)) | (np.isnan(r.view(rname)) & np.isnan(e.view(rname)))).all())
        m3 = (rng.standard_normal((64, 64, 256)) + 1j * rng.standard_normal((64, 64, 256))).astype(dt)
        m3[1, 2, 0] = complex(np.nan, 0.0)
        r3 = fnp.nancumprod(m3, axis=2); e3 = np.nancumprod(m3, axis=2)
        ok = ok and bool(((r3.view(rname) == e3.view(rname)) | (np.isnan(r3.view(rname)) & np.isnan(e3.view(rname)))).all())
        # MIDDLE axis (axis=1, outer>=2): native per-block nan-scan, both ops
        mm = (rng.standard_normal((64, 96, 96)) + 1j * rng.standard_normal((64, 96, 96))).astype(dt)
        mm[3, 0, 5] = complex(np.nan, 1.0); mm[5, 7, 0] = complex(np.inf, np.nan)
        for fn_fnp, fn_np in [(fnp.nancumsum, np.nancumsum), (fnp.nancumprod, np.nancumprod)]:
            rm = fn_fnp(mm, axis=1); em = fn_np(mm, axis=1)
            ok = ok and bool(((rm.view(rname) == em.view(rname)) | (np.isnan(rm.view(rname)) & np.isnan(em.view(rname)))).all())
    # axis-0 (non-last): c128 nancumprod is native at 1M; other combinations delegate
    for dt, uname in [(np.complex128, np.uint64), (np.complex64, np.uint32)]:
        a0 = (rng.standard_normal((256, 4096)) + 1j * rng.standard_normal((256, 4096))).astype(dt)
        a0[0, 9] = complex(np.nan, 1.0)
        a0[7, 3] = complex(np.inf, np.nan)
        for fn_fnp, fn_np in [(fnp.nancumsum, np.nancumsum), (fnp.nancumprod, np.nancumprod)]:
            r0 = fn_fnp(a0, axis=0); e0 = fn_np(a0, axis=0)
            ok = ok and r0.dtype == e0.dtype and r0.shape == e0.shape
            ok = ok and r0.view(uname).tobytes() == e0.view(uname).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "complex nancumsum/nancumprod boundary must be byte-identical to numpy (c128/c64 + 3-D + NaN + axis0): {result}"
    );
    Ok(())
}

#[test]
fn complex_cumulative_midaxis_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy runs a non-last complex cumsum/cumprod strided + single-threaded; each independent outer
    // block is a slab-by-slab scan (cumsum: re/im add; cumprod: naive cmul). The native per-block
    // parallel scan along a MIDDLE axis must be byte-identical for c128/c64, incl NaN/inf; axis-0 and
    // the last axis delegate (last axis handled by the dedicated last-axis paths).
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(54)
ok = True
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for dt, rname in [(np.complex128, np.float64), (np.complex64, np.float32)]:
        m3 = (rng.standard_normal((64, 96, 96)) + 1j * rng.standard_normal((64, 96, 96))).astype(dt)
        for fn_fnp, fn_np in [(fnp.cumsum, np.cumsum), (fnp.cumprod, np.cumprod)]:
            r = fn_fnp(m3, axis=1); e = fn_np(m3, axis=1)
            ok = ok and r.dtype == e.dtype and r.shape == e.shape and bool(((r.view(rname) == e.view(rname)) | (np.isnan(r.view(rname)) & np.isnan(e.view(rname)))).all())
        # NaN/inf along a middle lane
        mn = m3.copy(); mn[3, 7, 5] = complex(np.nan, 1.0); mn[5, 0, 9] = complex(np.inf, np.nan)
        rn = fnp.cumprod(mn, axis=1); en = np.cumprod(mn, axis=1)
        ok = ok and bool(((rn.view(rname) == en.view(rname)) | (np.isnan(rn.view(rname)) & np.isnan(en.view(rname)))).all())
        # 4-D middle axis (axis=2, still a middle axis -> outer>=2)
        m4 = (rng.standard_normal((8, 8, 24, 32)) + 1j * rng.standard_normal((8, 8, 24, 32))).astype(dt)
        r4 = fnp.cumsum(m4, axis=2); e4 = np.cumsum(m4, axis=2)
        ok = ok and bool(((r4.view(rname) == e4.view(rname)) | (np.isnan(r4.view(rname)) & np.isnan(e4.view(rname)))).all())
    # axis-0 (outer==1) -> native gather/scan/scatter column scan, byte-identical (2-D + 3-D, c128/c64, sum+prod)
    for dt, rname in [(np.complex128, np.float64), (np.complex64, np.float32)]:
        a2 = (rng.standard_normal((4096, 256)) + 1j * rng.standard_normal((4096, 256))).astype(dt)
        a3 = (rng.standard_normal((256, 128, 64)) + 1j * rng.standard_normal((256, 128, 64))).astype(dt)
        for arr in (a2, a3):
            for fn_fnp, fn_np in [(fnp.cumsum, np.cumsum), (fnp.cumprod, np.cumprod)]:
                r0 = fn_fnp(arr, axis=0); e0 = fn_np(arr, axis=0)
                ok = ok and r0.dtype == e0.dtype and r0.shape == e0.shape and bool(((r0.view(rname) == e0.view(rname)) | (np.isnan(r0.view(rname)) & np.isnan(e0.view(rname)))).all())
        # NaN/inf down an axis-0 column
        an = a2.copy(); an[7, 3] = complex(np.inf, np.nan); an[0, 9] = complex(np.nan, 1.0)
        rn = fnp.cumprod(an, axis=0); en = np.cumprod(an, axis=0)
        ok = ok and bool(((rn.view(rname) == en.view(rname)) | (np.isnan(rn.view(rname)) & np.isnan(en.view(rname)))).all())
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native complex middle-axis cumsum/cumprod must be byte-identical to numpy (c128/c64 + 3-D/4-D + NaN/inf + axis0 defer): {result}"
    );
    Ok(())
}

#[test]
fn f16_ldexp_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // np.ldexp(float16, int32) = x * 2^e. numpy has no f16 ALU -> widens, scalbnf, narrows single-
    // threaded. The native widen->exact-pow2-scale->narrow (one rounding) must be byte-identical over
    // the full f16 domain x exponents incl overflow(->inf)/underflow(->0/subnormal); 0/inf/nan identity.
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(51)
ok = True
n = (1 << 20) + 257
a = (rng.standard_normal(n)).astype(np.float16)
e = rng.integers(-30, 20, n).astype(np.int32)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = fnp.ldexp(a, e); ex = np.ldexp(a, e)
ok = ok and r.dtype == ex.dtype and r.shape == ex.shape
ok = ok and r.view(np.uint16).tobytes() == ex.view(np.uint16).tobytes()
# EXHAUSTIVE: every f16 value x a fixed exponent (incl overflow/underflow/inf/nan/zero)
allf16 = np.arange(65536, dtype=np.uint16).view(np.float16)
for ev in (-40, -14, -1, 0, 1, 14, 40):
    full = np.tile(allf16, (1 << 20) // 65536 + 2)
    efull = np.full(full.size, ev, dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf = fnp.ldexp(full, efull); eff = np.ldexp(full, efull)
    ok = ok and bool(((rf.view(np.uint16) == eff.view(np.uint16)) | (np.isnan(rf) & np.isnan(eff))).all())
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 ldexp must be byte-identical to numpy (full domain x exponents incl over/underflow): {result}"
    );
    Ok(())
}

#[test]
fn f16_frexp_full_domain_bit_exact_matches_numpy() -> Result<(), String> {
    // np.frexp(f16) -> (mantissa f16, exponent int32). It is an EXACT bit/exponent decomposition
    // (no rounding, no libm): mantissa = x/2^e in [0.5,1) is exactly representable in f16. numpy widens
    // f16->f32 single-threaded; the native parallel split must be byte-identical over the ENTIRE f16
    // domain (finite native + zero/inf/nan handled by frexp_one, no defer).
    let script = fnp_script(
        r#"
ok = True
allf16 = np.arange(65536, dtype=np.uint16).view(np.float16)
# EXHAUSTIVE over all finite f16 tiled past the parallel gate (1<<19)
finite = allf16[np.isfinite(allf16)]
eng = np.tile(finite, (1 << 20) // finite.size + 2)
m, e = fnp.frexp(eng); em, ee = np.frexp(eng)
ok = ok and m.dtype == em.dtype and e.dtype == ee.dtype
ok = ok and m.view(np.uint16).tobytes() == em.view(np.uint16).tobytes()
ok = ok and e.tobytes() == ee.tobytes()
# FULL domain incl zero/inf/nan (frexp_one handles them inline -> no defer)
full = np.tile(allf16, (1 << 19) // 65536 + 2)
mf, ef = fnp.frexp(full); emf, eef = np.frexp(full)
ok = ok and bool(((mf.view(np.uint16) == emf.view(np.uint16)) | (np.isnan(mf) & np.isnan(emf))).all())
ok = ok and ef.tobytes() == eef.tobytes()
# 2-D shape preserved (both outputs)
d2 = (np.random.default_rng(7).standard_normal((724, 724)) * 50).astype(np.float16)
m2, e2 = fnp.frexp(d2); em2, ee2 = np.frexp(d2)
ok = ok and m2.shape == em2.shape and e2.shape == ee2.shape
ok = ok and m2.view(np.uint16).tobytes() == em2.view(np.uint16).tobytes() and e2.tobytes() == ee2.tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native f16 frexp must be byte-identical to numpy over the full domain (mantissa + int32 exponent): {result}"
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
fn f16_unary_floor_ceil_trunc_rint_parallel_full_domain_bit_exact_matches_numpy()
-> Result<(), String> {
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

#[test]
fn zerocopy_binary_output_shape_survives_the_flat_output_helper() -> Result<(), String> {
    // Guards `finish_flat_output`, which every zero-copy route now shares
    // (`deadlock-audit-omyno`). It applies three different rules by rank and a bug in
    // any one of them is invisible to a value-only check: 1-D returns the flat buffer
    // UNCHANGED (skipping a reshape to the shape it already has, worth 200 ns per call
    // per `deadlock-audit-6twge`), 0-D reshapes to `()` and then extracts a SCALAR via
    // `get_item(())`, and everything else reshapes to the target shape.
    //
    // So this asserts SHAPE, NDIM, DTYPE and TYPE against numpy, not just values -
    // returning a flat 1-D array where numpy returns 2-D is exactly the regression the
    // 1-D skip could cause if it were ever applied to a rank it does not hold for, and
    // every element would still compare equal.
    let script = fnp_script(
        r#"
verdicts = []
rng = np.random.default_rng(20260816)

def check(name, a, b):
    for op in ("divide", "multiply", "add", "subtract", "maximum", "minimum", "power", "floor_divide"):
        ours = getattr(fnp, op)(a, b)
        theirs = getattr(np, op)(a, b)
        oa, ta = np.asarray(ours), np.asarray(theirs)
        if oa.shape != ta.shape:
            verdicts.append(f"FAIL {name} {op} shape {oa.shape} != {ta.shape}")
        if oa.ndim != ta.ndim:
            verdicts.append(f"FAIL {name} {op} ndim {oa.ndim} != {ta.ndim}")
        if str(oa.dtype) != str(ta.dtype):
            verdicts.append(f"FAIL {name} {op} dtype {oa.dtype} != {ta.dtype}")
        # a 0-D route must hand back a SCALAR, not a 0-D array, exactly as numpy does
        if type(ours).__name__ != type(theirs).__name__:
            verdicts.append(f"FAIL {name} {op} type {type(ours).__name__} != {type(theirs).__name__}")
        if oa.tobytes() != ta.tobytes():
            verdicts.append(f"FAIL {name} {op} bytes")

# rank 1 - the arm that skips the reshape
n = 1 << 12
a1 = 1.0 + (np.arange(n) % 1000) / 1000.0
b1 = 1.25 + (np.arange(n) % 997) / 997.0
check("1-D", a1, b1)

# rank 1 large enough to cross the parallel gates, so the skip is exercised on the
# rayon arm too (FLOAT_POWER_PARALLEL_MIN_LEN is 16_384; Max/Min/Div use 1<<21)
nbig = (1 << 21) + 3
abig = 1.0 + (np.arange(nbig) % 1000) / 1000.0
bbig = 1.25 + (np.arange(nbig) % 997) / 997.0
check("1-D above the parallel gate", abig, bbig)

# rank 2 and rank 3 - the arm that must still reshape
check("2-D", a1.reshape(64, 64), b1.reshape(64, 64))
check("3-D", a1.reshape(16, 16, 16), b1.reshape(16, 16, 16))

# rank 0 - the arm that reshapes to () and extracts a scalar
check("0-D", np.float64(3.5), np.float64(1.25))

# non-contiguous and broadcasting forms defer, but must still agree on shape
check("strided", a1[::3], b1[::3])
check("broadcast row", a1.reshape(64, 64), b1.reshape(64, 64)[0])
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "zero-copy binary routes must preserve shape/ndim/dtype/type through finish_flat_output: {result}"
    );
    Ok(())
}

/// Every NumPy ufunc name must behave as a ufunc OBJECT, not a bare function: 105 of
/// NumPy's 106 names used to fail `isinstance(x, np.ufunc)`, ~83 had no
/// `.reduce`/`.accumulate`/`.outer`/`.at`, and `out=`/`dtype=`/`subok=` raised TypeError on
/// the plain-function ones (deadlock-audit-rc0923-epic-71qy3.5). Iterates the LIVE numpy's
/// ufunc names so new ones are covered automatically; compares protocol attributes, method
/// results, keyword calls, and plain-call values against numpy. It does NOT prove the plain
/// call reaches fnp's native kernel rather than numpy's ufunc: equal values cannot tell the
/// two routes apart.
#[test]
fn every_numpy_ufunc_name_is_a_ufunc_object_with_numpy_protocol() {
    // Registered in sys.modules: fnp's ufuncs pickle by reference to `fnp_python.<name>`, as
    // numpy's pickle to `numpy.<name>`, so the module must be importable as it is when installed.
    let script = fnp_script_with(
        "import sys\n",
        true,
        r#"
import pickle, warnings
warnings.simplefilter("ignore")
names = sorted(n for n in dir(np) if not n.startswith("_") and isinstance(getattr(np, n), np.ufunc))
bad = []
x = np.linspace(0.25, 2.0, 8)
ints = np.arange(1, 9, dtype=np.int64)
def check(n, f, g):
    if not isinstance(f, np.ufunc):
        bad.append(f"{n}: not isinstance np.ufunc"); return
    for attr in ("nin", "nout", "nargs", "ntypes", "types", "identity", "signature", "__name__"):
        if getattr(f, attr) != getattr(g, attr):
            bad.append(f"{n}: .{attr} differs")
    if g.nin == 2 and g.nout == 1:
        # `reduction=` was missing from a hand-written forwarder on the native class.
        outcomes = []
        for u in (f, g):
            try:
                outcomes.append(u.resolve_dtypes((None, np.dtype("f8"), None), reduction=True))
            except Exception as e:
                outcomes.append(type(e).__name__)
        if outcomes[0] != outcomes[1]:
            bad.append(f"{n}.resolve_dtypes(reduction=True): {outcomes[0]} vs {outcomes[1]}")
    if repr(f) != repr(g):
        bad.append(f"{n}: repr {repr(f)!r}")
    # By reference, like numpy's own: the round trip returns the SAME object.
    if pickle.loads(pickle.dumps(f)) is not f:
        bad.append(f"{n}: pickle does not round-trip to the same object")
    if g.nin == 2 and g.nout == 1 and g.signature is None and "d" in "".join(g.types):
        # Compare OUTCOMES: numpy itself raises for some methods on float input (equal.reduce,
        # ldexp.outer, ...), and fnp must raise the same exception type there.
        def outcome(call):
            try:
                return ("ok", np.asarray(call()))
            except Exception as e:
                return ("err", type(e).__name__)
        def at_call(u):
            a = np.zeros(4)
            u.at(a, [0, 0, 2], 1.5)
            return a
        for meth, call in (("reduce", lambda u: u.reduce(x)), ("accumulate", lambda u: u.accumulate(x)),
                           ("outer", lambda u: u.outer(x[:3], x[:3])), ("at", at_call)):
            got, want = outcome(lambda: call(f)), outcome(lambda: call(g))
            if got[0] != want[0] or (got[0] == "err" and got[1] != want[1]) or (
                    got[0] == "ok" and not np.array_equal(got[1], want[1], equal_nan=True)):
                bad.append(f"{n}.{meth}: fnp {got[0]} {got[1] if got[0] == 'err' else ''} vs numpy {want[0]} {want[1] if want[0] == 'err' else ''}")
    if g.nin == 1 and g.nout == 1 and g.signature is None and "d->d" in g.types:
        o1, o2 = np.empty_like(x), np.empty_like(x)
        r1, r2 = f(x, out=o1), g(x, out=o2)
        if r1 is not o1 or not np.array_equal(o1, o2, equal_nan=True):
            bad.append(f"{n}(x, out=o): wrong")
        if not np.array_equal(f(x), g(x), equal_nan=True):
            bad.append(f"{n}(x): value differs")
        if np.asarray(f(x, dtype=np.float32)).dtype != np.asarray(g(x, dtype=np.float32)).dtype:
            bad.append(f"{n}(x, dtype=float32): dtype differs")
for n in names:
    # One name raising must not hide the verdicts of the others.
    try:
        check(n, getattr(fnp, n), getattr(np, n))
    except Exception as e:
        bad.append(f"{n}: raised {type(e).__name__}: {e}")
print(len(names))
print("OK" if not bad else " || ".join(bad))
"#
        .to_string(),
    );
    let result = match numpy_oracle(&script) {
        Ok(output) => output,
        Err(error) => panic!("ufunc protocol probe did not run: {error}"),
    };
    let lines: Vec<&str> = result.lines().collect();
    let count: usize = lines.first().and_then(|n| n.parse().ok()).unwrap_or(0);
    assert!(
        count >= 100,
        "expected ~106 numpy ufunc names, got {count}: {result}"
    );
    assert_eq!(
        lines.get(1).copied(),
        Some("OK"),
        "ufunc protocol diverges: {result}"
    );
}

/// The ufunc METHODS, not just their presence (bead .5's acceptance): every ufunc name of the
/// live numpy x eight dtypes (float64/float32/int64/int32/uint8/bool/complex128/timedelta64) x
/// __call__ with out= / where= / dtype= / casting= / broadcasting, reduce (axis 0/1/None,
/// keepdims, initial, where, dtype), accumulate, outer, reduceat and at (array and scalar
/// values), plus nin/nout/nargs/identity/signature/ntypes/types/__name__: fnp must match numpy's
/// result type, dtype, shape and bytes, or raise the same exception type. `where=` without
/// `out=` leaves the unselected outputs uninitialised in numpy, so only selected positions are
/// compared there.
#[test]
fn every_ufunc_method_matches_numpy_results() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(71)
names = sorted(n for n in dir(np) if isinstance(getattr(np, n), np.ufunc))
def operands(dt):
    kind = np.dtype(dt).kind
    if kind == "b":
        return rng.random((5, 6)) < 0.5
    if kind in "iu":
        return rng.integers(1, 9, (5, 6)).astype(dt)
    if kind == "c":
        return (rng.standard_normal((5, 6)) + 1j * rng.standard_normal((5, 6))).astype(dt)
    if kind == "m":
        return rng.integers(-9, 9, (5, 6)).astype("timedelta64[s]")
    return (rng.standard_normal((5, 6)) * 3).astype(dt)
cases = []
def add(name, fn, mask=None):
    cases.append((name, fn, mask))
def into_out(m, call):
    # numpy's own result fixes the out= buffers (one per output); each arm fills a fresh set.
    ref = call(np, None)
    outs = tuple(np.empty_like(np.asarray(r)) for r in (ref if isinstance(ref, tuple) else (ref,)))
    call(m, outs)
    return outs
for n in names:
    for attr in ("nin", "nout", "nargs", "identity", "signature", "ntypes", "types", "__name__"):
        add(f"{n}.{attr}", lambda m, n=n, a=attr: getattr(getattr(m, n), a))
    for dt in ("f8", "f4", "i8", "i4", "u1", "?", "c16", "m8[s]"):
        a, b = operands(dt), operands(dt)
        u = lambda m, n=n: getattr(m, n)
        tag = f"{n} {dt}"
        if getattr(np, n).nin == 1:
            sel = a.real > 0 if a.dtype.kind in "fic" else np.ones(a.shape, bool)
            add(f"{tag} call", lambda m, a=a, u=u: u(m)(a))
            add(f"{tag} call out", lambda m, a=a, u=u: into_out(m, lambda mm, o: u(mm)(a) if o is None else u(mm)(a, out=o)))
            add(f"{tag} call where", lambda m, a=a, u=u, w=sel: u(m)(a, where=w), sel)
            add(f"{tag} call dtype f8", lambda m, a=a, u=u: u(m)(a, dtype="f8"))
            add(f"{tag} at", lambda m, a=a, u=u: (lambda x: (u(m).at(x, [0, 2, 0]), x)[1])(a.copy().ravel()))
        else:
            add(f"{tag} call", lambda m, a=a, b=b, u=u: u(m)(a, b))
            add(f"{tag} call broadcast", lambda m, a=a, b=b, u=u: u(m)(a, b[:1]))
            add(f"{tag} call out", lambda m, a=a, b=b, u=u: into_out(m, lambda mm, o: u(mm)(a, b) if o is None else u(mm)(a, b, out=o)))
            add(f"{tag} call dtype f8", lambda m, a=a, b=b, u=u: u(m)(a, b, dtype="f8"))
            add(f"{tag} call casting", lambda m, a=a, b=b, u=u: u(m)(a, b, casting="same_kind", dtype="f4"))
            for axis in (0, 1, None):
                add(f"{tag} reduce axis={axis}", lambda m, a=a, u=u, ax=axis: u(m).reduce(a, axis=ax))
            add(f"{tag} reduce keepdims", lambda m, a=a, u=u: u(m).reduce(a, axis=1, keepdims=True))
            add(f"{tag} reduce initial", lambda m, a=a, u=u: u(m).reduce(a, axis=0, initial=1))
            add(f"{tag} reduce where", lambda m, a=a, u=u: u(m).reduce(a, axis=0, where=np.eye(5, 6, dtype=bool), initial=0))
            add(f"{tag} reduce dtype", lambda m, a=a, u=u: u(m).reduce(a, axis=0, dtype="f8"))
            add(f"{tag} accumulate", lambda m, a=a, u=u: u(m).accumulate(a, axis=1))
            add(f"{tag} accumulate axis0", lambda m, a=a, u=u: u(m).accumulate(a, axis=0))
            add(f"{tag} outer", lambda m, a=a, b=b, u=u: u(m).outer(a[0], b[:, 0]))
            add(f"{tag} reduceat", lambda m, a=a, u=u: u(m).reduceat(a, [0, 2, 5], axis=0))
            add(f"{tag} at", lambda m, a=a, b=b, u=u: (lambda x: (u(m).at(x, [0, 2, 0], b.ravel()[:3]), x)[1])(a.copy().ravel()))
            add(f"{tag} at scalar", lambda m, a=a, b=b, u=u: (lambda x: (u(m).at(x, [1, 1], b.ravel()[0]), x)[1])(a.copy().ravel()))
class Raised:
    def __init__(self, ex): self.name = type(ex).__name__
def same(r, s, mask):
    if isinstance(s, (tuple, list)):
        return isinstance(r, (tuple, list)) and len(r) == len(s) and all(same(x, y, mask) for x, y in zip(r, s))
    if s is None or isinstance(s, (str, int, float, bool)):
        return type(r) is type(s) and (r == s or (s != s and r != r))
    if type(r) is not type(s):
        return False
    r2, s2 = np.asarray(r), np.asarray(s)
    if r2.dtype != s2.dtype or r2.shape != s2.shape:
        return False
    if mask is not None:
        r2, s2 = r2[mask], s2[mask]
    return r2.tobytes() == s2.tobytes()
bad = []
compared = 0
for name, fn, mask in cases:
    try:
        s = fn(np)
    except Exception as ex:
        s = Raised(ex)
    try:
        r = fn(fnp)
    except Exception as ex:
        r = Raised(ex)
    if isinstance(s, Raised) or isinstance(r, Raised):
        if not (isinstance(s, Raised) and isinstance(r, Raised) and s.name == r.name):
            bad.append(f"{name}: fnp={getattr(r, 'name', 'ok')} numpy={getattr(s, 'name', 'ok')}")
        continue
    compared += 1
    if not same(r, s, mask):
        bad.append(name)
# One JSON verdict line per ufunc name (the failing cells, empty when it matches), then the
# summary line the assertions read.
import json
for n in names:
    failing = [b for b in bad if b.split(":")[0].split(" ")[0].split(".")[0] == n]
    print(json.dumps({"ufunc": n, "ok": not failing, "failing": failing}))
print(len(names), compared, bad)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    let (result, verdicts) = output
        .trim()
        .lines()
        .collect::<Vec<_>>()
        .split_last()
        .map(|(summary, verdicts)| (summary.to_string(), verdicts.join("\n")))
        .ok_or_else(|| format!("no output: {output}"))?;
    // Per-ufunc verdicts, shown by the test harness whenever this test fails.
    println!("{verdicts}");
    let mut fields = result.trim().splitn(3, ' ');
    let (ufuncs, compared, bad) = (
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or(""),
    );
    assert!(
        ufuncs.parse::<usize>().unwrap_or(0) >= 100,
        "expected ~106 numpy ufunc names: {result}"
    );
    assert!(
        compared.parse::<usize>().unwrap_or(0) >= 5000,
        "too few cells produced a result to compare: {result}"
    );
    assert_eq!(bad, "[]", "ufunc methods must match numpy: {result}");
    Ok(())
}

/// NumPy's binary ufuncs still take the legacy `sig=` spelling of `signature=`: they normalize
/// it before `__array_ufunc__` sees it, refuse it alongside `signature=`, and refuse
/// `sig=None`. fnp's `ufunc.__call__` named only its own keywords, so every `sig=` call -
/// and every other keyword numpy's ufunc owns - raised "unexpected keyword argument" (numpy's
/// own TestBinop::test_ufunc_override_normalize_signature under the drop-in harness).
#[test]
fn binary_ufunc_accepts_numpys_legacy_sig_keyword() -> Result<(), String> {
    let script = fnp_script(
        r#"
class Override:
    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        return sorted(kw.items())
def outcome(fn):
    try:
        r = fn()
        return ("ok", repr(r))
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))
x = np.array([1.5, 2.5])
cases = [
    lambda m: m.add(Override(), [1], sig="ii->i"),
    lambda m: m.multiply(Override(), [1], signature="ii->i"),
    lambda m: m.add(x, x, sig="dd->d"),
    lambda m: m.subtract(x, x, sig=("d", "d", "d")),
    lambda m: m.add(x, x, sig="ii->i"),
    lambda m: m.add(x, x, sig="dd->d", signature="dd->d"),
    lambda m: m.add(x, x, sig=None),
    lambda m: m.add(x, x, keepdims=True),
    lambda m: m.add(x, x, bogus=1),
    lambda m: m.add(x, x),
]
bad = [i for i, c in enumerate(cases) if outcome(lambda: c(fnp)) != outcome(lambda: c(np))]
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "sig= must behave as numpy's: {result}"
    );
    Ok(())
}

/// fnp's lazy byte-equality probes (numpy vs libm, run inside an op's FIRST call) evaluate numpy
/// on f64::MAX, subnormals and 1e300**2. They ran under the CALLER'S errstate, so the first
/// `fnp.sin([0.0])` of a process warned "underflow encountered in sin", the first
/// `fnp.sinh([0.0])` warned overflow and underflow, an `errstate(all='call')` handler received
/// events that were not the caller's, and under `raise` the failed probe silently disabled the
/// op's native route. This runs every probed op's FIRST call, in this fresh process, on a benign
/// in-domain operand under `call` and `raise`: fnp must report exactly what numpy reports
/// (nothing), then still agree with numpy on the values.
#[test]
fn first_call_host_probes_do_not_leak_fp_events_into_the_callers_errstate() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
seen = []
def handler(kind, flag):
    seen.append(kind)
def first_call(m, name, args):
    seen.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(all="call", call=handler):
            r1 = getattr(m, name)(*args)
        with np.errstate(all="raise"):
            try:
                r2 = getattr(m, name)(*args)
            except FloatingPointError as exc:
                r2 = "raised " + str(exc)
    return list(seen), [str(w.message) for w in caught], np.asarray(r1).tolist(), np.asarray(r2).tolist() if not isinstance(r2, str) else r2
x = np.array([0.0, 0.5])
unary = ["sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh",
         "arctanh", "cbrt", "expm1", "log1p", "exp", "exp2", "log", "log2", "log10"]
bad = []
for name in unary:
    if first_call(fnp, name, (x,)) != first_call(np, name, (x,)):
        bad.append(name)
for name in ["power", "arctan2"]:
    operands = (np.array([0.5, 2.0]), np.array([2.0, 0.5]))
    if first_call(fnp, name, operands) != first_call(np, name, operands):
        bad.append(name)
if first_call(fnp, "arccosh", (np.array([1.0, 1.5]),)) != first_call(np, "arccosh", (np.array([1.0, 1.5]),)):
    bad.append("arccosh")
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "a first call must not surface fnp's own probe events: {result}"
    );
    Ok(())
}

/// Bead .26 acceptance probe: under each of errstate warn / raise / ignore, every op's
/// (exception, sorted warning messages) equals numpy's on the same hazardous operands - zero,
/// negative, 1e308, +-inf, NaN, 1e-200 - in the same process. Each op is warmed once under
/// `ignore` first so a lazy host probe (see the first-call test above) cannot be what differs.
/// `ignore` is the negative case: fnp must then emit nothing at all.
#[test]
fn native_kernels_report_numpys_fp_events_under_every_errstate() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
x = np.array([0.0, -1.0, 1e308, np.inf, -np.inf, np.nan, 2.0, 1e-200])
y = np.array([0.0, 0.0, 1e308, np.inf, np.inf, 1.0, 0.0, 1e-200])
with np.errstate(all="ignore"):
    x32 = x.astype(np.float32)
unary = ["reciprocal", "square", "sqrt", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
         "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "exp", "exp2", "expm1", "log",
         "log2", "log10", "log1p", "cbrt", "rint", "negative"]
binary = ["add", "subtract", "multiply", "divide", "true_divide", "floor_divide", "remainder",
          "fmod", "power", "arctan2", "hypot", "maximum", "logaddexp"]
calls = [(n, (lambda n: lambda m: getattr(m, n)(x))(n)) for n in unary]
calls += [(n + "_f32", (lambda n: lambda m: getattr(m, n)(x32))(n)) for n in unary]
calls += [(n, (lambda n: lambda m: getattr(m, n)(x, y))(n)) for n in binary]
calls += [
    ("cumsum", lambda m: m.cumsum(x)),
    ("sum_pair", lambda m: m.sum(np.array([np.inf, -np.inf]))),
    ("nanmean_pair", lambda m: m.nanmean(np.array([np.inf, -np.inf]))),
    ("nanmean_axis", lambda m: m.nanmean(np.array([[np.inf, -np.inf], [1.0, 2.0]]), axis=1)),
    ("mean_pair", lambda m: m.mean(np.array([np.inf, -np.inf]))),
    ("prod_overflow", lambda m: m.prod(np.array([1e300, 1e300]))),
    ("cumprod_overflow", lambda m: m.cumprod(np.array([1e300, 1e300]))),
    ("var_overflow", lambda m: m.var(np.array([1e300, -1e300]))),
    ("std_inf", lambda m: m.std(np.array([np.inf, 1.0]))),
]
def outcome(fn, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(all=mode):
            try:
                fn()
                exc = None
            except Exception as e:
                exc = type(e).__name__ + ": " + str(e)
    return exc, sorted(str(w.message) for w in caught)
for name, call in calls:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            call(fnp)
        except Exception:
            pass
bad = []
for name, call in calls:
    for mode in ("warn", "raise", "ignore"):
        theirs, ours = outcome(lambda: call(np), mode), outcome(lambda: call(fnp), mode)
        if theirs != ours:
            bad.append(f"{name}/{mode}: numpy={theirs} fnp={ours}")
print(len(calls) * 3, "cells;", len(bad), "diverge")
for line in bad:
    print("  " + line)
print(True if not bad else False)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "FP events must match numpy under every errstate:\n{result}"
    );
    Ok(())
}

/// Native routes read buffers, so a layout they do not expect can give a quietly different answer:
/// a Fortran-order .npy loaded PERMUTED, and a byte-swapped `>f8` once made `isnan` answer all-False.
/// This sweeps 70 functions over {C, F, byte-swapped, byte-swapped F, strided, reversed, transposed}
/// x {f8, f4, i8} and requires numpy's exact bytes, dtype and shape. Failing cells before the fix:
/// nansum/nanmean on byte-swapped f8 (last-bit: sequential sum), and float32 trace (f64 fold) -
/// which the dedicated loop shows is not layout-specific (51% of random float32 matrices differed).
#[test]
fn functions_match_numpy_on_fortran_byteswapped_strided_and_reversed_layouts() -> Result<(), String>
{
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(7)
def layouts(dt):
    kind = np.dtype(dt).kind
    base = (rng.standard_normal((6, 8)) * 5).astype(dt) if kind == "f" else rng.integers(-50, 50, (6, 8)).astype(dt)
    if kind == "f":
        base.flat[3] = np.nan
    swapped = base.astype(base.dtype.newbyteorder(">"))
    return {"C": base.copy(), "F": np.asfortranarray(base), "byteswapped": swapped,
            "byteswapped_F": np.asfortranarray(swapped), "strided": base[:, ::2],
            "reversed": base[::-1, ::-1], "T": base.T}
names = ["abs", "negative", "sqrt", "exp", "log", "sin", "floor", "ceil", "rint", "sign", "square",
         "isnan", "isfinite", "isinf", "signbit", "reciprocal", "cbrt", "trunc", "fabs",
         "sum", "prod", "mean", "std", "var", "min", "max", "argmin", "argmax", "nansum",
         "nanmean", "nanmin", "nanmax", "ptp", "median", "any", "all", "count_nonzero",
         "cumsum", "cumprod", "sort", "argsort", "unique", "nonzero", "flatnonzero", "diff",
         "ravel", "flip", "round", "clip", "copy", "ascontiguousarray", "isin", "searchsorted",
         "nan_to_num", "where", "maximum", "add", "multiply", "subtract", "divide", "power",
         "dot", "matmul", "outer", "tile", "repeat", "cross", "trace", "diagonal", "transpose"]
def call(mod, name, a):
    f = getattr(mod, name)
    if name == "clip": return f(a, -2, 2)
    if name in ("maximum", "add", "multiply", "subtract", "divide", "power"): return f(a, a)
    if name == "where": return f(a > 0, a, 0)
    if name == "isin": return f(a, a[0])
    if name == "searchsorted": return f(np.sort(a.ravel()), a.ravel()[:5])
    if name in ("dot", "matmul"): return f(a, a.T)
    if name == "outer": return f(a.ravel()[:4], a.ravel()[:5])
    if name == "tile": return f(a, 2)
    if name == "repeat": return f(a, 2, axis=0)
    if name == "cross": return f(a[:, :3], a[:, :3])
    if name == "round": return f(a, 1)
    return f(a)
def same(r, s):
    if isinstance(s, tuple):
        return isinstance(r, tuple) and len(r) == len(s) and all(same(x, y) for x, y in zip(r, s))
    r, s = np.asarray(r), np.asarray(s)
    return r.dtype == s.dtype and r.shape == s.shape and r.tobytes() == s.tobytes()
bad, cells = [], 0
for dt in ("<f8", "<f4", "<i8"):
    for lname, a in layouts(dt).items():
        for name in names:
            try:
                s = call(np, name, a)
            except Exception:
                continue
            cells += 1
            try:
                r = call(fnp, name, a)
            except Exception as ex:
                bad.append(f"{name} {lname} {dt}: fnp raised {type(ex).__name__}")
                continue
            if not same(r, s):
                bad.append(f"{name} {lname} {dt}")
trace_bad = 0
for n in (3, 8, 50, 300):
    for _ in range(50):
        m = (rng.standard_normal((n, n)) * 7).astype(np.float32)
        r, s = fnp.trace(m), np.trace(m)
        trace_bad += type(r) is not type(s) or np.asarray(r).tobytes() != np.asarray(s).tobytes()
if trace_bad:
    bad.append(f"float32 trace differs in {trace_bad}/200 matrices")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert!(
        cells.parse::<usize>().unwrap_or(0) >= 1400,
        "cell table drifted: {result}"
    );
    assert_eq!(bad, "[]", "layout parity with numpy: {result}");
    Ok(())
}

/// Native routes switch on at size gates and dtype checks, so sweep 58 functions over 14 dtypes at
/// n = 7 and n = 70,000 and require numpy's exact bytes, dtype, shape and exception type. Before
/// the fixes this sweep was written with, `trapezoid` failed in four ways: float32/float64 last
/// bits (a sum shortcut), float16 result dtype, and bool values (numpy's `y[1:] + y[:-1]` is a
/// logical OR). `cross` failed in three: float32/float16 computed in f64, and bool answered where
/// numpy raises. `cov` is covered by its own tests (ledger row DIV-COV-GRAM-NO-FMA).
#[test]
fn functions_match_numpy_across_dtypes_and_sizes() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(11)
dtypes = ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "c8", "c16"]
def make(dt, n):
    d = np.dtype(dt)
    if d.kind == "b":
        return rng.integers(0, 2, n).astype(d)
    if d.kind in "iu":
        info = np.iinfo(d)
        return rng.integers(max(info.min, -1000), min(info.max, 1000), n).astype(d)
    if d.kind == "f":
        return (rng.standard_normal(n) * 50).astype(d)
    return (rng.standard_normal(n) * 50 + 1j * rng.standard_normal(n) * 50).astype(d)
funcs = {
    "sum": lambda m, a: m.sum(a), "prod": lambda m, a: m.prod(a[:20]), "mean": lambda m, a: m.mean(a),
    "cumsum": lambda m, a: m.cumsum(a), "min": lambda m, a: m.min(a), "max": lambda m, a: m.max(a),
    "argmin": lambda m, a: m.argmin(a), "argmax": lambda m, a: m.argmax(a), "ptp": lambda m, a: m.ptp(a),
    "sort": lambda m, a: m.sort(a), "argsort_stable": lambda m, a: m.argsort(a, kind="stable"),
    "unique": lambda m, a: m.unique(a), "abs": lambda m, a: m.abs(a), "negative": lambda m, a: m.negative(a),
    "square": lambda m, a: m.square(a), "sign": lambda m, a: m.sign(a), "clip": lambda m, a: m.clip(a, 1, 50),
    "add": lambda m, a: m.add(a, a), "subtract": lambda m, a: m.subtract(a, a[::-1]),
    "multiply": lambda m, a: m.multiply(a, a), "maximum": lambda m, a: m.maximum(a, a[::-1]),
    "equal": lambda m, a: m.equal(a, a[::-1]), "less": lambda m, a: m.less(a, a[::-1]),
    "where": lambda m, a: m.where(a > a[::-1], a, a[::-1]), "nonzero": lambda m, a: m.nonzero(a),
    "count_nonzero": lambda m, a: m.count_nonzero(a), "any": lambda m, a: m.any(a), "all": lambda m, a: m.all(a),
    "diff": lambda m, a: m.diff(a), "cumprod": lambda m, a: m.cumprod(a[:12]), "round": lambda m, a: m.round(a, 1),
    "isnan": lambda m, a: m.isnan(a), "isfinite": lambda m, a: m.isfinite(a),
    "searchsorted": lambda m, a: m.searchsorted(m.sort(a), a[:9]), "isin": lambda m, a: m.isin(a, a[:30]),
    "bincount": lambda m, a: m.bincount(np.abs(a.astype(np.int64)) % 64),
    "histogram": lambda m, a: m.histogram(a.real if a.dtype.kind == "c" else a, bins=7),
    "percentile": lambda m, a: m.percentile(a, 37), "median": lambda m, a: m.median(a),
    "var": lambda m, a: m.var(a), "std": lambda m, a: m.std(a), "dot": lambda m, a: m.dot(a, a),
    "convolve": lambda m, a: m.convolve(a[:200], a[:9]), "flip": lambda m, a: m.flip(a),
    "repeat": lambda m, a: m.repeat(a[:50], 3), "tile": lambda m, a: m.tile(a[:50], 3),
    "concatenate": lambda m, a: m.concatenate([a, a[:7]]), "cross": lambda m, a: m.cross(a[:3], a[3:6]),
    "cross_n3": lambda m, a: m.cross(a[:6].reshape(2, 3), a[1:7].reshape(2, 3)),
    "power": lambda m, a: m.power(a[:30], 2), "logical_and": lambda m, a: m.logical_and(a, a[::-1]),
    "trapezoid": lambda m, a: m.trapezoid(a), "trapezoid_dx": lambda m, a: m.trapezoid(a, dx=0.1),
    "trapezoid_x": lambda m, a: m.trapezoid(a, x=np.cumsum(np.ones(len(a))) * 0.5),
    "trapezoid_2d_last": lambda m, a: m.trapezoid(a[: len(a) // 7 * 7].reshape(-1, 7)),
    "trapezoid_2d_axis0": lambda m, a: m.trapezoid(a[: len(a) // 7 * 7].reshape(-1, 7), axis=0),
    "nan_to_num": lambda m, a: m.nan_to_num(a), "argwhere": lambda m, a: m.argwhere(a > 3),
}
class Raised:
    def __init__(self, ex): self.name = type(ex).__name__
def same(r, s):
    if isinstance(s, tuple):
        return isinstance(r, tuple) and len(r) == len(s) and all(same(x, y) for x, y in zip(r, s))
    if type(r) is not type(s) and not (isinstance(r, np.ndarray) and isinstance(s, np.ndarray)):
        return False
    r, s = np.asarray(r), np.asarray(s)
    return r.dtype == s.dtype and r.shape == s.shape and r.tobytes() == s.tobytes()
bad, cells = [], 0
for dt in dtypes:
    for n in (7, 70_000):
        a = make(dt, n)
        for name, f in funcs.items():
            try:
                s = f(np, a)
            except Exception as ex:
                s = Raised(ex)
            try:
                r = f(fnp, a)
            except Exception as ex:
                r = Raised(ex)
            cells += 1
            if isinstance(s, Raised) or isinstance(r, Raised):
                if not (isinstance(s, Raised) and isinstance(r, Raised) and s.name == r.name):
                    bad.append(f"{name} {dt}/{n}: exception fnp={getattr(r, 'name', '-')} numpy={getattr(s, 'name', '-')}")
            elif not same(r, s):
                bad.append(f"{name} {dt}/{n}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert!(
        cells.parse::<usize>().unwrap_or(0) >= 1600,
        "cell table drifted: {result}"
    );
    assert_eq!(bad, "[]", "dtype/size parity with numpy: {result}");
    Ok(())
}

/// Binary ops over every pairing of 11 array dtypes, a broadcasting column, Python scalars (int,
/// out-of-range int, negative int, float, complex, bool) and NumPy scalars: numpy's result type,
/// dtype, bytes, and exception type. Before the fix only the shifts failed, 302 cells for
/// left_shift and 289 for right_shift: wrong promotion (bool << bool gave bool, numpy int8),
/// ValueError on mixed widths numpy shifts, and ValueError where numpy raises TypeError or
/// OverflowError.
#[test]
fn binary_ops_match_numpy_promotion_scalars_and_exceptions() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(5)
dts = ["?", "i1", "u1", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "c16"]
def arr(dt, n=5):
    d = np.dtype(dt)
    if d.kind == "b": return rng.integers(0, 2, n).astype(d)
    if d.kind == "u": return rng.integers(1, 100, n).astype(d)
    if d.kind == "i": return rng.integers(-60, 60, n).astype(d)
    if d.kind == "f": return (rng.standard_normal(n) * 9).astype(d)
    return (rng.standard_normal(n) * 9 + 3j).astype(d)
operands = {f"arr_{d}": arr(d) for d in dts}
operands.update({f"col_{d}": arr(d).reshape(5, 1)[:3] for d in ("i4", "f8")})
operands.update({"py_int": 3, "py_big": 300, "py_neg": -2, "py_float": 2.5, "py_complex": 1 + 2j,
                 "py_bool": True, "np_i8": np.int8(3), "np_u8": np.uint8(200),
                 "np_f32": np.float32(2.5), "np_f64": np.float64(2.5)})
ops = ["add", "subtract", "multiply", "true_divide", "floor_divide", "remainder", "power", "maximum",
       "minimum", "fmax", "fmin", "arctan2", "hypot", "copysign", "logaddexp", "bitwise_and",
       "bitwise_or", "bitwise_xor", "left_shift", "right_shift", "equal", "less", "greater_equal",
       "logical_xor", "heaviside", "fmod", "divmod"]
class Raised:
    def __init__(self, ex): self.name = type(ex).__name__
def same(r, s):
    if isinstance(s, tuple):
        return isinstance(r, tuple) and len(r) == len(s) and all(same(x, y) for x, y in zip(r, s))
    if type(r) is not type(s):
        return False
    r, s = np.asarray(r), np.asarray(s)
    return r.dtype == s.dtype and r.shape == s.shape and r.tobytes() == s.tobytes()
bad, cells, shifts = [], 0, 0
names = list(operands)
for op in ops:
    for ln in names:
        for rn in names:
            if not (ln.startswith(("arr", "col")) or rn.startswith(("arr", "col"))):
                continue
            a, b = operands[ln], operands[rn]
            try:
                s = getattr(np, op)(a, b)
            except Exception as ex:
                s = Raised(ex)
            try:
                r = getattr(fnp, op)(a, b)
            except Exception as ex:
                r = Raised(ex)
            cells += 1
            shifts += op.endswith("_shift")
            if isinstance(s, Raised) or isinstance(r, Raised):
                if not (isinstance(s, Raised) and isinstance(r, Raised) and s.name == r.name):
                    bad.append(f"{op}({ln}, {rn}): fnp={getattr(r, 'name', 'ok')} numpy={getattr(s, 'name', 'ok')}")
            elif not same(r, s):
                bad.append(f"{op}({ln}, {rn})")
print(cells, shifts, bad[:40], len(bad))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(3, ' ');
    let cells: usize = fields.next().unwrap_or("").parse().unwrap_or(0);
    let shifts: usize = fields.next().unwrap_or("").parse().unwrap_or(0);
    assert!(cells >= 11_000, "cell table drifted: {result}");
    assert!(
        shifts >= 800,
        "shift cells must stay in the table: {result}"
    );
    assert!(
        fields.next().unwrap_or("").ends_with("[] 0"),
        "binary ops differ from numpy: {result}"
    );
    Ok(())
}

/// Creation, manipulation, set and search functions with edge parameters, compared with numpy by
/// result type, dtype, shape, bytes and exception type (140 cases). Before the fix only
/// `linspace(..., retstep=True)` failed: `num=1, endpoint=False` returned step NaN (numpy:
/// `stop - start`), and numpy's undefined step is the Python float `nan`, not `np.float64(nan)`.
#[test]
fn creation_and_manipulation_functions_match_numpy_on_edge_parameters() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(13)
f8 = rng.standard_normal(40) * 10
i8 = rng.integers(-20, 20, 40)
f4 = f8.astype(np.float32)
m = rng.standard_normal((5, 6))
cases = []
def add(name, fn):
    cases.append((name, fn))
for start, stop, num in [(0, 1, 7), (1, 10, 50), (-3.3, 7.1, 13), (0, 1, 1), (5, 5, 4), (0, 1e-300, 5), (1, 2, 0)]:
    for endpoint in (True, False):
        add(f"linspace({start},{stop},{num},{endpoint})", lambda m_, a=start, b=stop, n=num, e=endpoint: m_.linspace(a, b, n, endpoint=e))
        add(f"linspace_retstep({start},{stop},{num},{endpoint})", lambda m_, a=start, b=stop, n=num, e=endpoint: m_.linspace(a, b, n, endpoint=e, retstep=True))
add("linspace_int_dtype", lambda m_: m_.linspace(0, 10, 7, dtype=np.int64))
add("linspace_f32", lambda m_: m_.linspace(0, 1, 9, dtype=np.float32))
add("linspace_array", lambda m_: m_.linspace([0, 1], [5, 11], 4))
add("linspace_axis1", lambda m_: m_.linspace([0, 1], [5, 11], 4, axis=1))
for base in (10.0, 2.0, np.e):
    add(f"logspace base={base}", lambda m_, b=base: m_.logspace(0, 3, 7, base=b))
add("geomspace", lambda m_: m_.geomspace(1, 1000, 7))
add("geomspace_neg", lambda m_: m_.geomspace(-1, -1000, 5))
add("geomspace_complex", lambda m_: m_.geomspace(1j, 1000j, 4))
for args in [(10,), (0, 1, 0.1), (1, 2, 0.3), (-5, 5, 1.5), (0.0, 1.0, 1 / 3), (5, 0, -1), (0, 10, 3), (1e10, 1e10 + 5, 1)]:
    add(f"arange{args}", lambda m_, a=args: m_.arange(*a))
add("arange_f32", lambda m_: m_.arange(0, 1, 0.1, dtype=np.float32))
for mode in ("constant", "edge", "reflect", "symmetric", "wrap", "linear_ramp", "maximum", "mean", "median", "minimum"):
    add(f"pad {mode}", lambda m_, md=mode: m_.pad(m, ((1, 2), (3, 0)), mode=md))
add("pad reflect odd", lambda m_: m_.pad(f8[:5], 3, mode="reflect", reflect_type="odd"))
add("pad constant values", lambda m_: m_.pad(i8[:5], (2, 3), constant_values=(-1, 7)))
for sh in (3, -2, 45):
    add(f"roll {sh}", lambda m_, s=sh: m_.roll(m, s))
    add(f"roll axis {sh}", lambda m_, s=sh: m_.roll(m, s, axis=1))
add("roll tuple", lambda m_: m_.roll(m, (1, -2), axis=(0, 1)))
for k in (1, 2, 3, -1):
    add(f"rot90 {k}", lambda m_, kk=k: m_.rot90(m, kk))
for fn in ("union1d", "intersect1d", "setdiff1d", "setxor1d"):
    add(fn, lambda m_, f=fn: getattr(m_, f)(i8[:25], i8[15:]))
    add(fn + " f8", lambda m_, f=fn: getattr(m_, f)(np.round(f8[:25]), np.round(f8[15:])))
add("intersect1d indices", lambda m_: m_.intersect1d(i8[:25], i8[15:], return_indices=True))
add("unique all", lambda m_: m_.unique(i8, return_index=True, return_inverse=True, return_counts=True))
add("unique axis0", lambda m_: m_.unique(np.array([[1, 2], [1, 2], [0, 5]]), axis=0))
add("unique f8 nan", lambda m_: m_.unique(np.array([np.nan, 1.0, np.nan, -0.0, 0.0])))
add("unique equal_nan False", lambda m_: m_.unique(np.array([np.nan, 1.0, np.nan]), equal_nan=False))
for kth in (0, 5, -1, [2, 7]):
    add(f"partition {kth}", lambda m_, k=kth: np.sort(m_.partition(f8, k)))
    add(f"partition kth-element {kth}", lambda m_, k=kth: m_.partition(f8, k)[k])
    add(f"argpartition values {kth}", lambda m_, k=kth: f8[m_.argpartition(f8, k)][k])
for side in ("left", "right"):
    add(f"searchsorted {side}", lambda m_, s=side: m_.searchsorted(np.sort(i8), [-20, 0, 3, 19, 25], side=s))
    add(f"searchsorted sorter {side}", lambda m_, s=side: m_.searchsorted(i8, [0, 3], side=s, sorter=np.argsort(i8, kind="stable")))
for right in (False, True):
    add(f"digitize {right}", lambda m_, r=right: m_.digitize(f8, [-10, 0, 5, 10], right=r))
    add(f"digitize decreasing {right}", lambda m_, r=right: m_.digitize(f8, [10, 5, 0, -10], right=r))
add("interp", lambda m_: m_.interp([-50, -1, 0.5, 3, 99], np.sort(f8), np.arange(40.0)))
add("interp lr", lambda m_: m_.interp([-50, 99], np.sort(f8), np.arange(40.0), left=-7, right=7))
add("interp period", lambda m_: m_.interp([-50, 3, 400], [0, 90, 180, 270], [1, 2, 3, 4], period=360))
add("interp complex", lambda m_: m_.interp([0.5, 1.5], [0, 1, 2], [1 + 1j, 2, 3 - 1j]))
for mode in ("full", "same", "valid"):
    add(f"convolve {mode}", lambda m_, md=mode: m_.convolve(f8, f8[:7], mode=md))
    add(f"correlate {mode}", lambda m_, md=mode: m_.correlate(f8, f8[:7], mode=md))
    add(f"convolve int {mode}", lambda m_, md=mode: m_.convolve(i8, i8[:5], mode=md))
add("correlate complex", lambda m_: m_.correlate(f8[:9] + 1j * f8[9:18], f8[:3] - 2j))
add("lexsort", lambda m_: m_.lexsort((i8 % 3, i8 // 3)))
add("sort kind stable f4", lambda m_: m_.sort(f4, kind="stable"))
add("argsort kind stable f4", lambda m_: m_.argsort(f4, kind="stable"))
add("take_along_axis", lambda m_: m_.take_along_axis(m, np.argsort(m, axis=1), axis=1))
add("meshgrid ij", lambda m_: m_.meshgrid([1, 2, 3], [4, 5], indexing="ij"))
add("meshgrid sparse", lambda m_: m_.meshgrid([1, 2, 3], [4, 5], sparse=True))
add("tri", lambda m_: m_.tri(4, 5, 1))
add("eye k", lambda m_: m_.eye(4, 6, k=-2, dtype=np.int8))
add("diagflat", lambda m_: m_.diagflat([1, 2, 3], 1))
add("vander", lambda m_: m_.vander([1, 2, 3.5], 4))
add("histogram density", lambda m_: m_.histogram(f8, bins=5, density=True))
add("histogram range", lambda m_: m_.histogram(f8, bins="auto", range=(-5, 5)))
add("histogram weights", lambda m_: m_.histogram(f8, bins=6, weights=np.abs(f8)))
add("histogram2d", lambda m_: m_.histogram2d(f8[:20], f8[20:], bins=4))
add("bincount weights minlength", lambda m_: m_.bincount(np.abs(i8), weights=f8, minlength=30))
add("cumulative_sum include_initial", lambda m_: m_.cumulative_sum(f8, include_initial=True))
add("gradient", lambda m_: m_.gradient(m, 0.5, axis=1))
add("gradient edge2", lambda m_: m_.gradient(f8, edge_order=2))
add("ediff1d", lambda m_: m_.ediff1d(i8, to_begin=[-99], to_end=99))
add("polyfit", lambda m_: m_.polyfit(np.arange(40.0), f8, 3))
add("polyval", lambda m_: m_.polyval([1.5, -2, 0.25], f8))
add("round half", lambda m_: m_.round(np.array([0.5, 1.5, 2.5, -0.5, 2.675, 1.005]), 2))
add("around neg decimals", lambda m_: m_.around(i8 * 137, -2))
class Raised:
    def __init__(self, ex): self.name = type(ex).__name__
def same(r, s):
    if isinstance(s, (tuple, list)):
        return isinstance(r, (tuple, list)) and len(r) == len(s) and all(same(x, y) for x, y in zip(r, s))
    if type(r) is not type(s):
        return False
    r, s = np.asarray(r), np.asarray(s)
    return r.dtype == s.dtype and r.shape == s.shape and r.tobytes() == s.tobytes()
bad = []
for name, fn in cases:
    try:
        s = fn(np)
    except Exception as ex:
        s = Raised(ex)
    try:
        r = fn(fnp)
    except Exception as ex:
        r = Raised(ex)
    if isinstance(s, Raised) or isinstance(r, Raised):
        if not (isinstance(s, Raised) and isinstance(r, Raised) and s.name == r.name):
            bad.append(f"{name}: fnp={getattr(r, 'name', 'ok')} numpy={getattr(s, 'name', 'ok')}")
    elif not same(r, s):
        bad.append(name)
print(len(cases), bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cases, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert!(
        cases.parse::<usize>().unwrap_or(0) >= 140,
        "case table drifted: {result}"
    );
    assert_eq!(bad, "[]", "edge-parameter parity with numpy: {result}");
    Ok(())
}

/// Structural and indexing functions with the keywords that change semantics (order C/F/A,
/// axis None/negative/out of range, casting, dtype, take/choose modes), compared with numpy by
/// result type, dtype, shape, bytes, C/F contiguity and exception type (140 cases). Before the
/// fixes: `take(a3d, [7, -9], mode="clip")` read element 51 for -9 (clip mode disables negative
/// indexing; numpy reads 0), `argwhere` was C-contiguous where numpy's `transpose(nonzero(a))`
/// is F-contiguous, and `unravel_index` returned separate contiguous arrays where numpy returns
/// column views of one (n, ndim) array.
#[test]
fn structural_functions_match_numpy_on_order_axis_and_mode_keywords() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(21)
A = rng.integers(-9, 9, (3, 4, 5))
F = np.asfortranarray(rng.standard_normal((4, 6)))
v = rng.standard_normal(12)
b = rng.integers(0, 2, 12).astype(bool)
cases = []
def add(name, fn):
    cases.append((name, fn))
for order in ("C", "F", "A"):
    add(f"reshape {order}", lambda m, o=order: m.reshape(A, (4, 15), order=o))
    add(f"reshape F-src {order}", lambda m, o=order: m.reshape(F, (8, 3), order=o))
    add(f"ravel {order}", lambda m, o=order: m.ravel(F, order=o))
    add(f"copy {order}", lambda m, o=order: m.copy(F, order=o).flags.c_contiguous)
    add(f"asarray order {order}", lambda m, o=order: m.asarray(A, order=o).flags.f_contiguous)
add("ravel K", lambda m: m.ravel(F.T, order="K"))
add("resize", lambda m: m.resize(A, (7, 3)))
for axis in (0, 1, 2, -1, None):
    add(f"concatenate axis={axis}", lambda m, ax=axis: m.concatenate([A, A], axis=ax))
    add(f"flip axis={axis}", lambda m, ax=axis: m.flip(A, axis=ax))
    add(f"delete axis={axis}", lambda m, ax=axis: m.delete(A, [0, -1], axis=ax))
    add(f"insert axis={axis}", lambda m, ax=axis: m.insert(A, 1, 99, axis=ax))
    add(f"append axis={axis}", lambda m, ax=axis: m.append(A, A, axis=ax))
    add(f"take axis={axis}", lambda m, ax=axis: m.take(A, [2, 0, -1], axis=ax))
    for mode in ("clip", "wrap"):
        add(f"take {mode} list axis={axis}", lambda m, ax=axis, md=mode: m.take(A, [7, -9], axis=ax, mode=md))
        add(f"take {mode} ndarray axis={axis}", lambda m, ax=axis, md=mode: m.take(A, np.array([7, -9]), axis=ax, mode=md))
    add(f"compress axis={axis}", lambda m, ax=axis: m.compress([True, False, True], A, axis=ax))
    add(f"cumsum axis={axis}", lambda m, ax=axis: m.cumsum(A, axis=ax))
    add(f"expand_dims {axis}", lambda m, ax=axis: m.expand_dims(A, ax if ax is not None else 0))
add("concatenate dtype", lambda m: m.concatenate([A, A], axis=None, dtype=np.float32))
add("concatenate casting", lambda m: m.concatenate([v, A.ravel()], casting="same_kind", dtype=np.int32))
add("concatenate casting unsafe", lambda m: m.concatenate([v, v], casting="unsafe", dtype=np.int8))
add("concatenate bad axis", lambda m: m.concatenate([A, A], axis=3))
add("concatenate mismatch", lambda m: m.concatenate([A, A[:, :2]], axis=0))
for fn in ("stack", "hstack", "vstack", "dstack", "column_stack"):
    add(fn, lambda m, f=fn: getattr(m, f)([v, v * 2]))
add("stack axis -1", lambda m: m.stack([A, A], axis=-1))
add("stack dtype", lambda m: m.stack([v, v], dtype=np.float32))
add("block", lambda m: m.block([[A[0], A[1]], [A[2], A[0]]]))
add("split", lambda m: m.split(A, [1, 2]))
add("array_split", lambda m: m.array_split(A, 2))
add("hsplit", lambda m: m.hsplit(A, [1, 2]))
add("vsplit", lambda m: m.vsplit(A, [1, 2]))
add("dsplit", lambda m: m.dsplit(A, [1, 2]))
add("array_split uneven", lambda m: m.array_split(v, 5))
add("split unequal raises", lambda m: m.split(v, 5))
add("moveaxis", lambda m: m.moveaxis(A, [0, 1], [-1, -2]))
add("swapaxes", lambda m: m.swapaxes(A, 0, 2))
add("transpose axes", lambda m: m.transpose(A, (1, 2, 0)))
add("squeeze", lambda m: m.squeeze(A[:, :1, :1]))
add("squeeze axis bad", lambda m: m.squeeze(A, axis=0))
add("broadcast_to", lambda m: m.broadcast_to(v[:5], (3, 5)))
add("broadcast_arrays", lambda m: m.broadcast_arrays(A[:, :1], v[:5]))
add("tile", lambda m: m.tile(A, (2, 1, 1, 2)))
add("repeat axis", lambda m: m.repeat(A, [1, 0, 2], axis=0))
add("choose", lambda m: m.choose(A[0] % 3, [A[0], A[1], A[2]]))
add("choose clip", lambda m: m.choose(A[0], [A[0], A[1], A[2]], mode="clip"))
add("select", lambda m: m.select([A > 3, A < -3], [A, -A], default=7))
add("piecewise", lambda m: m.piecewise(v, [v < 0, v >= 0], [lambda x: -x, lambda x: x * 2]))
add("extract", lambda m: m.extract(b, v))
add("where 1arg", lambda m: m.where(A > 0))
add("argwhere 3-D", lambda m: m.argwhere(A > 4))
add("argwhere 2-D f64", lambda m: m.argwhere(F > 0))
add("argwhere large", lambda m: m.argwhere(np.arange(600_000).reshape(600, 1000) % 7 == 0))
add("argwhere bool 1-D", lambda m: m.argwhere(b))
add("nonzero", lambda m: m.nonzero(A))
add("flatnonzero", lambda m: m.flatnonzero(A))
add("tril", lambda m: m.tril(A, -1))
add("triu", lambda m: m.triu(F, 2))
add("diag", lambda m: m.diag(F, -1))
add("diagonal", lambda m: m.diagonal(A, 1, 0, 2))
add("fill_diagonal", lambda m: (lambda x: (m.fill_diagonal(x, 5), x)[1])(np.zeros((4, 4))))
add("put", lambda m: (lambda x: (m.put(x, [0, -1, 5], [7, 8, 9]), x)[1])(np.arange(10.0)))
add("put wrap", lambda m: (lambda x: (m.put(x, [11, -12], [7, 8], mode="wrap"), x)[1])(np.arange(10.0)))
add("putmask", lambda m: (lambda x: (m.putmask(x, x > 4, [-1, -2]), x)[1])(np.arange(10.0)))
add("place", lambda m: (lambda x: (m.place(x, x > 4, [-1, -2]), x)[1])(np.arange(10.0)))
add("put_along_axis", lambda m: (lambda x: (m.put_along_axis(x, np.argsort(x, axis=1)[:, :1], -1, axis=1), x)[1])(F.copy()))
add("ix_", lambda m: A[m.ix_([0, 2], [1, 3], [4])])
add("ravel_multi_index", lambda m: m.ravel_multi_index(([0, 2], [1, 3], [4, 0]), (3, 4, 5)))
for order in ("C", "F"):
    add(f"unravel_index {order}", lambda m, o=order: m.unravel_index(np.array([5, 17, 33]), (3, 4, 5), order=o))
    add(f"unravel_index 2-D {order}", lambda m, o=order: m.unravel_index(np.array([[5, 17], [33, 1]]), (3, 4, 5), order=o))
    add(f"unravel_index list {order}", lambda m, o=order: m.unravel_index([5, 17], (3, 4, 5), order=o))
    add(f"unravel_index scalar {order}", lambda m, o=order: m.unravel_index(17, (3, 4, 5), order=o))
add("unravel_index oob", lambda m: m.unravel_index(np.array([60]), (3, 4, 5)))
add("indices", lambda m: m.indices((2, 3)))
add("atleast_3d", lambda m: m.atleast_3d(v))
add("trim_zeros", lambda m: m.trim_zeros(np.array([0, 0, 1, 2, 0]), "b"))
add("rollaxis", lambda m: m.rollaxis(A, 2, 0))
class Raised:
    def __init__(self, ex): self.name = type(ex).__name__
def same(r, s):
    if isinstance(s, (tuple, list)):
        return isinstance(r, (tuple, list)) and len(r) == len(s) and all(same(x, y) for x, y in zip(r, s))
    if type(r) is not type(s):
        return False
    r2, s2 = np.asarray(r), np.asarray(s)
    if r2.dtype != s2.dtype or r2.shape != s2.shape or r2.tobytes() != s2.tobytes():
        return False
    if isinstance(s, np.ndarray):
        return r.flags.c_contiguous == s.flags.c_contiguous and r.flags.f_contiguous == s.flags.f_contiguous
    return True
bad = []
for name, fn in cases:
    try:
        s = fn(np)
    except Exception as ex:
        s = Raised(ex)
    try:
        r = fn(fnp)
    except Exception as ex:
        r = Raised(ex)
    if isinstance(s, Raised) or isinstance(r, Raised):
        if not (isinstance(s, Raised) and isinstance(r, Raised) and s.name == r.name):
            bad.append(f"{name}: fnp={getattr(r, 'name', 'ok')} numpy={getattr(s, 'name', 'ok')}")
    elif not same(r, s):
        bad.append(name)
print(len(cases), bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cases, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert!(
        cases.parse::<usize>().unwrap_or(0) >= 130,
        "case table drifted: {result}"
    );
    assert_eq!(bad, "[]", "structural parity with numpy: {result}");
    Ok(())
}

/// Every numpy.__all__ callable (344 after excluding IO, state and class entry points) that numpy
/// accepts with one or two same-dtype arrays, with no keyword and with axis=0 / axis=-1, over ten
/// dtypes in BOTH byte orders (f8, i4, c16, f4, u2, i8, f2, c8, i2, m8[s]) at (4, 6) and at
/// (160, 128), which clears the 2**14-element floors of most native routes: fnp must match
/// numpy's result type, dtype, shape and bytes, or raise the same exception type (14,865 cells).
/// Big-endian operands used to reach native kernels that `.view()` the data as a native integer
/// and compute on the reinterpreted bits (bead .8): argmax/argmin/min/max/ptp of '>m8' gave
/// wrong answers, angle('>c16') was off by up to 5.6, outer/kron/lexsort/frexp and the
/// nancumsum/nancumprod axis routes of '>f2' returned zeros or wrong orders, choose('>f8')
/// returned 4.585e-320 for 10.0 and take_along_axis('>i4') 33554432 for 2; triu/tril/extract/
/// setdiff1d and the '>m8' set-ops and axis min/max had the wrong byte order. The native-order
/// half found polyval of a 2-D coefficient array wrong by 1.9e22, float64 logaddexp2 off by one
/// ulp (it divided by ln 2 where numpy multiplies by log2 e), float32 i0 off in the last place,
/// and float32 histogramdd edges in float64. cov/corrcoef are excluded by name: their last-bit
/// difference is the documented DIV-COV-GRAM-NO-FMA contract.
#[test]
fn array_functions_match_numpy_on_native_and_byteswapped_operands() -> Result<(), String> {
    let script = fnp_script(
        r#"
import inspect, warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(12)
SKIP = {"save", "savez", "savez_compressed", "savetxt", "load", "loadtxt", "genfromtxt", "fromfile",
        "memmap", "seterr", "seterrcall", "setbufsize", "set_printoptions", "printoptions", "info",
        "show_config", "show_runtime", "test", "from_dlpack", "frompyfunc", "vectorize", "piecewise",
        "apply_along_axis", "apply_over_axes", "fromfunction", "fromregex", "nditer", "nested_iters",
        "busday_offset", "busday_count", "is_busday", "put", "place", "putmask", "copyto",
        "fill_diagonal", "shares_memory", "may_share_memory", "empty", "empty_like", "ndarray",
        "broadcast", "iinfo", "finfo", "dtype", "format_float_positional", "format_float_scientific",
        "getbufsize", "geterr", "geterrcall", "errstate", "asmatrix", "matrix", "bmat", "recarray",
        "record",
        # DIV-COV-GRAM-NO-FMA: fnp's Gram path is within 1e-12 of numpy's FMA-contracted BLAS
        "cov", "corrcoef"}
def make(dt, shape=(4, 6)):
    kind = np.dtype(dt).kind
    if kind == "c":
        base = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    elif kind == "f":
        base = rng.standard_normal(shape) * 5
    elif kind == "m":
        return rng.integers(-500, 500, shape).astype(np.dtype(dt).newbyteorder("<")).astype(dt)
    else:
        base = rng.integers(0, 50, shape)
    return base.astype(np.dtype(dt).newbyteorder("<")).astype(dt)
KINDS = ["f8", "i4", "c16", "f4", "u2", "i8", "f2", "c8", "i2", "m8[s]"]
# 4x6 and 160x128: the larger one clears the 2**14-element floors most native routes use,
# which is where several of the byte-order defects lived; the four functions with quadratic
# outputs are left out of it.
OPS = {(order + k, shape): make(order + k, shape) for shape in ((4, 6), (160, 128))
       for order in (">", "<") for k in KINDS}
QUADRATIC = {"outer", "kron", "meshgrid", "diagflat"}
names = [n for n in np.__all__ if callable(getattr(np, n, None)) and n not in SKIP
         and not inspect.isclass(getattr(np, n))]
class Raised:
    def __init__(self, ex): self.name = type(ex).__name__
def run(fn):
    try:
        return fn()
    except BaseException as ex:  # a Rust panic surfaces as PanicException, a BaseException
        return Raised(ex)
def same(r, s):
    if isinstance(s, Raised) or isinstance(r, Raised):
        return isinstance(s, Raised) and isinstance(r, Raised) and r.name == s.name
    if isinstance(s, (tuple, list)):
        return isinstance(r, (tuple, list)) and len(r) == len(s) and all(same(a, b) for a, b in zip(r, s))
    if type(r) is not type(s):
        return False
    if s is None or isinstance(s, (str, bool, int, float)):
        return r == s or (s != s and r != r)
    try:
        r2, s2 = np.asarray(r), np.asarray(s)
    except Exception:
        return repr(r) == repr(s)
    if r2.dtype != s2.dtype or r2.shape != s2.shape:
        return False
    if r2.dtype.kind == "O":
        return repr(r) == repr(s)
    return r2.tobytes() == s2.tobytes()
bad = []
cells = 0
for name in names:
    npf, fnf = getattr(np, name), getattr(fnp, name, None)
    if fnf is None:
        continue
    for (dt, shape), a in OPS.items():
        if shape != (4, 6) and name in QUADRATIC:
            continue
        for args, kw in (((a,), {}), ((a, a[::-1].copy()), {}), ((a,), {"axis": 0}), ((a,), {"axis": -1})):
            s = run(lambda: npf(*args, **kw))
            if isinstance(s, Raised):
                continue
            cells += 1
            if not same(run(lambda: fnf(*args, **kw)), s):
                bad.append(f"{name}{len(args)}{kw} {dt} {shape}")
print(len(names), cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(3, ' ');
    let (functions, cells, bad) = (
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or(""),
    );
    assert!(
        functions.parse::<usize>().unwrap_or(0) >= 300,
        "numpy callables drifted: {result}"
    );
    assert!(
        cells.parse::<usize>().unwrap_or(0) >= 14000,
        "cell table drifted: {result}"
    );
    assert_eq!(bad, "[]", "array functions must match numpy: {result}");
    Ok(())
}

/// Every numpy.__all__ callable on DEGENERATE operands: empty ((0,), (0, 4), (4, 0), (0, 0),
/// (2, 0, 3)), 0-d, and length-1 ((1,), (1, 1)) arrays of six dtypes, called as f(a), f(a, a),
/// f(a, axis=0) and f(a, axis=-1). Each arm gets fresh copies of its arguments. fnp must match
/// numpy's result type, dtype, shape and bytes, or raise the same exception type, and must never
/// panic (~23,700 cells). Before the fixes: argmin/argmax/nanargmin/nanargmax(axis=0) of a (4, 0)
/// array and cov/corrcoef of a (0, 4) one PANICKED on a zero chunk size; argsort and angle of a
/// 0-d array raised; nan_to_num of a 0-d int returned an array, not a scalar;
/// concatenate(arrays, None) stacked on axis 0 instead of flattening; histogram_bin_edges and
/// compress answered bins/conditions numpy rejects as not 1-D; clip(a, a_min) answered numpy's
/// "missing a_max" TypeError; bincount of an empty array returned float64 counts; histogram2d of
/// float16 returned float64 edges; unpackbits refused a positional axis; and rot90, diag,
/// diagflat, vander, einsum_path, trim_zeros and the index helpers raised PyO3's argument
/// TypeError where numpy raises or answers differently (336 cells and 10 panics in all).
#[test]
fn array_functions_match_numpy_on_empty_zero_dim_and_length_one_operands() -> Result<(), String> {
    let script = fnp_script(
        r#"
import copy, inspect, warnings
warnings.simplefilter("ignore")
SKIP = {"save", "savez", "savez_compressed", "savetxt", "load", "loadtxt", "genfromtxt", "fromfile",
        "memmap", "seterr", "seterrcall", "setbufsize", "set_printoptions", "printoptions", "info",
        "show_config", "show_runtime", "test", "from_dlpack", "frompyfunc", "vectorize", "piecewise",
        "apply_along_axis", "apply_over_axes", "fromfunction", "fromregex", "nditer", "nested_iters",
        "empty", "empty_like", "ndarray", "broadcast", "iinfo", "finfo", "dtype", "getbufsize",
        "geterr", "geterrcall", "errstate", "asmatrix", "matrix", "bmat", "recarray", "record",
        "put", "place", "putmask", "copyto", "fill_diagonal", "busday_offset", "busday_count",
        "is_busday", "shares_memory", "may_share_memory"}
SHAPES = [(0,), (0, 4), (4, 0), (0, 0), (), (1,), (1, 1), (2, 0, 3)]
DTS = ["f8", "i8", "?", "c16", "f2", "u1"]
names = [n for n in np.__all__ if callable(getattr(np, n, None)) and n not in SKIP
         and not inspect.isclass(getattr(np, n))]
class Raised:
    def __init__(self, ex): self.name = type(ex).__name__
def run(fn, args, kw):
    try:
        return fn(*copy.deepcopy(args), **kw)
    except BaseException as ex:  # a Rust panic surfaces as PanicException, a BaseException
        return Raised(ex)
def same(r, s):
    if isinstance(s, Raised) or isinstance(r, Raised):
        return isinstance(s, Raised) and isinstance(r, Raised) and r.name == s.name
    if isinstance(s, (tuple, list)):
        return isinstance(r, (tuple, list)) and len(r) == len(s) and all(same(a, b) for a, b in zip(r, s))
    if type(r) is not type(s):
        return False
    if s is None or isinstance(s, (str, bool, int, float)):
        return r == s or (s != s and r != r)
    try:
        r2, s2 = np.asarray(r), np.asarray(s)
    except Exception:
        return repr(r) == repr(s)
    if r2.dtype != s2.dtype or r2.shape != s2.shape:
        return False
    if r2.dtype.kind == "O":
        return repr(r) == repr(s)
    return r2.tobytes() == s2.tobytes()
rng = np.random.default_rng(5)
bad, panics, cells = [], [], 0
for name in names:
    npf, fnf = getattr(np, name), getattr(fnp, name, None)
    if fnf is None:
        continue
    for shape in SHAPES:
        for dt in DTS:
            a = (rng.random(shape) * 4).astype(dt) if shape else np.array(rng.random() * 4).astype(dt)
            for label, args, kw in (("1", (a,), {}), ("2", (a, a.copy()), {}), ("ax0", (a,), {"axis": 0}),
                                    ("ax-1", (a,), {"axis": -1})):
                s = run(npf, args, kw)
                r = run(fnf, args, kw)
                if isinstance(s, Raised) and s.name == "TypeError" and isinstance(r, Raised):
                    continue
                cells += 1
                if isinstance(r, Raised) and r.name == "PanicException":
                    panics.append(f"{name}{label} {dt}{shape}")
                elif not same(r, s):
                    bad.append(f"{name}{label} {dt}{shape}: fnp={getattr(r, 'name', type(r).__name__)} "
                               f"numpy={getattr(s, 'name', type(s).__name__)}")
print(len(names), cells, panics, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(3, ' ');
    let (functions, cells, rest) = (
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or(""),
    );
    assert!(
        functions.parse::<usize>().unwrap_or(0) >= 300,
        "numpy callables drifted: {result}"
    );
    assert!(
        cells.parse::<usize>().unwrap_or(0) >= 20000,
        "cell table drifted: {result}"
    );
    assert_eq!(
        rest, "[] []",
        "degenerate operands must never panic and must match numpy (panics, mismatches): {result}"
    );
    Ok(())
}

/// SCALAR operands: every numpy.__all__ callable on Python scalars (int, float, bool, complex),
/// NumPy scalars (float64/32/16, int64/8, uint8, bool_, complex128) and 0-d arrays, called as
/// f(x), f(x, x) and f(x, axis=0). fnp must return numpy's result TYPE (a NumPy scalar is not a
/// 0-d array is not a Python float), dtype and bytes, or raise the same exception type (~6,000
/// cells). Before the fix, 32 cells differed. The worst were silent wrong answers: linspace/
/// geomspace/logspace with `np.complex128` or `np.complex64` endpoints returned float64 arrays of
/// the REAL parts (a NumPy complex scalar implements `__float__`), and
/// nanpercentile/nanquantile of a `np.float16` scalar returned float64. ediff1d(True),
/// eye(True) and identity(True) answered where numpy refuses a bool, and
/// tril/triu/diag_indices_from and rollaxis raised a different exception type than numpy for a
/// non-array argument.
#[test]
fn array_functions_match_numpy_on_python_and_numpy_scalars() -> Result<(), String> {
    let script = fnp_script(
        r#"
import copy, inspect, warnings
warnings.simplefilter("ignore")
SKIP = {"save", "savez", "savez_compressed", "savetxt", "load", "loadtxt", "genfromtxt", "fromfile",
        "memmap", "seterr", "seterrcall", "setbufsize", "set_printoptions", "printoptions", "info",
        "show_config", "show_runtime", "test", "from_dlpack", "frompyfunc", "vectorize", "piecewise",
        "apply_along_axis", "apply_over_axes", "fromfunction", "fromregex", "nditer", "nested_iters",
        "empty", "empty_like", "ndarray", "broadcast", "iinfo", "finfo", "dtype", "getbufsize",
        "geterr", "geterrcall", "errstate", "asmatrix", "matrix", "bmat", "recarray", "record",
        "put", "place", "putmask", "copyto", "fill_diagonal", "busday_offset", "busday_count",
        "is_busday", "shares_memory", "may_share_memory"}
VALUES = {"py_int": 3, "py_float": 2.5, "py_neg": -1.5, "py_bool": True, "py_complex": 1.5 - 2j,
          "np_f8": np.float64(2.5), "np_f4": np.float32(-1.25), "np_f2": np.float16(0.5),
          "np_i8": np.int64(-7), "np_u1": np.uint8(200), "np_i1": np.int8(-3), "np_b": np.bool_(True),
          "np_c16": np.complex128(1 - 1j), "zd_f8": np.array(2.5), "zd_i4": np.array(4, dtype=np.int32)}
names = [n for n in np.__all__ if callable(getattr(np, n, None)) and n not in SKIP
         and not inspect.isclass(getattr(np, n))]
class Raised:
    def __init__(self, ex): self.name = type(ex).__name__
def run(fn, args, kw):
    try:
        return fn(*copy.deepcopy(args), **kw)
    except BaseException as ex:
        return Raised(ex)
def same(r, s):
    if isinstance(s, Raised) or isinstance(r, Raised):
        return isinstance(s, Raised) and isinstance(r, Raised) and r.name == s.name
    if isinstance(s, (tuple, list)):
        return type(r) is type(s) and len(r) == len(s) and all(same(a, b) for a, b in zip(r, s))
    if type(r) is not type(s):
        return False
    if s is None or isinstance(s, (str, bool, int, float, complex)):
        return r == s or (s != s and r != r)
    try:
        r2, s2 = np.asarray(r), np.asarray(s)
    except Exception:
        return repr(r) == repr(s)
    if r2.dtype.kind == "O":
        return repr(r) == repr(s)
    return r2.dtype == s2.dtype and r2.shape == s2.shape and r2.tobytes() == s2.tobytes()
bad, cells = [], 0
for name in names:
    npf, fnf = getattr(np, name), getattr(fnp, name, None)
    if fnf is None:
        continue
    for label, x in VALUES.items():
        for form, args, kw in (("1", (x,), {}), ("2", (x, x), {}), ("ax0", (x,), {"axis": 0})):
            s = run(npf, args, kw)
            r = run(fnf, args, kw)
            if isinstance(s, Raised) and s.name == "TypeError" and isinstance(r, Raised):
                continue
            cells += 1
            if not same(r, s):
                bad.append(f"{name}{form} {label}: fnp={getattr(r, 'name', type(r).__name__)} "
                           f"numpy={getattr(s, 'name', type(s).__name__)}")
print(len(names), cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(3, ' ');
    let (functions, cells, bad) = (
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or(""),
    );
    assert!(
        functions.parse::<usize>().unwrap_or(0) >= 300,
        "numpy callables drifted: {result}"
    );
    assert!(
        cells.parse::<usize>().unwrap_or(0) >= 5000,
        "cell table drifted: {result}"
    );
    assert_eq!(
        bad, "[]",
        "scalar operands must behave as in numpy: {result}"
    );
    Ok(())
}

/// ndarray SUBCLASSES: every numpy.__all__ callable on a user subclass (with
/// `__array_finalize__`), an `np.matrix` and an `np.ma.MaskedArray` of f8/i8/bool, called as
/// f(a), f(a, a) and f(a, axis=0). fnp must return numpy's exact result CLASS, dtype, shape,
/// bytes and (masked) mask, or raise the same exception type. Each arm runs in a forked child:
/// numpy 2.4.3 itself segfaults on `np.dstack(np.matrix(...))`, so a cell where BOTH arms die
/// is numpy's and skipped, while fnp dying alone fails. Before the fix, 163 cells differed:
/// native routes read a subclass's buffer as a plain ndarray, returning base ndarrays where
/// numpy keeps the subclass (meshgrid, concatenate of a matrix, ediff1d, diag, modf, degrees,
/// logical_not, set ops, ...) and computing on masked-out data (`trace` of a MaskedArray).
#[test]
fn array_functions_match_numpy_on_ndarray_subclasses() -> Result<(), String> {
    let script = fnp_script(
        r#"
import copy, hashlib, inspect, os, signal, warnings
warnings.simplefilter("ignore")
SKIP = {"save", "savez", "savez_compressed", "savetxt", "load", "loadtxt", "genfromtxt", "fromfile",
        "memmap", "seterr", "seterrcall", "setbufsize", "set_printoptions", "printoptions", "info",
        "show_config", "show_runtime", "test", "from_dlpack", "frompyfunc", "vectorize", "piecewise",
        "apply_along_axis", "apply_over_axes", "fromfunction", "fromregex", "nditer", "nested_iters",
        "empty", "empty_like", "ndarray", "broadcast", "iinfo", "finfo", "dtype", "getbufsize",
        "geterr", "geterrcall", "errstate", "asmatrix", "matrix", "bmat", "recarray", "record",
        "put", "place", "putmask", "copyto", "fill_diagonal", "busday_offset", "busday_count",
        "is_busday", "shares_memory", "may_share_memory"}
class Tagged(np.ndarray):
    def __array_finalize__(self, obj):
        self.tag = getattr(obj, "tag", "t")
rng = np.random.default_rng(9)
base = {"f8": rng.standard_normal((3, 4)) * 3, "i8": rng.integers(-5, 9, (3, 4)),
        "?": rng.random((3, 4)) < 0.5}
OPS = {}
for dt, a in base.items():
    OPS[("tagged", dt)] = a.view(Tagged)
    OPS[("matrix", dt)] = np.asmatrix(a)
    OPS[("masked", dt)] = np.ma.masked_array(a, mask=rng.random((3, 4)) < 0.25)
names = [n for n in np.__all__ if callable(getattr(np, n, None)) and n not in SKIP
         and not inspect.isclass(getattr(np, n))]
def digest(fn, args, kw):
    try:
        value = fn(*copy.deepcopy(args), **kw)
    except BaseException as ex:
        return "raised " + type(ex).__name__
    def one(value):
        if isinstance(value, (tuple, list)):
            return type(value).__name__ + "[" + ",".join(one(v) for v in value) + "]"
        if value is None or isinstance(value, (str, bool, int, float, complex)):
            return type(value).__name__ + ":" + repr(value)
        h = hashlib.sha256(type(value).__name__.encode())
        if isinstance(value, np.ma.MaskedArray):
            h.update(np.ma.getmaskarray(value).tobytes())
            value = np.ma.getdata(value)
        try:
            arr = np.asarray(value)
            h.update(f"{arr.dtype.str}{arr.shape}".encode())
            h.update(repr(value).encode() if arr.dtype.kind == "O" else arr.tobytes())
        except Exception:
            h.update(repr(value).encode())
        return type(value).__name__ + ":" + h.hexdigest()[:16]
    return one(value)
def isolated(fn, args, kw):
    rd, wr = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(rd)
        signal.alarm(60)
        os.write(wr, digest(fn, args, kw).encode())
        os._exit(0)
    os.close(wr)
    with os.fdopen(rd) as pipe:
        text = pipe.read()
    _, status = os.waitpid(pid, 0)
    return "SIGNAL" if os.WIFSIGNALED(status) else text
bad, cells = [], 0
for name in names:
    npf, fnf = getattr(np, name), getattr(fnp, name, None)
    if fnf is None:
        continue
    for (kind, dt), a in OPS.items():
        for label, args, kw in (("1", (a,), {}), ("2", (a, a.copy()), {}), ("ax0", (a,), {"axis": 0})):
            s = isolated(npf, args, kw)
            if s.startswith("raised") or s == "SIGNAL":
                continue
            cells += 1
            r = isolated(fnf, args, kw)
            if r != s:
                bad.append(f"{name}{label} {kind}/{dt}: fnp={r[:40]} numpy={s[:40]}")
print(len(names), cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(3, ' ');
    let (functions, cells, bad) = (
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or(""),
    );
    assert!(
        functions.parse::<usize>().unwrap_or(0) >= 300,
        "numpy callables drifted: {result}"
    );
    assert!(
        cells.parse::<usize>().unwrap_or(0) >= 2500,
        "cell table drifted: {result}"
    );
    assert_eq!(
        bad, "[]",
        "ndarray subclasses must behave as in numpy: {result}"
    );
    Ok(())
}

/// Floating-point EVENTS, not just values (bead .26): every numpy.__all__ callable on operands
/// that provoke numpy's warnings (0, -1, 1e308, +-inf, NaN, -0.0 in f8/f4/f2/c16, an all-NaN
/// array, an empty one, and integer/bool arrays - each (2, 4), and the floats again 1-D, 4096
/// long and 64x64), called as f(a), f(a, a) and f(a, axis=0) under
/// numpy's default errstate, under `errstate(all='raise', under='ignore')` and under
/// `errstate(all='ignore')`. fnp must end the same way as numpy (ok or the same exception type)
/// with the same set of (category, message) warnings. Under 'ignore' that set is empty, which is
/// the negative case. Each arm gets fresh copies (numpy's rot90 reduces an array `k` in place).
/// The sweep found 35 default-errstate cells (and the same 35 under `raise`) in 21 functions whose
/// native kernels return numpy's values silently; the 1-D / 4096 / 64x64 operands then found the
/// same class in sum, trace, trapezoid, polyval, vander and the float16 cumsum scan. FIXED, per
/// ROUTE, never per name: each native route that computes silently hands a non-finite result to
/// numpy (`native_or_numpy_on_non_finite`), and the f64/f32 nancumsum/nancumprod chains replay
/// their categories exactly (`report_native_accumulation_fp_events`, skip_nan). A route that
/// already returns numpy's own result on a hazard (the float16 diff) is left alone, and i0 wraps
/// only a native result (`native_unary_promoting_route`). A name-level recompute (6a050102) was
/// reverted: it warned twice wherever the native function had already returned numpy's own
/// result (fallbacks, nanvar's all-NaN deferral). RESIDUAL below is the ratchet for anything
/// still open: the test fails on any divergence outside it AND on a name in it that now matches.
/// Underflow is out of scope: it leaves no NaN/inf in the result to detect (arctan2/nextafter
/// under a non-default `under=`).
#[test]
fn array_functions_match_numpy_fp_warnings_and_errors() -> Result<(), String> {
    let script = fnp_script(
        r#"
import copy, inspect, warnings
SKIP = {"save", "savez", "savez_compressed", "savetxt", "load", "loadtxt", "genfromtxt", "fromfile",
        "memmap", "seterr", "seterrcall", "setbufsize", "set_printoptions", "printoptions", "info",
        "show_config", "show_runtime", "test", "from_dlpack", "frompyfunc", "vectorize", "piecewise",
        "apply_along_axis", "apply_over_axes", "fromfunction", "fromregex", "nditer", "nested_iters",
        "empty", "empty_like", "ndarray", "broadcast", "iinfo", "finfo", "dtype", "getbufsize",
        "geterr", "geterrcall", "errstate", "asmatrix", "matrix", "bmat", "recarray", "record",
        "put", "place", "putmask", "copyto", "fill_diagonal"}
SPECIAL = [0.0, -1.0, 1e308, np.inf, -np.inf, np.nan, 2.0, -0.0]
with np.errstate(all="ignore"):
    OPS = {
        "f8": np.array(SPECIAL, dtype="f8").reshape(2, 4),
        "f4": np.array(SPECIAL, dtype="f8").astype("f4").reshape(2, 4),
        "f2": np.array(SPECIAL, dtype="f8").astype("f2").reshape(2, 4),
        "i8": np.array([0, -1, 2, 3, 0, 5, -7, 1], dtype="i8").reshape(2, 4),
        "u1": np.array([0, 1, 2, 3, 0, 5, 7, 255], dtype="u1").reshape(2, 4),
        "?": np.array([True, False, True, True, False, False, True, False]).reshape(2, 4),
        "c16": (np.array(SPECIAL) + 1j * np.array(SPECIAL[::-1])).reshape(2, 4),
        "nan": np.full((2, 4), np.nan),
        "empty": np.empty((0, 4)),
        # 1-D and 4096-element operands reach the routes a (2, 4) one never does: the 1-D diff,
        # gradient and trapezoid kernels, size-gated sum/nansum trees, a 64x64 trace.
        "f8_1d": np.array(SPECIAL, dtype="f8"),
        "f4_1d": np.array(SPECIAL, dtype="f8").astype("f4"),
        "f2_1d": np.array(SPECIAL, dtype="f8").astype("f2"),
        "c16_1d": np.array(SPECIAL) + 1j * np.array(SPECIAL[::-1]),
        "f8_4k": np.tile(np.array(SPECIAL, dtype="f8"), 512),
        "f4_4k": np.tile(np.array(SPECIAL, dtype="f8").astype("f4"), 512),
        "f8_64x64": np.tile(np.array(SPECIAL, dtype="f8"), 512).reshape(64, 64),
    }
names = [n for n in np.__all__ if callable(getattr(np, n, None)) and n not in SKIP
         and not inspect.isclass(getattr(np, n))]
def run(fn, args, kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fn(*copy.deepcopy(args), **kw)
            outcome = "ok"
        except BaseException as ex:
            outcome = type(ex).__name__
    return outcome, sorted({(w.category.__name__, str(w.message)) for w in caught})
MODES = {"default": {}, "raise": {"all": "raise", "under": "ignore"}, "ignore": {"all": "ignore"}}
bad = []
cells = 0
for mode, settings in MODES.items():
    with np.errstate(**settings):
        for name in names:
            npf, fnf = getattr(np, name), getattr(fnp, name, None)
            if fnf is None:
                continue
            for dt, a in OPS.items():
                for label, args, kw in (("1", (a,), {}), ("2", (a, a[::-1].copy()), {}),
                                        ("ax0", (a,), {"axis": 0})):
                    s = run(npf, args, kw)
                    if s[0] not in ("ok", "FloatingPointError") and not s[1]:
                        continue
                    cells += 1
                    r = run(fnf, args, kw)
                    if r != s:
                        bad.append((name, f"{mode} {name}{label} {dt}: fnp={r} numpy={s}"))
# Bead .26's open residual: native kernels not yet converted to the per-ROUTE non-finite recompute.
# A divergence outside this set fails, and so does a name in it that no longer diverges - the list
# can only shrink.
RESIDUAL = set()
unexpected = [text for name, text in bad if name not in RESIDUAL]
stale = sorted(RESIDUAL - {name for name, _ in bad})
print(len(names), cells, "|", unexpected, "|", stale)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut parts = result.trim().split(" | ");
    let (head, unexpected, stale) = (
        parts.next().unwrap_or(""),
        parts.next().unwrap_or(""),
        parts.next().unwrap_or(""),
    );
    let mut fields = head.split(' ');
    let (functions, cells) = (fields.next().unwrap_or("0"), fields.next().unwrap_or("0"));
    assert!(
        functions.parse::<usize>().unwrap_or(0) >= 300,
        "numpy callables drifted: {result}"
    );
    assert!(
        cells.parse::<usize>().unwrap_or(0) >= 14000,
        "cell table drifted: {result}"
    );
    assert_eq!(
        unexpected, "[]",
        "fp warnings/errors must match numpy outside bead .26's named residual: {result}"
    );
    assert_eq!(
        stale, "[]",
        "these residual names now match numpy - drop them from RESIDUAL: {result}"
    );
    Ok(())
}

/// The parallel complex and float16 accumulation/reduction routes are size-gated, so the
/// small operands above never reach them: on a (4096, 256) complex128 and a (2048, 256) float16
/// holding +-inf / 1e308 / 60000, cumsum, cumprod, nancumsum, nancumprod, sum, prod, nansum,
/// nanprod and cumulative_sum/_prod must warn (default errstate) or raise (`all='raise'`) exactly
/// as numpy does; 26 cells were silent before the per-route fix (bead .26). The negative case:
/// unit-magnitude complex and small float16 operands, where both must stay silent - an
/// implementation that warns whenever it sees a large operand fails there.
#[test]
fn large_complex_and_float16_accumulations_report_fp_events_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
inf = np.inf
OPS = {
    "c16": np.tile(np.array([[inf + 0j, -inf + 0j, 1e308 + 0j, 0j]]), (4096, 64)),
    "f2": np.tile(np.array([60000.0, 60000.0, 1.0, 2.0], dtype="f2"), (2048, 64)),
    "c16_clean": np.tile(np.array([[1j, -1 + 0j, 1 + 0j, -1j]]), (4096, 64)),
    "f2_clean": np.tile(np.array([1.0, -1.0, 1.0, 0.5], dtype="f2"), (2048, 64)),
}
NAMES = ["cumsum", "cumprod", "nancumsum", "nancumprod", "sum", "prod", "nansum", "nanprod",
         "cumulative_sum", "cumulative_prod"]
def run(fn, a, kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fn(a.copy(), **kw)
            outcome = "ok"
        except BaseException as ex:
            outcome = type(ex).__name__
    return outcome, sorted({(w.category.__name__, str(w.message)) for w in caught})
bad, cells, warned = [], 0, 0
for mode, settings in (("default", {}), ("raise", {"all": "raise", "under": "ignore"})):
    with np.errstate(**settings):
        for name in NAMES:
            for key, a in OPS.items():
                for kw in ({"axis": -1}, {"axis": 0}):
                    s = run(getattr(np, name), a, kw)
                    r = run(getattr(fnp, name), a, kw)
                    cells += 1
                    warned += bool(s[1]) or s[0] == "FloatingPointError"
                    if key.endswith("_clean") and (s[1] or s[0] != "ok"):
                        bad.append(f"clean operand not clean in numpy: {mode} {name} {key} {kw} {s}")
                    if r != s:
                        bad.append(f"{mode} {name} {key} {kw}: fnp={r} numpy={s}")
print(cells, warned, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(3, ' ');
    let (cells, warned, bad) = (
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or(""),
    );
    assert_eq!(cells, "160", "cell table drifted: {result}");
    assert!(
        warned.parse::<usize>().unwrap_or(0) >= 40,
        "the hazard operands stopped provoking numpy: {result}"
    );
    assert_eq!(
        bad, "[]",
        "large complex/float16 accumulations must report like numpy: {result}"
    );
    Ok(())
}

/// Business-day and datetime helpers, bit packing, the indexing helpers over eight dtypes
/// (incl. unicode and datetime64), fft with every norm, the six polynomial classes, stride
/// tricks and fnp.testing assertions, compared with numpy by type, dtype, shape, layout, bytes
/// and exception type (218 cases). Before the fixes (bead .8): take_along_axis raised
/// AttributeError on unicode and datetime64 arrays (it viewed its gather back through
/// `numpy.<dtype.name>`, and 'str64' / 'datetime64[D]' are no numpy attributes), and choose of
/// int8 choices returned int64 (its extract fallback canonicalised narrow integers).
#[test]
fn datetime_packing_indexing_fft_polynomial_and_testing_helpers_match_numpy() -> Result<(), String>
{
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(8)
cases = []
def add(name, fn):
    cases.append((name, fn))
days = np.datetime64("2024-01-01") + rng.integers(-400, 400, 60).astype("timedelta64[D]")
days[5] = np.datetime64("NaT")
ends = days + rng.integers(0, 90, 60).astype("timedelta64[D]")
hol = ["2024-01-15", "2024-02-19", "2024-05-27", "2024-07-04"]
for roll in ("raise", "nat", "forward", "following", "backward", "preceding", "modifiedfollowing", "modifiedpreceding"):
    add(f"busday_offset {roll}", lambda m, r=roll: m.busday_offset(days, 3, roll=r))
    add(f"busday_offset holidays {roll}", lambda m, r=roll: m.busday_offset(days, -7, roll=r, holidays=hol))
for mask in ("1111100", "1010101", [1, 1, 1, 1, 1, 1, 0], "0000011"):
    add(f"busday_count {mask}", lambda m, w=mask: m.busday_count(days, ends, weekmask=w))
    add(f"is_busday {mask}", lambda m, w=mask: m.is_busday(days, weekmask=w, holidays=hol))
add("busday_count reversed", lambda m: m.busday_count(ends, days))
add("datetime_as_string", lambda m: m.datetime_as_string(days, unit="D"))
add("datetime_as_string tz", lambda m: m.datetime_as_string(days.astype("M8[s]"), unit="m", timezone="UTC"))
add("datetime floor_divide", lambda m: m.floor_divide(ends - days, np.timedelta64(7, "D")))
for dt in (np.uint8, np.int32, bool):
    bits = (rng.random((7, 13)) < 0.5).astype(dt)
    for order in ("big", "little"):
        for axis in (None, 0, 1):
            add(f"packbits {dt.__name__} {order} {axis}", lambda m, b=bits, o=order, a=axis: m.packbits(b, axis=a, bitorder=o))
packed = rng.integers(0, 256, (5, 3), dtype=np.uint8)
for order in ("big", "little"):
    for count in (None, 5, -3, 20):
        add(f"unpackbits {order} {count}", lambda m, o=order, c=count: m.unpackbits(packed, axis=1, count=c, bitorder=o))
for dt in (np.float64, np.float32, np.int8, np.uint64, np.complex128, bool, "U3", "M8[D]"):
    if dt == "U3":
        base = np.array([["ab", "c", "de"], ["f", "gh", "i"]])
    elif dt == "M8[D]":
        base = days[:6].reshape(2, 3)
    else:
        base = (rng.standard_normal((2, 3)) * 10).astype(dt)
    tag = getattr(dt, "__name__", dt)
    idx = np.array([[2, 0, 1], [1, 1, 0]])
    eye = np.eye(2, 3, dtype=bool)
    add(f"take_along_axis {tag}", lambda m, b=base, i=idx: m.take_along_axis(b, i, axis=1))
    add(f"take_along_axis None {tag}", lambda m, b=base: m.take_along_axis(b, np.array([5, 0, 3]), axis=None))
    add(f"compress {tag}", lambda m, b=base: m.compress([True, False, True], b, axis=1))
    add(f"where {tag}", lambda m, b=base: m.where(np.array([[True, False, True], [False, True, False]]), b, b[:, ::-1]))
    add(f"select {tag}", lambda m, b=base, e=eye: m.select([e], [b], default=b[0, 0]))
    add(f"extract {tag}", lambda m, b=base, e=eye: m.extract(e, b))
    add(f"place {tag}", lambda m, b=base, e=eye: (lambda x: (m.place(x, e, [b[1, 2]]), x)[1])(b.copy()))
    add(f"putmask {tag}", lambda m, b=base, e=eye: (lambda x: (m.putmask(x, e, b[::-1]), x)[1])(b.copy()))
    add(f"put_along_axis {tag}", lambda m, b=base, i=idx: (lambda x: (m.put_along_axis(x, i[:, :1], b[:, -1:], axis=1), x)[1])(b.copy()))
    add(f"choose {tag}", lambda m, b=base: m.choose(np.array([[0, 1, 0], [1, 0, 1]]), [b, b[::-1]]))
    add(f"repeat {tag}", lambda m, b=base: m.repeat(b, [1, 2, 0], axis=1))
    add(f"roll {tag}", lambda m, b=base: m.roll(b, -4))
    add(f"rot90 {tag}", lambda m, b=base: m.rot90(b, 3))
x = rng.standard_normal(64)
X = rng.standard_normal((8, 6)) + 1j * rng.standard_normal((8, 6))
for norm in (None, "ortho", "forward"):
    for f in ("fft", "ifft", "rfft", "hfft", "ihfft"):
        add(f"fft.{f} {norm}", lambda m, f=f, n=norm: getattr(m.fft, f)(x, norm=n))
    add(f"fft.irfft n=70 {norm}", lambda m, n=norm: m.fft.irfft(m.fft.rfft(x), n=70, norm=n))
    add(f"fft.fft2 {norm}", lambda m, n=norm: m.fft.fft2(X, norm=n))
    add(f"fft.rfftn s {norm}", lambda m, n=norm: m.fft.rfftn(X.real, s=(10, 4), norm=n))
add("fft.fft f32", lambda m: m.fft.fft(x.astype(np.float32)))
add("fft.fft prime n", lambda m: m.fft.fft(x[:61]))
add("fftfreq", lambda m: m.fft.fftfreq(9, d=0.3))
add("fftshift", lambda m: m.fft.fftshift(X, axes=1))
for cls in ("Polynomial", "Chebyshev", "Legendre", "Hermite", "HermiteE", "Laguerre"):
    add(f"{cls} eval", lambda m, c=cls: getattr(m.polynomial, c)([1, -2, 0.5, 3])(x[:10]))
    add(f"{cls} deriv integ", lambda m, c=cls: getattr(m.polynomial, c)([1, -2, 0.5, 3]).deriv(2).integ(1, k=[0.5]).coef)
    add(f"{cls} fit", lambda m, c=cls: getattr(m.polynomial, c).fit(x[:30], np.sin(x[:30]), 4).coef)
    add(f"{cls} mul pow", lambda m, c=cls: (getattr(m.polynomial, c)([1, 2]) * getattr(m.polynomial, c)([0, 1, 3]) ** 2).coef)
add("sliding_window_view", lambda m: m.lib.stride_tricks.sliding_window_view(np.arange(10), 3)[::2])
add("as_strided", lambda m: m.lib.stride_tricks.as_strided(np.arange(10), shape=(4, 3), strides=(16, 8)))
for f, args in (("assert_array_equal", (np.arange(3), np.array([0, 1, 3]))), ("assert_allclose", (np.array([1.0]), np.array([1.1]))),
                ("assert_equal", ({"a": 1}, {"a": 2})), ("assert_array_less", (np.arange(3), np.arange(1, 4))),
                ("assert_string_equal", ("abc", "abd")), ("assert_approx_equal", (1.0, 1.0000001))):
    add(f"testing.{f}", lambda m, f=f, a=args: getattr(m.testing, f)(*a))
class Raised:
    def __init__(self, ex): self.name = type(ex).__name__
def same(r, s):
    if isinstance(s, (tuple, list)):
        return isinstance(r, (tuple, list)) and len(r) == len(s) and all(same(a, b) for a, b in zip(r, s))
    if type(r) is not type(s):
        return False
    if s is None or isinstance(s, (str, int, float, bool, tuple)):
        return r == s
    r2, s2 = np.asarray(r), np.asarray(s)
    if r2.dtype != s2.dtype or r2.shape != s2.shape:
        return False
    if isinstance(s, np.ndarray) and (r.flags.c_contiguous != s.flags.c_contiguous or r.flags.f_contiguous != s.flags.f_contiguous):
        return False
    return r2.tobytes() == s2.tobytes()
bad = []
for name, fn in cases:
    try:
        s = fn(np)
    except Exception as ex:
        s = Raised(ex)
    try:
        r = fn(fnp)
    except Exception as ex:
        r = Raised(ex)
    if isinstance(s, Raised) or isinstance(r, Raised):
        if not (isinstance(s, Raised) and isinstance(r, Raised) and s.name == r.name):
            bad.append(f"{name}: fnp={getattr(r, 'name', 'ok')} numpy={getattr(s, 'name', 'ok')}")
    elif not same(r, s):
        bad.append(name)
print(len(cases), bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cases, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert!(
        cases.parse::<usize>().unwrap_or(0) >= 200,
        "case table drifted: {result}"
    );
    assert_eq!(bad, "[]", "helper parity with numpy: {result}");
    Ok(())
}

/// fnp's ufunc objects report NumPy's docstring. The proxy class for natively implemented ufunc
/// names carried a Rust `///` class docstring, which CPython writes into the type dict after
/// PyO3's `__doc__` getter and so replaces it: `fnp.sin.__doc__` was fnp's implementation note.
#[test]
fn ufunc_objects_report_numpys_docstring() -> Result<(), String> {
    let script = fnp_script(
        r#"
bad = [name for name in ("sin", "cos", "exp", "log", "sqrt", "isnan", "modf", "ldexp", "frexp",
                         "add", "multiply", "power", "reciprocal", "square", "absolute")
       if getattr(fnp, name).__doc__ != getattr(np, name).__doc__]
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "ufunc docstrings must be numpy's: {result}"
    );
    Ok(())
}

/// Divergences the panic audit's hostile-input runs found next to its sites (bead rc0923 .20),
/// each against live numpy (outcome type and, when both succeed, dtype and bytes):
/// ravel_multi_index silently WRAPPED an i128 stride product past 2**127 and returned
/// 4611686018427387905 where numpy raises "invalid dims"; ediff1d of a Python list accepted a
/// float `to_end`/`to_begin` numpy refuses under same_kind (TypeError) and truncated it; take
/// raised ValueError for a uint64 index past int64 where numpy wraps it and raises IndexError;
/// put accepted a uint64 index array numpy refuses under safe casting (TypeError);
/// histogram_bin_edges treated an explicit `bins=None` as omitted (numpy: TypeError); and
/// linalg.cholesky raised TypeError for `upper=None` / `upper=1`, which numpy reads for
/// truthiness. Controls in the same table (in-range uint32/int32 indices, int `to_end`, omitted
/// `bins`, in-range ravel_multi_index, `upper=True`) must keep succeeding.
#[test]
fn panic_audit_neighbour_divergences_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def o(f):
    try:
        r = f()
        if isinstance(r, np.ndarray) or isinstance(r, np.generic):
            return ("ok", str(np.asarray(r).dtype), np.asarray(r).tobytes())
        return ("ok", repr(r))
    except Exception as e:
        return (type(e).__name__,)
def put_result(m, idx):
    a = np.arange(5)
    m.put(a, idx, 9)
    return a
big = np.array([2**63 + 5], dtype=np.uint64)
cases = {
    "ravel_multi_index overflow": lambda m: m.ravel_multi_index((np.array([1]),) * 3, (2**62,) * 3),
    "ravel_multi_index control": lambda m: m.ravel_multi_index((np.array([1]), np.array([2])), (3, 4)),
    "ediff1d list float to_end": lambda m: m.ediff1d([1, 4, 9], to_end=2.5),
    "ediff1d list nan to_begin": lambda m: m.ediff1d([1, 4, 9], to_begin=np.nan),
    "ediff1d list int to_end (control)": lambda m: m.ediff1d([1, 4, 9], to_end=7),
    "take uint64 past int64": lambda m: m.take(np.arange(5), big),
    "take uint64 in range (control)": lambda m: m.take(np.arange(5), np.array([3], dtype=np.uint64)),
    "put uint64 index": lambda m: put_result(m, np.array([1], dtype=np.uint64)),
    "put uint32 index (control)": lambda m: put_result(m, np.array([1], dtype=np.uint32)),
    "put int32 index (control)": lambda m: put_result(m, np.array([1], dtype=np.int32)),
    "histogram_bin_edges bins=None": lambda m: m.histogram_bin_edges([1, 2, 3], bins=None),
    "histogram_bin_edges omitted (control)": lambda m: m.histogram_bin_edges([1, 2, 3]),
    "cholesky upper=None": lambda m: m.linalg.cholesky(np.array([[4.0, 2.0], [2.0, 3.0]]), upper=None),
    "cholesky upper=1": lambda m: m.linalg.cholesky(np.array([[4.0, 2.0], [2.0, 3.0]]), upper=1),
    "cholesky upper=True (control)": lambda m: m.linalg.cholesky(np.array([[4.0, 2.0], [2.0, 3.0]]), upper=True),
}
bad = []
for label, f in cases.items():
    s, r = o(lambda: f(np)), o(lambda: f(fnp))
    if s != r:
        bad.append(f"{label}: fnp={r[:2]} numpy={s[:2]}")
print(len(cases), bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "15 []",
        "neighbour divergences must match numpy: {result}"
    );
    Ok(())
}

/// numpy's own test_umath (TestSpecialFloats, through the drop-in harness, bead rc0923 .8) found
/// unary domain errors that did not raise under `errstate(<category>='raise')` when the operand
/// was 0-d, a list, a Python float or a numpy scalar - the operands that take the extract path:
/// `sqrt(-inf)`, `log1p(-inf)`, `arcsin`/`arccos(+-inf)` (an `is_finite()` guard in fnp-ufunc's
/// event classifier and in the direct f64 bridge dropped infinities from numpy's invalid set) and
/// `square(float32(1e32))` (computed in f64, the overflow only appeared when narrowing). Every
/// cell compares the outcome with numpy for each operand form. The negative half is numpy's
/// test_unary_spurious_fpexception data: on those, fnp must stay as silent as numpy.
#[test]
fn unary_domain_errors_raise_like_numpy_for_every_operand_form() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
inf, nan = np.inf, np.nan
raise_cells = []
for dt in "efd":
    for f in ("log", "log2", "log10"):
        raise_cells += [(f, 0.0, dt, "divide"), (f, -inf, dt, "invalid"), (f, -1.0, dt, "invalid")]
    raise_cells += [("log1p", -1.0, dt, "divide"), ("log1p", -inf, dt, "invalid")]
    for f in ("arcsin", "arccos"):
        raise_cells += [(f, v, dt, "invalid") for v in (inf, -inf, 2.0, -2.0)]
    raise_cells.append(("square", {"e": 1e3, "f": 1e32, "d": 1e200}[dt], dt, "over"))
    for f in ("sin", "cos", "tan"):
        raise_cells += [(f, inf, dt, "invalid"), (f, -inf, dt, "invalid")]
    raise_cells += [("sqrt", -1.0, dt, "invalid"), ("sqrt", -inf, dt, "invalid"),
                    ("arctanh", 2.0, dt, "invalid"), ("arctanh", inf, dt, "invalid"),
                    ("arctanh", 1.0, dt, "divide"), ("arccosh", 0.5, dt, "invalid"),
                    ("arccosh", -inf, dt, "invalid"), ("reciprocal", 0.0, dt, "divide"),
                    ("exp", 1e4, dt, "over"), ("sinh", 1e4, dt, "over")]
def forms(value, dt):
    scalar = np.dtype(dt).type(value)
    yield "0-d", np.array(value, dtype=dt)
    yield "(1,)", np.full((1,), value, dtype=dt)
    yield "(3,)", np.full((3,), value, dtype=dt)
    yield "numpy scalar", scalar
    if dt == "d":
        yield "list", [value]
        yield "float", float(value)
def raised(m, f, a, category):
    with np.errstate(**{category: "raise"}):
        try:
            getattr(m, f)(a)
            return "ok"
        except BaseException as ex:
            return type(ex).__name__
bad, cells = [], 0
for f, value, dt, category in raise_cells:
    for form, a in forms(value, dt):
        cells += 1
        s, r = raised(np, f, a, category), raised(fnp, f, a, category)
        if s != r:
            bad.append(f"{f}({value}) {dt} {form} [{category}=raise]: fnp={r} numpy={s}")
datas = [[0.03], [-1.0], [1.0], [0.0], [-0.0], [0.5, 0.5, 0.5, nan], [nan, 1.0, 1.0, 1.0], [nan],
         [0.5, 0.5, 0.5, inf], [inf], [0.5, 0.5, 0.5, -inf], [-inf]]
names = ["arctanh", "arccosh", "tan", "sin", "log2", "log10", "log", "cos", "arcsin", "arccos",
         "sqrt", "log1p", "spacing", "reciprocal", "exp", "expm1", "tanh", "arctan", "square"]
def warned(fn, a):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(a)
    return sorted({str(w.message) for w in caught})
for name in names:
    for dt in "efd":
        for data in datas:
            for reps in (1, 32):
                cells += 1
                a = np.array(data * reps, dtype=dt)
                s, r = warned(getattr(np, name), a), warned(getattr(fnp, name), a)
                if s != r:
                    bad.append(f"{name} {dt} {data} x{reps}: fnp warned {r} numpy {s}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(2, ' ');
    assert!(
        fields.next().unwrap_or("0").parse::<usize>().unwrap_or(0) >= 1800,
        "cell table drifted: {result}"
    );
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "unary domain errors must raise and stay silent exactly as numpy: {result}"
    );
    Ok(())
}

/// NaN sign, payload and signaling bits through every unary ufunc (bead rc0923 .8, from numpy's
/// test_signaling_nan_exceptions under the drop-in harness).
/// - `sign` and `spacing` returned a canonical +NaN. numpy returns the input NaN itself
///   (`sign`), or `x - x` (`spacing`), keeping sign and payload. So `sign` of x86's default NaN
///   (-nan, from 0/0 or inf - inf) differed in bits.
/// - A 0-d or scalar float32 signaling NaN went through a float64 extract. That warned
///   "invalid value encountered in cast" from `isnan`/`isinf`/`isfinite`/`signbit`, and
///   returned quieted bits from `negative`/`fabs`/`absolute`.
///
/// 40 of these cells failed on 25feaae5. Known residual, not asserted here: numpy's hardware
/// raises "invalid" when ARITHMETIC ops (sin, log, sqrt, ...) read a signaling NaN, and fnp's
/// event classifier does not flag a NaN input as invalid.
#[test]
fn nan_sign_payload_and_signaling_bits_match_numpy_through_every_unary_ufunc() -> Result<(), String>
{
    let script = fnp_script(
        r#"
import warnings

def bits32(pattern):
    return np.frombuffer(np.array([pattern], dtype=np.uint32).tobytes(), dtype=np.float32)

def bits64(pattern):
    return np.frombuffer(np.array([pattern], dtype=np.uint64).tobytes(), dtype=np.float64)

def outcome(module, name, x):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = np.asarray(getattr(module, name)(x))
            return (r.dtype.str, r.shape, r.tobytes(), sorted(str(w.message) for w in caught))
        except Exception as ex:
            return (type(ex).__name__,)

def forms(arr):
    return (("0-d", arr.reshape(())), ("scalar", arr[0]), ("1-d", arr), ("x5", np.repeat(arr, 5)))

unary = sorted(n for n in dir(np) if isinstance(getattr(np, n), np.ufunc)
               and getattr(np, n).nin == 1 and getattr(np, n).nout == 1)
# Quiet NaNs with a sign or a payload: -nan is x86's default NaN (0/0, inf - inf).
quiet = {
    "-qnan32": bits32(0xFFC00000), "qnan32 payload": bits32(0x7FC01234),
    "-qnan64": bits64(0xFFF8000000000000), "qnan64 payload": bits64(0x7FF8000000001234),
}
# Signaling NaNs through the operations numpy leaves silent and bit-preserving.
signaling = {"snan32": bits32(0xFFBFE000), "snan64": bits64(0x7FF4000000000000)}
silent_ops = ("isnan", "isinf", "isfinite", "signbit", "negative", "fabs", "absolute", "sign")
cells = 0
bad = []
for label, arr in quiet.items():
    for form, x in forms(arr):
        for name in unary:
            cells += 1
            ours, theirs = outcome(fnp, name, x), outcome(np, name, x)
            if ours != theirs:
                bad.append(f"{name}({label}, {form}): fnp={ours} numpy={theirs}")
for label, arr in signaling.items():
    for form, x in forms(arr):
        for name in silent_ops:
            cells += 1
            ours, theirs = outcome(fnp, name, x), outcome(np, name, x)
            if ours != theirs:
                bad.append(f"{name}({label}, {form}): fnp={ours} numpy={theirs}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(2, ' ');
    assert!(
        fields.next().unwrap_or("0").parse::<usize>().unwrap_or(0) >= 900,
        "cell table drifted: {result}"
    );
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "NaN sign / payload / signaling bits must match numpy: {result}"
    );
    Ok(())
}

/// `frompyfunc` objects against numpy's (bead rc0923 .8, numpy's test_ufunc_override_mro under
/// the drop-in harness). The native object handed an operand overriding `__array_ufunc__`
/// straight to the Python function (TypeError from `A * int`). It also refused every keyword
/// (`out=`, `where=`), and had no `accumulate`/`outer`/`at`/`reduceat`/`types`/`nargs`. Those
/// now run on `numpy.frompyfunc` over the same callable. 13 of these 16 cells failed on
/// 25feaae5.
#[test]
fn frompyfunc_overrides_keywords_and_methods_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
class A:
    def __array_ufunc__(self, func, method, *inputs, **kwargs): return ("A", method)
class ASub(A):
    def __array_ufunc__(self, func, method, *inputs, **kwargs): return ("ASub", method)
class C:
    def __array_ufunc__(self, func, method, *inputs, **kwargs): return NotImplemented
class N:
    __array_ufunc__ = None
def mul3(a, b, c): return a * b * c
def add2(a, b): return a + b
def outcome(f):
    try:
        r = f()
        return ("ok", repr(r.tolist() if hasattr(r, "tolist") else r)[:80])
    except Exception as ex:
        return (type(ex).__name__, str(ex)[:60])
def uf(m, fn, nin, nout, **kw): return m.frompyfunc(fn, nin, nout, **kw)
cases = {
    "override first": lambda m: uf(m, mul3, 3, 1)(A(), 1, 2),
    "override sub before super": lambda m: uf(m, mul3, 3, 1)(A(), ASub(), 2),
    "all NotImplemented": lambda m: uf(m, mul3, 3, 1)(C(), C(), 1),
    "__array_ufunc__ None": lambda m: uf(m, add2, 2, 1)(N(), 1),
    "plain call": lambda m: uf(m, add2, 2, 1)(np.arange(3), 10),
    "out=": lambda m: (lambda o: (uf(m, add2, 2, 1)(np.arange(3), 1, out=o), o)[1])(np.empty(3, dtype=object)),
    "where=": lambda m: uf(m, add2, 2, 1)(np.arange(3), 1, where=np.array([True, False, True]), out=np.zeros(3, dtype=object)),
    "accumulate": lambda m: uf(m, add2, 2, 1).accumulate(np.arange(5)),
    "outer": lambda m: uf(m, add2, 2, 1).outer(np.arange(2), np.arange(3)),
    "at": lambda m: (lambda a: (uf(m, add2, 2, 1).at(a, [0, 0], 1), a)[1])(np.zeros(3, dtype=object)),
    "reduceat": lambda m: uf(m, add2, 2, 1).reduceat(np.arange(6), [0, 2, 4]),
    "reduce override": lambda m: uf(m, add2, 2, 1).reduce(A()),
    "reduce plain": lambda m: uf(m, add2, 2, 1, identity=0).reduce(np.arange(5)),
    "types": lambda m: uf(m, add2, 2, 1).types,
    "nargs": lambda m: uf(m, add2, 2, 1).nargs,
    "identity": lambda m: uf(m, add2, 2, 1, identity=0).identity,
}
bad = []
for k, f in cases.items():
    a, b = outcome(lambda: f(np)), outcome(lambda: f(fnp))
    if a != b:
        bad.append(f"{k}: numpy={a} fnp={b}")
print(len(cases), bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(2, ' ');
    assert_eq!(
        fields.next().unwrap_or("0"),
        "16",
        "cell table drifted: {result}"
    );
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "frompyfunc must match numpy's: {result}"
    );
    Ok(())
}

/// The NaN-screened sort / argsort / sort_complex fast paths scan the caller's buffer for NaN and
/// read it again afterwards; a NaN that lands in between (another thread's `np.copyto`, which
/// drops the GIL - 25 sites reproduced that way - or here, deterministically, a patched
/// `np.empty` that the route calls between the two reads) hit `partial_cmp(..).expect("no NaN")`
/// and raised PanicException, which numpy never raises (bead rc0923 .20). Every cell below raised
/// PanicException on the pre-fix build (ba919374); with the NaN-last comparator each must return.
/// Sizes are the ones the parallel routes need (2**19 / 2**20); a host whose gates send a cell to
/// numpy instead passes that cell without exercising it.
#[test]
fn nan_screened_sort_routes_never_panic_when_the_operand_changes_after_the_screen()
-> Result<(), String> {
    let script = fnp_script(
        r#"
R = np.random.default_rng(1)
M, N = 1 << 19, 1 << 20
real_empty = np.empty
def planted(operand, index, value):
    state = {"done": False}
    def fake_empty(*args, **kwargs):
        if not state["done"]:
            state["done"] = True
            operand[index] = value
        return real_empty(*args, **kwargs)
    return fake_empty
def run(label, make, index, value, call):
    operand = make()
    np.empty = planted(operand, index, value)
    try:
        call(operand)
        outcome = "ok"
    except BaseException as ex:
        outcome = type(ex).__name__
    finally:
        np.empty = real_empty
    return label, outcome
nan = np.nan
cases = [
    run("sort f64 last axis", lambda: R.permutation(N).astype(float).reshape(4, -1), (0, 3), nan, fnp.sort),
    run("argsort c128 flat, NaN real", lambda: R.permutation(N) + 1j * R.permutation(N), 3, complex(nan, 1.0), fnp.argsort),
    run("argsort c128 flat, NaN imag", lambda: np.floor(R.permutation(N) / 2) + 1j * R.permutation(N), 3, complex(1.0, nan), fnp.argsort),
    run("argsort c64 flat", lambda: (R.permutation(N) + 1j * R.permutation(N)).astype(np.complex64), 3, complex(nan, 1.0), fnp.argsort),
    run("argsort c128 last axis", lambda: np.stack([R.permutation(M) + 1j * R.permutation(M) for _ in "ab"]), (0, 3), complex(nan, 1.0), fnp.argsort),
    run("argsort c64 last axis", lambda: np.stack([R.permutation(M) + 1j * R.permutation(M) for _ in "ab"]).astype(np.complex64), (0, 3), complex(nan, 1.0), fnp.argsort),
    run("argsort f64 last axis", lambda: np.stack([R.permutation(M) for _ in "ab"]).astype(float), (0, 3), nan, fnp.argsort),
    run("argsort f32 last axis", lambda: np.stack([R.permutation(M) for _ in "ab"]).astype(np.float32), (0, 3), nan, fnp.argsort),
    run("sort_complex f64 with -0.0", lambda: np.concatenate([R.permutation(N).astype(float), [-0.0]]), 3, nan, fnp.sort_complex),
    run("sort_complex f64", lambda: R.permutation(N) + 1.0, 3, nan, fnp.sort_complex),
]
print(len(cases), [c for c in cases if c[1] != "ok"])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "10 []",
        "a NaN written after the screen must not panic: {result}"
    );
    Ok(())
}

/// Singletons from numpy's own suites under the drop-in harness (bead rc0923 .8), each a
/// silently different answer before the fix:
/// - `digitize(2**54, [2**54 - 1, 2**54 + 1])` compared in float64 and answered 2 (numpy: 1).
/// - `histogram` / `histogram_bin_edges` over a range only ulps wide built bins that did not
///   increase and counted into them, where numpy raises "Too many bins for data range".
/// - `linspace` to a subnormal stop returned zeros (numpy divides first when the step
///   underflows, gh-5437).
/// - An operand carrying `__array_wrap__` (numpy hands it the result) or `__array_ufunc__`
///   (numpy dispatches it) got a bare ndarray from the native unary ufuncs.
///
/// 14 of these 46 cells failed on 0ba35c5c.
#[test]
fn digitize_histogram_linspace_and_ufunc_hook_singletons_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            if isinstance(r, Wrap):
                got = ("Wrap", r.ctx_name, r.arr.dtype.str, r.arr.tobytes())
            elif isinstance(r, tuple):
                got = ("tuple",) + tuple((np.asarray(x).dtype.str, np.asarray(x).tobytes()) for x in r)
            elif isinstance(r, (np.ndarray, np.generic)):
                a = np.asarray(r)
                got = ("ok", type(r).__name__, a.dtype.str, a.shape, a.tobytes())
            else:
                got = ("ok", type(r).__name__, repr(r))
        except Exception as ex:
            got = (type(ex).__name__, str(ex))
    return got + (sorted((w.category.__name__, str(w.message)) for w in caught),)

base = np.arange(4.0)
class Wrap:
    __array_interface__ = base.__array_interface__
    def __array_wrap__(self, arr, context=None, return_scalar=False):
        r = Wrap()
        r.arr = arr
        r.ctx_name = context[0].__name__ if context else None
        return r
class UF:
    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        return ("UF", ufunc.__name__, method)

big = 2**54
tiny_range = np.array([1, 1 + 2e-16] * 10)
cases = {
    # Integers past 2**53 must not collapse in float64.
    "digitize 2**54": lambda m: m.digitize(big, [big - 1, big + 1]),
    "digitize 2**54 right": lambda m: m.digitize(big, [big - 1, big + 1], right=True),
    "digitize 2**54 list": lambda m: m.digitize([big, big + 2], [big - 1, big + 1]),
    # Bins the float edges cannot separate are numpy's ValueError.
    "histogram tiny range": lambda m: m.histogram(tiny_range, bins=10),
    "histogram tiny range f32": lambda m: m.histogram(tiny_range.astype(np.float32), bins=10),
    "histogram_bin_edges tiny range": lambda m: m.histogram_bin_edges(tiny_range, bins=10),
    "histogram ordinary": lambda m: m.histogram(np.arange(20.0), bins=4),
    # A step that underflows to zero keeps the subnormal values (gh-5437).
    "linspace subnormal f64": lambda m: m.linspace(0, np.nextafter(0.0, 1.0) * 5, 10, endpoint=False),
    "linspace subnormal f32": lambda m: m.linspace(0, np.nextafter(np.float32(0), np.float32(1)) * 5, 10, endpoint=False, dtype=np.float32),
    "linspace subnormal endpoint": lambda m: m.linspace(0, np.nextafter(0.0, 1.0) * 5, 11),
    "linspace ordinary": lambda m: m.linspace(0.0, 1.0, 7),
}
# Operands carrying numpy's ufunc hooks: `__array_wrap__` receives the result, and
# `__array_ufunc__` is dispatched.
for name in ("abs", "absolute", "negative", "sqrt", "sin", "exp", "isnan", "i0", "fabs"):
    cases[f"{name}(Wrap)"] = lambda m, name=name: getattr(m, name)(Wrap())
    cases[f"{name}(UF)"] = lambda m, name=name: getattr(m, name)(UF())
for name in ("add", "multiply", "maximum", "power", "divide", "subtract", "arctan2"):
    cases[f"{name}(Wrap, 1)"] = lambda m, name=name: getattr(m, name)(Wrap(), 1.0)
    cases[f"{name}(1, UF)"] = lambda m, name=name: getattr(m, name)(1.0, UF())
cases["add.reduce(UF)"] = lambda m: m.add.reduce(UF())
cases["numpy scalar add"] = lambda m: m.add(np.float64(1.5), 2.0)
cases["numpy scalar sqrt"] = lambda m: m.sqrt(np.float32(2.0))
bad = []
for name, case in cases.items():
    ours, theirs = outcome(lambda: case(fnp)), outcome(lambda: case(np))
    if ours != theirs:
        bad.append(f"{name}: fnp={str(ours)[:160]} numpy={str(theirs)[:160]}")
print(len(cases), bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "46 []",
        "singletons must match numpy: {result}"
    );
    Ok(())
}
