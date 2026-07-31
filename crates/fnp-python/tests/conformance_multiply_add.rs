//! Conformance tests for fused arithmetic chains against the NumPy oracle.
//!
//! `fnp.multiply_add(a, b, c)` computes `a * b + c` in one pass. NumPy has no
//! fused public form, so the oracle is NumPy's own two-call expression. The
//! contract is BYTE equality, not closeness: the fused pass keeps both roundings
//! (multiply rounds, then add rounds) exactly as the two ufuncs do, and must
//! never contract into a single-rounding FMA.

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

/// The routed regime: above the parallel gate, so the Rayon banded path runs.
#[test]
fn multiply_add_f64_large_is_byte_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(7)
a = rng.standard_normal(300_000)
b = rng.standard_normal(300_000)
c = rng.standard_normal(300_000)
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(ours.dtype == theirs.dtype, ours.shape == theirs.shape, ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True True");
    Ok(())
}

/// Below the parallel gate, so the serial path runs. Same byte contract.
#[test]
fn multiply_add_f64_small_serial_path_is_byte_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(11)
a = rng.standard_normal(1000)
b = rng.standard_normal(1000)
c = rng.standard_normal(1000)
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(ours.dtype == theirs.dtype, ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True");
    Ok(())
}

/// THE FMA TRAP. These operands are chosen so that a single-rounding fused
/// multiply-add gives a DIFFERENT answer from multiply-then-add. If the kernel
/// ever contracts, this test fails and the byte-equality claim dies with it.
#[test]
fn multiply_add_does_not_contract_into_a_single_rounding_fma() -> Result<(), String> {
    let script = fnp_script(
        r#"
# a*b is inexact in f64; adding -(a*b rounded) exposes the rounding residue.
# With two roundings the result is exactly 0.0; a true FMA returns the residue.
a = np.array([1.0 + 2.0**-52, 1.0 + 2.0**-52, 0.1, 1e200], dtype=np.float64)
b = np.array([1.0 + 2.0**-52, 1.0 - 2.0**-52, 0.1, 1e200], dtype=np.float64)
c = -(a * b)
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(ours.tobytes() == theirs.tobytes(),
      bool(np.all(ours[:3] == 0.0)),
      bool(np.isnan(ours[3])))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True True");
    Ok(())
}

/// Non-finite propagation must match element for element, including NaN bits.
#[test]
fn multiply_add_non_finite_matches_numpy_bitwise() -> Result<(), String> {
    let script = fnp_script(
        r#"
inf = np.inf
nan = np.nan
a = np.array([inf, -inf, nan, 0.0, -0.0, inf, 1.0, -0.0], dtype=np.float64)
b = np.array([0.0, 0.0, 1.0, inf, inf, inf, nan, -0.0], dtype=np.float64)
c = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -inf, 0.0, 0.0], dtype=np.float64)
with np.errstate(all='ignore'):
    ours = fnp.multiply_add(a, b, c)
    theirs = a * b + c
print(ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True");
    Ok(())
}

#[test]
fn multiply_add_f32_is_byte_identical_and_keeps_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(3)
a = rng.standard_normal(200_000).astype(np.float32)
b = rng.standard_normal(200_000).astype(np.float32)
c = rng.standard_normal(200_000).astype(np.float32)
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(str(ours.dtype), ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "float32 True");
    Ok(())
}

/// Every fixed-width integer dtype takes the fused path. Full-range random
/// bytes exercise wrapping overflow rather than merely the no-overflow subset.
#[test]
fn multiply_add_all_integer_widths_match_numpy_with_overflow() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(17)
checks = []
for dtype in [np.int8, np.uint8, np.int16, np.uint16,
              np.int32, np.uint32, np.int64, np.uint64]:
    dt = np.dtype(dtype)
    n = 300_000
    a = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()
    b = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()
    c = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()
    with np.errstate(over='ignore'):
        ours = fnp.multiply_add(a, b, c)
        theirs = a * b + c
    checks.append(ours.dtype == theirs.dtype and ours.tobytes() == theirs.tobytes())
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 8");
    Ok(())
}

/// Multi-dimensional input must come back with the original shape, not flat.
#[test]
fn multiply_add_preserves_multidimensional_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(5)
a = rng.standard_normal((400, 900))
b = rng.standard_normal((400, 900))
c = rng.standard_normal((400, 900))
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(ours.shape == theirs.shape, ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True");
    Ok(())
}

/// Everything outside the routed regime must defer to NumPy's own expression
/// and therefore still agree: mixed dtypes, broadcasting, non-contiguity,
/// mixed integer widths, and complex.
#[test]
fn multiply_add_deferred_regimes_still_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(13)
checks = []

# broadcasting (shapes differ) -> deferred
a = rng.standard_normal((512, 4))
b = rng.standard_normal((4,))
c = rng.standard_normal((512, 1))
checks.append(np.array_equal(fnp.multiply_add(a, b, c), a * b + c, equal_nan=True))

# non-contiguous view -> deferred
base = rng.standard_normal((256, 8))
a2 = base[:, ::2]
b2 = base[:, 1::2]
c2 = base[:, ::2]
checks.append(np.array_equal(fnp.multiply_add(a2, b2, c2), a2 * b2 + c2, equal_nan=True))

# mixed float widths -> deferred, numpy owns promotion
a3 = rng.standard_normal(1000).astype(np.float32)
b3 = rng.standard_normal(1000)
c3 = rng.standard_normal(1000)
out3 = fnp.multiply_add(a3, b3, c3)
ref3 = a3 * b3 + c3
checks.append(out3.dtype == ref3.dtype and out3.tobytes() == ref3.tobytes())

# mixed integer widths -> deferred, numpy owns promotion
a4 = rng.integers(-1000, 1000, size=5000, dtype=np.int32)
b4 = rng.integers(-1000, 1000, size=5000, dtype=np.int64)
c4 = rng.integers(-1000, 1000, size=5000, dtype=np.int64)
out4 = fnp.multiply_add(a4, b4, c4)
ref4 = a4 * b4 + c4
checks.append(out4.dtype == ref4.dtype and out4.tobytes() == ref4.tobytes())

# complex dtype -> routed natively (see the contraction tests below), same bytes
a5 = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
b5 = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
c5 = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
out5 = fnp.multiply_add(a5, b5, c5)
ref5 = a5 * b5 + c5
checks.append(out5.dtype == ref5.dtype and out5.tobytes() == ref5.tobytes())

# python lists -> deferred
checks.append(np.array_equal(fnp.multiply_add([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]),
                             np.array([8.0, 14.0])))

print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 6");
    Ok(())
}

/// The longer chain exercises the parallel path and preserves shape, dtype, and
/// every output bit across both routed floating widths.
#[test]
fn subtract_multiply_add_float_routes_match_numpy_bitwise() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(29)
checks = []
for dtype in [np.float64, np.float32]:
    arrays = [rng.standard_normal((300, 1000)).astype(dtype) for _ in range(4)]
    a, b, c, d = arrays
    ours = fnp.subtract_multiply_add(a, b, c, d)
    theirs = (a - b) * c + d
    checks.append(ours.dtype == theirs.dtype and
                  ours.shape == theirs.shape and
                  ours.tobytes() == theirs.tobytes())
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 2");
    Ok(())
}

/// This makes a contracted multiply-add observably different from NumPy's
/// subtract, multiply, then add sequence.
#[test]
fn subtract_multiply_add_retains_all_float_roundings() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0 + 2.0**-52, 1.0 + 2.0**-52, 0.1, 1e200], dtype=np.float64)
b = np.zeros_like(a)
c = np.array([1.0 + 2.0**-52, 1.0 - 2.0**-52, 0.1, 1e200], dtype=np.float64)
d = -((a - b) * c)
with np.errstate(all='ignore'):
    ours = fnp.subtract_multiply_add(a, b, c, d)
    theirs = (a - b) * c + d
print(ours.tobytes() == theirs.tobytes(),
      bool(np.all(ours[:3] == 0.0)),
      bool(np.isnan(ours[3])))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True True");
    Ok(())
}

/// Full-range bytes force modular overflow at every fixed integer width.
#[test]
fn subtract_multiply_add_integer_routes_match_numpy_with_overflow() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(31)
checks = []
for dtype in [np.int8, np.uint8, np.int16, np.uint16,
              np.int32, np.uint32, np.int64, np.uint64]:
    dt = np.dtype(dtype)
    n = 300_000
    a, b, c, d = [np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()
                  for _ in range(4)]
    with np.errstate(over='ignore'):
        ours = fnp.subtract_multiply_add(a, b, c, d)
        theirs = (a - b) * c + d
    checks.append(ours.dtype == theirs.dtype and ours.tobytes() == theirs.tobytes())
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 8");
    Ok(())
}

/// A caller-owned output removes the candidate's final allocation. Disjoint
/// outputs stay on the one-pass route; every exact/partial alias must defer so
/// it observes the same mutations as NumPy's three in-place ufuncs.
#[test]
fn subtract_multiply_add_out_is_exact_and_alias_safe() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(41)
checks = []

for dtype in [np.float64, np.float32, np.int8, np.uint8, np.int16,
              np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
    dt = np.dtype(dtype)
    n = 300_000
    if np.issubdtype(dt, np.integer):
        a, b, c, d = [np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()
                      for _ in range(4)]
    else:
        a, b, c, d = [rng.standard_normal(n).astype(dt) for _ in range(4)]
    ours = np.empty_like(a)
    theirs = np.empty_like(a)
    with np.errstate(all='ignore'):
        returned = fnp.subtract_multiply_add(a, b, c, d, out=ours)
        np.subtract(a, b, out=theirs)
        np.multiply(theirs, c, out=theirs)
        np.add(theirs, d, out=theirs)
    checks.append(returned is ours and ours.tobytes() == theirs.tobytes())

original = [rng.standard_normal(4096) for _ in range(4)]
for output_index in range(4):
    ours_inputs = [value.copy() for value in original]
    numpy_inputs = [value.copy() for value in original]
    ours = ours_inputs[output_index]
    theirs = numpy_inputs[output_index]
    returned = fnp.subtract_multiply_add(*ours_inputs, out=ours)
    np.subtract(numpy_inputs[0], numpy_inputs[1], out=theirs)
    np.multiply(theirs, numpy_inputs[2], out=theirs)
    np.add(theirs, numpy_inputs[3], out=theirs)
    checks.append(returned is ours and ours.tobytes() == theirs.tobytes())

# Distinct ndarray objects whose byte ranges overlap must also defer.
ours_base = np.linspace(-3.0, 7.0, 4097)
numpy_base = ours_base.copy()
ours_a, ours_out = ours_base[:-1], ours_base[1:]
numpy_a, numpy_out = numpy_base[:-1], numpy_base[1:]
b, c, d = [rng.standard_normal(4096) for _ in range(3)]
returned = fnp.subtract_multiply_add(ours_a, b, c, d, out=ours_out)
np.subtract(numpy_a, b, out=numpy_out)
np.multiply(numpy_out, c, out=numpy_out)
np.add(numpy_out, d, out=numpy_out)
checks.append(returned is ours_out and ours_base.tobytes() == numpy_base.tobytes())

# Unsupported output layouts and casts retain NumPy's exact behavior.
a, b, c, d = [rng.standard_normal(4096) for _ in range(4)]
ours_storage = np.empty(8192)
numpy_storage = np.empty(8192)
ours_out = ours_storage[::2]
numpy_out = numpy_storage[::2]
returned = fnp.subtract_multiply_add(a, b, c, d, out=ours_out)
np.subtract(a, b, out=numpy_out)
np.multiply(numpy_out, c, out=numpy_out)
np.add(numpy_out, d, out=numpy_out)
checks.append(returned is ours_out and ours_out.tobytes() == numpy_out.tobytes())

ours_out = np.empty(4096, dtype=np.float32)
numpy_out = np.empty(4096, dtype=np.float32)
returned = fnp.subtract_multiply_add(a, b, c, d, out=ours_out)
np.subtract(a, b, out=numpy_out)
np.multiply(numpy_out, c, out=numpy_out)
np.add(numpy_out, d, out=numpy_out)
checks.append(returned is ours_out and ours_out.tobytes() == numpy_out.tobytes())

readonly = np.empty_like(a)
readonly.flags.writeable = False
try:
    fnp.subtract_multiply_add(a, b, c, d, out=readonly)
except Exception as exc:
    ours_error = (type(exc).__name__, str(exc))
try:
    np.subtract(a, b, out=readonly)
except Exception as exc:
    numpy_error = (type(exc).__name__, str(exc))
checks.append(ours_error == numpy_error)

print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 18");
    Ok(())
}

/// Broadcasting, mixed widths, non-contiguity, complex values, and Python
/// sequences all stay on NumPy's own three-ufunc semantics.
#[test]
fn subtract_multiply_add_deferred_regimes_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(37)
checks = []

a = rng.standard_normal((512, 4))
b = rng.standard_normal((4,))
c = rng.standard_normal((512, 1))
d = rng.standard_normal((512, 4))
checks.append(np.array_equal(fnp.subtract_multiply_add(a, b, c, d),
                             (a - b) * c + d, equal_nan=True))

base = rng.standard_normal((256, 16))
a2, b2, c2, d2 = base[:, ::2], base[:, 1::2], base[:, ::2], base[:, 1::2]
checks.append(np.array_equal(fnp.subtract_multiply_add(a2, b2, c2, d2),
                             (a2 - b2) * c2 + d2, equal_nan=True))

a3 = rng.standard_normal(1000).astype(np.float32)
b3 = rng.standard_normal(1000)
c3 = rng.standard_normal(1000)
d3 = rng.standard_normal(1000)
ours3 = fnp.subtract_multiply_add(a3, b3, c3, d3)
theirs3 = (a3 - b3) * c3 + d3
checks.append(ours3.dtype == theirs3.dtype and ours3.tobytes() == theirs3.tobytes())

a4 = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
b4 = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
c4 = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
d4 = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
ours4 = fnp.subtract_multiply_add(a4, b4, c4, d4)
theirs4 = (a4 - b4) * c4 + d4
checks.append(ours4.dtype == theirs4.dtype and ours4.tobytes() == theirs4.tobytes())

checks.append(np.array_equal(
    fnp.subtract_multiply_add([5, 8], [1, 2], [3, 4], [7, 9]),
    np.array([19, 33])))

print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 5");
    Ok(())
}

// ---------------------------------------------------------------------------
// float16 — NumPy has no f16 ALU, so each ufunc widens to f32 and narrows back.
// The fused pass must reproduce BOTH narrowings, in order.
// ---------------------------------------------------------------------------

/// The routed regime, above the parallel gate.
#[test]
fn multiply_add_f16_large_is_byte_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(21)
a = (rng.standard_normal(300_000) * 4).astype(np.float16)
b = (rng.standard_normal(300_000) * 4).astype(np.float16)
c = (rng.standard_normal(300_000) * 4).astype(np.float16)
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(ours.dtype == theirs.dtype, ours.shape == theirs.shape, ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True True");
    Ok(())
}

/// Below the parallel gate, so the serial path runs. Same byte contract.
#[test]
fn multiply_add_f16_small_serial_path_is_byte_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(22)
a = (rng.standard_normal(1000) * 4).astype(np.float16)
b = (rng.standard_normal(1000) * 4).astype(np.float16)
c = (rng.standard_normal(1000) * 4).astype(np.float16)
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(ours.dtype == theirs.dtype, ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True");
    Ok(())
}

/// The intermediate narrow to `float16` is LOAD-BEARING, not an optimization
/// detail. Keeping the product in `float32` across the add — the obvious
/// "faster" spelling — produces different bytes from NumPy. This test fails
/// loudly if the kernel ever drops that narrowing, and asserts the
/// discriminating condition actually occurs so it can never pass vacuously.
#[test]
fn multiply_add_f16_intermediate_narrowing_is_load_bearing() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(23)
a = (rng.standard_normal(200_000) * 4).astype(np.float16)
b = (rng.standard_normal(200_000) * 4).astype(np.float16)
c = (rng.standard_normal(200_000) * 4).astype(np.float16)
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
# The spelling that skips the intermediate narrow.
unnarrowed = (a.astype(np.float32) * b.astype(np.float32) + c.astype(np.float32)).astype(np.float16)
discriminating = theirs.tobytes() != unnarrowed.tobytes()
print(ours.tobytes() == theirs.tobytes(), discriminating)
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True");
    Ok(())
}

/// f16 special values: inf, -inf, nan, signed zeros, and overflow of the narrow
/// range (f16 maxes out near 65504, so ordinary products saturate to inf).
#[test]
fn multiply_add_f16_special_values_are_byte_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
vals = np.array([0.0, -0.0, 1.0, -1.0, 2.0, 65504.0, -65504.0, 6e-8,
                 np.inf, -np.inf, np.nan], dtype=np.float16)
A, B, C = (g.ravel().copy() for g in np.meshgrid(vals, vals, vals, indexing='ij'))
with np.errstate(all='ignore'):
    ours = fnp.multiply_add(A, B, C)
    theirs = A * B + C
print(ours.dtype == theirs.dtype, ours.tobytes() == theirs.tobytes(), A.size)
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True 1331");
    Ok(())
}

// ---------------------------------------------------------------------------
// complex64 / complex128 — the INVERSE of the real-float contract. NumPy's
// complex multiply loop is compiled with FP contraction enabled, so it emits a
// genuine FMA; matching it bit-for-bit REQUIRES fusing, where the real f64
// route requires NOT fusing.
// ---------------------------------------------------------------------------

/// The routed regime, above the parallel gate.
#[test]
fn multiply_add_complex128_large_is_byte_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(31)
mk = lambda: rng.standard_normal(300_000) + 1j * rng.standard_normal(300_000)
a, b, c = mk(), mk(), mk()
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(ours.dtype == theirs.dtype, ours.shape == theirs.shape, ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True True");
    Ok(())
}

/// complex64 carries its own component width; the kernel must fuse in `f32`.
#[test]
fn multiply_add_complex64_large_is_byte_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(32)
mk = lambda: (rng.standard_normal(300_000) + 1j * rng.standard_normal(300_000)).astype(np.complex64)
a, b, c = mk(), mk(), mk()
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(ours.dtype == theirs.dtype, ours.shape == theirs.shape, ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True True");
    Ok(())
}

/// Below the parallel gate, so the serial path runs. Same byte contract.
#[test]
fn multiply_add_complex128_small_serial_path_is_byte_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(33)
mk = lambda: rng.standard_normal(1000) + 1j * rng.standard_normal(1000)
a, b, c = mk(), mk(), mk()
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(ours.dtype == theirs.dtype, ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True");
    Ok(())
}

/// THE CONTRACTION TEST, and the mirror image of
/// `multiply_add_anti_contraction_guard` above. For complex operands the naive
/// schoolbook `ar*br - ai*bi` disagrees with NumPy on a large fraction of
/// random components, because NumPy's loop contracts to
/// `fma(ar, br, -(ai*bi))`. This asserts two things: that we match NumPy, and
/// that the naive form genuinely differs on this data — so the test cannot pass
/// vacuously if the discriminating condition ever stops occurring.
#[test]
fn multiply_add_complex_must_contract_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(34)
mk = lambda: rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000)
a, b, c = mk(), mk(), mk()
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
# The naive schoolbook spelling, evaluated with separate roundings.
ar, ai, br, bi = a.real, a.imag, b.real, b.imag
naive = np.empty_like(a)
naive.real = (ar * br - ai * bi) + c.real
naive.imag = (ar * bi + ai * br) + c.imag
discriminating = theirs.tobytes() != naive.tobytes()
print(ours.tobytes() == theirs.tobytes(), discriminating)
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True");
    Ok(())
}

/// Complex special values: infinities and NaNs in either component, where
/// NumPy's complex arithmetic has its own propagation rules.
#[test]
fn multiply_add_complex128_special_values_are_byte_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
parts = [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan]
vals = np.array([complex(re, im) for re in parts for im in parts], dtype=np.complex128)
A, B = (g.ravel().copy() for g in np.meshgrid(vals, vals, indexing='ij'))
C = np.tile(vals, A.size // vals.size)
with np.errstate(all='ignore'):
    ours = fnp.multiply_add(A, B, C)
    theirs = A * B + C
# NaN payloads may differ bit-wise; compare NaN-aware per component.
ov, tv = ours.view(np.float64), theirs.view(np.float64)
same = np.array_equal(ov, tv) or ((ov == tv) | (np.isnan(ov) & np.isnan(tv))).all()
print(ours.dtype == theirs.dtype, bool(same), A.size)
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True 2401");
    Ok(())
}

/// Shape preservation for the new dtypes: multi-dimensional inputs must come
/// back with the original shape, not the flat buffer the kernel writes.
#[test]
fn multiply_add_new_dtypes_preserve_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(35)
checks = []
for dtype in (np.float16, np.complex64, np.complex128):
    if np.issubdtype(dtype, np.complexfloating):
        mk = lambda: (rng.standard_normal((400, 900)) + 1j * rng.standard_normal((400, 900))).astype(dtype)
    else:
        mk = lambda: (rng.standard_normal((400, 900)) * 4).astype(dtype)
    a, b, c = mk(), mk(), mk()
    ours = fnp.multiply_add(a, b, c)
    theirs = a * b + c
    checks.append(ours.shape == theirs.shape and ours.dtype == theirs.dtype
                  and ours.tobytes() == theirs.tobytes())
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 3");
    Ok(())
}

/// Non-contiguous and mismatched-shape inputs in the new dtypes must DEFER to
/// NumPy's own two-call form rather than take the fused route.
#[test]
fn multiply_add_new_dtypes_defer_outside_routed_regime() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(36)
checks = []

# non-contiguous complex128 (stride 2) -> deferred
base = rng.standard_normal(400_000) + 1j * rng.standard_normal(400_000)
a = base[::2]
b = (rng.standard_normal(400_000) + 1j * rng.standard_normal(400_000))[::2]
c = (rng.standard_normal(400_000) + 1j * rng.standard_normal(400_000))[::2]
out = fnp.multiply_add(a, b, c)
checks.append(out.dtype == (a * b + c).dtype and out.tobytes() == (a * b + c).tobytes())

# mixed complex widths -> deferred, numpy owns promotion
a2 = (rng.standard_normal(5000) + 1j * rng.standard_normal(5000)).astype(np.complex64)
b2 = rng.standard_normal(5000) + 1j * rng.standard_normal(5000)
c2 = rng.standard_normal(5000) + 1j * rng.standard_normal(5000)
out2 = fnp.multiply_add(a2, b2, c2)
ref2 = a2 * b2 + c2
checks.append(out2.dtype == ref2.dtype and out2.tobytes() == ref2.tobytes())

# f16 mixed with f32 -> deferred
a3 = (rng.standard_normal(5000) * 4).astype(np.float16)
b3 = (rng.standard_normal(5000) * 4).astype(np.float32)
c3 = (rng.standard_normal(5000) * 4).astype(np.float16)
out3 = fnp.multiply_add(a3, b3, c3)
ref3 = a3 * b3 + c3
checks.append(out3.dtype == ref3.dtype and out3.tobytes() == ref3.tobytes())

# broadcasting in the new dtypes -> deferred
a4 = (rng.standard_normal(5000) * 4).astype(np.float16)
b4 = np.float16(2.5)
c4 = (rng.standard_normal(5000) * 4).astype(np.float16)
out4 = fnp.multiply_add(a4, b4, c4)
ref4 = a4 * b4 + c4
checks.append(out4.dtype == ref4.dtype and out4.tobytes() == ref4.tobytes())

print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 4");
    Ok(())
}
