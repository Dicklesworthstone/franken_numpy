//! Conformance tests for numpy.frexp against NumPy oracle.
//!
//! Tests frexp (extract mantissa and exponent).

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
fn frexp_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 4.0, 8.0])
fnp_mant, fnp_exp = fnp.frexp(x)
np_mant, np_exp = np.frexp(x)
print(np.allclose(fnp_mant, np_mant) and np.array_equal(fnp_exp, np_exp))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "frexp basic should match numpy");
    Ok(())
}

#[test]
fn frexp_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.0, -2.0, -4.0])
fnp_mant, fnp_exp = fnp.frexp(x)
np_mant, np_exp = np.frexp(x)
print(np.allclose(fnp_mant, np_mant) and np.array_equal(fnp_exp, np_exp))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "frexp with negative values should match numpy"
    );
    Ok(())
}

#[test]
fn frexp_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, np.inf, -np.inf])
fnp_mant, fnp_exp = fnp.frexp(x)
np_mant, np_exp = np.frexp(x)
mant_match = np.allclose(fnp_mant, np_mant, equal_nan=True) or (np.isinf(fnp_mant) == np.isinf(np_mant)).all()
exp_match = np.array_equal(fnp_exp, np_exp)
print(mant_match and exp_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "frexp with special values should match numpy"
    );
    Ok(())
}

#[test]
fn frexp_f64_zerocopy_bit_exact_golden_sha256() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib
import warnings

cases = []
base = np.array([
    -0.0,
    0.0,
    1.0,
    -2.5,
    8.0,
    np.inf,
    -np.inf,
    np.finfo(np.float64).tiny,
    np.finfo(np.float64).tiny / 2,
    np.nextafter(0.0, 1.0),
], dtype=np.float64)
custom_nan = np.array([
    0x7ff8000000000001,
    0x7ff0000000000001,
    0xfff8000000000002,
], dtype=np.uint64).view(np.float64)
cases.append(base)
cases.append(custom_nan)
cases.append(np.linspace(-1024.0, 1024.0, 64, dtype=np.float64).reshape(8, 8))
cases.append(np.array([], dtype=np.float64))

chunks = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for x in cases:
        fnp_m, fnp_e = fnp.frexp(x)
        np_m, np_e = np.frexp(x)
        for got, expected in [(fnp_m, np_m), (fnp_e, np_e)]:
            got = np.ascontiguousarray(got)
            expected = np.ascontiguousarray(expected)
            assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
            assert got.shape == expected.shape, (got.shape, expected.shape)
            assert got.tobytes() == expected.tobytes(), (got, expected)
            chunks.append(str(got.dtype).encode())
            chunks.append(str(got.shape).encode())
            chunks.append(got.tobytes())

print(hashlib.sha256(b"".join(chunks)).hexdigest())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "26994a6d71efb33eb3b046511ba6e3b631a3a4cc2feb1435e46035f8c6f5885f",
        "frexp mantissa/exponent dtype/shape/raw bytes should match numpy"
    );
    Ok(())
}

#[test]
fn frexp_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(4.0)
fnp_mant, fnp_exp = fnp.frexp(x)
np_mant, np_exp = np.frexp(x)
mant_type_match = type(fnp_mant).__name__ == type(np_mant).__name__
exp_type_match = type(fnp_exp).__name__ == type(np_exp).__name__
print(mant_type_match and exp_type_match, fnp_mant, fnp_exp, np_mant, np_exp)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "frexp scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn frexp_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.nan])
fnp_mant, fnp_exp = fnp.frexp(x)
np_mant, np_exp = np.frexp(x)
print(np.isnan(fnp_mant[0]) and np.isnan(np_mant[0]) and fnp_exp[0] == np_exp[0])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "frexp nan should match numpy");
    Ok(())
}

#[test]
fn frexp_signed_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, -0.0])
fnp_mant, fnp_exp = fnp.frexp(x)
np_mant, np_exp = np.frexp(x)
# Check values and sign bits match
value_match = np.allclose(fnp_mant, np_mant) and np.array_equal(fnp_exp, np_exp)
sign_match = np.array_equal(np.signbit(fnp_mant), np.signbit(np_mant))
print(value_match and sign_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "frexp signed zero should match numpy"
    );
    Ok(())
}

#[test]
fn frexp_subnormal() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.finfo(np.float64).tiny / 2, np.finfo(np.float64).tiny / 4])
fnp_mant, fnp_exp = fnp.frexp(x)
np_mant, np_exp = np.frexp(x)
print(np.allclose(fnp_mant, np_mant) and np.array_equal(fnp_exp, np_exp))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "frexp subnormal should match numpy");
    Ok(())
}

#[test]
fn frexp_powers_of_two() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([2**i for i in range(-5, 10)])
fnp_mant, fnp_exp = fnp.frexp(x)
np_mant, np_exp = np.frexp(x)
# For powers of 2, mantissa should be 0.5 and exp is n+1
print(np.allclose(fnp_mant, np_mant) and np.array_equal(fnp_exp, np_exp))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "frexp powers of two should match numpy"
    );
    Ok(())
}
