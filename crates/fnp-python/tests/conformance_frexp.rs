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
    assert_eq!(result.trim(), "True", "frexp signed zero should match numpy");
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
    assert_eq!(result.trim(), "True", "frexp powers of two should match numpy");
    Ok(())
}
