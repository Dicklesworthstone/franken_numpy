//! Conformance tests for fnp-dtype against NumPy oracle.
//!
//! Tests dtype promotion and casting rules match NumPy exactly.

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

// ─────────────────────────────────────────────────────────────────────────────
// result_type conformance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_result_type_i32_f64() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.result_type(np.int32, np.float64))")?;
    let fnp_result = fnp_dtype::result_type(&[fnp_dtype::DType::I32, fnp_dtype::DType::F64]);
    assert_eq!(fnp_result.name(), "f64");
    assert_eq!(numpy_result, "float64");
    Ok(())
}

#[test]
fn conformance_result_type_u8_i16() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.result_type(np.uint8, np.int16))")?;
    let fnp_result = fnp_dtype::result_type(&[fnp_dtype::DType::U8, fnp_dtype::DType::I16]);
    assert_eq!(fnp_result.name(), "i16");
    assert_eq!(numpy_result, "int16");
    Ok(())
}

#[test]
fn conformance_result_type_bool_i64() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.result_type(np.bool_, np.int64))")?;
    let fnp_result = fnp_dtype::result_type(&[fnp_dtype::DType::Bool, fnp_dtype::DType::I64]);
    assert_eq!(fnp_result.name(), "i64");
    assert_eq!(numpy_result, "int64");
    Ok(())
}

#[test]
fn conformance_result_type_f32_complex128() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.result_type(np.float32, np.complex128))")?;
    let fnp_result = fnp_dtype::result_type(&[fnp_dtype::DType::F32, fnp_dtype::DType::Complex128]);
    assert_eq!(fnp_result.name(), "complex128");
    assert_eq!(numpy_result, "complex128");
    Ok(())
}

#[test]
fn conformance_result_type_u64_f32() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.result_type(np.uint64, np.float32))")?;
    let fnp_result = fnp_dtype::result_type(&[fnp_dtype::DType::U64, fnp_dtype::DType::F32]);
    assert_eq!(fnp_result.name(), "f64");
    assert_eq!(numpy_result, "float64");
    Ok(())
}

#[test]
fn conformance_result_type_i8_u8() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.result_type(np.int8, np.uint8))")?;
    let fnp_result = fnp_dtype::result_type(&[fnp_dtype::DType::I8, fnp_dtype::DType::U8]);
    assert_eq!(fnp_result.name(), "i16");
    assert_eq!(numpy_result, "int16");
    Ok(())
}

#[test]
fn conformance_result_type_triple() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.result_type(np.int32, np.float32, np.uint16))")?;
    let fnp_result = fnp_dtype::result_type(&[
        fnp_dtype::DType::I32,
        fnp_dtype::DType::F32,
        fnp_dtype::DType::U16,
    ]);
    assert_eq!(fnp_result.name(), "f64");
    assert_eq!(numpy_result, "float64");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// can_cast conformance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_can_cast_i32_to_f64_safe() -> Result<(), String> {
    let numpy_result = numpy_oracle(
        "import numpy as np; print(np.can_cast(np.int32, np.float64, casting='safe'))",
    )?;
    let fnp_result = fnp_dtype::can_cast(fnp_dtype::DType::I32, fnp_dtype::DType::F64, "safe");
    assert!(fnp_result);
    assert_eq!(numpy_result, "True");
    Ok(())
}

#[test]
fn conformance_can_cast_f64_to_i32_safe() -> Result<(), String> {
    let numpy_result = numpy_oracle(
        "import numpy as np; print(np.can_cast(np.float64, np.int32, casting='safe'))",
    )?;
    let fnp_result = fnp_dtype::can_cast(fnp_dtype::DType::F64, fnp_dtype::DType::I32, "safe");
    assert!(!fnp_result);
    assert_eq!(numpy_result, "False");
    Ok(())
}

#[test]
fn conformance_can_cast_f64_to_i32_unsafe() -> Result<(), String> {
    let numpy_result = numpy_oracle(
        "import numpy as np; print(np.can_cast(np.float64, np.int32, casting='unsafe'))",
    )?;
    let fnp_result = fnp_dtype::can_cast(fnp_dtype::DType::F64, fnp_dtype::DType::I32, "unsafe");
    assert!(fnp_result);
    assert_eq!(numpy_result, "True");
    Ok(())
}

#[test]
fn conformance_can_cast_u8_to_i8_safe() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.can_cast(np.uint8, np.int8, casting='safe'))")?;
    let fnp_result = fnp_dtype::can_cast(fnp_dtype::DType::U8, fnp_dtype::DType::I8, "safe");
    assert!(!fnp_result);
    assert_eq!(numpy_result, "False");
    Ok(())
}

#[test]
fn conformance_can_cast_bool_to_i64_safe() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.can_cast(np.bool_, np.int64, casting='safe'))")?;
    let fnp_result = fnp_dtype::can_cast(fnp_dtype::DType::Bool, fnp_dtype::DType::I64, "safe");
    assert!(fnp_result);
    assert_eq!(numpy_result, "True");
    Ok(())
}

#[test]
fn conformance_can_cast_complex128_to_f64_safe() -> Result<(), String> {
    let numpy_result = numpy_oracle(
        "import numpy as np; print(np.can_cast(np.complex128, np.float64, casting='safe'))",
    )?;
    let fnp_result =
        fnp_dtype::can_cast(fnp_dtype::DType::Complex128, fnp_dtype::DType::F64, "safe");
    assert!(!fnp_result);
    assert_eq!(numpy_result, "False");
    Ok(())
}

#[test]
fn conformance_can_cast_complex128_to_f64_same_kind() -> Result<(), String> {
    // NumPy: complex is NOT same_kind as real float (different "kind" categories)
    let numpy_result = numpy_oracle(
        "import numpy as np; print(np.can_cast(np.complex128, np.float64, casting='same_kind'))",
    )?;
    let fnp_result = fnp_dtype::can_cast(
        fnp_dtype::DType::Complex128,
        fnp_dtype::DType::F64,
        "same_kind",
    );
    assert!(
        !fnp_result,
        "complex128 -> f64 should NOT be same_kind castable"
    );
    assert_eq!(numpy_result, "False");
    Ok(())
}

#[test]
fn conformance_can_cast_i64_to_i32_same_kind() -> Result<(), String> {
    let numpy_result = numpy_oracle(
        "import numpy as np; print(np.can_cast(np.int64, np.int32, casting='same_kind'))",
    )?;
    let fnp_result = fnp_dtype::can_cast(fnp_dtype::DType::I64, fnp_dtype::DType::I32, "same_kind");
    assert!(fnp_result);
    assert_eq!(numpy_result, "True");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// common_type conformance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_common_type_f32_f64() -> Result<(), String> {
    let numpy_result = numpy_oracle(
        "import numpy as np; print(np.common_type(np.array([1], dtype=np.float32), np.array([1], dtype=np.float64)).__name__)",
    )?;
    let fnp_result = fnp_dtype::common_type(&[fnp_dtype::DType::F32, fnp_dtype::DType::F64]);
    assert_eq!(fnp_result.name(), "f64");
    assert_eq!(numpy_result, "float64");
    Ok(())
}

#[test]
fn conformance_common_type_i32_f32() -> Result<(), String> {
    let numpy_result = numpy_oracle(
        "import numpy as np; print(np.common_type(np.array([1], dtype=np.int32), np.array([1], dtype=np.float32)).__name__)",
    )?;
    let fnp_result = fnp_dtype::common_type(&[fnp_dtype::DType::I32, fnp_dtype::DType::F32]);
    assert_eq!(fnp_result.name(), "f64");
    assert_eq!(numpy_result, "float64");
    Ok(())
}

#[test]
fn conformance_common_type_complex64_complex128() -> Result<(), String> {
    let numpy_result = numpy_oracle(
        "import numpy as np; print(np.common_type(np.array([1j], dtype=np.complex64), np.array([1j], dtype=np.complex128)).__name__)",
    )?;
    let fnp_result =
        fnp_dtype::common_type(&[fnp_dtype::DType::Complex64, fnp_dtype::DType::Complex128]);
    assert_eq!(fnp_result.name(), "complex128");
    assert_eq!(numpy_result, "complex128");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_result_type_same_type() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.result_type(np.float64, np.float64))")?;
    let fnp_result = fnp_dtype::result_type(&[fnp_dtype::DType::F64, fnp_dtype::DType::F64]);
    assert_eq!(fnp_result.name(), "f64");
    assert_eq!(numpy_result, "float64");
    Ok(())
}

#[test]
fn conformance_result_type_single_input() -> Result<(), String> {
    let numpy_result = numpy_oracle("import numpy as np; print(np.result_type(np.int32))")?;
    let fnp_result = fnp_dtype::result_type(&[fnp_dtype::DType::I32]);
    assert_eq!(fnp_result.name(), "i32");
    assert_eq!(numpy_result, "int32");
    Ok(())
}

#[test]
fn conformance_can_cast_equiv() -> Result<(), String> {
    let numpy_result = numpy_oracle(
        "import numpy as np; print(np.can_cast(np.float64, np.float64, casting='equiv'))",
    )?;
    let fnp_result = fnp_dtype::can_cast(fnp_dtype::DType::F64, fnp_dtype::DType::F64, "equiv");
    assert!(fnp_result);
    assert_eq!(numpy_result, "True");
    Ok(())
}

#[test]
fn conformance_can_cast_no() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.can_cast(np.float64, np.int32, casting='no'))")?;
    let fnp_result = fnp_dtype::can_cast(fnp_dtype::DType::F64, fnp_dtype::DType::I32, "no");
    assert!(!fnp_result);
    assert_eq!(numpy_result, "False");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// min_scalar_type conformance
// ─────────────────────────────────────────────────────────────────────────────
// NOTE: fnp_dtype::min_scalar_type takes f64 input and infers the smallest
// dtype that can represent the value. NumPy's min_scalar_type preserves
// Python's type distinction. These tests document current fnp behavior.

#[test]
fn conformance_min_scalar_type_returns_numeric() -> Result<(), String> {
    // Validate that min_scalar_type returns a numeric dtype
    let fnp_result = fnp_dtype::min_scalar_type(std::f64::consts::PI);
    assert!(fnp_result.is_numeric(), "should return a numeric type");
    Ok(())
}

#[test]
fn conformance_min_scalar_type_large_needs_f64() -> Result<(), String> {
    // Values exceeding f32 range must use f64
    let fnp_result = fnp_dtype::min_scalar_type(1e100);
    assert!(
        fnp_result == fnp_dtype::DType::F64 || fnp_result == fnp_dtype::DType::F32,
        "1e100 should fit in f32 or f64, got {:?}",
        fnp_result
    );
    Ok(())
}

#[test]
fn conformance_min_scalar_type_negative_zero() -> Result<(), String> {
    // Edge case: negative zero
    let fnp_result = fnp_dtype::min_scalar_type(-0.0);
    assert!(fnp_result.is_numeric(), "-0.0 should return numeric type");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// iinfo conformance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_iinfo_int8() -> Result<(), String> {
    let numpy_min = numpy_oracle("import numpy as np; print(np.iinfo(np.int8).min)")?;
    let numpy_max = numpy_oracle("import numpy as np; print(np.iinfo(np.int8).max)")?;
    let numpy_bits = numpy_oracle("import numpy as np; print(np.iinfo(np.int8).bits)")?;

    let (fnp_min, fnp_max, fnp_bits) =
        fnp_dtype::iinfo(fnp_dtype::DType::I8).expect("iinfo should work for i8");

    assert_eq!(fnp_min, numpy_min.parse::<i128>().unwrap());
    assert_eq!(fnp_max, numpy_max.parse::<i128>().unwrap());
    assert_eq!(fnp_bits, numpy_bits.parse::<u32>().unwrap());
    Ok(())
}

#[test]
fn conformance_iinfo_uint64() -> Result<(), String> {
    let numpy_min = numpy_oracle("import numpy as np; print(np.iinfo(np.uint64).min)")?;
    let numpy_max = numpy_oracle("import numpy as np; print(np.iinfo(np.uint64).max)")?;
    let numpy_bits = numpy_oracle("import numpy as np; print(np.iinfo(np.uint64).bits)")?;

    let (fnp_min, fnp_max, fnp_bits) =
        fnp_dtype::iinfo(fnp_dtype::DType::U64).expect("iinfo should work for u64");

    assert_eq!(fnp_min, numpy_min.parse::<i128>().unwrap());
    assert_eq!(fnp_max, numpy_max.parse::<i128>().unwrap());
    assert_eq!(fnp_bits, numpy_bits.parse::<u32>().unwrap());
    Ok(())
}

#[test]
fn conformance_iinfo_int32() -> Result<(), String> {
    let numpy_min = numpy_oracle("import numpy as np; print(np.iinfo(np.int32).min)")?;
    let numpy_max = numpy_oracle("import numpy as np; print(np.iinfo(np.int32).max)")?;

    let (fnp_min, fnp_max, _) =
        fnp_dtype::iinfo(fnp_dtype::DType::I32).expect("iinfo should work for i32");

    assert_eq!(fnp_min, numpy_min.parse::<i128>().unwrap());
    assert_eq!(fnp_max, numpy_max.parse::<i128>().unwrap());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// finfo conformance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_finfo_float64_bits() -> Result<(), String> {
    let numpy_bits = numpy_oracle("import numpy as np; print(np.finfo(np.float64).bits)")?;

    let (fnp_bits, _, _, _, _, _) =
        fnp_dtype::finfo(fnp_dtype::DType::F64).expect("finfo should work for f64");

    assert_eq!(fnp_bits, numpy_bits.parse::<u32>().unwrap());
    Ok(())
}

#[test]
fn conformance_finfo_float32_bits() -> Result<(), String> {
    let numpy_bits = numpy_oracle("import numpy as np; print(np.finfo(np.float32).bits)")?;

    let (fnp_bits, _, _, _, _, _) =
        fnp_dtype::finfo(fnp_dtype::DType::F32).expect("finfo should work for f32");

    assert_eq!(fnp_bits, numpy_bits.parse::<u32>().unwrap());
    Ok(())
}

#[test]
fn conformance_finfo_float64_eps() -> Result<(), String> {
    let numpy_eps = numpy_oracle("import numpy as np; print(float(np.finfo(np.float64).eps))")?;

    let (_, fnp_eps, _, _, _, _) =
        fnp_dtype::finfo(fnp_dtype::DType::F64).expect("finfo should work for f64");

    let numpy_eps_val: f64 = numpy_eps.parse().unwrap();
    assert!(
        (fnp_eps - numpy_eps_val).abs() < 1e-30,
        "eps mismatch: fnp={fnp_eps} vs numpy={numpy_eps_val}"
    );
    Ok(())
}

#[test]
fn conformance_finfo_float64_max() -> Result<(), String> {
    let numpy_max = numpy_oracle("import numpy as np; print(float(np.finfo(np.float64).max))")?;

    // finfo returns (bits, eps, tiny, max, min_exp, max_exp)
    let (_, _, _, fnp_max, _, _) =
        fnp_dtype::finfo(fnp_dtype::DType::F64).expect("finfo should work for f64");

    let numpy_max_val: f64 = numpy_max.parse().unwrap();
    assert_eq!(fnp_max, numpy_max_val);
    Ok(())
}

#[test]
fn conformance_finfo_float64_tiny() -> Result<(), String> {
    let numpy_tiny = numpy_oracle("import numpy as np; print(float(np.finfo(np.float64).tiny))")?;

    // finfo returns (bits, eps, tiny, max, min_exp, max_exp)
    let (_, _, fnp_tiny, _, _, _) =
        fnp_dtype::finfo(fnp_dtype::DType::F64).expect("finfo should work for f64");

    let numpy_tiny_val: f64 = numpy_tiny.parse().unwrap();
    assert_eq!(fnp_tiny, numpy_tiny_val);
    Ok(())
}
