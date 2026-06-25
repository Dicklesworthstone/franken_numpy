//! Conformance tests for numpy.frexp and numpy.modf against NumPy oracle.
//!
//! Tests tuple-returning decomposition functions:
//! - frexp(x): returns (mantissa, exponent) where x = mantissa * 2**exponent
//! - modf(x): returns (fractional, integral) parts

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
fn frexp_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1.0])",
        "np.array([2.0])",
        "np.array([4.0])",
        "np.array([0.5])",
        "np.array([0.25])",
        "np.array([0.0])",
        "np.array([-1.0])",
        "np.array([-2.0])",
        "np.array([-0.5])",
        "np.array([3.0])",
        "np.array([1.5])",
        "np.array([1e10])",
        "np.array([1e-10])",
        "np.array([1e100])",
        "np.array([1e-100])",
        "np.array([1e308])",
        "np.array([1e-308])",
        "np.array([1.0, 2.0, 3.0, 4.0])",
        "np.array([-1.0, -2.0, -3.0, -4.0])",
        "np.array([0.5, 1.0, 1.5, 2.0])",
        "np.linspace(0.1, 10, 10)",
        "np.array([[1.0, 2.0], [4.0, 8.0]])",
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([np.nan])",
        "np.array([0.0, np.inf, np.nan])",
        "np.array([-0.0])",
        "np.array([np.finfo(float).eps])",
        "np.array([np.finfo(float).tiny])",
    ];

    for arr_expr in &test_cases {
        let script = format!(
            "import numpy as np; m, e = np.frexp({arr_expr}); print(m.flatten().tolist(), e.flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "m, e = fnp.frexp({arr_expr}); print(m.flatten().tolist(), e.flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "frexp mismatch for {arr_expr}"
        );
    }

    Ok(())
}

#[test]
fn modf_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1.5])",
        "np.array([2.7])",
        "np.array([0.5])",
        "np.array([0.0])",
        "np.array([-1.5])",
        "np.array([-2.7])",
        "np.array([-0.5])",
        "np.array([3.14159])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.array([10.25])",
        "np.array([-10.25])",
        "np.array([0.999])",
        "np.array([-0.999])",
        "np.array([1e10 + 0.5])",
        "np.array([1.5, 2.5, 3.5, 4.5])",
        "np.array([-1.5, -2.5, -3.5, -4.5])",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.linspace(0, 5, 6)",
        "np.array([[1.5, 2.5], [3.5, 4.5]])",
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([np.nan])",
        "np.array([0.0, np.inf, np.nan])",
        "np.array([-0.0])",
        "np.array([123.456])",
        "np.array([-123.456])",
        "np.array([0.123456789])",
    ];

    for arr_expr in &test_cases {
        let script = format!(
            "import numpy as np; f, i = np.modf({arr_expr}); print(f.flatten().tolist(), i.flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "f, i = fnp.modf({arr_expr}); print(f.flatten().tolist(), i.flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "modf mismatch for {arr_expr}"
        );
    }

    Ok(())
}

#[test]
fn frexp_empty_array_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; m, e = np.frexp(np.array([], dtype=np.float64)); print(m.tolist(), e.tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        "m, e = fnp.frexp(np.array([], dtype=np.float64)); print(m.tolist(), e.tolist())".into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "frexp empty array mismatch"
    );

    Ok(())
}

#[test]
fn frexp_f64_zerocopy_bit_exact_golden_sha256() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE

cases = [
    np.array([], dtype=np.float64),
    np.array([-0.0, 0.0, 1.0, -2.5, 8.0, -16.0], dtype=np.float64),
    np.array(
        [
            [np.inf, -np.inf, np.nan],
            [np.finfo(np.float64).tiny, np.finfo(np.float64).tiny / 2.0, -1.0e-308],
        ],
        dtype=np.float64,
    ),
    np.linspace(-1.0e6, 1.0e6, 257, dtype=np.float64).reshape(257, 1),
]

h = hashlib.sha256()
for x in cases:
    got_m, got_e = mod.frexp(x)
    expected_m, expected_e = np.frexp(x)
    for got, expected in [(got_m, expected_m), (got_e, expected_e)]:
        assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
        assert got.shape == expected.shape, (got.shape, expected.shape)
        assert got.flags["C_CONTIGUOUS"]
        assert got.flags["WRITEABLE"]
        assert got.tobytes() == expected.tobytes(), (got, expected)
        h.update(str(got.dtype).encode())
        h.update(str(got.shape).encode())
        h.update(got.tobytes())

print(h.hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "fnp.frexp f64 array outputs must be bit-identical to numpy"
    );
    assert_eq!(
        fnp_hash, "1af900613e40fc30184fe39d712e8c1f561975f9bd6c85d4825b7db68e8dc3d5",
        "golden sha256 of frexp mantissa/exponent dtype/shape/raw-output bytes"
    );

    Ok(())
}

#[test]
fn modf_empty_array_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; f, i = np.modf(np.array([], dtype=np.float64)); print(f.tolist(), i.tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        "f, i = fnp.modf(np.array([], dtype=np.float64)); print(f.tolist(), i.tolist())".into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "modf empty array mismatch"
    );

    Ok(())
}

#[test]
fn modf_f64_zerocopy_bit_exact_golden_sha256() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE

cases = [
    np.array([], dtype=np.float64),
    np.array([-0.0, 0.0, 1.5, -2.5, 8.0, -16.0], dtype=np.float64),
    np.array(
        [
            1.0,
            -1.0,
            123456.75,
            -123456.75,
            np.nextafter(0.0, 1.0),
            -np.nextafter(0.0, 1.0),
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [np.inf, -np.inf, np.nan],
            [np.finfo(np.float64).tiny, np.finfo(np.float64).tiny / 2.0, -1.0e-308],
        ],
        dtype=np.float64,
    ),
    np.linspace(-1.0e6 - 0.75, 1.0e6 + 0.75, 257, dtype=np.float64).reshape(257, 1),
]

h = hashlib.sha256()
for x in cases:
    got_f, got_i = mod.modf(x)
    expected_f, expected_i = np.modf(x)
    for got, expected in [(got_f, expected_f), (got_i, expected_i)]:
        assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
        assert got.shape == expected.shape, (got.shape, expected.shape)
        assert got.flags["C_CONTIGUOUS"]
        assert got.flags["WRITEABLE"]
        assert got.tobytes() == expected.tobytes(), (got, expected)
        h.update(str(got.dtype).encode())
        h.update(str(got.shape).encode())
        h.update(got.tobytes())

print(h.hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "fnp.modf f64 array outputs must be bit-identical to numpy"
    );
    assert_eq!(
        fnp_hash, "ee45bb6c4a9a21df366f3698b20d11a1a1fb41a91986f8b296c2fb7b56b3ec7b",
        "golden sha256 of modf fractional/integral dtype/shape/raw-output bytes"
    );

    Ok(())
}

#[test]
fn modf_f64_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = (1 << 21) + 65
x = np.linspace(-1.0e6 - 0.75, 1.0e6 + 0.75, n, dtype=np.float64)
x[0] = -0.0
x[1] = 0.0
x[2] = np.inf
x[3] = -np.inf
x[4] = np.nan
x[5] = np.nextafter(0.0, 1.0)
actual_f, actual_i = fnp.modf(x)
expected_f, expected_i = np.modf(x)
print(
    actual_f.dtype == expected_f.dtype
    and actual_i.dtype == expected_i.dtype
    and actual_f.shape == expected_f.shape
    and actual_i.shape == expected_i.shape
    and actual_f.tobytes() == expected_f.tobytes()
    and actual_i.tobytes() == expected_i.tobytes()
)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large f64 modf parallel path must be bit-identical to numpy"
    );
    Ok(())
}

#[test]
fn frexp_dtype_matches_numpy() -> Result<(), String> {
    let script =
        "import numpy as np; m, e = np.frexp(np.array([1.0, 2.0])); print(m.dtype, e.dtype)";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("m, e = fnp.frexp(np.array([1.0, 2.0])); print(m.dtype, e.dtype)".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "frexp dtype mismatch"
    );

    Ok(())
}

#[test]
fn modf_dtype_matches_numpy() -> Result<(), String> {
    let script =
        "import numpy as np; f, i = np.modf(np.array([1.5, 2.5])); print(f.dtype, i.dtype)";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("f, i = fnp.modf(np.array([1.5, 2.5])); print(f.dtype, i.dtype)".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "modf dtype mismatch"
    );

    Ok(())
}

#[test]
fn frexp_roundtrip_matches_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 0.25])
m, e = np.frexp(x)
reconstructed = np.ldexp(m, e)
print(np.allclose(x, reconstructed))
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 0.25])
m, e = fnp.frexp(x)
reconstructed = fnp.ldexp(m, e)
print(np.allclose(x, reconstructed))
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "frexp roundtrip mismatch"
    );

    Ok(())
}

#[test]
fn modf_sum_matches_original() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([1.5, 2.7, -3.2, 4.9, -0.5])
f, i = np.modf(x)
print(np.allclose(f + i, x))
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([1.5, 2.7, -3.2, 4.9, -0.5])
f, i = fnp.modf(x)
print(np.allclose(f + i, x))
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(numpy_result.trim(), rust_result.trim(), "modf sum mismatch");

    Ok(())
}

#[test]
fn frexp_shape_preserved() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
m, e = np.frexp(x)
print(m.shape, e.shape)
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
m, e = fnp.frexp(x)
print(m.shape, e.shape)
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "frexp shape mismatch"
    );

    Ok(())
}

#[test]
fn modf_shape_preserved() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
f, i = np.modf(x)
print(f.shape, i.shape)
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
f, i = fnp.modf(x)
print(f.shape, i.shape)
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "modf shape mismatch"
    );

    Ok(())
}

#[test]
fn frexp_modf_scalar_return_type_matches_numpy() -> Result<(), String> {
    let frexp_script = fnp_script(
        r#"
x = np.float64(3.5)
fnp_m, fnp_e = fnp.frexp(x)
np_m, np_e = np.frexp(x)
types_match = (type(fnp_m).__name__ == type(np_m).__name__ and
               type(fnp_e).__name__ == type(np_e).__name__)
print(types_match, fnp_m, fnp_e, np_m, np_e)
"#
        .into(),
    );
    let result = numpy_oracle(&frexp_script)?;
    assert!(
        result.trim().starts_with("True"),
        "frexp scalar return type should match numpy: {result}"
    );

    let modf_script = fnp_script(
        r#"
x = np.float64(3.5)
fnp_f, fnp_i = fnp.modf(x)
np_f, np_i = np.modf(x)
types_match = (type(fnp_f).__name__ == type(np_f).__name__ and
               type(fnp_i).__name__ == type(np_i).__name__)
print(types_match, fnp_f, fnp_i, np_f, np_i)
"#
        .into(),
    );
    let result = numpy_oracle(&modf_script)?;
    assert!(
        result.trim().starts_with("True"),
        "modf scalar return type should match numpy: {result}"
    );

    Ok(())
}
