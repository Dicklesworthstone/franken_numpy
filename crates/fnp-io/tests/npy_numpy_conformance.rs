use fnp_io::{
    IOSupportedDType, LoadBytes, load, load_auto, load_complex, load_npz, read_npy_bytes,
};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

const NUMPY_ORACLE_SCRIPT: &str = r#"
import io
import numpy as np
import sys

case = sys.argv[1]

def raw_payload(npy):
    version = tuple(npy[6:8])
    if version == (1, 0):
        header_len = int.from_bytes(npy[8:10], "little")
        offset = 10
    elif version in ((2, 0), (3, 0)):
        header_len = int.from_bytes(npy[8:12], "little")
        offset = 12
    else:
        raise AssertionError(f"unexpected npy version {version}")
    return npy[offset + header_len:]

def emit_npy(arr):
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    npy = buf.getvalue()
    print("npy_hex=" + npy.hex())
    print("payload_hex=" + raw_payload(npy).hex())
    print("dtype=" + arr.dtype.str)
    print("shape=" + ",".join(str(dim) for dim in arr.shape))
    fortran = arr.flags.f_contiguous and not arr.flags.c_contiguous
    print("fortran=" + ("1" if fortran else "0"))
    flat = arr.ravel(order="A")
    if np.issubdtype(arr.dtype, np.complexfloating):
        values = ",".join(
            f"{float(value.real):.17g}:{float(value.imag):.17g}" for value in flat
        )
        print("complex_values=" + values)
    else:
        values = ",".join(f"{float(value):.17g}" for value in flat)
        print("values=" + values)

def emit_npz(compressed):
    buf = io.BytesIO()
    floats = np.array([1.25, -2.5], dtype=np.dtype("<f8"))
    ints = np.array([1, 255], dtype=np.dtype("|u1"))
    if compressed:
        np.savez_compressed(buf, floats=floats, ints=ints)
    else:
        np.savez(buf, floats=floats, ints=ints)
    print("npz_hex=" + buf.getvalue().hex())

if case == "f64_c_order":
    emit_npy(np.arange(6, dtype=np.dtype("<f8")).reshape(2, 3))
elif case == "i16_big_endian":
    emit_npy(np.array([-2, 0, 257], dtype=np.dtype(">i2")))
elif case == "f32_fortran_order":
    emit_npy(np.asfortranarray(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.dtype("<f4"))))
elif case == "complex128":
    emit_npy(np.array([1 + 2j, -3 + 0.5j], dtype=np.dtype("<c16")))
elif case == "u8_empty":
    emit_npy(np.array([], dtype=np.dtype("|u1")))
elif case == "npz_store":
    emit_npz(False)
elif case == "npz_deflate":
    emit_npz(True)
else:
    raise AssertionError(f"unknown case {case}")
"#;

#[derive(Debug)]
struct NumpyNpyOracle {
    npy_bytes: Vec<u8>,
    payload: Vec<u8>,
    dtype_descr: String,
    shape: Vec<usize>,
    fortran_order: bool,
    values: Vec<f64>,
    complex_values: Vec<(f64, f64)>,
}

fn repo_python() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(".venv-numpy314/bin/python3")
}

fn python_has_numpy(python: &str) -> bool {
    Command::new(python)
        .arg("-c")
        .arg("import numpy")
        .output()
        .is_ok_and(|output| output.status.success())
}

fn numpy_python() -> Result<&'static str, String> {
    static PYTHON: OnceLock<Option<String>> = OnceLock::new();
    PYTHON
        .get_or_init(|| {
            if let Ok(configured) = std::env::var("FNP_ORACLE_PYTHON")
                && python_has_numpy(&configured)
            {
                return Some(configured);
            }

            let repo = repo_python();
            if repo.is_file() {
                let candidate = repo.display().to_string();
                if python_has_numpy(&candidate) {
                    return Some(candidate);
                }
            }

            if python_has_numpy("python3") {
                return Some("python3".to_string());
            }

            None
        })
        .as_deref()
        .ok_or_else(|| {
            "npy_numpy_conformance requires FNP_ORACLE_PYTHON, repo .venv-numpy314, or python3 with numpy"
                .to_string()
        })
}

fn run_numpy_case(case: &str) -> Result<BTreeMap<String, String>, String> {
    let output = Command::new(numpy_python()?)
        .arg("-c")
        .arg(NUMPY_ORACLE_SCRIPT)
        .arg(case)
        .output()
        .map_err(|err| format!("failed to run NumPy oracle: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "NumPy oracle case {case} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8(output.stdout)
        .map_err(|err| format!("NumPy oracle output should be UTF-8: {err}"))?;
    stdout
        .lines()
        .map(|line| {
            line.split_once('=')
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .ok_or_else(|| format!("malformed oracle output line {line:?}"))
        })
        .collect()
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, String> {
    if !hex.len().is_multiple_of(2) {
        return Err("hex string must have even length".to_string());
    }
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digits = std::str::from_utf8(pair)
                .map_err(|err| format!("hex digits should be ASCII: {err}"))?;
            u8::from_str_radix(digits, 16)
                .map_err(|err| format!("oracle should emit valid hex: {err}"))
        })
        .collect()
}

fn parse_shape(raw: &str) -> Result<Vec<usize>, String> {
    if raw.trim().is_empty() {
        Ok(Vec::new())
    } else {
        raw.split(',')
            .map(|part| {
                part.parse::<usize>()
                    .map_err(|err| format!("shape dimension {part:?} failed to parse: {err}"))
            })
            .collect()
    }
}

fn parse_values(raw: Option<&String>) -> Result<Vec<f64>, String> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    if raw.trim().is_empty() {
        Ok(Vec::new())
    } else {
        raw.split(',')
            .map(|part| {
                part.parse::<f64>()
                    .map_err(|err| format!("float oracle value {part:?} failed to parse: {err}"))
            })
            .collect()
    }
}

fn parse_complex_values(raw: Option<&String>) -> Result<Vec<(f64, f64)>, String> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    if raw.trim().is_empty() {
        Ok(Vec::new())
    } else {
        raw.split(',')
            .map(|pair| {
                let (real, imag) = pair.split_once(':').ok_or_else(|| {
                    format!("complex oracle value should use real:imag, got {pair:?}")
                })?;
                let real = real
                    .parse::<f64>()
                    .map_err(|err| format!("complex real component failed to parse: {err}"))?;
                let imag = imag
                    .parse::<f64>()
                    .map_err(|err| format!("complex imaginary component failed to parse: {err}"))?;
                Ok((real, imag))
            })
            .collect()
    }
}

fn require_line<'a>(lines: &'a BTreeMap<String, String>, key: &str) -> Result<&'a String, String> {
    lines
        .get(key)
        .ok_or_else(|| format!("NumPy oracle omitted required line {key}"))
}

fn numpy_npy_oracle(case: &str) -> Result<NumpyNpyOracle, String> {
    let lines = run_numpy_case(case)?;
    Ok(NumpyNpyOracle {
        npy_bytes: hex_to_bytes(require_line(&lines, "npy_hex")?)?,
        payload: hex_to_bytes(require_line(&lines, "payload_hex")?)?,
        dtype_descr: require_line(&lines, "dtype")?.clone(),
        shape: parse_shape(require_line(&lines, "shape")?)?,
        fortran_order: require_line(&lines, "fortran")? == "1",
        values: parse_values(lines.get("values"))?,
        complex_values: parse_complex_values(lines.get("complex_values"))?,
    })
}

fn numpy_npz_oracle(case: &str) -> Result<Vec<u8>, String> {
    let lines = run_numpy_case(case)?;
    hex_to_bytes(require_line(&lines, "npz_hex")?)
}

fn ensure(condition: bool, message: impl Into<String>) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn ensure_eq<T>(actual: T, expected: T, context: impl Into<String>) -> Result<(), String>
where
    T: PartialEq + Debug,
{
    ensure(
        actual == expected,
        format!(
            "{} mismatch: actual={actual:?} expected={expected:?}",
            context.into()
        ),
    )
}

fn assert_close(actual: &[f64], expected: &[f64]) -> Result<(), String> {
    ensure_eq(actual.len(), expected.len(), "value length")?;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let delta = (actual - expected).abs();
        ensure(
            delta <= 1e-6,
            format!(
                "value mismatch at {index}: actual={actual:?} expected={expected:?} delta={delta:?}"
            ),
        )?;
    }
    Ok(())
}

#[test]
fn numpy_generated_npy_payloads_parse_with_matching_headers_and_values() -> Result<(), String> {
    let cases = [
        ("f64_c_order", IOSupportedDType::F64),
        ("i16_big_endian", IOSupportedDType::I16Be),
        ("f32_fortran_order", IOSupportedDType::F32),
        ("u8_empty", IOSupportedDType::U8),
    ];

    for (case_id, expected_dtype) in cases {
        let oracle = numpy_npy_oracle(case_id)?;
        ensure_eq(
            IOSupportedDType::decode(&oracle.dtype_descr)
                .map_err(|err| format!("{case_id}: oracle dtype should decode: {err}"))?,
            expected_dtype,
            format!("{case_id}: dtype descriptor sanity check"),
        )?;

        let parsed = read_npy_bytes(&oracle.npy_bytes, false)
            .map_err(|err| format!("{case_id}: NumPy NPY should parse: {err}"))?;
        ensure_eq(parsed.version, (1, 0), format!("{case_id}: version"))?;
        ensure_eq(
            parsed.header.descr,
            expected_dtype,
            format!("{case_id}: dtype"),
        )?;
        ensure_eq(
            parsed.header.shape,
            oracle.shape.clone(),
            format!("{case_id}: shape"),
        )?;
        ensure_eq(
            parsed.header.fortran_order,
            oracle.fortran_order,
            format!("{case_id}: fortran_order"),
        )?;
        ensure_eq(
            parsed.payload.as_ref(),
            oracle.payload.as_slice(),
            format!("{case_id}: raw payload bytes"),
        )?;

        let (shape, values, dtype) = load(&oracle.npy_bytes)
            .map_err(|err| format!("{case_id}: NumPy NPY should load: {err}"))?;
        ensure_eq(shape, oracle.shape, format!("{case_id}: loaded shape"))?;
        ensure_eq(dtype, expected_dtype, format!("{case_id}: loaded dtype"))?;
        assert_close(&values, &oracle.values)?;
    }
    Ok(())
}

#[test]
fn numpy_generated_complex_npy_loads_interleaved_complex_values() -> Result<(), String> {
    let oracle = numpy_npy_oracle("complex128")?;
    let parsed = read_npy_bytes(&oracle.npy_bytes, false)
        .map_err(|err| format!("NumPy complex NPY should parse: {err}"))?;
    ensure_eq(parsed.version, (1, 0), "complex128: version")?;
    ensure_eq(
        parsed.header.descr,
        IOSupportedDType::Complex128,
        "complex128: dtype",
    )?;
    ensure_eq(
        parsed.header.shape,
        oracle.shape.clone(),
        "complex128: shape",
    )?;
    ensure(
        !parsed.header.fortran_order,
        "complex128 should not set fortran_order",
    )?;
    ensure_eq(
        parsed.payload.as_ref(),
        oracle.payload.as_slice(),
        "complex128: raw payload bytes",
    )?;

    let (shape, values, dtype) = load_complex(&oracle.npy_bytes)
        .map_err(|err| format!("NumPy complex NPY should load: {err}"))?;
    ensure_eq(shape, oracle.shape, "complex128: loaded shape")?;
    ensure_eq(
        dtype,
        IOSupportedDType::Complex128,
        "complex128: loaded dtype",
    )?;
    ensure_eq(
        values.len(),
        oracle.complex_values.len(),
        "complex128: loaded value length",
    )?;
    for (index, ((actual_real, actual_imag), (expected_real, expected_imag))) in
        values.iter().zip(&oracle.complex_values).enumerate()
    {
        ensure(
            (actual_real - expected_real).abs() <= 1e-12,
            format!("real component mismatch at {index}"),
        )?;
        ensure(
            (actual_imag - expected_imag).abs() <= 1e-12,
            format!("imag component mismatch at {index}"),
        )?;
    }
    Ok(())
}

#[test]
fn numpy_generated_npz_archives_dispatch_and_decode_members() -> Result<(), String> {
    for case_id in ["npz_store", "npz_deflate"] {
        let npz_bytes = numpy_npz_oracle(case_id)?;
        match load_auto(&npz_bytes, false)
            .map_err(|err| format!("{case_id}: NumPy NPZ should auto-dispatch: {err}"))?
        {
            LoadBytes::Npz(entries) => {
                let names = entries
                    .iter()
                    .map(|entry| entry.name.as_str())
                    .collect::<Vec<_>>();
                ensure_eq(
                    names,
                    vec!["floats", "ints"],
                    format!("{case_id}: member names"),
                )?;
            }
            other => return Err(format!("{case_id}: expected NPZ dispatch, got {other:?}")),
        }

        let decoded = load_npz(&npz_bytes, false)
            .map_err(|err| format!("{case_id}: NumPy NPZ should decode: {err}"))?;
        ensure_eq(decoded.len(), 2, format!("{case_id}: member count"))?;

        let floats = decoded
            .iter()
            .find(|(name, _, _, _)| name == "floats")
            .ok_or_else(|| format!("{case_id}: missing floats member"))?;
        ensure_eq(
            floats.1.clone(),
            vec![2],
            format!("{case_id}: floats shape"),
        )?;
        ensure_eq(
            floats.3,
            IOSupportedDType::F64,
            format!("{case_id}: floats dtype"),
        )?;
        assert_close(&floats.2, &[1.25, -2.5])?;

        let ints = decoded
            .iter()
            .find(|(name, _, _, _)| name == "ints")
            .ok_or_else(|| format!("{case_id}: missing ints member"))?;
        ensure_eq(ints.1.clone(), vec![2], format!("{case_id}: ints shape"))?;
        ensure_eq(
            ints.3,
            IOSupportedDType::U8,
            format!("{case_id}: ints dtype"),
        )?;
        assert_close(&ints.2, &[1.0, 255.0])?;
    }
    Ok(())
}
