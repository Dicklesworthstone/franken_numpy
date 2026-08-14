use fnp_io::{
    IOSupportedDType, LoadBytes, load, load_auto, load_complex, load_npz, read_npy_bytes, savez,
    savez_compressed, write_npy_bytes,
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
import zipfile

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

def emit_loaded_npz(raw_hex):
    raw = bytes.fromhex(raw_hex)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        methods = [f"{info.filename}:{info.compress_type}" for info in zf.infolist()]
        print("zip_methods=" + ",".join(methods))
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        print("npz_names=" + ",".join(archive.files))
        for name in archive.files:
            arr = archive[name]
            print(f"{name}_dtype=" + arr.dtype.str)
            print(f"{name}_shape=" + ",".join(str(dim) for dim in arr.shape))
            values = ",".join(f"{float(value):.17g}" for value in arr.ravel(order="C"))
            print(f"{name}_values=" + values)

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
elif case.startswith("version_"):
    # numpy can be told which .npy format version to emit, which is the only
    # practical way to obtain a 2.0/3.0 header: 2.0 otherwise requires a header
    # over 65535 bytes and 3.0 requires utf8 field names.
    import numpy.lib.format as fmt
    major = int(case.split("_")[1])
    arr = np.arange(6, dtype=np.dtype("<f8")).reshape(2, 3)
    buf = io.BytesIO()
    fmt.write_array(buf, arr, version=(major, 0), allow_pickle=False)
    npy = buf.getvalue()
    print("npy_hex=" + npy.hex())
    print("payload_hex=" + raw_payload(npy).hex())
    print("dtype=" + arr.dtype.str)
    print("shape=" + ",".join(str(dim) for dim in arr.shape))
    print("fortran=0")
    print("values=" + ",".join(f"{float(v):.17g}" for v in arr.ravel()))
elif case == "bool_mixed":
    emit_npy(np.array([True, False, True, True], dtype=np.dtype("|b1")))
elif case == "i32_range":
    emit_npy(np.array([-2147483648, -1, 0, 1, 2147483647], dtype=np.dtype("<i4")))
elif case == "i64_range":
    emit_npy(
        np.array(
            [-9223372036854775808, -1, 0, 1, 9223372036854775807], dtype=np.dtype("<i8")
        )
    )
elif case == "f32_specials":
    emit_npy(np.array([np.inf, -np.inf, np.nan, -0.0, 0.0, 1.5], dtype=np.dtype("<f4")))
elif case == "f64_specials":
    emit_npy(np.array([np.inf, -np.inf, np.nan, -0.0, 0.0, 1.5], dtype=np.dtype("<f8")))
elif case == "complex64_specials":
    emit_npy(
        np.array(
            [complex(np.inf, -0.0), complex(np.nan, 1.5), complex(-0.0, -np.inf)],
            dtype=np.dtype("<c8"),
        )
    )
elif case == "complex128_specials":
    emit_npy(
        np.array(
            [complex(np.inf, -0.0), complex(np.nan, 1.5), complex(-0.0, -np.inf)],
            dtype=np.dtype("<c16"),
        )
    )
elif case == "npz_store":
    emit_npz(False)
elif case == "npz_deflate":
    emit_npz(True)
elif case == "load_npz_hex":
    emit_loaded_npz(sys.argv[2])
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
    run_numpy_case_with_extra_arg(case, None)
}

fn run_numpy_case_with_extra_arg(
    case: &str,
    extra_arg: Option<&str>,
) -> Result<BTreeMap<String, String>, String> {
    let mut command = Command::new(numpy_python()?);
    command.arg("-c").arg(NUMPY_ORACLE_SCRIPT).arg(case);
    if let Some(extra_arg) = extra_arg {
        command.arg(extra_arg);
    }

    let output = command
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

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        for nibble in [byte >> 4, byte & 0x0f] {
            let digit = match nibble {
                0..=9 => b'0' + nibble,
                10..=15 => b'a' + (nibble - 10),
                _ => b'?',
            };
            out.push(char::from(digit));
        }
    }
    out
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, String> {
    if !hex.len().is_multiple_of(2) {
        return Err("hex string must have even length".to_string());
    }
    hex.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
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

fn numpy_load_npz_summary(npz_bytes: &[u8]) -> Result<BTreeMap<String, String>, String> {
    run_numpy_case_with_extra_arg("load_npz_hex", Some(&bytes_to_hex(npz_bytes)))
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
fn numpy_written_v2_and_v3_headers_parse_with_the_right_preamble() -> Result<(), String> {
    // Our reader claims to handle .npy versions 1.0, 2.0 and 3.0 — the 2.0/3.0 arms use a
    // 4-byte header length where 1.0 uses 2 bytes. Nothing exercised those arms against a
    // file numpy actually wrote: the only version test in the crate is named
    // numpy_oracle_npy_magic_and_version but calls no numpy at all, checking our own
    // writer's v1.0 output against hand-written byte offsets.
    //
    // A 2.0 header normally needs to exceed 65535 bytes and 3.0 needs utf8 field names,
    // which is why they were never covered incidentally. numpy.lib.format.write_array
    // takes an explicit version, so both are reachable directly.
    for (case, major) in [("version_2", 2u8), ("version_3", 3u8)] {
        let oracle = numpy_npy_oracle(case)?;
        // The preamble numpy emitted really is the version we asked for, and really does
        // use the wide header-length field — otherwise this would silently be a 1.0 test.
        ensure_eq(
            oracle.npy_bytes[6],
            major,
            format!("{case}: numpy should have written major version {major}"),
        )?;
        ensure_eq(oracle.npy_bytes[7], 0, format!("{case}: minor version"))?;
        let header_len = u32::from_le_bytes([
            oracle.npy_bytes[8],
            oracle.npy_bytes[9],
            oracle.npy_bytes[10],
            oracle.npy_bytes[11],
        ]) as usize;
        ensure_eq(
            oracle.npy_bytes.len(),
            12 + header_len + oracle.payload.len(),
            format!("{case}: total length implies a 4-byte header-length field"),
        )?;

        let parsed = read_npy_bytes(&oracle.npy_bytes, false)
            .map_err(|err| format!("{case}: numpy-written v{major}.0 should parse: {err}"))?;
        ensure_eq(
            parsed.version,
            (major, 0),
            format!("{case}: parsed version"),
        )?;
        ensure_eq(
            parsed.header.descr,
            IOSupportedDType::F64,
            format!("{case}: dtype"),
        )?;
        ensure_eq(
            parsed.header.shape.clone(),
            oracle.shape.clone(),
            format!("{case}: shape"),
        )?;
        ensure_eq(
            bytes_to_hex(parsed.payload.as_ref()),
            bytes_to_hex(&oracle.payload),
            format!("{case}: payload bytes"),
        )?;

        let (shape, values, dtype) =
            load(&oracle.npy_bytes).map_err(|err| format!("{case}: load should succeed: {err}"))?;
        ensure_eq(shape, oracle.shape, format!("{case}: loaded shape"))?;
        ensure_eq(
            dtype,
            IOSupportedDType::F64,
            format!("{case}: loaded dtype"),
        )?;
        assert_close(&values, &oracle.values)?;
    }
    Ok(())
}

#[test]
fn uncovered_dtypes_and_special_floats_round_trip_byte_exactly() -> Result<(), String> {
    // The existing conformance cases compare decimal `values=` lines, which cannot
    // distinguish -0.0 from 0.0 and cannot see a NaN's payload bits at all. These dtypes
    // (bool/i32/i64/f32/f64/complex64/complex128) are checked on RAW BYTES in both
    // directions: NumPy's payload must survive our parse untouched, and re-serialising the
    // parsed header + payload must reproduce NumPy's whole .npy file byte for byte.
    let cases = [
        ("bool_mixed", IOSupportedDType::Bool),
        ("i32_range", IOSupportedDType::I32),
        ("i64_range", IOSupportedDType::I64),
        ("f32_specials", IOSupportedDType::F32),
        ("f64_specials", IOSupportedDType::F64),
        ("complex64_specials", IOSupportedDType::Complex64),
        ("complex128_specials", IOSupportedDType::Complex128),
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
        ensure_eq(
            parsed.header.descr,
            expected_dtype,
            format!("{case_id}: dtype"),
        )?;
        ensure_eq(
            parsed.header.shape.clone(),
            oracle.shape.clone(),
            format!("{case_id}: shape"),
        )?;
        ensure_eq(
            bytes_to_hex(parsed.payload.as_ref()),
            bytes_to_hex(&oracle.payload),
            format!("{case_id}: raw payload bytes"),
        )?;

        // Round trip: our writer must reproduce NumPy's file exactly, header padding and
        // all — not merely a file NumPy would accept.
        let rewritten = write_npy_bytes(&parsed.header, parsed.payload.as_ref(), false)
            .map_err(|err| format!("{case_id}: re-serialisation should succeed: {err}"))?;
        ensure_eq(
            bytes_to_hex(&rewritten),
            bytes_to_hex(&oracle.npy_bytes),
            format!("{case_id}: re-serialised .npy bytes"),
        )?;
    }

    // Signed zero and the non-finite patterns, asserted on the exact bytes NumPy wrote.
    // Layout is [inf, -inf, nan, -0.0, 0.0, 1.5] in both float widths.
    let f64_specials = numpy_npy_oracle("f64_specials")?;
    let f64_words: Vec<u64> = f64_specials
        .payload
        .as_chunks::<8>()
        .0
        .iter()
        .map(|w| u64::from_le_bytes(*w))
        .collect();
    ensure_eq(f64_words.len(), 6, "f64_specials: word count")?;
    ensure_eq(
        f64_words[0],
        f64::INFINITY.to_bits(),
        "f64_specials: +inf bits",
    )?;
    ensure_eq(
        f64_words[1],
        f64::NEG_INFINITY.to_bits(),
        "f64_specials: -inf bits",
    )?;
    ensure(
        f64::from_bits(f64_words[2]).is_nan(),
        "f64_specials: slot 2 must be NaN",
    )?;
    ensure_eq(
        f64_words[3],
        (-0.0f64).to_bits(),
        "f64_specials: -0.0 must keep its sign bit, not collapse to +0.0",
    )?;
    ensure_eq(f64_words[4], 0.0f64.to_bits(), "f64_specials: +0.0 bits")?;
    ensure(
        f64_words[3] != f64_words[4],
        "f64_specials: -0.0 and +0.0 must not share a bit pattern",
    )?;

    let f32_specials = numpy_npy_oracle("f32_specials")?;
    let f32_words: Vec<u32> = f32_specials
        .payload
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    ensure_eq(f32_words.len(), 6, "f32_specials: word count")?;
    ensure_eq(
        f32_words[3],
        (-0.0f32).to_bits(),
        "f32_specials: -0.0 must keep its sign bit",
    )?;
    ensure(
        f32::from_bits(f32_words[2]).is_nan(),
        "f32_specials: slot 2 must be NaN",
    )?;

    // A complex element's imaginary half carries its own sign bit; -0.0j is the half most
    // easily lost by a parse that reconstructs through a real-valued intermediate.
    let c128 = numpy_npy_oracle("complex128_specials")?;
    let c128_words: Vec<u64> = c128
        .payload
        .as_chunks::<8>()
        .0
        .iter()
        .map(|w| u64::from_le_bytes(*w))
        .collect();
    ensure_eq(c128_words.len(), 6, "complex128_specials: word count")?;
    ensure_eq(
        c128_words[1],
        (-0.0f64).to_bits(),
        "complex128_specials: imaginary -0.0 must keep its sign bit",
    )?;
    ensure_eq(
        c128_words[4],
        (-0.0f64).to_bits(),
        "complex128_specials: real -0.0 must keep its sign bit",
    )?;
    ensure_eq(
        c128_words[5],
        f64::NEG_INFINITY.to_bits(),
        "complex128_specials: imaginary -inf bits",
    )?;
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

#[test]
fn rust_generated_npz_archives_load_in_numpy() -> Result<(), String> {
    let floats = [1.25, -2.5];
    let floats_shape = [2_usize];
    let ints = [1.0, 255.0];
    let ints_shape = [2_usize];
    let entries = [
        (
            "floats",
            floats_shape.as_slice(),
            floats.as_slice(),
            IOSupportedDType::F64,
        ),
        (
            "ints",
            ints_shape.as_slice(),
            ints.as_slice(),
            IOSupportedDType::U8,
        ),
    ];

    for (case_id, npz_bytes, expected_method) in [
        (
            "rust_store",
            savez(&entries).map_err(|err| format!("rust_store: save failed: {err}"))?,
            "0",
        ),
        (
            "rust_deflate",
            savez_compressed(&entries)
                .map_err(|err| format!("rust_deflate: save failed: {err}"))?,
            "8",
        ),
    ] {
        let lines = numpy_load_npz_summary(&npz_bytes)?;
        let expected_methods = format!("floats.npy:{expected_method},ints.npy:{expected_method}");
        ensure_eq(
            require_line(&lines, "zip_methods")?.as_str(),
            expected_methods.as_str(),
            format!("{case_id}: ZIP compression methods"),
        )?;
        ensure_eq(
            require_line(&lines, "npz_names")?.as_str(),
            "floats,ints",
            format!("{case_id}: NumPy-visible member names"),
        )?;
        ensure_eq(
            require_line(&lines, "floats_dtype")?.as_str(),
            "<f8",
            format!("{case_id}: floats dtype"),
        )?;
        ensure_eq(
            parse_shape(require_line(&lines, "floats_shape")?)?,
            vec![2],
            format!("{case_id}: floats shape"),
        )?;
        assert_close(&parse_values(lines.get("floats_values"))?, &floats)?;
        ensure_eq(
            require_line(&lines, "ints_dtype")?.as_str(),
            "|u1",
            format!("{case_id}: ints dtype"),
        )?;
        ensure_eq(
            parse_shape(require_line(&lines, "ints_shape")?)?,
            vec![2],
            format!("{case_id}: ints shape"),
        )?;
        assert_close(&parse_values(lines.get("ints_values"))?, &ints)?;
    }
    Ok(())
}
