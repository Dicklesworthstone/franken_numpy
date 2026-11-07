//! NumPy's invalid float casts depend on the host conversion instructions.
//! These fixtures cover the Linux x86-64 kernels implemented by `ArrayStorage`.
#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use fnp_dtype::{ArrayStorage, DType};
use half::f16;
use std::fmt::Write as _;
use std::io::Write as _;
use std::process::{Command, Stdio};

const INTEGER_TARGETS: [(DType, &str); 8] = [
    (DType::I8, "i1"),
    (DType::I16, "i2"),
    (DType::I32, "i4"),
    (DType::I64, "i8"),
    (DType::U8, "u1"),
    (DType::U16, "u2"),
    (DType::U32, "u4"),
    (DType::U64, "u8"),
];

fn integer_values(storage: &ArrayStorage) -> Vec<i128> {
    (0..storage.len())
        .map(|index| storage.get_i128(index).unwrap())
        .collect()
}

#[test]
fn invalid_casts_use_signed_indefinite_and_unsigned_split() {
    let source = ArrayStorage::F64(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
    for target in [DType::I8, DType::I16, DType::U8, DType::U16] {
        assert_eq!(
            integer_values(&source.cast_to(target).unwrap()),
            vec![0; 3],
            "{target:?} keeps the low bits of the signed indefinite value"
        );
    }
    assert_eq!(
        integer_values(&source.cast_to(DType::I32).unwrap()),
        vec![i128::from(i32::MIN); 3]
    );
    assert_eq!(
        integer_values(&source.cast_to(DType::I64).unwrap()),
        vec![i128::from(i64::MIN); 3]
    );
    assert_eq!(
        integer_values(&source.cast_to(DType::U64).unwrap()),
        vec![1_i128 << 63, 0, 1_i128 << 63]
    );
}

#[test]
fn finite_overflow_preserves_the_destination_conversion_width() {
    let source = ArrayStorage::F64(vec![
        -2_147_483_649.0,
        -2_147_483_648.9,
        -257.9,
        -129.9,
        127.9,
        255.9,
        257.9,
        65_535.9,
        2_147_483_647.9,
        2_147_483_648.0,
        4_294_967_297.0,
        f64::MAX,
        -f64::MAX,
    ]);
    assert_eq!(
        integer_values(&source.cast_to(DType::I8).unwrap()),
        vec![0, 0, -1, 127, 127, -1, 1, -1, -1, 0, 0, 0, 0]
    );
    assert_eq!(
        integer_values(&source.cast_to(DType::I16).unwrap()),
        vec![0, 0, -257, -129, 127, 255, 257, -1, -1, 0, 0, 0, 0]
    );
    assert_eq!(
        integer_values(&source.cast_to(DType::U8).unwrap()),
        vec![0, 0, 255, 127, 127, 255, 1, 255, 255, 0, 0, 0, 0]
    );
    assert_eq!(
        integer_values(&source.cast_to(DType::U16).unwrap()),
        vec![
            0, 0, 65_279, 65_407, 127, 255, 257, 65_535, 65_535, 0, 0, 0, 0
        ]
    );
    assert_eq!(
        integer_values(&source.cast_to(DType::I32).unwrap()),
        vec![
            i128::from(i32::MIN),
            i128::from(i32::MIN),
            -257,
            -129,
            127,
            255,
            257,
            65_535,
            i128::from(i32::MAX),
            i128::from(i32::MIN),
            i128::from(i32::MIN),
            i128::from(i32::MIN),
            i128::from(i32::MIN),
        ]
    );
}

#[test]
fn sixty_four_bit_boundaries_keep_integer_payloads_exact() {
    let two_to_63 = 9_223_372_036_854_775_808.0_f64;
    let two_to_64 = 18_446_744_073_709_551_616.0_f64;
    let source = ArrayStorage::F64(vec![
        -two_to_63,
        (-two_to_63).next_down(),
        -1.9,
        two_to_63.next_down(),
        two_to_63,
        two_to_63.next_up(),
        two_to_64.next_down(),
        two_to_64,
        f64::MAX,
        -f64::MAX,
    ]);
    assert_eq!(
        integer_values(&source.cast_to(DType::I64).unwrap()),
        vec![
            i128::from(i64::MIN),
            i128::from(i64::MIN),
            -1,
            (1_i128 << 63) - 1024,
            i128::from(i64::MIN),
            i128::from(i64::MIN),
            i128::from(i64::MIN),
            i128::from(i64::MIN),
            i128::from(i64::MIN),
            i128::from(i64::MIN),
        ]
    );
    assert_eq!(
        integer_values(&source.cast_to(DType::U64).unwrap()),
        vec![
            1_i128 << 63,
            1_i128 << 63,
            i128::from(u64::MAX),
            (1_i128 << 63) - 1024,
            1_i128 << 63,
            (1_i128 << 63) + 2048,
            (1_i128 << 64) - 2048,
            0,
            0,
            1_i128 << 63,
        ]
    );
}

#[test]
fn source_precision_is_applied_before_integer_conversion() {
    let values = [2_147_483_647.0, -2_147_483_649.0, 65_535.0];
    let sources = [
        ArrayStorage::F16(values.iter().map(|&value| f16::from_f64(value)).collect()),
        ArrayStorage::F32(values.iter().map(|&value| value as f32).collect()),
        ArrayStorage::F64(values.to_vec()),
    ];
    assert_eq!(
        integer_values(&sources[0].cast_to(DType::I32).unwrap()),
        vec![i128::from(i32::MIN); 3]
    );
    assert_eq!(
        integer_values(&sources[1].cast_to(DType::I32).unwrap()),
        vec![i128::from(i32::MIN), i128::from(i32::MIN), 65_535]
    );
    assert_eq!(
        integer_values(&sources[2].cast_to(DType::I32).unwrap()),
        vec![i128::from(i32::MAX), i128::from(i32::MIN), 65_535]
    );
    for source in sources {
        for (target, _) in INTEGER_TARGETS {
            let mut assigned = ArrayStorage::zeros(target, source.len());
            for index in 0..source.len() {
                assigned
                    .set_f64(index, source.get_f64(index).unwrap())
                    .unwrap();
            }
            assert_eq!(
                integer_values(&source.cast_to(target).unwrap()),
                integer_values(&assigned),
                "{:?} -> {target:?}: assignment and bulk conversion disagree",
                source.dtype()
            );
        }
    }
}

#[test]
fn invalid_status_distinguishes_conversion_overflow_from_narrow_wrapping() {
    for target in [DType::I8, DType::I16, DType::I32, DType::U8, DType::U16] {
        for (value, invalid) in [
            (-2_147_483_649.0, true),
            (-2_147_483_648.9, false),
            (-257.9, false),
            (65_535.9, false),
            (2_147_483_647.9, false),
            (2_147_483_648.0, true),
            (f64::NAN, true),
            (f64::INFINITY, true),
            (f64::NEG_INFINITY, true),
        ] {
            assert_eq!(
                ArrayStorage::F64(vec![value]).has_invalid_float_to_integer_cast(target),
                invalid,
                "{value:?} -> {target:?}"
            );
        }
    }
    let two_to_63 = 9_223_372_036_854_775_808.0_f64;
    let two_to_64 = 18_446_744_073_709_551_616.0_f64;
    for (value, signed_invalid, unsigned_invalid) in [
        ((-two_to_63).next_down(), true, true),
        (-two_to_63, false, false),
        (-1.9, false, false),
        (two_to_63.next_down(), false, false),
        (two_to_63, true, false),
        (two_to_64.next_down(), true, false),
        (two_to_64, true, true),
    ] {
        let source = ArrayStorage::F64(vec![value]);
        assert_eq!(
            source.has_invalid_float_to_integer_cast(DType::I64),
            signed_invalid
        );
        assert_eq!(
            source.has_invalid_float_to_integer_cast(DType::U32),
            signed_invalid
        );
        assert_eq!(
            source.has_invalid_float_to_integer_cast(DType::U64),
            unsigned_invalid
        );
    }
}

#[test]
fn invalid_status_only_inspects_stored_real_floating_values() {
    let finite_half = ArrayStorage::F16(
        (0..=u16::MAX)
            .map(f16::from_bits)
            .filter(|value| value.is_finite())
            .collect(),
    );
    for (target, _) in INTEGER_TARGETS {
        // Every finite float16 fits the signed 32-bit conversion intermediate,
        // even when it wraps after narrowing to an 8-bit destination.
        assert!(!finite_half.has_invalid_float_to_integer_cast(target));
        for source in [
            ArrayStorage::F64(Vec::new()),
            ArrayStorage::I64(vec![i64::MIN, i64::MAX]),
            ArrayStorage::U64(vec![u64::MAX]),
            ArrayStorage::Bool(vec![true]),
            ArrayStorage::String(vec!["nan".into()]),
            ArrayStorage::Complex64(vec![(1.9, f32::from_bits(0x7f80_0001)); 8]),
            ArrayStorage::Complex128(vec![(1.9, f64::INFINITY); 8]),
        ] {
            assert!(!source.has_invalid_float_to_integer_cast(target));
        }
    }
    let invalid = ArrayStorage::F64(vec![f64::NAN, f64::INFINITY]);
    for target in [
        DType::Bool,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
        DType::Str,
        DType::Structured,
    ] {
        assert!(!invalid.has_invalid_float_to_integer_cast(target));
    }
}

fn uint32_sources(values: &[f64]) -> [(ArrayStorage, &'static str); 5] {
    [
        (
            ArrayStorage::F16(values.iter().map(|&value| f16::from_f64(value)).collect()),
            "f2",
        ),
        (
            ArrayStorage::F32(values.iter().map(|&value| value as f32).collect()),
            "f4",
        ),
        (ArrayStorage::F64(values.to_vec()), "f8"),
        (
            ArrayStorage::Complex64(
                values
                    .iter()
                    .map(|&value| (value as f32, f32::NAN))
                    .collect(),
            ),
            "c8",
        ),
        (
            ArrayStorage::Complex128(values.iter().map(|&value| (value, f64::INFINITY)).collect()),
            "c16",
        ),
    ]
}

#[test]
fn uint32_scalar_assignments_use_a_signed_64_bit_intermediate() {
    let two_to_63 = 9_223_372_036_854_775_808.0_f64;
    let values = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        -f64::MAX,
        -2_147_483_649.0,
        4_294_967_297.0,
        -4_294_967_295.0,
        two_to_63.next_down(),
        two_to_63,
        -two_to_63,
        (-two_to_63).next_up(),
    ];
    let expected = [
        0,
        0,
        0,
        0,
        0,
        2_147_483_647,
        1,
        1,
        4_294_966_272,
        0,
        0,
        1024,
    ];
    let mut assigned = ArrayStorage::zeros(DType::U32, values.len());
    let mut cases = Vec::new();
    for (index, (&value, &expected)) in values.iter().zip(&expected).enumerate() {
        assigned.set_f64(index, value).unwrap();
        assert_eq!(assigned.get_i128(index).unwrap(), expected);
        cases.push((ArrayStorage::F64(vec![value]), "f8", DType::U32, "u4"));
    }
    numpy_float_cast_oracle(&cases);
}

#[test]
fn uint32_invalid_casts_preserve_source_specific_vector_tails() {
    // (length, real vector prefix, complex64 prefix, complex128 prefix).
    // These independent NumPy fixtures include the different small-complex
    // thresholds and prove that 8192 does not restart a contiguous cast loop.
    for (len, real_prefix, complex64_prefix, complex128_prefix) in [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (3, 0, 0, 0),
        (4, 4, 0, 0),
        (5, 4, 4, 0),
        (6, 4, 4, 4),
        (7, 4, 4, 4),
        (8, 8, 4, 4),
        (9, 8, 8, 8),
        (12, 12, 8, 8),
        (13, 12, 12, 12),
        (8191, 8188, 8188, 8188),
        (8192, 8192, 8188, 8188),
        (8193, 8192, 8192, 8192),
        (16384, 16384, 16380, 16380),
        (16385, 16384, 16384, 16384),
    ] {
        for (source, _) in uint32_sources(&vec![f64::NAN; len]) {
            let prefix = match source.dtype() {
                DType::F16 => 0,
                DType::F32 | DType::F64 => real_prefix,
                DType::Complex64 => complex64_prefix,
                DType::Complex128 => complex128_prefix,
                _ => unreachable!(),
            };
            let mut expected = vec![1_i128 << 31; prefix];
            expected.resize(len, 0);
            assert_eq!(
                integer_values(&source.cast_to(DType::U32).unwrap()),
                expected,
                "{:?} -> uint32 at length {len}",
                source.dtype(),
            );
        }
    }
}

#[test]
fn uint32_vector_boundaries_and_disabled_dispatch_match_numpy() {
    let pattern = [
        f64::NAN,
        f64::NEG_INFINITY,
        -2_147_483_649.0,
        2_147_483_649.0,
        f64::INFINITY,
        4_294_967_297.0,
        -4_294_967_295.0,
        f64::MAX,
        -f64::MAX,
        -1.9,
        1.9,
        -0.0,
        4_294_967_295.9,
        -2_147_483_648.9,
        9_223_372_036_854_775_808.0_f64.next_down(),
        9_223_372_036_854_775_808.0,
        -9_223_372_036_854_775_808.0,
        f64::from_bits(0xfff8_0000_0000_0001),
        f64::from_bits(0x7ff0_0000_0000_0001),
    ];
    let lengths: Vec<usize> = (0..=33)
        .chain([8191, 8192, 8193, 16383, 16384, 16385])
        .collect();
    let mut cases = Vec::new();
    for len in lengths {
        let values = (0..len)
            .map(|index| pattern[index % pattern.len()])
            .collect::<Vec<_>>();
        for (source, name) in uint32_sources(&values) {
            cases.push((source, name, DType::U32, "u4"));
        }
    }
    numpy_float_cast_oracle(&cases);
    numpy_float_cast_oracle_with_dispatch(&cases, true);
}

fn raw_bytes(storage: &ArrayStorage) -> Vec<u8> {
    let mut out = Vec::new();
    match storage {
        ArrayStorage::I8(values) => values
            .iter()
            .for_each(|value| out.extend(value.to_le_bytes())),
        ArrayStorage::I16(values) => values
            .iter()
            .for_each(|value| out.extend(value.to_le_bytes())),
        ArrayStorage::I32(values) => values
            .iter()
            .for_each(|value| out.extend(value.to_le_bytes())),
        ArrayStorage::I64(values) => values
            .iter()
            .for_each(|value| out.extend(value.to_le_bytes())),
        ArrayStorage::U8(values) => out.extend(values.iter().copied()),
        ArrayStorage::U16(values) => values
            .iter()
            .for_each(|value| out.extend(value.to_le_bytes())),
        ArrayStorage::U32(values) => values
            .iter()
            .for_each(|value| out.extend(value.to_le_bytes())),
        ArrayStorage::U64(values) => values
            .iter()
            .for_each(|value| out.extend(value.to_le_bytes())),
        ArrayStorage::F16(values) => values
            .iter()
            .for_each(|value| out.extend(value.to_bits().to_le_bytes())),
        ArrayStorage::F32(values) => values
            .iter()
            .for_each(|value| out.extend(value.to_bits().to_le_bytes())),
        ArrayStorage::F64(values) => values
            .iter()
            .for_each(|value| out.extend(value.to_bits().to_le_bytes())),
        ArrayStorage::Complex64(values) => values.iter().for_each(|(real, imag)| {
            out.extend(real.to_bits().to_le_bytes());
            out.extend(imag.to_bits().to_le_bytes());
        }),
        ArrayStorage::Complex128(values) => values.iter().for_each(|(real, imag)| {
            out.extend(real.to_bits().to_le_bytes());
            out.extend(imag.to_bits().to_le_bytes());
        }),
        _ => panic!("unsupported oracle fixture {:?}", storage.dtype()),
    }
    out
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(text, "{byte:02x}").unwrap();
    }
    text
}

fn numpy_float_cast_oracle(cases: &[(ArrayStorage, &str, DType, &str)]) {
    numpy_float_cast_oracle_with_dispatch(cases, false);
}

fn numpy_float_cast_oracle_with_dispatch(
    cases: &[(ArrayStorage, &str, DType, &str)],
    disable_optional_dispatch: bool,
) {
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let available = Command::new(&python)
        .args(["-c", "import numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(
            std::env::var("FNP_REQUIRE_REAL_NUMPY_ORACLE").as_deref() != Ok("1"),
            "NumPy oracle is required but unavailable through {python}"
        );
        eprintln!("skipping float-cast NumPy oracle: unavailable through {python}");
        return;
    }
    // The entire request is read before writing output, so large exhaustive
    // float16 fixtures cannot fill both pipes and deadlock the parent process.
    let script = r#"
import sys, warnings
import numpy as np
if len(sys.argv) > 1:
    from numpy._core import _multiarray_umath as umath
    assert not any(umath.__cpu_features__[name] for name in umath.__cpu_dispatch__)
requests = sys.stdin.read().splitlines()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for request in requests:
        source, target, payload = request.split(':')
        array = np.frombuffer(bytes.fromhex(payload), dtype=np.dtype('<' + source))
        dtype = np.dtype('<' + target)
        with np.errstate(all='ignore'):
            converted = array.astype(dtype)
        try:
            with np.errstate(all='ignore', invalid='raise'):
                array.astype(dtype)
        except FloatingPointError as error:
            assert str(error) == 'invalid value encountered in cast', str(error)
            invalid = True
        else:
            invalid = False
        print(f'{int(invalid)}:{converted.tobytes().hex()}')
"#;
    let mut request = String::new();
    for (source, source_name, _, target_name) in cases {
        writeln!(
            request,
            "{source_name}:{target_name}:{}",
            hex_bytes(&raw_bytes(source))
        )
        .unwrap();
    }
    let mut command = Command::new(&python);
    command.args(["-c", script]);
    if disable_optional_dispatch {
        let features = Command::new(&python)
            .args([
                "-c",
                "from numpy._core import _multiarray_umath as u; print(','.join(u.__cpu_dispatch__))",
            ])
            .output()
            .expect("read NumPy optional dispatch features");
        assert!(features.status.success());
        let features = String::from_utf8(features.stdout).unwrap();
        command
            .arg("verify-disabled-dispatch")
            .env("NPY_DISABLE_CPU_FEATURES", features.trim());
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start NumPy float-cast oracle");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "NumPy oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<_> = actual.lines().collect();
    assert_eq!(lines.len(), cases.len(), "incomplete NumPy oracle response");
    for ((source, _, target, _), response) in cases.iter().zip(lines) {
        let (invalid, expected) = response.split_once(':').expect("status and cast bytes");
        assert!(matches!(invalid, "0" | "1"), "unexpected status: {invalid}");
        assert_eq!(
            source.has_invalid_float_to_integer_cast(*target),
            invalid == "1",
            "{:?} -> {target:?}, length {}: invalid-cast status disagrees with NumPy",
            source.dtype(),
            source.len(),
        );
        let converted = source.cast_to(*target).unwrap();
        let bytes = hex_bytes(&raw_bytes(&converted));
        if bytes != expected {
            let width = raw_bytes(&ArrayStorage::zeros(*target, 1)).len() * 2;
            let mismatch = bytes
                .as_bytes()
                .chunks(width)
                .zip(expected.as_bytes().chunks(width))
                .position(|(actual, oracle)| actual != oracle)
                .unwrap();
            panic!(
                "{:?} -> {target:?}, element {mismatch} ({:?}): got {}, NumPy {}",
                source.dtype(),
                source.get_f64(mismatch).unwrap(),
                &bytes[mismatch * width..(mismatch + 1) * width],
                &expected[mismatch * width..(mismatch + 1) * width]
            );
        }
    }
}

#[test]
fn invalid_status_matches_numpy_at_boundaries_and_each_uint32_vector_lane() {
    let mut values = vec![
        0.0,
        -0.0,
        1.9,
        -1.9,
        257.9,
        -257.9,
        65_504.0,
        65_519.0,
        65_520.0,
        65_535.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        -f64::MAX,
    ];
    for exponent in [31, 32, 63, 64, 127] {
        for sign in [-1.0, 1.0] {
            let value = sign * 2_f64.powi(exponent);
            values.extend([
                value.next_down(),
                value,
                value.next_up(),
                value - 1.0,
                value + 1.0,
                value - 0.5,
                value + 0.5,
            ]);
        }
    }
    let mut cases = Vec::new();
    // Singleton requests prevent one NaN or overflow from hiding an incorrect
    // status for a different value in a large array.
    for value in values {
        for (source, source_name) in uint32_sources(&[value]) {
            for (target, target_name) in INTEGER_TARGETS {
                cases.push((source.clone(), source_name, target, target_name));
            }
        }
    }
    // These finite values are valid in a uint32 scalar tail's signed 64-bit
    // intermediate but invalid in the vector body's biased signed 32-bit one.
    // Place exactly one such value at each position across every small boundary.
    for len in 0..=10 {
        for (source, source_name) in uint32_sources(&vec![0.0; len]) {
            cases.push((source, source_name, DType::U32, "u4"));
        }
        for index in 0..len {
            for value in [-4_294_967_296.0, 4_294_967_296.0] {
                let mut values = vec![0.0; len];
                values[index] = value;
                for (source, source_name) in uint32_sources(&values) {
                    cases.push((source, source_name, DType::U32, "u4"));
                }
            }
        }
    }
    numpy_float_cast_oracle(&cases);
    numpy_float_cast_oracle_with_dispatch(&cases, true);
}

#[test]
fn every_float16_bit_pattern_matches_numpy_integer_casts() {
    let source = ArrayStorage::F16((0..=u16::MAX).map(f16::from_bits).collect());
    let cases: Vec<_> = INTEGER_TARGETS
        .into_iter()
        .map(|(target, name)| (source.clone(), "f2", target, name))
        .collect();
    numpy_float_cast_oracle(&cases);
}

#[test]
fn float_and_complex_integer_boundaries_match_numpy() {
    let mut values = vec![
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.9,
        -1.9,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::from_bits(0x7ff0_0000_0000_0001),
        f64::from_bits(0xfff8_0000_0000_0001),
    ];
    for exponent in [7, 8, 15, 16, 31, 32, 63, 64, 127] {
        for sign in [-1.0, 1.0] {
            let boundary = sign * 2_f64.powi(exponent);
            values.extend([
                boundary.next_down(),
                boundary,
                boundary.next_up(),
                boundary - 1.0,
                boundary + 1.0,
                boundary - 0.5,
                boundary + 0.5,
            ]);
            let boundary32 = boundary as f32;
            values.extend([
                f64::from(boundary32.next_down()),
                f64::from(boundary32),
                f64::from(boundary32.next_up()),
            ]);
        }
    }
    // Deterministic raw bit coverage exercises finite overflow, subnormals,
    // signed NaNs, and ordinary values independently of decimal parsing.
    let mut bits = 0xa076_1d64_78bd_642f_u64;
    for _ in 0..4096 {
        bits = bits.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
        values.push(f64::from_bits(bits));
        values.push(f64::from(f32::from_bits(bits as u32)));
    }
    let sources = [
        (
            ArrayStorage::F32(values.iter().map(|&value| value as f32).collect()),
            "f4",
        ),
        (ArrayStorage::F64(values.clone()), "f8"),
        (
            ArrayStorage::Complex64(
                values
                    .iter()
                    .map(|&value| (value as f32, f32::NAN))
                    .collect(),
            ),
            "c8",
        ),
        (
            ArrayStorage::Complex128(values.iter().map(|&value| (value, f64::INFINITY)).collect()),
            "c16",
        ),
    ];
    let mut cases = Vec::new();
    for (source, name) in sources {
        for (target, target_name) in INTEGER_TARGETS {
            cases.push((source.clone(), name, target, target_name));
        }
    }
    // Retain explicit valid-range fixtures in addition to the invalid vector
    // prefix/tail cases above, including the largest representable uint32.
    for (source, name) in [
        (
            ArrayStorage::F16(vec![f16::ZERO, f16::from_f32(1.9), f16::MAX]),
            "f2",
        ),
        (
            ArrayStorage::F32(vec![0.0, 1.9, 65_535.9, (4_294_967_296.0_f32).next_down()]),
            "f4",
        ),
        (
            ArrayStorage::F64(vec![0.0, 1.9, 65_535.9, 4_294_967_295.9]),
            "f8",
        ),
    ] {
        cases.push((source, name, DType::U32, "u4"));
    }
    numpy_float_cast_oracle(&cases);
}
