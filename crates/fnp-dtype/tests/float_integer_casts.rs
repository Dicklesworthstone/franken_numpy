//! NumPy's invalid float casts depend on the host conversion instructions.
//! These fixtures cover the Linux x86-64 kernels implemented by `ArrayStorage`.
#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use fnp_dtype::{ArrayStorage, DType};
use half::f16;
use std::fmt::Write as _;
use std::io::Write as _;
use std::process::{Command, Stdio};

const INTEGER_TARGETS: [(DType, &str); 7] = [
    (DType::I8, "i1"),
    (DType::I16, "i2"),
    (DType::I32, "i4"),
    (DType::I64, "i8"),
    (DType::U8, "u1"),
    (DType::U16, "u2"),
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
requests = sys.stdin.read().splitlines()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for request in requests:
        source, target, payload = request.split(':')
        array = np.frombuffer(bytes.fromhex(payload), dtype=np.dtype('<' + source))
        print(array.astype(np.dtype('<' + target)).tobytes().hex())
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
    let mut child = Command::new(&python)
        .args(["-c", script])
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
    for ((source, _, target, _), expected) in cases.iter().zip(lines) {
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
    // uint32 invalid casts are intentionally not asserted: NumPy's SIMD prefix
    // and scalar tail disagree. Include that width for the fully defined range.
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
