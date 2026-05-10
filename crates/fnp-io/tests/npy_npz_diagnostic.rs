#![forbid(unsafe_code)]

use fnp_io::{
    IOSupportedDType, MAX_ARCHIVE_MEMBERS, MAX_DISPATCH_RETRIES, MAX_HEADER_BYTES,
    MAX_MEMMAP_VALIDATION_RETRIES, MemmapMode, NpyHeader, classify_load_dispatch, load_auto,
    read_npy_bytes, read_npz_bytes, save, savez, savez_compressed, validate_header_schema,
    validate_io_policy_metadata, validate_magic_version, validate_memmap_contract,
    validate_npz_archive_budget, validate_read_payload, validate_write_contract, write_npy_bytes,
};

#[test]
fn npy_npz_diagnostic_reason_codes_cover_malformed_boundaries() {
    let cases = [
        DiagnosticCase {
            id: "npy_magic_short_payload",
            reason_code: "io_magic_invalid",
            run: || validate_magic_version(&[0u8; 4]).map(|_| ()),
        },
        DiagnosticCase {
            id: "npy_magic_unknown_tuple",
            reason_code: "io_magic_invalid",
            run: || {
                let payload = [0x93, b'N', b'U', b'M', b'P', b'Y', 9, 9];
                validate_magic_version(&payload).map(|_| ())
            },
        },
        DiagnosticCase {
            id: "npy_header_oversized_budget",
            reason_code: "io_header_schema_invalid",
            run: || validate_header_schema(&[2, 3], false, "<f8", MAX_HEADER_BYTES + 1).map(|_| ()),
        },
        DiagnosticCase {
            id: "npy_header_shape_overflow",
            reason_code: "io_header_schema_invalid",
            run: || validate_header_schema(&[usize::MAX, 2], false, "<f8", 128).map(|_| ()),
        },
        DiagnosticCase {
            id: "npy_dtype_descriptor_invalid",
            reason_code: "io_dtype_descriptor_invalid",
            run: || validate_header_schema(&[1], false, ">i3", 128).map(|_| ()),
        },
        DiagnosticCase {
            id: "npy_write_count_mismatch",
            reason_code: "io_write_contract_violation",
            run: || validate_write_contract(&[3, 3], 8, IOSupportedDType::F64).map(|_| ()),
        },
        DiagnosticCase {
            id: "npy_read_payload_short",
            reason_code: "io_read_payload_incomplete",
            run: || validate_read_payload(&[2, 3], 5 * 8, IOSupportedDType::F64).map(|_| ()),
        },
        DiagnosticCase {
            id: "npy_read_payload_long",
            reason_code: "io_read_payload_incomplete",
            run: || validate_read_payload(&[2, 3], 7 * 8, IOSupportedDType::F64).map(|_| ()),
        },
        DiagnosticCase {
            id: "npy_write_item_size_misalignment",
            reason_code: "io_write_contract_violation",
            run: || {
                let header = NpyHeader {
                    shape: vec![1],
                    fortran_order: false,
                    descr: IOSupportedDType::F64,
                };
                write_npy_bytes(&header, &[0u8; 7], false).map(|_| ())
            },
        },
        DiagnosticCase {
            id: "npy_reader_truncated_payload",
            reason_code: "io_read_payload_incomplete",
            run: || {
                let header = NpyHeader {
                    shape: vec![2, 2],
                    fortran_order: false,
                    descr: IOSupportedDType::F64,
                };
                let payload = vec![0u8; 4 * 8];
                let mut encoded =
                    write_npy_bytes(&header, &payload, false).expect("valid fixture encoding");
                encoded.pop();
                read_npy_bytes(&encoded, false).map(|_| ())
            },
        },
        DiagnosticCase {
            id: "npy_reader_declared_header_truncated",
            reason_code: "io_header_schema_invalid",
            run: || {
                let header = NpyHeader {
                    shape: vec![2, 2],
                    fortran_order: false,
                    descr: IOSupportedDType::F64,
                };
                let payload = vec![0u8; 4 * 8];
                let mut encoded =
                    write_npy_bytes(&header, &payload, false).expect("valid fixture encoding");
                for (slot, value) in encoded.iter_mut().skip(8).take(2).zip([0xFF, 0x7F]) {
                    *slot = value;
                }
                encoded.truncate(64);
                read_npy_bytes(&encoded, false).map(|_| ())
            },
        },
        DiagnosticCase {
            id: "npy_object_read_pickle_disallowed",
            reason_code: "io_pickle_policy_violation",
            run: || {
                let header = NpyHeader {
                    shape: vec![1],
                    fortran_order: false,
                    descr: IOSupportedDType::Object,
                };
                let encoded = write_npy_bytes(&header, &[0x80, 0x05, 0x4B, 0x01, 0x2E], true)
                    .expect("valid object fixture encoding");
                read_npy_bytes(&encoded, false).map(|_| ())
            },
        },
        DiagnosticCase {
            id: "npy_object_write_pickle_disallowed",
            reason_code: "io_pickle_policy_violation",
            run: || {
                let header = NpyHeader {
                    shape: vec![1],
                    fortran_order: false,
                    descr: IOSupportedDType::Object,
                };
                write_npy_bytes(&header, &[0x80, 0x05, 0x4B, 0x01, 0x2E], false).map(|_| ())
            },
        },
        DiagnosticCase {
            id: "npy_object_write_missing_pickle_marker",
            reason_code: "io_write_contract_violation",
            run: || {
                let header = NpyHeader {
                    shape: vec![1],
                    fortran_order: false,
                    descr: IOSupportedDType::Object,
                };
                write_npy_bytes(&header, b"not-pickle", true).map(|_| ())
            },
        },
        DiagnosticCase {
            id: "memmap_object_dtype_rejected",
            reason_code: "io_memmap_contract_violation",
            run: || {
                validate_memmap_contract(
                    MemmapMode::ReadOnly,
                    IOSupportedDType::Object,
                    4096,
                    1024,
                    0,
                )
            },
        },
        DiagnosticCase {
            id: "memmap_retry_budget_rejected",
            reason_code: "io_memmap_contract_violation",
            run: || {
                validate_memmap_contract(
                    MemmapMode::ReadOnly,
                    IOSupportedDType::F64,
                    4096,
                    1024,
                    MAX_MEMMAP_VALIDATION_RETRIES + 1,
                )
            },
        },
        DiagnosticCase {
            id: "load_dispatch_pickle_policy_rejected",
            reason_code: "io_load_dispatch_invalid",
            run: || classify_load_dispatch(&[0x80, 0x05, 0x95], false).map(|_| ()),
        },
        DiagnosticCase {
            id: "load_auto_pickle_policy_rejected",
            reason_code: "io_load_dispatch_invalid",
            run: || load_auto(&[0x80, 0x05, 0x95, 0x00], false).map(|_| ()),
        },
        DiagnosticCase {
            id: "npz_archive_member_budget_rejected",
            reason_code: "io_npz_archive_contract_violation",
            run: || validate_npz_archive_budget(MAX_ARCHIVE_MEMBERS + 1, 1024, 0),
        },
        DiagnosticCase {
            id: "npz_archive_uncompressed_budget_rejected",
            reason_code: "io_npz_archive_contract_violation",
            run: || validate_npz_archive_budget(4, usize::MAX, 0),
        },
        DiagnosticCase {
            id: "npz_archive_retry_budget_rejected",
            reason_code: "io_load_dispatch_invalid",
            run: || validate_npz_archive_budget(4, 1024, MAX_DISPATCH_RETRIES + 1),
        },
        DiagnosticCase {
            id: "npz_reader_bad_zip_rejected",
            reason_code: "io_npz_archive_contract_violation",
            run: || read_npz_bytes(b"PK\x03\x04bad", false).map(|_| ()),
        },
        DiagnosticCase {
            id: "npz_save_duplicate_member_rejected",
            reason_code: "io_npz_archive_contract_violation",
            run: || {
                let entries: Vec<(&str, &[usize], &[f64], IOSupportedDType)> = vec![
                    ("arr", &[1], &[1.0], IOSupportedDType::F64),
                    ("arr", &[1], &[2.0], IOSupportedDType::F64),
                ];
                savez(&entries).map(|_| ())
            },
        },
        DiagnosticCase {
            id: "npz_save_compressed_duplicate_member_rejected",
            reason_code: "io_npz_archive_contract_violation",
            run: || {
                let entries: Vec<(&str, &[usize], &[f64], IOSupportedDType)> = vec![
                    ("arr", &[1], &[1.0], IOSupportedDType::F64),
                    ("arr.npy", &[1], &[2.0], IOSupportedDType::F64),
                ];
                savez_compressed(&entries).map(|_| ())
            },
        },
        DiagnosticCase {
            id: "io_policy_unknown_mode_rejected",
            reason_code: "io_policy_unknown_metadata",
            run: || validate_io_policy_metadata("mystery", "known_compatible_low_risk"),
        },
    ];

    assert!(cases.len() >= 20);
    for case in cases {
        let error = (case.run)().expect_err(case.id);
        assert_eq!(error.reason_code(), case.reason_code, "{}", case.id);
    }
}

#[test]
fn npy_npz_diagnostic_success_controls_keep_strict_paths_distinct() {
    let npy = save(&[2], &[1.0, 2.0], IOSupportedDType::F64).expect("valid npy");
    read_npy_bytes(&npy, false).expect("valid npy should decode");

    let npz = savez(&[("arr", &[2], &[1.0, 2.0], IOSupportedDType::F64)]).expect("valid npz");
    read_npz_bytes(&npz, false).expect("valid npz should decode");

    let compressed = savez_compressed(&[("arr", &[2], &[1.0, 2.0], IOSupportedDType::F64)])
        .expect("valid compressed npz");
    read_npz_bytes(&compressed, false).expect("valid compressed npz should decode");
}

struct DiagnosticCase {
    id: &'static str,
    reason_code: &'static str,
    run: fn() -> Result<(), fnp_io::IOError>,
}
