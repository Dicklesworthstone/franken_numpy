//! Regression tests from fuzz findings.
//!
//! Each test case captures a crash input found by fuzzing and ensures it
//! doesn't panic (though it may return an error).

#[test]
fn regression_header_null_descr() {
    // Found by fuzz_header: input with null byte in descr and large header_len
    // Input: shape=[], fortran_order=false, descr="\0Ճ", header_len=65024
    // This triggered a panic in validate_header_schema
    let shape: &[usize] = &[];
    let fortran_order = false;
    let descr = "\0\u{0543}"; // null byte + Armenian letter
    let header_len = 65024usize;

    // Should return an error, not panic
    let result = fnp_io::validate_header_schema(shape, fortran_order, descr, header_len);
    assert!(result.is_err(), "Expected error for invalid header params");
}

#[test]
fn regression_header_empty_bytes() {
    // Ensure empty header_bytes don't panic any validation functions
    let empty: &[u8] = &[];

    // These should all return errors, not panic
    let _ = fnp_io::validate_magic_version(empty);
    let _ = fnp_io::classify_load_dispatch(empty, false);
    let _ = fnp_io::classify_load_dispatch(empty, true);
}
