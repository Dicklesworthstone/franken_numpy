#![no_main]

use fnp_linalg::{MatrixNormOrder, QrMode, VectorNormOrder};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 256 {
        return;
    }

    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };

    // Fuzz QrMode parser
    let _ = QrMode::from_mode_token(text);
    let _ = QrMode::from_mode_token(text.trim());

    // Fuzz VectorNormOrder parser (handles numeric parsing)
    let _ = VectorNormOrder::from_token(text);
    let _ = VectorNormOrder::from_token(text.trim());

    // Fuzz MatrixNormOrder parser
    let _ = MatrixNormOrder::from_token(text);
    let _ = MatrixNormOrder::from_token(text.trim());

    // Try various case variants
    let upper = text.to_uppercase();
    let lower = text.to_lowercase();
    let _ = QrMode::from_mode_token(&upper);
    let _ = QrMode::from_mode_token(&lower);
    let _ = VectorNormOrder::from_token(&upper);
    let _ = VectorNormOrder::from_token(&lower);
    let _ = MatrixNormOrder::from_token(&upper);
    let _ = MatrixNormOrder::from_token(&lower);

    // Round-trip invariant: if parsed successfully, the canonical form should re-parse
    if let Ok(qr) = QrMode::from_mode_token(text) {
        let canonical = match qr {
            QrMode::Reduced => "reduced",
            QrMode::Complete => "complete",
            QrMode::R => "r",
            QrMode::Raw => "raw",
        };
        assert!(
            QrMode::from_mode_token(canonical).is_ok(),
            "QrMode canonical form failed to re-parse"
        );
    }

    if let Ok(norm) = MatrixNormOrder::from_token(text) {
        let canonical = match norm {
            MatrixNormOrder::Fro => "fro",
            MatrixNormOrder::One => "1",
            MatrixNormOrder::NegOne => "-1",
            MatrixNormOrder::Inf => "inf",
            MatrixNormOrder::NegInf => "-inf",
            MatrixNormOrder::Two => "2",
            MatrixNormOrder::NegTwo => "-2",
            MatrixNormOrder::Nuclear => "nuc",
        };
        assert!(
            MatrixNormOrder::from_token(canonical).is_ok(),
            "MatrixNormOrder canonical form failed to re-parse"
        );
    }
});
