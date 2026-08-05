//! Conformance tests for `numpy.nonzero` on NON-ndarray public surfaces.
//!
//! The existing `nonzero`/`flatnonzero` coverage feeds every input through
//! `np.array(...)` first, which means the container-coercion path the public API
//! actually exposes — Python lists, tuples, nested sequences, scalars, object
//! arrays, and ragged input — was never differentially compared against NumPy
//! (bead `deadlock-audit-63cen`).
//!
//! `nonzero` returns a TUPLE of index arrays rather than one array, so the
//! outcome capture below records the tuple length and every component's dtype,
//! shape, and values. Two of the cases are negative on purpose: a 0-d input and
//! a ragged list must both RAISE, with NumPy's exact exception type and message.
//! An implementation that quietly coerced either one would pass a values-only
//! check and fail here.

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

/// Capture the full observable outcome of one `nonzero` call.
///
/// A `\` line-continuation eats the SOURCE indentation, so the Python indent is
/// injected through `{I4}`/`{I8}`/`{I12}` placeholders rather than written as
/// literal spaces; literal spaces would emit a flat script and the oracle would
/// raise `IndentationError` (the harness bug already recorded in
/// `conformance_flatnonzero`).
fn nonzero_outcome_body(function_expr: &str, input_expr: &str) -> String {
    format!(
        "def outcome(fn):\n\
         {I4}try:\n\
         {I8}value = fn({input_expr})\n\
         {I8}print('ok')\n\
         {I8}print(type(value).__name__)\n\
         {I8}print(len(value))\n\
         {I8}for part in value:\n\
         {I12}arr = np.asarray(part)\n\
         {I12}print(str(arr.dtype))\n\
         {I12}print(tuple(arr.shape))\n\
         {I12}print(arr.tolist())\n\
         {I4}except Exception as exc:\n\
         {I8}print('err')\n\
         {I8}print(type(exc).__name__)\n\
         {I8}print(str(exc))\n\
         outcome({function_expr})",
        I4 = "    ",
        I8 = "        ",
        I12 = "            ",
    )
}

fn numpy_outcome_script(input_expr: &str) -> String {
    format!(
        "import numpy as np\n{}",
        nonzero_outcome_body("np.nonzero", input_expr)
    )
}

fn fnp_outcome_script(input_expr: &str) -> String {
    fnp_script(nonzero_outcome_body("fnp.nonzero", input_expr))
}

/// Every non-ndarray surface the bead enumerates, compared outcome-for-outcome.
#[test]
fn nonzero_python_container_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        ("list fallback", "[0, 2, 0, 3]"),
        ("tuple fallback", "(0, 0, 5, 0)"),
        ("bool list", "[False, True, False, True]"),
        ("all-zero list", "[0, 0, 0]"),
        ("empty list", "[]"),
        ("nested list coordinates", "[[0, 1], [2, 0]]"),
        ("nested tuple coordinates", "((0, 1), (2, 0))"),
        ("nested bool list", "[[False, True], [True, False]]"),
        ("three-dimensional nested list", "[[[0, 1]], [[2, 0]]]"),
        ("mixed int float list", "[0, 1.5, 0.0, -2]"),
        ("signed zero and nan list", "[-0.0, float('nan'), 2.5, 0.0]"),
        (
            "object truthiness",
            "np.array(['', 'x', '0'], dtype=object)",
        ),
        // Negative cases: both MUST raise, with NumPy's exact type and message.
        ("scalar nonzero is an error", "7"),
        ("scalar zero is an error", "0"),
        ("zero-d ndarray is an error", "np.array(7)"),
        ("ragged list error", "[[1], [0, 2]]"),
    ];

    for (label, input_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script(input_expr))?;
        let rust_result = numpy_oracle(&fnp_outcome_script(input_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "nonzero Python-container surface mismatch for {label} (input {input_expr})"
        );
    }

    Ok(())
}

/// Guard the guard: the two 0-d cases and the ragged case must actually be
/// raising in the oracle. If a future NumPy started returning a value for them
/// the case above would still pass while silently testing nothing negative, so
/// the negative expectation is asserted directly rather than inferred.
#[test]
fn nonzero_negative_cases_really_raise_in_numpy() -> Result<(), String> {
    for (input_expr, expected_exception) in [
        ("7", "ValueError"),
        ("0", "ValueError"),
        ("np.array(7)", "ValueError"),
        ("[[1], [0, 2]]", "ValueError"),
    ] {
        let outcome = numpy_oracle(&numpy_outcome_script(input_expr))?;
        let mut lines = outcome.lines();
        assert_eq!(
            lines.next(),
            Some("err"),
            "numpy.nonzero({input_expr}) was expected to raise"
        );
        assert_eq!(
            lines.next(),
            Some(expected_exception),
            "numpy.nonzero({input_expr}) raised an unexpected exception type"
        );
    }

    Ok(())
}
