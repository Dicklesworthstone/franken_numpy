//! Codebase hygiene tests that ensure no stub/mock/placeholder code exists.
//!
//! These tests verify the codebase stays clean of TODO markers, unimplemented
//! macros, and other indicators of incomplete code.

use std::process::Command;

/// Run ripgrep with the given `pattern` and extra glob filters, return the
/// total per-file match count summed across the workspace's `crates/` tree.
///
/// All callers share the same baseline flags: `-c` (count mode), `--type rust`,
/// and the standard `!target/` / `!.rch-target/` excludes. Callers can pass
/// additional `-g` glob patterns via `extra_globs` (e.g. positive includes
/// like `"**/src/*.rs"`, or further excludes like `"!fuzz/"`).
fn run_ripgrep(pattern: &str, extra_globs: &[&str]) -> usize {
    let crates_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir");

    let mut args: Vec<&str> = vec![
        "-c",
        pattern,
        "--type",
        "rust",
        "-g",
        "!target/",
        "-g",
        "!.rch-target/",
    ];
    for glob in extra_globs {
        args.push("-g");
        args.push(glob);
    }

    let output = Command::new("rg")
        .args(&args)
        .arg(crates_dir)
        .output()
        .expect("rg should be available");

    if output.status.code() == Some(1) {
        return 0;
    }
    assert!(
        output.status.success(),
        "rg failed while scanning hygiene pattern {pattern:?}: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.split(':').next_back()?.parse::<usize>().ok())
        .sum()
}

/// Default ripgrep helper used by the stub-marker / TODO / unimplemented tests.
/// Excludes the `fuzz/` tree and the two GATE FILES that necessarily quote the
/// very markers they enforce, which would otherwise self-match.
///
/// `ledger_hygiene.rs` earned its exclusion the hard way: its worker/harness
/// provenance gates define a `PLACEHOLDERS` list (`unavailable`, `unknown`,
/// `n/a`, ...) so a row cannot satisfy them with a non-answer, and the doc
/// comments explaining that rule used the word "placeholder" in prose. That
/// turned `no_stub_comments` red on main for a false positive — the file
/// contains no stub code, it contains a definition OF placeholders. Excluding a
/// second gate file for the same self-match reason is not a weakening of the
/// stub rule; renaming the concept to dodge a grep would have been the worse
/// change.
fn grep_pattern(pattern: &str) -> usize {
    run_ripgrep(
        pattern,
        &["!fuzz/", "!codebase_hygiene.rs", "!ledger_hygiene.rs"],
    )
}

#[test]
fn no_unimplemented_macros() {
    let count = grep_pattern("unimplemented!");
    assert_eq!(
        count, 0,
        "found {count} unimplemented! macros — these should be replaced with real implementations"
    );
}

#[test]
fn no_todo_macros() {
    let count = grep_pattern(r"todo!\(");
    assert_eq!(
        count, 0,
        "found {count} todo! macros — these should be completed"
    );
}

#[test]
fn no_stub_comments() {
    let count = grep_pattern(
        r"//[!/]*.*\b([sS][tT][uU][bB]|[pP][lL][aA][cC][eE][hH][oO][lL][dD][eE][rR])\b|//[!/]*.*\b[mM][oO][cC][kK]\b.*\bimpl\b",
    );
    assert_eq!(
        count, 0,
        "found {count} stub/placeholder comments — code should be complete"
    );
}

#[test]
fn no_not_implemented_panics() {
    let count = grep_pattern(r#"panic!\("not implemented"#);
    assert_eq!(
        count, 0,
        "found {count} 'not implemented' panics — implement the functionality"
    );
}

#[test]
fn test_count_sanity_check() {
    // Regression-guard for total #[test] count across the workspace.
    // README + FEATURE_PARITY cite ~6,392 tests; the >6,000 floor leaves
    // a ~390-test buffer for legitimate refactor consolidation while
    // still catching catastrophic test-deletion. When the cited count
    // grows substantially (e.g. past 7,000), raise this floor in lockstep.
    let test_count = grep_pattern(r"#\[test\]");
    assert!(
        test_count > 6000,
        "expected >6000 test functions, found {test_count} — test coverage may have regressed"
    );
}

#[test]
fn no_fixme_hack_markers() {
    let count = grep_pattern(r"//.*\b(FIXME|HACK|XXX)\b");
    assert_eq!(
        count, 0,
        "found {count} FIXME/HACK/XXX comment markers — address or convert to tracked issues"
    );
}

#[test]
fn no_dbg_macros_in_library_code() {
    let count = run_ripgrep(r"dbg!\(", &["**/src/*.rs"]);
    assert_eq!(
        count, 0,
        "found {count} dbg! macros in library code — remove before release"
    );
}

#[test]
fn no_allow_unused_in_library_code() {
    let count = run_ripgrep(
        r"#\[allow\(dead_code\)\]|#\[allow\(unused",
        &["**/src/lib.rs"],
    );
    // Current inventory is 63 across fnp-conformance and fnp-python; includes
    // PyUFunc native path functions preserved for future optimization.
    assert!(
        count <= 65,
        "found {count} allow(dead_code/unused) in lib.rs files — clean up unused code"
    );
}

#[test]
fn no_unsafe_code_blocks_or_items() {
    // The 9 numeric-core crates declare `#![forbid(unsafe_code)]` and must stay
    // hand-written-unsafe-free. `fnp-python` is the sanctioned opt-out (per
    // AGENTS.md): as the PyO3 boundary it uses hand-written `unsafe` to
    // reinterpret borrowed PyBuffer bytes as typed slices without copying — the
    // zero-copy fast paths that back the performance work. It is excluded here
    // so this test enforces the invariant on the crates that actually hold it.
    let count = run_ripgrep(
        r"\bunsafe\s*(\{|fn|impl|trait|extern\b)",
        &["!fuzz/", "!codebase_hygiene.rs", "!fnp-python/"],
    );
    assert_eq!(
        count, 0,
        "found {count} unsafe blocks/items outside fnp-python — keep the forbid(unsafe_code) crates on the safe-Rust path"
    );
}

#[test]
fn no_allow_unsafe_code_lint_overrides() {
    let count = run_ripgrep(
        r"#\[allow\(unsafe_code\)\]|#!\[allow\(unsafe_code\)\]",
        &["!fuzz/", "!codebase_hygiene.rs"],
    );
    assert_eq!(
        count, 0,
        "found {count} allow(unsafe_code) overrides — do not relax the workspace unsafe-code invariant"
    );
}

#[test]
fn no_arch_intrinsics_in_crate_sources() {
    // `is_x86_feature_detected!` is a SAFE macro and the standard way to gate a
    // SIMD/ISA fast path at runtime (e.g. the AVX-512-gated linalg and libm
    // paths); it is permitted. Any OTHER `core/std::arch` reference — real
    // intrinsics like `_mm256_*` that require `unsafe` — is still forbidden, so
    // subtract the feature-detection calls from the total.
    let total = run_ripgrep(
        r"\buse\s+(core|std)::arch\b|\b(core|std)::arch::",
        &["!fuzz/", "!codebase_hygiene.rs"],
    );
    let feature_detected = run_ripgrep(
        r"\b(core|std)::arch::is_x86_feature_detected\b",
        &["!fuzz/", "!codebase_hygiene.rs"],
    );
    let count = total.saturating_sub(feature_detected);
    assert_eq!(
        count, 0,
        "found {count} core/std::arch intrinsic references (excluding is_x86_feature_detected!) — use portable safe SIMD or scalar fallbacks instead"
    );
}

/// A ufunc-named `#[pyfunction]` must never be REGISTERED under a name that
/// `fnp_python` also binds to a `PyUFunc` object.
///
/// Seventeen ops in `fnp-python` carry both: an `#[allow(dead_code)]`
/// `#[pyfunction]` (e.g. `fn divide`) and `m.add("divide", Py::new(py, PyUFunc {
/// .. })?)`. The module attribute wins, so the pyfunction body is unreachable from
/// Python. That is fine as long as it stays unregistered — the danger is the
/// reverse: those bodies LOOK like optimisation targets
/// (`native_binary_divide_or_passthrough` copies both operands and scans the whole
/// divisor, roughly 3x NumPy's traffic) while moving no measured number. It has
/// already cost two reading cycles, the second while hunting the live 1.16-1.22x
/// `fnp.divide` deficit against `numpy.divide`. Beads `deadlock-audit-uxkqi` and
/// `deadlock-audit-su0i6`.
///
/// This test fails if anyone wires one of them up for real, which would make the
/// same name resolve two ways depending on registration order and silently change
/// which code the benchmarks measure.
#[test]
fn no_ufunc_shadowed_pyfunction_is_registered() {
    const UFUNC_BOUND_NAMES: [&str; 20] = [
        "add",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "divide",
        "equal",
        "floor_divide",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "maximum",
        "minimum",
        "multiply",
        "not_equal",
        "power",
        "subtract",
    ];

    let mut registered: Vec<&str> = Vec::new();
    for name in UFUNC_BOUND_NAMES {
        // `wrap_pyfunction!(divide, m)` is the only way a pyfunction reaches the
        // module, so its absence is what makes the shadowed body unreachable.
        let pattern = format!(r"wrap_pyfunction!\(\s*{name}\s*,");
        if run_ripgrep(&pattern, &["!fuzz/", "!codebase_hygiene.rs"]) > 0 {
            registered.push(name);
        }
    }

    assert!(
        registered.is_empty(),
        "these ops are bound as PyUFunc objects AND registered as pyfunctions, so the \
         attribute now resolves by registration order and the benchmarks may be measuring \
         a different body than the one you read: {registered:?}. Either drop the \
         wrap_pyfunction! call, or remove the PyUFunc binding — do not ship both."
    );
}
