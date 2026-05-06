//! Codebase hygiene tests that ensure no stub/mock/placeholder code exists.
//!
//! These tests verify the codebase stays clean of TODO markers, unimplemented
//! macros, and other indicators of incomplete code.

use std::process::Command;

fn grep_pattern(pattern: &str) -> usize {
    let crates_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir");

    let output = Command::new("rg")
        .args([
            "-c",
            pattern,
            "--type",
            "rust",
            "-g",
            "!target/",
            "-g",
            "!.rch-target/",
            "-g",
            "!fuzz/",
            "-g",
            "!codebase_hygiene.rs",
        ])
        .arg(crates_dir)
        .output()
        .expect("rg should be available");

    if !output.status.success() {
        return 0;
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.split(':').last()?.parse::<usize>().ok())
        .sum()
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
    let count = grep_pattern(r"//.*STUB|//.*PLACEHOLDER|//.*MOCK.*impl");
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
    let test_count = grep_pattern(r"#\[test\]");
    assert!(
        test_count > 2000,
        "expected >2000 test functions, found {test_count} — test coverage may have regressed"
    );
}
