use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use fnp_conformance::{HarnessConfig, run_all_core_suites, run_smoke};
use fnp_iter::{Nditer, NditerOptions, NditerOrder, NditerStep, nditer_python_with_interpreter};
use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct OracleNditerState {
    iterindex: usize,
    multi_index: Vec<usize>,
}

fn repo_numpy_venv_python() -> PathBuf {
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

fn bootstrap_repo_numpy_venv(python_path: &Path, bootstrap_python: &str) -> Result<String, String> {
    let venv_dir = python_path
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| format!("invalid oracle venv path {}", python_path.display()))?;

    // Reuse the venv only if it is fully bootstrapped (python binary exists).
    // A bare directory with no python3 inside means a prior bootstrap failed
    // partway — in that case we pass --clear to force recreation. Skipping
    // --clear would leave `uv venv` to fail with "venv already exists" and
    // `uv pip install` to fail with "not a valid venv".
    let venv_already_exists = python_path.is_file();
    let venv_dir_partial = venv_dir.exists() && !venv_already_exists;

    if let Ok(uv_check) = Command::new("uv").arg("--version").output()
        && uv_check.status.success()
    {
        if !venv_already_exists {
            let mut cmd = Command::new("uv");
            cmd.arg("venv").arg("--python").arg("3.14").arg(venv_dir);
            if venv_dir_partial {
                cmd.arg("--clear");
            }
            let create = cmd
                .output()
                .map_err(|err| format!("failed to bootstrap oracle venv via uv venv: {err}"))?;
            if !create.status.success() {
                let stderr = String::from_utf8_lossy(&create.stderr);
                let stdout = String::from_utf8_lossy(&create.stdout);
                return Err(format!(
                    "failed to bootstrap oracle venv via uv venv (stdout={} stderr={})",
                    stdout.trim(),
                    stderr.trim()
                ));
            }
        }

        let install = Command::new("uv")
            .arg("pip")
            .arg("install")
            .arg("--python")
            .arg(python_path)
            .arg("numpy")
            .output()
            .map_err(|err| format!("failed to install numpy into oracle venv via uv pip: {err}"))?;
        if !install.status.success() {
            let stderr = String::from_utf8_lossy(&install.stderr);
            let stdout = String::from_utf8_lossy(&install.stdout);
            return Err(format!(
                "failed to install numpy into oracle venv via uv pip (stdout={} stderr={})",
                stdout.trim(),
                stderr.trim()
            ));
        }

        return Ok(python_path.display().to_string());
    }

    if !venv_already_exists {
        let create = Command::new(bootstrap_python)
            .arg("-m")
            .arg("venv")
            .arg(venv_dir)
            .output()
            .map_err(|err| {
                format!("failed to bootstrap oracle venv via `{bootstrap_python} -m venv`: {err}")
            })?;
        if !create.status.success() {
            let stderr = String::from_utf8_lossy(&create.stderr);
            let stdout = String::from_utf8_lossy(&create.stdout);
            return Err(format!(
                "failed to bootstrap oracle venv via `{bootstrap_python} -m venv` (stdout={} stderr={})",
                stdout.trim(),
                stderr.trim()
            ));
        }
    }

    let install = Command::new(python_path)
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg("numpy")
        .output()
        .map_err(|err| format!("failed to install numpy into oracle venv via pip: {err}"))?;
    if !install.status.success() {
        let stderr = String::from_utf8_lossy(&install.stderr);
        let stdout = String::from_utf8_lossy(&install.stdout);
        return Err(format!(
            "failed to install numpy into oracle venv via pip (stdout={} stderr={})",
            stdout.trim(),
            stderr.trim()
        ));
    }

    Ok(python_path.display().to_string())
}

fn real_numpy_python() -> &'static str {
    static PYTHON: OnceLock<String> = OnceLock::new();
    PYTHON
        .get_or_init(|| {
            if let Ok(configured) = std::env::var("FNP_ORACLE_PYTHON")
                && python_has_numpy(&configured)
            {
                return configured;
            }

            let repo_python = repo_numpy_venv_python();
            if repo_python.is_file() {
                let candidate = repo_python.display().to_string();
                if python_has_numpy(&candidate) {
                    return candidate;
                }
            }

            let bootstrap_python =
                std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".to_string());
            bootstrap_repo_numpy_venv(&repo_python, &bootstrap_python)
                .expect("bootstrap real numpy oracle")
        })
        .as_str()
}

fn numpy_nditer_states(order: &str, seek_multi_index: Option<&[usize]>) -> Vec<OracleNditerState> {
    let python = real_numpy_python();
    let script = r#"
import json
import numpy as np
import sys

order = sys.argv[1]
seek = json.loads(sys.argv[2])

it = np.nditer(np.arange(6).reshape(2, 3), flags=['multi_index'], order=order)
if seek is not None:
    it.multi_index = tuple(seek)

states = []
while not it.finished:
    states.append({
        "iterindex": int(it.iterindex),
        "multi_index": list(it.multi_index),
    })
    it.iternext()

print(json.dumps(states))
"#;
    let seek_json =
        serde_json::to_string(&seek_multi_index).expect("seek_multi_index should serialize");
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(order)
        .arg(seek_json)
        .output()
        .expect("failed to run numpy oracle");
    assert!(
        output.status.success(),
        "numpy oracle failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    serde_json::from_slice(&output.stdout).expect("numpy oracle should emit valid JSON")
}

#[test]
fn smoke_report_is_stable() {
    let cfg = HarnessConfig::default_paths();
    let report = run_smoke(&cfg);
    assert_eq!(report.suite, "smoke");
    assert!(report.fixture_count >= 1);
    assert!(report.oracle_present);

    let fixture_path = cfg.fixture_root.join("smoke_case.json");
    assert!(Path::new(&fixture_path).exists());
}

#[test]
fn core_conformance_suites_pass() {
    let cfg = HarnessConfig::default_paths();
    let suites = run_all_core_suites(&cfg).expect("core suites should execute");

    for suite in suites {
        assert!(
            suite.all_passed(),
            "suite {} failed with {:?}",
            suite.suite,
            suite.failures
        );
    }
}

#[test]
fn nditer_wrapper_matches_numpy_iterindex_and_multi_index_progression() {
    for (order_name, order) in [("C", NditerOrder::C), ("F", NditerOrder::F)] {
        let oracle_states = numpy_nditer_states(order_name, None);
        let mut iter = Nditer::new(
            vec![2, 3],
            8,
            NditerOptions {
                order,
                external_loop: false,
            },
        )
        .expect("nditer");

        let mut rust_states = Vec::new();
        while !iter.finished() {
            rust_states.push(OracleNditerState {
                iterindex: iter.iterindex().expect("iterindex"),
                multi_index: iter.multi_index().expect("multi_index"),
            });
            iter.iternext();
        }

        assert_eq!(rust_states, oracle_states, "order={order_name}");
    }
}

#[test]
fn nditer_wrapper_seek_matches_numpy_multi_index_assignment() {
    let oracle_states = numpy_nditer_states("C", Some(&[0, 2]));
    let mut iter = Nditer::new(vec![2, 3], 8, NditerOptions::default()).expect("nditer");
    iter.set_multi_index(&[0, 2])
        .expect("multi-index seek should succeed");

    let mut rust_states = Vec::new();
    while !iter.finished() {
        rust_states.push(OracleNditerState {
            iterindex: iter.iterindex().expect("iterindex"),
            multi_index: iter.multi_index().expect("multi_index"),
        });
        iter.iternext();
    }

    assert_eq!(rust_states, oracle_states);
}

#[test]
fn nditer_python_bridge_matches_rust_external_loop_c_order_chunks() {
    let python = real_numpy_python();
    let options = NditerOptions {
        order: NditerOrder::C,
        external_loop: true,
    };
    let bridge =
        nditer_python_with_interpreter(vec![2, 3, 4], 8, options, python).expect("python bridge");
    let rust_steps: Vec<NditerStep> = Nditer::new(vec![2, 3, 4], 8, options)
        .expect("rust nditer")
        .collect();

    assert_eq!(bridge.steps().expect("python bridge steps"), rust_steps);
}

#[test]
fn nditer_python_bridge_matches_rust_f_order_seek_from_iterindex() {
    let python = real_numpy_python();
    let options = NditerOptions {
        order: NditerOrder::F,
        external_loop: true,
    };
    let bridge =
        nditer_python_with_interpreter(vec![2, 3, 4], 8, options, python).expect("python bridge");
    let mut rust_iter = Nditer::new(vec![2, 3, 4], 8, options).expect("rust nditer");
    rust_iter
        .set_iterindex(2)
        .expect("aligned external_loop seek should succeed");
    let rust_steps: Vec<NditerStep> = rust_iter.collect();

    assert_eq!(
        bridge
            .steps_from_iterindex(2)
            .expect("python bridge seek-by-iterindex"),
        rust_steps
    );
}
