use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use fnp_dtype::DType;
use fnp_linalg::{det_2x2, inv_2x2, solve_2x2};
use fnp_random::Generator;
use fnp_ufunc::{BinaryOp, UFuncArray};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct OracleArray {
    shape: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct RawOracleArray {
    shape: Vec<usize>,
    values: Vec<serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NumpyVersion {
    major: u64,
    minor: u64,
    patch: u64,
}

impl NumpyVersion {
    fn parse(text: &str) -> Result<Self, String> {
        let release = text
            .trim()
            .split(|ch: char| !ch.is_ascii_digit() && ch != '.')
            .next()
            .unwrap_or("");
        let mut parts = release.split('.');
        let major = parts
            .next()
            .filter(|part| !part.is_empty())
            .ok_or_else(|| format!("missing NumPy major version in {text:?}"))?
            .parse()
            .map_err(|err| format!("invalid NumPy major version in {text:?}: {err}"))?;
        let minor = parts
            .next()
            .unwrap_or("0")
            .parse()
            .map_err(|err| format!("invalid NumPy minor version in {text:?}: {err}"))?;
        let patch = parts
            .next()
            .unwrap_or("0")
            .parse()
            .map_err(|err| format!("invalid NumPy patch version in {text:?}: {err}"))?;
        Ok(Self {
            major,
            minor,
            patch,
        })
    }

    fn exposes_trapezoid(self) -> bool {
        self.major >= 2
    }
}

fn decode_oracle_scalar(case_id: &str, value: serde_json::Value) -> Result<f64, String> {
    match value {
        serde_json::Value::Number(number) => number
            .as_f64()
            .ok_or_else(|| format!("{case_id}: oracle emitted a non-f64 JSON number")),
        serde_json::Value::String(text) if text == "NaN" => Ok(f64::NAN),
        serde_json::Value::String(text) if text == "Infinity" => Ok(f64::INFINITY),
        serde_json::Value::String(text) if text == "-Infinity" => Ok(f64::NEG_INFINITY),
        other => Err(format!(
            "{case_id}: unexpected oracle scalar encoding {other:?}"
        )),
    }
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
                return Err(format!(
                    "failed to bootstrap oracle venv via uv venv (stdout={} stderr={})",
                    String::from_utf8_lossy(&create.stdout).trim(),
                    String::from_utf8_lossy(&create.stderr).trim()
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
            return Err(format!(
                "failed to install numpy into oracle venv via uv pip (stdout={} stderr={})",
                String::from_utf8_lossy(&install.stdout).trim(),
                String::from_utf8_lossy(&install.stderr).trim()
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
            return Err(format!(
                "failed to bootstrap oracle venv via `{bootstrap_python} -m venv` (stdout={} stderr={})",
                String::from_utf8_lossy(&create.stdout).trim(),
                String::from_utf8_lossy(&create.stderr).trim()
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
        return Err(format!(
            "failed to install numpy into oracle venv via pip (stdout={} stderr={})",
            String::from_utf8_lossy(&install.stdout).trim(),
            String::from_utf8_lossy(&install.stderr).trim()
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
                .expect("bootstrap live NumPy oracle")
        })
        .as_str()
}

fn live_numpy_version() -> NumpyVersion {
    static VERSION: OnceLock<NumpyVersion> = OnceLock::new();
    *VERSION.get_or_init(|| {
        let output = Command::new(real_numpy_python())
            .arg("-c")
            .arg("import numpy as np; print(np.__version__)")
            .output()
            .expect("failed to query live NumPy version");
        assert!(
            output.status.success(),
            "NumPy version query failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let version_text = String::from_utf8(output.stdout).expect("NumPy version should be UTF-8");
        NumpyVersion::parse(version_text.trim()).expect("live NumPy version should parse")
    })
}

fn numpy_case(case_id: &str) -> OracleArray {
    let script = r#"
import json
import sys

import numpy as np

case_id = sys.argv[1]

def emit(value):
    arr = np.asarray(value)
    def scalar(item):
        value = float(item)
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return value
    if np.iscomplexobj(arr):
        values = []
        for item in arr.reshape(-1):
            values.extend([scalar(np.real(item)), scalar(np.imag(item))])
        shape = list(arr.shape) + [2]
    else:
        values = [scalar(item) for item in arr.reshape(-1)]
        shape = list(arr.shape)
    print(json.dumps({"shape": shape, "values": values}))

if case_id == "ufunc_add_broadcast":
    emit(np.add(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.array([10.0, 20.0, 30.0]),
    ))
elif case_id == "ufunc_add_0d_scalar_lhs_broadcast":
    emit(np.add(
        np.array(5.0),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ))
elif case_id == "ufunc_subtract_0d_scalar_lhs_broadcast":
    emit(np.subtract(
        np.array(10.0),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ))
elif case_id == "ufunc_greater_0d_scalar_rhs_broadcast":
    emit(np.greater(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.array(3.0),
    ))
elif case_id == "ufunc_reduce_keepdims_broadcast_add":
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    emit(np.add(values, np.sum(values, axis=1, keepdims=True)))
elif case_id == "ufunc_floor_divide_inf":
    emit(np.floor_divide(np.array([np.inf, -np.inf, 9.0]), np.array([2.0, 2.0, 4.0])))
elif case_id == "ufunc_logical_or_nan":
    emit(np.logical_or(np.array([np.nan, 0.0, 2.0]), np.array([0.0, 0.0, np.nan])))
elif case_id == "ufunc_reduce_sum_axis1":
    emit(np.sum(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), axis=1))
elif case_id == "linalg_solve_2x2":
    emit(np.linalg.solve(np.array([[3.0, 1.0], [1.0, 2.0]]), np.array([9.0, 8.0])))
elif case_id == "linalg_det_2x2":
    emit(np.linalg.det(np.array([[3.0, 1.0], [1.0, 2.0]])))
elif case_id == "linalg_inv_2x2":
    emit(np.linalg.inv(np.array([[3.0, 1.0], [1.0, 2.0]])))
elif case_id == "fft_fft_len4":
    emit(np.fft.fft(np.array([0.0, 1.0, 2.0, 3.0])))
elif case_id == "fft_rfft_len5":
    emit(np.fft.rfft(np.array([0.0, 1.0, 2.0, 3.0]), n=5))
elif case_id == "fft_fftfreq_even":
    emit(np.fft.fftfreq(6, d=0.5))
elif case_id == "random_pcg64dxsm_random":
    rng = np.random.Generator(np.random.PCG64DXSM(12345))
    emit(rng.random(8))
elif case_id == "random_pcg64dxsm_standard_normal":
    rng = np.random.Generator(np.random.PCG64DXSM(12345))
    emit(rng.standard_normal(6))
elif case_id == "random_pcg64dxsm_integers":
    rng = np.random.Generator(np.random.PCG64DXSM(12345))
    emit(rng.integers(-3, 7, size=8))
elif case_id == "trapz_axis0":
    emit(np.trapz(np.array([[1.0, 2.0], [3.0, 4.0], [7.0, 11.0]]), dx=0.5, axis=0))
elif case_id == "trapezoid_axis0":
    emit(np.trapezoid(np.array([[1.0, 2.0], [3.0, 4.0], [7.0, 11.0]]), dx=0.5, axis=0))
else:
    raise SystemExit(f"unknown case_id: {case_id}")
"#;
    let output = Command::new(real_numpy_python())
        .arg("-c")
        .arg(script)
        .arg(case_id)
        .output()
        .expect("failed to run live NumPy oracle");
    assert!(
        output.status.success(),
        "NumPy oracle failed for {case_id}: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let raw: RawOracleArray =
        serde_json::from_slice(&output.stdout).expect("NumPy oracle should emit JSON");
    let values = raw
        .values
        .into_iter()
        .map(|value| decode_oracle_scalar(case_id, value))
        .collect::<Result<Vec<_>, _>>()
        .expect("NumPy oracle scalar encoding should be valid");
    OracleArray {
        shape: raw.shape,
        values,
    }
}

fn assert_oracle_match(case_id: &str, actual_shape: &[usize], actual_values: &[f64], tol: f64) {
    let oracle = numpy_case(case_id);
    assert_eq!(
        actual_shape, oracle.shape,
        "{case_id}: shape diverged from live NumPy"
    );
    assert_eq!(
        actual_values.len(),
        oracle.values.len(),
        "{case_id}: flattened value length diverged from live NumPy"
    );
    for (index, (actual, expected)) in actual_values.iter().zip(oracle.values.iter()).enumerate() {
        let matches = if actual.is_nan() || expected.is_nan() {
            actual.is_nan() && expected.is_nan()
        } else {
            (actual - expected).abs() <= tol.max(tol * expected.abs())
        };
        assert!(
            matches,
            "{case_id}[{index}] diverged: actual={actual:?} expected={expected:?}"
        );
    }
}

fn array(shape: &[usize], values: &[f64]) -> UFuncArray {
    UFuncArray::new(shape.to_vec(), values.to_vec(), DType::F64).expect("valid test array")
}

#[test]
fn ufunc_ops_match_live_numpy_reference() {
    let lhs = array(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = array(&[3], &[10.0, 20.0, 30.0]);
    let add = lhs
        .elementwise_binary(&rhs, BinaryOp::Add)
        .expect("broadcast add");
    assert_oracle_match("ufunc_add_broadcast", add.shape(), add.values(), 1e-12);

    let scalar_add = UFuncArray::scalar(5.0, DType::F64)
        .elementwise_binary(&lhs, BinaryOp::Add)
        .expect("0-d scalar lhs add broadcast");
    assert_oracle_match(
        "ufunc_add_0d_scalar_lhs_broadcast",
        scalar_add.shape(),
        scalar_add.values(),
        1e-12,
    );

    let scalar_subtract = UFuncArray::scalar(10.0, DType::F64)
        .elementwise_binary(&lhs, BinaryOp::Sub)
        .expect("0-d scalar lhs subtract broadcast");
    assert_oracle_match(
        "ufunc_subtract_0d_scalar_lhs_broadcast",
        scalar_subtract.shape(),
        scalar_subtract.values(),
        1e-12,
    );

    let scalar_greater = lhs
        .elementwise_binary(&UFuncArray::scalar(3.0, DType::F64), BinaryOp::Greater)
        .expect("0-d scalar rhs comparison broadcast");
    assert_oracle_match(
        "ufunc_greater_0d_scalar_rhs_broadcast",
        scalar_greater.shape(),
        scalar_greater.values(),
        0.0,
    );

    let row_sums = lhs.reduce_sum(Some(1), true).expect("sum axis=1 keepdims");
    let keepdims_broadcast = lhs
        .elementwise_binary(&row_sums, BinaryOp::Add)
        .expect("keepdims reduction should broadcast back over source rows");
    assert_oracle_match(
        "ufunc_reduce_keepdims_broadcast_add",
        keepdims_broadcast.shape(),
        keepdims_broadcast.values(),
        1e-12,
    );

    let inf_lhs = array(&[3], &[f64::INFINITY, f64::NEG_INFINITY, 9.0]);
    let divisors = array(&[3], &[2.0, 2.0, 4.0]);
    let floor_divide = inf_lhs
        .elementwise_binary(&divisors, BinaryOp::FloorDivide)
        .expect("floor_divide");
    assert_oracle_match(
        "ufunc_floor_divide_inf",
        floor_divide.shape(),
        floor_divide.values(),
        1e-12,
    );

    let logical_lhs = array(&[3], &[f64::NAN, 0.0, 2.0]);
    let logical_rhs = array(&[3], &[0.0, 0.0, f64::NAN]);
    let logical_or = logical_lhs
        .elementwise_binary(&logical_rhs, BinaryOp::LogicalOr)
        .expect("logical_or");
    assert_oracle_match(
        "ufunc_logical_or_nan",
        logical_or.shape(),
        logical_or.values(),
        1e-12,
    );

    let reduced = lhs.reduce_sum(Some(1), false).expect("sum axis=1");
    assert_oracle_match(
        "ufunc_reduce_sum_axis1",
        reduced.shape(),
        reduced.values(),
        1e-12,
    );
}

#[test]
fn linalg_ops_match_live_numpy_reference() {
    let matrix = [[3.0, 1.0], [1.0, 2.0]];
    let solution = solve_2x2(matrix, [9.0, 8.0]).expect("solve_2x2");
    assert_oracle_match("linalg_solve_2x2", &[2], &solution, 1e-12);

    let det = det_2x2(matrix).expect("det_2x2");
    assert_oracle_match("linalg_det_2x2", &[], &[det], 1e-12);

    let inv = inv_2x2(matrix).expect("inv_2x2");
    let inv_flat = [inv[0][0], inv[0][1], inv[1][0], inv[1][1]];
    assert_oracle_match("linalg_inv_2x2", &[2, 2], &inv_flat, 1e-12);
}

#[test]
fn fft_ops_match_live_numpy_reference() {
    let signal = array(&[4], &[0.0, 1.0, 2.0, 3.0]);
    let fft = signal.fft(None).expect("fft");
    assert_oracle_match("fft_fft_len4", fft.shape(), fft.values(), 1e-10);

    let rfft = signal.rfft(Some(5)).expect("rfft n=5");
    assert_oracle_match("fft_rfft_len5", rfft.shape(), rfft.values(), 1e-9);

    let fftfreq = UFuncArray::fftfreq(6, 0.5);
    assert_oracle_match("fft_fftfreq_even", fftfreq.shape(), fftfreq.values(), 1e-12);
}

#[test]
fn random_ops_match_live_numpy_reference() {
    let mut rng = Generator::from_pcg64_dxsm(12345).expect("pcg64dxsm seed");
    let random = rng.random(8);
    assert_oracle_match("random_pcg64dxsm_random", &[8], &random, 1e-12);

    let mut rng = Generator::from_pcg64_dxsm(12345).expect("pcg64dxsm seed");
    let normal = rng.standard_normal(6);
    assert_oracle_match("random_pcg64dxsm_standard_normal", &[6], &normal, 1e-12);

    let mut rng = Generator::from_pcg64_dxsm(12345).expect("pcg64dxsm seed");
    let integers: Vec<f64> = rng
        .integers(-3, 7, 8)
        .expect("integers")
        .into_iter()
        .map(|value| value as f64)
        .collect();
    assert_oracle_match("random_pcg64dxsm_integers", &[8], &integers, 0.0);
}

#[test]
fn version_specific_trapezoid_reference_is_gated_by_live_numpy_version() {
    let y = array(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 7.0, 11.0]);
    let version = live_numpy_version();
    let (case_id, actual) = if version.exposes_trapezoid() {
        (
            "trapezoid_axis0",
            y.trapezoid(0.5, Some(0))
                .expect("trapezoid axis=0 should match NumPy 2.x"),
        )
    } else {
        (
            "trapz_axis0",
            y.trapz(0.5, Some(0))
                .expect("trapz axis=0 should match NumPy 1.x"),
        )
    };

    assert_oracle_match(case_id, actual.shape(), actual.values(), 1e-12);

    let alias = y
        .trapz(0.5, Some(0))
        .expect("trapz alias should stay behaviorally equivalent");
    assert_eq!(alias.shape(), actual.shape());
    assert_eq!(alias.values(), actual.values());
}

#[test]
fn numpy_version_parser_accepts_release_and_prerelease_text() {
    assert_eq!(
        NumpyVersion::parse("2.4.3").expect("release version"),
        NumpyVersion {
            major: 2,
            minor: 4,
            patch: 3,
        }
    );
    assert_eq!(
        NumpyVersion::parse("2.5.0.dev0+git20260503").expect("dev version"),
        NumpyVersion {
            major: 2,
            minor: 5,
            patch: 0,
        }
    );
}
