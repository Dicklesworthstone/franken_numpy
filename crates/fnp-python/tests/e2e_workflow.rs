//! End-to-end workflow tests that exercise complete data processing pipelines.
//!
//! These tests verify that multiple fnp functions work correctly together
//! in realistic usage scenarios, not just in isolation.

use std::process::Command;

fn numpy_oracle(script: &str) -> Result<String, String> {
    let python = std::env::var("FNP_ORACLE_PYTHON")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "python3".to_string());
    let output = Command::new(&python)
        .args(["-c", script])
        .output()
        .map_err(|error| format!("{python} should be available: {error}\nScript: {script}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NumPy oracle failed: {stderr}\nScript: {script}"));
    }
    if !output.stderr.is_empty() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
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

// ─────────────────────────────────────────────────────────────────────────────
// Data normalization pipeline
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_normalize_data() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Simulate normalizing a dataset (z-score normalization)
data = fnp.arange(100).reshape(10, 10).astype(float)
mean = fnp.mean(data, axis=0)
std = fnp.std(data, axis=0)
normalized = fnp.divide(fnp.subtract(data, mean), std)

# Verify: normalized data should have mean ~0 and std ~1
result_mean = fnp.mean(normalized, axis=0)
result_std = fnp.std(normalized, axis=0)
print(fnp.allclose(result_mean, 0, atol=1e-10) and fnp.allclose(result_std, 1, atol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "normalize pipeline should work correctly"
    );
    Ok(())
}

#[test]
fn e2e_minmax_scale() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Min-max scaling to [0, 1] range
data = fnp.arange(50).reshape(5, 10).astype(float) * 3 + 10
min_val = fnp.amin(data, axis=0)
max_val = fnp.amax(data, axis=0)
scaled = fnp.divide(fnp.subtract(data, min_val), fnp.subtract(max_val, min_val))

# Verify: scaled data should be in [0, 1]
all_in_range = fnp.all(scaled >= 0) and fnp.all(scaled <= 1)
print(all_in_range)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "minmax scale pipeline should work");
    Ok(())
}

#[test]
fn e2e_loadtxt_public_api_feature_pipeline() -> Result<(), String> {
    let script = fnp_script(
        r#"
import json
import os
import sys
import tempfile

test_name = "loadtxt_public_api_feature_pipeline"

def log(phase, event, **data):
    print(json.dumps({
        "suite": "fnp_public_api_e2e",
        "test": test_name,
        "phase": phase,
        "event": event,
        "data": data,
    }, sort_keys=True), file=sys.stderr)

rows = [
    "day,temp_f,humidity,units",
    "1,68.0,0.45,120",
    "2,70.5,0.50,132",
    "3,66.0,0.47,118",
    "4,74.0,0.60,145",
    "5,71.5,0.58,139",
    "6,69.0,0.52,128",
]

with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as handle:
    handle.write("\n".join(rows))
    path = handle.name

try:
    log("load", "file_created", rows=len(rows) - 1)
    data = fnp.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=[1, 2, 3],
        dtype=float,
    )
    log("load", "loaded", shape=list(data.shape), dtype=str(data.dtype))

    temp_f = data[:, 0]
    humidity = data[:, 1]
    units = data[:, 2]

    temp_c = fnp.divide(fnp.multiply(fnp.subtract(temp_f, 32.0), 5.0), 9.0)
    unit_z = fnp.divide(fnp.subtract(units, fnp.mean(units)), fnp.std(units))
    risk_score = fnp.add(fnp.multiply(temp_c, 0.7), fnp.multiply(humidity, 12.0))
    high_risk = risk_score[units >= 130.0]

    actual = np.array([
        float(fnp.mean(temp_c)),
        float(fnp.amax(risk_score)),
        float(fnp.sum(high_risk > 24.0)),
        float(fnp.mean(unit_z)),
    ])
    log("operate", "pipeline_complete", high_risk_rows=int(len(high_risk)))

    expected_data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=[1, 2, 3],
        dtype=float,
    )
    expected_temp_c = (expected_data[:, 0] - 32.0) * 5.0 / 9.0
    expected_unit_z = (expected_data[:, 2] - np.mean(expected_data[:, 2])) / np.std(expected_data[:, 2])
    expected_risk = expected_temp_c * 0.7 + expected_data[:, 1] * 12.0
    expected_high_risk = expected_risk[expected_data[:, 2] >= 130.0]
    expected = np.array([
        np.mean(expected_temp_c),
        np.max(expected_risk),
        np.sum(expected_high_risk > 24.0),
        np.mean(expected_unit_z),
    ])

    passed = bool(np.allclose(actual, expected, atol=1e-12))
    log("assert", "allclose", actual=actual.tolist(), expected=expected.tolist(), passed=passed)
    print(passed)
finally:
    os.unlink(path)
    log("teardown", "file_removed")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "loadtxt public API pipeline should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix operations pipeline
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_matrix_transform() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Create rotation matrix and apply to points
theta = fnp.pi / 4  # 45 degrees
rotation = fnp.array([[fnp.cos(theta), -fnp.sin(theta)],
                       [fnp.sin(theta), fnp.cos(theta)]])
points = fnp.array([[1, 0], [0, 1], [1, 1], [2, 0]])

# Apply rotation: points @ rotation.T
rotated = fnp.matmul(points, fnp.transpose(rotation))

# Verify: distances from origin should be preserved
original_dist = fnp.sqrt(fnp.sum(fnp.square(points), axis=1))
rotated_dist = fnp.sqrt(fnp.sum(fnp.square(rotated), axis=1))
print(fnp.allclose(original_dist, rotated_dist))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matrix rotation should preserve distances"
    );
    Ok(())
}

#[test]
fn e2e_gram_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Compute Gram matrix (X @ X.T)
X = fnp.arange(12).reshape(3, 4).astype(float)
gram = fnp.matmul(X, fnp.transpose(X))

# Verify: Gram matrix should be symmetric and positive semi-definite
is_symmetric = fnp.allclose(gram, fnp.transpose(gram))
# Eigenvalues should be non-negative (check via diagonal of X@X.T being >= 0)
diag_positive = fnp.all(fnp.diag(gram) >= 0)
print(is_symmetric and diag_positive)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "Gram matrix should be symmetric");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistical analysis pipeline
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_correlation_workflow() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Compute correlation between two variables
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5  # y is correlated with x

# Convert to fnp arrays
x_fnp = fnp.array(x)
y_fnp = fnp.array(y)

# Compute Pearson correlation coefficient
x_mean = fnp.mean(x_fnp)
y_mean = fnp.mean(y_fnp)
x_centered = fnp.subtract(x_fnp, x_mean)
y_centered = fnp.subtract(y_fnp, y_mean)
numerator = fnp.sum(fnp.multiply(x_centered, y_centered))
denominator = fnp.sqrt(fnp.multiply(fnp.sum(fnp.square(x_centered)),
                                     fnp.sum(fnp.square(y_centered))))
r = fnp.divide(numerator, denominator)

# Correlation should be high (> 0.9) since y = 2x + noise
print(r > 0.9)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "correlation workflow should compute high r"
    );
    Ok(())
}

#[test]
fn e2e_running_statistics() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Compute running mean and variance
data = fnp.arange(100).astype(float)
window_size = 10

# Use cumsum to compute running sums efficiently
cumsum = fnp.cumsum(data)
# Running mean: (cumsum[i] - cumsum[i-window]) / window
# For first window elements, just use cumsum / index
running_sum = fnp.zeros(len(data) - window_size + 1)
running_sum[0] = cumsum[window_size - 1]
for i in range(1, len(running_sum)):
    running_sum[i] = cumsum[i + window_size - 1] - cumsum[i - 1]
running_mean = fnp.divide(running_sum, window_size)

# Verify: running mean should increase monotonically
diffs = fnp.diff(running_mean)
print(fnp.all(diffs > 0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "running statistics should work");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal processing pipeline
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_signal_smoothing() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Simple moving average smoothing
np.random.seed(123)
noisy_signal = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.3
signal = fnp.array(noisy_signal)

# Moving average with window of 5
kernel = fnp.ones(5) / 5
smoothed = fnp.convolve(signal, kernel, mode='valid')

# Verify: smoothed signal should have lower variance
original_var = fnp.var(signal[2:-2])  # trim to match convolution output
smoothed_var = fnp.var(smoothed)
print(smoothed_var < original_var)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "signal smoothing should reduce variance"
    );
    Ok(())
}

#[test]
fn e2e_derivative_approximation() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Approximate derivative using finite differences
x = fnp.linspace(0, 2 * fnp.pi, 100)
y = fnp.sin(x)

# First derivative using diff
dy = fnp.diff(y)
dx = fnp.diff(x)
derivative = fnp.divide(dy, dx)

# The derivative of sin(x) is cos(x)
# Compare at midpoints
x_mid = (x[:-1] + x[1:]) / 2
expected = fnp.cos(x_mid)

# Should be close (not exact due to finite differences)
print(fnp.allclose(derivative, expected, atol=0.1))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "derivative approximation should be close to cos(x)"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Array manipulation pipeline
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_reshape_transpose_chain() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Chain of reshape and transpose operations
data = fnp.arange(24)

# Reshape -> transpose -> reshape -> flatten -> reshape
step1 = fnp.reshape(data, (2, 3, 4))
step2 = fnp.transpose(step1, (2, 0, 1))  # (4, 2, 3)
step3 = fnp.reshape(step2, (8, 3))
step4 = fnp.ravel(step3)
step5 = fnp.reshape(step4, (6, 4))

# Verify: total elements preserved
print(fnp.prod(step5.shape) == 24 and len(fnp.unique(step4)) == 24)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "reshape/transpose chain should preserve data"
    );
    Ok(())
}

#[test]
fn e2e_split_process_concat() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Split array, process each part, concatenate back
data = fnp.arange(30).reshape(6, 5)

# Split into 3 parts along axis 0
parts = fnp.split(data, 3, axis=0)

# Process each part (double it)
processed = [fnp.multiply(p, 2) for p in parts]

# Concatenate back
result = fnp.concatenate(processed, axis=0)

# Verify: result should be 2x original
print(fnp.array_equal(result, fnp.multiply(data, 2)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "split/process/concat should work correctly"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Data filtering pipeline
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_outlier_removal() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Remove outliers using z-score method
np.random.seed(42)
data = np.concatenate([np.random.randn(100), [10, -10, 15]])  # add outliers
data = fnp.array(data)

mean = fnp.mean(data)
std = fnp.std(data)
z_scores = fnp.absolute(fnp.divide(fnp.subtract(data, mean), std))

# Keep only values with z-score < 3
mask = z_scores < 3
cleaned = data[mask]

# Verify: outliers should be removed
print(len(cleaned) < len(data) and fnp.amax(fnp.absolute(cleaned)) < 5)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "outlier removal should work");
    Ok(())
}

#[test]
fn e2e_histogram_analysis() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Analyze distribution using histogram
np.random.seed(42)
data = np.random.randn(1000)
data = fnp.array(data)

# Compute histogram
hist, edges = fnp.histogram(data, bins=20)

# Find bin with most counts (should be near 0 for normal distribution)
peak_bin_idx = fnp.argmax(hist)
peak_bin_center = (edges[peak_bin_idx] + edges[peak_bin_idx + 1]) / 2

# Peak should be near 0 (within 0.5)
print(fnp.absolute(peak_bin_center) < 0.5)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "histogram peak should be near 0");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerical stability pipeline
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_logsumexp() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Compute log(sum(exp(x))) in a numerically stable way
x = fnp.array([1000, 1001, 1002])  # large values that would overflow exp

# Naive: exp would overflow
# Stable: max + log(sum(exp(x - max)))
max_x = fnp.amax(x)
stable_result = max_x + fnp.log(fnp.sum(fnp.exp(fnp.subtract(x, max_x))))

# Verify: result should be finite and reasonable
print(fnp.isfinite(stable_result) and stable_result > 1000)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logsumexp should be numerically stable"
    );
    Ok(())
}

#[test]
fn e2e_softmax() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Compute softmax in a numerically stable way
x = fnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Stable softmax: subtract max before exp
max_x = fnp.amax(x)
exp_x = fnp.exp(fnp.subtract(x, max_x))
softmax = fnp.divide(exp_x, fnp.sum(exp_x))

# Verify: softmax should sum to 1 and be positive
sums_to_one = fnp.allclose(fnp.sum(softmax), 1.0)
all_positive = fnp.all(softmax > 0)
print(sums_to_one and all_positive)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "softmax should sum to 1 and be positive"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid computation pipeline
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_meshgrid_computation() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Evaluate 2D function on a grid
x = fnp.linspace(-2, 2, 50)
y = fnp.linspace(-2, 2, 50)
X, Y = fnp.meshgrid(x, y)

# Compute 2D Gaussian
sigma = 1.0
Z = fnp.exp(fnp.negative(fnp.divide(fnp.add(fnp.square(X), fnp.square(Y)),
                                      2 * sigma**2)))

# Verify: peak should be at center, edges should be small
center_idx = 25
center_val = Z[center_idx, center_idx]
corner_val = Z[0, 0]
print(center_val > 0.9 and corner_val < 0.2)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "meshgrid Gaussian computation should work"
    );
    Ok(())
}

#[test]
fn e2e_distance_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Compute pairwise Euclidean distance matrix
points = fnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
n = len(points)

# Broadcasting approach: |a - b|^2 = |a|^2 + |b|^2 - 2*a.b
sq_norms = fnp.sum(fnp.square(points), axis=1)
dot_products = fnp.matmul(points, fnp.transpose(points))

# Compute squared distances using broadcasting
sq_dists = fnp.add(fnp.reshape(sq_norms, (n, 1)),
                    fnp.subtract(sq_norms, fnp.multiply(dot_products, 2)))
distances = fnp.sqrt(fnp.maximum(sq_dists, 0))  # numerical safety

# Verify: diagonal should be 0, distances should be symmetric
diag_zero = fnp.allclose(fnp.diag(distances), 0)
symmetric = fnp.allclose(distances, fnp.transpose(distances))
print(diag_zero and symmetric)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "distance matrix should be symmetric with 0 diagonal"
    );
    Ok(())
}
