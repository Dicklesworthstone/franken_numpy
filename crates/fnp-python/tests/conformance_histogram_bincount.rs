//! Conformance tests for numpy histogram and bincount functions against NumPy oracle.
//!
//! Tests histogram, histogram_bin_edges, bincount, digitize.

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

mod support;
use support::fnp_script;

#[test]
fn histogram_explicit_edges_validates_allocator_outputs() -> Result<(), String> {
    let script = fnp_script(
        r#"
original_empty = np.empty
source = (np.arange((1 << 20) + 7, dtype=np.float64) % 23).reshape(-1)
for edge_values in [[0., 3., 7., 12., 19., 23.],
                    [-np.inf, 0., 0., 7., 23., np.inf]]:
    expected = np.histogram(source, bins=np.array(edge_values))
    for mode in ['normal', 'dirty', 'mutate_input', 'alias_input', 'short',
                 'shape', 'strided', 'readonly', 'overlap', 'reverse_overlap',
                 'mutate_edges', 'reshape_edges']:
        x = source.copy()
        backing = np.zeros(8, dtype=np.float64)
        offset = 1 if mode == 'reverse_overlap' else 0
        edges = backing[offset:offset + len(edge_values)]
        edges[:] = edge_values
        calls = []
        def empty(shape, dtype):
            calls.append(shape)
            if mode == 'mutate_input':
                x.fill(99)
            if mode == 'mutate_edges':
                edges[1] += 1
            if mode == 'reshape_edges':
                edges.shape = (2, 3)
            if mode == 'alias_input':
                return x.view(np.int64)[:shape]
            if mode in ['overlap', 'reverse_overlap']:
                start = 1 if mode == 'overlap' else 0
                return backing.view(np.int64)[start:start + shape]
            if mode == 'short':
                return original_empty(0, dtype=dtype)
            if mode == 'shape':
                return original_empty((shape, 1), dtype=dtype)
            if mode == 'strided':
                return original_empty(shape * 2, dtype=dtype)[::2]
            result = original_empty(shape, dtype=dtype)
            result.fill(123)
            if mode == 'readonly':
                result.flags.writeable = False
            return result
        np.empty = empty
        try:
            if mode in ['normal', 'dirty', 'mutate_input', 'alias_input']:
                actual = fnp.histogram(x, bins=edges)
                assert all(parity_equal(a, b) for a, b in zip(actual, expected)), (
                    mode, edge_values, actual, expected)
                assert actual[1] is edges
                if mode == 'alias_input':
                    assert np.shares_memory(actual[0], x)
            else:
                try:
                    fnp.histogram(x, bins=edges)
                except ValueError as error:
                    assert 'histogram allocator returned' in str(error), str(error)
                else:
                    raise AssertionError((mode, 'invalid output accepted'))
        finally:
            np.empty = original_empty
        assert calls == [5], (mode, calls)
print('ok')
"#
        .into(),
    );
    let output = Command::new("python3")
        .env("RAYON_NUM_THREADS", "2")
        .args(["-c", &script])
        .output()
        .map_err(|error| error.to_string())?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "ok");
    Ok(())
}

#[test]
fn histogram_f32_callbacks_preserve_snapshot_and_validate_outputs() -> Result<(), String> {
    let script = fnp_script(
        r#"
original_zeros, original_linspace, original_float32 = np.zeros, np.linspace, np.float32
for order in ['empty', 'sorted', 'unsorted']:
    source = np.array([] if order == 'empty' else list(range(24)), dtype=np.float32)
    if order == 'unsorted':
        source = source[::-1].copy()
    expected = np.histogram(source, bins=5)
    modes = ['mutate_scalar', 'mutate_edges', 'mutate_counts', 'dirty_counts',
             'short_edges', 'shape_edges', 'strided_edges', 'short_counts',
             'shape_counts', 'strided_counts', 'readonly_counts', 'overlap',
             'reverse_overlap', 'reshape_edges']
    if len(source):
        modes += ['alias_input']
    for mode in modes:
        x = source.copy()
        calls, produced_edges, scalar_calls = [], [], []
        def scalar(value):
            scalar_calls.append(value)
            x.fill(99)
            return original_float32(value)
        def linspace(start, stop, num, dtype):
            calls.append('edges')
            if mode == 'mutate_edges':
                x.fill(99)
            result = original_linspace(start, stop, num, dtype=dtype)
            if mode == 'short_edges':
                result = result[:0]
            elif mode == 'shape_edges':
                result = result.reshape(2, 3)
            elif mode == 'strided_edges':
                result = original_linspace(start, stop, num * 2, dtype=dtype)[::2]
            elif mode in ['overlap', 'reverse_overlap']:
                backing = original_zeros(16, dtype='float32')
                offset = 0 if mode == 'overlap' else 8
                backing[offset:offset + num] = result
                result = backing[offset:offset + num]
            produced_edges.append(result)
            return result
        def zeros(shape, dtype):
            calls.append('counts')
            if mode == 'mutate_counts':
                x.fill(99)
            if mode == 'reshape_edges':
                produced_edges[0].shape = (2, 3)
            if mode == 'short_counts':
                return original_zeros(0, dtype=dtype)
            if mode == 'shape_counts':
                return original_zeros((shape, 1), dtype=dtype)
            if mode == 'strided_counts':
                return original_zeros(shape * 2, dtype=dtype)[::2]
            if mode in ['overlap', 'reverse_overlap']:
                offset = 2 if mode == 'overlap' else 0
                return produced_edges[0].base.view(np.int64)[offset:offset + shape]
            if mode == 'alias_input':
                return x.view(np.int64)[:shape]
            result = original_zeros(shape, dtype=dtype)
            if mode == 'readonly_counts':
                result.flags.writeable = False
            if mode == 'dirty_counts':
                result.fill(123)
            return result
        np.linspace, np.zeros = linspace, zeros
        if mode == 'mutate_scalar':
            np.float32 = scalar
        try:
            if mode in ['mutate_scalar', 'mutate_edges', 'mutate_counts', 'dirty_counts', 'alias_input']:
                actual = fnp.histogram(x, bins=5)
                assert all(parity_equal(a, b) for a, b in zip(actual, expected)), (
                    order, mode, actual, expected)
                if mode == 'alias_input':
                    assert np.shares_memory(actual[0], x)
            else:
                try:
                    fnp.histogram(x, bins=5)
                except ValueError as error:
                    assert 'histogram allocator returned' in str(error), str(error)
                else:
                    raise AssertionError((order, mode, 'invalid output accepted'))
        finally:
            np.linspace, np.zeros, np.float32 = original_linspace, original_zeros, original_float32
        assert calls == ['edges', 'counts'], (order, mode, calls)
        if mode == 'mutate_scalar':
            assert len(scalar_calls) == 2, (order, scalar_calls)
            assert np.all(x == 99), (order, x)
print('ok')
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "ok");
    Ok(())
}

#[test]
fn histogram_allocator_callbacks_validate_buffers_and_preserve_input_snapshot() -> Result<(), String>
{
    let script = fnp_script(
        r#"
original_zeros, original_linspace = np.zeros, np.linspace
for dtype in [np.float64, np.int64, np.uint8]:
    for n in [0, 17, (1 << 21) + 7]:
        source = (np.arange(n, dtype=np.int64) % 11).astype(dtype)
        expected = np.histogram(source, bins=5)
        modes = ['mutate_edges', 'mutate_counts', 'short_edges', 'shape_edges',
                 'strided_edges', 'short_counts', 'shape_counts', 'readonly_counts',
                 'strided_counts', 'overlap', 'reverse_overlap', 'reshape_edges', 'dirty_counts']
        if n >= 5 and np.dtype(dtype).itemsize == 8:
            modes += ['alias_input']
        for mode in modes:
            x = source.copy()
            calls = []
            produced_edges = []
            def linspace(start, stop, num):
                calls.append('edges')
                if mode == 'mutate_edges':
                    x.fill(99)
                result = original_linspace(start, stop, num)
                if mode == 'short_edges':
                    result = result[:0]
                elif mode == 'shape_edges':
                    result = result.reshape(2, 3)
                elif mode == 'strided_edges':
                    result = original_linspace(start, stop, num * 2)[::2]
                elif mode == 'reverse_overlap':
                    backing = np.empty(num + 1, dtype=np.float64)
                    backing[1:] = result
                    result = backing[1:]
                produced_edges.append(result)
                return result
            def zeros(shape, dtype):
                calls.append('counts')
                if mode == 'mutate_counts':
                    x.fill(99)
                if mode == 'reshape_edges':
                    produced_edges[0].shape = (2, 3)
                if mode == 'short_counts':
                    return original_zeros(0, dtype=dtype)
                if mode == 'shape_counts':
                    return original_zeros((shape, 1), dtype=dtype)
                if mode == 'strided_counts':
                    return original_zeros(shape * 2, dtype=dtype)[::2]
                if mode == 'overlap':
                    return produced_edges[0].view(np.int64)[1:]
                if mode == 'reverse_overlap':
                    return produced_edges[0].base.view(np.int64)[:shape]
                if mode == 'alias_input':
                    return x.view(np.int64)[:shape]
                result = original_zeros(shape, dtype=dtype)
                if mode == 'readonly_counts':
                    result.flags.writeable = False
                if mode == 'dirty_counts':
                    result.fill(123)
                return result
            np.linspace, np.zeros = linspace, zeros
            try:
                if mode in ['mutate_edges', 'mutate_counts', 'alias_input', 'dirty_counts']:
                    actual = fnp.histogram(x, bins=5)
                    assert all(parity_equal(a, b) for a, b in zip(actual, expected)), (
                        dtype, n, mode, actual, expected)
                    if mode == 'alias_input':
                        assert np.shares_memory(actual[0], x)
                else:
                    try:
                        fnp.histogram(x, bins=5)
                    except ValueError as error:
                        assert 'histogram allocator returned' in str(error), str(error)
                    else:
                        raise AssertionError((dtype, n, mode, 'invalid output accepted'))
            finally:
                np.linspace, np.zeros = original_linspace, original_zeros
            assert calls == ['edges', 'counts'], (dtype, n, mode, calls)
print('ok')
"#
        .into(),
    );
    let output = Command::new("python3")
        .env("RAYON_NUM_THREADS", "2")
        .args(["-c", &script])
        .output()
        .map_err(|error| error.to_string())?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "ok");
    Ok(())
}

#[test]
fn unweighted_bincount_checks_indices_after_python_allocation() -> Result<(), String> {
    let script = fnp_script(
        r#"
original_zeros = np.zeros
calls = []
def allocating_zeros(shape, dtype):
    calls.append((x.dtype.name, mode))
    if mode == 'negative':
        x[0] = -1
    elif mode == 'too_large':
        x[0] = 8
    elif mode == 'short_output':
        return original_zeros(0, dtype=dtype)
    return original_zeros(shape, dtype=dtype)
np.zeros = allocating_zeros
expected_calls = []
try:
    for dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32]:
        x = np.array([0], dtype=dtype)
        modes = ['too_large', 'short_output']
        if x.dtype.kind == 'i':
            modes.append('negative')
        for mode in modes:
            x[0] = 0
            expected_calls.append((x.dtype.name, mode))
            try:
                fnp.bincount(x)
            except ValueError as error:
                assert 'outside the allocated output' in str(error), (dtype, mode, str(error))
            else:
                raise AssertionError((dtype, mode, 'invalid index was accepted'))
    assert calls == expected_calls, (calls, expected_calls)
finally:
    np.zeros = original_zeros
print('ok')
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "ok");
    Ok(())
}

#[test]
fn unweighted_bincount_parallel_snapshot_survives_allocation_callback() -> Result<(), String> {
    let script = fnp_script(
        r#"
original_zeros = np.zeros
x = original_zeros(1 << 26, dtype=np.uint8)
calls = []
def allocating_zeros(shape, dtype):
    calls.append(mode)
    x[0] = 8
    return original_zeros(0 if mode == 'short_output' else shape, dtype=dtype)
np.zeros = allocating_zeros
try:
    mode = 'mutate'
    result = fnp.bincount(x)
    assert result.tolist() == [len(x)], result
    mode = 'short_output'
    x[0] = 0
    try:
        fnp.bincount(x)
    except ValueError as error:
        assert 'outside the allocated output' in str(error), str(error)
    else:
        raise AssertionError('short output was accepted')
    assert calls == ['mutate', 'short_output'], calls
finally:
    np.zeros = original_zeros
print('ok')
"#
        .into(),
    );
    // Set the pool size only in this fresh child; never mutate the test
    // runner's environment while other tests are executing.
    let output = Command::new("python3")
        .env("RAYON_NUM_THREADS", "2")
        .args(["-c", &script])
        .output()
        .map_err(|error| error.to_string())?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "ok");
    Ok(())
}

#[test]
fn weighted_bincount_checks_indices_after_python_allocation() -> Result<(), String> {
    // Each script gets a fresh interpreter, so poisoning NumPy's allocator
    // cannot affect another test or depend on an already initialized cache.
    let script = fnp_script(
        r#"
original_zeros = np.zeros
calls = []
x = np.array([0], dtype=np.int64)
weights = np.array([1.0], dtype=np.float64)
def allocating_zeros(shape, dtype):
    calls.append(mode)
    if mode == 'negative':
        x[0] = -1
    elif mode == 'too_large':
        x[0] = 8
    elif mode == 'short_output':
        return original_zeros(0, dtype=dtype)
    return original_zeros(shape, dtype=dtype)
np.zeros = allocating_zeros
try:
    for mode in ['negative', 'too_large', 'short_output']:
        x[0] = 0
        try:
            fnp.bincount(x, weights=weights)
        except ValueError as error:
            assert 'outside the allocated output' in str(error), (mode, str(error))
        else:
            raise AssertionError((mode, 'invalid index was accepted'))
    assert calls == ['negative', 'too_large', 'short_output'], calls
finally:
    np.zeros = original_zeros
print('ok')
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "ok");
    Ok(())
}

fn outcome_body(setup: &str, call_expr: &str) -> String {
    format!(
        "{setup}\n\
         def outcome(op):\n\
         {I4}try:\n\
         {I8}value = {call_expr}\n\
         {I8}arr = np.asarray(value)\n\
         {I8}print('ok')\n\
         {I8}print(type(value).__name__)\n\
         {I8}print(str(arr.dtype))\n\
         {I8}print(tuple(arr.shape))\n\
         {I8}print(repr(arr.tolist()))\n\
         {I4}except Exception as exc:\n\
         {I8}print('err')\n\
         {I8}print(type(exc).__name__)\n\
         outcome(op)",
        I4 = "    ",
        I8 = "        ",
    )
}

fn numpy_outcome_script(function_expr: &str, setup: &str, call_expr: &str) -> String {
    format!(
        "import numpy as np\nop = {function_expr}\n{}",
        outcome_body(setup, call_expr)
    )
}

fn fnp_outcome_script(function_name: &str, setup: &str, call_expr: &str) -> String {
    fnp_script(format!(
        "op = fnp.{function_name}\n{}",
        outcome_body(setup, call_expr)
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// histogram
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn histogram_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
hist, edges = fnp.histogram(a)
np_hist, np_edges = np.histogram(a)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "histogram basic should match numpy");
    Ok(())
}

#[test]
fn histogram_with_bins() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
hist, edges = fnp.histogram(a, bins=5)
np_hist, np_edges = np.histogram(a, bins=5)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram with bins should match numpy"
    );
    Ok(())
}

#[test]
fn histogram_with_range() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
hist, edges = fnp.histogram(a, bins=5, range=(2, 8))
np_hist, np_edges = np.histogram(a, bins=5, range=(2, 8))
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram with range should match numpy"
    );
    Ok(())
}

#[test]
fn histogram_with_explicit_edges() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
bin_edges = np.array([1, 2, 3, 4, 5])
hist, edges = fnp.histogram(a, bins=bin_edges)
np_hist, np_edges = np.histogram(a, bins=bin_edges)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram with explicit edges should match numpy"
    );
    Ok(())
}

#[test]
fn histogram_density() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
hist, edges = fnp.histogram(a, density=True)
np_hist, np_edges = np.histogram(a, density=True)
print(np.allclose(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram density should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// bincount
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bincount_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 1, 2, 2, 2, 3])
result = fnp.bincount(a)
expected = np.bincount(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bincount basic should match numpy");
    Ok(())
}

#[test]
fn bincount_with_weights() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 1, 2, 2, 2, 3])
w = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
result = fnp.bincount(a, weights=w)
expected = np.bincount(a, weights=w)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount with weights should match numpy"
    );
    Ok(())
}

#[test]
fn bincount_with_minlength() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 1, 2])
result = fnp.bincount(a, minlength=5)
expected = np.bincount(a, minlength=5)
print(np.array_equal(result, expected) and len(result) == 5)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount with minlength should match numpy"
    );
    Ok(())
}

#[test]
fn bincount_empty() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([], dtype=int)
result = fnp.bincount(a)
expected = np.bincount(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bincount empty should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// digitize
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn histogram_bin_edges_python_container_keyword_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        ("list data with int bins", "", "op([0, 1, 2, 3], bins=4)"),
        (
            "tuple data with explicit edge list",
            "",
            "op((0.2, 0.8, 1.4), bins=[0.0, 0.5, 1.0, 1.5])",
        ),
        (
            "range and weights keywords",
            "",
            "op([0, 1, 2, 3], bins=3, range=(0, 3), weights=[1, 2, 3, 4])",
        ),
        (
            "auto bin estimator fallback",
            "",
            "op([0.0, 0.5, 1.0, 1.5, 2.0], bins='auto')",
        ),
        (
            "invalid range error type",
            "",
            "op([0, 1], bins=3, range=(2, 1))",
        ),
    ];

    for (label, setup, call_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script(
            "np.histogram_bin_edges",
            setup,
            call_expr,
        ))?;
        let rust_result =
            numpy_oracle(&fnp_outcome_script("histogram_bin_edges", setup, call_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "histogram_bin_edges Python-container keyword surface mismatch for {label}"
        );
    }

    Ok(())
}

#[test]
fn digitize_python_container_keyword_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        (
            "list x with tuple bins",
            "",
            "op([0.2, 1.5, 2.3], (1.0, 2.0, 3.0))",
        ),
        (
            "tuple x with right keyword",
            "",
            "op((1, 2, 3), [1, 2, 3], right=True)",
        ),
        ("scalar x output", "", "op(np.float64(2.5), [1, 2, 3, 4])"),
        (
            "decreasing bins with right keyword",
            "",
            "op([0.5, 1.5, 3.5], [4, 3, 2, 1], right=True)",
        ),
        ("nonmonotonic bins error type", "", "op([1, 2], [0, 2, 1])"),
    ];

    for (label, setup, call_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script("np.digitize", setup, call_expr))?;
        let rust_result = numpy_oracle(&fnp_outcome_script("digitize", setup, call_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "digitize Python-container keyword surface mismatch for {label}"
        );
    }

    Ok(())
}

#[test]
fn digitize_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.2, 0.8, 1.5, 2.3, 3.8, 5.0])
bins = np.array([1, 2, 3, 4])
result = fnp.digitize(x, bins)
expected = np.digitize(x, bins)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "digitize basic should match numpy");
    Ok(())
}

#[test]
fn digitize_right() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4])
bins = np.array([1, 2, 3, 4])
result = fnp.digitize(x, bins, right=True)
expected = np.digitize(x, bins, right=True)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "digitize right should match numpy");
    Ok(())
}

#[test]
fn digitize_decreasing() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.2, 0.8, 1.5, 2.3, 3.8, 5.0])
bins = np.array([4, 3, 2, 1])  # decreasing
result = fnp.digitize(x, bins)
expected = np.digitize(x, bins)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "digitize decreasing bins should match numpy"
    );
    Ok(())
}

#[test]
fn digitize_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = (1 << 21) + 4096
shape = (2052, 1024)

def check(values, bins, right):
    result = fnp.digitize(values, bins, right=right)
    expected = np.digitize(values, bins, right=right)
    return (
        result.dtype == expected.dtype
        and result.shape == expected.shape
        and result.flags.c_contiguous
        and result.tobytes() == expected.tobytes()
    )

ok = True

x64 = np.linspace(-5.0, 5.0, n, dtype=np.float64).reshape(shape)
x64_flat = x64.ravel()
x64_flat[0] = np.nan
x64_flat[97] = np.inf
x64_flat[211] = -np.inf
x64_flat[4096] = -0.0
bins64 = np.linspace(-4.0, 4.0, 50, dtype=np.float64)
ok = ok and check(x64, bins64, False)
ok = ok and check(x64, bins64, True)

x32 = x64.astype(np.float32, copy=True)
bins32 = bins64.astype(np.float32)
ok = ok and check(x32, bins32, False)
ok = ok and check(x32, bins32, True)

i64 = ((np.arange(n, dtype=np.int64) % 1000) - 500).reshape(shape)
ibins = np.array([-500, -1, 0, 0, 1, 499], dtype=np.int64)
ok = ok and check(i64, ibins, False)
ok = ok and check(i64, ibins, True)

u16 = (np.arange(n, dtype=np.uint32) % 1000).astype(np.uint16).reshape(shape)
ubins = np.array([0, 1, 1, 500, 999], dtype=np.uint16)
ok = ok and check(u16, ubins, False)
ok = ok and check(u16, ubins, True)

dec_x = np.array([0.5, 1.5, 3.5], dtype=np.float64)
dec_bins = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64)
ok = ok and check(dec_x, dec_bins, False)
ok = ok and check(dec_x, dec_bins, True)

print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large digitize parallel path should match numpy bit-exactly"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn histogram_sum_equals_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
hist, _ = fnp.histogram(a)
print(np.sum(hist) == len(a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram sum should equal array length"
    );
    Ok(())
}

#[test]
fn bincount_sum_equals_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 1, 2, 2, 2, 3])
result = fnp.bincount(a)
print(np.sum(result) == len(a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount sum should equal array length"
    );
    Ok(())
}

#[test]
fn digitize_searchsorted_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.2, 1.5, 2.8, 4.2])
bins = np.array([1, 2, 3, 4])
digitize_result = fnp.digitize(x, bins)
searchsorted_result = fnp.searchsorted(bins, x, side='right')
print(np.array_equal(digitize_result, searchsorted_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "digitize should equal searchsorted for increasing bins"
    );
    Ok(())
}

#[test]
fn digitize_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(2.5)
bins = np.array([1, 2, 3, 4])
fnp_result = fnp.digitize(x, bins)
np_result = np.digitize(x, bins)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "digitize scalar return type should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn histogram_empty() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([], dtype=np.float64)
hist, edges = fnp.histogram(a, bins=5)
np_hist, np_edges = np.histogram(a, bins=5)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "histogram empty should match numpy");
    Ok(())
}

#[test]
fn histogram_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5.0])
hist, edges = fnp.histogram(a, bins=3)
np_hist, np_edges = np.histogram(a, bins=3)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram single element should match numpy"
    );
    Ok(())
}

#[test]
fn histogram_all_same_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
hist, edges = fnp.histogram(a)
np_hist, np_edges = np.histogram(a)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram all same value should match numpy"
    );
    Ok(())
}

#[test]
fn histogram_edge_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
bins = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
hist, edges = fnp.histogram(a, bins=bins)
np_hist, np_edges = np.histogram(a, bins=bins)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram edge values should match numpy"
    );
    Ok(())
}

/// Locks the typed uniform-bin histogram fast path to NumPy's raw output bytes:
/// int64 counts, f32 edges for f32 inputs, f64 edges for f64 and supported
/// integer inputs, and fallback/error parity for unsupported cases.
#[test]
fn histogram_typed_uniform_bins_bit_exact_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib
chunks = []

def capture(func):
    try:
        hist, edges = func()
    except Exception as exc:
        return ("E", type(exc).__name__, None, None)
    return ("O", None, np.asarray(hist), np.asarray(edges))

def record(label, values, bins=10, **kwargs):
    got_kind, got_exc, got_hist, got_edges = capture(
        lambda: fnp.histogram(values, bins=bins, **kwargs)
    )
    exp_kind, exp_exc, exp_hist, exp_edges = capture(
        lambda: np.histogram(values, bins=bins, **kwargs)
    )
    assert got_kind == exp_kind, (label, got_kind, exp_kind)
    chunks.append(label.encode())
    chunks.append(b'\0')
    if got_kind == "E":
        assert got_exc == exp_exc, (label, got_exc, exp_exc)
        chunks.append(b'E')
        chunks.append(got_exc.encode())
        return
    assert got_hist.dtype == exp_hist.dtype, (label, got_hist.dtype, exp_hist.dtype)
    assert got_edges.dtype == exp_edges.dtype, (label, got_edges.dtype, exp_edges.dtype)
    assert got_hist.shape == exp_hist.shape, (label, got_hist.shape, exp_hist.shape)
    assert got_edges.shape == exp_edges.shape, (label, got_edges.shape, exp_edges.shape)
    assert got_hist.tobytes() == exp_hist.tobytes(), (
        label,
        got_hist.tolist(),
        exp_hist.tolist(),
    )
    assert got_edges.tobytes() == exp_edges.tobytes(), (
        label,
        got_edges.tolist(),
        exp_edges.tolist(),
    )
    chunks.append(b'O')
    chunks.append(got_hist.dtype.str.encode())
    chunks.append(str(got_hist.shape).encode())
    chunks.append(got_hist.tobytes())
    chunks.append(got_edges.dtype.str.encode())
    chunks.append(str(got_edges.shape).encode())
    chunks.append(got_edges.tobytes())

for dtype in (np.float32, np.float64):
    record(f'{dtype.__name__}:empty', np.array([], dtype=dtype), bins=5)
    record(f'{dtype.__name__}:same', np.array([7, 7, 7], dtype=dtype), bins=5)
    record(f'{dtype.__name__}:linear', np.linspace(-1000, 1000, 10000, dtype=dtype), bins=50)
    record(f'{dtype.__name__}:edges',
           np.array([-1, -0.8, -0.2, 0, 0.2, 0.8, 1], dtype=dtype), bins=5)

for dtype in (np.int8, np.int16, np.int32, np.int64):
    record(f'{dtype.__name__}:signed', ((np.arange(1000) % 37) - 13).astype(dtype), bins=11)

for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
    record(f'{dtype.__name__}:unsigned', (np.arange(1000) % 37).astype(dtype), bins=11)

record('int64:exact-boundary', np.array([-2**53, -1, 0, 2**53], dtype=np.int64), bins=4)
record('uint64:exact-boundary', np.array([0, 1, 2**32, 2**53], dtype=np.uint64), bins=4)
record('int64:large-error', np.array([2**60, 2**60 + 3], dtype=np.int64), bins=5)
record('float32:nonfinite-error', np.array([1, np.inf], dtype=np.float32), bins=5)
record('float32:strided-defer', np.arange(20, dtype=np.float32)[::2], bins=5)
record('float32:range-defer', np.arange(20, dtype=np.float32), bins=5, range=(2, 18))
record('float32:density-defer', np.arange(20, dtype=np.float32), bins=5, density=True)

print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#
        .into(),
    );
    let hash = numpy_oracle(&script)?;
    assert_eq!(
        hash, "dca1ab4a9b56fc672a88e951bcd68f25b8db593ee67dfd5364f17214b39f5739",
        "typed uniform-bin histogram must be bit-identical to numpy (sha256 of dtype/shape/raw output bytes)"
    );
    Ok(())
}

#[test]
fn bincount_zeros_only() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 0, 0, 0, 0])
result = fnp.bincount(a)
expected = np.bincount(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount zeros only should match numpy"
    );
    Ok(())
}

#[test]
fn bincount_single_large_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([100])
result = fnp.bincount(a)
expected = np.bincount(a)
print(np.array_equal(result, expected) and len(result) == 101)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount single large value should match numpy"
    );
    Ok(())
}

#[test]
fn bincount_sparse_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 50, 100])
result = fnp.bincount(a)
expected = np.bincount(a)
print(np.array_equal(result, expected) and result[0] == 1 and result[50] == 1 and result[100] == 1)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount sparse values should match numpy"
    );
    Ok(())
}

#[test]
fn bincount_with_zero_weights() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 2, 3])
w = np.array([1.0, 0.0, 0.0, 1.0])
result = fnp.bincount(a, weights=w)
expected = np.bincount(a, weights=w)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount with zero weights should match numpy"
    );
    Ok(())
}

#[test]
fn digitize_empty() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([], dtype=np.float64)
bins = np.array([1, 2, 3, 4])
result = fnp.digitize(x, bins)
expected = np.digitize(x, bins)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "digitize empty should match numpy");
    Ok(())
}

#[test]
fn digitize_single_bin() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
bins = np.array([1.0])
result = fnp.digitize(x, bins)
expected = np.digitize(x, bins)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "digitize single bin should match numpy"
    );
    Ok(())
}

#[test]
fn digitize_exact_matches() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0])
bins = np.array([1.0, 2.0, 3.0, 4.0])
result_left = fnp.digitize(x, bins, right=False)
result_right = fnp.digitize(x, bins, right=True)
expected_left = np.digitize(x, bins, right=False)
expected_right = np.digitize(x, bins, right=True)
print(np.array_equal(result_left, expected_left) and np.array_equal(result_right, expected_right))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "digitize exact matches should match numpy"
    );
    Ok(())
}

#[test]
fn digitize_inf_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-np.inf, 0.0, np.inf])
bins = np.array([1.0, 2.0, 3.0])
result = fnp.digitize(x, bins)
expected = np.digitize(x, bins)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "digitize inf values should match numpy"
    );
    Ok(())
}

/// Locks the zero-copy bincount fast path (`try_zerocopy_bincount`, the 1-D
/// non-negative int64 no-weights case that tallies the buffer directly into an
/// int64 output) to bit-exact parity with numpy, including the int64 result
/// dtype and the minlength-driven output length. Compares the sha256 of raw
/// output bytes across sparse and dense ranges and explicit minlength.
#[test]
fn bincount_zerocopy_int64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for n in [1000, 100003]:
    x = rng.integers(0, 500, n)
    out = np.asarray(mod.bincount(x))
    chunks.append(bytes([1 if out.dtype == np.int64 else 0]))
    chunks.append(out.tobytes())
    chunks.append(np.asarray(mod.bincount(x, minlength=1000)).tobytes())
chunks.append(np.asarray(mod.bincount(np.array([0, 5, 5, 2, 9, 0], dtype=np.int64))).tobytes())
chunks.append(np.asarray(mod.bincount(np.array([], dtype=np.int64), minlength=10)).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy bincount must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}

/// numpy's bincount converts a SEQUENCE with an intp target, so an EMPTY sequence is an empty
/// int64 count (length `minlength`), not the float64 safe-cast TypeError fnp raised - `asarray`
/// makes `[]` float64 (numpy's own TestBincount::test_empty_list). A float LIST is
/// deprecated-but-accepted on numpy 2.1+ (DeprecationWarning, then truncation) while a float
/// ARRAY raises; the outcome, warnings included, must be whatever the live numpy does.
#[test]
fn bincount_empty_sequence_is_an_empty_int_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
def outcome(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = fn()
            value = ("ok", type(r).__name__, str(r.dtype), r.tolist())
        except Exception as exc:
            value = ("err", type(exc).__name__)
    return value, sorted(c.category.__name__ for c in caught)
class DtypeDirected:
    def __array__(self, dtype=None, copy=None):
        return np.array([] if dtype is None else [1, 1, 2], dtype=dtype)
cases = [
    lambda m: m.bincount([]),
    lambda m: m.bincount([], minlength=3),
    lambda m: m.bincount((), weights=[]),
    lambda m: m.bincount([1.0, 2.0]),
    lambda m: m.bincount(np.array([1.0, 2.0])),
    lambda m: m.bincount([0, 2, 2]),
    # A list of strings, bytes or objects is numpy's per-element conversion (numpy's
    # TestBincount::test_bad_list: ['0', '1', '1'] is [1 2] with a DeprecationWarning); a str
    # ARRAY is its safe-cast TypeError. fnp raised its own TypeError for all four, so the three
    # list cells failed on a8d9a337.
    lambda m: m.bincount(["0", "1", "1"]),
    lambda m: m.bincount([b"1"]),
    lambda m: m.bincount([None]),
    lambda m: m.bincount(np.array(["0", "1"])),
    lambda m: m.bincount(DtypeDirected()),
]
bad = [i for i, c in enumerate(cases) if outcome(lambda: c(fnp)) != outcome(lambda: c(np))]
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "bincount on empty sequences must match numpy: {result}"
    );
    Ok(())
}

/// The serial uniform-bin tally (below the 2^21 parallel gate) takes its ±1 edge corrections
/// branch-free against a +inf sentinel above the last bin, which is only right if a value on an
/// edge, on the last edge, or in the first / last bin lands where numpy puts it. The grid holds
/// every integer width and float64 over flat data (first and last bins each take 1/bins of the
/// values), normal data, values sitting exactly on the edges, a constant array (numpy widens the
/// range by 0.5), two values, and bins 1 / 2 / 10 / 100 / 1000, at sizes on both sides of the
/// parallel gate. Plus the ranges numpy raises on: a width that overflows (`[-1e308, 1e308]`
/// with bins=1 is numpy's IndexError; it was answered with counts by the extract path, for an
/// ndarray and a list alike), a range too narrow for the bins, >2^53 integers, inf and NaN.
/// Counts, edges, dtypes and raise type must be numpy's.
#[test]
fn histogram_uniform_bins_edge_placement_grid_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
def outcome(f):
    try:
        c, e = f()
        c, e = np.asarray(c), np.asarray(e)
        return ("ok", c.dtype.str, c.tobytes(), e.dtype.str, e.tobytes())
    except BaseException as ex:
        return (type(ex).__name__, str(ex)[:80])
rng = np.random.default_rng(37)
def data_for(dt, n, kind):
    if kind == "flat":
        if np.dtype(dt).kind == "f":
            return rng.uniform(-3, 3, n).astype(dt)
        info = np.iinfo(dt)
        return rng.integers(info.min, info.max, n, endpoint=True, dtype=dt)
    if kind == "normal":
        v = rng.standard_normal(n) * 50
        if np.dtype(dt).kind == "f":
            return v.astype(dt)
        return np.clip(v, np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)
    if kind == "const":
        return np.full(n, 7, dtype=dt)
    if kind == "two":
        return np.where(rng.integers(0, 2, n) == 0, 0, 100).astype(dt)
    return (rng.integers(0, 11, n) * 10).astype(dt)
cells = 0
bad = []
for dt in ("f8", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"):
    for n in (1, 2, 17, 4096, (1 << 21) + 5):
        for kind in ("flat", "normal", "const", "two", "on_edges"):
            if n > 4096 and kind in ("const", "two"):
                continue
            a = data_for(dt, n, kind)
            for bins in (1, 2, 10, 100, 1000):
                if n > 4096 and bins not in (10, 1000):
                    continue
                cells += 1
                ours = outcome(lambda: fnp.histogram(a, bins=bins))
                theirs = outcome(lambda: np.histogram(a, bins=bins))
                if ours != theirs:
                    bad.append(f"{dt} n={n} {kind} bins={bins}: fnp={ours[:2]} numpy={theirs[:2]}")
specials = {
    "huge range": np.array([-1e308, 1e308]),
    "huge range list": [-1e308, 1e308],
    "tiny range": np.array([1.0, 1.0 + 2**-50, 1.0 + 2**-49]),
    "signed zeros": np.array([0.0, -0.0, -0.0, 0.0, 1.0, -1.0]),
    "i8 above 2^53": np.array([0, 2**53 + 1], dtype=np.int64),
    "u8 extremes": np.array([0, 2**64 - 1], dtype=np.uint64),
    "inf": np.array([1.0, np.inf, 2.0]),
    "nan": np.array([1.0, np.nan, 2.0]),
}
for name, a in specials.items():
    for bins in (1, 3, 10):
        cells += 1
        ours = outcome(lambda: fnp.histogram(a, bins=bins))
        theirs = outcome(lambda: np.histogram(a, bins=bins))
        if ours != theirs:
            bad.append(f"{name} bins={bins}: fnp={ours[:2]} numpy={theirs[:2]}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(2, ' ');
    assert_eq!(
        fields.next().unwrap_or("0"),
        "978",
        "cell table drifted: {result}"
    );
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "histogram must place every value in numpy's bin: {result}"
    );
    Ok(())
}
