//! Conformance tests for numpy memory utility functions against NumPy oracle.
//!
//! Tests may_share_memory, shares_memory, result_type, and allocation-failure parity
//! (MemoryError, never a dead interpreter).

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
fn memory_utils_python_container_and_keyword_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def normalize(value):
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, np.dtype):
        return ("dtype", str(value))
    return ("other", type(value).__name__, str(value))

def outcome(call_fn, *args, **kwargs):
    try:
        return ("ok", normalize(call_fn(*args, **kwargs)))
    except Exception as exc:
        return ("err", type(exc).__name__)

base = np.arange(12).reshape(3, 4)
rev = base[::-1, ::-1]
even = base[:, ::2]
odd = base[:, 1::2]
broadcast = np.broadcast_to(base[:1, :], (3, 4))

cases = [
    ("may_share reversed view", "may_share_memory", lambda: ((base, rev), {})),
    ("may_share disjoint strided views", "may_share_memory", lambda: ((even, odd), {})),
    (
        "may_share bounded max_work",
        "may_share_memory",
        lambda: ((broadcast, base), {"max_work": 1}),
    ),
    ("shares reversed view", "shares_memory", lambda: ((base, rev), {})),
    ("shares disjoint strided views", "shares_memory", lambda: ((even, odd), {})),
    (
        "shares bounded max_work",
        "shares_memory",
        lambda: ((broadcast, base), {"max_work": 1}),
    ),
    (
        "result_type Python scalar and dtype string",
        "result_type",
        lambda: ((1, "float32"), {}),
    ),
    (
        "result_type array and scalar",
        "result_type",
        lambda: ((np.array([1, 2], dtype=np.int16), 3.5), {}),
    ),
    (
        "result_type bool complex scalar",
        "result_type",
        lambda: ((True, np.complex64(1 + 2j)), {}),
    ),
    ("result_type invalid dtype error", "result_type", lambda: (("not-a-dtype",), {})),
    (
        "shares_memory Python list error",
        "shares_memory",
        lambda: (([1, 2, 3], [1, 2, 3]), {}),
    ),
]

ok = True
for label, name, factory in cases:
    args, kwargs = factory()
    actual = outcome(getattr(fnp, name), *args, **kwargs)
    args, kwargs = factory()
    expected = outcome(getattr(np, name), *args, **kwargs)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "memory utility Python-container and keyword surfaces should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// may_share_memory
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn may_share_memory_same_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.may_share_memory(a, a)
expected = np.may_share_memory(a, a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "may_share_memory same array should match numpy"
    );
    Ok(())
}

#[test]
fn may_share_memory_different_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.may_share_memory(a, b)
expected = np.may_share_memory(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "may_share_memory different arrays should match numpy"
    );
    Ok(())
}

#[test]
fn may_share_memory_view() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
b = a[1:4]  # view of a
result = fnp.may_share_memory(a, b)
expected = np.may_share_memory(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "may_share_memory view should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// shares_memory
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn shares_memory_same_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.shares_memory(a, a)
expected = np.shares_memory(a, a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "shares_memory same array should match numpy"
    );
    Ok(())
}

#[test]
fn shares_memory_different_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.shares_memory(a, b)
expected = np.shares_memory(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "shares_memory different arrays should match numpy"
    );
    Ok(())
}

#[test]
fn shares_memory_view() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
b = a[1:4]  # view of a
result = fnp.shares_memory(a, b)
expected = np.shares_memory(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "shares_memory view should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// result_type
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn result_type_int_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2], dtype='int32')
b = np.array([1.0, 2.0], dtype='float64')
result = fnp.result_type(a, b)
expected = np.result_type(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "result_type int+float should match numpy"
    );
    Ok(())
}

#[test]
fn result_type_dtypes() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.result_type('float32', 'float64')
expected = np.result_type('float32', 'float64')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "result_type dtypes should match numpy"
    );
    Ok(())
}

#[test]
fn result_type_scalars() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.result_type(3, 3.0)
expected = np.result_type(3, 3.0)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "result_type scalars should match numpy"
    );
    Ok(())
}

#[test]
fn result_type_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2], dtype='int8')
b = np.array([1, 2], dtype='int16')
c = np.array([1, 2], dtype='int32')
result = fnp.result_type(a, b, c)
expected = np.result_type(a, b, c)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "result_type multiple should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn shares_vs_may_share_same() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
# For same array, both should be True
shares = fnp.shares_memory(a, a)
may_share = fnp.may_share_memory(a, a)
print(shares == True and may_share == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "both should be True for same array");
    Ok(())
}

#[test]
fn shares_vs_may_share_different() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# For different arrays, both should be False
shares = fnp.shares_memory(a, b)
may_share = fnp.may_share_memory(a, b)
print(shares == False and may_share == False)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "both should be False for different arrays"
    );
    Ok(())
}

/// Allocation failure is numpy's MemoryError, never a dead interpreter (bead .31). Each arm runs
/// in a forked child whose address space is capped at its own size plus 6x a 4 Mi-element
/// operand. On 21 calls numpy fits that cap, but fnp's routes built a float64 copy of a
/// bool/uint8/int8/float16 operand: min/max of bool never reached its bool kernel (a 0-d output
/// buffer yields no slice); histogram, the four set ops, extract, logical_not,
/// isposinf/isneginf, convolve/correlate, ediff1d, count_nonzero and bitwise_count extracted to
/// float64; roll extracted an array shift into a Vec<i64>. Those aborted the process (SIGABRT
/// from Rust's alloc-error handler) or raised MemoryError. On 4 calls numpy itself raises
/// MemoryError, and fnp aborted there too. fnp must match numpy's outcome and result bytes. A
/// build of 95809bc2, before the fix, fails 18 of the 25: 15 by SIGABRT and 3 with MemoryError
/// where numpy succeeds.
#[test]
fn allocation_failure_is_memory_error_and_never_kills_the_interpreter() -> Result<(), String> {
    let script = fnp_script(
        r#"
import ctypes, hashlib, os, resource, signal, warnings
warnings.simplefilter("ignore")
# The cap must measure the CALL, not allocator/thread-pool bookkeeping: one malloc arena (glibc
# reserves 64 MiB of address space per extra arena) and a small rayon pool.
try:
    ctypes.CDLL(None).mallopt(-8, 1)  # M_ARENA_MAX
except Exception:
    pass
os.environ["RAYON_NUM_THREADS"] = "4"
N = 1 << 22
BUDGET = 6 * N
rng = np.random.default_rng(31)
u1 = rng.integers(0, 200, N, dtype=np.uint8)
i1 = rng.integers(-100, 100, N, dtype=np.int8)
f2 = rng.standard_normal(N).astype(np.float16)
b = rng.random(N) < 0.5
CASES = [
    # numpy fits the cap; fnp's former routes built a float64 copy (4-8x) and aborted
    ("min bool", lambda m: m.min(b)),
    ("max bool", lambda m: m.max(b)),
    ("amax bool keepdims", lambda m: m.amax(b, keepdims=True)),
    ("histogram bool", lambda m: m.histogram(b)),
    ("setxor1d bool", lambda m: m.setxor1d(b, b[::-1])),
    ("setdiff1d bool", lambda m: m.setdiff1d(b, b[::-1])),
    ("intersect1d bool", lambda m: m.intersect1d(b, b[::-1])),
    ("extract bool strided", lambda m: m.extract(b, b[::-1])),
    ("logical_not uint8", lambda m: m.logical_not(u1)),
    ("logical_not int8", lambda m: m.logical_not(i1)),
    ("logical_not float16", lambda m: m.logical_not(f2)),
    ("isposinf float16", lambda m: m.isposinf(f2)),
    ("isneginf float16", lambda m: m.isneginf(f2)),
    ("convolve bool", lambda m: m.convolve(b, b[:64])),
    ("correlate float16", lambda m: m.correlate(f2, f2[:64])),
    ("ediff1d uint8 to_end", lambda m: m.ediff1d(u1, to_end=u1[:3])),
    ("roll uint8 array shift", lambda m: m.roll(u1, u1[::-1])),
    ("count_nonzero float16", lambda m: m.count_nonzero(f2)),
    ("bitwise_count bool", lambda m: m.bitwise_count(b)),
    ("kron float16", lambda m: m.kron(f2[:2048], f2[:2048])),
    ("outer float16", lambda m: m.outer(f2[:2048], f2[:2048])),
    # numpy itself raises MemoryError here; fnp must too, never die by a signal
    ("vstack uint8 1-D", lambda m: m.vstack(u1)),
    ("tile uint8", lambda m: m.tile(u1, u1[::-1])),
    ("triu uint8 1-D", lambda m: m.triu(u1)),
    ("union1d bool", lambda m: m.union1d(b, b[::-1])),
]
def vm_bytes():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024
def digest(value):
    parts = value if isinstance(value, tuple) else (value,)
    h = hashlib.sha256()
    for part in parts:
        arr = np.asarray(part)
        h.update(f"{type(part).__name__}|{arr.dtype.str}|{arr.shape}|".encode())
        h.update(arr.tobytes())
    return h.hexdigest()
def outcome(module, fn):
    rd, wr = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(rd)
        cap = vm_bytes() + BUDGET
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        signal.alarm(120)
        try:
            text = "ok " + digest(fn(module))
        except MemoryError:
            text = "MemoryError"
        except BaseException as ex:  # PanicException is a BaseException
            text = "raised " + type(ex).__name__
        os.write(wr, text.encode())
        os._exit(0)
    os.close(wr)
    with os.fdopen(rd) as pipe:
        text = pipe.read()
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status):
        return "signal " + signal.Signals(os.WTERMSIG(status)).name
    return text
bad = []
survived = memory_errors = 0
for name, fn in CASES:
    s = outcome(np, fn)
    r = outcome(fnp, fn)
    survived += s.startswith("ok")
    memory_errors += s == "MemoryError"
    if r != s:
        bad.append(f"{name}: fnp={r[:40]} numpy={s[:40]}")
print(len(CASES), survived, memory_errors, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(4, ' ');
    let (cases, survived, memory_errors, bad) = (
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or("0"),
        fields.next().unwrap_or(""),
    );
    assert_eq!(cases, "25", "case table drifted: {result}");
    // Both outcome classes must be exercised, or the cap is mis-sized and the test is vacuous.
    assert!(
        survived.parse::<usize>().unwrap_or(0) >= 16,
        "numpy should fit the cap on most cells: {result}"
    );
    assert!(
        memory_errors.parse::<usize>().unwrap_or(0) >= 2,
        "numpy should raise MemoryError on the oversized cells: {result}"
    );
    assert_eq!(bad, "[]", "allocation failure must match numpy: {result}");
    Ok(())
}
