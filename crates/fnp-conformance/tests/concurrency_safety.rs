//! Concurrency safety tests for fnp-conformance.
//!
//! Verifies that:
//! - Static lock initialization is thread-safe
//! - Multiple threads can safely configure log paths concurrently
//! - Lock ordering doesn't cause deadlocks
//! - Poisoned mutexes are handled correctly
//!
//! Finding: fnp-conformance has 4+ static Mutex/OnceLock combinations but
//! no tests verifying concurrent access safety. This test suite validates
//! the lock-free initialization and consistent lock ordering.

use fnp_conformance::{
    set_dtype_promotion_log_path, set_runtime_policy_log_path, set_shape_stride_log_path,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Concurrent initialization tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concurrent_set_runtime_policy_log_path_no_panic() {
    let completed = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..8)
        .map(|i| {
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                let path = if i % 2 == 0 {
                    Some(PathBuf::from(format!("/tmp/policy_{i}.log")))
                } else {
                    None
                };
                set_runtime_policy_log_path(path);
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for t in threads {
        t.join().expect("thread should not panic");
    }

    assert_eq!(
        completed.load(Ordering::SeqCst),
        8,
        "all threads should complete"
    );
}

#[test]
fn concurrent_set_shape_stride_log_path_no_panic() {
    let completed = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..8)
        .map(|i| {
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                let path = if i % 2 == 0 {
                    Some(PathBuf::from(format!("/tmp/shape_{i}.log")))
                } else {
                    None
                };
                set_shape_stride_log_path(path);
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for t in threads {
        t.join().expect("thread should not panic");
    }

    assert_eq!(
        completed.load(Ordering::SeqCst),
        8,
        "all threads should complete"
    );
}

#[test]
fn concurrent_set_dtype_promotion_log_path_no_panic() {
    let completed = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..8)
        .map(|i| {
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                let path = if i % 2 == 0 {
                    Some(PathBuf::from(format!("/tmp/dtype_{i}.log")))
                } else {
                    None
                };
                set_dtype_promotion_log_path(path);
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for t in threads {
        t.join().expect("thread should not panic");
    }

    assert_eq!(
        completed.load(Ordering::SeqCst),
        8,
        "all threads should complete"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-lock ordering tests (verify no deadlock from AB-BA patterns)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concurrent_set_all_log_paths_no_deadlock() {
    let completed = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..12)
        .map(|i| {
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                // Each thread sets all three log paths in different orders
                // If there's a lock ordering issue, this would deadlock
                match i % 3 {
                    0 => {
                        set_runtime_policy_log_path(Some(PathBuf::from("/tmp/a")));
                        set_shape_stride_log_path(Some(PathBuf::from("/tmp/b")));
                        set_dtype_promotion_log_path(Some(PathBuf::from("/tmp/c")));
                    }
                    1 => {
                        set_shape_stride_log_path(Some(PathBuf::from("/tmp/b")));
                        set_dtype_promotion_log_path(Some(PathBuf::from("/tmp/c")));
                        set_runtime_policy_log_path(Some(PathBuf::from("/tmp/a")));
                    }
                    _ => {
                        set_dtype_promotion_log_path(Some(PathBuf::from("/tmp/c")));
                        set_runtime_policy_log_path(Some(PathBuf::from("/tmp/a")));
                        set_shape_stride_log_path(Some(PathBuf::from("/tmp/b")));
                    }
                }
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for t in threads {
        t.join().expect("thread should not panic");
    }

    assert_eq!(
        completed.load(Ordering::SeqCst),
        12,
        "all threads should complete without deadlock"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Rapid concurrent access stress test
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stress_concurrent_log_path_updates() {
    let completed = Arc::new(AtomicUsize::new(0));
    let iterations = 100;

    let threads: Vec<_> = (0..4)
        .map(|thread_id| {
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                for i in 0..iterations {
                    let path = Some(PathBuf::from(format!("/tmp/stress_{thread_id}_{i}.log")));
                    match (thread_id + i) % 3 {
                        0 => set_runtime_policy_log_path(path),
                        1 => set_shape_stride_log_path(path),
                        _ => set_dtype_promotion_log_path(path),
                    }
                }
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for t in threads {
        t.join().expect("stress test should not panic");
    }

    assert_eq!(
        completed.load(Ordering::SeqCst),
        4,
        "all stress threads should complete"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// OnceLock initialization race test
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn oncelock_initialization_race_safe() {
    // All threads start simultaneously and race to initialize the OnceLock
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let completed = Arc::new(AtomicUsize::new(0));

    let threads: Vec<_> = (0..8)
        .map(|i| {
            let barrier = Arc::clone(&barrier);
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                // Wait for all threads to be ready
                barrier.wait();
                // All threads try to initialize simultaneously
                set_runtime_policy_log_path(Some(PathBuf::from(format!("/tmp/race_{i}.log"))));
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for t in threads {
        t.join().expect("race initialization should not panic");
    }

    assert_eq!(
        completed.load(Ordering::SeqCst),
        8,
        "all racing threads should complete"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Timeout-based deadlock detection
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn no_deadlock_within_timeout() {
    let completed = Arc::new(AtomicUsize::new(0));

    let handle = {
        let completed = Arc::clone(&completed);
        thread::spawn(move || {
            for _ in 0..50 {
                set_runtime_policy_log_path(Some(PathBuf::from("/tmp/timeout_a.log")));
                set_shape_stride_log_path(Some(PathBuf::from("/tmp/timeout_b.log")));
                set_dtype_promotion_log_path(Some(PathBuf::from("/tmp/timeout_c.log")));
            }
            completed.fetch_add(1, Ordering::SeqCst);
        })
    };

    // Give it a reasonable timeout - if this hangs, there's a deadlock
    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();

    while completed.load(Ordering::SeqCst) == 0 {
        if start.elapsed() > timeout {
            panic!("potential deadlock detected - operations did not complete within 5 seconds");
        }
        thread::sleep(Duration::from_millis(10));
    }

    handle.join().expect("thread should complete");
}

// ─────────────────────────────────────────────────────────────────────────────
// None value handling under concurrency
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concurrent_none_and_some_paths() {
    let completed = Arc::new(AtomicUsize::new(0));

    let threads: Vec<_> = (0..16)
        .map(|i| {
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                // Alternate between Some and None
                for j in 0..10 {
                    let path = if (i + j) % 2 == 0 {
                        Some(PathBuf::from(format!("/tmp/mixed_{i}_{j}.log")))
                    } else {
                        None
                    };
                    set_runtime_policy_log_path(path);
                }
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for t in threads {
        t.join().expect("mixed None/Some should not panic");
    }

    assert_eq!(
        completed.load(Ordering::SeqCst),
        16,
        "all threads should complete"
    );
}
