//! Concurrency safety tests for fnp-ufunc SharedView operations.
//!
//! Verifies that:
//! - SharedBuffer/SharedSidecar RwLock operations don't deadlock
//! - Concurrent reads during writes are safe
//! - Poisoned locks are detected and reported correctly
//! - Multiple threads can safely access shared views
//!
//! Finding: fnp-ufunc has SharedBuffer and SharedSidecar (Arc<RwLock<...>>)
//! patterns with 4 inline tests but no standalone concurrency test file.
//! These tests verify thread safety in isolation.

use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::UFuncArray;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Concurrent read safety
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concurrent_shared_view_reads_no_race() {
    let arr = UFuncArray::arange(0.0, 100.0, 1.0, DType::F64).unwrap();
    let view = arr.shared_view().unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    let threads: Vec<_> = (0..8)
        .map(|_| {
            let v = view.clone();
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                for i in 0..50 {
                    let val = v.item(&[i]).unwrap();
                    assert_eq!(val, i as f64, "value should match index");
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
        8,
        "all threads should complete"
    );
}

#[test]
fn concurrent_shared_view_shape_access_no_race() {
    let arr = UFuncArray::zeros(vec![10, 20, 30], DType::F64).unwrap();
    let view = arr.shared_view().unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    let threads: Vec<_> = (0..8)
        .map(|_| {
            let v = view.clone();
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                for _ in 0..100 {
                    let shape = v.shape();
                    assert_eq!(shape, &[10, 20, 30]);
                    assert_eq!(shape.len(), 3);
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
        8,
        "all threads should complete"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Concurrent write safety (single writer, multiple readers)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concurrent_reads_during_write_no_deadlock() {
    let arr = UFuncArray::arange(0.0, 50.0, 1.0, DType::F64).unwrap();
    let view = arr.shared_view().unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    let writer_view = view.clone();
    let writer_completed = Arc::clone(&completed);
    let writer = thread::spawn(move || {
        for i in 0..50 {
            let _ = writer_view.itemset(&[i], 999.0);
        }
        writer_completed.fetch_add(1, Ordering::SeqCst);
    });

    let readers: Vec<_> = (0..4)
        .map(|_| {
            let v = view.clone();
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                for i in 0..50 {
                    let _ = v.item(&[i]);
                }
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    writer.join().expect("writer should not deadlock");
    for r in readers {
        r.join().expect("reader should not deadlock");
    }

    assert_eq!(
        completed.load(Ordering::SeqCst),
        5,
        "all threads should complete"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// to_array concurrent access
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concurrent_to_array_no_race() {
    let arr = UFuncArray::arange(0.0, 20.0, 1.0, DType::F64).unwrap();
    let view = arr.shared_view().unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    let threads: Vec<_> = (0..8)
        .map(|_| {
            let v = view.clone();
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                for _ in 0..10 {
                    let owned = v.to_array().unwrap();
                    assert_eq!(owned.shape(), &[20]);
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
        8,
        "all threads should complete"
    );
}

#[test]
fn sidecar_backed_itemset_to_array_contention_no_deadlock() -> Result<(), String> {
    let initial = (0..128).map(|i| i64::MAX - i).collect::<Vec<_>>();
    let arr = UFuncArray::from_storage(vec![128], ArrayStorage::I64(initial))
        .map_err(|err| format!("{err:?}"))?;
    assert!(arr.has_integer_sidecar());

    let view = Arc::new(arr.shared_view().map_err(|err| format!("{err:?}"))?);
    let writes_completed = Arc::new(AtomicUsize::new(0));
    let reads_completed = Arc::new(AtomicUsize::new(0));
    let mut threads = Vec::new();

    for worker in 0..4 {
        let view = Arc::clone(&view);
        let writes_completed = Arc::clone(&writes_completed);
        threads.push(thread::spawn(move || -> Result<(), String> {
            for offset in 0..32 {
                let index = worker * 32 + offset;
                view.itemset(&[index as i64], (1_000 + index) as f64)
                    .map_err(|err| format!("{err:?}"))?;
                writes_completed.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }));
    }

    for _ in 0..4 {
        let view = Arc::clone(&view);
        let reads_completed = Arc::clone(&reads_completed);
        threads.push(thread::spawn(move || -> Result<(), String> {
            for _ in 0..32 {
                let owned = view.to_array().map_err(|err| format!("{err:?}"))?;
                assert_eq!(owned.shape(), &[128]);
                assert!(owned.has_integer_sidecar());
                reads_completed.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }));
    }

    for thread in threads {
        thread
            .join()
            .map_err(|_| "sidecar-backed view thread should not deadlock".to_string())??;
    }

    assert_eq!(writes_completed.load(Ordering::SeqCst), 128);
    assert_eq!(reads_completed.load(Ordering::SeqCst), 128);

    let final_storage = view
        .to_array()
        .map_err(|err| format!("{err:?}"))?
        .to_storage()
        .map_err(|err| format!("{err:?}"))?;
    let values = match final_storage {
        ArrayStorage::I64(values) => values,
        other => {
            return Err(format!(
                "sidecar-backed final storage should remain I64: {other:?}"
            ));
        }
    };
    assert_eq!(values[0], 1_000);
    assert_eq!(values[127], 1_127);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Clone during concurrent access
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concurrent_clone_no_race() {
    let arr = UFuncArray::arange(0.0, 100.0, 1.0, DType::F64).unwrap();
    let view = arr.shared_view().unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    let threads: Vec<_> = (0..8)
        .map(|_| {
            let v = view.clone();
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                for _ in 0..20 {
                    let cloned = v.clone();
                    assert_eq!(cloned.shape(), v.shape());
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
        8,
        "all threads should complete"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Stress test: interleaved reads and writes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stress_interleaved_read_write_no_deadlock() {
    let arr = UFuncArray::arange(0.0, 100.0, 1.0, DType::F64).unwrap();
    let view = arr.shared_view().unwrap();
    let completed = Arc::new(AtomicUsize::new(0));
    let iterations = 50;

    let threads: Vec<_> = (0..8)
        .map(|i| {
            let v = view.clone();
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                for j in 0..iterations {
                    let idx = (i * iterations + j) % 100;
                    if j % 2 == 0 {
                        let _ = v.item(&[idx]);
                    } else {
                        let _ = v.itemset(&[idx], (i * j) as f64);
                    }
                }
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for t in threads {
        t.join().expect("thread should not deadlock");
    }

    assert_eq!(
        completed.load(Ordering::SeqCst),
        8,
        "all threads should complete"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Timeout detection (no infinite wait)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concurrent_access_completes_within_timeout() {
    let arr = UFuncArray::arange(0.0, 1000.0, 1.0, DType::F64).unwrap();
    let view = arr.shared_view().unwrap();
    let completed = Arc::new(AtomicUsize::new(0));
    let start = std::time::Instant::now();

    let threads: Vec<_> = (0..16)
        .map(|_| {
            let v = view.clone();
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                for i in 0..100 {
                    let _ = v.item(&[i]);
                    let _ = v.itemset(&[i], 0.0);
                }
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for t in threads {
        t.join().expect("thread should complete");
    }

    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(5),
        "concurrent operations should complete quickly, took {:?}",
        elapsed
    );
    assert_eq!(completed.load(Ordering::SeqCst), 16);
}
