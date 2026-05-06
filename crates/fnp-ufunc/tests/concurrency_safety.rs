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

// ─────────────────────────────────────────────────────────────────────────────
// View creation operations under concurrency
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concurrent_slice_axis_view_creation_no_race() {
    let arr = UFuncArray::arange(0.0, 100.0, 1.0, DType::F64).unwrap();
    let view = arr.shared_view().unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    let threads: Vec<_> = (0..8)
        .map(|i| {
            let v = view.clone();
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                for _ in 0..20 {
                    let start = (i * 10) as i64;
                    let stop = start + 10;
                    let sliced = v.slice_axis(0, Some(start), Some(stop), 1).unwrap();
                    assert_eq!(sliced.shape(), &[10]);
                    let _ = sliced.item(&[0]).unwrap();
                }
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for t in threads {
        t.join().expect("slice_axis thread should not panic");
    }

    assert_eq!(completed.load(Ordering::SeqCst), 8);
}

#[test]
fn concurrent_nested_slice_operations_no_deadlock() {
    let arr = UFuncArray::zeros(vec![100], DType::F64).unwrap();
    let view = arr.shared_view().unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    let threads: Vec<_> = (0..4)
        .map(|_| {
            let v = view.clone();
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                for _ in 0..10 {
                    let slice1 = v.slice_axis(0, Some(0), Some(80), 1).unwrap();
                    let slice2 = slice1.slice_axis(0, Some(10), Some(50), 1).unwrap();
                    let slice3 = slice2.slice_axis(0, Some(5), Some(30), 2).unwrap();
                    assert!(slice3.shape()[0] > 0);
                }
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for t in threads {
        t.join().expect("nested slice thread should not deadlock");
    }

    assert_eq!(completed.load(Ordering::SeqCst), 4);
}

#[test]
fn concurrent_view_creation_with_read_write_no_race() {
    let arr = UFuncArray::arange(0.0, 200.0, 1.0, DType::F64).unwrap();
    let view = arr.shared_view().unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    let threads: Vec<_> = (0..8)
        .map(|i| {
            let v = view.clone();
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                for j in 0..20 {
                    match (i + j) % 3 {
                        0 => {
                            let sliced = v.slice_axis(0, Some(0), Some(100), 1).unwrap();
                            let _ = sliced.item(&[j % 100]);
                        }
                        1 => {
                            let _ = v.item(&[((i * 20 + j) % 200) as i64]);
                        }
                        _ => {
                            let _ = v.itemset(&[((i * 20 + j) % 200) as i64], 999.0);
                        }
                    }
                }
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for t in threads {
        t.join().expect("mixed operations should not race");
    }

    assert_eq!(completed.load(Ordering::SeqCst), 8);
}

#[test]
fn concurrent_view_to_array_with_slice_no_race() {
    let arr = UFuncArray::arange(0.0, 50.0, 1.0, DType::F64).unwrap();
    let view = arr.shared_view().unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    let threads: Vec<_> = (0..6)
        .map(|i| {
            let v = view.clone();
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                for _ in 0..10 {
                    let sliced = v.slice_axis(0, Some((i * 5) as i64), Some(((i + 1) * 5) as i64), 1).unwrap();
                    let owned = sliced.to_array().unwrap();
                    assert_eq!(owned.shape(), &[5]);
                }
                completed.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for t in threads {
        t.join().expect("slice+to_array should not race");
    }

    assert_eq!(completed.load(Ordering::SeqCst), 6);
}

// ─────────────────────────────────────────────────────────────────────────────
// Data consistency: buffer and sidecar must be read atomically
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn to_array_buffer_sidecar_consistency_under_contention() {
    // Tests that to_array() returns arrays where buffer[i] and sidecar[i] for
    // the SAME index come from the same point in time.
    //
    // Without the fix: to_array read buffer (release), then read sidecar.
    // An itemset between these reads could update both, causing buffer[i] = old
    // but sidecar[i] = new for the returned array.
    //
    // With the fix: to_array holds both locks, so buffer[i] and sidecar[i]
    // are always from the same snapshot.
    //
    // Strategy: use one large value (i64::MAX) to force sidecar creation, then
    // test small values at other indices where f64 IS exact.

    // Element 0 is large (forces sidecar), elements 1-63 are small (f64-exact)
    let mut initial: Vec<i64> = (1..64).map(|i| 1_000_000 + i).collect();
    initial.insert(0, i64::MAX); // Force sidecar creation
    let arr = UFuncArray::from_storage(vec![64], ArrayStorage::I64(initial))
        .expect("create sidecar-backed array");
    assert!(arr.has_integer_sidecar());

    let view = Arc::new(arr.shared_view().expect("shared view"));
    let inconsistencies = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();

    // Writers: update indices 1-63 with exact-representable values
    // (leave index 0 as i64::MAX to keep sidecar alive)
    for writer_id in 0..4i64 {
        let view = Arc::clone(&view);
        handles.push(thread::spawn(move || {
            for round in 0..500i64 {
                let value = (2_000_000 + writer_id * 100_000 + round) as f64;
                for i in 1..64i64 {
                    let _ = view.itemset(&[i], value);
                }
            }
        }));
    }

    // Readers: call to_array and verify buffer[i] == sidecar[i] for indices 1-63
    for _ in 0..4 {
        let view = Arc::clone(&view);
        let inconsistencies = Arc::clone(&inconsistencies);
        handles.push(thread::spawn(move || {
            for _ in 0..500 {
                let owned = view.to_array().expect("to_array");
                let storage = owned.to_storage().expect("to_storage");
                let sidecar_vals = match storage {
                    ArrayStorage::I64(v) => v,
                    _ => panic!("expected I64 storage"),
                };

                // buffer[i] as i64 should match sidecar[i] for small-integer indices
                for i in 1..64 {
                    let buffer_val = owned.item(&[i as i64]).expect("item");
                    let sidecar_val = sidecar_vals[i];
                    if buffer_val as i64 != sidecar_val {
                        inconsistencies.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread should complete");
    }

    let total_inconsistencies = inconsistencies.load(Ordering::SeqCst);
    assert_eq!(
        total_inconsistencies, 0,
        "to_array returned {} buffer/sidecar mismatches — race condition!",
        total_inconsistencies
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// itemset_cow atomic clone: buffer and sidecar must be cloned atomically
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn itemset_cow_atomic_clone_no_buffer_sidecar_skew() {
    // Regression test for non-atomic clone bug in itemset_cow.
    //
    // Before the fix: itemset_cow cloned buffer (releasing lock), then cloned
    // sidecar separately. A concurrent itemset between the two clones could
    // update both buffer and sidecar atomically, causing the COW copy to have
    // buffer from time T1 but sidecar from time T2 — an inconsistent snapshot.
    //
    // After the fix: itemset_cow holds both locks (sidecar first, buffer second)
    // during the clone operation, ensuring atomicity.
    //
    // Strategy: create a shared view with sidecar (requires at least one value
    // outside f64 exact range to force sidecar creation), spawn writers that call
    // itemset on the shared view while another thread calls itemset_cow to
    // create a private copy. The private copy's to_array should always return
    // consistent buffer/sidecar values.

    // Include one large value (i64::MAX) to force sidecar creation, rest are small
    let mut initial: Vec<i64> = (1..64).map(|i| 1_000_000 + i).collect();
    initial.insert(0, i64::MAX);
    let arr = UFuncArray::from_storage(vec![64], ArrayStorage::I64(initial))
        .expect("create sidecar-backed array");
    assert!(arr.has_integer_sidecar(), "need sidecar for this test");

    let shared_view = Arc::new(arr.shared_view().expect("shared view"));
    let inconsistencies = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();

    // Writers: continuously modify indices 1-63 (skip index 0 which has i64::MAX)
    for writer_id in 0..2i64 {
        let view = Arc::clone(&shared_view);
        handles.push(thread::spawn(move || {
            for round in 0..1000i64 {
                let value = (2_000_000 + writer_id * 100_000 + round) as f64;
                for i in 1..64i64 {
                    let _ = view.itemset(&[i], value);
                }
            }
        }));
    }

    // COW readers: call itemset_cow to create private copies, then verify consistency
    for _ in 0..4 {
        let view = Arc::clone(&shared_view);
        let inconsistencies = Arc::clone(&inconsistencies);
        handles.push(thread::spawn(move || {
            for _ in 0..200 {
                // Clone the shared view to get our own view struct (not the Arc data)
                let mut cow_view = (*view).clone();

                // Trigger COW clone by writing index 1 (leave 0 alone, it's i64::MAX)
                if cow_view.itemset_cow(&[1], 9_999_999.0).is_err() {
                    continue;
                }

                // The private copy should be internally consistent
                let owned = match cow_view.to_array() {
                    Ok(a) => a,
                    Err(_) => continue,
                };
                let storage = match owned.to_storage() {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let sidecar_vals = match storage {
                    ArrayStorage::I64(v) => v,
                    _ => continue,
                };

                // Check consistency: buffer[i] as i64 should match sidecar[i]
                // for indices 2-63 (skip 0=i64::MAX anchor, 1=our write)
                for i in 2..64 {
                    let buffer_val = match owned.item(&[i as i64]) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let sidecar_val = sidecar_vals[i];
                    if buffer_val as i64 != sidecar_val {
                        inconsistencies.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread should complete");
    }

    let total_inconsistencies = inconsistencies.load(Ordering::SeqCst);
    assert_eq!(
        total_inconsistencies, 0,
        "itemset_cow produced {} buffer/sidecar mismatches — non-atomic clone race!",
        total_inconsistencies
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Poisoned lock detection
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn poisoned_buffer_lock_detected_via_thread_panic() {
    use std::sync::RwLock;

    let buffer = Arc::new(RwLock::new(vec![1.0, 2.0, 3.0]));
    let buffer_clone = Arc::clone(&buffer);

    let handle = thread::spawn(move || {
        let _guard = buffer_clone.write().unwrap();
        panic!("intentional panic to poison lock");
    });

    let _ = handle.join();

    assert!(buffer.is_poisoned(), "lock should be poisoned after thread panic");

    let read_result = buffer.read();
    assert!(read_result.is_err(), "read on poisoned lock should return Err");

    let write_result = buffer.write();
    assert!(write_result.is_err(), "write on poisoned lock should return Err");
}

#[test]
fn shared_view_recovers_from_thread_panic() {
    let arr = UFuncArray::arange(0.0, 10.0, 1.0, DType::F64).unwrap();
    let view = arr.shared_view().unwrap();
    let view_for_panic = view.clone();

    let handle = thread::spawn(move || {
        let _val = view_for_panic.item(&[0]).unwrap();
        panic!("intentional panic while using view");
    });

    let _ = handle.join();

    let result = view.item(&[0]);
    match result {
        Ok(val) => assert_eq!(val, 0.0, "view still usable after sibling thread panic"),
        Err(err) => {
            let msg = format!("{err:?}");
            assert!(
                msg.contains("poison") || msg.contains("Poison"),
                "if error, should mention poisoning: {msg}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lock ordering verification
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lock_ordering_sidecar_then_buffer_no_deadlock() {
    let initial: Vec<i64> = (0..100).map(|i| i64::MAX - i).collect();
    let arr = UFuncArray::from_storage(vec![100], ArrayStorage::I64(initial))
        .expect("create sidecar-backed array");
    assert!(arr.has_integer_sidecar(), "need sidecar for lock ordering test");

    let view = Arc::new(arr.shared_view().expect("shared view"));
    let completed = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();

    for _ in 0..4 {
        let view = Arc::clone(&view);
        let completed = Arc::clone(&completed);
        handles.push(thread::spawn(move || {
            for i in 0..50i64 {
                let _ = view.itemset(&[i], 999.0);
            }
            completed.fetch_add(1, Ordering::SeqCst);
        }));
    }

    for _ in 0..4 {
        let view = Arc::clone(&view);
        let completed = Arc::clone(&completed);
        handles.push(thread::spawn(move || {
            for _ in 0..50 {
                let _ = view.to_array();
            }
            completed.fetch_add(1, Ordering::SeqCst);
        }));
    }

    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();
    for h in handles {
        let remaining = timeout.saturating_sub(start.elapsed());
        if remaining.is_zero() {
            panic!("lock ordering test timed out — potential deadlock");
        }
        h.join().expect("thread should complete without deadlock");
    }

    assert_eq!(
        completed.load(Ordering::SeqCst),
        8,
        "all threads should complete — lock ordering is correct"
    );
}
