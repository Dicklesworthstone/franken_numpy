//! Metamorphic tests for fnp-random RNG implementations.
//!
//! Tests RNG invariants that must hold regardless of seed:
//! - advance(n) equivalence to n sequential calls
//! - bounded output range constraints
//! - f64 output in [0, 1) interval
//! - fill length correctness
//! - seed distinctness (different seeds → different sequences)
//! - spawn distinctness (spawned children → different sequences)
//! - state roundtrip preservation
//!
//! Finding: fnp-random had only golden tests. RNGs have rich metamorphic
//! properties that can catch subtle bugs in state advancement and output mapping.

use fnp_random::{Pcg64DxsmRng, Pcg64Rng, SeedSequence};

// ─────────────────────────────────────────────────────────────────────────────
// MR1: advance(n) ≡ n sequential next() calls
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_advance_equivalence_pcg64dxsm_small() {
    for seed in [0u64, 1, 42, 12345, u64::MAX] {
        let mut rng_seq = Pcg64DxsmRng::from_u64_seed(seed).unwrap();
        let mut rng_adv = Pcg64DxsmRng::from_u64_seed(seed).unwrap();

        for _ in 0..100 {
            let _ = rng_seq.next_u64();
        }
        rng_adv.advance(100);

        let seq_vals: Vec<u64> = (0..10).map(|_| rng_seq.next_u64()).collect();
        let adv_vals: Vec<u64> = (0..10).map(|_| rng_adv.next_u64()).collect();

        assert_eq!(seq_vals, adv_vals, "advance(100) should match 100 sequential calls for seed {seed}");
    }
}

#[test]
fn mr_advance_equivalence_pcg64dxsm_large() {
    let mut rng_seq = Pcg64DxsmRng::from_u64_seed(999).unwrap();
    let mut rng_adv = Pcg64DxsmRng::from_u64_seed(999).unwrap();

    for _ in 0..10000 {
        let _ = rng_seq.next_u64();
    }
    rng_adv.advance(10000);

    let seq_vals: Vec<u64> = (0..5).map(|_| rng_seq.next_u64()).collect();
    let adv_vals: Vec<u64> = (0..5).map(|_| rng_adv.next_u64()).collect();

    assert_eq!(seq_vals, adv_vals, "advance(10000) should match 10000 sequential calls");
}

#[test]
fn mr_advance_zero_is_noop() {
    let mut rng1 = Pcg64DxsmRng::from_u64_seed(42).unwrap();
    let mut rng2 = Pcg64DxsmRng::from_u64_seed(42).unwrap();

    rng1.advance(0);

    let vals1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
    let vals2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();

    assert_eq!(vals1, vals2, "advance(0) should be a no-op");
}

// ─────────────────────────────────────────────────────────────────────────────
// MR2: bounded_u64(bound) < bound (always)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_bounded_u64_always_less_than_bound() {
    let mut rng = Pcg64DxsmRng::from_u64_seed(12345).unwrap();

    for bound in [1u64, 2, 10, 100, 1000, 1_000_000, u64::MAX / 2, u64::MAX] {
        for _ in 0..1000 {
            let value = rng.bounded_u64(bound).unwrap();
            assert!(
                value < bound,
                "bounded_u64({bound}) returned {value} which is >= bound"
            );
        }
    }
}

#[test]
fn mr_bounded_u64_covers_range() {
    let mut rng = Pcg64DxsmRng::from_u64_seed(7777).unwrap();
    let bound = 10u64;
    let n = 10000;

    let mut seen = [false; 10];
    for _ in 0..n {
        let v = rng.bounded_u64(bound).unwrap() as usize;
        seen[v] = true;
    }

    assert!(
        seen.iter().all(|&s| s),
        "bounded_u64(10) should eventually produce all values 0..10"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR3: next_f64() ∈ [0, 1)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_next_f64_in_unit_interval_pcg64() {
    let mut rng = Pcg64Rng::from_u64_seed(54321).unwrap();

    for _ in 0..10000 {
        let v = rng.next_f64();
        assert!(v >= 0.0, "next_f64() returned {v} < 0.0");
        assert!(v < 1.0, "next_f64() returned {v} >= 1.0");
    }
}

#[test]
fn mr_next_f64_in_unit_interval_pcg64dxsm() {
    let mut rng = Pcg64DxsmRng::from_u64_seed(54321).unwrap();

    for _ in 0..10000 {
        let v = rng.next_f64();
        assert!(v >= 0.0, "next_f64() returned {v} < 0.0");
        assert!(v < 1.0, "next_f64() returned {v} >= 1.0");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR4: fill_u64(n).len() == n
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_fill_u64_length_correct() {
    let mut rng = Pcg64DxsmRng::from_u64_seed(111).unwrap();

    for len in [0, 1, 10, 100, 1000] {
        let filled = rng.fill_u64(len);
        assert_eq!(filled.len(), len, "fill_u64({len}) should return exactly {len} elements");
    }
}

#[test]
fn mr_fill_u64_deterministic() {
    let mut rng1 = Pcg64DxsmRng::from_u64_seed(222).unwrap();
    let mut rng2 = Pcg64DxsmRng::from_u64_seed(222).unwrap();

    let fill1 = rng1.fill_u64(100);
    let fill2 = rng2.fill_u64(100);

    assert_eq!(fill1, fill2, "fill_u64 should be deterministic for same seed");
}

// ─────────────────────────────────────────────────────────────────────────────
// MR5: Different seeds → different sequences (statistical)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_different_seeds_produce_different_sequences() {
    for (seed1, seed2) in [(0, 1), (42, 43), (100, 200), (u64::MAX - 1, u64::MAX)] {
        let mut rng1 = Pcg64DxsmRng::from_u64_seed(seed1).unwrap();
        let mut rng2 = Pcg64DxsmRng::from_u64_seed(seed2).unwrap();

        let vals1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let vals2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();

        assert_ne!(
            vals1, vals2,
            "seeds {seed1} and {seed2} should produce different sequences"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR6: Spawned children → distinct sequences
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_spawn_produces_distinct_children() {
    let entropy = [123u32, 456, 789, 0];
    let mut parent = SeedSequence::new(&entropy).expect("parent");

    let children = parent.spawn(5).expect("spawn 5 children");

    let mut sequences: Vec<Vec<u64>> = Vec::new();
    for child in &children {
        let mut rng = Pcg64DxsmRng::from_seed_sequence(child).expect("rng from child");
        let seq: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();
        sequences.push(seq);
    }

    for i in 0..sequences.len() {
        for j in (i + 1)..sequences.len() {
            assert_ne!(
                sequences[i], sequences[j],
                "spawned children {i} and {j} should produce distinct sequences"
            );
        }
    }
}

#[test]
fn mr_spawn_is_deterministic() {
    let entropy = [999u32, 888, 777, 666];

    let mut parent1 = SeedSequence::new(&entropy).expect("parent1");
    let mut parent2 = SeedSequence::new(&entropy).expect("parent2");

    let children1 = parent1.spawn(3).expect("spawn from parent1");
    let children2 = parent2.spawn(3).expect("spawn from parent2");

    for (c1, c2) in children1.iter().zip(children2.iter()) {
        let mut rng1 = Pcg64DxsmRng::from_seed_sequence(c1).expect("rng1");
        let mut rng2 = Pcg64DxsmRng::from_seed_sequence(c2).expect("rng2");

        let seq1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let seq2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();

        assert_eq!(seq1, seq2, "spawn should be deterministic");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR7: State roundtrip preservation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_state_roundtrip_preserves_sequence() {
    let mut rng = Pcg64DxsmRng::from_u64_seed(42).unwrap();

    for _ in 0..500 {
        let _ = rng.next_u64();
    }

    let state = rng.to_state_entries();
    let expected: Vec<u64> = (0..20).map(|_| rng.next_u64()).collect();

    let mut restored = Pcg64DxsmRng::from_state_entries(&state).expect("restore");
    let actual: Vec<u64> = (0..20).map(|_| restored.next_u64()).collect();

    assert_eq!(expected, actual, "state roundtrip should preserve RNG position");
}

#[test]
fn mr_state_roundtrip_at_initial_position() {
    let rng = Pcg64DxsmRng::from_u64_seed(12345).unwrap();

    let state = rng.to_state_entries();
    let mut restored = Pcg64DxsmRng::from_state_entries(&state).expect("restore");
    let mut original = Pcg64DxsmRng::from_u64_seed(12345).unwrap();

    let restored_vals: Vec<u64> = (0..10).map(|_| restored.next_u64()).collect();
    let original_vals: Vec<u64> = (0..10).map(|_| original.next_u64()).collect();

    assert_eq!(restored_vals, original_vals);
}

// ─────────────────────────────────────────────────────────────────────────────
// MR8: Composition - advance then fill equals sequential fill
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_advance_then_fill_equals_skip_then_fill() {
    let mut rng1 = Pcg64DxsmRng::from_u64_seed(333).unwrap();
    let mut rng2 = Pcg64DxsmRng::from_u64_seed(333).unwrap();

    // Method 1: advance then fill
    rng1.advance(50);
    let fill1 = rng1.fill_u64(20);

    // Method 2: sequential skip then fill
    for _ in 0..50 {
        let _ = rng2.next_u64();
    }
    let fill2 = rng2.fill_u64(20);

    assert_eq!(fill1, fill2, "advance then fill should equal sequential skip then fill");
}

// ─────────────────────────────────────────────────────────────────────────────
// MR9: Pcg64 vs Pcg64Dxsm with same seed produce different sequences
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_pcg64_and_pcg64dxsm_are_distinct() {
    let seed = 42u64;
    let mut pcg64 = Pcg64Rng::from_u64_seed(seed).unwrap();
    let mut pcg64dxsm = Pcg64DxsmRng::from_u64_seed(seed).unwrap();

    let vals1: Vec<u64> = (0..10).map(|_| pcg64.next_u64()).collect();
    let vals2: Vec<u64> = (0..10).map(|_| pcg64dxsm.next_u64()).collect();

    assert_ne!(
        vals1, vals2,
        "Pcg64 and Pcg64Dxsm should produce different sequences even with same seed"
    );
}
