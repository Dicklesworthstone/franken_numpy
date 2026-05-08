//! Golden artifact tests for fnp-random RNG implementations.
//!
//! These tests verify deterministic output for fixed seeds, enabling
//! regression testing without external oracles.

use fnp_random::{Pcg64DxsmRng, Pcg64Rng, SeedSequence};

// ─────────────────────────────────────────────────────────────────────────────
// PCG64 deterministic output
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_pcg64_seed_42_deterministic() {
    let mut rng1 = Pcg64Rng::from_u64_seed(42).expect("seed should succeed");
    let mut rng2 = Pcg64Rng::from_u64_seed(42).expect("seed should succeed");

    let first_run: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
    let second_run: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();

    assert_eq!(
        first_run, second_run,
        "PCG64 should be deterministic for same seed"
    );
}

#[test]
fn golden_pcg64_seed_0_distinct_from_seed_1() {
    let mut rng0 = Pcg64Rng::from_u64_seed(0).expect("seed 0");
    let mut rng1 = Pcg64Rng::from_u64_seed(1).expect("seed 1");

    let vals0: Vec<u64> = (0..5).map(|_| rng0.next_u64()).collect();
    let vals1: Vec<u64> = (0..5).map(|_| rng1.next_u64()).collect();

    assert_ne!(
        vals0, vals1,
        "different seeds should produce different sequences"
    );
}

#[test]
fn golden_pcg64_f64_in_unit_interval() {
    let mut rng = Pcg64Rng::from_u64_seed(12345).expect("seed");
    for _ in 0..1000 {
        let v = rng.next_f64();
        assert!((0.0..1.0).contains(&v), "f64 output {v} not in [0, 1)");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PCG64DXSM deterministic output
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_pcg64dxsm_seed_42_deterministic() {
    let mut rng1 = Pcg64DxsmRng::from_u64_seed(42).expect("seed");
    let mut rng2 = Pcg64DxsmRng::from_u64_seed(42).expect("seed");

    let run1: Vec<u64> = (0..20).map(|_| rng1.next_u64()).collect();
    let run2: Vec<u64> = (0..20).map(|_| rng2.next_u64()).collect();

    assert_eq!(run1, run2, "PCG64DXSM should be deterministic");
}

#[test]
fn golden_pcg64dxsm_state_roundtrip() {
    let mut rng = Pcg64DxsmRng::from_u64_seed(999).expect("seed");

    // Advance the RNG a bit
    for _ in 0..100 {
        let _ = rng.next_u64();
    }

    // Capture state
    let state_entries = rng.to_state_entries();

    // Generate some values
    let expected: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();

    // Restore from state
    let mut restored =
        Pcg64DxsmRng::from_state_entries(&state_entries).expect("state restoration should succeed");
    let actual: Vec<u64> = (0..10).map(|_| restored.next_u64()).collect();

    assert_eq!(
        expected, actual,
        "state roundtrip should preserve RNG position"
    );
}

#[test]
fn golden_pcg64dxsm_bounded_u64_in_range() {
    let mut rng = Pcg64DxsmRng::from_u64_seed(7777).expect("seed");

    for bound in [10u64, 100, 1000, 1_000_000] {
        for _ in 0..100 {
            let v = rng.bounded_u64(bound).expect("bounded should succeed");
            assert!(v < bound, "bounded_u64({bound}) returned {v} >= bound");
        }
    }
}

#[test]
fn golden_pcg64dxsm_bounded_u64_uniform_distribution() {
    let mut rng = Pcg64DxsmRng::from_u64_seed(54321).expect("seed");
    let bound = 10u64;
    let n = 10000;

    let mut counts = [0u64; 10];
    for _ in 0..n {
        let v = rng.bounded_u64(bound).expect("bounded") as usize;
        counts[v] += 1;
    }

    // Each bucket should have ~n/10 = 1000 samples
    // Allow 30% deviation for statistical noise
    let expected = n as f64 / bound as f64;
    for (i, &count) in counts.iter().enumerate() {
        let ratio = count as f64 / expected;
        assert!(
            ratio > 0.7 && ratio < 1.3,
            "bucket {i} has {count} samples, expected ~{expected}, ratio={ratio}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SeedSequence
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_seed_sequence_deterministic() {
    let entropy = [42u32, 0, 0, 0];
    let ss1 = SeedSequence::new(&entropy).expect("seed seq 1");
    let ss2 = SeedSequence::new(&entropy).expect("seed seq 2");

    let mut rng1 = Pcg64DxsmRng::from_seed_sequence(&ss1).expect("rng1");
    let mut rng2 = Pcg64DxsmRng::from_seed_sequence(&ss2).expect("rng2");

    let run1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
    let run2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();

    assert_eq!(run1, run2, "SeedSequence should produce deterministic RNG");
}

#[test]
fn golden_seed_sequence_spawn_children_distinct() {
    let entropy = [123u32, 456, 789, 0];
    let mut parent = SeedSequence::new(&entropy).expect("parent");

    let children = parent.spawn(2).expect("spawn 2 children");

    let mut rng1 = Pcg64DxsmRng::from_seed_sequence(&children[0]).expect("rng1");
    let mut rng2 = Pcg64DxsmRng::from_seed_sequence(&children[1]).expect("rng2");

    let run1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
    let run2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();

    assert_ne!(
        run1, run2,
        "spawned children should produce distinct sequences"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Fill operations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_fill_u64_correct_length() {
    let mut rng = Pcg64DxsmRng::from_u64_seed(999).expect("seed");

    for len in [0, 1, 10, 100, 1000] {
        let filled = rng.fill_u64(len);
        assert_eq!(
            filled.len(),
            len,
            "fill_u64({len}) should return {len} elements"
        );
    }
}

#[test]
fn golden_fill_u64_deterministic() {
    let mut rng1 = Pcg64DxsmRng::from_u64_seed(888).expect("seed");
    let mut rng2 = Pcg64DxsmRng::from_u64_seed(888).expect("seed");

    let fill1 = rng1.fill_u64(100);
    let fill2 = rng2.fill_u64(100);

    assert_eq!(fill1, fill2, "fill_u64 should be deterministic");
}

// ─────────────────────────────────────────────────────────────────────────────
// Advance/jump operations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_pcg64dxsm_advance_matches_sequential() {
    let mut rng_seq = Pcg64DxsmRng::from_u64_seed(555).expect("seed");
    let mut rng_adv = Pcg64DxsmRng::from_u64_seed(555).expect("seed");

    // Advance sequentially
    for _ in 0..1000 {
        let _ = rng_seq.next_u64();
    }

    // Advance in one jump
    rng_adv.advance(1000);

    // Both should now produce the same sequence
    let seq_vals: Vec<u64> = (0..10).map(|_| rng_seq.next_u64()).collect();
    let adv_vals: Vec<u64> = (0..10).map(|_| rng_adv.next_u64()).collect();

    assert_eq!(
        seq_vals, adv_vals,
        "advance(1000) should match 1000 sequential calls"
    );
}
