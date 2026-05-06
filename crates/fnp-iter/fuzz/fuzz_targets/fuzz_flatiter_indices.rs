#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use fnp_iter::FlatIterIndex;

#[derive(Debug, Arbitrary)]
enum FuzzFlatIterIndex {
    Single(u16),
    Slice { start: u16, stop: u16, step: u8 },
    Fancy(Vec<u16>),
    BoolMask(Vec<bool>),
}

impl FuzzFlatIterIndex {
    fn to_flat_iter_index(&self) -> FlatIterIndex {
        match self {
            Self::Single(i) => FlatIterIndex::Single(*i as usize),
            Self::Slice { start, stop, step } => FlatIterIndex::Slice {
                start: *start as usize,
                stop: *stop as usize,
                step: (*step).max(1) as usize,
            },
            Self::Fancy(indices) => {
                FlatIterIndex::Fancy(indices.iter().map(|&i| i as usize).collect())
            }
            Self::BoolMask(mask) => FlatIterIndex::BoolMask(mask.clone()),
        }
    }
}

#[derive(Debug, Arbitrary)]
struct FlatIterInput {
    len: u16,
    index: FuzzFlatIterIndex,
}

fuzz_target!(|input: FlatIterInput| {
    let len = input.len as usize;
    let index = input.index.to_flat_iter_index();

    if let FlatIterIndex::BoolMask(ref mask) = index {
        if mask.len() > 10_000 {
            return;
        }
    }
    if let FlatIterIndex::Fancy(ref indices) = index {
        if indices.len() > 10_000 {
            return;
        }
    }

    let result = fnp_iter::resolve_flatiter_indices(len, &index);

    if let Ok(resolved) = result {
        for &idx in &resolved {
            assert!(idx < len, "resolved index must be < len");
        }

        match &index {
            FlatIterIndex::Single(i) => {
                if *i < len {
                    assert_eq!(resolved.len(), 1);
                    assert_eq!(resolved[0], *i);
                }
            }
            FlatIterIndex::Slice { start, stop, step } => {
                if *start < len && *stop <= len && *step > 0 {
                    let mut expected = Vec::new();
                    let mut i = *start;
                    while i < *stop && i < len {
                        expected.push(i);
                        i += step;
                    }
                    assert_eq!(resolved, expected, "slice resolution mismatch");
                }
            }
            FlatIterIndex::Fancy(indices) => {
                if indices.iter().all(|&i| i < len) {
                    assert_eq!(resolved.len(), indices.len());
                }
            }
            FlatIterIndex::BoolMask(mask) => {
                if mask.len() == len {
                    let expected_count = mask.iter().filter(|&&b| b).count();
                    assert_eq!(resolved.len(), expected_count);
                }
            }
        }
    }
});
