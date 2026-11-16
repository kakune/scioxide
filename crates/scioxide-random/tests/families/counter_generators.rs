//! Published Random123 answers and exact finite-stream protocol boundaries.
#![allow(clippy::unwrap_used)]

use scioxide_core::{RealScalar, SciOxideError};
use scioxide_random::{
    BitGenerator, CountDistribution, Distribution, JumpableRng, OutputPolicy, Philox4x64, Rng,
    SamplingPolicy, SamplingWorkspace, SeedableRng, SerializableState, SkipAhead, SplittableRng,
    Threefry4x64, Uniform, philox4x64_block, threefry4x64_block,
};

#[test]
fn published_round_known_answers() {
    let mut cases = 0;
    for line in include_str!("../data/counter_reference.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        let words: Vec<_> = fields[2..]
            .iter()
            .map(|w| u64::from_str_radix(w, 16).unwrap())
            .collect();
        let counter = words[..4].try_into().unwrap();
        let (actual, expected) = match fields[0] {
            "philox4x64" => {
                assert_eq!(fields[1], "10");
                (
                    philox4x64_block(counter, words[4..6].try_into().unwrap()),
                    &words[6..],
                )
            }
            "threefry4x64" => {
                assert_eq!(fields[1], "20");
                (
                    threefry4x64_block(counter, words[4..8].try_into().unwrap()),
                    &words[8..],
                )
            }
            other => unreachable!("unexpected fixture {other}"),
        };
        assert_eq!(actual, expected, "{}", fields[0]);
        cases += 1;
    }
    assert_eq!(cases, 6);
}

// Independent bitwise CRC for malformed-but-checksummed test payloads.
fn checksum(bytes: &mut [u8]) {
    let end = bytes.len() - 4;
    let mut crc = u32::MAX;
    for &byte in &bytes[..end] {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 != 0 { 0x82f6_3b78 } else { 0 };
        }
    }
    bytes[end..].copy_from_slice(&(!crc).to_le_bytes());
}

macro_rules! engine_contract {
    ($module:ident, $engine:ident, $block:ident, $n:expr) => {
        mod $module {
            use super::*;

            #[test]
            fn independent_seeded_words_decoders_and_state_bytes() {
                let name = if $n == 2 {
                    "philox4x64"
                } else {
                    "threefry4x64"
                };
                let row = include_str!("../data/counter_streams.tsv")
                    .lines()
                    .find(|row| row.starts_with(name))
                    .unwrap();
                let expected: Vec<_> = row
                    .split_whitespace()
                    .skip(1)
                    .map(|word| u64::from_str_radix(word, 16).unwrap())
                    .collect();
                let mut engine = $engine::from_seed(42);
                let mut singles = engine.clone();
                let mut doubles = engine.clone();
                let mut sums = [0_u64; 3];
                for _ in 0..1024 {
                    sums[0] ^= engine.next_u64().unwrap();
                    let single: f32 = singles.next_float().unwrap();
                    let double: f64 = doubles.next_float().unwrap();
                    sums[1] ^= u64::from(single.to_bits());
                    sums[2] ^= double.to_bits();
                }
                assert_eq!(sums, expected[..3]);
                let base = engine
                    .substream(13, 29)
                    .unwrap()
                    .derive(&[0, 255, 4])
                    .unwrap();
                for (index, position) in [0, 1, 2, 3, 4, 7, 1024].into_iter().enumerate() {
                    let mut selected = base.clone();
                    selected.seek(position).unwrap();
                    assert_eq!(selected.next_u64().unwrap(), expected[index + 3]);
                }
                let row = include_str!("../data/counter_state.tsv")
                    .lines()
                    .find(|row| row.starts_with(name))
                    .unwrap();
                let hex = row.split_whitespace().nth(1).unwrap();
                let bytes: Vec<_> = hex
                    .as_bytes()
                    .chunks_exact(2)
                    .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
                    .collect();
                let mut engine = $engine::from_seed(42);
                engine.seek(3).unwrap();
                assert_eq!(engine.export_state().unwrap(), bytes);
                let mut restored = $engine::import_state(&bytes).unwrap();
                assert_eq!(restored.next_u64().unwrap(), engine.next_u64().unwrap());
            }

            #[test]
            fn lanes_mixed_widths_bytes_and_checkpoints() {
                let key = core::array::from_fn(|i| i as u64 + 37);
                let base = $engine::from_key(key)
                    .substream(13, 29)
                    .unwrap()
                    .derive(&[0, 255, 4])
                    .unwrap();
                let path = base.path_code();
                assert_eq!(path, 0x0100_ff04);
                for position in 0..13 {
                    let mut engine = base.clone();
                    engine.seek(position).unwrap();
                    let expected =
                        $block([(position / 4) as u64, path, 13, 29], key)[(position % 4) as usize];
                    assert_eq!(engine.clone().next_u64().unwrap(), expected);
                    assert_eq!(engine.clone().next_u32().unwrap(), (expected >> 32) as u32);
                    let checkpoint = engine.export_state().unwrap();
                    let mut restored = $engine::import_state(&checkpoint).unwrap();
                    assert_eq!(restored.export_state().unwrap(), checkpoint);
                    assert_eq!(restored.stream_identity(), engine.stream_identity());
                    for _ in 0..8 {
                        let a: f32 = engine.next_float().unwrap();
                        let b: f32 = restored.next_float().unwrap();
                        assert_eq!(a.to_bits(), b.to_bits());
                        let a: f64 = engine.next_float().unwrap();
                        let b: f64 = restored.next_float().unwrap();
                        assert_eq!(a.to_bits(), b.to_bits());
                    }
                }
                for size in [0, 1, 7, 8, 9, 31, 32, 33, 127] {
                    let mut engine = base.clone();
                    let mut expected = base.clone();
                    let mut bytes = vec![0; size];
                    engine.fill_bytes(&mut bytes).unwrap();
                    let words: Vec<_> = (0..size.div_ceil(8))
                        .flat_map(|_| expected.next_u64().unwrap().to_le_bytes())
                        .collect();
                    assert_eq!(bytes, words[..size]);
                    assert_eq!(engine.position(), expected.position());
                }
            }

            #[test]
            fn exhaustion_and_movement_are_transactional() {
                let mut engine = $engine::from_seed(42);
                let limit = $engine::WORD_CAPACITY;
                let identity_engine = engine.clone();
                let identity = identity_engine.stream_identity();
                for remaining in 0..=9 {
                    engine.seek(limit - remaining).unwrap();
                    let before = engine.export_state().unwrap();
                    let mut bytes = vec![0x5a; remaining as usize * 8 + 1];
                    let error = engine.fill_bytes(&mut bytes).unwrap_err();
                    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::OutOfBounds);
                    assert!(bytes.iter().all(|&b| b == 0x5a));
                    assert_eq!(engine.export_state().unwrap(), before);
                    assert!(engine.skip_ahead(remaining + 1).is_err());
                    assert!(engine.skip_ahead(u128::MAX).is_err());
                    assert!(engine.jump().is_err());
                    assert!(engine.long_jump().is_err());
                    assert_eq!(engine.export_state().unwrap(), before);
                    for position in limit - remaining..limit {
                        let counter = engine.counter().unwrap();
                        let expected = $block(counter, *engine.key())[(position % 4) as usize];
                        assert_eq!(engine.next_u64().unwrap(), expected);
                    }
                    assert_eq!(engine.position(), Some(limit));
                    assert_eq!(engine.counter(), None);
                    assert_eq!(engine.stream_identity(), identity);
                    let exhausted = engine.export_state().unwrap();
                    let error = engine.next_u64().unwrap_err();
                    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::OutOfBounds);
                    assert!(engine.next_u32().is_err());
                    assert!(<_ as Rng<f32>>::next_float(&mut engine).is_err());
                    assert!(<_ as Rng<f64>>::next_float(&mut engine).is_err());
                    engine.fill_bytes(&mut []).unwrap();
                    engine.skip_ahead(0).unwrap();
                    assert_eq!(engine.export_state().unwrap(), exhausted);
                    let mut restored = $engine::import_state(&exhausted).unwrap();
                    assert!(restored.next_u64().is_err());
                    assert_eq!(restored.export_state().unwrap(), exhausted);
                }
                engine.seek(3).unwrap();
                engine.jump().unwrap();
                assert_eq!(engine.position(), Some(3 + (1 << 32)));
                engine.long_jump().unwrap();
                assert_eq!(engine.position(), Some(3 + (1 << 32) + (1 << 48)));
                let before = engine.export_state().unwrap();
                assert!(engine.skip_to(0).is_err());
                assert!(engine.seek(limit + 1).is_err());
                assert_eq!(engine.export_state().unwrap(), before);
                engine.seek(0).unwrap();
                let mut sequential = engine.clone();
                for index in 0..20 {
                    engine.skip_to(index).unwrap();
                    assert_eq!(
                        engine.clone().next_u64().unwrap(),
                        sequential.next_u64().unwrap()
                    );
                }
            }

            #[test]
            fn exact_tree_identity_and_absolute_selection() {
                let mut root = $engine::from_seed(42);
                let original = root.export_state().unwrap();
                let child = root.fork(0).unwrap().fork(255).unwrap();
                assert_eq!(
                    child.export_state().unwrap(),
                    root.derive(&[0, 255]).unwrap().export_state().unwrap()
                );
                assert_eq!(root.export_state().unwrap(), original);
                root.skip_ahead(999).unwrap();
                assert_eq!(
                    child.export_state().unwrap(),
                    root.derive(&[0, 255]).unwrap().export_state().unwrap()
                );
                assert_eq!(
                    child.derive(&[1]).unwrap().export_state().unwrap(),
                    root.derive(&[1]).unwrap().export_state().unwrap()
                );
                assert_eq!(
                    child.substream(4, 5).unwrap().export_state().unwrap(),
                    root.substream(4, 5).unwrap().export_state().unwrap()
                );
                assert_eq!(
                    root.split(4).unwrap().export_state().unwrap(),
                    root.substream(4, 0).unwrap().export_state().unwrap()
                );
                let deepest = root.derive(&[255; 7]).unwrap();
                assert!(deepest.fork(0).is_err());
                assert!(root.derive(&[0; 8]).is_err());
                let mut identities = std::collections::BTreeSet::new();
                identities.insert(root.path_code());
                for first in 0..=255 {
                    let child = root.fork(first).unwrap();
                    assert!(identities.insert(child.path_code()));
                    for second in 0..=255 {
                        assert!(identities.insert(child.fork(second).unwrap().path_code()));
                    }
                }
            }

            #[test]
            fn malformed_state_and_foreign_restore_do_not_commit() {
                let mut engine = $engine::from_seed(19)
                    .substream(7, 8)
                    .unwrap()
                    .fork(9)
                    .unwrap();
                engine.seek(7).unwrap();
                let original = engine.export_state().unwrap();
                for index in 0..original.len() {
                    let mut bad = original.clone();
                    bad[index] ^= 1;
                    assert!(engine.restore_state(&bad).is_err());
                    assert_eq!(engine.export_state().unwrap(), original);
                }
                for len in 0..original.len() {
                    assert!($engine::import_state(&original[..len]).is_err());
                }
                let mut trailing = original.clone();
                trailing.push(0);
                assert!($engine::import_state(&trailing).is_err());
                for path in [0_u64, 2, 0x0200, u64::MAX] {
                    let mut bad = original.clone();
                    let offset = 16 + 8 * ($n + 2);
                    bad[offset..offset + 8].copy_from_slice(&path.to_le_bytes());
                    checksum(&mut bad);
                    assert!($engine::import_state(&bad).is_err());
                }
                let mut bad = original.clone();
                let offset = 16 + 8 * ($n + 3);
                bad[offset..offset + 16]
                    .copy_from_slice(&($engine::WORD_CAPACITY + 1).to_le_bytes());
                checksum(&mut bad);
                assert!($engine::import_state(&bad).is_err());
                for (offset, byte) in [(4, 2), (6, 255), (7, 2), (8, 2), (9, 1), (10, 1)] {
                    let mut bad = original.clone();
                    bad[offset] = byte;
                    checksum(&mut bad);
                    assert!($engine::import_state(&bad).is_err());
                }
                let foreign = engine.fork(0).unwrap().export_state().unwrap();
                assert!(engine.restore_state(&foreign).is_err());
                assert!(
                    engine
                        .restore_state(&$engine::from_seed(20).export_state().unwrap())
                        .is_err()
                );
                assert_eq!(engine.export_state().unwrap(), original);
                let mut wrong_size = vec![0x5a; original.len() + 1];
                assert!(engine.export_state_into(&mut wrong_size).is_err());
                assert!(wrong_size.iter().all(|&b| b == 0x5a));
            }

            fn sampling_failure<S: RealScalar>()
            where
                $engine: Rng<S>,
            {
                let distribution = Uniform::new(S::ZERO, S::ONE).unwrap();
                for rollback in [false, true] {
                    for policy in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
                        for remaining in 0..4 {
                            let mut engine = $engine::from_seed(42);
                            engine.seek($engine::WORD_CAPACITY - remaining).unwrap();
                            let mut rng = engine;
                            let before = rng.export_state().unwrap();
                            let mut output = [-S::ONE; 4];
                            let mut workspace = SamplingWorkspace::new();
                            let result = if rollback {
                                distribution.sample_n_into(
                                    &mut rng,
                                    SamplingPolicy::rollback(policy),
                                    &mut output,
                                    &mut workspace,
                                )
                            } else {
                                distribution.sample_n_into(
                                    &mut rng,
                                    SamplingPolicy::advance_on_error(policy),
                                    &mut output,
                                    &mut workspace,
                                )
                            };
                            let error = result.unwrap_err();
                            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::OutOfBounds);
                            if rollback {
                                assert_eq!(rng.export_state().unwrap(), before);
                            } else {
                                assert_eq!(rng.position(), Some($engine::WORD_CAPACITY));
                            }
                            let published = if policy == OutputPolicy::Atomic {
                                0
                            } else {
                                remaining as usize
                            };
                            assert!(
                                output[..published]
                                    .iter()
                                    .all(|&x| x >= S::ZERO && x < S::ONE)
                            );
                            assert!(output[published..].iter().all(|&x| x == -S::ONE));
                        }
                    }
                }
            }
            #[test]
            fn sampling_policies_propagate_exhaustion() {
                sampling_failure::<f32>();
                sampling_failure::<f64>();
            }
        }
    };
}

engine_contract!(philox, Philox4x64, philox4x64_block, 2);
engine_contract!(threefry, Threefry4x64, threefry4x64_block, 4);

#[test]
fn algorithm_identity_is_checked() {
    let philox = Philox4x64::from_seed(0).export_state().unwrap();
    let threefry = Threefry4x64::from_seed(0).export_state().unwrap();
    assert!(Philox4x64::import_state(&threefry).is_err());
    assert!(Threefry4x64::import_state(&philox).is_err());
}

#[test]
fn raw_provider_errors_reach_existing_samplers() {
    struct Failed;
    fn failure() -> SciOxideError {
        SciOxideError::capability_failure(scioxide_core::Capability::Entropy, "offline")
    }
    impl BitGenerator for Failed {
        #[inline]
        fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.next_u32()
        }
        #[inline]
        fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.next_u64()
        }
        fn next_u32(&mut self) -> Result<u32, SciOxideError> {
            Err(failure())
        }
        fn next_u64(&mut self) -> Result<u64, SciOxideError> {
            Err(failure())
        }
        fn fill_bytes(&mut self, _: &mut [u8]) -> Result<(), SciOxideError> {
            Err(failure())
        }
    }
    impl Rng<f64> for Failed {
        fn next_float(&mut self) -> Result<f64, SciOxideError> {
            Err(failure())
        }
    }
    let mut provider = Failed;
    let error = Uniform::new(0.0, 1.0)
        .unwrap()
        .sample(&mut provider)
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::CapabilityFailure
    );
    for n in [10, 100, 1000] {
        let error = scioxide_random::Binomial::new(n, 0.25_f64)
            .unwrap()
            .prepare_sampler()
            .unwrap()
            .sample(&mut provider)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::CapabilityFailure
        );
    }
    for rate in [1.0_f64, 100.0] {
        let error = scioxide_random::Poisson::new(rate)
            .unwrap()
            .prepare_sampler()
            .unwrap()
            .sample(&mut provider)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::CapabilityFailure
        );
    }
    let mut output = [1, 2, 3];
    let error = scioxide_random::shuffle_in_place(
        &mut output,
        &mut provider,
        SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
        &mut SamplingWorkspace::new(),
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::CapabilityFailure
    );
    assert_eq!(output, [1, 2, 3]);
}
