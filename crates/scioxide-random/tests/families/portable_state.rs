//! Portable reference fixtures and adversarial checkpoint/identity contracts.
//! The same tests run with either scalar width, std/no_std, and every target.
#![allow(clippy::unwrap_used)]

use scioxide_core::{InvalidInputReason, SciOxideError};
use scioxide_random::{
    BitGenerator, JumpableRng, Lcg64, Mrg32k3a, Pcg64, Rng, SeedableRng, SerializableState,
    SkipAhead, SplitMix64, SplittableRng, Xoshiro128StarStar, Xoshiro256StarStar,
};

fn unhex(text: &str) -> Vec<u8> {
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn pure_decoders_preserve_endpoints_and_discard_only_documented_bits() {
    use scioxide_random::{decode_u32, decode_u64};
    assert_eq!(decode_u64::<f32>(0).to_bits(), 0);
    assert_eq!(decode_u64::<f64>(0).to_bits(), 0);
    assert_eq!(decode_u32::<f32>(0).to_bits(), 0);
    assert_eq!(decode_u32::<f64>(0).to_bits(), 0);
    assert_eq!(decode_u64::<f32>(u64::MAX).to_bits(), 0x3f7f_ffff);
    assert_eq!(decode_u64::<f64>(u64::MAX).to_bits(), 0x3fef_ffff_ffff_ffff);
    assert_eq!(decode_u32::<f32>(u32::MAX).to_bits(), 0x3f7f_ffff);
    assert_eq!(decode_u32::<f64>(u32::MAX).to_bits(), 0x3fef_ffff_ffe0_0000);
    assert_eq!(decode_u64::<f32>((1 << 40) - 1).to_bits(), 0);
    assert_eq!(decode_u64::<f32>(1 << 40).to_bits(), 0x3380_0000); // 2^-24
    assert_eq!(decode_u64::<f64>((1 << 11) - 1).to_bits(), 0);
    assert_eq!(decode_u64::<f64>(1 << 11).to_bits(), 0x3ca0_0000_0000_0000); // 2^-53
    assert_eq!(decode_u32::<f32>(255).to_bits(), 0);
    assert_eq!(decode_u32::<f32>(256).to_bits(), 0x3380_0000);
    assert_eq!(decode_u32::<f64>(1).to_bits(), 0x3df0_0000_0000_0000); // 2^-32
}
fn fixture<B>(engine: B, columns: &[&str])
where
    B: BitGenerator + SerializableState + Clone,
    B: Rng<f32> + Rng<f64>,
{
    let bytes = unhex(columns[3]);
    assert_eq!(
        engine.export_state().unwrap(),
        bytes,
        "{} {} {}",
        columns[0],
        columns[1],
        columns[2]
    );
    let restored = B::import_state(&bytes).unwrap();
    assert_eq!(engine.stream_identity(), restored.stream_identity());
    assert_eq!(engine.position(), restored.position());
    assert_eq!(restored.export_state().unwrap(), bytes);
    let mut raw = restored.clone();
    let mut f32_rng = restored.clone();
    let mut f64_rng = restored;
    for expected in columns[4].split(',') {
        assert_eq!(
            raw.next_u64().unwrap(),
            u64::from_str_radix(expected, 16).unwrap()
        );
    }
    for expected in columns[5].split(',') {
        let value: f32 = f32_rng.next_float().unwrap();
        assert_eq!(value.to_bits(), u32::from_str_radix(expected, 16).unwrap());
    }
    for expected in columns[6].split(',') {
        let value: f64 = f64_rng.next_float().unwrap();
        assert_eq!(value.to_bits(), u64::from_str_radix(expected, 16).unwrap());
    }
}
fn advanced<B: SeedableRng + BitGenerator>(seed: u64) -> B {
    let mut engine = B::from_seed(seed);
    (0..3).for_each(|_| {
        engine.next_u64().unwrap();
    });
    engine
}
fn jumped<B: SeedableRng + JumpableRng>(seed: u64, operation: &str) -> B {
    let mut engine = B::from_seed(seed);
    if operation == "jump" {
        engine.jump().unwrap();
    } else {
        engine.long_jump().unwrap();
    }
    engine
}
fn selected<B: SeedableRng + SplittableRng>(seed: u64, operation: &str) -> B {
    let (stream, sequence) = if operation == "substream-max" {
        (u64::MAX, u64::MAX)
    } else {
        (1, 2)
    };
    B::from_seed(seed).substream(stream, sequence).unwrap()
}

#[test]
fn independent_integer_matrix_and_ieee_vectors_match_every_engine() {
    for line in include_str!("../data/portable_state.tsv")
        .lines()
        .filter(|line| !line.starts_with("//"))
    {
        let columns: Vec<_> = line.split('\t').collect();
        let seed = u64::from_str_radix(columns[1], 16).unwrap();
        let operation = columns[2];
        if operation == "three-u64" {
            match columns[0] {
                "1" => fixture(advanced::<SplitMix64>(seed), &columns),
                "2" => fixture(advanced::<Xoshiro256StarStar>(seed), &columns),
                "3" => fixture(advanced::<Xoshiro128StarStar>(seed), &columns),
                "4" => fixture(advanced::<Pcg64>(seed), &columns),
                "5" => fixture(advanced::<Lcg64>(seed), &columns),
                "6" => fixture(advanced::<Mrg32k3a>(seed), &columns),
                _ => assert_eq!(columns[0], "known engine"),
            }
        } else {
            match columns[0] {
                "2" => fixture(jumped::<Xoshiro256StarStar>(seed, operation), &columns),
                "3" => fixture(jumped::<Xoshiro128StarStar>(seed, operation), &columns),
                "4" => fixture(selected::<Pcg64>(seed, operation), &columns),
                "6" => fixture(selected::<Mrg32k3a>(seed, operation), &columns),
                _ => assert_eq!(columns[0], "engine with movement"),
            }
        }
    }
}

fn assert_invalid<T>(result: Result<T, SciOxideError>, reason: InvalidInputReason) {
    let error = result.err().unwrap();
    assert!(
        matches!(error.view(), scioxide_core::SciOxideErrorView::InvalidInput { reason: actual, .. } if actual == reason)
    );
}
// Independent bit-serial CRC lets structural tests get past integrity checking.
fn recheck(bytes: &mut [u8]) {
    let end = bytes.len() - 4;
    let mut crc = u32::MAX;
    for byte in &bytes[..end] {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 == 0 { 0 } else { 0x82f6_3b78 };
        }
    }
    bytes[end..].copy_from_slice(&(!crc).to_le_bytes());
}

fn adversarial<B: SeedableRng + SerializableState + Clone>() {
    let mut engine = advanced::<B>(42);
    let good = engine.export_state().unwrap();
    for length in 0..good.len() {
        assert_invalid(
            engine.restore_state(&good[..length]),
            InvalidInputReason::Malformed,
        );
        assert_eq!(engine.export_state().unwrap(), good);
    }
    for bit in 0..good.len() * 8 {
        let mut corrupt = good.clone();
        corrupt[bit / 8] ^= 1 << (bit % 8);
        assert!(
            engine.restore_state(&corrupt).is_err(),
            "accepted corrupt bit {bit}"
        );
        assert_eq!(engine.export_state().unwrap(), good);
    }
    let mut trailing = good.clone();
    trailing.push(0);
    assert_invalid(B::import_state(&trailing), InvalidInputReason::Malformed);
    assert_invalid(
        B::import_state(&good[16..good.len() - 4]),
        InvalidInputReason::Malformed,
    );
    let mut old = b"GRNG".to_vec();
    old.push(good[6]);
    old.extend_from_slice(&good[16..good.len() - 4]);
    assert_invalid(B::import_state(&old), InvalidInputReason::Malformed);
    for index in [4, 7, 8] {
        let mut invalid = good.clone();
        invalid[index] = 2;
        recheck(&mut invalid);
        assert_invalid(
            B::import_state(&invalid),
            InvalidInputReason::UnsupportedConfiguration,
        );
    }
    let mut unknown = good.clone();
    unknown[6] = 255;
    recheck(&mut unknown);
    assert_invalid(
        B::import_state(&unknown),
        InvalidInputReason::UnsupportedConfiguration,
    );
    for index in [9, 10, 11] {
        let mut invalid = good.clone();
        invalid[index] |= 128;
        recheck(&mut invalid);
        assert_invalid(B::import_state(&invalid), InvalidInputReason::Malformed);
    }
    let mut inconsistent = good.clone();
    inconsistent[9] &= !1;
    recheck(&mut inconsistent);
    assert_invalid(
        B::import_state(&inconsistent),
        InvalidInputReason::Malformed,
    );
    let foreign = B::from_seed(43).export_state().unwrap();
    assert_invalid(
        engine.restore_state(&foreign),
        InvalidInputReason::Inconsistent,
    );
    assert_eq!(engine.export_state().unwrap(), good);
    assert_eq!(
        engine.next_u64().unwrap(),
        B::import_state(&good).unwrap().next_u64().unwrap()
    );
    for size in [0, B::STATE_BYTES - 1, B::STATE_BYTES + 1] {
        let mut output = vec![0xa5; size];
        let error = engine.export_state_into(&mut output).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(output, vec![0xa5; size]);
    }
    let mut output = vec![0; B::STATE_BYTES];
    engine.export_state_into(&mut output).unwrap();
    assert_eq!(output, engine.export_state().unwrap());
    engine.restore_state(&good).unwrap();
    assert_eq!(engine.export_state().unwrap(), good);
}

#[test]
fn all_engines_reject_invalid_checkpoints_transactionally() {
    adversarial::<SplitMix64>();
    adversarial::<Lcg64>();
    adversarial::<Pcg64>();
    adversarial::<Xoshiro128StarStar>();
    adversarial::<Xoshiro256StarStar>();
    adversarial::<Mrg32k3a>();
    let states = [
        SplitMix64::from_seed(42).export_state().unwrap(),
        Xoshiro256StarStar::from_seed(42).export_state().unwrap(),
        Xoshiro128StarStar::from_seed(42).export_state().unwrap(),
        Pcg64::from_seed(42).export_state().unwrap(),
        Lcg64::from_seed(42).export_state().unwrap(),
        Mrg32k3a::from_seed(42).export_state().unwrap(),
    ];
    type Importer = fn(&[u8]) -> Result<(), SciOxideError>;
    let imports: [Importer; 6] = [
        |s| SplitMix64::import_state(s).map(drop),
        |s| Xoshiro256StarStar::import_state(s).map(drop),
        |s| Xoshiro128StarStar::import_state(s).map(drop),
        |s| Pcg64::import_state(s).map(drop),
        |s| Lcg64::import_state(s).map(drop),
        |s| Mrg32k3a::import_state(s).map(drop),
    ];
    for (i, state) in states.iter().enumerate() {
        for (j, import) in imports.iter().enumerate().filter(|(j, _)| *j != i) {
            assert_invalid(import(state), InvalidInputReason::Inconsistent);
            assert_ne!(i, j);
        }
    }
}

#[test]
fn valid_checksums_do_not_hide_noncanonical_engine_invariants() {
    for (bytes, word_bytes) in [
        (
            Xoshiro128StarStar::from_seed(42).export_state().unwrap(),
            16,
        ),
        (
            Xoshiro256StarStar::from_seed(42).export_state().unwrap(),
            32,
        ),
    ] {
        for start in [16, 16 + word_bytes] {
            let mut bad = bytes.clone();
            bad[start..start + word_bytes].fill(0);
            recheck(&mut bad);
            if word_bytes == 16 {
                assert_invalid(
                    Xoshiro128StarStar::import_state(&bad),
                    InvalidInputReason::Malformed,
                );
            } else {
                assert_invalid(
                    Xoshiro256StarStar::import_state(&bad),
                    InvalidInputReason::Malformed,
                );
            }
        }
    }
    let bytes = Mrg32k3a::from_seed(42).export_state().unwrap();
    for start in [16, 40, 64, 88] {
        let mut bad = bytes.clone();
        bad[start..start + 24].fill(0);
        recheck(&mut bad);
        assert_invalid(Mrg32k3a::import_state(&bad), InvalidInputReason::Malformed);
        let mut bad = bytes.clone();
        let modulus = if start == 16 || start == 64 {
            4294967087_u64
        } else {
            4294944443_u64
        };
        bad[start..start + 8].copy_from_slice(&modulus.to_le_bytes());
        recheck(&mut bad);
        assert_invalid(Mrg32k3a::import_state(&bad), InvalidInputReason::Malformed);
    }
    let mut bad = Pcg64::from_seed(42)
        .split(9)
        .unwrap()
        .export_state()
        .unwrap();
    bad[48] ^= 2;
    recheck(&mut bad); // odd increment, inconsistent with stream selector
    assert_invalid(Pcg64::import_state(&bad), InvalidInputReason::Malformed);
}

fn identity<B: SeedableRng + SerializableState + SplittableRng + SkipAhead + Clone>() {
    let mut base = B::from_seed(42);
    let identity_base = base.clone();
    let root = identity_base.stream_identity();
    let child = base.substream(5, 7).unwrap();
    let checkpoint = child.export_state().unwrap();
    base.skip_ahead(123).unwrap();
    base.fill_bytes(&mut [0; 17]).unwrap();
    assert_eq!(base.stream_identity(), root);
    assert_eq!(
        base.substream(5, 7).unwrap().export_state().unwrap(),
        checkpoint
    );
    let mut unrelated = base.substream(19, 23).unwrap();
    unrelated.next_u64().unwrap();
    assert_eq!(
        unrelated.substream(5, 7).unwrap().export_state().unwrap(),
        checkpoint
    );
    let mut restored = B::import_state(&checkpoint).unwrap();
    restored.next_u64().unwrap();
    restored.restore_state(&checkpoint).unwrap();
    assert_eq!(restored.position(), Some(0));
    assert_eq!(restored.stream_identity(), child.stream_identity());
    assert_invalid(
        restored.restore_state(&base.split(5).unwrap().export_state().unwrap()),
        InvalidInputReason::Inconsistent,
    );
}
#[test]
fn stream_identity_survives_parent_progress_and_restore() {
    identity::<Pcg64>();
    identity::<Mrg32k3a>();
}

#[test]
fn mixed_scalar_and_raw_calls_replay_through_the_same_snapshot() {
    let mut rng = Xoshiro128StarStar::from_seed(42);
    let _: f32 = rng.next_float().unwrap();
    rng.fill_bytes(&mut [0; 7]).unwrap();
    let _: f64 = rng.next_float().unwrap();
    let bytes = rng.export_state().unwrap();
    let mut replay = Xoshiro128StarStar::import_state(&bytes).unwrap();
    let a: f64 = rng.next_open_float().unwrap();
    let b: f64 = replay.next_open_float().unwrap();
    assert_eq!(a.to_bits(), b.to_bits());
    assert_eq!(rng.next_u64().unwrap(), replay.next_u64().unwrap());
    let a: f32 = rng.next_float().unwrap();
    let b: f32 = replay.next_float().unwrap();
    assert_eq!(a.to_bits(), b.to_bits());
    assert_eq!(rng.export_state().unwrap(), replay.export_state().unwrap());
}

#[test]
fn entropy_is_acquired_only_by_the_explicit_seed_operation() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    struct CountingEntropy(AtomicUsize);
    impl scioxide_core::EntropySource for CountingEntropy {
        fn fill_bytes(&self, output: &mut [u8]) -> Result<(), scioxide_core::CapabilityError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            assert_eq!(output.len(), 8);
            output.copy_from_slice(&42_u64.to_le_bytes());
            Ok(())
        }
    }
    fn check<B: SeedableRng + SerializableState + Clone>(source: &CountingEntropy) {
        let before = source.0.load(Ordering::SeqCst);
        let mut engine = B::seed_from_entropy(source).unwrap();
        assert_eq!(source.0.load(Ordering::SeqCst), before + 1);
        let identity_root = B::from_seed(42);
        let identity = identity_root.stream_identity();
        let checkpoint = engine.export_state().unwrap();
        engine.next_u64().unwrap();
        let mut restored = B::import_state(&checkpoint).unwrap();
        engine.restore_state(&checkpoint).unwrap();
        assert_eq!(engine.stream_identity(), identity);
        assert_eq!(restored.next_u64().unwrap(), engine.next_u64().unwrap());
        assert_eq!(source.0.load(Ordering::SeqCst), before + 1);
    }
    let source = CountingEntropy(AtomicUsize::new(0));
    check::<SplitMix64>(&source);
    check::<Lcg64>(&source);
    check::<Pcg64>(&source);
    check::<Xoshiro128StarStar>(&source);
    check::<Xoshiro256StarStar>(&source);
    check::<Mrg32k3a>(&source);
}

#[test]
fn full_width_and_untracked_positions_survive_export() {
    fn check<B: SeedableRng + SkipAhead + SerializableState>() {
        let mut engine = B::from_seed(42);
        let identity_root = B::from_seed(42);
        let identity = identity_root.stream_identity();
        engine.skip_ahead(u128::from(u64::MAX) + 1).unwrap();
        assert_eq!(engine.position(), Some(1_u128 << 64));
        let restored = B::import_state(&engine.export_state().unwrap()).unwrap();
        assert_eq!(restored.position(), engine.position());
        engine.skip_ahead(u128::MAX).unwrap();
        assert_eq!(engine.position(), None);
        let mut restored = B::import_state(&engine.export_state().unwrap()).unwrap();
        assert_eq!(restored.position(), None);
        assert_eq!(restored.stream_identity(), identity);
        assert_eq!(restored.next_u64().unwrap(), engine.next_u64().unwrap());
        assert!(restored.skip_to(0).is_err());
    }
    check::<SplitMix64>();
    check::<Lcg64>();
    check::<Pcg64>();
    check::<Mrg32k3a>();
}

#[test]
fn native_draws_carry_full_width_positions_and_preserve_untracked_status() {
    fn check<B: SeedableRng + SerializableState>() {
        for position in [u128::from(u64::MAX), u128::MAX - 1, u128::MAX] {
            let mut bytes = B::from_seed(42).export_state().unwrap();
            let end = bytes.len() - 4;
            bytes[end - 16..end].copy_from_slice(&position.to_le_bytes());
            recheck(&mut bytes);
            let mut engine = B::import_state(&bytes).unwrap();
            for count in 1..=3 {
                engine.next_u32().unwrap();
                assert_eq!(engine.position(), position.checked_add(count));
                let replay = B::import_state(&engine.export_state().unwrap()).unwrap();
                assert_eq!(replay.position(), engine.position());
            }
        }
    }
    check::<SplitMix64>();
    check::<Lcg64>();
    check::<Pcg64>();
    check::<Xoshiro128StarStar>();
    check::<Xoshiro256StarStar>();
    check::<Mrg32k3a>();
}
