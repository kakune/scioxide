//! Shared movement and native-byte protocols across engine families.
#![allow(clippy::unwrap_used)]

use scioxide_core::{InvalidInputReason, SciOxideErrorKind, SciOxideErrorView};
use scioxide_random::*;

fn movement<B: SeedableRng + SkipAhead + SerializableState>() {
    let mut rng = B::from_seed(42);
    for index in [0, 1, 17, 1_u128 << 65] {
        rng.skip_to(index).unwrap();
        assert_eq!(rng.position(), Some(index));
        let before = rng.export_state().unwrap();
        rng.skip_to(index).unwrap();
        assert_eq!(rng.export_state().unwrap(), before);
        if index > 0 {
            let error = rng.skip_to(index - 1).unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::OutOfBounds);
            assert_eq!(rng.export_state().unwrap(), before);
        }
        let mut relative = B::from_seed(42);
        relative.skip_ahead(index).unwrap();
        assert_eq!(relative.export_state().unwrap(), before);
    }
}

#[test]
fn forward_movement_and_backward_errors_are_consistent() {
    movement::<SplitMix64>();
    movement::<Lcg64>();
    movement::<Pcg64>();
    movement::<Mrg32k3a>();
    movement::<Philox4x64>();
    movement::<Threefry4x64>();
}

fn untracked<B: SeedableRng + SkipAhead + SerializableState>() {
    let mut rng = B::from_seed(42);
    rng.skip_ahead(u128::MAX).unwrap();
    rng.skip_ahead(1).unwrap();
    assert_eq!(rng.position(), None);
    let before = rng.export_state().unwrap();
    for index in [0, u128::MAX] {
        let error = rng.skip_to(index).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::InvalidInput {
                reason: InvalidInputReason::UnsupportedConfiguration,
                ..
            }
        ));
        assert_eq!(rng.export_state().unwrap(), before);
    }
}

#[test]
fn untracked_absolute_movement_preserves_the_stream() {
    untracked::<SplitMix64>();
    untracked::<Lcg64>();
    untracked::<Pcg64>();
    untracked::<Mrg32k3a>();
}

#[test]
fn counter_absolute_movement_rejects_every_outside_target() {
    fn check<B: SeedableRng + SkipAhead + SerializableState>(limit: u128) {
        let mut rng = B::from_seed(42);
        let before = rng.export_state().unwrap();
        for index in [limit + 1, u128::MAX] {
            assert_eq!(
                rng.skip_to(index).unwrap_err().kind(),
                SciOxideErrorKind::OutOfBounds
            );
            assert_eq!(rng.export_state().unwrap(), before);
        }
        rng.skip_to(limit).unwrap();
        let before = rng.export_state().unwrap();
        rng.skip_to(limit).unwrap();
        assert_eq!(rng.export_state().unwrap(), before);
    }
    check::<Philox4x64>(Philox4x64::WORD_CAPACITY);
    check::<Threefry4x64>(Threefry4x64::WORD_CAPACITY);
}

#[test]
fn compact_engine_bytes_preserve_native_consumption_and_tail_discard() {
    fn check<B: SeedableRng + BitGenerator>(width: usize) {
        for length in 0..=2 * width + 1 {
            let mut expected = B::from_seed(42);
            let mut actual = B::from_seed(42);
            let mut bytes = vec![0; length];
            actual.fill_bytes(&mut bytes).unwrap();
            let mut reference = Vec::new();
            while reference.len() < length {
                let word = if width == 4 {
                    u64::from(expected.next_u32().unwrap())
                } else {
                    expected.next_u64().unwrap()
                };
                reference.extend_from_slice(&word.to_le_bytes()[..width]);
            }
            assert_eq!(bytes, reference[..length]);
            assert_eq!(actual.position(), expected.position());
            assert_eq!(actual.next_u64().unwrap(), expected.next_u64().unwrap());
        }
    }
    check::<SplitMix64>(8);
    check::<Lcg64>(8);
    check::<Pcg64>(8);
    check::<Xoshiro256StarStar>(8);
    check::<Xoshiro128StarStar>(4);
    check::<Mrg32k3a>(4);
}

fn decoded_floats<S: scioxide_core::RealScalar, B: SeedableRng + Rng<S>>() {
    let mut single = B::from_seed(42);
    let mut batch = B::from_seed(42);
    let mut values = [S::ZERO; 9];
    let mut attempts = 0;
    fill_uniforms(&mut batch, &mut values, &mut attempts).unwrap();
    assert_eq!(attempts, values.len());
    for &value in &values {
        let before = single.position();
        let expected: S = single.next_float().unwrap();
        assert_eq!(value.into_f64().to_bits(), expected.into_f64().to_bits());
        assert!(expected >= S::ZERO && expected < S::ONE);
        if let Some(position) = before {
            assert_eq!(single.position(), Some(position + 1));
        }
    }
    assert_eq!(single.position(), batch.position());
}

fn bounded_floats<S: scioxide_core::RealScalar, B: SeedableRng + Rng<S> + SkipAhead + Clone>() {
    let mut expected = B::from_seed(42);
    let mut bounded = BoundedSubstream::for_chunk(&B::from_seed(42), 0, 3).unwrap();
    for _ in 0..3 {
        let value: S = bounded.next_float().unwrap();
        let reference: S = expected.next_float().unwrap();
        assert_eq!(value.into_f64().to_bits(), reference.into_f64().to_bits());
    }
    let error = Rng::<S>::next_float(&mut bounded).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::OutOfBounds);
    assert_eq!(bounded.position(), Some(3));
}

#[test]
fn every_engine_decodes_floats_directly_in_both_precisions() {
    macro_rules! engines {
        ($($engine:ty),+ $(,)?) => {$(
            decoded_floats::<f32, $engine>();
            decoded_floats::<f64, $engine>();
        )+};
    }
    engines!(
        SplitMix64,
        Lcg64,
        Pcg64,
        Xoshiro256StarStar,
        Xoshiro128StarStar,
        Mt19937,
        Ranlux,
        Ranlxs,
        Ranlxd,
        Taus2,
        Taus113,
        Gfsr4,
        Mrg,
        Cmrg,
        Mrg32k3a,
        Philox4x64,
        Threefry4x64,
    );
    macro_rules! bounded {
        ($($engine:ty),+ $(,)?) => {$(
            bounded_floats::<f32, $engine>();
            bounded_floats::<f64, $engine>();
        )+};
    }
    bounded!(SplitMix64, Lcg64, Pcg64, Mrg32k3a, Philox4x64, Threefry4x64);
}
