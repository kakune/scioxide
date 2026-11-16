//! Uniform-word composition, forwarding, bounded state, and reference preservation.
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use scioxide_core::{RealScalar, SciOxideError};
use scioxide_random::*;

fn replay<B: BitGenerator + SerializableState + Clone + Rng<f32> + Rng<f64>>(mut engine: B) {
    engine.next_uniform_u32().unwrap();
    engine.next_uniform_u64().unwrap();
    let bytes = engine.export_state().unwrap();
    let mut restored = B::import_state(&bytes).unwrap();
    let borrowed = &mut engine;
    for _ in 0..50 {
        assert_eq!(
            restored.next_uniform_u32().unwrap(),
            borrowed.next_uniform_u32().unwrap()
        );
        assert_eq!(
            restored.next_uniform_u64().unwrap(),
            borrowed.next_uniform_u64().unwrap()
        );
        assert_eq!(restored.next_u32().unwrap(), borrowed.next_u32().unwrap());
        assert_eq!(restored.next_u64().unwrap(), borrowed.next_u64().unwrap());
        let a: f64 = restored.next_float().unwrap();
        let b: f64 = borrowed.next_float().unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
        let a: f32 = restored.next_float().unwrap();
        let b: f32 = borrowed.next_float().unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
        assert_eq!(restored.position(), borrowed.position());
    }
    let mut object: &mut dyn BitGenerator = &mut engine;
    assert_eq!(
        restored.next_uniform_u64().unwrap(),
        object.next_uniform_u64().unwrap()
    );
    // Explicitly exercise the blanket borrowed implementation, too.
    assert_eq!(
        restored.next_uniform_u32().unwrap(),
        BitGenerator::next_uniform_u32(&mut object).unwrap()
    );
    assert_eq!(
        restored.export_state().unwrap(),
        engine.export_state().unwrap()
    );
}

#[test]
fn restricted_engines_replay_mixed_calls_and_borrowed_adapters() {
    replay(Ranlux::from_seed(42));
    replay(Ranlxs::from_seed(42));
    replay(Ranlxd::from_seed(42));
    replay(Mrg::from_seed(42));
    replay(Cmrg::from_seed(42));
    replay(Mrg32k3a::from_seed(42));
}

fn unchanged<B: BitGenerator + Clone>(mut native: B) {
    let mut uniform = native.clone();
    for _ in 0..128 {
        assert_eq!(
            native.next_u32().unwrap(),
            uniform.next_uniform_u32().unwrap()
        );
        assert_eq!(
            native.next_u64().unwrap(),
            uniform.next_uniform_u64().unwrap()
        );
        assert_eq!(native.position(), uniform.position());
    }
}

#[test]
fn full_width_engines_preserve_words_and_consumption() {
    unchanged(SplitMix64::from_seed(42));
    unchanged(Lcg64::from_seed(42));
    unchanged(Pcg64::from_seed(42));
    unchanged(Xoshiro128StarStar::from_seed(42));
    unchanged(Xoshiro256StarStar::from_seed(42));
    unchanged(Mt19937::from_seed(42));
    unchanged(Taus2::from_seed(42));
    unchanged(Taus113::from_seed(42));
    unchanged(Gfsr4::from_seed(42));
    unchanged(Philox4x64::from_seed(42));
    unchanged(Threefry4x64::from_seed(42));
    let mut native = Ranlxd::from_seed(42);
    let mut uniform = native.clone();
    for _ in 0..128 {
        assert_eq!(
            native.next_u32().unwrap(),
            uniform.next_uniform_u32().unwrap()
        );
    }
}

#[test]
fn normalization_uses_native_ranges_and_transition_counts() {
    let mut native = Ranlux::from_seed(42);
    let low = u64::from(native.next_u32().unwrap());
    let middle = u64::from(native.next_u32().unwrap());
    let high = u64::from(native.next_u32().unwrap());
    let mut uniform = Ranlux::from_seed(42);
    assert_eq!(
        uniform.next_uniform_u64().unwrap(),
        low | middle << 24 | (high & 0xffff) << 48
    );
    assert_eq!(uniform.position(), Some(3));

    let mut native = Mrg32k3a::from_seed(42);
    let digits = [
        native.next_u32().unwrap(),
        native.next_u32().unwrap(),
        native.next_u32().unwrap(),
    ];
    let radix = 4_294_967_087_u128;
    let candidate = u128::from(digits[0] - 1)
        + radix * u128::from(digits[1] - 1)
        + radix * radix * u128::from(digits[2] - 1);
    let mut uniform = Mrg32k3a::from_seed(42);
    assert_eq!(
        u128::from(uniform.next_uniform_u64().unwrap()),
        candidate % (1_u128 << 64)
    );
    assert_eq!(uniform.position(), Some(3));
}

#[test]
fn bounded_normalized_reads_commit_only_whole_words() {
    let base = Mrg32k3a::from_seed(42);
    let mut bounded = BoundedSubstream::for_chunk(&base, 0, 2).unwrap();
    assert!(bounded.next_uniform_u64().is_err());
    assert_eq!(bounded.position(), Some(0));
    assert_eq!(bounded.remaining(), 2);
    let mut expected = base.clone();
    assert_eq!(
        bounded.next_uniform_u32().unwrap(),
        expected.next_uniform_u32().unwrap()
    );
    assert_eq!(bounded.position(), Some(2));
    assert!(bounded.next_uniform_u32().is_err());
    assert_eq!(bounded.position(), Some(2));

    let mut bounded = BoundedSubstream::for_chunk(&base, 0, 3).unwrap();
    assert_eq!(
        bounded.next_uniform_u64().unwrap(),
        base.clone().next_uniform_u64().unwrap()
    );
    assert_eq!(bounded.remaining(), 0);
    let mut workspace = SamplingWorkspace::new();
    let mut bounded = BoundedSubstream::for_chunk(&base, 0, 3).unwrap();
    let mut values = [0, 1, 2, 3, 4];
    assert!(
        shuffle_in_place(
            &mut values,
            &mut bounded,
            SamplingPolicy::rollback(OutputPolicy::Atomic),
            &mut workspace
        )
        .is_err()
    );
    assert_eq!(values, [0, 1, 2, 3, 4]);
    assert_eq!(bounded.position(), Some(0));
}

/// Native calls fail, so consumers cannot accidentally fall back to native words.
struct UniformOnly(SplitMix64);
impl BitGenerator for UniformOnly {
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Err(SciOxideError::not_implemented("native test words"))
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        Err(SciOxideError::not_implemented("native test words"))
    }
    fn fill_bytes(&mut self, _: &mut [u8]) -> Result<(), SciOxideError> {
        Err(SciOxideError::not_implemented("native test bytes"))
    }
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        self.0.next_uniform_u32()
    }
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        self.0.next_uniform_u64()
    }
}
impl<S: RealScalar> Rng<S> for UniformOnly {
    fn next_float(&mut self) -> Result<S, SciOxideError> {
        self.0.next_float()
    }
}

#[test]
fn nested_samplers_weighted_sampling_and_qmc_use_uniform_words() {
    let mut rng = UniformOnly(SplitMix64::from_seed(42));
    let beta = BetaBinomial::new(100, 2.0_f64, 3.0).unwrap();
    let direction = UniformDirection::<f64>::new(4).unwrap();
    let weights = WeightedSampler::new(&[1.0_f64, 2.0, 3.0]).unwrap();
    for _ in 0..100 {
        assert!(beta.sample(&mut rng).unwrap() <= 100);
        assert!(
            direction
                .contains_support(&direction.sample(&mut rng).unwrap())
                .unwrap()
        );
        assert!(*weights.sample(&mut rng, &[0, 1, 2]).unwrap() < 3);
    }
    for (plan, scramble) in [
        (QmcPlan::halton(2).unwrap(), QmcRandomization::RandomShift),
        (
            QmcPlan::sobol(2).unwrap(),
            QmcRandomization::LinearMatrixShift,
        ),
    ] {
        let plan = plan.with_randomization(scramble, &mut rng).unwrap();
        let mut cursor = plan.sequence::<f64>().unwrap();
        let mut point = [0.; 2];
        cursor.next_point_into(&mut point).unwrap();
        assert!(point.into_iter().all(|x| (0.0..1.0).contains(&x)));
    }
    let mut output = [0.; 32];
    latin_hypercube_into(
        2,
        &mut rng,
        AdvanceOnError,
        LatinHypercubeOptions::default(),
        &mut output,
        &mut SamplingWorkspace::new(),
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert!(output.into_iter().all(|x| (0.0..1.0).contains(&x)));
}

#[test]
fn restricted_engine_weighted_draws_and_random_shifts_cover_the_unit_interval() {
    let mut rng = Ranlux::from_seed(12345);
    let sampler = WeightedSampler::new(&[1.0_f64, 2.0, 3.0]).unwrap();
    let mut counts = [0; 3];
    for _ in 0..20_000 {
        counts[sampler.sample_index(&mut rng).unwrap()] += 1;
    }
    for (count, probability) in counts.into_iter().zip([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]) {
        assert!((f64::from(count) / 20_000.0 - probability).abs() < 0.025);
    }
    for scramble in [
        QmcRandomization::RandomShift,
        QmcRandomization::LinearMatrixShift,
    ] {
        let mut sum = 0.0;
        for seed in 0..128 {
            let plan = if scramble == QmcRandomization::RandomShift {
                QmcPlan::halton(1).unwrap()
            } else {
                QmcPlan::sobol(1).unwrap()
            };
            let plan = plan
                .with_randomization(scramble, &mut Ranlux::from_seed(seed))
                .unwrap();
            let mut cursor = plan.sequence::<f64>().unwrap();
            sum += cursor.next_point().unwrap()[0];
            let restored = QmcPlan::import_state(&cursor.export_state().unwrap()).unwrap();
            assert_eq!(
                cursor.next_point().unwrap(),
                restored.sequence::<f64>().unwrap().next_point().unwrap()
            );
        }
        assert!(
            (sum / 128.0 - 0.5).abs() < 0.1,
            "{scramble:?}: mean {}",
            sum / 128.0
        );
    }
}
