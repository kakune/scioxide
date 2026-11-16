//! External contracts for the canonical crate-root API.

#![allow(clippy::unwrap_used)]

use core::num::NonZeroUsize;

use scioxide_core::{ExecutionContext, ExecutionPolicy, RealScalar};
use scioxide_linalg::{Matrix, Vector};
use scioxide_random::{
    Beta, Binomial, BitGenerator, BrownianBridgePlan, Cauchy, Cdf, ChiSquared, Density,
    DirectionNumberSet, Distribution, Exponential, Gamma, JumpableRng, Lcg64, LogNormal, Mass,
    Mean, Mrg32k3a, MultivariateNormal, Normal, Pcg64, Poisson, QmcPlan, Quantile, QuasiRng, Rng,
    SamplingWorkspace, SeedableRng, SerializableState, SkipAhead, SplitMix64, SplittableRng,
    StreamPartitioner, StudentT, Uniform, Variance, WeightedSampler, Xoshiro128StarStar,
    Xoshiro256StarStar, choose, sample_indices, seed_sequence, shuffle, shuffle_in_place,
};

struct ScriptedRng<S> {
    values: std::vec::IntoIter<S>,
    fallback: S,
}

impl<S> ScriptedRng<S> {
    fn new(values: Vec<S>, fallback: S) -> Self {
        Self {
            values: values.into_iter(),
            fallback,
        }
    }
}

impl<S: Copy> BitGenerator for ScriptedRng<S> {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        Ok(0)
    }

    fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        Ok(0)
    }

    fn fill_bytes(&mut self, destination: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
        destination.fill(0);
        Ok(())
    }
}

impl<S: RealScalar> Rng<S> for ScriptedRng<S> {
    fn next_float(&mut self) -> Result<S, scioxide_core::SciOxideError> {
        Ok(self.values.next().unwrap_or(self.fallback))
    }
}

#[test]
fn supported_families_are_reachable_from_the_crate_root() {
    let mut splitmix = SplitMix64::from_seed(42);
    let mut lcg = Lcg64::from_seed(42);
    let mut pcg = Pcg64::from_seed(42);
    let mrg = Mrg32k3a::from_seed(42);
    let mut xoshiro128 = Xoshiro128StarStar::from_seed(42);
    let mut xoshiro256 = Xoshiro256StarStar::from_seed(42);
    let _: f64 = splitmix.next_float().unwrap();
    let _ = lcg.next_u64().unwrap();
    pcg.skip_ahead(1).unwrap();
    let _ = mrg.split(1).unwrap();
    xoshiro128.jump().unwrap();
    xoshiro256.long_jump().unwrap();
    let state = splitmix.export_state().unwrap();
    let _ = SplitMix64::import_state(&state).unwrap();

    let uniform = Uniform::new(-1.0_f64, 1.0).unwrap();
    let normal = Normal::new(0.0_f64, 1.0).unwrap();
    let exponential = Exponential::new(2.0_f64).unwrap();
    let gamma = Gamma::new(2.0_f64, 3.0).unwrap();
    let beta = Beta::new(2.0_f64, 3.0).unwrap();
    let chi_squared = ChiSquared::new(4.0_f64).unwrap();
    let student_t = StudentT::new(5.0_f64).unwrap();
    let cauchy = Cauchy::new(0.0_f64, 1.0).unwrap();
    let log_normal = LogNormal::new(0.0_f64, 1.0).unwrap();
    let poisson = Poisson::new(4.0_f64).unwrap();
    let binomial = Binomial::new(10, 0.25_f64).unwrap();
    assert_eq!(uniform.cdf(0.0).unwrap(), 0.5);
    assert_eq!(normal.quantile(0.5).unwrap(), 0.0);
    assert!(exponential.mean().unwrap().is_finite());
    assert!(gamma.variance().unwrap().is_finite());
    assert!(beta.pdf(0.5).unwrap().is_finite());
    assert!(chi_squared.cdf(1.0).unwrap().is_finite());
    assert!(student_t.pdf(0.0).unwrap().is_finite());
    assert!(cauchy.cdf(0.0).unwrap().is_finite());
    assert!(log_normal.mean().unwrap().is_finite());
    assert!(poisson.pmf(2).unwrap().is_finite());
    assert!(binomial.cdf(2).unwrap().is_finite());

    let mean = Vector::from_vec(vec![0.0_f64, 0.0]);
    let covariance = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let multivariate = MultivariateNormal::new(mean.as_slice(), &covariance).unwrap();
    let policy = scioxide_random::SamplingPolicy::rollback(scioxide_random::OutputPolicy::Atomic);
    let workspace = multivariate.workspace(policy, 1).unwrap();
    assert_eq!(
        workspace
            .check(&multivariate.workspace_requirements(policy, 1).unwrap())
            .unwrap(),
        scioxide_core::Capacity::Sufficient
    );

    let directions = DirectionNumberSet::JoeKuoD6;
    let sobol_plan = QmcPlan::sobol_with_direction_numbers(2, directions).unwrap();
    let mut sobol = sobol_plan.sequence::<f64>().unwrap();
    let halton_plan = QmcPlan::halton(2).unwrap();
    let mut halton = halton_plan.sequence::<f64>().unwrap();
    assert_eq!(sobol.points(3).unwrap().len(), 6);
    assert_eq!(halton.points(3).unwrap().len(), 6);
    let plan = BrownianBridgePlan::new(&[0.0_f64, 0.5, 1.0]).unwrap();
    let mut path = vec![0.0; plan.output_len()];
    let mut bridge_workspace = SamplingWorkspace::new();
    plan.apply_into(&[0.0, 1.0], &mut path, &mut bridge_workspace)
        .unwrap();
    let mut data = [1, 2, 3, 4];
    let mut shuffle_workspace = SamplingWorkspace::new();
    shuffle_in_place(
        &mut data,
        &mut splitmix,
        scioxide_random::SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic),
        &mut shuffle_workspace,
    )
    .unwrap();
    shuffle(&mut data, &mut splitmix).unwrap();
    assert!(choose(&data, &mut splitmix).unwrap().is_some());
    assert!(
        WeightedSampler::new(&[1.0_f64; 4])
            .and_then(|sampler| sampler.sample(&mut splitmix, &data))
            .is_ok()
    );
    assert_eq!(sample_indices(20, 4, &mut splitmix).unwrap().len(), 4);
    assert_eq!(
        scioxide_random::latin_hypercube::<f64, _>(4, 2, &mut splitmix)
            .unwrap()
            .len(),
        8
    );
    assert_eq!(seed_sequence(42, 2).unwrap().len(), 2);
    let plan = ExecutionContext::sequential().plan(
        ExecutionPolicy::Sequential,
        2,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: NonZeroUsize::MIN,
        },
    );
    let partitioner = StreamPartitioner::from_plan(plan, 42);
    assert_eq!(partitioner.num_partitions(), 1);

    let _ = <f64 as RealScalar>::next_after(1.0, f64::NEG_INFINITY);
}

#[test]
fn checked_consumers_classify_malformed_external_providers() {
    let distribution = Uniform::new(-2.0_f64, 3.0).unwrap();

    [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
        .into_iter()
        .for_each(|value| {
            let mut rng = ScriptedRng::new(vec![value], 0.5);
            let error = distribution.sample(&mut rng).unwrap_err();
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
        });

    [-0.5_f64, 1.0, 2.0].into_iter().for_each(|value| {
        let mut rng = ScriptedRng::new(vec![value], 0.5);
        let error = distribution.sample(&mut rng).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    });
}

#[test]
fn uniform_sampling_preserves_half_open_represented_endpoints() {
    let f64_cases = [(-f64::MAX, f64::MAX), (1.0_f64, 1.0_f64.next_up())];
    f64_cases.into_iter().for_each(|(low, high)| {
        [0.0_f64, 0.5, 1.0_f64.next_down()]
            .into_iter()
            .for_each(|value| {
                let mut rng = ScriptedRng::new(vec![value], value);
                let sample = Uniform::new(low, high).unwrap().sample(&mut rng).unwrap();
                assert!(sample.is_finite() && sample >= low && sample < high);
            });
    });

    let f32_cases = [(-f32::MAX, f32::MAX), (1.0_f32, 1.0_f32.next_up())];
    f32_cases.into_iter().for_each(|(low, high)| {
        [0.0_f32, 0.5, 1.0_f32.next_down()]
            .into_iter()
            .for_each(|value| {
                let mut rng = ScriptedRng::new(vec![value], value);
                let sample = Uniform::new(low, high).unwrap().sample(&mut rng).unwrap();
                assert!(sample.is_finite() && sample >= low && sample < high);
            });
    });
}

#[test]
fn prepared_weighted_sampling_validates_inputs() {
    let values = ["zero", "positive"];
    let weights = [0.0_f64, 1.0];
    let sampler = WeightedSampler::new(&weights).unwrap();
    let mut prepared_rng = ScriptedRng::new(vec![0.0], 0.0);
    assert_eq!(
        sampler.sample(&mut prepared_rng, &values).unwrap(),
        &"positive"
    );

    [
        Vec::<f64>::new(),
        vec![-1.0, 1.0],
        vec![f64::NAN, 1.0],
        vec![0.0, 0.0],
    ]
    .iter()
    .for_each(|invalid| {
        assert!(WeightedSampler::new(invalid).is_err());
    });
    let mut rng = ScriptedRng::new(vec![0.5], 0.5);
    assert_eq!(
        sampler.sample(&mut rng, &values[..1]).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
}

#[test]
fn errors_identify_parameters_with_consistent_kinds() {
    use scioxide_core::{SciOxideErrorKind, SciOxideErrorView, argument};
    use scioxide_random::{Cosine, DoubleGamma, Histogram, Philox4x64};
    let mut rng = ScriptedRng::new(vec![2.0_f64], 0.5);
    let cases = [
        (
            Normal::new(0.0_f64, -1.0).unwrap_err(),
            SciOxideErrorKind::DomainError,
            argument!("std_dev"),
        ),
        (
            DoubleGamma::<f64>::new(-1.0).unwrap_err(),
            SciOxideErrorKind::DomainError,
            argument!("shape"),
        ),
        (
            Cosine::<f64>::new().unwrap().quantile(2.0).unwrap_err(),
            SciOxideErrorKind::DomainError,
            argument!("p"),
        ),
        (
            Histogram::from_densities(&[0.0_f64, 1.0], &[-1.0]).unwrap_err(),
            SciOxideErrorKind::DomainError,
            argument!("densities"),
        ),
        (
            Philox4x64::from_seed(1).derive(&[0; 8]).unwrap_err(),
            SciOxideErrorKind::InvalidInput,
            argument!("path"),
        ),
        (
            rng.next_open_float().unwrap_err(),
            SciOxideErrorKind::InvalidInput,
            argument!("self"),
        ),
    ];
    for (error, kind, expected) in cases {
        assert_eq!(error.kind(), kind);
        let (SciOxideErrorView::InvalidInput {
            argument: actual, ..
        }
        | SciOxideErrorView::DomainError {
            argument: actual, ..
        }) = error.view()
        else {
            unreachable!("unexpected error kind")
        };
        assert_eq!(actual, expected);
    }
}
