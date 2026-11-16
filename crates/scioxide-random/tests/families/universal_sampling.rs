//! Analytic numerical and statistical tests for user-density samplers.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Budget, ExtendedInterval, RealScalar, SciOxideError};
use scioxide_random::*;
#[path = "../support/universal_density.rs"]
mod providers;
use providers::{Gaussian, Unit};

fn calibration<S: RealScalar>(mut draw: impl FnMut() -> S) {
    // Standard normal quartiles and probabilities are independent analytic fixtures.
    // 20,000 draws: each bin threshold is > 6 standard deviations from 0.25.
    let mut bins = [0usize; 4];
    let mut sum = 0.0;
    let mut squares = 0.0;
    for _ in 0..20_000 {
        let x = draw().into_f64();
        sum += x;
        squares += x * x;
        let i = if x < -0.6744897501960817 {
            0
        } else if x < 0.0 {
            1
        } else if x < 0.6744897501960817 {
            2
        } else {
            3
        };
        bins[i] += 1;
    }
    for n in bins {
        assert!((n as f64 / 20_000.0 - 0.25).abs() < 0.025, "{bins:?}");
    }
    assert!((sum / 20_000.0).abs() < 0.04);
    assert!((squares / 20_000.0 - 1.0).abs() < 0.07);
}
fn normal<S: RealScalar>() {
    let s = |v| S::try_from_f64_lossy(v).unwrap();
    let options = UniversalSamplingOptions::default();
    let p = Gaussian { offset: 0.0 };
    let ratio = RatioUniformsSampler::<S, _>::with_options(&p, options).unwrap();
    let mut rng = SplitMix64::from_seed(543);
    calibration(|| ratio.sample(&mut rng).unwrap());
    let ars =
        AdaptiveRejectionSampler::with_options(p, &[s(-1.0), s(0.0), s(1.0)], options).unwrap();
    let mut rng = SplitMix64::from_seed(544);
    calibration(|| ars.sample(&mut rng).unwrap());
    let mut workspace = ars.workspace().unwrap();
    let mut rng = SplitMix64::from_seed(545);
    calibration(|| ars.sample_adaptive(&mut rng, &mut workspace).unwrap().value);
    assert!(workspace.nodes() > 3);
    assert!(workspace.nodes() <= options.table_nodes());
    let tdr =
        TransformedDensitySampler::with_options(&p, &[s(-1.0), s(0.0), s(1.0)], options).unwrap();
    let mut rng = SplitMix64::from_seed(546);
    calibration(|| tdr.sample(&mut rng).unwrap());
    let mut workspace = tdr.workspace().unwrap();
    let mut rng = SplitMix64::from_seed(547);
    calibration(|| tdr.sample_adaptive(&mut rng, &mut workspace).unwrap().value);
    assert_eq!(ars.preparation_diagnostics().evaluations, 6);
}
#[test]
fn normal_f64() {
    normal::<f64>();
}
#[test]
fn normal_f32() {
    normal::<f32>();
}
fn uniform<S: RealScalar>() {
    let s = |v| S::try_from_f64_lossy(v).unwrap();
    let options = UniversalSamplingOptions::default();
    let inv = InversionSampler::with_options(Unit, options).unwrap();
    let tolerance = 64.0 * S::EPSILON.into_f64();
    for p in [0.001, 0.25, 0.5, 0.75, 0.999] {
        assert!((inv.transport(s(p)).unwrap().value.into_f64() - p).abs() < tolerance);
    }
    assert!(inv.transport(S::ZERO).is_err());
    assert!(inv.transport(S::ONE).is_err());
    let design = QmcPlan::halton(1).unwrap();
    let mut sequence = design.sequence::<S>().unwrap();
    sequence.skip_to(1).unwrap(); // Caller explicitly excludes the zero design point.
    let mut point = [S::ZERO];
    for _ in 0..64 {
        sequence.next_point_into(&mut point).unwrap();
        let value = inv.transport(point[0]).unwrap().value;
        assert!((value - point[0]).abs().into_f64() < tolerance);
    }
    let ars = AdaptiveRejectionSampler::with_options(Unit, &[s(0.25), s(0.75)], options).unwrap();
    let tdr = TransformedDensitySampler::with_options(Unit, &[s(0.25), s(0.75)], options).unwrap();
    for p in [0.01, 0.25, 0.49, 0.5, 0.51, 0.75, 0.99] {
        assert!((ars.proposal_quantile(s(p)).unwrap().into_f64() - p).abs() < tolerance);
        assert!((tdr.proposal_quantile(s(p)).unwrap().into_f64() - p).abs() < tolerance);
    }
}
#[test]
fn analytic_uniform_both_widths() {
    uniform::<f32>();
    uniform::<f64>();
}
#[test]
fn invalid_knots_and_budgets() {
    let options = UniversalSamplingOptions::default();
    let p = Gaussian { offset: 0.0 };
    for knots in [
        &[][..],
        &[0.0][..],
        &[1.0, -1.0][..],
        &[0.0, 0.0][..],
        &[f64::NAN, 1.0][..],
    ] {
        assert!(AdaptiveRejectionSampler::with_options(p, knots, options).is_err());
    }
    assert!(AdaptiveRejectionSampler::with_options(p, &[1.0, 2.0], options).is_err());
    assert!(
        AdaptiveRejectionSampler::with_options(
            p,
            &[-1.0, 1.0],
            options.with_preparation_evaluations(Budget::try_from(3).unwrap())
        )
        .is_err()
    );
    assert!(
        AdaptiveRejectionSampler::with_options(p, &[-1.0, 1.0], options.with_max_bytes(1)).is_err()
    );
    let capped =
        AdaptiveRejectionSampler::with_options(p, &[-1.0, 1.0], options.with_table_nodes(2))
            .unwrap();
    let mut w = capped.workspace().unwrap();
    let mut rng = SplitMix64::from_seed(543);
    for _ in 0..100 {
        capped.sample_adaptive(&mut rng, &mut w).unwrap();
    }
    assert_eq!(w.nodes(), 2);
}

#[derive(Clone, Copy)]
enum Shape {
    Exponential,
    GammaTwo,
    Cauchy,
    TruncatedExponential,
}
impl<S: RealScalar> Support<S> for Shape {
    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        match self {
            Self::Cauchy => ExtendedInterval::new(None, None),
            Self::TruncatedExponential => ExtendedInterval::new(Some(S::ZERO), Some(S::TWO)),
            _ => ExtendedInterval::new(Some(S::ZERO), None),
        }
    }
}
impl<S: RealScalar> LogDensity<S> for Shape {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        let x = x.into_f64();
        let l = match self {
            Self::Cauchy => -(x * x).ln_1p(),
            Self::GammaTwo => x.ln() - x,
            _ => -x,
        };
        if l == f64::NEG_INFINITY {
            Ok(S::NEG_INFINITY)
        } else {
            S::try_from_f64_lossy(l)
        }
    }
}
impl<S: RealScalar> UserDensity<S> for Shape {
    fn validate_density(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
}
impl<S: RealScalar> ConcaveDensity<S> for Shape {
    fn log_density_derivative(&self, x: S) -> Result<S, SciOxideError> {
        let x = x.into_f64();
        S::try_from_f64_lossy(match self {
            Self::Cauchy => -2.0 * x / (1.0 + x * x),
            Self::GammaTwo => 1.0 / x - 1.0,
            _ => -1.0,
        })
    }
    fn validate_concavity(&self, t: DensityTransform) -> Result<(), SciOxideError> {
        if matches!(self, Self::Cauchy) && t == DensityTransform::Log {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("concavity"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Cauchy is not globally log-concave",
            ))
        } else {
            Ok(())
        }
    }
}
fn cdf(shape: Shape, x: f64) -> f64 {
    match shape {
        Shape::Cauchy => 0.5 + x.atan() / core::f64::consts::PI,
        Shape::GammaTwo => 1.0 - (-x).exp() * (1.0 + x),
        Shape::TruncatedExponential => -(-x).exp_m1() / -(-2.0_f64).exp_m1(),
        Shape::Exponential => -(-x).exp_m1(),
    }
}
fn probability_calibration<S: RealScalar>(shape: Shape, mut sample: impl FnMut() -> S) {
    let mut bins = [0usize; 4];
    for _ in 0..12_000 {
        let p = cdf(shape, sample().into_f64());
        assert!((0.0..=1.0).contains(&p));
        bins[((p * 4.0) as usize).min(3)] += 1;
    }
    for n in bins {
        assert!((n as f64 / 12_000.0 - 0.25).abs() < 0.03, "{bins:?}");
    }
}
fn positive_and_heavy<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let options = UniversalSamplingOptions::default();
    for shape in [
        Shape::Exponential,
        Shape::GammaTwo,
        Shape::TruncatedExponential,
    ] {
        let knots = match shape {
            Shape::TruncatedExponential => [s(0.5), s(1.5)],
            _ => [s(0.5), s(2.0)],
        };
        let ars = AdaptiveRejectionSampler::with_options(shape, &knots, options).unwrap();
        let mut rng = SplitMix64::from_seed(549);
        probability_calibration(shape, || ars.sample(&mut rng).unwrap());
        let tdr = TransformedDensitySampler::with_options(shape, &knots, options).unwrap();
        let mut rng = SplitMix64::from_seed(550);
        probability_calibration(shape, || tdr.sample(&mut rng).unwrap());
    }
    let tdr =
        TransformedDensitySampler::with_options(Shape::Cauchy, &[s(-1.0), s(0.0), s(1.0)], options)
            .unwrap();
    let mut rng = SplitMix64::from_seed(551);
    probability_calibration(Shape::Cauchy, || tdr.sample(&mut rng).unwrap());
    assert!(
        AdaptiveRejectionSampler::with_options(Shape::Cauchy, &[s(-1.0), s(1.0)], options).is_err()
    );
    // Exponential hat is exactly the target, giving an independent analytic inverse.
    let exp =
        AdaptiveRejectionSampler::with_options(Shape::Exponential, &[s(0.5), s(2.0)], options)
            .unwrap();
    for p in [0.001, 0.1, 0.5, 0.9, 0.999] {
        let actual = exp.proposal_quantile(s(p)).unwrap().into_f64();
        let expected = -(-s(p).into_f64()).ln_1p();
        assert!((actual - expected).abs() < 64.0 * S::EPSILON.into_f64() * (1.0 + expected));
    }
}
#[test]
fn positive_and_heavy_f32() {
    positive_and_heavy::<f32>();
}
#[test]
fn positive_and_heavy_f64() {
    positive_and_heavy::<f64>();
}
#[test]
fn unknown_normalization_does_not_change_target() {
    for offset in [-1000.0, 1000.0] {
        let p = Gaussian { offset };
        let ars = AdaptiveRejectionSampler::new(p, &[-1.0, 0.0, 1.0]).unwrap();
        let mut rng = SplitMix64::from_seed(552);
        calibration(|| ars.sample(&mut rng).unwrap());
        let ratio = RatioUniformsSampler::<f64, _>::new(p).unwrap();
        let mut rng = SplitMix64::from_seed(553);
        calibration(|| ratio.sample(&mut rng).unwrap());
        let tdr = TransformedDensitySampler::new(p, &[-1.0, 0.0, 1.0]).unwrap();
        let mut rng = SplitMix64::from_seed(554);
        calibration(|| tdr.sample(&mut rng).unwrap());
    }
}

#[test]
fn inversion_needs_only_tails_and_support() {
    /// Uniform tails on `[0, 1]`; no density is provided.
    struct Tails;
    impl Support<f64> for Tails {
        fn support(&self) -> Result<ExtendedInterval<f64>, SciOxideError> {
            ExtendedInterval::new(Some(0.0), Some(1.0))
        }
    }
    impl LogCdf<f64> for Tails {
        fn log_cdf(&self, x: f64) -> Result<f64, SciOxideError> {
            Ok(x.clamp(0.0, 1.0).ln())
        }
    }
    impl LogSurvival<f64> for Tails {
        fn log_sf(&self, x: f64) -> Result<f64, SciOxideError> {
            Ok((1.0 - x.clamp(0.0, 1.0)).ln())
        }
    }
    let sampler = InversionSampler::<f64, _>::new(Tails).unwrap();
    let quartile = sampler.transport(0.25).unwrap();
    assert!((quartile.value - 0.25).abs() < 1e-12);
    assert!(quartile.diagnostics.evaluations > 0 && quartile.diagnostics.proposals == 0);
    let draw = sampler
        .sample_report(&mut SplitMix64::from_seed(3))
        .unwrap();
    assert_eq!(
        (draw.diagnostics.proposals, draw.diagnostics.rng_calls),
        (1, 1)
    );
    // The one per-sample limit bounds the tail work of each inversion.
    let limited = InversionSampler::<f64, _>::with_options(
        Tails,
        UniversalSamplingOptions::default().with_sample_evaluations(Budget::try_from(3).unwrap()),
    )
    .unwrap();
    assert_eq!(
        limited.transport(0.25).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
}

/// Asserts that two samplers give the same 64 draws, bit for bit, from equal seeds.
fn same_draws(a: &impl Distribution<f64>, b: &impl Distribution<f64>, seed: u64) {
    let (mut first, mut second) = (SplitMix64::from_seed(seed), SplitMix64::from_seed(seed));
    for _ in 0..64 {
        assert_eq!(
            a.sample(&mut first).unwrap().to_bits(),
            b.sample(&mut second).unwrap().to_bits()
        );
    }
}

#[test]
fn closure_adapters_reproduce_trait_implementations() {
    let gaussian = Gaussian { offset: 0.0 };
    let line = ExtendedInterval::new(None, None).unwrap();
    // The closures compute what the struct implementations compute.
    let log_pdf = |x: f64| Ok(0.0 - 0.5 * x * x);
    let concave = ConcaveDensityFn::new(log_pdf, |x: f64| Ok(-x), DensityTransform::Log, line);
    let knots = [-1.0, 0.0, 1.0];
    same_draws(
        &AdaptiveRejectionSampler::new(concave, &knots).unwrap(),
        &AdaptiveRejectionSampler::new(gaussian, &knots).unwrap(),
        561,
    );
    same_draws(
        &TransformedDensitySampler::new(concave, &knots).unwrap(),
        &TransformedDensitySampler::new(gaussian, &knots).unwrap(),
        562,
    );
    let bounds = BoundedDensity::<f64>::ratio_uniforms_bounds(&gaussian).unwrap();
    same_draws(
        &RatioUniformsSampler::new(BoundedDensityFn::new(log_pdf, bounds, line)).unwrap(),
        &RatioUniformsSampler::<f64, _>::new(gaussian).unwrap(),
        563,
    );
    let tails = LogTailsFn::new(
        |x: f64| LogCdf::<f64>::log_cdf(&Unit, x),
        |x: f64| LogSurvival::<f64>::log_sf(&Unit, x),
        ExtendedInterval::new(Some(0.0), Some(1.0)).unwrap(),
    );
    same_draws(
        &InversionSampler::new(tails).unwrap(),
        &InversionSampler::<f64, _>::new(Unit).unwrap(),
        564,
    );
}

#[test]
fn declared_transform_limits_the_concave_samplers() {
    let line = ExtendedInterval::new(None, None).unwrap();
    // Cauchy density 1/(1 + x²) is −1/√f-concave but not log-concave.
    let cauchy = ConcaveDensityFn::new(
        |x: f64| Ok(-(x * x).ln_1p()),
        |x: f64| Ok(-2.0 * x / (1.0 + x * x)),
        DensityTransform::NegativeInverseSqrt,
        line,
    );
    // Closures are not `Debug`, so the sampler is not either: take the error with `err`.
    let error = AdaptiveRejectionSampler::new(cauchy, &[-1.0, 0.0, 1.0])
        .err()
        .unwrap();
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput {
                argument,
                reason: scioxide_core::InvalidInputReason::Inconsistent,
                ..
            } if argument.as_str() == "transform"
        ),
        "{error}"
    );
    assert!(TransformedDensitySampler::new(cauchy, &[-1.0, 0.0, 1.0]).is_ok());
    assert_eq!(
        cauchy
            .validate_concavity(DensityTransform::NegativeInverseSqrt)
            .ok(),
        Some(())
    );
    // A log-concave declaration covers both transforms.
    let normal = ConcaveDensityFn::new(
        |x: f64| Ok(-0.5 * x * x),
        |x: f64| Ok(-x),
        DensityTransform::Log,
        line,
    );
    for transform in [DensityTransform::Log, DensityTransform::NegativeInverseSqrt] {
        assert!(normal.validate_concavity(transform).is_ok());
    }
    assert_eq!(normal.log_density_derivative(2.0).unwrap(), -2.0);
    assert_eq!(normal.support().unwrap(), line);
    assert!(normal.validate_density().is_ok());
}
