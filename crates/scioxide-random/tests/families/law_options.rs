//! One constructor-time accuracy contract for every configurable law.
#![allow(clippy::unwrap_used, clippy::type_complexity)]
use scioxide_core::{Budget, SciOxideError, SciOxideErrorKind, SciOxideErrorView, Tolerance};
use scioxide_linalg::Matrix;
use scioxide_random::*;

macro_rules! law {
    ($scalar:ty; $name:ident $(, $argument:expr)*) => {
        (stringify!($name), |options| $name::<$scalar>::with_options($($argument,)* options).map(|_| ()))
    };
}

macro_rules! validation_matrix {
    ($scalar:ty) => {{
        type S = $scalar;
        let constructors: &[(&str, fn(LawOptions<S>) -> Result<(), SciOxideError>)] = &[
            law!($scalar; Alpha, 2.0),
            law!($scalar; ExponentiallyModifiedNormal, 1.0),
            law!($scalar; DoubleParetoLogNormal, 0.0, 1.0, 3.0, 2.0),
            law!($scalar; GeneralizedNormal, 2.0),
            law!($scalar; PowerNormal, 2.0),
            law!($scalar; GeneralizedExponential, 1.0, 1.0, 1.0),
            law!($scalar; Cosine),
            law!($scalar; InverseGaussian, 1.0),
            law!($scalar; ReciprocalInverseGaussian, 1.0),
            law!($scalar; Gamma, 2.0, 1.0),
            law!($scalar; DoubleGamma, 2.0),
            law!($scalar; GeneralizedGamma, 2.0, 1.0),
            law!($scalar; Pearson3, 0.2),
            law!($scalar; Beta, 2.0, 3.0),
            law!($scalar; BetaPrime, 2.0, 3.0),
            law!($scalar; ChiSquared, 2.0),
            law!($scalar; StudentT, 4.0),
            law!($scalar; Poisson, 2.0),
            law!($scalar; JonesFaddySkewT, 2.0, 3.0),
            law!($scalar; Argus, 2.0),
            ("Mixture", |o| Mixture::with_options(vec![Normal::<S>::new(0.0,1.0)?],&[1.0],o).map(|_|())),
            ("Truncated", |o| Truncated::with_options(Normal::<S>::new(0.0,1.0)?,-1.0,1.0,o).map(|_|())),
            ("OrderStatistic", |o| OrderStatistic::with_options(Normal::<S>::new(0.0,1.0)?,2,3,o).map(|_|())),
            ("Transformed::abs", |o| Transformed::abs_with_options(Normal::<S>::new(0.0,1.0)?,o).map(|_|())),
            ("Transformed::exp", |o| Transformed::exp_with_options(Normal::<S>::new(0.0,1.0)?,o).map(|_|())),
            ("Transformed::log", |o| Transformed::log_with_options(LogNormal::<S>::new(0.0,1.0)?,o).map(|_|())),
            ("MultivariateNormal", |o| MultivariateNormal::with_options(&[0.0],&Matrix::identity(1)?,MultivariateNormalOptions::default(),o).map(|_|())),
            law!($scalar; LevyStable, 2.0, 0.0),
            law!($scalar; RelativisticBreitWigner, 1.0),
            law!($scalar; GaussHypergeometric, 1.0, 1.0, 0.0, 0.0),
            law!($scalar; GeneralizedInverseGaussian, 1.0, 1.0),
            law!($scalar; GeneralizedHyperbolic, 1.0, 2.0, 0.0),
            law!($scalar; NormalInverseGaussian, 2.0, 0.0),
            law!($scalar; SkewNormal, 0.0),
            law!($scalar; Geometric, 0.5),
            ("Geometric::from_rate", |o| Geometric::<S>::from_rate_with_options(1.0, o).map(|_| ())),
            law!($scalar; NegativeBinomial, 2.0, 0.5),
            law!($scalar; BetaBinomial, 4, 2.0, 3.0),
            law!($scalar; BetaNegativeBinomial, 2.0, 2.0, 3.0),
            law!($scalar; LogSeries, 0.5),
            law!($scalar; YuleSimon, 3.0),
            law!($scalar; Zipf, 3.0),
            law!($scalar; Zipfian, 2.0, 4),
            law!($scalar; Skellam, 1.0, 2.0),
            law!($scalar; Boltzmann, 1.0, 4),
            law!($scalar; DiscreteLaplace, 1.0),
            law!($scalar; DiscreteUniform, 0, 4),
            law!($scalar; Hypergeometric, 8, 4, 2),
            law!($scalar; FisherNoncentralHypergeometric, 8, 4, 2, 1.0),
            law!($scalar; WalleniusNoncentralHypergeometric, 8, 4, 2, 1.0),
            law!($scalar; PoissonBinomial, &[0.25, 0.5]),
            law!($scalar; VonMises, 0.0, 1.0),
            law!($scalar; VonMisesLine, 0.0, 1.0),
            law!($scalar; WrappedCauchy, 0.0, 0.5),
            law!($scalar; UniformDirection, 3),
            law!($scalar; VonMisesFisher, &[1.0, 0.0, 0.0], 1.0),
            law!($scalar; JohnsonSB, 0.0, 1.0),
            law!($scalar; Kappa4, 0.0, 0.0),
            law!($scalar; IrwinHall, 3),
            law!($scalar; ExponentialPower, 1.0),
            law!($scalar; Gompertz, 1.0),
            law!($scalar; Binomial, 4, 0.5),
            law!($scalar; FisherF, 4.0, 8.0),
            law!($scalar; NoncentralChiSquared, 4.0, 1.0),
            law!($scalar; NoncentralF, 4.0, 8.0, 1.0),
            law!($scalar; NoncentralStudentT, 4.0, 1.0),
            law!($scalar; Rice, 1.0),
            law!($scalar; Kolmogorov),
            law!($scalar; KolmogorovSmirnovOneSided, 4),
            law!($scalar; KolmogorovSmirnovTwoSided, 4),
            law!($scalar; StudentizedRange, 3.0, 8.0),
            law!($scalar; MultivariateHypergeometric, &[4, 4], 2),
            law!($scalar; RandomTable, &[2, 2], &[2, 2]),
            law!($scalar; Dirichlet, &[1.0, 2.0]),
            law!($scalar; DirichletMultinomial, 4, &[1.0, 2.0]),
            law!($scalar; Multinomial, 4, &[0.5, 0.5]),
            law!($scalar; NormalInverseGamma, 0.0, 1.0, 3.0, 1.0),
            law!($scalar; MultivariateStudentT, &[0.0], &Matrix::identity(1)?, 4.0, Default::default()),
            law!($scalar; Wishart, 4.0, &Matrix::identity(1)?),
            law!($scalar; InverseWishart, 4.0, &Matrix::identity(1)?),
            law!($scalar; MatrixNormal, &Matrix::zeros(1, 1)?, &Matrix::identity(1)?, &Matrix::identity(1)?),
            law!($scalar; MatrixT, &Matrix::zeros(1, 1)?, &Matrix::identity(1)?, &Matrix::identity(1)?, 4.0),
            law!($scalar; OrthogonalGroup, 2),
            law!($scalar; SpecialOrthogonalGroup, 2),
            law!($scalar; UnitaryGroup, 2),
            law!($scalar; RandomCorrelation, &[1.0, 1.0]),
        ];
        let floor = 64.0 * S::EPSILON;
        let below_floor = S::from_bits(floor.to_bits() - 1);
        let below_ceiling = S::from_bits((0.1 as S).to_bits() - 1);
        for &(name, construct) in constructors {
            construct(LawOptions::default()).unwrap_or_else(|e| panic!("{name} default: {e}"));
            for relative in [floor, S::EPSILON.sqrt(), below_ceiling] {
                let options = LawOptions::default()
                    .with_tolerance(Tolerance::new(0.0, relative).unwrap());
                construct(options).unwrap_or_else(|e| panic!("{name} valid {relative}: {e}"));
            }
            for relative in [0.0, below_floor, 0.1, 1.0] {
                // Builders accept the policy; every constructor reports the same error.
                let options = LawOptions::default()
                    .with_tolerance(Tolerance::new(0.0, relative).unwrap());
                let error = construct(options).unwrap_err();
                assert!(matches!(error.view(), SciOxideErrorView::InvalidInput { argument, .. }
                    if argument == scioxide_core::argument!("options.tolerance")),
                    "{name} relative={relative}: {error}");
            }
        }
    }};
}

#[test]
#[allow(clippy::panic)]
fn every_law_validates_the_same_accuracy_range() {
    validation_matrix!(f32);
    validation_matrix!(f64);
}

#[test]
fn moments_and_entropy_use_the_stored_budget() {
    let tiny = LawOptions::<f64>::default().with_work_budget(Budget::ONE);
    let moments = [
        JohnsonSB::with_options(0.0, 1.0, tiny).unwrap().mean(),
        JohnsonSB::with_options(0.0, 1.0, tiny).unwrap().variance(),
        Kappa4::with_options(0.0, 0.0, tiny).unwrap().mean(),
        Kappa4::with_options(0.0, 0.0, tiny).unwrap().variance(),
        ExponentialPower::with_options(1.0, tiny).unwrap().mean(),
        ExponentialPower::with_options(1.0, tiny)
            .unwrap()
            .variance(),
        Gompertz::with_options(1.0, tiny).unwrap().mean(),
        Gompertz::with_options(1.0, tiny).unwrap().variance(),
        Binomial::with_options(4, 0.5, tiny).unwrap().entropy(),
    ];
    for result in moments {
        assert_eq!(
            result.unwrap_err().kind(),
            SciOxideErrorKind::ConvergenceFailure
        );
    }
    assert_eq!(
        Binomial::with_options(100, 0.5, tiny)
            .unwrap()
            .quantile(0.3)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ConvergenceFailure,
    );
    // Two support atoms fit exactly; one does not. Degenerate support needs no sum.
    let two = tiny.with_work_budget(Budget::ONE.saturating_add(1));
    let entropy = Binomial::with_options(1, 0.5, two)
        .unwrap()
        .entropy()
        .unwrap();
    assert!((entropy - core::f64::consts::LN_2).abs() < 8.0 * f64::EPSILON);
    assert_eq!(
        Binomial::with_options(20, 0.0, tiny)
            .unwrap()
            .entropy()
            .unwrap(),
        0.0
    );
    assert_eq!(
        Kappa4::with_options(0.0, -1.0, tiny)
            .unwrap()
            .mean()
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DomainError
    );
}

#[test]
fn owned_matrix_constructors_forward_resource_limits() {
    let identity = Matrix::<f64>::identity(2).unwrap();
    let options = LawOptions::default().with_max_bytes(1);
    for error in [
        Wishart::with_options(4.0, &identity, options).unwrap_err(),
        InverseWishart::with_options(4.0, &identity, options).unwrap_err(),
        MatrixNormal::with_options(&identity, &identity, &identity, options).unwrap_err(),
        MatrixT::with_options(&identity, &identity, &identity, 4.0, options).unwrap_err(),
    ] {
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
    }
}

#[test]
fn numerical_formula_and_irwin_hall_use_the_configured_policy() {
    let tiny = LawOptions::default().with_work_budget(Budget::ONE);
    let alpha = Alpha::with_options(2.0_f64, tiny).unwrap();
    // The lower inverse is closed-form and needs only one work unit. The
    // upper inverse still charges its residual refinement to the same budget.
    assert!((alpha.cdf(alpha.quantile(0.3).unwrap()).unwrap() - 0.3).abs() < 1e-8);
    assert_eq!(
        alpha.isf(0.1).unwrap_err().kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    let alpha = Alpha::new(2.0_f64).unwrap();
    assert!((alpha.cdf(alpha.quantile(0.3).unwrap()).unwrap() - 0.3).abs() < 1e-8);

    let limited = LawOptions::default().with_work_budget(Budget::try_from(16).unwrap());
    let irwin = IrwinHall::with_options(2, limited).unwrap();
    assert_eq!(
        irwin.quantile(0.3).unwrap_err().kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(
        IrwinHall::with_options(2, tiny).unwrap_err().kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    let loose = LawOptions::default().with_tolerance(Tolerance::new(0.0, 0.01).unwrap());
    let strict =
        LawOptions::default().with_tolerance(Tolerance::new(0.0, 64.0 * f64::EPSILON).unwrap());
    let loose = IrwinHall::with_options(3, loose).unwrap();
    let strict = IrwinHall::with_options(3, strict).unwrap();
    let (coarse, refined) = (
        loose.quantile(0.37).unwrap(),
        strict.quantile(0.37).unwrap(),
    );
    assert!((coarse - refined).abs() > 1e-7);
    assert!((strict.cdf(refined).unwrap() - 0.37).abs() < 2e-14);
    assert!((loose.cdf(coarse).unwrap() - 0.37).abs() < 0.001);
}

#[test]
fn analytic_quantiles_remain_independent_of_moment_policy() {
    let tiny = LawOptions::default().with_work_budget(Budget::ONE);
    let kappa = Kappa4::with_options(0.2_f64, 0.3, tiny).unwrap();
    let johnson = JohnsonSB::with_options(0.4_f64, 1.2, tiny).unwrap();
    for p in [0.01, 0.3, 0.8, 0.99] {
        assert_eq!(
            kappa.quantile(p).unwrap(),
            Kappa4::new(0.2_f64, 0.3).unwrap().quantile(p).unwrap()
        );
        assert_eq!(
            johnson.isf(p).unwrap(),
            JohnsonSB::new(0.4_f64, 1.2).unwrap().isf(p).unwrap()
        );
    }
    for error in [kappa.mean().unwrap_err(), johnson.mean().unwrap_err()] {
        assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    }
    assert!(
        Kappa4::new(0.2_f64, 0.3)
            .unwrap()
            .mean()
            .unwrap()
            .is_finite()
    );
    assert!(
        JohnsonSB::new(0.4_f64, 1.2)
            .unwrap()
            .mean()
            .unwrap()
            .is_finite()
    );
}

#[derive(Clone)]
struct CountingProvider {
    requests: usize,
}
impl BitGenerator for CountingProvider {
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        self.next_u64()
    }
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        self.next_u32()
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        self.requests += 1;
        Ok(1_u64 << 63)
    }
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        self.requests += 1;
        Ok(1_u32 << 31)
    }
    fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), SciOxideError> {
        output.fill(0);
        Ok(())
    }
}
impl Rng<f64> for CountingProvider {
    fn next_float(&mut self) -> Result<f64, SciOxideError> {
        self.requests += 1;
        Ok(0.75)
    }
}

#[test]
fn poisson_sampling_and_nested_tails_stop_before_exceeding_their_allowances() {
    let limited = LawOptions::default().with_sample_word_budget(Budget::ONE);
    let sampler = Poisson::with_options(30.0_f64, limited)
        .unwrap()
        .prepare_sampler()
        .unwrap();
    let mut provider = CountingProvider { requests: 0 };
    assert_eq!(
        sampler.sample(&mut provider).unwrap_err().kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(provider.requests, 1);
    let sufficient = limited.with_sample_word_budget(Budget::try_from(2).unwrap());
    assert!(
        Poisson::with_options(30.0_f64, sufficient)
            .unwrap()
            .prepare_sampler()
            .unwrap()
            .sample(&mut provider)
            .is_ok()
    );
    let zero = Poisson::with_options(0.0_f64, limited)
        .unwrap()
        .prepare_sampler()
        .unwrap();
    let before = provider.requests;
    assert_eq!(zero.sample(&mut provider).unwrap(), 0);
    assert_eq!(provider.requests, before);

    let low_work = LawOptions::default().with_work_budget(Budget::ONE);
    assert_eq!(
        Poisson::with_options(30.0_f64, low_work)
            .unwrap()
            .quantile(0.3)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(
        Skellam::with_options(3.0_f64, 2.0, low_work)
            .unwrap()
            .log_cdf(1)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert!(
        Skellam::new(3.0_f64, 2.0)
            .unwrap()
            .log_cdf(1)
            .unwrap()
            .is_finite()
    );
}

#[test]
fn multivariate_normal_charges_requests_per_draw_including_batches() {
    let covariance = Matrix::identity(2).unwrap();
    let options = LawOptions::default().with_sample_word_budget(Budget::ONE);
    let law = MultivariateNormal::with_options(
        &[0.0_f64; 2],
        &covariance,
        MultivariateNormalOptions::default(),
        options,
    )
    .unwrap();
    let mut provider = CountingProvider { requests: 0 };
    assert_eq!(
        law.sample(&mut provider).unwrap_err().kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(provider.requests, 1);
    let law = MultivariateNormal::with_options(
        &[0.0_f64; 2],
        &covariance,
        MultivariateNormalOptions::default(),
        options.with_sample_word_budget(Budget::try_from(2).unwrap()),
    )
    .unwrap();
    let before = provider.requests;
    let mut samples = [0.0; 4];
    law.sample_n_into(
        &mut provider,
        SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
        &mut samples,
        &mut SamplingWorkspace::new(),
    )
    .unwrap();
    assert_eq!(provider.requests - before, 4);
    let point = MultivariateNormal::with_options(
        &[1.0_f64; 2],
        &Matrix::zeros(2, 2).unwrap(),
        MultivariateNormalOptions::default(),
        options,
    )
    .unwrap();
    let before = provider.requests;
    assert_eq!(point.sample(&mut provider).unwrap().as_slice(), &[1.0, 1.0]);
    assert_eq!(provider.requests, before);
}

#[test]
fn student_t_forwards_the_selected_gaussian_rank_policy() {
    use scioxide_linalg::RankPolicy;
    let shape = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1e-18]).unwrap();
    let exact = MultivariateNormalOptions::default().with_rank_policy(RankPolicy::Exact);
    let ordinary = MultivariateStudentT::new(&[0.0, 0.0], &shape, 4.0).unwrap();
    let precise =
        MultivariateStudentT::with_options(&[0.0, 0.0], &shape, 4.0, exact, LawOptions::default())
            .unwrap();
    assert_eq!((ordinary.rank(), precise.rank()), (1, 2));
    let gaussian =
        MultivariateNormal::with_options(&[0.0, 0.0], &shape, exact, LawOptions::default())
            .unwrap();
    let limit = MultivariateStudentT::gaussian_limit_with_options(
        &[0.0, 0.0],
        &shape,
        exact,
        LawOptions::default(),
    )
    .unwrap();
    assert_eq!(limit.rank(), gaussian.rank());
}

#[test]
fn absolute_transform_shares_quadrature_and_inversion_work() {
    use core::cell::Cell;
    struct CountedNormal<'a> {
        density: &'a Cell<usize>,
        normal: Normal<f64>,
    }
    impl LogDensity<f64> for CountedNormal<'_> {
        fn log_pdf(&self, x: f64) -> Result<f64, SciOxideError> {
            self.density.set(self.density.get() + 1);
            self.normal.log_pdf(x)
        }
    }
    impl Density<f64> for CountedNormal<'_> {
        fn pdf(&self, x: f64) -> Result<f64, SciOxideError> {
            Ok(self.log_pdf(x)?.exp())
        }
    }
    impl LogCdf<f64> for CountedNormal<'_> {
        fn log_cdf(&self, x: f64) -> Result<f64, SciOxideError> {
            self.normal.log_cdf(x)
        }
    }
    impl LogSurvival<f64> for CountedNormal<'_> {
        fn log_sf(&self, x: f64) -> Result<f64, SciOxideError> {
            self.normal.log_sf(x)
        }
    }
    impl Cdf<f64> for CountedNormal<'_> {
        fn cdf(&self, x: f64) -> Result<f64, SciOxideError> {
            Ok(self.log_cdf(x)?.exp())
        }
    }
    impl Survival<f64> for CountedNormal<'_> {
        fn sf(&self, x: f64) -> Result<f64, SciOxideError> {
            Ok(self.log_sf(x)?.exp())
        }
    }
    impl Support<f64> for CountedNormal<'_> {
        fn support(&self) -> Result<scioxide_core::ExtendedInterval<f64>, SciOxideError> {
            self.normal.support()
        }
    }
    let density = Cell::new(0);
    let counted = CountedNormal {
        density: &density,
        normal: Normal::new(0.0, 1.0).unwrap(),
    };
    let low = LawOptions::default().with_work_budget(Budget::try_from(10).unwrap());
    let folded =
        Transformed::abs_with_options(Affine::new(&counted, 0.0, 1.0).unwrap(), low).unwrap();
    assert_eq!(
        folded.log_cdf(1e-12).unwrap_err().kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert!(density.get() > 0 && density.get() <= 10);
    density.set(0);
    let folded = Transformed::abs(&counted).unwrap();
    assert!(
        (folded.log_cdf(1e-12).unwrap() - (1e-12 * (2.0 / core::f64::consts::PI).sqrt()).ln())
            .abs()
            < 1e-12
    );
    density.set(0);
    let components: Vec<&dyn ContinuousLaw<f64>> = vec![&folded];
    let mixture = Mixture::with_options(
        components,
        &[1.0],
        low.with_work_budget(Budget::try_from(256).unwrap()),
    )
    .unwrap();
    assert_eq!(
        mixture.quantile(1e-15).unwrap_err().kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert!(density.get() > 0 && density.get() <= 256);
}

#[test]
fn mixture_moments_and_draws_share_the_outer_allowance() {
    let tiny = LawOptions::default().with_work_budget(Budget::ONE);
    let johnson = JohnsonSB::new(0.4_f64, 1.2).unwrap();
    let mixture = Mixture::with_options(vec![johnson], &[1.0], tiny).unwrap();
    assert_eq!(
        mixture.mean().unwrap_err().kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    let alpha = Alpha::new(2.0_f64).unwrap();
    let mixture = Mixture::with_options(vec![alpha], &[1.0], tiny).unwrap();
    let mut provider = CountingProvider { requests: 0 };
    assert_eq!(
        mixture.sample(&mut provider).unwrap_err().kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
}

/// `F(p)` and `F_with_options(p, LawOptions::default())` build the same law.
macro_rules! same_construction {
    ($($name:ident $(:: $with:ident)? ($($argument:expr),*)),+ $(,)?) => {$(
        assert_eq!(
            format!("{:?}", same_construction!(@plain $name $(:: $with)? ($($argument),*))),
            format!("{:?}", same_construction!(@options $name $(:: $with)? ($($argument),*))),
            stringify!($name)
        );
    )+};
    (@plain $name:ident ($($argument:expr),*)) => {
        $name::<f64>::new($($argument),*).unwrap()
    };
    (@options $name:ident ($($argument:expr),*)) => {
        $name::<f64>::with_options($($argument,)* LawOptions::default()).unwrap()
    };
}

fn same_scalar_draws<D: Distribution<f64>>(plain: &D, configured: &D) {
    let (mut left, mut right) = (SplitMix64::from_seed(64), SplitMix64::from_seed(64));
    for _ in 0..64 {
        assert_eq!(
            plain.sample(&mut left).unwrap().to_bits(),
            configured.sample(&mut right).unwrap().to_bits()
        );
    }
}

fn same_continuous<D: Distribution<f64> + LogDensity<f64> + Quantile<f64>>(
    plain: D,
    configured: D,
    x: f64,
) {
    // Errors, such as an exhausted default work limit, must agree as well.
    let bits = |value: Result<f64, SciOxideError>| format!("{:?}", value.map(f64::to_bits));
    assert_eq!(bits(plain.log_pdf(x)), bits(configured.log_pdf(x)));
    assert_eq!(bits(plain.quantile(0.3)), bits(configured.quantile(0.3)));
    same_scalar_draws(&plain, &configured);
}

fn same_count_values<D: LogMass<f64> + Quantile<f64, usize>>(plain: &D, configured: &D, k: usize) {
    assert_eq!(
        plain.log_pmf(k).unwrap().to_bits(),
        configured.log_pmf(k).unwrap().to_bits()
    );
    assert_eq!(
        plain.quantile(0.3).unwrap(),
        configured.quantile(0.3).unwrap()
    );
}

fn same_count_draws<D: CountDistribution>(plain: &D, configured: &D)
where
    D::Count: PartialEq + core::fmt::Debug,
{
    let (mut left, mut right) = (SplitMix64::from_seed(64), SplitMix64::from_seed(64));
    for _ in 0..64 {
        assert_eq!(
            plain.sample(&mut left).unwrap(),
            configured.sample(&mut right).unwrap()
        );
    }
}

fn same_count<D: CountDistribution<Count = usize> + LogMass<f64> + Quantile<f64, usize>>(
    plain: D,
    configured: D,
    k: usize,
) {
    same_count_values(&plain, &configured, k);
    same_count_draws(&plain, &configured);
}

#[test]
fn default_constructors_equal_their_configured_forms() {
    let identity = Matrix::<f64>::identity(2).unwrap();
    let zeros = Matrix::<f64>::zeros(2, 2).unwrap();
    same_construction!(
        JonesFaddySkewT(2.0, 3.0),
        SkewNormal(1.0),
        Beta(2.0, 3.0),
        BetaPrime(2.0, 3.0),
        Binomial(20, 0.3),
        Cosine(),
        Argus(1.0),
        RelativisticBreitWigner(1.0),
        ChiSquared(3.0),
        VonMises(0.0, 2.0),
        VonMisesLine(0.0, 2.0),
        WrappedCauchy(0.0, 0.5),
        UniformDirection(3),
        VonMisesFisher(&[1.0, 0.0, 0.0], 2.0),
        Geometric(0.25),
        NegativeBinomial(3.0, 0.4),
        BetaBinomial(10, 2.0, 3.0),
        BetaNegativeBinomial(3.0, 2.0, 3.0),
        LogSeries(0.5),
        YuleSimon(3.0),
        Zipf(3.0),
        Zipfian(2.0, 10),
        Boltzmann(1.0, 5),
        DiscreteLaplace(1.0),
        DiscreteUniform(0, 9),
        ExponentialPower(1.5),
        GeneralizedExponential(1.0, 1.0, 1.0),
        Kappa4(0.1, 0.1),
        Gompertz(1.0),
        Hypergeometric(20, 7, 5),
        FisherNoncentralHypergeometric(20, 7, 5, 2.0),
        WalleniusNoncentralHypergeometric(20, 7, 5, 2.0),
        Gamma(2.0, 3.0),
        DoubleGamma(2.0),
        GeneralizedGamma(2.0, 1.5),
        Pearson3(0.2),
        GaussHypergeometric(1.0, 1.0, 0.0, 0.0),
        GeneralizedInverseGaussian(1.0, 1.0),
        GeneralizedHyperbolic(1.0, 2.0, 0.5),
        NormalInverseGaussian(2.0, 0.5),
        InverseGaussian(1.0),
        ReciprocalInverseGaussian(1.0),
        IrwinHall(3),
        JohnsonSB(0.5, 1.0),
        MultivariateHypergeometric(&[4, 6], 3),
        RandomTable(&[2, 3], &[3, 2]),
        DirichletMultinomial(5, &[1.0, 2.0]),
        OrthogonalGroup(3),
        SpecialOrthogonalGroup(3),
        UnitaryGroup(2),
        RandomCorrelation(&[0.5, 1.5]),
        Multinomial(5, &[0.25, 0.75]),
        NormalInverseGamma(0.0, 1.0, 3.0, 1.0),
        Dirichlet(&[1.0, 2.0]),
        Alpha(2.0),
        GeneralizedNormal(2.0),
        ExponentiallyModifiedNormal(1.0),
        PowerNormal(2.0),
        DoubleParetoLogNormal(0.0, 1.0, 3.0, 2.0),
        Poisson(4.5),
        PoissonBinomial(&[0.25, 0.5, 0.75]),
        Skellam(1.0, 2.0),
        LevyStable(1.5, 0.5),
        StudentT(4.0),
        FisherF(4.0, 8.0),
        NoncentralChiSquared(4.0, 1.0),
        NoncentralF(4.0, 8.0, 1.0),
        NoncentralStudentT(4.0, 1.0),
        Kolmogorov(),
        KolmogorovSmirnovOneSided(5),
        KolmogorovSmirnovTwoSided(5),
        StudentizedRange(3.0, 8.0),
        Rice(1.0),
        Wishart(4.0, &identity),
        InverseWishart(4.0, &identity),
        MatrixNormal(&zeros, &identity, &identity),
        MatrixT(&zeros, &identity, &identity, 4.0),
    );
    assert_eq!(
        format!("{:?}", Geometric::<f64>::from_rate(0.5).unwrap()),
        format!(
            "{:?}",
            Geometric::<f64>::from_rate_with_options(0.5, LawOptions::default()).unwrap()
        )
    );
    let normal = || Normal::<f64>::new(1.0, 2.0).unwrap();
    assert_eq!(
        format!(
            "{:?}",
            Mixture::new(vec![normal(), normal()], &[1.0, 3.0]).unwrap()
        ),
        format!(
            "{:?}",
            Mixture::with_options(vec![normal(), normal()], &[1.0, 3.0], LawOptions::default())
                .unwrap()
        )
    );
    assert_eq!(
        format!("{:?}", Truncated::new(normal(), 0.0, 3.0).unwrap()),
        format!(
            "{:?}",
            Truncated::with_options(normal(), 0.0, 3.0, LawOptions::default()).unwrap()
        )
    );
    assert_eq!(
        format!("{:?}", OrderStatistic::new(normal(), 2, 5).unwrap()),
        format!(
            "{:?}",
            OrderStatistic::with_options(normal(), 2, 5, LawOptions::default()).unwrap()
        )
    );
    assert_eq!(
        format!("{:?}", Transformed::abs(normal()).unwrap()),
        format!(
            "{:?}",
            Transformed::abs_with_options(normal(), LawOptions::default()).unwrap()
        )
    );
    assert_eq!(
        format!("{:?}", Transformed::exp(normal()).unwrap()),
        format!(
            "{:?}",
            Transformed::exp_with_options(normal(), LawOptions::default()).unwrap()
        )
    );
    let log_normal = || LogNormal::<f64>::new(0.0, 1.0).unwrap();
    assert_eq!(
        format!("{:?}", Transformed::log(log_normal()).unwrap()),
        format!(
            "{:?}",
            Transformed::log_with_options(log_normal(), LawOptions::default()).unwrap()
        )
    );
    let mean = [1.0, -1.0];
    let covariance = Matrix::from_vec(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap();
    let default_rank = MultivariateNormalOptions::default();
    assert_eq!(
        format!("{:?}", MultivariateNormal::new(&mean, &covariance).unwrap()),
        format!(
            "{:?}",
            MultivariateNormal::with_options(
                &mean,
                &covariance,
                default_rank,
                LawOptions::default()
            )
            .unwrap()
        )
    );
    assert_eq!(
        format!(
            "{:?}",
            MultivariateStudentT::new(&mean, &covariance, 4.0).unwrap()
        ),
        format!(
            "{:?}",
            MultivariateStudentT::with_options(
                &mean,
                &covariance,
                4.0,
                default_rank,
                LawOptions::default()
            )
            .unwrap()
        )
    );
    assert_eq!(
        format!(
            "{:?}",
            MultivariateStudentT::gaussian_limit(&mean, &covariance).unwrap()
        ),
        format!(
            "{:?}",
            MultivariateStudentT::gaussian_limit_with_options(
                &mean,
                &covariance,
                default_rank,
                LawOptions::default()
            )
            .unwrap()
        )
    );
}

#[test]
fn default_constructors_draw_and_evaluate_as_their_configured_forms() {
    let options = LawOptions::default;
    same_continuous(
        SkewNormal::new(1.0).unwrap(),
        SkewNormal::with_options(1.0, options()).unwrap(),
        0.5,
    );
    same_continuous(
        Beta::new(2.0, 3.0).unwrap(),
        Beta::with_options(2.0, 3.0, options()).unwrap(),
        0.4,
    );
    same_continuous(
        Argus::new(1.0).unwrap(),
        Argus::with_options(1.0, options()).unwrap(),
        0.5,
    );
    same_continuous(
        RelativisticBreitWigner::new(1.0).unwrap(),
        RelativisticBreitWigner::with_options(1.0, options()).unwrap(),
        1.0,
    );
    same_continuous(
        ChiSquared::new(3.0).unwrap(),
        ChiSquared::with_options(3.0, options()).unwrap(),
        2.0,
    );
    same_continuous(
        VonMises::new(0.0, 2.0).unwrap(),
        VonMises::with_options(0.0, 2.0, options()).unwrap(),
        0.5,
    );
    same_continuous(
        ExponentialPower::new(1.5).unwrap(),
        ExponentialPower::with_options(1.5, options()).unwrap(),
        0.5,
    );
    same_continuous(
        Kappa4::new(0.1, 0.1).unwrap(),
        Kappa4::with_options(0.1, 0.1, options()).unwrap(),
        0.5,
    );
    same_continuous(
        Gompertz::new(1.0).unwrap(),
        Gompertz::with_options(1.0, options()).unwrap(),
        0.5,
    );
    same_continuous(
        Gamma::new(2.0, 3.0).unwrap(),
        Gamma::with_options(2.0, 3.0, options()).unwrap(),
        3.0,
    );
    same_continuous(
        GaussHypergeometric::new(2.0, 3.0, 0.5, 0.5).unwrap(),
        GaussHypergeometric::with_options(2.0, 3.0, 0.5, 0.5, options()).unwrap(),
        0.4,
    );
    same_continuous(
        NormalInverseGaussian::new(2.0, 0.5).unwrap(),
        NormalInverseGaussian::with_options(2.0, 0.5, options()).unwrap(),
        0.5,
    );
    same_continuous(
        InverseGaussian::new(1.0).unwrap(),
        InverseGaussian::with_options(1.0, options()).unwrap(),
        0.5,
    );
    same_continuous(
        IrwinHall::new(3).unwrap(),
        IrwinHall::with_options(3, options()).unwrap(),
        1.2,
    );
    same_continuous(
        JohnsonSB::new(0.5, 1.0).unwrap(),
        JohnsonSB::with_options(0.5, 1.0, options()).unwrap(),
        0.4,
    );
    same_continuous(
        Alpha::new(2.0).unwrap(),
        Alpha::with_options(2.0, options()).unwrap(),
        0.5,
    );
    same_continuous(
        LevyStable::new(1.5, 0.5).unwrap(),
        LevyStable::with_options(1.5, 0.5, options()).unwrap(),
        0.5,
    );
    same_continuous(
        StudentT::new(4.0).unwrap(),
        StudentT::with_options(4.0, options()).unwrap(),
        0.5,
    );
    same_continuous(
        FisherF::new(4.0, 8.0).unwrap(),
        FisherF::with_options(4.0, 8.0, options()).unwrap(),
        1.5,
    );
    same_continuous(
        KolmogorovSmirnovOneSided::new(5).unwrap(),
        KolmogorovSmirnovOneSided::with_options(5, options()).unwrap(),
        0.3,
    );
    same_continuous(
        StudentizedRange::new(3.0, 8.0).unwrap(),
        StudentizedRange::with_options(3.0, 8.0, options()).unwrap(),
        2.0,
    );
    same_continuous(
        Rice::new(1.0).unwrap(),
        Rice::with_options(1.0, options()).unwrap(),
        1.0,
    );
    let normal = || Normal::<f64>::new(1.0, 2.0).unwrap();
    same_continuous(
        Truncated::new(normal(), 0.0, 3.0).unwrap(),
        Truncated::with_options(normal(), 0.0, 3.0, options()).unwrap(),
        1.0,
    );
    same_continuous(
        Transformed::exp(normal()).unwrap(),
        Transformed::exp_with_options(normal(), options()).unwrap(),
        2.0,
    );
    same_continuous(
        Mixture::new(vec![normal()], &[1.0]).unwrap(),
        Mixture::with_options(vec![normal()], &[1.0], options()).unwrap(),
        0.5,
    );
    let (plain, configured) = (
        Binomial::new(20, 0.3).unwrap(),
        Binomial::with_options(20, 0.3, options()).unwrap(),
    );
    same_count_values(&plain, &configured, 6);
    same_count_draws(
        &plain.prepare_sampler().unwrap(),
        &configured.prepare_sampler().unwrap(),
    );
    let (plain, configured) = (
        Poisson::new(4.5).unwrap(),
        Poisson::with_options(4.5, options()).unwrap(),
    );
    same_count_values(&plain, &configured, 4);
    same_count_draws(
        &plain.prepare_sampler().unwrap(),
        &configured.prepare_sampler().unwrap(),
    );
    same_count(
        NegativeBinomial::new(3.0, 0.4).unwrap(),
        NegativeBinomial::with_options(3.0, 0.4, options()).unwrap(),
        4,
    );
    same_count(
        Boltzmann::new(1.0, 5).unwrap(),
        Boltzmann::with_options(1.0, 5, options()).unwrap(),
        1,
    );
    same_count(
        Hypergeometric::new(20, 7, 5).unwrap(),
        Hypergeometric::with_options(20, 7, 5, options()).unwrap(),
        2,
    );
    same_count(
        PoissonBinomial::new(&[0.25, 0.5, 0.75]).unwrap(),
        PoissonBinomial::with_options(&[0.25, 0.5, 0.75], options()).unwrap(),
        1,
    );
    let (plain, configured) = (
        Skellam::<f64>::new(1.0, 2.0).unwrap(),
        Skellam::with_options(1.0, 2.0, options()).unwrap(),
    );
    assert_eq!(
        plain.log_pmf(-1).unwrap().to_bits(),
        configured.log_pmf(-1).unwrap().to_bits()
    );
    same_count_draws(
        &plain.prepare_sampler().unwrap(),
        &configured.prepare_sampler().unwrap(),
    );
    let mean = [1.0, -1.0];
    let covariance = Matrix::from_vec(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap();
    let plain = MultivariateNormal::new(&mean, &covariance).unwrap();
    let configured = MultivariateNormal::with_options(
        &mean,
        covariance.view(),
        MultivariateNormalOptions::default(),
        options(),
    )
    .unwrap();
    assert_eq!(
        plain.log_pdf(&[0.5, 0.0]).unwrap().to_bits(),
        configured.log_pdf(&[0.5, 0.0]).unwrap().to_bits()
    );
    let (mut left, mut right) = (SplitMix64::from_seed(64), SplitMix64::from_seed(64));
    for _ in 0..64 {
        assert_eq!(
            plain.sample(&mut left).unwrap(),
            configured.sample(&mut right).unwrap()
        );
    }
}

fn standard_normal_equals_new<S: scioxide_core::RealScalar>() {
    let standard = Normal::<S>::standard();
    let explicit = Normal::new(S::ZERO, S::ONE).unwrap();
    for x in [-2.5, -0.5, 0.0, 1.25] {
        let x = S::try_from_f64_lossy(x).unwrap();
        assert_eq!(
            standard.cdf(x).unwrap().into_f64().to_bits(),
            explicit.cdf(x).unwrap().into_f64().to_bits()
        );
    }
    for p in [1e-6, 0.3, 0.975] {
        let p = S::try_from_f64_lossy(p).unwrap();
        assert_eq!(
            standard.quantile(p).unwrap().into_f64().to_bits(),
            explicit.quantile(p).unwrap().into_f64().to_bits()
        );
    }
    let (mut left, mut right) = (SplitMix64::from_seed(64), SplitMix64::from_seed(64));
    for _ in 0..64 {
        assert_eq!(
            standard.sample(&mut left).unwrap().into_f64().to_bits(),
            explicit.sample(&mut right).unwrap().into_f64().to_bits()
        );
    }
}

#[test]
fn standard_normal_equals_the_explicit_construction() {
    standard_normal_equals_new::<f32>();
    standard_normal_equals_new::<f64>();
}
