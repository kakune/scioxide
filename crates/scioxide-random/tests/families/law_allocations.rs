//! Prepared evaluation and sampling allocation contracts.
#![allow(clippy::unwrap_used)]

use scioxide_random::*;

#[test]
fn prepared_scalar_densities_and_direct_tails_execute() {
    let beta = Beta::new(2.5_f64, 3.5).unwrap();
    let gamma = Gamma::new(2.5_f64, 0.5).unwrap();
    let student = StudentT::new(5.0_f64).unwrap();
    let fisher = FisherF::new(5.0_f64, 10.0).unwrap();
    let count = NegativeBinomial::new(2.5_f64, 0.3).unwrap();
    for _ in 0..8 {
        beta.pdf(0.5).unwrap();
        gamma.pdf(0.5).unwrap();
        student.pdf(0.5).unwrap();
        fisher.pdf(0.5).unwrap();
        fisher.cdf(0.5).unwrap();
        fisher.sf(10.0).unwrap();
        count.pmf(5).unwrap();
        count.cdf(5).unwrap();
        count.sf(100).unwrap();
    }
}

#[test]
fn prepared_scalar_samplers_execute() {
    let budget = LawOptions::default()
        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215));
    let normal = Normal::new(2.0_f64, 3.0).unwrap();
    let exponential = Exponential::new(1.5_f64).unwrap();
    let weibull = WeibullMin::new(1.5_f64).unwrap();
    let pareto = Pareto::new(3.0_f64).unwrap();
    let logistic = Logistic::new(0.0_f64, 1.0).unwrap();
    let beta = Beta::new(2.0_f64, 3.0).unwrap();
    let fisher = FisherF::with_options(5.0_f64, 7.0, budget).unwrap();
    let noncentral = NoncentralChiSquared::with_options(0.5_f64, 3.0, budget).unwrap();
    let noncentral_t = NoncentralStudentT::with_options(5.0_f64, 1.0, budget).unwrap();
    let range = StudentizedRange::with_options(3.0_f64, 10.0, budget).unwrap();
    let skew = JonesFaddySkewT::new(2.0_f64, 3.0).unwrap();
    let negative = NegativeBinomial::new(5.0_f64, 0.4).unwrap();
    let beta_negative = BetaNegativeBinomial::new(2.0_f64, 3.0, 4.0).unwrap();
    let log_series = LogSeries::new(0.6_f64).unwrap();
    let mut rng = Mt19937::from_seed(1345);
    for _ in 0..64 {
        normal.sample(&mut rng).unwrap();
        exponential.sample(&mut rng).unwrap();
        weibull.sample(&mut rng).unwrap();
        pareto.sample(&mut rng).unwrap();
        logistic.sample(&mut rng).unwrap();
        beta.sample(&mut rng).unwrap();
        fisher.sample(&mut rng).unwrap();
        noncentral.sample(&mut rng).unwrap();
        noncentral_t.sample(&mut rng).unwrap();
        range.sample(&mut rng).unwrap();
        skew.sample(&mut rng).unwrap();
        CountDistribution::sample(&negative, &mut rng).unwrap();
        CountDistribution::sample(&beta_negative, &mut rng).unwrap();
        CountDistribution::sample(&log_series, &mut rng).unwrap();
        uniform_index(1000, &mut rng).unwrap();
    }
}

#[test]
fn single_draws_through_prepared_samplers_and_shuffles_execute() {
    let binomial = Binomial::new(1_000, 0.4_f64).unwrap();
    let poisson = Poisson::new(30.0_f64).unwrap();
    let skellam = Skellam::new(3.0_f64, 1.0).unwrap();
    let mut deck: Vec<u32> = (0..52).collect();
    let mut rng = Mt19937::from_seed(2469);
    for _ in 0..64 {
        CountDistribution::sample(&binomial, &mut rng).unwrap();
        CountDistribution::sample(&poisson, &mut rng).unwrap();
        CountDistribution::sample(&skellam, &mut rng).unwrap();
        shuffle(&mut deck, &mut rng).unwrap();
    }
}

#[test]
fn bounded_substream_reads_execute() {
    let base = Philox4x64::from_seed(1346);
    let mut chunk = BoundedSubstream::for_chunk(&base, 2, 1 << 20).unwrap();
    let mut bytes = [0_u8; 64];
    {
        for _ in 0..64 {
            chunk.next_u64().unwrap();
            chunk.next_uniform_u32().unwrap();
            let _: f64 = chunk.next_float().unwrap();
        }
        chunk.fill_bytes(&mut bytes).unwrap();
    };
}

#[test]
fn stable_evaluation_and_inversion_execute() {
    for (alpha, beta) in [(0.1, 0.5), (0.99, 0.5), (1., 1.), (1.01, -1.), (1.5, 0.)] {
        let law = LevyStable::new(alpha, beta).unwrap();
        {
            law.log_cdf(1.).unwrap();
            law.log_sf(1.).unwrap();
        };
        {
            law.quantile(0.1).unwrap();
            law.isf(0.1).unwrap();
        };
    }
}
#[test]
fn circular_prepared_queries_and_buffers() {
    let law = VonMises::new(0.0, 10.0).unwrap();
    law.log_pdf(0.3).unwrap();
    law.log_sf(0.3).unwrap();
    law.isf(0.1).unwrap();
    let mut rng = SplitMix64::from_seed(541);
    law.sample(&mut rng).unwrap();
    let law = VonMisesFisher::new(&[1.0, 0.0, 0.0, 0.0, 0.0], 10.0).unwrap();
    law.log_pdf(&[1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let mut output = [0.0; 20];
    let batch = SamplingPolicy::advance_on_error(OutputPolicy::Atomic);
    let mut workspace = batch
        .workspace(scioxide_random::SamplingRequest::rows(4, 5))
        .unwrap();
    law.sample_n_into(&mut rng, batch, &mut output, &mut workspace)
        .unwrap();
}
#[test]
fn prepared_queries() {
    let law = Truncated::new(Normal::new(0.0, 1.0).unwrap(), 10.0, 11.0).unwrap();
    law.log_pdf(10.5).unwrap();
    law.isf(0.01).unwrap();
    let histogram = Histogram::from_masses(&[0.0, 1.0, 2.0], &[1.0, 2.0]).unwrap();
    histogram.quantile(0.5).unwrap();
    let mix = Mixture::new(
        vec![
            Normal::new(0.0, 1.0).unwrap(),
            Normal::new(4.0, 1.0).unwrap(),
        ],
        &[1.0, 2.0],
    )
    .unwrap();
    mix.log_sf(40.0).unwrap();
    let mut rng = SplitMix64::from_seed(531);
    mix.sample(&mut rng).unwrap();
    histogram.sample(&mut rng).unwrap();
    PowerNormal::new(2.0).unwrap().sample(&mut rng).unwrap();
    let mut out = [0.0; 4];
    let batch = SamplingPolicy::advance_on_error(OutputPolicy::Atomic);
    let mut workspace = batch
        .workspace(scioxide_random::SamplingRequest::scalars(4))
        .unwrap();
    law.sample_n_into(&mut rng, batch, &mut out, &mut workspace)
        .unwrap();
    mix.sample_n_into(&mut rng, batch, &mut out, &mut workspace)
        .unwrap();
    histogram
        .sample_n_into(&mut rng, batch, &mut out, &mut workspace)
        .unwrap();
    PowerNormal::new(2.0)
        .unwrap()
        .sample_n_into(&mut rng, batch, &mut out, &mut workspace)
        .unwrap();
}

#[test]
fn asymmetric_prepared_queries_and_buffers() {
    fn check<D: Distribution<f64> + LogDensity<f64> + LogSurvival<f64> + Quantile<f64>>(d: D) {
        let mut rng = SplitMix64::from_seed(539);
        let mut out = [0.; 8];
        let batch = SamplingPolicy::advance_on_error(OutputPolicy::Atomic);
        let mut workspace = batch
            .workspace(scioxide_random::SamplingRequest::scalars(8))
            .unwrap();
        d.log_pdf(1.).unwrap();
        d.log_sf(1.).unwrap();
        d.quantile(0.4).unwrap();
        d.sample_n_into(&mut rng, batch, &mut out, &mut workspace)
            .unwrap();
    }
    check(CrystalBall::new(2., 4.).unwrap());
    check(SkewCauchy::new(0.8).unwrap());
    check(JonesFaddySkewT::new(2., 5.).unwrap());
    check(SkewNormal::new(4.).unwrap());
    check(GeneralizedInverseGaussian::new(1., 2.).unwrap());
    check(
        GeneralizedHyperbolic::with_options(
            1.,
            2.,
            0.5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    check(
        NormalInverseGaussian::with_options(
            2.,
            0.5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
}

#[test]
fn accelerated_quantiles_execute() {
    fn check<D: Quantile<f64> + InverseSurvival<f64>>(law: D) {
        {
            law.quantile(0.1).unwrap();
            law.isf(1e-10).unwrap();
        };
    }
    check(
        FisherF::with_options(
            5.0,
            10.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    check(
        NoncentralChiSquared::with_options(
            4.0,
            2.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    check(
        NoncentralF::with_options(
            5.0,
            10.0,
            2.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    check(
        NoncentralStudentT::with_options(
            5.0,
            1.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    check(
        Rice::with_options(
            1.5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    let extreme = NoncentralStudentT::with_options(
        1.0_f64,
        -1.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    {
        extreme.isf(1e-309).unwrap();
    };
    let law = Binomial::new(10000, 0.3).unwrap();
    {
        law.quantile(0.1).unwrap();
        law.isf(1e-10).unwrap();
    };
}

#[test]
fn closure_adapter_queries_execute() {
    use scioxide_core::ExtendedInterval;
    let line = ExtendedInterval::new(None, None).unwrap();
    // Logistic tails: ln F(x) = −ln(1 + e^(−x)) and ln P(X > x) = −ln(1 + e^x).
    let tails = LogTailsFn::new(
        |x: f64| Ok(-(-x).exp().ln_1p()),
        |x: f64| Ok(-x.exp().ln_1p()),
        line,
    );
    let concave = ConcaveDensityFn::new(
        |x: f64| Ok(-0.5 * x * x),
        |x: f64| Ok(-x),
        DensityTransform::Log,
        line,
    );
    let bounds = RatioUniformsBounds {
        shift: 0.0,
        u_max: 1.0,
        v_min: -0.86,
        v_max: 0.86,
    };
    let bounded = BoundedDensityFn::new(|x: f64| Ok(-0.5 * x * x), bounds, line);
    {
        for x in [-2.0, 0.0, 1.5] {
            tails.log_cdf(x).unwrap();
            tails.log_sf(x).unwrap();
            concave.log_pdf(x).unwrap();
            concave.log_density_derivative(x).unwrap();
            bounded.log_pdf(x).unwrap();
        }
        assert_eq!(tails.support().unwrap(), line);
        assert_eq!(concave.support().unwrap(), line);
        concave.validate_density().unwrap();
        concave
            .validate_concavity(DensityTransform::NegativeInverseSqrt)
            .unwrap();
        assert_eq!(bounded.support().unwrap(), line);
        bounded.validate_density().unwrap();
        bounded.ratio_uniforms_bounds().unwrap();
    };
    let inversion = InversionSampler::new(tails).unwrap();
    let ratio = RatioUniformsSampler::new(bounded).unwrap();
    let mut rng = SplitMix64::from_seed(5);
    {
        inversion.sample(&mut rng).unwrap();
        ratio.sample(&mut rng).unwrap();
        log_interval_probability(&tails, -1.0, 2.0).unwrap();
        invert_log_probability(&tails, 0.3_f64.ln(), false).unwrap();
    };
}
#[test]
fn scalar_batches_keep_configured_byte_limits_when_borrowed_or_wrapped() {
    use scioxide_random::*;
    let options = LawOptions::<f64>::default().with_max_bytes(8);
    fn check<D: Distribution<f64>>(law: D) {
        let mut rng = SplitMix64::from_seed(7);
        let before = rng.position();
        assert_eq!(
            law.sample_n(&mut rng, 2).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(rng.position(), before);
        let mut out = [123.0; 2];
        assert_eq!(
            law.sample_n_into(
                &mut rng,
                SamplingPolicy::rollback(OutputPolicy::Atomic),
                &mut out,
                &mut SamplingWorkspace::new()
            )
            .unwrap_err()
            .kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(out, [123.0; 2]);
        assert_eq!(rng.position(), before);
        assert_eq!(law.sample_n(&mut rng, 0).unwrap().len(), 0);
        assert_eq!(law.sample_n(&mut rng, 1).unwrap().len(), 1);
    }
    let gamma = Gamma::with_options(2.0, 1.0, options).unwrap();
    check(&gamma);
    check(Affine::new(gamma, 1.0, 2.0).unwrap());
    check(Beta::with_options(2.0, 3.0, options).unwrap());
    check(StudentT::with_options(4.0, options).unwrap());
    check(ChiSquared::with_options(4.0, options).unwrap());
    check(Gompertz::with_options(2.0, options).unwrap());
    check(InverseGaussian::with_options(2.0, options).unwrap());
}
