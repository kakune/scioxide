#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/reductions.rs"]
mod reduction_test;
use scioxide_core::{Budget, ExecutionContext};
use scioxide_linalg::MatrixView;
use scioxide_random::{InversionOptions, Mass, Mean, Quantile, Variance};
use scioxide_stats::*;
#[path = "support/executor.rs"]
mod executor;

fn accurate_options() -> InversionOptions<f64> {
    let root = InversionOptions::default()
        .root_options()
        .clone()
        .with_step_tolerance(scioxide_core::StepTolerance::new(1e-12, 1e-12).unwrap());
    InversionOptions::default()
        .with_root_options(root)
        .with_work_budget(Budget::ONE.saturating_add(4095))
}
fn close(x: f64, y: f64, tol: f64) {
    assert!((x - y).abs() <= tol * y.abs().max(1.0), "{x} != {y}");
}

#[test]
fn beta_identities_replication_and_predictive_normalization() {
    let prior = BetaBinomialPrior::<f64>::new(2., 3.).unwrap();
    let batch = prior.update_batch(&[(3, 5), (2, 4)]).unwrap();
    assert_eq!(batch.parameters().unwrap(), (7., 7.));
    assert_eq!(batch, prior.update(3, 5).unwrap().update(2, 4).unwrap());
    assert_eq!(
        prior.update_frequency(2, 4, 3).unwrap(),
        prior.update_batch(&[(2, 4); 3]).unwrap()
    );
    let predictive = batch.predictive(10).unwrap();
    close(
        (0..=10).map(|k| predictive.pmf(k).unwrap()).sum(),
        1.,
        1e-12,
    );
    close(batch.posterior_law().unwrap().mean().unwrap(), 0.5, 1e-12);
    assert_eq!(prior.update_frequency(1, 2, 0).unwrap(), prior);
    assert_eq!(prior.update(0, 0).unwrap(), prior);
    assert!(prior.update(3, 2).is_err());
    assert!(prior.update_frequency(3, 2, 0).is_err());
    assert_eq!(prior.parameters().unwrap(), (2., 3.));
}
#[test]
fn gamma_rate_exposure_and_predictive_normalization() {
    let prior = GammaPoissonPrior::<f64>::new(2., 3.).unwrap();
    let post = prior.update_batch(&[(4, 2.), (3, 0.5)]).unwrap();
    assert_eq!(post.parameters().unwrap(), (9., 5.5));
    assert_eq!(post, prior.update(4, 2.).unwrap().update(3, 0.5).unwrap());
    close(
        post.posterior_law().unwrap().mean().unwrap(),
        9. / 5.5,
        1e-12,
    );
    let law = post.predictive(2.).unwrap();
    close((0..200).map(|k| law.pmf(k).unwrap()).sum(), 1., 1e-11);
    close(law.mean().unwrap(), 18. / 5.5, 1e-12);
    close(post.predictive(0.).unwrap().pmf(0).unwrap(), 1., 0.);
    assert_eq!(
        prior.update_frequency(2, 1., 3).unwrap(),
        prior.update_batch(&[(2, 1.); 3]).unwrap()
    );
    assert_eq!(prior.update(0, 0.).unwrap(), prior);
    assert!(prior.update(1, 0.).is_err());
    assert!(prior.update(0, -1.).is_err());
    assert!(prior.update(0, f64::NAN).is_err());
}
#[test]
fn explicit_improper_boundaries_only_proper_posteriors_have_laws() {
    let beta = BetaBinomialPrior::<f64>::improper(0., 0.).unwrap();
    assert!(!beta.is_proper());
    assert!(beta.posterior_law().is_err());
    assert!(beta.update(0, 4).unwrap().predictive(2).is_err());
    assert!(beta.update(2, 4).unwrap().is_proper());
    for (a, b) in [(0., 2.), (2., 0.)] {
        assert!(
            !BetaBinomialPrior::<f64>::improper(a, b)
                .unwrap()
                .is_proper()
        );
    }
    let gamma = GammaPoissonPrior::<f64>::improper(0., 0.).unwrap();
    assert!(gamma.update(0, 2.).unwrap().posterior_law().is_err());
    assert!(gamma.update(1, 2.).unwrap().is_proper());
    for (a, b) in [(0., 2.), (2., 0.)] {
        assert!(
            !GammaPoissonPrior::<f64>::improper(a, b)
                .unwrap()
                .is_proper()
        );
    }
    assert!(GammaPoissonPrior::<f64>::improper(1., 1.).is_err());
    assert!(BetaBinomialPrior::<f64>::improper(1., 1.).is_err());
    assert!(BetaBinomialPrior::<f64>::new(0., 1.).is_err());
    assert!(GammaPoissonPrior::<f64>::new(1., 0.).is_err());
    assert!(NormalInverseGammaPrior::<f64>::new(0., 0., 1., 1.).is_err());
    for bad in [f64::NAN, f64::INFINITY, -1.] {
        assert!(BetaBinomialPrior::<f64>::improper(bad, 0.).is_err());
        assert!(GammaPoissonPrior::<f64>::improper(0., bad).is_err());
        assert!(ConfidenceProbability::new(bad).is_err());
    }
    assert!(ConfidenceProbability::new(0.).is_err());
    assert!(ConfidenceProbability::new(1.).is_err());
    for values in [&[][..], &[1.][..], &[1., 1.][..]] {
        assert!(normal_posterior_marginals(values).is_err());
    }
}
#[test]
fn normal_posterior_identity_sequential_batch_and_merged() {
    let prior = NormalInverseGammaPrior::<f64>::new(0., 2., 3., 4.).unwrap();
    let post = prior.update_batch(&[1., 2., 3.]).unwrap();
    let (m, k, a, b) = post.parameters().unwrap();
    close(m, 1.2, 1e-14);
    close(k, 5., 0.);
    close(a, 4.5, 0.);
    close(b, 7.4, 1e-14);
    let sequential = [1., 2., 3.]
        .into_iter()
        .try_fold(prior, |p, x| p.update(x))
        .unwrap();
    for (x, y) in [sequential.parameters().unwrap()]
        .into_iter()
        .flat_map(|(m, k, a, b)| [(m, 1.2), (k, 5.), (a, 4.5), (b, 7.4)])
    {
        close(x, y, 1e-14);
    }
    let mut left = OnlineMoments::from_slice(&[1., 2.], MomentOrder::Second).unwrap();
    let right = OnlineMoments::from_slice(&[3.], MomentOrder::Second).unwrap();
    left.merge(&right).unwrap();
    assert_eq!(post, prior.update_moments(&left).unwrap());
    assert!(post.posterior_law().is_ok());
    let prediction = post.predictive(1.).unwrap();
    close(prediction.mean().unwrap(), 1.2, 1e-13);
    close(
        prediction.variance().unwrap(),
        7.4 / (4.5 - 1.) * (1. + 1. / 5.),
        1e-12,
    );
    assert!(post.predictive(0.).is_err());
    assert_eq!(prior.update_batch(&[]).unwrap(), prior);
    assert!(prior.update(f64::NAN).is_err());
}
#[test]
fn normal_frequency_and_known_precision_weights() {
    let prior = NormalInverseGammaPrior::<f64>::new(0., 2., 3., 4.).unwrap();
    let frequency = OnlineWeightedMoments::from_slices(
        &[1., 3.],
        Weights::<_, Frequency>::new(&[2., 1.]),
        MomentOrder::Second,
    )
    .unwrap();
    let a = prior
        .update_weighted(&frequency)
        .unwrap()
        .parameters()
        .unwrap();
    let b = prior
        .update_batch(&[1., 1., 3.])
        .unwrap()
        .parameters()
        .unwrap();
    for (x, y) in [(a.0, b.0), (a.1, b.1), (a.2, b.2), (a.3, b.3)] {
        close(x, y, 1e-14);
    }
    let analytic = OnlineWeightedMoments::from_slices(
        &[1., 3.],
        Weights::<_, Analytic>::new(&[2., 1.]),
        MomentOrder::Second,
    )
    .unwrap();
    let post = prior.update_weighted(&analytic).unwrap();
    let (m, k, a, b) = post.parameters().unwrap();
    close(m, 1., 1e-14);
    close(k, 5., 0.);
    close(a, 4., 0.);
    close(b, 7., 1e-14);
    let mut left = OnlineWeightedMoments::from_slices(
        &[1.],
        Weights::<_, Analytic>::new(&[2.]),
        MomentOrder::Second,
    )
    .unwrap();
    left.merge(
        &OnlineWeightedMoments::from_slices(
            &[3.],
            Weights::<_, Analytic>::new(&[1.]),
            MomentOrder::Second,
        )
        .unwrap(),
    )
    .unwrap();
    // Batch and partition merges retain compensated centers with different
    // rounding histories; compare the projected numerical parameters.
    let merged = prior.update_weighted(&left).unwrap().parameters().unwrap();
    for (actual, expected) in [merged.0, merged.1, merged.2, merged.3]
        .into_iter()
        .zip([m, k, a, b])
    {
        close(actual, expected, 1e-14);
    }
}
#[test]
fn reference_prior_exact_marginals_and_undefined_moments() {
    let values = [6., 9., 12., 7., 8., 8., 13.];
    let posterior = NormalInverseGammaPrior::<f64>::reference()
        .update_batch(&values)
        .unwrap();
    assert_eq!(posterior.parameters().unwrap(), (9., 7., 3., 20.));
    let marginals = normal_posterior_marginals(&values).unwrap();
    close(marginals.mean_law().mean().unwrap(), 9., 1e-14);
    close(marginals.variance_law().mean().unwrap(), 10., 1e-12);
    // Exact mean identity; comparator outputs alone are not the oracle.
    close(
        marginals.standard_deviation_law().mean().unwrap(),
        20_f64.sqrt() * 3. * std::f64::consts::PI.sqrt() / 8.,
        1e-12,
    );
    let summary = normal_credible_summary(
        &[0., 2.],
        ConfidenceProbability::new(0.9).unwrap(),
        &accurate_options(),
    )
    .unwrap();
    assert!(summary.mean.mean.is_none());
    assert!(summary.variance.mean.is_none());
    assert!(summary.standard_deviation.mean.is_none());
    let expected = 1. + (0.45 * std::f64::consts::PI).tan();
    close(summary.mean.interval.bounds().upper(), expected, 1e-8);
    let large: Vec<_> = (0..1001).map(|i| (i % 7) as f64).collect();
    let exact = normal_posterior_marginals(&large).unwrap();
    let p = NormalInverseGammaPrior::reference()
        .update_batch(&large)
        .unwrap();
    let (_, _, shape, scale) = p.parameters().unwrap();
    close(
        exact.variance_law().mean().unwrap(),
        scale / (shape - 1.),
        1e-11,
    );
}
#[test]
fn credible_quantiles_closed_form_beta_and_exponential() {
    let p = ConfidenceProbability::new(0.9).unwrap();
    let beta = BetaBinomialPrior::<f64>::new(1., 1.)
        .unwrap()
        .posterior_law()
        .unwrap();
    let interval = credible_interval(&beta, p, &accurate_options()).unwrap();
    close(interval.bounds().lower(), 0.05, 5e-8);
    close(interval.bounds().upper(), 0.95, 5e-8);
    assert_eq!(interval.probability(), p);
    let gamma = GammaPoissonPrior::<f64>::new(1., 2.)
        .unwrap()
        .posterior_law()
        .unwrap();
    let interval = credible_interval(&gamma, p, &accurate_options()).unwrap();
    close(interval.bounds().lower(), -0.95_f64.ln() / 2., 1e-8);
    close(interval.bounds().upper(), -0.05_f64.ln() / 2., 1e-8);
    let root = InversionOptions::default().root_options().clone();
    let limited = InversionOptions::default()
        .with_root_options(root)
        .with_work_budget(Budget::ONE);
    assert!(credible_interval(&beta, p, &limited).is_err());
}
#[test]
fn real_widths_extreme_scale_and_checked_projection() {
    let beta = BetaBinomialPrior::<f32>::new(1., 2.)
        .unwrap()
        .update(3, 5)
        .unwrap();
    assert_eq!(beta.parameters().unwrap(), (4., 4.));
    close(
        beta.posterior_law().unwrap().mean().unwrap() as f64,
        0.5,
        2e-5,
    );
    let gamma = GammaPoissonPrior::<f32>::new(2., 3.)
        .unwrap()
        .update(2, 1.)
        .unwrap();
    close(
        gamma.posterior_law().unwrap().mean().unwrap() as f64,
        1.,
        2e-5,
    );
    let normal = NormalInverseGammaPrior::<f32>::new(0., 2., 3., 4.)
        .unwrap()
        .update_batch(&[1., 2., 3.])
        .unwrap();
    close(normal.parameters().unwrap().0 as f64, 1.2, 2e-5);
    // Squaring the mean difference overflows f64, but the small precision keeps beta finite.
    let tiny = NormalInverseGammaPrior::<f64>::new(-1e200, 1e-200, 2., 1.)
        .unwrap()
        .update(1e200)
        .unwrap();
    close(tiny.parameters().unwrap().3, 2e200, 1e-14);
    let huge = GammaPoissonPrior::<f64>::new(1., f64::MAX)
        .unwrap()
        .update(0, f64::MAX)
        .unwrap();
    assert!(huge.parameters().is_err());
    assert!(huge.posterior_law().is_ok()); // reciprocal rate remains representable
    let overflow = BetaBinomialPrior::<f32>::new(f32::MAX, 1.)
        .unwrap()
        .update(usize::MAX, usize::MAX)
        .unwrap();
    assert!(overflow.is_proper());
}
#[test]
fn owned_borrowed_views_and_executor_reductions() {
    let values = [1., 2., 3., 4.];
    let view_shape = [2, 2];
    let view_strides = [2, 1];
    let view = BatchView::new(
        &values,
        scioxide_core::StridedLayout::new(&view_shape, &view_strides).unwrap(),
        Axis::All,
    )
    .unwrap();
    let prior = NormalInverseGammaPrior::new(0., 2., 3., 4.).unwrap();
    let executor = executor::PartitionExecutor(std::num::NonZeroUsize::new(2).unwrap());
    let context = ExecutionContext::new(&executor);
    let options_axis = Axis::All;
    let options = StatisticsOptions::new().with_merge_policy(MergePolicy::StablePartitions);
    let result =
        reduction_test::moments(view.with_axis(options_axis).unwrap(), options, context).unwrap();
    let a = prior
        .update_snapshot(result[0])
        .unwrap()
        .parameters()
        .unwrap();
    let b = prior.update_batch(&values).unwrap().parameters().unwrap();
    for (x, y) in [(a.0, b.0), (a.1, b.1), (a.2, b.2), (a.3, b.3)] {
        close(x, y, 1e-14);
    }
}

#[test]
fn borrowed_batch_staging_executor_equivalence_and_rollback() {
    use scioxide_core::ExecutionPolicy;
    let values = [1., 2., 3., 4.];
    let weights = [1., 2., 1., 2.];
    let view = MatrixView::from_slice(2, 2, &values).unwrap();
    let prior = NormalInverseGammaPrior::new(0., 2., 3., 4.).unwrap();
    let requests = [
        NormalPosteriorRequest::new(prior, view),
        NormalPosteriorRequest::new(prior, view.transpose()).with_weights(
            Weights::<_, Analytic>::new(
                MatrixView::from_slice(2, 2, &weights).unwrap().transpose(),
            ),
        ),
    ];
    let mut stage = [None; 2];
    let mut output = [None; 2];
    let executor = executor::PartitionExecutor(std::num::NonZeroUsize::new(2).unwrap());
    normal_posteriors_into(
        &requests,
        ExecutionPolicy::Auto,
        &mut output,
        &mut stage,
        ExecutionContext::new(&executor),
    )
    .unwrap();
    close(
        output[0].unwrap().parameters().unwrap().3,
        prior.update_batch(&values).unwrap().parameters().unwrap().3,
        1e-14,
    );
    let mut sequential = [None; 2];
    normal_posteriors_into(
        &requests,
        ExecutionPolicy::Sequential,
        &mut sequential,
        &mut stage,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output, sequential);
    let old = output;
    assert!(
        normal_posteriors_into(
            &requests,
            ExecutionPolicy::Sequential,
            &mut output,
            &mut [],
            ExecutionContext::sequential(),
        )
        .is_err()
    );
    assert_eq!(output, old);
    let bad = [f64::NAN];
    let invalid = [
        requests[0],
        NormalPosteriorRequest::new(prior, MatrixView::from_slice(1, 1, &bad).unwrap()),
    ];
    assert!(
        normal_posteriors_into(
            &invalid,
            ExecutionPolicy::Auto,
            &mut output,
            &mut stage,
            ExecutionContext::new(&executor),
        )
        .is_err()
    );
    assert_eq!(output, old);
    let wrong_weights = [requests[0].with_weights(Weights::<_, Frequency>::new(
        MatrixView::from_slice(1, 1, &weights[..1]).unwrap(),
    ))];
    assert!(
        normal_posteriors_into(
            &wrong_weights,
            ExecutionPolicy::Sequential,
            &mut output[..1],
            &mut stage,
            ExecutionContext::sequential(),
        )
        .is_err()
    );
    normal_posteriors_into::<f64>(
        &[],
        ExecutionPolicy::Sequential,
        &mut [],
        &mut [],
        ExecutionContext::sequential(),
    )
    .unwrap();
}

#[test]
fn independent_high_precision_marginal_quantiles_both_widths() {
    let m64 = normal_posterior_marginals(&[6_f64, 9., 12., 7., 8., 8., 13.]).unwrap();
    let m32 = normal_posterior_marginals(&[6_f32, 9., 12., 7., 8., 8., 13.]).unwrap();
    let b64 = BetaBinomialPrior::<f64>::new(3., 7.)
        .unwrap()
        .posterior_law()
        .unwrap();
    let b32 = BetaBinomialPrior::<f32>::new(3., 7.)
        .unwrap()
        .posterior_law()
        .unwrap();
    let g64 = GammaPoissonPrior::<f64>::new(2.5, 3.)
        .unwrap()
        .posterior_law()
        .unwrap();
    let g32 = GammaPoissonPrior::<f32>::new(2.5, 3.)
        .unwrap()
        .posterior_law()
        .unwrap();
    for line in include_str!("data/conjugate.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<_> = line.split('\t').collect();
        let p: f64 = fields[2].parse().unwrap();
        let expected: f64 = fields[3].parse().unwrap();
        let reference32 = f32::from_bits(u32::from_str_radix(fields[4], 16).unwrap());
        let reference64 = f64::from_bits(u64::from_str_radix(fields[5], 16).unwrap());
        assert_eq!(expected.to_bits(), reference64.to_bits());
        assert_eq!((expected as f32).to_bits(), reference32.to_bits());
        let (x64, x32) = match fields[0] {
            "beta" => (b64.quantile(p).unwrap(), b32.quantile(p as f32).unwrap()),
            "gamma" => (g64.quantile(p).unwrap(), g32.quantile(p as f32).unwrap()),
            "normal_mean" => (
                m64.mean_law().quantile(p).unwrap(),
                m32.mean_law().quantile(p as f32).unwrap(),
            ),
            "normal_variance" => (
                m64.variance_law().quantile(p).unwrap(),
                m32.variance_law().quantile(p as f32).unwrap(),
            ),
            "normal_std" => (
                m64.standard_deviation_law().quantile(p).unwrap(),
                m32.standard_deviation_law().quantile(p as f32).unwrap(),
            ),
            other => panic!("unknown fixture {other}"),
        };
        close(x64, expected, 8e-8);
        close(x32 as f64, expected, 1e-4);
        // Independently check the public credible inversion path at each tail.
        if p < 0.5 {
            let probability = ConfidenceProbability::new(1. - 2. * p).unwrap();
            let law: &dyn scioxide_random::ContinuousLaw<f64> = match fields[0] {
                "beta" => &b64,
                "gamma" => &g64,
                "normal_mean" => m64.mean_law(),
                "normal_variance" => m64.variance_law(),
                _ => m64.standard_deviation_law(),
            };
            let interval = credible_interval(law, probability, &accurate_options()).unwrap();
            close(interval.bounds().lower(), expected, 2e-10);
            close(law.cdf(interval.bounds().lower()).unwrap(), p, 2e-10);
            close(law.cdf(interval.bounds().upper()).unwrap(), 1. - p, 2e-10);
        }
    }
}

#[test]
fn portable_cancellation_and_f32_credible_accuracy() {
    struct Cancelled;
    impl scioxide_core::Cancellation for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    let values = [6_f32, 9., 12., 7., 8., 8., 13.];
    let request = [NormalPosteriorRequest::new(
        NormalInverseGammaPrior::reference(),
        MatrixView::from_slice(1, 7, &values).unwrap(),
    )];
    let mut stage = [None];
    let mut output = [None];
    assert!(
        normal_posteriors_into(
            &request,
            scioxide_core::ExecutionPolicy::Sequential,
            &mut output,
            &mut stage,
            ExecutionContext::sequential().with_cancellation(&Cancelled),
        )
        .is_err()
    );
    assert_eq!(output, [None]);
    normal_posteriors_into(
        &request,
        scioxide_core::ExecutionPolicy::Sequential,
        &mut output,
        &mut stage,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output[0].unwrap().parameters().unwrap(), (9., 7., 3., 20.));
    let root = InversionOptions::default()
        .root_options()
        .clone()
        .with_step_tolerance(scioxide_core::StepTolerance::new(1e-6_f32, 1e-6).unwrap());
    let options = InversionOptions::default()
        .with_root_options(root)
        .with_work_budget(Budget::ONE.saturating_add(4095));
    let summary =
        normal_credible_summary(&values, ConfidenceProbability::new(0.9).unwrap(), &options)
            .unwrap();
    close(summary.variance.mean.unwrap() as f64, 10., 2e-5);
    for line in include_str!("data/conjugate.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<_> = line.split('\t').collect();
        let p: f64 = fields[2].parse().unwrap();
        if p != 0.05 && p != 0.95 {
            continue;
        }
        let expected: f64 = fields[3].parse().unwrap();
        let reference32 = f32::from_bits(u32::from_str_radix(fields[4], 16).unwrap());
        let reference64 = f64::from_bits(u64::from_str_radix(fields[5], 16).unwrap());
        assert_eq!(expected.to_bits(), reference64.to_bits());
        assert_eq!((expected as f32).to_bits(), reference32.to_bits());
        let interval = match fields[0] {
            "normal_mean" => summary.mean.interval,
            "normal_variance" => summary.variance.interval,
            "normal_std" => summary.standard_deviation.interval,
            _ => continue,
        };
        close(
            if p == 0.05 {
                interval.bounds().lower()
            } else {
                interval.bounds().upper()
            } as f64,
            expected,
            3e-5,
        );
    }
    let limited_root = accurate_options()
        .root_options()
        .clone()
        .with_max_iterations(Budget::ONE);
    let limited = InversionOptions::default()
        .with_root_options(limited_root)
        .with_work_budget(Budget::ONE.saturating_add(4095));
    let beta = BetaBinomialPrior::<f64>::new(2., 3.)
        .unwrap()
        .posterior_law()
        .unwrap();
    assert!(credible_interval(&beta, ConfidenceProbability::default(), &limited).is_err());
}

#[test]
fn extreme_analytic_scatter_projects_posterior_instead_of_population_variance() {
    let values = [-1e200_f64, 1e200];
    let weights = [1e-200_f64; 2];
    let requests = [NormalPosteriorRequest::new(
        NormalInverseGammaPrior::new(0., 1., 2., 1.).unwrap(),
        MatrixView::from_slice(1, 2, &values).unwrap(),
    )
    .with_weights(Weights::<_, Analytic>::new(
        MatrixView::from_slice(1, 2, &weights).unwrap(),
    ))];
    let mut stage = [None];
    let mut output = [None];
    normal_posteriors_into(
        &requests,
        scioxide_core::ExecutionPolicy::Sequential,
        &mut output,
        &mut stage,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let p = output[0].unwrap();
    close(p.parameters().unwrap().3, 1e200, 2e-14);
    close(
        p.marginals().unwrap().variance_law().mean().unwrap(),
        5e199,
        2e-14,
    );
    assert!(p.posterior_law().is_ok());
}

#[test]
fn default_options_resolve_iterative_beta_tails() {
    use scioxide_random::Cdf;
    let p = ConfidenceProbability::new(0.95).unwrap();
    let law = BetaBinomialPrior::<f64>::new(0.5, 0.5)
        .unwrap()
        .update(5, 30)
        .unwrap()
        .posterior_law()
        .unwrap();
    let interval = credible_interval(&law, p, &InversionOptions::default()).unwrap();
    close(law.cdf(interval.bounds().lower()).unwrap(), 0.025, 1e-12);
    close(law.cdf(interval.bounds().upper()).unwrap(), 0.975, 1e-12);
}
