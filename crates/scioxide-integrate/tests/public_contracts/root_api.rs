//! External coverage for every supported crate-root integration contract.

#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Budget, ErrorTolerance, ExecutionContext, Interval, NormSelection, SciOxideError,
};
use scioxide_integrate::GaussianRulePlan;
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, ClenshawCurtis, DoubleExponentialDomain,
    DoubleExponentialIntegrator, FixedRule, FourierWeight, GaussKronrod, GaussKronrodRule,
    GaussianSupport, IntegrationReport, IntegrationTermination, LevelOptions, MonteCarloIntegrator,
    MonteCarloOptions, MonteCarloStatistics, MonteCarloWorkspace, NewtonCotes,
    NewtonCotesEndpoints, ProductRule, RandomStreams, Romberg, SubdivisionOptions, VarianceBasis,
    WeightedQuadraturePlan,
};
use scioxide_poly::OrthogonalFamily;
use scioxide_random::{Philox4x64, SeedableRng};
use scioxide_stats::ConfidenceProbability;

type ErasedCallback<'a> = &'a (dyn Fn(f64) -> Result<f64, SciOxideError> + Sync);

fn apply_fixed<I: FixedRule<f64>>(
    rule: &I,
    integrand: ErasedCallback<'_>,
    interval: Interval<f64>,
) -> Result<(f64, IntegrationReport<f64>), SciOxideError> {
    // The metadata protocol is object safe.
    let erased: &dyn FixedRule<f64> = rule;
    assert!(erased.node_count() > 0);
    rule.integrate(integrand, interval)
}

#[test]
fn crate_root_exposes_every_one_dimensional_algorithm() -> Result<(), SciOxideError> {
    let interval = Interval::new(0.0_f64, 1.0)?;
    let function = |x: f64| -> Result<f64, SciOxideError> { Ok(x * x) };
    let erased: ErasedCallback<'_> = &function;
    let erased_callback: ErasedCallback<'_> = &function;

    let legendre = GaussianRulePlan::gauss_legendre(4)?;
    let jacobi = GaussianRulePlan::new(
        OrthogonalFamily::Jacobi {
            alpha: 0.0_f64,
            beta: 0.0,
        },
        4,
    )?;
    let gegenbauer = GaussianRulePlan::new(OrthogonalFamily::Gegenbauer { lambda: 0.5 }, 4)?;
    let kronrod = GaussKronrod::new(GaussKronrodRule::G7K15).unwrap();
    let clenshaw_curtis = ClenshawCurtis::new(5)?;
    let newton_cotes = NewtonCotes::new(3, NewtonCotesEndpoints::Closed, Budget::try_from(4)?)?;

    // Every unweighted fixed rule integrates x^2 exactly on [0, 1].
    [
        apply_fixed(&legendre, erased_callback, interval)?,
        apply_fixed(&jacobi, erased_callback, interval)?,
        apply_fixed(&gegenbauer, erased_callback, interval)?,
        apply_fixed(&kronrod, erased_callback, interval)?,
        apply_fixed(&clenshaw_curtis, erased_callback, interval)?,
        apply_fixed(&newton_cotes, erased_callback, interval)?,
    ]
    .into_iter()
    .for_each(|(value, report)| {
        assert!(report.is_success());
        assert!((value - 1.0 / 3.0).abs() < 1.0e-15, "{value}");
    });

    // A weighted family never answers an unweighted request; its weight is explicit.
    let chebyshev = GaussianRulePlan::new(OrthogonalFamily::ChebyshevT, 4)?;
    assert!(matches!(
        chebyshev
            .integrate(erased_callback, interval)
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    let (weighted, _) =
        chebyshev.integrate_weighted(erased_callback, GaussianSupport::Interval(interval))?;
    // ∫_0^1 x^2 / sqrt(x (1 - x)) dx = 3π/8.
    assert!((weighted - 3.0 * std::f64::consts::PI / 8.0).abs() < 1.0e-14);

    let tolerance = ErrorTolerance::scalar(1.0e-10, 1.0e-10, NormSelection::Infinity)?;
    let subdivision = SubdivisionOptions::default()
        .with_tolerance(tolerance.clone())
        .with_max_subdivisions(Budget::ONE.saturating_add(14))
        .with_max_evaluations(Some(Budget::ONE.saturating_add(254)));
    assert_eq!(subdivision.max_subdivisions().get(), 15);
    assert_eq!(subdivision.max_evaluations().map(Budget::get), Some(255));
    assert_eq!(
        subdivision.tolerance().absolute().component(0, 1).unwrap(),
        1.0e-10
    );

    let adaptive_gk = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, subdivision.clone())?;
    let adaptive_simpson = AdaptiveIntegrator::new(AdaptiveRule::Simpson, subdivision)?;
    for (value, report) in [
        adaptive_gk.integrate(erased_callback, interval)?,
        adaptive_simpson.integrate(erased_callback, interval)?,
    ] {
        assert!(value.is_finite());
        assert!(report.error_estimate().is_some());
    }

    let refinement = LevelOptions::default()
        .with_tolerance(tolerance.clone())
        .with_max_levels(Budget::ONE.saturating_add(7))
        .with_max_evaluations(Some(Budget::ONE.saturating_add(1023)));
    assert_eq!(refinement.max_levels().get(), 8);
    assert_eq!(refinement.max_evaluations().map(Budget::get), Some(1024));

    let romberg = Romberg::new(refinement)?;
    assert!(romberg.integrate(erased_callback, interval)?.0.is_finite());

    let finite = DoubleExponentialDomain::Finite {
        interval,
        singular_points: &[],
    };
    let transform = DoubleExponentialIntegrator::new(
        finite,
        LevelOptions::default()
            .with_tolerance(tolerance)
            .with_max_levels(Budget::ONE.saturating_add(7))
            .with_max_evaluations(Some(Budget::ONE.saturating_add(1023))),
    )?;
    assert!(
        transform
            .integrate(
                |point: scioxide_integrate::DoubleExponentialPoint<_>| erased_callback(
                    point.coordinate()
                )
            )?
            .0
            .is_finite()
    );
    assert!(
        DoubleExponentialIntegrator::new(
            DoubleExponentialDomain::HalfLine { lower: 0.0 },
            transform.options().clone()
        )
        .unwrap()
        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<f64>| {
            let x = point.coordinate();
            Ok((-x).exp())
        })?
        .0
        .is_finite()
    );
    assert!(
        DoubleExponentialIntegrator::new(
            DoubleExponentialDomain::RealLine,
            transform.options().clone()
        )
        .unwrap()
        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<f64>| {
            let x = point.coordinate();
            Ok((-x * x).exp())
        })?
        .0
        .is_finite()
    );
    assert!(
        WeightedQuadraturePlan::breakpoints(
            interval,
            &[],
            scioxide_integrate::WeightedQuadratureOptions::new(SubdivisionOptions::default())
        )?
        .integrate(erased)?
        .0
        .is_finite()
    );
    assert!(
        WeightedQuadraturePlan::fourier(
            interval,
            &[],
            2.0,
            FourierWeight::Sine,
            scioxide_integrate::WeightedQuadratureOptions::new(SubdivisionOptions::default())
                .with_moment_levels(Budget::try_from(16)?)
        )?
        .integrate(|_: f64| Ok(1.0))?
        .0
        .is_finite()
    );

    let hermite = GaussianRulePlan::new(OrthogonalFamily::HermiteHe, 4)?;
    let laguerre = GaussianRulePlan::new(OrthogonalFamily::Laguerre, 4)?;
    // ∫ x^2 exp(-x^2/2) dx = sqrt(2π) and ∫_0^∞ x^2 exp(-x) dx = 2.
    let (hermite_value, _) =
        hermite.integrate_weighted(erased_callback, GaussianSupport::RealLine)?;
    assert!((hermite_value - (2.0 * std::f64::consts::PI).sqrt()).abs() < 1.0e-14);
    let (laguerre_value, _) =
        laguerre.integrate_weighted(erased_callback, GaussianSupport::HalfLine(0.0))?;
    assert!((laguerre_value - 2.0).abs() < 1.0e-14);

    Ok::<(), SciOxideError>(())
}

#[test]
fn crate_root_exposes_multidimensional_results_and_statistics() -> Result<(), SciOxideError> {
    let bounds = [Interval::new(0.0_f64, 1.0)?, Interval::new(0.0, 1.0)?];
    let integrand = |point: &[f64]| Ok(point[0] + point[1]);
    let erased: &(dyn Fn(&[f64]) -> Result<f64, SciOxideError> + Sync) = &integrand;

    let legendre = GaussianRulePlan::gauss_legendre(3)?;
    let product = ProductRule::new(&legendre, scioxide_integrate::ProductOptions::default());
    let product_result: (f64, scioxide_integrate::IntegrationReport<f64>) =
        product.integrate(erased, &bounds)?;
    assert_eq!(
        product_result.1.termination(),
        IntegrationTermination::FixedRuleCompleted
    );
    assert!(product_result.1.error_components().is_none());

    let cubature =
        scioxide_integrate::CubaturePlan::new(&bounds, SubdivisionOptions::default()).unwrap();
    assert!(cubature.integrate(erased)?.0.is_finite());

    let confidence = ConfidenceProbability::new(0.975)?;
    let options = MonteCarloOptions::default()
        .with_samples(Budget::ONE.saturating_add(63))
        .with_confidence_probability(confidence);
    assert_eq!(options.samples().get(), 64);
    assert_eq!(options.replicates(), 1);
    assert_eq!(options.confidence_probability(), confidence);

    let monte_carlo = MonteCarloIntegrator::new(&bounds, options)?;
    assert_eq!(monte_carlo.options(), &options);
    assert_eq!(monte_carlo.bounds(), &bounds);
    let mut workspace: MonteCarloWorkspace<f64> =
        monte_carlo.workspace(1, ExecutionContext::sequential_native())?;
    let root = Philox4x64::from_seed(27);
    let mut value = 0.0;
    let report: IntegrationReport<f64> = monte_carlo.integrate_into(
        |point: &[f64]| Ok(point[0] + point[1]),
        RandomStreams::new(&root, 0),
        &mut value,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    let mut statistics: MonteCarloStatistics<f64> = workspace.statistics()[0].clone().unwrap();
    assert_eq!(statistics.samples(), 64);
    assert_eq!(statistics.basis(), VarianceBasis::Observations);
    assert_eq!(statistics.confidence_probability(), confidence);
    assert_eq!(statistics.moments().unwrap().mean()?, Some(value));
    assert!(statistics.estimator_variance().round()?.is_finite());
    assert!(statistics.standard_error().is_finite());
    assert!(statistics.confidence_interval()?.width()?.is_finite());
    assert_eq!(
        report.termination(),
        IntegrationTermination::SampleBudgetCompleted
    );
    assert_eq!(report.evaluations(), 64);
    assert_eq!(workspace.children().len(), 1);

    Ok::<(), SciOxideError>(())
}
